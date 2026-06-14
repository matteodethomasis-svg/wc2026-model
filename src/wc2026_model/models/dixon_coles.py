from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

from wc2026_model.types import MatchInfo, PredictionRecord, ThreeWayProbabilities


def exponential_time_decay_weights(
    match_dates: pd.Series,
    *,
    xi: float = 0.001,
    reference_date: pd.Timestamp | None = None,
) -> np.ndarray:
    if reference_date is None:
        reference_date = pd.Timestamp(match_dates.max())
    day_gaps = (reference_date - pd.to_datetime(match_dates)).dt.days.to_numpy(dtype=float)
    return np.exp(-xi * np.maximum(day_gaps, 0.0))


@dataclass(frozen=True)
class DixonColesFitResult:
    success: bool
    message: str
    iterations: int
    objective_value: float


class DixonColesModel:
    def __init__(
        self,
        *,
        teams: list[str],
        attack: np.ndarray,
        defense: np.ndarray,
        intercept: float,
        home_advantage: float,
        elo_weight: float,
        rho: float,
        fit_result: DixonColesFitResult,
    ) -> None:
        self.teams = teams
        self.attack = attack
        self.defense = defense
        self.intercept = intercept
        self.home_advantage = home_advantage
        self.elo_weight = elo_weight
        self.rho = rho
        self.fit_result = fit_result
        self._team_to_index = {team: index for index, team in enumerate(teams)}

    @classmethod
    def fit(
        cls,
        matches: pd.DataFrame,
        *,
        l2_penalty: float = 0.0,
        maxiter: int = 1000,
    ) -> "DixonColesModel":
        required_columns = {
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
            "neutral",
            "elo_diff_pre",
            "sample_weight",
        }
        missing_columns = required_columns.difference(matches.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Missing required columns for Dixon-Coles fit: {missing}")

        teams = sorted(set(matches["home_team"]).union(matches["away_team"]))
        team_to_index = {team: index for index, team in enumerate(teams)}

        home_index = matches["home_team"].map(team_to_index).to_numpy(dtype=int)
        away_index = matches["away_team"].map(team_to_index).to_numpy(dtype=int)
        home_goals = matches["home_goals"].to_numpy(dtype=int)
        away_goals = matches["away_goals"].to_numpy(dtype=int)
        home_field = (~matches["neutral"]).to_numpy(dtype=float)
        elo_diff = matches["elo_diff_pre"].to_numpy(dtype=float) / 400.0
        sample_weight = matches["sample_weight"].to_numpy(dtype=float)

        team_count = len(teams)
        initial_params = _build_initial_parameters(
            matches=matches,
            teams=teams,
            team_to_index=team_to_index,
        )

        objective = lambda params: _objective_and_gradient(
            params=params,
            home_index=home_index,
            away_index=away_index,
            home_goals=home_goals,
            away_goals=away_goals,
            home_field=home_field,
            elo_diff=elo_diff,
            sample_weight=sample_weight,
            l2_penalty=l2_penalty,
        )

        optimization = minimize(
            objective,
            initial_params,
            jac=True,
            method="L-BFGS-B",
            options={
                "maxiter": maxiter,
                "maxfun": max(50000, maxiter * 100),
            },
        )

        attack, defense, intercept, home_advantage, elo_weight, rho = _unpack_parameters(
            optimization.x, team_count
        )
        fit_result = DixonColesFitResult(
            success=bool(optimization.success),
            message=str(optimization.message),
            iterations=int(getattr(optimization, "nit", 0)),
            objective_value=float(optimization.fun),
        )
        return cls(
            teams=teams,
            attack=attack,
            defense=defense,
            intercept=float(intercept),
            home_advantage=float(home_advantage),
            elo_weight=float(elo_weight),
            rho=float(rho),
            fit_result=fit_result,
        )

    def predict_expected_goals(
        self,
        home_team: str,
        away_team: str,
        *,
        neutral_site: bool = False,
        elo_diff_pre: float = 0.0,
    ) -> tuple[float, float]:
        home_index = self._lookup_team(home_team)
        away_index = self._lookup_team(away_team)
        home_field = 0.0 if neutral_site else 1.0
        elo_term = elo_diff_pre / 400.0

        lambda_home = math.exp(
            self.intercept
            + (self.home_advantage * home_field)
            + (self.elo_weight * elo_term)
            + self.attack[home_index]
            - self.defense[away_index]
        )
        lambda_away = math.exp(
            self.intercept
            - (self.elo_weight * elo_term)
            + self.attack[away_index]
            - self.defense[home_index]
        )
        return lambda_home, lambda_away

    def predict_score_matrix(
        self,
        home_team: str,
        away_team: str,
        *,
        neutral_site: bool = False,
        elo_diff_pre: float = 0.0,
        max_goals: int = 10,
    ) -> np.ndarray:
        lambda_home, lambda_away = self.predict_expected_goals(
            home_team,
            away_team,
            neutral_site=neutral_site,
            elo_diff_pre=elo_diff_pre,
        )
        home_range = np.arange(max_goals + 1)
        away_range = np.arange(max_goals + 1)
        matrix = np.outer(
            poisson.pmf(home_range, lambda_home),
            poisson.pmf(away_range, lambda_away),
        )
        for home_goals in range(min(2, max_goals) + 1):
            for away_goals in range(min(2, max_goals) + 1):
                matrix[home_goals, away_goals] *= _tau(
                    home_goals, away_goals, lambda_home, lambda_away, self.rho
                )
        matrix_sum = matrix.sum()
        if matrix_sum <= 0.0:
            raise ValueError("Predicted score matrix has non-positive mass.")
        return matrix / matrix_sum

    def predict_outcome_probabilities(
        self,
        home_team: str,
        away_team: str,
        *,
        neutral_site: bool = False,
        elo_diff_pre: float = 0.0,
        max_goals: int = 10,
    ) -> ThreeWayProbabilities:
        matrix = self.predict_score_matrix(
            home_team,
            away_team,
            neutral_site=neutral_site,
            elo_diff_pre=elo_diff_pre,
            max_goals=max_goals,
        )
        home_probability = float(np.tril(matrix, k=-1).sum())
        draw_probability = float(np.trace(matrix))
        away_probability = float(np.triu(matrix, k=1).sum())
        return ThreeWayProbabilities(
            home=home_probability,
            draw=draw_probability,
            away=away_probability,
        )

    def predict_match(
        self,
        match: MatchInfo,
        *,
        elo_diff_pre: float = 0.0,
        max_goals: int = 10,
    ) -> PredictionRecord:
        probabilities = self.predict_outcome_probabilities(
            match.home_team,
            match.away_team,
            neutral_site=match.is_neutral_site,
            elo_diff_pre=elo_diff_pre,
            max_goals=max_goals,
        )
        return PredictionRecord(
            match=match,
            probabilities=probabilities,
            model_name="dixon_coles_elo",
        )

    def team_strengths(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "team": self.teams,
                "attack": self.attack,
                "defense": self.defense,
            }
        ).sort_values("attack", ascending=False, kind="stable")

    def _lookup_team(self, team: str) -> int:
        if team not in self._team_to_index:
            raise KeyError(f"Unknown team '{team}' for fitted Dixon-Coles model.")
        return self._team_to_index[team]


def _unpack_parameters(
    parameters: np.ndarray,
    team_count: int,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
    raw_attack = parameters[:team_count]
    raw_defense = parameters[team_count : 2 * team_count]
    intercept = float(parameters[(2 * team_count)])
    home_advantage = float(parameters[(2 * team_count) + 1])
    elo_weight = float(parameters[(2 * team_count) + 2])
    rho = float(0.2 * np.tanh(parameters[(2 * team_count) + 3]))

    attack = raw_attack - raw_attack.mean()
    defense = raw_defense - raw_defense.mean()
    return attack, defense, intercept, home_advantage, elo_weight, rho


def _negative_log_likelihood(
    *,
    params: np.ndarray,
    home_index: np.ndarray,
    away_index: np.ndarray,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    home_field: np.ndarray,
    elo_diff: np.ndarray,
    sample_weight: np.ndarray,
    l2_penalty: float,
) -> float:
    objective_value, _ = _objective_and_gradient(
        params=params,
        home_index=home_index,
        away_index=away_index,
        home_goals=home_goals,
        away_goals=away_goals,
        home_field=home_field,
        elo_diff=elo_diff,
        sample_weight=sample_weight,
        l2_penalty=l2_penalty,
    )
    return objective_value


def _objective_and_gradient(
    *,
    params: np.ndarray,
    home_index: np.ndarray,
    away_index: np.ndarray,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    home_field: np.ndarray,
    elo_diff: np.ndarray,
    sample_weight: np.ndarray,
    l2_penalty: float,
) -> tuple[float, np.ndarray]:
    team_count = int(len(params[:-4]) / 2)
    attack, defense, intercept, home_advantage, elo_weight, rho = _unpack_parameters(
        params, team_count
    )

    log_lambda_home = (
        intercept
        + (home_advantage * home_field)
        + (elo_weight * elo_diff)
        + attack[home_index]
        - defense[away_index]
    )
    log_lambda_away = (
        intercept
        - (elo_weight * elo_diff)
        + attack[away_index]
        - defense[home_index]
    )

    lambda_home = np.exp(log_lambda_home)
    lambda_away = np.exp(log_lambda_away)

    tau, d_tau_d_lambda_home, d_tau_d_lambda_away, d_tau_d_rho = _tau_vectorized(
        home_goals=home_goals,
        away_goals=away_goals,
        lambda_home=lambda_home,
        lambda_away=lambda_away,
        rho=rho,
    )
    if np.any(tau <= 0.0):
        return 1e12, np.zeros_like(params)

    log_probability = (
        poisson.logpmf(home_goals, lambda_home)
        + poisson.logpmf(away_goals, lambda_away)
        + np.log(tau)
    )
    weighted_log_probability = sample_weight * log_probability
    penalty = l2_penalty * float(np.square(attack).sum() + np.square(defense).sum())
    objective_value = float(-weighted_log_probability.sum() + penalty)

    score_home = (home_goals - lambda_home) + ((lambda_home * d_tau_d_lambda_home) / tau)
    score_away = (away_goals - lambda_away) + ((lambda_away * d_tau_d_lambda_away) / tau)
    score_rho = d_tau_d_rho / tau

    weighted_score_home = sample_weight * score_home
    weighted_score_away = sample_weight * score_away

    grad_attack_centered = (
        np.bincount(home_index, weights=-weighted_score_home, minlength=team_count)
        + np.bincount(away_index, weights=-weighted_score_away, minlength=team_count)
    )
    grad_defense_centered = (
        np.bincount(away_index, weights=weighted_score_home, minlength=team_count)
        + np.bincount(home_index, weights=weighted_score_away, minlength=team_count)
    )
    grad_attack_centered = grad_attack_centered + (2.0 * l2_penalty * attack)
    grad_defense_centered = grad_defense_centered + (2.0 * l2_penalty * defense)

    grad_attack_raw = grad_attack_centered - grad_attack_centered.mean()
    grad_defense_raw = grad_defense_centered - grad_defense_centered.mean()

    grad_intercept = float(-(weighted_score_home + weighted_score_away).sum())
    grad_home_advantage = float(-(weighted_score_home * home_field).sum())
    grad_elo_weight = float(-(sample_weight * elo_diff * (score_home - score_away)).sum())

    raw_rho_parameter = float(params[(2 * team_count) + 3])
    d_rho_d_raw_rho = 0.2 * (1.0 - math.tanh(raw_rho_parameter) ** 2)
    grad_raw_rho = float(-(sample_weight * score_rho).sum() * d_rho_d_raw_rho)

    gradient = np.concatenate(
        [
            grad_attack_raw,
            grad_defense_raw,
            np.array(
                [
                    grad_intercept,
                    grad_home_advantage,
                    grad_elo_weight,
                    grad_raw_rho,
                ],
                dtype=float,
            ),
        ]
    )
    return objective_value, gradient


def _build_initial_parameters(
    *,
    matches: pd.DataFrame,
    teams: list[str],
    team_to_index: dict[str, int],
) -> np.ndarray:
    team_count = len(teams)
    initial_params = np.zeros((2 * team_count) + 4, dtype=float)

    sample_weight = matches["sample_weight"].to_numpy(dtype=float)
    home_goals = matches["home_goals"].to_numpy(dtype=float)
    away_goals = matches["away_goals"].to_numpy(dtype=float)
    weighted_goal_mean = float(
        np.average((home_goals + away_goals) / 2.0, weights=sample_weight)
    )
    initial_params[(2 * team_count)] = math.log(max(weighted_goal_mean, 0.2))

    non_neutral_mask = ~matches["neutral"].to_numpy(dtype=bool)
    if non_neutral_mask.any():
        weighted_home_mean = float(
            np.average(home_goals[non_neutral_mask], weights=sample_weight[non_neutral_mask])
        )
        weighted_away_mean = float(
            np.average(away_goals[non_neutral_mask], weights=sample_weight[non_neutral_mask])
        )
        if weighted_home_mean > 0.0 and weighted_away_mean > 0.0:
            initial_params[(2 * team_count) + 1] = math.log(
                weighted_home_mean / weighted_away_mean
            )

    team_weight = np.zeros(team_count, dtype=float)
    goals_scored = np.zeros(team_count, dtype=float)
    goals_conceded = np.zeros(team_count, dtype=float)
    for row in matches.itertuples(index=False):
        home_team_index = team_to_index[str(row.home_team)]
        away_team_index = team_to_index[str(row.away_team)]
        row_weight = float(row.sample_weight)

        team_weight[home_team_index] += row_weight
        team_weight[away_team_index] += row_weight
        goals_scored[home_team_index] += row_weight * float(row.home_goals)
        goals_scored[away_team_index] += row_weight * float(row.away_goals)
        goals_conceded[home_team_index] += row_weight * float(row.away_goals)
        goals_conceded[away_team_index] += row_weight * float(row.home_goals)

    safe_team_weight = np.maximum(team_weight, 1e-8)
    average_scored = goals_scored / safe_team_weight
    average_conceded = goals_conceded / safe_team_weight

    attack = np.log(np.maximum(average_scored / max(weighted_goal_mean, 1e-8), 1e-4))
    defense = np.log(np.maximum(max(weighted_goal_mean, 1e-8) / average_conceded, 1e-4))
    attack = attack - attack.mean()
    defense = defense - defense.mean()

    initial_params[:team_count] = attack
    initial_params[team_count : 2 * team_count] = defense
    initial_params[(2 * team_count) + 2] = 0.15
    initial_params[(2 * team_count) + 3] = math.atanh(-0.05 / 0.2)
    return initial_params


def _tau(
    home_goals: int,
    away_goals: int,
    lambda_home: float,
    lambda_away: float,
    rho: float,
) -> float:
    if home_goals == 0 and away_goals == 0:
        return 1.0 - (lambda_home * lambda_away * rho)
    if home_goals == 0 and away_goals == 1:
        return 1.0 + (lambda_home * rho)
    if home_goals == 1 and away_goals == 0:
        return 1.0 + (lambda_away * rho)
    if home_goals == 1 and away_goals == 1:
        return 1.0 - rho
    return 1.0


def _tau_vectorized(
    *,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    lambda_home: np.ndarray,
    lambda_away: np.ndarray,
    rho: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tau = np.ones_like(lambda_home, dtype=float)
    d_tau_d_lambda_home = np.zeros_like(lambda_home, dtype=float)
    d_tau_d_lambda_away = np.zeros_like(lambda_away, dtype=float)
    d_tau_d_rho = np.zeros_like(lambda_home, dtype=float)

    mask_00 = (home_goals == 0) & (away_goals == 0)
    tau[mask_00] = 1.0 - (lambda_home[mask_00] * lambda_away[mask_00] * rho)
    d_tau_d_lambda_home[mask_00] = -(lambda_away[mask_00] * rho)
    d_tau_d_lambda_away[mask_00] = -(lambda_home[mask_00] * rho)
    d_tau_d_rho[mask_00] = -(lambda_home[mask_00] * lambda_away[mask_00])

    mask_01 = (home_goals == 0) & (away_goals == 1)
    tau[mask_01] = 1.0 + (lambda_home[mask_01] * rho)
    d_tau_d_lambda_home[mask_01] = rho
    d_tau_d_rho[mask_01] = lambda_home[mask_01]

    mask_10 = (home_goals == 1) & (away_goals == 0)
    tau[mask_10] = 1.0 + (lambda_away[mask_10] * rho)
    d_tau_d_lambda_away[mask_10] = rho
    d_tau_d_rho[mask_10] = lambda_away[mask_10]

    mask_11 = (home_goals == 1) & (away_goals == 1)
    tau[mask_11] = 1.0 - rho
    d_tau_d_rho[mask_11] = -1.0
    return tau, d_tau_d_lambda_home, d_tau_d_lambda_away, d_tau_d_rho
