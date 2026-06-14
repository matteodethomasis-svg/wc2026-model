import numpy as np
import pandas as pd

from wc2026_model.tournament import load_played_group_results
from wc2026_model.tournament.simulation import simulate_world_cup_2026
from wc2026_model.types import ThreeWayProbabilities


class EloDrivenDeterministicModel:
    def predict_score_matrix(
        self,
        home_team: str,
        away_team: str,
        *,
        neutral_site: bool,
        elo_diff_pre: float,
        max_goals: int,
    ) -> np.ndarray:
        matrix = np.zeros((max_goals + 1, max_goals + 1), dtype=float)
        if elo_diff_pre > 0:
            matrix[1, 0] = 1.0
        elif elo_diff_pre < 0:
            matrix[0, 1] = 1.0
        else:
            matrix[0, 0] = 1.0
        return matrix

    def predict_expected_goals(
        self,
        home_team: str,
        away_team: str,
        *,
        neutral_site: bool,
        elo_diff_pre: float,
    ) -> tuple[float, float]:
        if elo_diff_pre > 0:
            return (1.0, 0.0)
        if elo_diff_pre < 0:
            return (0.0, 1.0)
        return (0.0, 0.0)

    def predict_outcome_probabilities(
        self,
        home_team: str,
        away_team: str,
        *,
        neutral_site: bool,
        elo_diff_pre: float,
        max_goals: int,
    ) -> ThreeWayProbabilities:
        if elo_diff_pre > 0:
            return ThreeWayProbabilities(home=1.0, draw=0.0, away=0.0)
        if elo_diff_pre < 0:
            return ThreeWayProbabilities(home=0.0, draw=0.0, away=1.0)
        return ThreeWayProbabilities(home=0.0, draw=1.0, away=0.0)


def test_simulation_conditions_on_played_group_results() -> None:
    groups = {}
    elo_ratings = {}
    for group_name in "ABCDEFGHIJKL":
        teams = [f"{group_name}_team_{slot}" for slot in range(1, 5)]
        groups[group_name] = teams
        elo_ratings[teams[0]] = 1800.0
        elo_ratings[teams[1]] = 1700.0
        elo_ratings[teams[2]] = 1600.0
        elo_ratings[teams[3]] = 1500.0

    played_group_results = pd.DataFrame(
        [
            {
                "group": "A",
                "home_team": "A_team_1",
                "away_team": "A_team_2",
                "home_goals": 0,
                "away_goals": 2,
            }
        ]
    )

    probabilities = simulate_world_cup_2026(
        model=EloDrivenDeterministicModel(),
        groups=groups,
        elo_ratings=elo_ratings,
        played_group_results=played_group_results,
        simulations=1,
        random_state=2026,
    )

    group_a = probabilities.loc[probabilities["group"] == "A"].set_index("team")
    assert group_a.loc["A_team_2", "group_winner_probability"] == 1.0
    assert group_a.loc["A_team_1", "group_runner_up_probability"] == 1.0


def test_load_played_group_results_ignores_historical_world_cup_pairings(tmp_path) -> None:
    raw_results = pd.DataFrame(
        [
            {
                "date": "2002-05-31",
                "home_team": "France",
                "away_team": "Senegal",
                "home_score": 0,
                "away_score": 1,
                "tournament": "FIFA World Cup",
                "city": "Seoul",
                "country": "South Korea",
                "neutral": True,
            },
            {
                "date": "2026-06-11",
                "home_team": "France",
                "away_team": "Iraq",
                "home_score": 2,
                "away_score": 0,
                "tournament": "FIFA World Cup",
                "city": "Toronto",
                "country": "Canada",
                "neutral": True,
            },
        ]
    )
    results_path = tmp_path / "results.csv"
    raw_results.to_csv(results_path, index=False)

    played_group_results = load_played_group_results(
        results_path,
        groups={"I": ["France", "Senegal", "Iraq", "Norway"]},
        as_of_date="2026-06-12",
        tournament_year=2026,
    )

    assert played_group_results.to_dict(orient="records") == [
        {
            "group": "I",
            "home_team": "France",
            "away_team": "Iraq",
            "home_goals": 2,
            "away_goals": 0,
        }
    ]
