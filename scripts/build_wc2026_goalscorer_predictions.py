"""Build WC2026 individual goalscorer model predictions for the two Polymarket
markets we found: Golden Boot winner and Player-to-score (anytime over the tournament).

Method (validated 2026-06-15, see memory individual-goal-share-result): each player's
expected tournament goals lambda_i = (team expected tournament goals) * p_i, where
p_i is the player's share of team goals from his PRE-TOURNAMENT CLUB goal rate
(goals/90 from appearances.csv, latest 365d) over the squad, smoothed by a position
prior. Team expected tournament goals = (expected matches played, from the sim's
reach-round probabilities) * (team goals per match, from the team's relative attack).

Outputs (model side only; market comparison is a separate step):
  reports/wc2026_goalscorer_model_predictions.csv  (player, team, lambda_goals,
      anytime_score_probability, golden_boot_probability)
"""

from __future__ import annotations

import argparse
import math
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

POSITION_PRIOR = {"GK": 0.02, "DF": 0.5, "MF": 1.0, "FW": 2.2}
WC2026_KICKOFF = pd.Timestamp("2026-06-08")
# League-average goals a team scores per WC match (both teams ~2.5 total goals/game).
BASE_TEAM_GOALS_PER_MATCH = 1.35
# How strongly the Elo gap (own attack vs opponent defence) moves goals/match. ~400 Elo
# (a strong side vs a weak one) ~ doubles the goals via the exp() below — i.e. it IS
# easier to score against weak teams, which the flat-base version ignored (user, 2026-06-15).
ELO_GAP_GOALS_SENSITIVITY = 1.0 / 500.0


def _norm(t: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", str(t)) if not unicodedata.combining(c)).lower().strip()


def _tokens(n: str) -> set[str]:
    return {x for x in _norm(n).replace(".", " ").split() if len(x) > 1}


def _surname(n: str) -> str:
    p = _norm(n).replace(".", " ").split()
    return p[-1] if p else ""


def _coarse_position(pos: str) -> str:
    p = str(pos).lower()
    if "goalkeeper" in p or p == "gk":
        return "GK"
    if "back" in p or "defen" in p or p in ("df", "cb", "lb", "rb"):
        return "DF"
    if "forward" in p or "striker" in p or "wing" in p or p in ("fw", "cf", "st"):
        return "FW"
    return "MF"


def _index_from_rate_frame(g: pd.DataFrame) -> tuple[list, dict]:
    """Build the (token-index, surname-index) name lookups from a frame with columns
    name + per90."""
    tok = [(_tokens(r.name), float(r.per90)) for r in g.itertuples(index=False)]
    sur: dict[str, float] = {}
    for r in g.itertuples(index=False):
        sur.setdefault(_surname(r.name), float(r.per90))
    return tok, sur


def _build_club_rate(appearances: pd.DataFrame, *, cutoff: pd.Timestamp,
                     window_days: int = 365) -> tuple[list, dict]:
    """Compute per-player club goals/90 from the raw appearances file (the heavy
    142MB Transfermarkt CSV; used locally / to refresh the small precomputed artifact)."""
    lo = cutoff - pd.Timedelta(days=window_days)
    w = appearances[(appearances["date"] >= lo) & (appearances["date"] < cutoff)]
    g = w.groupby("player_id").agg(
        name=("player_name", "first"), goals=("goals", "sum"), mins=("minutes_played", "sum"),
    ).reset_index()
    g = g[g["mins"] >= 270]
    g["per90"] = g["goals"] / g["mins"] * 90.0
    return _index_from_rate_frame(g)


def _load_rate_index(rate_input: str, appearances_input: str) -> tuple[list, dict]:
    """Prefer the small precomputed per-player rate artifact (committed, available in
    CI). Fall back to the heavy raw appearances file when the artifact is missing
    (e.g. first build, or to regenerate it). The raw file is gitignored (142MB), so in
    CI only the precomputed path works — that's why it's the default."""
    if Path(rate_input).exists():
        g = pd.read_csv(rate_input)
        return _index_from_rate_frame(g)
    if Path(appearances_input).exists():
        appearances = pd.read_csv(
            appearances_input,
            usecols=["player_id", "player_name", "date", "goals", "minutes_played"],
        )
        appearances["date"] = pd.to_datetime(appearances["date"], errors="coerce")
        return _build_club_rate(appearances, cutoff=WC2026_KICKOFF)
    raise FileNotFoundError(
        f"Neither rate artifact ({rate_input}) nor appearances ({appearances_input}) found."
    )


def _match_rate(name: str, tok_index: list, sur_index: dict) -> float | None:
    tk = _tokens(name)
    best, best_ov = None, 0
    for kt, rec in tok_index:
        ov = len(tk & kt)
        if ov >= 2 and ov > best_ov:
            best_ov, best = ov, rec
    return best if best is not None else sur_index.get(_surname(name))


_KO_COLS = ("reach_round_of_32_probability", "reach_round_of_16_probability",
            "reach_quarterfinal_probability", "reach_semifinal_probability",
            "reach_final_probability")


def _goals_per_match_vs(team_elo: float, opponent_elo: float) -> float:
    """Goals a team is expected to score in one match, scaled by the Elo gap (own
    attack vs the OPPONENT's defence). Weak opponents -> more goals."""
    gap = (team_elo - opponent_elo) * ELO_GAP_GOALS_SENSITIVITY
    return max(0.25, BASE_TEAM_GOALS_PER_MATCH * math.exp(gap))


def _expected_team_tournament_goals(
    *, team: str, team_elo: float, sim_row, group_opponent_elos: list[float],
    mean_elo: float,
) -> float:
    """Sum of expected goals across the team's likely schedule, opponent-adjusted:
    the 3 REAL group games (known opponents), plus knockout rounds weighted by the
    probability of playing them, against average-strength KO opposition (mean Elo,
    which rises as the bracket deepens — approximated by the field mean)."""
    total = 0.0
    # Group stage: exact opponents from the real draw.
    for opp_elo in group_opponent_elos:
        total += _goals_per_match_vs(team_elo, opp_elo)
    # Knockout: each round weighted by reach prob; opponents are tournament-average
    # (KO sides are above-average, so use mean field Elo as a conservative opponent).
    for col in _KO_COLS:
        play_prob = float(getattr(sim_row, col, 0.0) or 0.0)
        total += play_prob * _goals_per_match_vs(team_elo, mean_elo)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--squad-input", default="reports/wc2026_squad_players_with_player_elo.csv")
    parser.add_argument("--sim-input",
                        default="reports/wc2026_simulation_expected_xi_plus_goalkeeper_probabilities.csv")
    parser.add_argument("--elo-input", default="reports/baseline_latest_elo_ratings.csv")
    parser.add_argument("--groups-input", default="data/reference/wc2026_groups_actual.csv")
    parser.add_argument(
        "--rate-input", default="data/reference/wc2026_player_goal_rate.csv",
        help="Small precomputed per-player club goals/90 (committed, used in CI). "
             "Regenerate from raw appearances with scripts/build_player_goal_rate.py.",
    )
    parser.add_argument("--appearances-input", default="data/raw/transfermarkt/appearances.csv",
                        help="Heavy raw Transfermarkt appearances (gitignored); fallback only.")
    parser.add_argument("--output", default="reports/wc2026_goalscorer_model_predictions.csv")
    args = parser.parse_args()

    squad = pd.read_csv(args.squad_input)
    sim = pd.read_csv(args.sim_input)
    elo = pd.read_csv(args.elo_input)
    groups = pd.read_csv(args.groups_input)

    tok_index, sur_index = _load_rate_index(args.rate_input, args.appearances_input)

    sim_by_team = {str(r.team): r for r in sim.itertuples(index=False)}
    elo_map = {str(r.team): float(r.elo_rating) for r in elo.itertuples(index=False)}
    mean_elo = np.mean(list(elo_map.values())) if elo_map else 1500.0

    # Real group-stage opponents per team (for opponent-adjusted goals).
    group_opponents: dict[str, list[float]] = {}
    for _, sub in groups.groupby("group"):
        teams_in = list(sub["team"])
        for t in teams_in:
            group_opponents[str(t)] = [
                elo_map.get(str(o), mean_elo) for o in teams_in if str(o) != str(t)
            ]

    rows: list[dict[str, object]] = []
    for team, grp in squad.groupby("team"):
        team = str(team)
        sim_row = sim_by_team.get(team)
        # Opponent-adjusted expected tournament goals: easier to score vs weak sides
        # (real group draw) and weighted by how far the team is likely to advance.
        team_tournament_goals = _expected_team_tournament_goals(
            team=team, team_elo=elo_map.get(team, mean_elo), sim_row=sim_row,
            group_opponent_elos=group_opponents.get(team, [mean_elo, mean_elo, mean_elo]),
            mean_elo=mean_elo,
        ) if sim_row is not None else 3.0 * BASE_TEAM_GOALS_PER_MATCH

        names = list(grp["player"])
        positions = list(grp["position"]) if "position" in grp.columns else ["MF"] * len(names)
        rates = np.array([(lambda r: 0.0 if r is None else max(r, 0.0))(
            _match_rate(n, tok_index, sur_index)) for n in names])
        priors = np.array([POSITION_PRIOR[_coarse_position(p)] for p in positions])
        weight = rates + 0.15 * priors
        if weight.sum() <= 0:
            weight = priors.copy()
        p_share = weight / weight.sum()

        for name, share in zip(names, p_share):
            lam = float(team_tournament_goals * share)
            rows.append({
                "player": name,
                "team": team,
                "lambda_goals": round(lam, 4),
                "anytime_score_probability": round(1.0 - math.exp(-lam), 4),
            })

    frame = pd.DataFrame(rows)
    # Golden Boot: probability of being THE top scorer. A softmax over ALL ~700 squad
    # players smears the mass so thin that even Kane lands at <1% (the field can't all
    # be 0). The Golden Boot is realistically a contest among the ~40-50 genuine
    # contenders, so we softmax over expected goals but RESTRICT the normalization to
    # players with a real chance (lambda above a floor), and add a small residual mass
    # for the long tail. This makes the levels comparable to the market, not just the
    # ranking. Temperature controls how peaked the favourites are.
    temp = 0.55
    lam = frame["lambda_goals"].to_numpy()
    exp_lam = np.exp(lam / temp)
    # Long-tail dampening: players below a goal-expectation floor barely contend.
    floor = np.quantile(lam, 0.85)  # only the top ~15% by expected goals truly contend
    exp_lam = np.where(lam >= floor, exp_lam, exp_lam * 0.02)
    # Reserve ~8% for "a non-favourite / surprise wins it" so favourites don't overfill.
    gb = 0.92 * exp_lam / exp_lam.sum()
    frame["golden_boot_probability"] = gb.round(4)

    frame = frame.sort_values("golden_boot_probability", ascending=False).reset_index(drop=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Wrote {args.output}: {len(frame)} players across {frame['team'].nunique()} teams.")
    print("\nTop 12 by Golden Boot probability:")
    print(frame.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
