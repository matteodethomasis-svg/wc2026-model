import numpy as np
import pandas as pd
import pytest

from wc2026_model.features import EloConfig, build_latest_elo_ratings
from wc2026_model.tournament import (
    build_group_stage_schedule,
    rank_group_standings,
    rank_third_place_teams,
    resolve_round_of_32_matchups,
    simulate_world_cup_2026,
)
from wc2026_model.types import ThreeWayProbabilities


class ToyTournamentModel:
    def predict_score_matrix(
        self,
        home_team: str,
        away_team: str,
        *,
        neutral_site: bool = False,
        elo_diff_pre: float = 0.0,
        max_goals: int = 10,
    ) -> np.ndarray:
        matrix = np.zeros((3, 3), dtype=float)
        matrix[1, 0] = 0.45
        matrix[1, 1] = 0.25
        matrix[0, 1] = 0.30
        return matrix

    def predict_expected_goals(
        self,
        home_team: str,
        away_team: str,
        *,
        neutral_site: bool = False,
        elo_diff_pre: float = 0.0,
    ) -> tuple[float, float]:
        return 1.4, 1.1

    def predict_outcome_probabilities(
        self,
        home_team: str,
        away_team: str,
        *,
        neutral_site: bool = False,
        elo_diff_pre: float = 0.0,
        max_goals: int = 10,
    ) -> ThreeWayProbabilities:
        return ThreeWayProbabilities(home=0.45, draw=0.25, away=0.30)


def test_build_latest_elo_ratings_returns_current_team_table(
    sample_international_results: pd.DataFrame,
) -> None:
    latest_ratings = build_latest_elo_ratings(
        sample_international_results,
        config=EloConfig(),
    )

    assert set(latest_ratings["team"]) == {"Alpha", "Beta", "Gamma", "Delta"}
    assert latest_ratings["matches_played"].sum() == 24
    assert latest_ratings.iloc[0]["elo_rating"] > latest_ratings.iloc[-1]["elo_rating"]
    assert latest_ratings["last_match_date"].notna().all()


def test_rank_group_standings_uses_head_to_head_for_overall_ties() -> None:
    group_results = pd.DataFrame(
        [
            {"group": "A", "home_team": "Alpha", "away_team": "Beta", "home_goals": 1, "away_goals": 0},
            {"group": "A", "home_team": "Gamma", "away_team": "Delta", "home_goals": 1, "away_goals": 0},
            {"group": "A", "home_team": "Alpha", "away_team": "Gamma", "home_goals": 0, "away_goals": 1},
            {"group": "A", "home_team": "Delta", "away_team": "Beta", "home_goals": 1, "away_goals": 0},
            {"group": "A", "home_team": "Delta", "away_team": "Alpha", "home_goals": 0, "away_goals": 1},
            {"group": "A", "home_team": "Beta", "away_team": "Gamma", "home_goals": 1, "away_goals": 0},
        ]
    )

    standings = rank_group_standings(group_results).sort_values("group_rank", kind="stable")

    assert standings.iloc[0]["team"] == "Gamma"
    assert standings.iloc[1]["team"] == "Alpha"
    assert standings.iloc[0]["points"] == standings.iloc[1]["points"] == 6
    assert standings.iloc[0]["goal_difference"] == standings.iloc[1]["goal_difference"] == 1


def test_resolve_round_of_32_matchups_uses_2026_lookup_table() -> None:
    standings_rows = []
    for group_name in "ABCDEFGHIJKL":
        standings_rows.extend(
            [
                {"group": group_name, "team": f"{group_name}1", "group_rank": 1},
                {"group": group_name, "team": f"{group_name}2", "group_rank": 2},
                {"group": group_name, "team": f"{group_name}3", "group_rank": 3},
                {"group": group_name, "team": f"{group_name}4", "group_rank": 4},
            ]
        )
    standings = pd.DataFrame(standings_rows)

    third_place_ranking = pd.DataFrame(
        [
            {"group": "E", "team": "E3", "third_place_rank": 1},
            {"group": "F", "team": "F3", "third_place_rank": 2},
            {"group": "G", "team": "G3", "third_place_rank": 3},
            {"group": "H", "team": "H3", "third_place_rank": 4},
            {"group": "I", "team": "I3", "third_place_rank": 5},
            {"group": "J", "team": "J3", "third_place_rank": 6},
            {"group": "K", "team": "K3", "third_place_rank": 7},
            {"group": "L", "team": "L3", "third_place_rank": 8},
            {"group": "A", "team": "A3", "third_place_rank": 9},
            {"group": "B", "team": "B3", "third_place_rank": 10},
            {"group": "C", "team": "C3", "third_place_rank": 11},
            {"group": "D", "team": "D3", "third_place_rank": 12},
        ]
    )

    round_of_32 = resolve_round_of_32_matchups(standings, third_place_ranking)

    match_74 = round_of_32.loc[round_of_32["match_number"] == 74].iloc[0]
    match_79 = round_of_32.loc[round_of_32["match_number"] == 79].iloc[0]
    match_80 = round_of_32.loc[round_of_32["match_number"] == 80].iloc[0]
    match_85 = round_of_32.loc[round_of_32["match_number"] == 85].iloc[0]

    assert match_74["home_team"] == "E1"
    assert match_74["away_team"] == "F3"
    assert match_79["home_team"] == "A1"
    assert match_79["away_team"] == "E3"
    assert match_80["away_team"] == "K3"
    assert match_85["away_team"] == "J3"


def test_simulate_world_cup_2026_produces_coherent_probability_mass() -> None:
    groups = {
        group_name: [f"{group_name}{slot}" for slot in range(1, 5)]
        for group_name in "ABCDEFGHIJKL"
    }
    elo_ratings = {team: 1500.0 for teams in groups.values() for team in teams}

    probabilities = simulate_world_cup_2026(
        model=ToyTournamentModel(),
        groups=groups,
        elo_ratings=elo_ratings,
        simulations=25,
        random_state=7,
        max_goals=2,
    )

    assert len(probabilities) == 48
    assert probabilities["group_winner_probability"].sum() == pytest.approx(12.0)
    assert probabilities["reach_round_of_32_probability"].sum() == pytest.approx(32.0)
    assert probabilities["reach_round_of_16_probability"].sum() == pytest.approx(16.0)
    assert probabilities["reach_quarterfinal_probability"].sum() == pytest.approx(8.0)
    assert probabilities["reach_semifinal_probability"].sum() == pytest.approx(4.0)
    assert probabilities["reach_final_probability"].sum() == pytest.approx(2.0)
    assert probabilities["champion_probability"].sum() == pytest.approx(1.0)
    assert probabilities["average_group_points"].between(0.0, 9.0).all()


def test_build_group_stage_schedule_creates_six_matches_per_group() -> None:
    groups = {"A": ["Alpha", "Beta", "Gamma", "Delta"]}

    schedule = build_group_stage_schedule(groups)

    assert len(schedule) == 6
    assert set(schedule["group_match_number"]) == {1, 2, 3, 4, 5, 6}
    assert set(schedule["home_team"]).union(schedule["away_team"]) == {
        "Alpha",
        "Beta",
        "Gamma",
        "Delta",
    }
