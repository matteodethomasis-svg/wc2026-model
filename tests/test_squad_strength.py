from __future__ import annotations

import math

import pandas as pd

from wc2026_model.features import (
    aggregate_team_squad_strength,
    attach_team_strength_ratings,
    build_squad_player_club_elo_frame,
    normalize_club_name,
)


def test_normalize_club_name_handles_common_aliases() -> None:
    assert normalize_club_name("Paris Saint-Germain") == "paris sg"
    assert normalize_club_name("Manchester City") == "man city"
    assert normalize_club_name("AC Milan") == "milan"


def test_build_squad_player_club_elo_frame_matches_alias_and_fuzzy_clubs() -> None:
    squads = pd.DataFrame(
        [
            {"team": "France", "player": "Kylian Mbappe", "club": "Paris Saint-Germain", "caps": 94, "age": 27},
            {"team": "France", "player": "Aurelien Tchouameni", "club": "Real Madrid", "caps": 41, "age": 26},
            {"team": "Norway", "player": "Erling Haaland", "club": "Manchester City", "caps": 47, "age": 25},
            {"team": "Norway", "player": "Martin Odegaard", "club": "Arsenal FC", "caps": 67, "age": 27},
        ]
    )
    club_elo = pd.DataFrame(
        [
            {"club": "Paris SG", "club_elo": 1968.0},
            {"club": "Real Madrid", "club_elo": 1923.0},
            {"club": "Man City", "club_elo": 1971.0},
            {"club": "Arsenal", "club_elo": 2064.0},
        ]
    )

    enriched = build_squad_player_club_elo_frame(squads, club_elo)

    assert enriched["matched_club"].tolist() == [
        "Paris SG",
        "Real Madrid",
        "Man City",
        "Arsenal",
    ]
    assert enriched["club_elo_match_method"].tolist() == ["exact", "exact", "exact", "exact"]
    assert enriched["club_elo"].tolist() == [1968.0, 1923.0, 1971.0, 2064.0]


def test_aggregate_team_squad_strength_and_attach_ratings() -> None:
    squad_players = pd.DataFrame(
        [
            {"team": "France", "player": "Kylian Mbappe", "club_elo": 1968.0, "caps": 94, "age": 27},
            {"team": "France", "player": "Aurelien Tchouameni", "club_elo": 1923.0, "caps": 41, "age": 26},
            {"team": "Norway", "player": "Erling Haaland", "club_elo": 1971.0, "caps": 47, "age": 25},
            {"team": "Norway", "player": "Martin Odegaard", "club_elo": 2064.0, "caps": 67, "age": 27},
        ]
    )

    strengths = aggregate_team_squad_strength(
        squad_players,
        top_player_count=2,
        core_player_count=2,
        star_player_count=1,
    )
    france = strengths.loc[strengths["team"] == "France"].iloc[0]
    norway = strengths.loc[strengths["team"] == "Norway"].iloc[0]

    assert math.isclose(float(france["squad_club_elo_rating"]), 1945.5)
    assert math.isclose(float(norway["squad_club_elo_rating"]), 2017.5)
    assert math.isclose(float(norway["squad_club_elo_star_rating"]), 2064.0)
    assert float(france["mapped_player_share"]) == 1.0

    matches = pd.DataFrame(
        [
            {"home_team": "France", "away_team": "Norway"},
            {"home_team": "Norway", "away_team": "France"},
        ]
    )
    attached = attach_team_strength_ratings(matches, strengths)

    assert math.isclose(float(attached.iloc[0]["team_strength_rating_diff"]), -72.0)
    assert math.isclose(float(attached.iloc[1]["team_strength_rating_diff"]), 72.0)


def test_aggregate_team_squad_strength_shrinks_partial_coverage() -> None:
    squad_players = pd.DataFrame(
        [
            {"team": "A", "player": "One", "club_elo": 2000.0},
            {"team": "B", "player": "One", "club_elo": 1800.0},
            {"team": "B", "player": "Two", "club_elo": 1790.0},
        ]
    )

    strengths = aggregate_team_squad_strength(
        squad_players,
        top_player_count=2,
        core_player_count=2,
        star_player_count=1,
        fallback_club_elo=1700.0,
    )
    team_a = strengths.loc[strengths["team"] == "A"].iloc[0]
    team_b = strengths.loc[strengths["team"] == "B"].iloc[0]

    assert math.isclose(float(team_a["mapped_only_squad_club_elo_rating"]), 2000.0)
    assert math.isclose(float(team_a["squad_club_elo_rating"]), 1850.0)
    assert math.isclose(float(team_b["squad_club_elo_rating"]), 1795.0)


def test_aggregate_team_squad_strength_builds_expected_xi_proxy() -> None:
    squad_players = pd.DataFrame(
        [
            {"team": "Alpha", "player": "GK1", "position": "GK", "club_elo": 1800.0, "caps": 40, "goals": 0, "age": 28},
            {"team": "Alpha", "player": "GK2", "position": "GK", "club_elo": 1700.0, "caps": 5, "goals": 0, "age": 23},
            {"team": "Alpha", "player": "DF1", "position": "DF", "club_elo": 1900.0, "caps": 50, "goals": 2, "age": 27},
            {"team": "Alpha", "player": "DF2", "position": "DF", "club_elo": 1890.0, "caps": 45, "goals": 1, "age": 29},
            {"team": "Alpha", "player": "DF3", "position": "DF", "club_elo": 1880.0, "caps": 40, "goals": 0, "age": 26},
            {"team": "Alpha", "player": "DF4", "position": "DF", "club_elo": 1870.0, "caps": 35, "goals": 0, "age": 28},
            {"team": "Alpha", "player": "DF5", "position": "DF", "club_elo": 1750.0, "caps": 10, "goals": 0, "age": 23},
            {"team": "Alpha", "player": "MF1", "position": "MF", "club_elo": 1820.0, "caps": 42, "goals": 6, "age": 27},
            {"team": "Alpha", "player": "MF2", "position": "MF", "club_elo": 1810.0, "caps": 38, "goals": 4, "age": 28},
            {"team": "Alpha", "player": "MF3", "position": "MF", "club_elo": 1800.0, "caps": 30, "goals": 3, "age": 25},
            {"team": "Alpha", "player": "MF4", "position": "MF", "club_elo": 1790.0, "caps": 20, "goals": 2, "age": 24},
            {"team": "Alpha", "player": "MF5", "position": "MF", "club_elo": 1780.0, "caps": 18, "goals": 1, "age": 22},
            {"team": "Alpha", "player": "FW1", "position": "FW", "club_elo": 1860.0, "caps": 44, "goals": 20, "age": 27},
            {"team": "Alpha", "player": "FW2", "position": "FW", "club_elo": 1850.0, "caps": 36, "goals": 12, "age": 29},
            {"team": "Alpha", "player": "FW3", "position": "FW", "club_elo": 1840.0, "caps": 25, "goals": 10, "age": 26},
            {"team": "Alpha", "player": "FW4", "position": "FW", "club_elo": 1760.0, "caps": 15, "goals": 4, "age": 24},
        ]
    )

    strengths = aggregate_team_squad_strength(
        squad_players,
        top_player_count=15,
        core_player_count=11,
        star_player_count=3,
        fallback_club_elo=1700.0,
    )
    alpha = strengths.loc[strengths["team"] == "Alpha"].iloc[0]

    assert alpha["expected_xi_formation"] == "4-3-3"
    assert math.isclose(float(alpha["expected_xi_club_elo_rating"]), 1847.2727272727273)
    assert math.isclose(float(alpha["expected_xi_mapped_player_share"]), 1.0)
    assert math.isclose(float(alpha["expected_xi_goalkeeper_club_elo_rating"]), 1800.0)
    assert math.isclose(float(alpha["expected_xi_defense_club_elo_rating"]), 1885.0)
    assert math.isclose(float(alpha["expected_xi_midfield_club_elo_rating"]), 1810.0)
    assert math.isclose(float(alpha["expected_xi_attack_club_elo_rating"]), 1850.0)
    assert float(alpha["expected_xi_selection_score"]) > float(alpha["expected_xi_club_elo_rating"])
