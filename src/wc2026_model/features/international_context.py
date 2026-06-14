from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from wc2026_model.data import canonicalize_team_name
from wc2026_model.types import OUTCOME_AWAY, OUTCOME_DRAW, OUTCOME_HOME

CONFEDERATIONS = ("AFC", "CAF", "CONCACAF", "CONMEBOL", "OFC", "UEFA")
DEFAULT_H2H_DECAY_HALF_LIFE_DAYS = 365.25 * 4.0

TEAM_CONFEDERATION = {
    "Albania": "UEFA",
    "Algeria": "CAF",
    "Angola": "CAF",
    "Argentina": "CONMEBOL",
    "Australia": "AFC",
    "Austria": "UEFA",
    "Belgium": "UEFA",
    "Bolivia": "CONMEBOL",
    "Bosnia and Herzegovina": "UEFA",
    "Brazil": "CONMEBOL",
    "Burkina Faso": "CAF",
    "Cameroon": "CAF",
    "Canada": "CONCACAF",
    "Cape Verde": "CAF",
    "Chile": "CONMEBOL",
    "Colombia": "CONMEBOL",
    "Costa Rica": "CONCACAF",
    "Croatia": "UEFA",
    "Curaçao": "CONCACAF",
    "CuraÃ§ao": "CONCACAF",
    "Czech Republic": "UEFA",
    "DR Congo": "CAF",
    "Denmark": "UEFA",
    "Ecuador": "CONMEBOL",
    "Egypt": "CAF",
    "England": "UEFA",
    "Equatorial Guinea": "CAF",
    "Finland": "UEFA",
    "France": "UEFA",
    "Gambia": "CAF",
    "Georgia": "UEFA",
    "Germany": "UEFA",
    "Ghana": "CAF",
    "Guinea": "CAF",
    "Guinea-Bissau": "CAF",
    "Haiti": "CONCACAF",
    "Hungary": "UEFA",
    "Iceland": "UEFA",
    "Iran": "AFC",
    "Iraq": "AFC",
    "Italy": "UEFA",
    "Ivory Coast": "CAF",
    "Jamaica": "CONCACAF",
    "Japan": "AFC",
    "Jordan": "AFC",
    "Mali": "CAF",
    "Mauritania": "CAF",
    "Mexico": "CONCACAF",
    "Morocco": "CAF",
    "Mozambique": "CAF",
    "Namibia": "CAF",
    "Netherlands": "UEFA",
    "New Zealand": "OFC",
    "Nigeria": "CAF",
    "North Macedonia": "UEFA",
    "Norway": "UEFA",
    "Panama": "CONCACAF",
    "Paraguay": "CONMEBOL",
    "Peru": "CONMEBOL",
    "Poland": "UEFA",
    "Portugal": "UEFA",
    "Qatar": "AFC",
    "Romania": "UEFA",
    "Russia": "UEFA",
    "Saudi Arabia": "AFC",
    "Scotland": "UEFA",
    "Senegal": "CAF",
    "Serbia": "UEFA",
    "Slovakia": "UEFA",
    "Slovenia": "UEFA",
    "South Africa": "CAF",
    "South Korea": "AFC",
    "Spain": "UEFA",
    "Sweden": "UEFA",
    "Switzerland": "UEFA",
    "Tanzania": "CAF",
    "Tunisia": "CAF",
    "Turkey": "UEFA",
    "Ukraine": "UEFA",
    "United States": "CONCACAF",
    "Uruguay": "CONMEBOL",
    "Uzbekistan": "AFC",
    "Venezuela": "CONMEBOL",
    "Wales": "UEFA",
    "Zambia": "CAF",
}


def get_team_confederation(team: str) -> str | None:
    canonical = canonicalize_team_name(team)
    return TEAM_CONFEDERATION.get(canonical)


def attach_confederation_features(matches: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"home_team", "away_team"}
    missing_columns = required_columns.difference(matches.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required match columns for confederation features: {missing}")

    dataframe = matches.copy()
    dataframe["home_team"] = dataframe["home_team"].astype(str).map(canonicalize_team_name)
    dataframe["away_team"] = dataframe["away_team"].astype(str).map(canonicalize_team_name)
    dataframe["home_confederation"] = dataframe["home_team"].map(get_team_confederation)
    dataframe["away_confederation"] = dataframe["away_team"].map(get_team_confederation)
    dataframe["same_confederation"] = (
        dataframe["home_confederation"].notna()
        & dataframe["home_confederation"].eq(dataframe["away_confederation"])
    ).astype(float)
    for confederation in CONFEDERATIONS:
        slug = confederation.lower()
        dataframe[f"home_is_{slug}"] = (
            dataframe["home_confederation"].eq(confederation).astype(float)
        )
        dataframe[f"away_is_{slug}"] = (
            dataframe["away_confederation"].eq(confederation).astype(float)
        )
    return dataframe


def augment_with_pre_match_h2h_features(
    matches: pd.DataFrame,
    *,
    decay_half_life_days: float = DEFAULT_H2H_DECAY_HALF_LIFE_DAYS,
) -> pd.DataFrame:
    required_columns = {"match_date", "home_team", "away_team"}
    missing_columns = required_columns.difference(matches.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required match columns for h2h augmentation: {missing}")
    if decay_half_life_days <= 0.0:
        raise ValueError("decay_half_life_days must be positive")

    dataframe = matches.copy()
    dataframe["match_date"] = pd.to_datetime(dataframe["match_date"], errors="raise")
    dataframe["home_team"] = dataframe["home_team"].astype(str).map(canonicalize_team_name)
    dataframe["away_team"] = dataframe["away_team"].astype(str).map(canonicalize_team_name)
    dataframe = dataframe.sort_values(
        ["match_date", "home_team", "away_team"],
        kind="stable",
    ).reset_index(drop=True)

    directional_histories: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: {"match_count": 0.0, "home_wins": 0.0, "draws": 0.0, "away_wins": 0.0}
    )
    directional_events: dict[tuple[str, str], list[tuple[pd.Timestamp, str]]] = defaultdict(list)

    records: list[dict[str, float]] = []
    for row in dataframe.itertuples(index=False):
        home_team = str(row.home_team)
        away_team = str(row.away_team)
        match_date = pd.Timestamp(row.match_date)
        history = directional_histories[(home_team, away_team)]
        record = _summarize_h2h_history(history)
        record.update(
            _summarize_decayed_h2h_history(
                directional_events[(home_team, away_team)],
                as_of_date=match_date,
                half_life_days=decay_half_life_days,
            )
        )
        records.append(record)

        outcome = _resolve_home_result(row)
        _apply_directional_h2h_update(
            directional_histories[(home_team, away_team)],
            outcome=outcome,
        )
        _append_directional_h2h_event(
            directional_events[(home_team, away_team)],
            match_date=match_date,
            outcome=outcome,
        )
        reverse_outcome = {
            OUTCOME_HOME: OUTCOME_AWAY,
            OUTCOME_DRAW: OUTCOME_DRAW,
            OUTCOME_AWAY: OUTCOME_HOME,
        }[outcome]
        _apply_directional_h2h_update(
            directional_histories[(away_team, home_team)],
            outcome=reverse_outcome,
        )
        _append_directional_h2h_event(
            directional_events[(away_team, home_team)],
            match_date=match_date,
            outcome=reverse_outcome,
        )

    h2h_frame = pd.DataFrame.from_records(records)
    return pd.concat([dataframe, h2h_frame], axis=1)


def attach_fixture_h2h_features(
    fixtures: pd.DataFrame,
    historical_matches: pd.DataFrame,
    *,
    as_of_date_column: str = "match_date",
) -> pd.DataFrame:
    required_fixture_columns = {"home_team", "away_team", as_of_date_column}
    missing_fixture_columns = required_fixture_columns.difference(fixtures.columns)
    if missing_fixture_columns:
        missing = ", ".join(sorted(missing_fixture_columns))
        raise ValueError(f"Missing required fixture columns for h2h attachment: {missing}")

    historical = augment_with_pre_match_h2h_features(historical_matches)
    latest_lookup: dict[tuple[str, str], dict[str, float]] = {}
    directional_histories: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: {"match_count": 0.0, "home_wins": 0.0, "draws": 0.0, "away_wins": 0.0}
    )
    historical_sorted = historical.sort_values(
        ["match_date", "home_team", "away_team"],
        kind="stable",
    ).reset_index(drop=True)
    for row in historical_sorted.itertuples(index=False):
        home_team = str(row.home_team)
        away_team = str(row.away_team)
        outcome = _resolve_home_result(row)
        _apply_directional_h2h_update(
            directional_histories[(home_team, away_team)],
            outcome=outcome,
        )
        reverse_outcome = {
            OUTCOME_HOME: OUTCOME_AWAY,
            OUTCOME_DRAW: OUTCOME_DRAW,
            OUTCOME_AWAY: OUTCOME_HOME,
        }[outcome]
        _apply_directional_h2h_update(
            directional_histories[(away_team, home_team)],
            outcome=reverse_outcome,
        )
        latest_lookup[(home_team, away_team)] = directional_histories[(home_team, away_team)].copy()
        latest_lookup[(away_team, home_team)] = directional_histories[(away_team, home_team)].copy()

    enriched = fixtures.copy()
    enriched["home_team"] = enriched["home_team"].astype(str).map(canonicalize_team_name)
    enriched["away_team"] = enriched["away_team"].astype(str).map(canonicalize_team_name)
    h2h_rows = []
    for row in enriched.itertuples(index=False):
        history = latest_lookup.get(
            (str(row.home_team), str(row.away_team)),
            {"match_count": 0.0, "home_wins": 0.0, "draws": 0.0, "away_wins": 0.0},
        )
        h2h_rows.append(_summarize_h2h_history(history))
    h2h_frame = pd.DataFrame.from_records(h2h_rows)
    return pd.concat([enriched.reset_index(drop=True), h2h_frame], axis=1)


def _resolve_home_result(row: object) -> str:
    home_result = getattr(row, "home_result", None)
    if isinstance(home_result, str) and home_result in {OUTCOME_HOME, OUTCOME_DRAW, OUTCOME_AWAY}:
        return home_result
    home_goals = float(getattr(row, "home_goals"))
    away_goals = float(getattr(row, "away_goals"))
    if home_goals > away_goals:
        return OUTCOME_HOME
    if home_goals < away_goals:
        return OUTCOME_AWAY
    return OUTCOME_DRAW


def _apply_directional_h2h_update(history: dict[str, float], *, outcome: str) -> None:
    history["match_count"] += 1.0
    if outcome == OUTCOME_HOME:
        history["home_wins"] += 1.0
    elif outcome == OUTCOME_DRAW:
        history["draws"] += 1.0
    else:
        history["away_wins"] += 1.0


def _append_directional_h2h_event(
    history: list[tuple[pd.Timestamp, str]],
    *,
    match_date: pd.Timestamp,
    outcome: str,
) -> None:
    history.append((match_date, outcome))


def _summarize_h2h_history(history: dict[str, float]) -> dict[str, float]:
    match_count = float(history["match_count"])
    if match_count <= 0.0:
        return {
            "h2h_match_count": 0.0,
            "h2h_home_win_rate": np.nan,
            "h2h_draw_rate": np.nan,
            "h2h_away_win_rate": np.nan,
        }
    return {
        "h2h_match_count": match_count,
        "h2h_home_win_rate": float(history["home_wins"] / match_count),
        "h2h_draw_rate": float(history["draws"] / match_count),
        "h2h_away_win_rate": float(history["away_wins"] / match_count),
    }


def _summarize_decayed_h2h_history(
    history: list[tuple[pd.Timestamp, str]],
    *,
    as_of_date: pd.Timestamp,
    half_life_days: float,
) -> dict[str, float]:
    if not history:
        return {
            "h2h_decayed_match_weight": 0.0,
            "h2h_decayed_home_win_rate": np.nan,
            "h2h_decayed_draw_rate": np.nan,
            "h2h_decayed_away_win_rate": np.nan,
            "h2h_days_since_last_match": np.nan,
        }

    weights = []
    weighted_home_wins = 0.0
    weighted_draws = 0.0
    weighted_away_wins = 0.0
    latest_match_date = max(match_date for match_date, _ in history)
    for match_date, outcome in history:
        age_days = max(float((as_of_date - match_date).days), 0.0)
        weight = 0.5 ** (age_days / half_life_days)
        weights.append(weight)
        if outcome == OUTCOME_HOME:
            weighted_home_wins += weight
        elif outcome == OUTCOME_DRAW:
            weighted_draws += weight
        else:
            weighted_away_wins += weight

    total_weight = float(sum(weights))
    if total_weight <= 0.0:
        return {
            "h2h_decayed_match_weight": 0.0,
            "h2h_decayed_home_win_rate": np.nan,
            "h2h_decayed_draw_rate": np.nan,
            "h2h_decayed_away_win_rate": np.nan,
            "h2h_days_since_last_match": np.nan,
        }

    return {
        "h2h_decayed_match_weight": total_weight,
        "h2h_decayed_home_win_rate": float(weighted_home_wins / total_weight),
        "h2h_decayed_draw_rate": float(weighted_draws / total_weight),
        "h2h_decayed_away_win_rate": float(weighted_away_wins / total_weight),
        "h2h_days_since_last_match": float((as_of_date - latest_match_date).days),
    }
