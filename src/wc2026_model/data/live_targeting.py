from __future__ import annotations

from pathlib import Path

import pandas as pd

from .international_results import canonicalize_team_name, load_scheduled_matches
from .provider_team_registry import normalize_provider_team_registry

_PROVIDER_ID_COLUMNS = {
    "sportmonks": "sportmonks_team_id",
    "api_football": "api_football_team_id",
}


def build_upcoming_fixture_team_window(
    fixtures_input: str | Path,
    *,
    tournament: str,
    start_date: str,
    window_days: int,
) -> pd.DataFrame:
    if window_days <= 0:
        raise ValueError("window_days must be a positive integer.")

    fixtures = load_scheduled_matches(
        fixtures_input,
        tournament=tournament,
        start_date=start_date,
    )
    if fixtures.empty:
        return pd.DataFrame(
            columns=["match_id", "match_date", "team", "opponent", "team_role", "neutral"]
        )

    window_end = pd.Timestamp(start_date) + pd.Timedelta(days=window_days - 1)
    window_fixtures = fixtures.loc[fixtures["match_date"] <= window_end].copy()

    rows: list[dict[str, object]] = []
    for row in window_fixtures.itertuples(index=False):
        rows.append(
            {
                "match_id": row.match_id,
                "match_date": row.match_date,
                "team": canonicalize_team_name(str(row.home_team)),
                "opponent": canonicalize_team_name(str(row.away_team)),
                "team_role": "home",
                "neutral": bool(row.neutral),
            }
        )
        rows.append(
            {
                "match_id": row.match_id,
                "match_date": row.match_date,
                "team": canonicalize_team_name(str(row.away_team)),
                "opponent": canonicalize_team_name(str(row.home_team)),
                "team_role": "away",
                "neutral": bool(row.neutral),
            }
        )

    return pd.DataFrame.from_records(rows).sort_values(
        ["match_date", "match_id"],
        kind="stable",
    ).reset_index(drop=True)


def select_provider_team_ids_for_fixture_window(
    registry: pd.DataFrame,
    *,
    provider: str,
    fixtures_input: str | Path,
    tournament: str,
    start_date: str,
    window_days: int,
    max_teams: int | None = None,
) -> dict[str, object]:
    provider_key = provider.strip().lower()
    if provider_key not in _PROVIDER_ID_COLUMNS:
        raise ValueError(f"Unsupported provider '{provider}'.")

    normalized_registry = normalize_provider_team_registry(registry)
    target_window = build_upcoming_fixture_team_window(
        fixtures_input,
        tournament=tournament,
        start_date=start_date,
        window_days=window_days,
    )

    ordered_target_teams = _ordered_unique(target_window["team"].astype(str).tolist())
    if max_teams is not None:
        ordered_target_teams = ordered_target_teams[:max_teams]

    provider_column = _PROVIDER_ID_COLUMNS[provider_key]
    registry_lookup = (
        normalized_registry.loc[:, ["team", provider_column]]
        .drop_duplicates("team", keep="first")
        .set_index("team")[provider_column]
        .to_dict()
    )

    selected_team_ids: list[int] = []
    missing_teams: list[str] = []
    covered_teams: list[str] = []
    for team in ordered_target_teams:
        team_id = registry_lookup.get(team)
        if pd.isna(team_id):
            missing_teams.append(team)
            continue
        covered_teams.append(team)
        selected_team_ids.append(int(team_id))

    if ordered_target_teams:
        filtered_window = target_window.loc[target_window["team"].isin(ordered_target_teams)].copy()
    else:
        filtered_window = target_window.iloc[0:0].copy()

    return {
        "provider": provider_key,
        "provider_column": provider_column,
        "team_ids": _ordered_unique(selected_team_ids),
        "target_teams": ordered_target_teams,
        "covered_teams": covered_teams,
        "missing_teams": missing_teams,
        "window_days": int(window_days),
        "window_match_count": int(filtered_window["match_id"].nunique()) if not filtered_window.empty else 0,
        "window_team_count": int(len(ordered_target_teams)),
        "window_frame": filtered_window,
    }


def _ordered_unique(values: list[object]) -> list[object]:
    return list(dict.fromkeys(values))
