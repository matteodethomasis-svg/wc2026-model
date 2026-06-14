from __future__ import annotations

from pathlib import Path

import pandas as pd

from .international_results import canonicalize_team_name
from .live_provider_feeds import load_expected_lineups_feed, load_injuries_feed

_REGISTRY_COLUMNS = [
    "group",
    "slot",
    "team",
    "sportmonks_team_id",
    "sportmonks_team_name",
    "api_football_team_id",
    "api_football_team_name",
    "notes",
]


def build_wc2026_provider_team_registry(
    *,
    groups: pd.DataFrame | None = None,
    squads: pd.DataFrame | None = None,
) -> pd.DataFrame:
    teams = _resolve_wc2026_team_universe(groups=groups, squads=squads)
    if groups is not None and not groups.empty:
        teams = teams.merge(
            groups.copy(),
            on=[column for column in ("group", "slot", "team") if column in groups.columns and column in teams.columns],
            how="left",
        )
        teams = teams.loc[:, ~teams.columns.duplicated()].copy()
    registry = teams.copy()
    return normalize_provider_team_registry(registry)


def enrich_registry_with_sportmonks_feed(
    registry: pd.DataFrame,
    expected_lineups: pd.DataFrame,
) -> pd.DataFrame:
    return _enrich_registry_with_provider_ids(
        registry,
        provider_frame=expected_lineups,
        provider_column="sportmonks_team_id",
        provider_name_column="sportmonks_team_name",
    )


def enrich_registry_with_api_football_feed(
    registry: pd.DataFrame,
    injuries: pd.DataFrame,
) -> pd.DataFrame:
    return _enrich_registry_with_provider_ids(
        registry,
        provider_frame=injuries,
        provider_column="api_football_team_id",
        provider_name_column="api_football_team_name",
    )


def build_wc2026_provider_team_registry_summary(registry: pd.DataFrame) -> dict[str, object]:
    _require_registry_columns(registry)
    summary = {
        "team_count": int(len(registry)),
        "groups_count": int(registry["group"].replace("", pd.NA).dropna().nunique()),
        "sportmonks_team_ids_filled": int(registry["sportmonks_team_id"].notna().sum()),
        "api_football_team_ids_filled": int(registry["api_football_team_id"].notna().sum()),
        "missing_sportmonks_teams": registry.loc[
            registry["sportmonks_team_id"].isna(),
            "team",
        ].astype(str).tolist(),
        "missing_api_football_teams": registry.loc[
            registry["api_football_team_id"].isna(),
            "team",
        ].astype(str).tolist(),
    }
    return summary


def load_provider_team_registry(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return normalize_provider_team_registry(frame)


def load_and_enrich_registry_from_feeds(
    *,
    registry: pd.DataFrame,
    sportmonks_input: str | Path | None = None,
    api_football_input: str | Path | None = None,
) -> pd.DataFrame:
    enriched = registry.copy()
    if sportmonks_input is not None:
        expected_lineups = load_expected_lineups_feed(sportmonks_input, provider="auto")
        enriched = enrich_registry_with_sportmonks_feed(enriched, expected_lineups)
    if api_football_input is not None:
        injuries = load_injuries_feed(api_football_input, provider="auto")
        enriched = enrich_registry_with_api_football_feed(enriched, injuries)
    return enriched


def read_provider_team_ids(
    registry_path: str | Path,
    *,
    provider: str,
) -> list[int]:
    registry = pd.read_csv(registry_path)
    provider_key = provider.strip().lower()
    if provider_key == "sportmonks":
        column = "sportmonks_team_id"
    elif provider_key == "api_football":
        column = "api_football_team_id"
    else:
        raise ValueError(f"Unsupported provider '{provider}'.")

    if column not in registry.columns:
        raise ValueError(f"Provider registry is missing column '{column}'.")

    series = pd.to_numeric(registry[column], errors="coerce").dropna().astype(int)
    return series.drop_duplicates().tolist()


def _resolve_wc2026_team_universe(
    *,
    groups: pd.DataFrame | None,
    squads: pd.DataFrame | None,
) -> pd.DataFrame:
    if groups is not None and not groups.empty:
        _require_columns(groups, required={"team"}, frame_name="groups")
        frame = groups.copy()
        frame["team"] = frame["team"].astype(str).map(canonicalize_team_name)
        if "group" not in frame.columns:
            frame["group"] = ""
        if "slot" not in frame.columns:
            frame["slot"] = pd.NA
        frame = frame.loc[:, ["group", "slot", "team"]].drop_duplicates("team", keep="first")
        return frame.reset_index(drop=True)

    if squads is not None and not squads.empty:
        _require_columns(squads, required={"team"}, frame_name="squads")
        teams = (
            squads.loc[:, ["team"]]
            .assign(group="", slot=pd.NA)
            .drop_duplicates("team", keep="first")
            .reset_index(drop=True)
        )
        teams["team"] = teams["team"].astype(str).map(canonicalize_team_name)
        return teams.loc[:, ["group", "slot", "team"]]

    raise ValueError("Provide at least one non-empty groups or squads dataframe.")


def normalize_provider_team_registry(registry: pd.DataFrame) -> pd.DataFrame:
    _require_columns(registry, required={"team"}, frame_name="registry")
    normalized = registry.copy()
    for column in _REGISTRY_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = pd.NA
    normalized = normalized.loc[:, _REGISTRY_COLUMNS].copy()
    normalized["group"] = normalized["group"].fillna("").astype(str)
    normalized["slot"] = pd.to_numeric(normalized["slot"], errors="coerce")
    normalized["team"] = normalized["team"].astype(str).map(canonicalize_team_name)
    for id_column in ("sportmonks_team_id", "api_football_team_id"):
        normalized[id_column] = pd.to_numeric(normalized[id_column], errors="coerce")
    normalized["sportmonks_team_name"] = normalized["sportmonks_team_name"].fillna("").astype(str)
    normalized["api_football_team_name"] = normalized["api_football_team_name"].fillna("").astype(str)
    normalized["notes"] = normalized["notes"].fillna("").astype(str)
    normalized = normalized.drop_duplicates("team", keep="first")
    return normalized.sort_values(["group", "slot", "team"], kind="stable").reset_index(drop=True)


def _enrich_registry_with_provider_ids(
    registry: pd.DataFrame,
    *,
    provider_frame: pd.DataFrame,
    provider_column: str,
    provider_name_column: str,
) -> pd.DataFrame:
    _require_registry_columns(registry)
    if provider_frame.empty:
        return registry.copy()
    _require_columns(provider_frame, required={"team", "team_id"}, frame_name="provider_frame")

    normalized_registry = registry.copy()
    normalized_registry["team"] = normalized_registry["team"].astype(str).map(canonicalize_team_name)

    provider_lookup = _build_provider_team_lookup(provider_frame)
    provider_ids = []
    provider_names = []
    notes = []
    for row in normalized_registry.itertuples(index=False):
        team = canonicalize_team_name(str(row.team))
        existing_id = getattr(row, provider_column)
        existing_name = getattr(row, provider_name_column)
        match = provider_lookup.get(team)
        provider_ids.append(existing_id if pd.notna(existing_id) else (match or {}).get("team_id"))
        provider_names.append(
            existing_name if str(existing_name).strip() != "" else (match or {}).get("team_name", "")
        )
        notes.append(_merge_note_strings(str(getattr(row, "notes")), (match or {}).get("note", "")))

    normalized_registry[provider_column] = provider_ids
    normalized_registry[provider_name_column] = provider_names
    normalized_registry["notes"] = notes
    return normalized_registry


def _build_provider_team_lookup(provider_frame: pd.DataFrame) -> dict[str, dict[str, object]]:
    frame = provider_frame.copy()
    frame["team"] = frame["team"].astype(str).map(canonicalize_team_name)
    frame["team_id"] = pd.to_numeric(frame["team_id"], errors="coerce")
    if "source" not in frame.columns:
        frame["source"] = ""

    rows: dict[str, dict[str, object]] = {}
    for team, group in frame.groupby("team", sort=True):
        valid_ids = group["team_id"].dropna().astype(int).drop_duplicates().tolist()
        team_names = (
            group["team"].astype(str).dropna().drop_duplicates().tolist()
        )
        note = ""
        if len(valid_ids) > 1:
            note = f"multiple_ids_detected:{','.join(map(str, valid_ids))}"
        rows[team] = {
            "team_id": valid_ids[0] if valid_ids else pd.NA,
            "team_name": team_names[0] if team_names else "",
            "note": note,
        }
    return rows


def _merge_note_strings(left: str, right: str) -> str:
    left_clean = left.strip()
    right_clean = right.strip()
    if left_clean == "":
        return right_clean
    if right_clean == "" or right_clean in left_clean:
        return left_clean
    return f"{left_clean}; {right_clean}"


def _require_registry_columns(registry: pd.DataFrame) -> None:
    _require_columns(registry, required={"team", "notes"}, frame_name="registry")


def _require_columns(
    dataframe: pd.DataFrame,
    *,
    required: set[str],
    frame_name: str,
) -> None:
    missing = required.difference(dataframe.columns)
    if missing:
        missing_columns = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns in {frame_name}: {missing_columns}")
