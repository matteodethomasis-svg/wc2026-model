from __future__ import annotations

import json
from collections.abc import Sequence
from urllib.request import urlopen

import pandas as pd

from .international_results import (
    canonicalize_team_name,
    canonicalize_tournament_name,
    normalize_standardized_results,
)

DEFAULT_STATSBOMB_OPEN_DATA_BASE_URL = (
    "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
)

_SHOTS_ON_TARGET_OUTCOMES = {"Goal", "Saved", "Saved to Post"}


def fetch_statsbomb_competitions(
    *,
    base_url: str = DEFAULT_STATSBOMB_OPEN_DATA_BASE_URL,
) -> pd.DataFrame:
    payload = _load_statsbomb_json("competitions.json", base_url=base_url)
    if not isinstance(payload, list):
        raise ValueError("StatsBomb competitions payload must be a list.")

    frame = pd.DataFrame.from_records(payload)
    if frame.empty:
        return frame

    if "competition_name" in frame.columns:
        frame["competition_name"] = frame["competition_name"].map(canonicalize_tournament_name)
    return frame.sort_values(
        ["competition_name", "season_name"],
        kind="stable",
    ).reset_index(drop=True)


def select_statsbomb_competitions(
    competitions: pd.DataFrame,
    *,
    competition_name: str | None = "FIFA World Cup",
    competition_names: Sequence[str] | None = None,
    competition_gender: str | None = "male",
    season_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    required_columns = {"competition_id", "season_id", "competition_name", "season_name"}
    missing_columns = required_columns.difference(competitions.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required competition columns: {missing}")

    filtered = competitions.copy()
    filtered["competition_name"] = filtered["competition_name"].map(canonicalize_tournament_name)

    normalized_names = (
        [canonicalize_tournament_name(name) for name in competition_names]
        if competition_names is not None
        else [canonicalize_tournament_name(competition_name or "FIFA World Cup")]
    )
    mask = filtered["competition_name"].isin(normalized_names)

    if competition_gender is not None and "competition_gender" in filtered.columns:
        mask &= filtered["competition_gender"].fillna("").astype(str).str.lower() == (
            competition_gender.lower()
        )

    if season_names:
        allowed = {str(season_name) for season_name in season_names}
        mask &= filtered["season_name"].astype(str).isin(allowed)

    selected = filtered.loc[mask].copy()
    return selected.sort_values(["competition_name", "season_name"], kind="stable").reset_index(
        drop=True
    )


def fetch_statsbomb_matches(
    competition_id: int,
    season_id: int,
    *,
    base_url: str = DEFAULT_STATSBOMB_OPEN_DATA_BASE_URL,
) -> pd.DataFrame:
    payload = _load_statsbomb_json(
        f"matches/{int(competition_id)}/{int(season_id)}.json",
        base_url=base_url,
    )
    return standardize_statsbomb_matches(payload)


def standardize_statsbomb_matches(payload: list[dict[str, object]]) -> pd.DataFrame:
    if not isinstance(payload, list):
        raise ValueError("StatsBomb matches payload must be a list.")

    rows: list[dict[str, object]] = []
    for match in payload:
        if not isinstance(match, dict):
            raise ValueError("StatsBomb matches must be JSON objects.")

        competition = _as_dict(match.get("competition"))
        season = _as_dict(match.get("season"))
        home_team = _as_dict(match.get("home_team"))
        away_team = _as_dict(match.get("away_team"))
        stadium = _as_dict(match.get("stadium"))
        competition_stage = _as_dict(match.get("competition_stage"))

        home_team_name = canonicalize_team_name(str(home_team.get("home_team_name") or ""))
        away_team_name = canonicalize_team_name(str(away_team.get("away_team_name") or ""))
        home_country_name = _nested_name(home_team.get("country"))
        away_country_name = _nested_name(away_team.get("country"))
        stadium_name = stadium.get("name")
        stadium_country = _nested_name(stadium.get("country"))
        competition_name = canonicalize_tournament_name(competition.get("competition_name"))

        rows.append(
            {
                "match_date": match.get("match_date"),
                "home_team": home_team_name,
                "away_team": away_team_name,
                "home_goals": match.get("home_score"),
                "away_goals": match.get("away_score"),
                "tournament": competition_name,
                "city": stadium_name,
                "country": stadium_country,
                "neutral": _infer_neutral_site(
                    stadium_country=stadium_country,
                    home_country=home_country_name,
                    away_country=away_country_name,
                ),
                "source_match_id": match.get("match_id"),
                "source_competition_id": competition.get("competition_id"),
                "source_season_id": season.get("season_id"),
                "source_competition_name": competition.get("competition_name"),
                "source_season_name": season.get("season_name"),
                "competition_stage": competition_stage.get("name"),
                "kick_off": match.get("kick_off"),
                "match_week": match.get("match_week"),
                "stadium": stadium_name,
                "home_group": home_team.get("home_team_group"),
                "away_group": away_team.get("away_team_group"),
                "home_manager": _first_manager_name(home_team.get("managers")),
                "away_manager": _first_manager_name(away_team.get("managers")),
                "home_country": home_country_name,
                "away_country": away_country_name,
                "source_updated_at_utc": match.get("last_updated"),
            }
        )

    standardized = normalize_standardized_results(pd.DataFrame.from_records(rows))
    standardized["source"] = "statsbomb_open_data"
    return standardized


def fetch_statsbomb_events(
    match_id: int,
    *,
    base_url: str = DEFAULT_STATSBOMB_OPEN_DATA_BASE_URL,
) -> list[dict[str, object]]:
    payload = _load_statsbomb_json(f"events/{int(match_id)}.json", base_url=base_url)
    if not isinstance(payload, list):
        raise ValueError("StatsBomb events payload must be a list.")
    return payload


def summarize_statsbomb_match_events(
    events: Sequence[dict[str, object]],
    *,
    home_team: str,
    away_team: str,
) -> dict[str, float]:
    team_side_lookup = {
        canonicalize_team_name(home_team): "home",
        canonicalize_team_name(away_team): "away",
    }
    summary: dict[str, float] = {
        "home_xg": 0.0,
        "away_xg": 0.0,
        "home_shots": 0.0,
        "away_shots": 0.0,
        "home_shots_on_target": 0.0,
        "away_shots_on_target": 0.0,
        "home_passes": 0.0,
        "away_passes": 0.0,
        "home_completed_passes": 0.0,
        "away_completed_passes": 0.0,
        "home_pressures": 0.0,
        "away_pressures": 0.0,
    }

    for event in events:
        if not isinstance(event, dict):
            continue

        team_name = canonicalize_team_name(_nested_name(event.get("team")))
        side = team_side_lookup.get(team_name)
        if side is None:
            continue

        event_type = _nested_name(event.get("type"))
        if event_type == "Shot":
            summary[f"{side}_shots"] += 1.0
            shot = _as_dict(event.get("shot"))
            summary[f"{side}_xg"] += _coerce_float(shot.get("statsbomb_xg"))
            if _nested_name(shot.get("outcome")) in _SHOTS_ON_TARGET_OUTCOMES:
                summary[f"{side}_shots_on_target"] += 1.0
            continue

        if event_type == "Pass":
            summary[f"{side}_passes"] += 1.0
            pass_payload = _as_dict(event.get("pass"))
            if not _nested_name(pass_payload.get("outcome")):
                summary[f"{side}_completed_passes"] += 1.0
            continue

        if event_type == "Pressure":
            summary[f"{side}_pressures"] += 1.0

    summary["home_pass_completion_rate"] = _safe_rate(
        numerator=summary["home_completed_passes"],
        denominator=summary["home_passes"],
    )
    summary["away_pass_completion_rate"] = _safe_rate(
        numerator=summary["away_completed_passes"],
        denominator=summary["away_passes"],
    )
    summary["xg_diff"] = summary["home_xg"] - summary["away_xg"]
    summary["shot_diff"] = summary["home_shots"] - summary["away_shots"]
    return summary


def build_statsbomb_world_cup_match_features(
    *,
    competition_name: str = "FIFA World Cup",
    competition_gender: str | None = "male",
    season_names: Sequence[str] | None = None,
    base_url: str = DEFAULT_STATSBOMB_OPEN_DATA_BASE_URL,
) -> pd.DataFrame:
    return build_statsbomb_competition_match_features(
        competition_names=[competition_name],
        competition_gender=competition_gender,
        season_names=season_names,
        base_url=base_url,
    )


def build_statsbomb_competition_match_features(
    *,
    competition_names: Sequence[str],
    competition_gender: str | None = "male",
    season_names: Sequence[str] | None = None,
    base_url: str = DEFAULT_STATSBOMB_OPEN_DATA_BASE_URL,
) -> pd.DataFrame:
    if not competition_names:
        raise ValueError("competition_names must contain at least one competition.")

    competitions = fetch_statsbomb_competitions(base_url=base_url)
    selected = select_statsbomb_competitions(
        competitions,
        competition_name=None,
        competition_names=competition_names,
        competition_gender=competition_gender,
        season_names=season_names,
    )
    if selected.empty:
        return pd.DataFrame()

    match_frames = [
        fetch_statsbomb_matches(
            int(row.competition_id),
            int(row.season_id),
            base_url=base_url,
        )
        for row in selected.itertuples(index=False)
    ]
    matches = pd.concat(match_frames, ignore_index=True)

    summaries: list[dict[str, float | int]] = []
    for row in matches.itertuples(index=False):
        summary = summarize_statsbomb_match_events(
            fetch_statsbomb_events(int(row.source_match_id), base_url=base_url),
            home_team=str(row.home_team),
            away_team=str(row.away_team),
        )
        summaries.append({"source_match_id": int(row.source_match_id)} | summary)

    event_summary_frame = pd.DataFrame.from_records(summaries)
    merged = matches.merge(
        event_summary_frame,
        on="source_match_id",
        how="left",
        validate="1:1",
    )
    return merged.sort_values(["match_date", "home_team", "away_team"], kind="stable").reset_index(
        drop=True
    )


def build_statsbomb_team_xg_summary(matches: pd.DataFrame) -> pd.DataFrame:
    required_columns = {
        "match_date",
        "home_team",
        "away_team",
        "home_xg",
        "away_xg",
        "home_shots",
        "away_shots",
        "home_pressures",
        "away_pressures",
        "home_passes",
        "away_passes",
        "home_completed_passes",
        "away_completed_passes",
    }
    missing_columns = required_columns.difference(matches.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required match feature columns: {missing}")

    team_rows: list[dict[str, object]] = []
    for row in matches.itertuples(index=False):
        team_rows.append(
            {
                "match_date": row.match_date,
                "team": row.home_team,
                "opponent": row.away_team,
                "xg_for": row.home_xg,
                "xg_against": row.away_xg,
                "shots_for": row.home_shots,
                "shots_against": row.away_shots,
                "pressures": row.home_pressures,
                "passes": row.home_passes,
                "completed_passes": row.home_completed_passes,
            }
        )
        team_rows.append(
            {
                "match_date": row.match_date,
                "team": row.away_team,
                "opponent": row.home_team,
                "xg_for": row.away_xg,
                "xg_against": row.home_xg,
                "shots_for": row.away_shots,
                "shots_against": row.home_shots,
                "pressures": row.away_pressures,
                "passes": row.away_passes,
                "completed_passes": row.away_completed_passes,
            }
        )

    team_match_frame = pd.DataFrame.from_records(team_rows)
    aggregate = (
        team_match_frame.groupby("team", sort=True)
        .agg(
            matches=("team", "size"),
            xg_for=("xg_for", "sum"),
            xg_against=("xg_against", "sum"),
            shots_for=("shots_for", "sum"),
            shots_against=("shots_against", "sum"),
            pressures=("pressures", "sum"),
            passes=("passes", "sum"),
            completed_passes=("completed_passes", "sum"),
        )
        .reset_index()
    )
    aggregate["xg_diff"] = aggregate["xg_for"] - aggregate["xg_against"]
    aggregate["xg_for_per_match"] = aggregate["xg_for"] / aggregate["matches"]
    aggregate["xg_against_per_match"] = aggregate["xg_against"] / aggregate["matches"]
    aggregate["xg_diff_per_match"] = aggregate["xg_diff"] / aggregate["matches"]
    aggregate["shots_for_per_match"] = aggregate["shots_for"] / aggregate["matches"]
    aggregate["shots_against_per_match"] = aggregate["shots_against"] / aggregate["matches"]
    aggregate["pressures_per_match"] = aggregate["pressures"] / aggregate["matches"]
    aggregate["pass_completion_rate"] = aggregate.apply(
        lambda row: _safe_rate(
            numerator=float(row["completed_passes"]),
            denominator=float(row["passes"]),
        ),
        axis=1,
    )
    return aggregate.sort_values(["xg_diff_per_match", "team"], ascending=[False, True]).reset_index(
        drop=True
    )


def _load_statsbomb_json(relative_path: str, *, base_url: str) -> object:
    url = f"{base_url.rstrip('/')}/{relative_path.lstrip('/')}"
    with urlopen(url) as response:
        return json.load(response)


def _as_dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _nested_name(value: object) -> str:
    if isinstance(value, dict):
        raw_name = value.get("name")
        return "" if raw_name is None else str(raw_name)
    return ""


def _first_manager_name(value: object) -> str | None:
    if not isinstance(value, list) or not value:
        return None
    first_manager = value[0]
    if not isinstance(first_manager, dict):
        return None
    raw_name = first_manager.get("name")
    return None if raw_name is None else str(raw_name)


def _infer_neutral_site(
    *,
    stadium_country: str,
    home_country: str,
    away_country: str,
) -> bool:
    country_set = {
        country.strip().lower()
        for country in (stadium_country, home_country, away_country)
        if isinstance(country, str) and country.strip()
    }
    if len(country_set) < 2:
        return True
    return stadium_country.strip().lower() not in {
        home_country.strip().lower(),
        away_country.strip().lower(),
    }


def _coerce_float(value: object) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_rate(*, numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return numerator / denominator
