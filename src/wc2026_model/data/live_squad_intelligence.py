from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

from .international_results import canonicalize_team_name

_BOOLEAN_TRUE_VALUES = {"1", "true", "yes", "y", "starter", "starting", "predicted"}
_UNAVAILABLE_STATUS_HINTS = {
    "injured",
    "out",
    "ruled out",
    "sidelined",
    "suspended",
    "ill",
    "absence",
    "unavailable",
}
_DOUBTFUL_STATUS_HINTS = {
    "doubtful",
    "questionable",
    "late fitness test",
    "fitness test",
    "minor doubt",
    "knock",
}


def standardize_official_squads(
    official_squads: pd.DataFrame,
    *,
    source_label: str = "official_squad",
) -> pd.DataFrame:
    _require_columns(official_squads, required={"team", "player"}, frame_name="official_squads")

    frame = official_squads.copy()
    frame["team"] = frame["team"].astype(str).map(canonicalize_team_name)
    frame["player"] = frame["player"].map(_clean_text)
    frame["player_key"] = frame["player"].map(normalize_person_name)
    frame["squad_position"] = _optional_text_column(frame, "position")
    frame["squad_number"] = _optional_numeric_column(frame, "squad_number")
    frame["club"] = _optional_text_column(frame, "club")
    frame["caps"] = _optional_numeric_column(frame, "caps")
    frame["goals"] = _optional_numeric_column(frame, "goals")
    frame["age"] = _optional_numeric_column(frame, "age")
    frame["birth_date"] = pd.to_datetime(frame.get("birth_date"), errors="coerce")
    frame["official_squad_source"] = str(source_label)
    frame["in_official_squad"] = True

    standardized = frame.loc[
        :,
        [
            "team",
            "player",
            "player_key",
            "squad_position",
            "squad_number",
            "club",
            "caps",
            "goals",
            "age",
            "birth_date",
            "official_squad_source",
            "in_official_squad",
        ],
    ].copy()
    standardized = standardized[
        standardized["team"].ne("")
        & standardized["player"].ne("")
        & standardized["player_key"].ne("")
    ].drop_duplicates(["team", "player_key"], keep="first")
    return standardized.reset_index(drop=True)


def standardize_expected_lineup_players(
    expected_lineups: pd.DataFrame,
    *,
    source_label: str = "expected_lineups",
) -> pd.DataFrame:
    _require_columns(expected_lineups, required={"team", "player"}, frame_name="expected_lineups")

    frame = expected_lineups.copy()
    frame["team"] = frame["team"].astype(str).map(canonicalize_team_name)
    frame["player"] = frame["player"].map(_clean_text)
    frame["player_key"] = frame["player"].map(normalize_person_name)
    frame["fixture_id"] = _optional_text_column(frame, "fixture_id")
    frame["match_date"] = pd.to_datetime(frame.get("match_date"), errors="coerce")
    frame["expected_lineup_position"] = _optional_text_column(frame, "position")
    frame["expected_lineup_formation"] = _optional_text_column(frame, "formation")
    frame["lineup_confidence"] = _optional_numeric_column(frame, "lineup_confidence").fillna(1.0)
    frame["is_expected_starter"] = _optional_bool_column(
        frame,
        "is_expected_starter",
        default=True,
    )
    frame["expected_lineup_source"] = str(source_label)

    standardized = frame.loc[
        :,
        [
            "fixture_id",
            "match_date",
            "team",
            "player",
            "player_key",
            "expected_lineup_position",
            "expected_lineup_formation",
            "lineup_confidence",
            "is_expected_starter",
            "expected_lineup_source",
        ],
    ].copy()
    standardized = standardized[
        standardized["team"].ne("")
        & standardized["player"].ne("")
        & standardized["player_key"].ne("")
    ].drop_duplicates(
        ["fixture_id", "match_date", "team", "player_key"],
        keep="last",
    )
    return standardized.reset_index(drop=True)


def standardize_injury_reports(
    injuries: pd.DataFrame,
    *,
    source_label: str = "injury_reports",
) -> pd.DataFrame:
    _require_columns(injuries, required={"team", "player"}, frame_name="injuries")

    frame = injuries.copy()
    frame["team"] = frame["team"].astype(str).map(canonicalize_team_name)
    frame["player"] = frame["player"].map(_clean_text)
    frame["player_key"] = frame["player"].map(normalize_person_name)
    frame["injury_status_raw"] = _optional_text_column(frame, "status")
    if "availability_status" in frame.columns:
        availability_status = _optional_text_column(frame, "availability_status").map(
            _normalize_availability_status_label
        )
    else:
        availability_status = frame["injury_status_raw"].map(_derive_availability_status)
    frame["availability_status"] = availability_status.fillna("available")
    frame["injury_reason"] = _optional_text_column(frame, "reason")
    frame["report_date"] = pd.to_datetime(frame.get("report_date"), errors="coerce")
    frame["expected_return_date"] = pd.to_datetime(
        frame.get("expected_return_date"),
        errors="coerce",
    )
    frame["injury_source"] = str(source_label)
    frame["availability_weight"] = frame["availability_status"].map(
        {
            "available": 1.0,
            "doubtful": 0.5,
            "unavailable": 0.0,
        }
    ).fillna(1.0)
    frame["has_injury_report"] = (
        frame["injury_status_raw"].ne("") | frame["injury_reason"].ne("")
    )

    standardized = frame.loc[
        :,
        [
            "team",
            "player",
            "player_key",
            "injury_status_raw",
            "injury_reason",
            "availability_status",
            "availability_weight",
            "report_date",
            "expected_return_date",
            "injury_source",
            "has_injury_report",
        ],
    ].copy()
    standardized = standardized[
        standardized["team"].ne("")
        & standardized["player"].ne("")
        & standardized["player_key"].ne("")
    ].drop_duplicates(["team", "player_key"], keep="last")
    return standardized.reset_index(drop=True)


def build_live_squad_intelligence(
    official_squads: pd.DataFrame,
    *,
    expected_lineups: pd.DataFrame | None = None,
    injuries: pd.DataFrame | None = None,
    official_source_label: str = "official_squad",
    expected_lineup_source_label: str = "expected_lineups",
    injury_source_label: str = "injury_reports",
) -> pd.DataFrame:
    official = standardize_official_squads(
        official_squads,
        source_label=official_source_label,
    )
    lineups = (
        standardize_expected_lineup_players(
            expected_lineups,
            source_label=expected_lineup_source_label,
        )
        if expected_lineups is not None and not expected_lineups.empty
        else pd.DataFrame()
    )
    injury_reports = (
        standardize_injury_reports(
            injuries,
            source_label=injury_source_label,
        )
        if injuries is not None and not injuries.empty
        else pd.DataFrame()
    )

    base = _expand_official_squads_across_fixtures(official, lineups=lineups)
    if not lineups.empty:
        lineup_merge_keys = _lineup_merge_keys(lineups)
        base = base.merge(
            lineups.loc[
                :,
                lineup_merge_keys
                + [
                    "expected_lineup_position",
                    "expected_lineup_formation",
                    "lineup_confidence",
                    "is_expected_starter",
                    "expected_lineup_source",
                ],
            ],
            on=lineup_merge_keys,
            how="left",
        )
    else:
        base["expected_lineup_position"] = pd.NA
        base["expected_lineup_formation"] = pd.NA
        base["lineup_confidence"] = pd.NA
        base["is_expected_starter"] = False
        base["expected_lineup_source"] = pd.NA

    if not injury_reports.empty:
        base = base.merge(
            injury_reports.loc[
                :,
                [
                    "team",
                    "player_key",
                    "injury_status_raw",
                    "injury_reason",
                    "availability_status",
                    "availability_weight",
                    "report_date",
                    "expected_return_date",
                    "injury_source",
                    "has_injury_report",
                ],
            ],
            on=["team", "player_key"],
            how="left",
        )
    else:
        base["injury_status_raw"] = pd.NA
        base["injury_reason"] = pd.NA
        base["availability_status"] = "available"
        base["availability_weight"] = 1.0
        base["report_date"] = pd.NaT
        base["expected_return_date"] = pd.NaT
        base["injury_source"] = pd.NA
        base["has_injury_report"] = False

    base["is_expected_starter"] = base["is_expected_starter"].fillna(False).astype(bool)
    base["availability_status"] = base["availability_status"].fillna("available")
    base["availability_weight"] = pd.to_numeric(
        base["availability_weight"],
        errors="coerce",
    ).fillna(1.0)
    base["has_injury_report"] = base["has_injury_report"].fillna(False).astype(bool)
    base["is_unavailable"] = base["availability_status"].eq("unavailable")
    base["is_doubtful"] = base["availability_status"].eq("doubtful")
    base["expected_starter_availability_weight"] = (
        base["availability_weight"] * base["is_expected_starter"].astype(float)
    )
    base["player_match_key"] = base.apply(
        lambda row: _build_player_match_key(
            fixture_id=row.get("fixture_id"),
            match_date=row.get("match_date"),
            team=row.get("team"),
            player_key=row.get("player_key"),
        ),
        axis=1,
    )

    ordered_columns = [
        "fixture_id",
        "match_date",
        "team",
        "player",
        "player_key",
        "squad_position",
        "expected_lineup_position",
        "expected_lineup_formation",
        "lineup_confidence",
        "is_expected_starter",
        "availability_status",
        "availability_weight",
        "expected_starter_availability_weight",
        "is_unavailable",
        "is_doubtful",
        "has_injury_report",
        "injury_status_raw",
        "injury_reason",
        "report_date",
        "expected_return_date",
        "club",
        "caps",
        "goals",
        "age",
        "birth_date",
        "official_squad_source",
        "expected_lineup_source",
        "injury_source",
        "in_official_squad",
        "player_match_key",
    ]
    return base.loc[:, ordered_columns].sort_values(
        ["match_date", "fixture_id", "team", "is_expected_starter", "player"],
        ascending=[True, True, True, False, True],
        kind="stable",
    ).reset_index(drop=True)


def aggregate_team_availability_features(player_intelligence: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        player_intelligence,
        required={"team", "player", "availability_weight", "is_expected_starter"},
        frame_name="player_intelligence",
    )

    group_keys = ["team"]
    if "fixture_id" in player_intelligence.columns and player_intelligence["fixture_id"].notna().any():
        group_keys.insert(0, "fixture_id")
    if "match_date" in player_intelligence.columns and player_intelligence["match_date"].notna().any():
        if "match_date" not in group_keys:
            group_keys.append("match_date")

    rows: list[dict[str, object]] = []
    for keys, group in player_intelligence.groupby(group_keys, dropna=False, sort=True):
        key_map = _group_key_map(group_keys, keys)
        starter_mask = group["is_expected_starter"].astype(bool)
        unavailable_mask = group["is_unavailable"].astype(bool)
        doubtful_mask = group["is_doubtful"].astype(bool)
        goalkeeper_mask = (
            group["expected_lineup_position"].fillna(group["squad_position"]).astype(str).eq("GK")
        )
        lineup_confidence_values = pd.to_numeric(group["lineup_confidence"], errors="coerce")
        row: dict[str, object] = {
            **key_map,
            "official_squad_player_count": int(len(group)),
            "expected_starter_count": int(starter_mask.sum()),
            "expected_goalkeeper_count": int((starter_mask & goalkeeper_mask).sum()),
            "injury_reported_player_count": int(group["has_injury_report"].astype(bool).sum()),
            "unavailable_player_count": int(unavailable_mask.sum()),
            "doubtful_player_count": int(doubtful_mask.sum()),
            "unavailable_expected_starter_count": int((starter_mask & unavailable_mask).sum()),
            "doubtful_expected_starter_count": int((starter_mask & doubtful_mask).sum()),
            "expected_lineup_completeness": float(starter_mask.mean() * len(group) / 11.0),
            "expected_starter_availability_weight_sum": float(
                group["expected_starter_availability_weight"].sum()
            ),
            "expected_starter_availability_weight_mean": float(
                group.loc[starter_mask, "availability_weight"].mean()
                if starter_mask.any()
                else 0.0
            ),
            "lineup_confidence": float(lineup_confidence_values.max())
            if lineup_confidence_values.notna().any()
            else 0.0,
            "goalkeeper_starter_available": bool(
                ((starter_mask & goalkeeper_mask & ~unavailable_mask).sum()) >= 1
            ),
        }
        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        group_keys,
        kind="stable",
    ).reset_index(drop=True)


def build_live_squad_intelligence_summary(
    player_intelligence: pd.DataFrame,
    *,
    expected_lineups: pd.DataFrame | None = None,
    injuries: pd.DataFrame | None = None,
) -> dict[str, object]:
    summary: dict[str, object] = {
        "player_rows": int(len(player_intelligence)),
        "team_count": int(player_intelligence["team"].nunique()),
        "fixture_team_rows": int(
            player_intelligence.loc[:, ["fixture_id", "team"]]
            .drop_duplicates()
            .shape[0]
            if "fixture_id" in player_intelligence.columns
            else player_intelligence["team"].nunique()
        ),
        "expected_starter_rows": int(player_intelligence["is_expected_starter"].sum()),
        "unavailable_rows": int(player_intelligence["is_unavailable"].sum()),
        "doubtful_rows": int(player_intelligence["is_doubtful"].sum()),
        "injury_reported_rows": int(player_intelligence["has_injury_report"].sum()),
    }

    if expected_lineups is not None and not expected_lineups.empty:
        lineups = standardize_expected_lineup_players(expected_lineups)
        lineup_merge_keys = _lineup_merge_keys(lineups)
        unmatched_lineups = lineups.merge(
            player_intelligence.loc[:, lineup_merge_keys].drop_duplicates(),
            on=lineup_merge_keys,
            how="left",
            indicator=True,
        )
        summary["expected_lineup_input_rows"] = int(len(lineups))
        summary["expected_lineup_unmatched_rows"] = int(
            unmatched_lineups["_merge"].eq("left_only").sum()
        )
    else:
        summary["expected_lineup_input_rows"] = 0
        summary["expected_lineup_unmatched_rows"] = 0

    if injuries is not None and not injuries.empty:
        injury_reports = standardize_injury_reports(injuries)
        unmatched_injuries = injury_reports.merge(
            player_intelligence.loc[:, ["team", "player_key"]].drop_duplicates(),
            on=["team", "player_key"],
            how="left",
            indicator=True,
        )
        summary["injury_input_rows"] = int(len(injury_reports))
        summary["injury_unmatched_rows"] = int(unmatched_injuries["_merge"].eq("left_only").sum())
    else:
        summary["injury_input_rows"] = 0
        summary["injury_unmatched_rows"] = 0

    return summary


def load_flat_file_table(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file_path)
    if suffix == ".json":
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return pd.DataFrame.from_records(payload)
        if isinstance(payload, dict):
            for key in ("data", "response", "rows", "items"):
                if isinstance(payload.get(key), list):
                    return pd.DataFrame.from_records(payload[key])
            return pd.DataFrame.from_records([payload])
    raise ValueError(f"Unsupported file type for {file_path}. Use CSV or flat JSON records.")


def normalize_person_name(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _expand_official_squads_across_fixtures(
    official_squads: pd.DataFrame,
    *,
    lineups: pd.DataFrame,
) -> pd.DataFrame:
    if lineups.empty:
        base = official_squads.copy()
        base["fixture_id"] = pd.NA
        base["match_date"] = pd.NaT
        return base

    fixture_teams = lineups.loc[
        :,
        [
            "fixture_id",
            "match_date",
            "team",
        ],
    ].drop_duplicates()
    return fixture_teams.merge(official_squads, on="team", how="left")


def _lineup_merge_keys(lineups: pd.DataFrame) -> list[str]:
    keys = ["team", "player_key"]
    if "fixture_id" in lineups.columns and lineups["fixture_id"].notna().any():
        keys.insert(0, "fixture_id")
    elif "match_date" in lineups.columns and lineups["match_date"].notna().any():
        keys.insert(0, "match_date")
    return keys


def _build_player_match_key(
    *,
    fixture_id: object,
    match_date: object,
    team: object,
    player_key: object,
) -> str:
    fixture_key = ""
    if pd.notna(fixture_id):
        fixture_key = str(fixture_id)
    elif pd.notna(match_date):
        fixture_key = pd.Timestamp(match_date).strftime("%Y-%m-%d")
    return "|".join([fixture_key, str(team), str(player_key)])


def _optional_text_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(pd.NA, index=frame.index, dtype="object")
    return frame[column].map(_clean_text)


def _optional_numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(float("nan"), index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _optional_bool_column(frame: pd.DataFrame, column: str, *, default: bool) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=bool)

    values = frame[column]
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(default)

    normalized = values.astype(str).str.strip().str.lower()
    return normalized.map(lambda value: value in _BOOLEAN_TRUE_VALUES).fillna(default)


def _clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _normalize_availability_status_label(value: object) -> str:
    lowered = _clean_text(value).lower()
    if lowered in {"available", "fit", "healthy"}:
        return "available"
    if lowered in {"doubtful", "questionable"}:
        return "doubtful"
    if lowered in {"unavailable", "out", "injured", "suspended"}:
        return "unavailable"
    return _derive_availability_status(lowered)


def _derive_availability_status(value: object) -> str:
    lowered = _clean_text(value).lower()
    if lowered == "":
        return "available"
    if any(hint in lowered for hint in _UNAVAILABLE_STATUS_HINTS):
        return "unavailable"
    if any(hint in lowered for hint in _DOUBTFUL_STATUS_HINTS):
        return "doubtful"
    return "available"


def _group_key_map(group_keys: list[str], keys: object) -> dict[str, object]:
    if not isinstance(keys, tuple):
        keys = (keys,)
    return dict(zip(group_keys, keys, strict=True))


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
