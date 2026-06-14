from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from .live_squad_intelligence import load_flat_file_table

SPORTMONKS_FOOTBALL_API_BASE_URL = "https://api.sportmonks.com/v3/football"
API_FOOTBALL_BASE_URL = "https://v3.football.api-sports.io"

_TRUE_VALUES = {"1", "true", "yes", "y", "starter", "starting", "predicted"}
_FALSE_VALUES = {"0", "false", "no", "n", "bench", "substitute", "sub", "reserve"}
_SPORTMONKS_EXPECTED_LINEUPS_COLUMNS = [
    "fixture_id",
    "match_date",
    "team",
    "player",
    "position",
    "formation",
    "is_expected_starter",
    "lineup_confidence",
    "expected_lineup_id",
    "team_id",
    "player_id",
    "jersey_number",
    "formation_field",
    "formation_position",
    "type_id",
    "type_name",
    "source",
]
_API_FOOTBALL_INJURY_COLUMNS = [
    "fixture_id",
    "match_date",
    "team",
    "player",
    "status",
    "reason",
    "availability_status",
    "report_date",
    "expected_return_date",
    "team_id",
    "player_id",
    "league_id",
    "league_name",
    "source",
]


def load_expected_lineups_feed(
    path: str | Path,
    *,
    provider: str = "auto",
) -> pd.DataFrame:
    file_path = Path(path)
    provider_key = provider.strip().lower()
    if provider_key == "flat" or file_path.suffix.lower() != ".json":
        return load_flat_file_table(file_path)

    payload = _load_json_payload(file_path)
    inferred_provider = (
        infer_expected_lineups_provider(payload) if provider_key == "auto" else provider_key
    )
    if inferred_provider == "sportmonks":
        return standardize_sportmonks_expected_lineups_payload(payload)
    return load_flat_file_table(file_path)


def load_injuries_feed(
    path: str | Path,
    *,
    provider: str = "auto",
) -> pd.DataFrame:
    file_path = Path(path)
    provider_key = provider.strip().lower()
    if provider_key == "flat" or file_path.suffix.lower() != ".json":
        return load_flat_file_table(file_path)

    payload = _load_json_payload(file_path)
    inferred_provider = infer_injuries_provider(payload) if provider_key == "auto" else provider_key
    if inferred_provider == "api_football":
        return standardize_api_football_injuries_payload(payload)
    return load_flat_file_table(file_path)


def standardize_sportmonks_expected_lineups_payload(
    payload: dict[str, Any] | list[dict[str, Any]],
) -> pd.DataFrame:
    rows = _records_from_payload(payload)
    records: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue

        team_name = _first_non_empty(
            row.get("team_name"),
            _relation_value(row.get("participant"), "name"),
            _relation_value(row.get("team"), "name"),
            _relation_value(row.get("participant"), "short_name"),
            row.get("team"),
        )
        player_name = _first_non_empty(
            row.get("player_name"),
            _relation_value(row.get("player"), "display_name"),
            _relation_value(row.get("player"), "name"),
            row.get("player"),
        )
        fixture_id = _first_non_empty(
            row.get("fixture_id"),
            _relation_value(row.get("fixture"), "id"),
        )
        match_date = _first_non_empty(
            row.get("match_date"),
            _relation_value(row.get("fixture"), "starting_at"),
            _relation_value(row.get("fixture"), "date"),
            _relation_value(row.get("fixture"), "starting_at_date"),
        )
        position_label = _normalize_position_label(
            _first_non_empty(
                row.get("position"),
                row.get("position_name"),
                row.get("detailed_position_name"),
                _relation_value(row.get("position"), "name"),
                _relation_value(row.get("detailed_position"), "name"),
                _relation_value(row.get("player"), "position"),
            )
        )
        formation = _normalize_formation(
            _first_non_empty(
                row.get("formation"),
                row.get("formation_name"),
                _relation_value(row.get("fixture"), "formation"),
                _relation_value(row.get("participant"), "formation"),
                _relation_value(row.get("team"), "formation"),
            )
        )
        type_name = _first_non_empty(
            row.get("type_name"),
            _relation_value(row.get("type"), "name"),
            _relation_value(row.get("lineup_type"), "name"),
        )
        records.append(
            {
                "fixture_id": fixture_id,
                "match_date": match_date,
                "team": team_name,
                "player": player_name,
                "position": position_label,
                "formation": formation,
                "is_expected_starter": _infer_sportmonks_starter_flag(
                    row,
                    type_name=type_name,
                ),
                "lineup_confidence": _coerce_float(
                    _first_non_empty(
                        row.get("lineup_confidence"),
                        row.get("confidence"),
                        row.get("probability"),
                        1.0,
                    )
                ),
                "expected_lineup_id": _coerce_int(row.get("id")),
                "team_id": _coerce_int(
                    _first_non_empty(
                        row.get("team_id"),
                        _relation_value(row.get("participant"), "id"),
                        _relation_value(row.get("team"), "id"),
                    )
                ),
                "player_id": _coerce_int(
                    _first_non_empty(
                        row.get("player_id"),
                        _relation_value(row.get("player"), "id"),
                    )
                ),
                "jersey_number": _coerce_int(row.get("jersey_number")),
                "formation_field": _clean_text(row.get("formation_field")),
                "formation_position": _coerce_int(row.get("formation_position")),
                "type_id": _coerce_int(
                    _first_non_empty(
                        row.get("type_id"),
                        _relation_value(row.get("type"), "id"),
                    )
                ),
                "type_name": _clean_text(type_name),
                "source": "sportmonks_expected_lineups",
            }
        )

    return pd.DataFrame.from_records(records, columns=_SPORTMONKS_EXPECTED_LINEUPS_COLUMNS)


def standardize_api_football_injuries_payload(
    payload: dict[str, Any] | list[dict[str, Any]],
) -> pd.DataFrame:
    rows = _records_from_payload(payload)
    records: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue

        team_block = row.get("team")
        player_block = row.get("player")
        fixture_block = row.get("fixture")
        league_block = row.get("league")

        status = _first_non_empty(
            row.get("status"),
            row.get("availability_status"),
            _relation_value(player_block, "status"),
            _relation_value(player_block, "type"),
        )
        reason = _first_non_empty(
            row.get("reason"),
            _relation_value(player_block, "reason"),
            _relation_value(player_block, "injury"),
            _relation_value(player_block, "comment"),
        )
        records.append(
            {
                "fixture_id": _coerce_int(
                    _first_non_empty(
                        row.get("fixture_id"),
                        _relation_value(fixture_block, "id"),
                    )
                ),
                "match_date": _first_non_empty(
                    row.get("match_date"),
                    _relation_value(fixture_block, "date"),
                    row.get("date"),
                ),
                "team": _first_non_empty(
                    row.get("team_name"),
                    _relation_value(team_block, "name"),
                    row.get("team"),
                ),
                "player": _first_non_empty(
                    row.get("player_name"),
                    _relation_value(player_block, "name"),
                    row.get("player"),
                ),
                "status": status,
                "reason": reason,
                "availability_status": _infer_api_football_availability_status(
                    status=status,
                    reason=reason,
                ),
                "report_date": _first_non_empty(
                    row.get("report_date"),
                    _relation_value(fixture_block, "date"),
                    row.get("updated_at"),
                    row.get("date"),
                ),
                "expected_return_date": _first_non_empty(
                    row.get("expected_return_date"),
                    row.get("return_date"),
                    _relation_value(player_block, "expected_return"),
                    _relation_value(player_block, "return_date"),
                ),
                "team_id": _coerce_int(
                    _first_non_empty(
                        row.get("team_id"),
                        _relation_value(team_block, "id"),
                    )
                ),
                "player_id": _coerce_int(
                    _first_non_empty(
                        row.get("player_id"),
                        _relation_value(player_block, "id"),
                    )
                ),
                "league_id": _coerce_int(
                    _first_non_empty(
                        row.get("league_id"),
                        _relation_value(league_block, "id"),
                    )
                ),
                "league_name": _first_non_empty(
                    row.get("league_name"),
                    _relation_value(league_block, "name"),
                ),
                "source": "api_football_injuries",
            }
        )

    return pd.DataFrame.from_records(records, columns=_API_FOOTBALL_INJURY_COLUMNS)


def infer_expected_lineups_provider(
    payload: dict[str, Any] | list[dict[str, Any]],
) -> str | None:
    first_row = _first_record(payload)
    if not isinstance(first_row, dict):
        return None
    if "player_name" in first_row and "fixture_id" in first_row:
        if any(key in first_row for key in ("team_id", "participant", "team")):
            return "sportmonks"
    return None


def infer_injuries_provider(
    payload: dict[str, Any] | list[dict[str, Any]],
) -> str | None:
    first_row = _first_record(payload)
    if not isinstance(first_row, dict):
        return None
    if isinstance(first_row.get("team"), dict) and isinstance(first_row.get("player"), dict):
        return "api_football"
    return None


def _load_json_payload(path: Path) -> dict[str, Any] | list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _records_from_payload(payload: dict[str, Any] | list[dict[str, Any]]) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "response", "rows", "items"):
            if isinstance(payload.get(key), list):
                return payload[key]
        return [payload]
    return []


def _first_record(payload: dict[str, Any] | list[dict[str, Any]]) -> Any:
    records = _records_from_payload(payload)
    return records[0] if records else None


def _relation_value(value: Any, field: str) -> Any:
    if isinstance(value, dict):
        if field in value:
            return value.get(field)
        data = value.get("data")
        if isinstance(data, dict):
            return data.get(field)
    if isinstance(value, list):
        for item in value:
            relation_value = _relation_value(item, field)
            if relation_value not in (None, ""):
                return relation_value
    return None


def _infer_sportmonks_starter_flag(
    row: dict[str, Any],
    *,
    type_name: str,
) -> bool:
    explicit_flag = _first_non_empty(
        row.get("is_expected_starter"),
        row.get("is_starter"),
        row.get("starter"),
        row.get("predicted"),
    )
    if explicit_flag != "":
        return _coerce_bool(explicit_flag, default=True)

    lowered_type_name = _clean_text(type_name).lower()
    if any(token in lowered_type_name for token in ("bench", "substitute", "reserve", "backup")):
        return False
    if any(
        token in lowered_type_name
        for token in ("starting", "starter", "first xi", "starting xi", "lineup")
    ):
        return True

    return _first_non_empty(row.get("formation_position"), row.get("formation_field")) != ""


def _infer_api_football_availability_status(
    *,
    status: Any,
    reason: Any,
) -> str:
    lowered_status = _clean_text(status).lower()
    lowered_reason = _clean_text(reason).lower()
    if any(token in lowered_status for token in ("doubt", "question", "late fitness", "knock")):
        return "doubtful"
    if any(token in lowered_reason for token in ("doubt", "question", "late fitness", "knock")):
        return "doubtful"
    if lowered_status in {"available", "fit", "healthy"}:
        return "available"
    return "unavailable"


def _normalize_position_label(value: Any) -> str:
    text = _clean_text(value)
    if text == "":
        return ""
    lowered = text.lower()
    if lowered in {"gk", "goalkeeper", "keeper"} or "goalkeeper" in lowered:
        return "GK"
    if lowered in {"df", "defender"} or any(
        token in lowered for token in ("defender", "centre-back", "center-back", "full-back", "back")
    ):
        return "DF"
    if lowered in {"mf", "midfielder"} or "midfield" in lowered:
        return "MF"
    if lowered in {"fw", "forward", "attacker", "striker"} or any(
        token in lowered for token in ("forward", "attacker", "striker", "winger")
    ):
        return "FW"
    return text.upper()


def _normalize_formation(value: Any) -> str:
    text = _clean_text(value)
    if re.fullmatch(r"\d(?:-\d){2,4}", text):
        return text
    return ""


def _coerce_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    text = _clean_text(value).lower()
    if text in _TRUE_VALUES:
        return True
    if text in _FALSE_VALUES:
        return False
    return default


def _coerce_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_non_empty(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and value.strip() == "":
            continue
        return value
    return ""


def _clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()
