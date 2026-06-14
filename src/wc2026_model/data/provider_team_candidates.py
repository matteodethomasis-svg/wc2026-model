from __future__ import annotations

import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pandas as pd

from .international_results import canonicalize_team_name
from .provider_team_registry import normalize_provider_team_registry

_PROVIDER_COLUMNS = {
    "sportmonks": ("sportmonks_team_id", "sportmonks_team_name"),
    "api_football": ("api_football_team_id", "api_football_team_name"),
}

_TEAM_SEARCH_QUERY_ALIASES = {
    "Bosnia and Herzegovina": [
        "Bosnia & Herzegovina",
        "Bosnia-Herzegovina",
        "Bosnia Herzegovina",
    ],
    "Cape Verde": [
        "Cape Verde Islands",
    ],
    "Curaçao": [
        "Curacao",
    ],
    "Czech Republic": [
        "Czechia",
    ],
    "DR Congo": [
        "Democratic Republic of the Congo",
        "Congo DR",
        "Congo, DR",
    ],
    "Iran": [
        "IR Iran",
    ],
    "Ivory Coast": [
        "Cote d'Ivoire",
        "Côte d'Ivoire",
    ],
    "South Korea": [
        "Korea Republic",
        "Republic of Korea",
        "Korea, Republic of",
    ],
    "Turkey": [
        "Türkiye",
        "Turkiye",
    ],
    "United States": [
        "USA",
        "United States of America",
    ],
}
_API_FOOTBALL_TEAM_CODES = {
    "Algeria": "ALG",
    "Argentina": "ARG",
    "Australia": "AUS",
    "Austria": "AUT",
    "Belgium": "BEL",
    "Bosnia and Herzegovina": "BIH",
    "Brazil": "BRA",
    "Canada": "CAN",
    "Cape Verde": "CPV",
    "Colombia": "COL",
    "Croatia": "CRO",
    "CuraÃ§ao": "CUW",
    "Czech Republic": "CZE",
    "DR Congo": "COD",
    "Ecuador": "ECU",
    "Egypt": "EGY",
    "England": "ENG",
    "France": "FRA",
    "Germany": "GER",
    "Ghana": "GHA",
    "Haiti": "HAI",
    "Iran": "IRN",
    "Iraq": "IRQ",
    "Ivory Coast": "CIV",
    "Japan": "JPN",
    "Jordan": "JOR",
    "Mexico": "MEX",
    "Morocco": "MAR",
    "Netherlands": "NED",
    "New Zealand": "NZL",
    "Norway": "NOR",
    "Panama": "PAN",
    "Paraguay": "PAR",
    "Portugal": "POR",
    "Qatar": "QAT",
    "Saudi Arabia": "KSA",
    "Scotland": "SCO",
    "Senegal": "SEN",
    "South Africa": "RSA",
    "South Korea": "KOR",
    "Spain": "ESP",
    "Sweden": "SWE",
    "Switzerland": "SUI",
    "Tunisia": "TUN",
    "Turkey": "TUR",
    "United States": "USA",
    "Uruguay": "URU",
    "Uzbekistan": "UZB",
}

_NON_SENIOR_MENS_PATTERN = re.compile(
    r"\b("
    r"u\d{2}|under\s*\d{2}|women|wnt|feminina|femenina|ladies|olympic|olympics|"
    r"reserve|reserves|b team|ii|iii|amateurs|beach|futsal|esports"
    r")\b",
    flags=re.IGNORECASE,
)
_NATIONAL_TEAM_PATTERN = re.compile(r"\bnational team\b", flags=re.IGNORECASE)
_PROVIDER_TEAM_SEARCH_COLUMNS = [
    "provider",
    "target_team",
    "search_query",
    "team_id",
    "candidate_name",
    "candidate_short_code",
    "candidate_type",
    "candidate_gender",
    "candidate_country_id",
    "candidate_country_name",
    "candidate_last_played_at",
    "candidate_placeholder",
    "source",
]


def build_team_search_queries(team: str) -> list[str]:
    canonical_team = canonicalize_team_name(team)
    queries = [canonical_team]
    queries.extend(_TEAM_SEARCH_QUERY_ALIASES.get(canonical_team, []))
    return list(dict.fromkeys([query.strip() for query in queries if str(query).strip() != ""]))


def build_api_football_team_search_queries(team: str) -> list[str]:
    canonical_team = canonicalize_team_name(team)
    queries = build_team_search_queries(canonical_team)
    code = _API_FOOTBALL_TEAM_CODES.get(canonical_team)
    if code:
        queries.append(f"code:{code}")
    return list(dict.fromkeys([query.strip() for query in queries if str(query).strip() != ""]))


def load_provider_team_search_candidates(
    path: str | Path,
    *,
    provider: str,
) -> pd.DataFrame:
    file_path = Path(path)
    provider_key = provider.strip().lower()
    if file_path.suffix.lower() != ".json":
        return pd.read_csv(file_path)

    payload = json.loads(file_path.read_text(encoding="utf-8"))
    if provider_key == "sportmonks":
        return standardize_sportmonks_team_search_payload(payload)
    if provider_key == "api_football":
        return standardize_api_football_team_search_payload(payload)
    raise ValueError(f"Unsupported provider '{provider}'.")


def standardize_sportmonks_team_search_payload(
    payload: dict[str, Any] | list[dict[str, Any]],
    *,
    target_team: str | None = None,
    search_query: str | None = None,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    if isinstance(payload, dict) and isinstance(payload.get("queries"), list):
        for query_block in payload["queries"]:
            if not isinstance(query_block, dict):
                continue
            nested = standardize_sportmonks_team_search_payload(
                query_block.get("response") or {},
                target_team=query_block.get("target_team"),
                search_query=query_block.get("search_query"),
            )
            if not nested.empty:
                records.extend(nested.to_dict(orient="records"))
        return pd.DataFrame.from_records(records, columns=_PROVIDER_TEAM_SEARCH_COLUMNS)

    rows = _records_from_payload(payload)
    canonical_target_team = canonicalize_team_name(target_team) if target_team else None
    cleaned_query = _clean_text(search_query)
    for row in rows:
        if not isinstance(row, dict):
            continue
        candidate_name = _first_non_empty(row.get("name"), row.get("short_name"))
        records.append(
            {
                "provider": "sportmonks",
                "target_team": canonical_target_team,
                "search_query": cleaned_query,
                "team_id": _coerce_int(row.get("id")),
                "candidate_name": candidate_name,
                "candidate_short_code": _clean_text(row.get("short_code")),
                "candidate_type": _clean_text(row.get("type")),
                "candidate_gender": _clean_text(row.get("gender")),
                "candidate_country_id": _coerce_int(_relation_value(row.get("country"), "id"))
                or _coerce_int(row.get("country_id")),
                "candidate_country_name": _first_non_empty(
                    _relation_value(row.get("country"), "name"),
                    row.get("country_name"),
                ),
                "candidate_last_played_at": _clean_text(row.get("last_played_at")),
                "candidate_placeholder": bool(row.get("placeholder", False)),
                "source": "sportmonks_team_search",
            }
        )
    return pd.DataFrame.from_records(records, columns=_PROVIDER_TEAM_SEARCH_COLUMNS)


def standardize_api_football_team_search_payload(
    payload: dict[str, Any] | list[dict[str, Any]],
    *,
    target_team: str | None = None,
    search_query: str | None = None,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    if isinstance(payload, dict) and isinstance(payload.get("queries"), list):
        for query_block in payload["queries"]:
            if not isinstance(query_block, dict):
                continue
            nested = standardize_api_football_team_search_payload(
                query_block.get("response") or {},
                target_team=query_block.get("target_team"),
                search_query=query_block.get("search_query"),
            )
            if not nested.empty:
                records.extend(nested.to_dict(orient="records"))
        return pd.DataFrame.from_records(records, columns=_PROVIDER_TEAM_SEARCH_COLUMNS)

    rows = _records_from_payload(payload)
    canonical_target_team = canonicalize_team_name(target_team) if target_team else None
    cleaned_query = _clean_text(search_query)
    for row in rows:
        if not isinstance(row, dict):
            continue
        team = row.get("team")
        if not isinstance(team, dict):
            team = row
        candidate_name = _first_non_empty(team.get("name"), team.get("code"))
        is_national = bool(team.get("national"))
        records.append(
            {
                "provider": "api_football",
                "target_team": canonical_target_team,
                "search_query": cleaned_query,
                "team_id": _coerce_int(team.get("id")),
                "candidate_name": candidate_name,
                "candidate_short_code": _clean_text(team.get("code")),
                "candidate_type": "national" if is_national else "club",
                "candidate_gender": "",
                "candidate_country_id": None,
                "candidate_country_name": _clean_text(team.get("country")),
                "candidate_last_played_at": "",
                "candidate_placeholder": False,
                "source": "api_football_team_search",
            }
        )
    return pd.DataFrame.from_records(records, columns=_PROVIDER_TEAM_SEARCH_COLUMNS)


def score_provider_team_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    frame = candidates.copy()
    for column in ("target_team", "candidate_name", "team_id"):
        if column not in frame.columns:
            frame[column] = pd.Series(dtype=object)
    required_columns = {"target_team", "candidate_name", "team_id"}
    if not frame.empty:
        missing = required_columns.difference(candidates.columns)
        if missing:
            missing_columns = ", ".join(sorted(missing))
            raise ValueError(f"Missing required candidate columns: {missing_columns}")

    if frame.empty:
        empty = frame.copy()
        empty["match_score"] = pd.Series(dtype=float)
        empty["score_margin"] = pd.Series(dtype=float)
        empty["rank"] = pd.Series(dtype=int)
        return empty

    frame["target_team"] = frame["target_team"].astype(str).map(canonicalize_team_name)
    frame["candidate_name"] = frame["candidate_name"].fillna("").astype(str)
    for column, default_value in (
        ("search_query", ""),
        ("candidate_type", ""),
        ("candidate_gender", ""),
        ("candidate_short_code", ""),
        ("candidate_placeholder", False),
    ):
        if column not in frame.columns:
            frame[column] = default_value
    frame["search_query"] = frame["search_query"].fillna("").astype(str)
    frame["candidate_type"] = frame["candidate_type"].fillna("").astype(str)
    frame["candidate_gender"] = frame["candidate_gender"].fillna("").astype(str)
    frame["candidate_short_code"] = frame["candidate_short_code"].fillna("").astype(str)
    frame["candidate_placeholder"] = frame["candidate_placeholder"].fillna(False).astype(bool)

    frame["target_name_key"] = frame["target_team"].map(_normalize_match_text)
    frame["candidate_name_key"] = frame["candidate_name"].map(_normalize_match_text)
    frame["search_query_key"] = frame["search_query"].map(_normalize_match_text)
    frame["candidate_code_key"] = frame["candidate_short_code"].map(_normalize_match_text)
    frame["code_query_key"] = frame["search_query"].map(_extract_code_query_key)
    frame["name_similarity"] = frame.apply(
        lambda row: SequenceMatcher(
            None,
            row["target_name_key"],
            row["candidate_name_key"],
        ).ratio(),
        axis=1,
    )
    frame["is_exact_name_match"] = frame["target_name_key"] == frame["candidate_name_key"]
    frame["is_exact_query_match"] = (
        frame["search_query_key"] != ""
    ) & (frame["search_query_key"] == frame["candidate_name_key"])
    frame["is_exact_code_query_match"] = (
        frame["code_query_key"] != ""
    ) & (frame["code_query_key"] == frame["candidate_code_key"])
    frame["type_bonus"] = frame["candidate_type"].map(_candidate_type_bonus)
    frame["gender_bonus"] = frame["candidate_gender"].map(_candidate_gender_bonus)
    frame["qualifier_penalty"] = frame["candidate_name"].map(_candidate_name_penalty)
    frame["placeholder_penalty"] = frame["candidate_placeholder"].map(
        lambda is_placeholder: -2.0 if bool(is_placeholder) else 0.0
    )
    frame["match_score"] = (
        4.0 * frame["name_similarity"]
        + 2.0 * frame["is_exact_name_match"].astype(float)
        + 0.5 * frame["is_exact_query_match"].astype(float)
        + 1.5 * frame["is_exact_code_query_match"].astype(float)
        + frame["type_bonus"]
        + frame["gender_bonus"]
        + frame["qualifier_penalty"]
        + frame["placeholder_penalty"]
    )

    frame = frame.sort_values(
        ["target_team", "match_score", "is_exact_name_match", "team_id"],
        ascending=[True, False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    frame["rank"] = frame.groupby("target_team").cumcount() + 1
    frame["score_margin"] = (
        frame.groupby("target_team")["match_score"].transform(lambda series: _score_margin(series.tolist()))
    )
    return frame


def build_provider_team_match_suggestions(
    registry: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    provider: str,
    min_score: float = 5.0,
    min_margin: float = 0.75,
) -> pd.DataFrame:
    provider_key = _normalize_provider(provider)
    id_column, name_column = _PROVIDER_COLUMNS[provider_key]

    normalized_registry = normalize_provider_team_registry(registry)
    existing_columns = ["team", id_column, name_column]
    registry_subset = normalized_registry.loc[:, existing_columns].copy()

    ranked = score_provider_team_candidates(candidates)
    if ranked.empty:
        suggestions = registry_subset.copy()
        suggestions["provider"] = provider_key
        suggestions["selection_status"] = "no_candidates"
        suggestions["selection_reason"] = "no_candidates"
        suggestions["selected"] = False
        suggestions["provider_team_id"] = pd.NA
        suggestions["provider_team_name"] = ""
        suggestions["match_score"] = pd.NA
        suggestions["score_margin"] = pd.NA
        suggestions["search_query"] = ""
        suggestions["candidate_type"] = ""
        suggestions["candidate_gender"] = ""
        suggestions["candidate_count"] = 0
        return suggestions

    ranked["target_team"] = ranked["target_team"].astype(str).map(canonicalize_team_name)
    top_candidates = ranked.loc[ranked["rank"] == 1].copy()
    candidate_counts = ranked.groupby("target_team").size().rename("candidate_count")
    top_candidates = top_candidates.merge(
        candidate_counts,
        left_on="target_team",
        right_index=True,
        how="left",
    )
    top_candidates["candidate_count"] = top_candidates["candidate_count"].fillna(0).astype(int)
    top_candidates["selection_status"] = top_candidates.apply(
        lambda row: _selection_status(
            match_score=float(row["match_score"]),
            score_margin=float(row["score_margin"]),
            min_score=min_score,
            min_margin=min_margin,
        ),
        axis=1,
    )
    top_candidates["selection_reason"] = top_candidates.apply(
        lambda row: _selection_reason(
            status=str(row["selection_status"]),
            match_score=float(row["match_score"]),
            score_margin=float(row["score_margin"]),
            min_score=min_score,
            min_margin=min_margin,
        ),
        axis=1,
    )
    top_candidates["selected"] = top_candidates["selection_status"] == "auto_selected"

    suggestions = registry_subset.merge(
        top_candidates.loc[
            :,
            [
                "target_team",
                "team_id",
                "candidate_name",
                "match_score",
                "score_margin",
                "search_query",
                "candidate_type",
                "candidate_gender",
                "candidate_count",
                "selection_status",
                "selection_reason",
                "selected",
            ],
        ],
        left_on="team",
        right_on="target_team",
        how="left",
    )
    suggestions["provider"] = provider_key
    suggestions["provider_team_id"] = suggestions["team_id"]
    suggestions["provider_team_name"] = suggestions["candidate_name"].fillna("")
    suggestions["selection_status"] = suggestions["selection_status"].fillna("no_candidates")
    suggestions["selection_reason"] = suggestions["selection_reason"].fillna("no_candidates")
    suggestions["selected"] = suggestions["selected"].fillna(False).astype(bool)
    suggestions["candidate_count"] = suggestions["candidate_count"].fillna(0).astype(int)
    suggestions["search_query"] = suggestions["search_query"].fillna("")
    suggestions["candidate_type"] = suggestions["candidate_type"].fillna("")
    suggestions["candidate_gender"] = suggestions["candidate_gender"].fillna("")
    suggestions = suggestions.drop(columns=["team_id", "candidate_name", "target_team"])
    return suggestions.loc[
        :,
        [
            "team",
            "provider",
            id_column,
            name_column,
            "provider_team_id",
            "provider_team_name",
            "selected",
            "selection_status",
            "selection_reason",
            "match_score",
            "score_margin",
            "search_query",
            "candidate_type",
            "candidate_gender",
            "candidate_count",
        ],
    ]


def apply_provider_team_match_suggestions(
    registry: pd.DataFrame,
    suggestions: pd.DataFrame,
    *,
    provider: str,
    overwrite: bool = False,
) -> pd.DataFrame:
    provider_key = _normalize_provider(provider)
    id_column, name_column = _PROVIDER_COLUMNS[provider_key]
    normalized_registry = normalize_provider_team_registry(registry)

    if suggestions.empty:
        return normalized_registry

    selected = suggestions.loc[suggestions["selected"].fillna(False)].copy()
    if selected.empty:
        return normalized_registry

    selected["team"] = selected["team"].astype(str).map(canonicalize_team_name)
    selected = selected.drop_duplicates("team", keep="first")
    suggestion_lookup = selected.set_index("team").to_dict(orient="index")

    updated_rows: list[dict[str, Any]] = []
    for row in normalized_registry.to_dict(orient="records"):
        team = canonicalize_team_name(str(row.get("team", "")))
        suggestion = suggestion_lookup.get(team)
        if suggestion is None:
            updated_rows.append(row)
            continue

        if overwrite or pd.isna(row.get(id_column)):
            row[id_column] = suggestion.get("provider_team_id")
        if overwrite or _clean_text(row.get(name_column)) == "":
            row[name_column] = _clean_text(suggestion.get("provider_team_name"))

        note_fragment = (
            f"{provider_key}_auto_selected"
            f"(query={_clean_text(suggestion.get('search_query'))},score={float(suggestion.get('match_score')):.2f})"
        )
        row["notes"] = _merge_note_strings(row.get("notes"), note_fragment)
        updated_rows.append(row)

    return normalize_provider_team_registry(pd.DataFrame.from_records(updated_rows))


def _normalize_provider(provider: str) -> str:
    provider_key = provider.strip().lower()
    if provider_key not in _PROVIDER_COLUMNS:
        raise ValueError(f"Unsupported provider '{provider}'.")
    return provider_key


def _records_from_payload(payload: dict[str, Any] | list[dict[str, Any]]) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "response", "rows", "items"):
            if isinstance(payload.get(key), list):
                return payload[key]
        return [payload]
    return []


def _relation_value(value: Any, field: str) -> Any:
    if isinstance(value, dict):
        if field in value:
            return value.get(field)
        data = value.get("data")
        if isinstance(data, dict):
            return data.get(field)
    if isinstance(value, list):
        for item in value:
            nested = _relation_value(item, field)
            if nested not in (None, ""):
                return nested
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


def _normalize_match_text(value: Any) -> str:
    text = canonicalize_team_name(_clean_text(value))
    text = _NATIONAL_TEAM_PATTERN.sub("", text)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text).strip().lower()
    return re.sub(r"\s+", " ", text)


def _extract_code_query_key(value: Any) -> str:
    cleaned = _clean_text(value)
    if ":" not in cleaned:
        return ""
    prefix, code = cleaned.split(":", 1)
    if prefix.strip().lower() != "code":
        return ""
    return _normalize_match_text(code)


def _candidate_type_bonus(candidate_type: str) -> float:
    lowered = _clean_text(candidate_type).lower()
    if lowered in {"national", "national_team", "country"} or "national" in lowered:
        return 1.5
    if lowered == "domestic" or "club" in lowered:
        return -1.0
    return 0.0


def _candidate_gender_bonus(candidate_gender: str) -> float:
    lowered = _clean_text(candidate_gender).lower()
    if lowered in {"", "male", "men"}:
        return 0.0
    if lowered in {"female", "women"}:
        return -2.0
    return -0.5


def _candidate_name_penalty(candidate_name: str) -> float:
    if _NON_SENIOR_MENS_PATTERN.search(_clean_text(candidate_name)):
        return -2.0
    return 0.0


def _score_margin(scores: list[float]) -> float:
    if not scores:
        return 0.0
    if len(scores) == 1:
        return float(scores[0])
    return float(scores[0] - scores[1])


def _selection_status(
    *,
    match_score: float,
    score_margin: float,
    min_score: float,
    min_margin: float,
) -> str:
    if match_score < min_score:
        return "review_required_low_score"
    if score_margin < min_margin:
        return "review_required_low_margin"
    return "auto_selected"


def _selection_reason(
    *,
    status: str,
    match_score: float,
    score_margin: float,
    min_score: float,
    min_margin: float,
) -> str:
    if status == "auto_selected":
        return f"score={match_score:.2f}; margin={score_margin:.2f}"
    if status == "review_required_low_score":
        return f"score={match_score:.2f} below threshold {min_score:.2f}"
    if status == "review_required_low_margin":
        return f"margin={score_margin:.2f} below threshold {min_margin:.2f}"
    return status


def _merge_note_strings(left: Any, right: Any) -> str:
    left_clean = _clean_text(left)
    right_clean = _clean_text(right)
    if left_clean == "":
        return right_clean
    if right_clean == "" or right_clean in left_clean:
        return left_clean
    return f"{left_clean}; {right_clean}"
