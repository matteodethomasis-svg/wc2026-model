from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from time import sleep
from typing import Any, Iterable
from urllib.error import HTTPError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

import pandas as pd

from .env import load_project_env
from .live_provider_feeds import (
    API_FOOTBALL_BASE_URL,
    SPORTMONKS_FOOTBALL_API_BASE_URL,
    standardize_api_football_injuries_payload,
    standardize_sportmonks_expected_lineups_payload,
)

DEFAULT_SPORTMONKS_EXPECTED_LINEUPS_INCLUDES = (
    "fixture",
    "participant",
    "type",
)
DEFAULT_SPORTMONKS_TEAM_SEARCH_INCLUDES = ("country",)
DEFAULT_API_FOOTBALL_HOST = "v3.football.api-sports.io"


def get_sportmonks_api_token(env_var: str = "SPORTMONKS_API_TOKEN") -> str:
    load_project_env()
    token = os.getenv(env_var)
    if not token:
        raise RuntimeError(
            f"Missing {env_var}. Create a Sportmonks token and export it before downloading."
        )
    return token


def get_api_football_api_key(env_var: str = "API_FOOTBALL_API_KEY") -> str:
    load_project_env()
    api_key = os.getenv(env_var)
    if not api_key:
        raise RuntimeError(
            f"Missing {env_var}. Create an API-Football key and export it before downloading."
        )
    return api_key


def get_api_football_api_key_header(
    env_var: str = "API_FOOTBALL_API_KEY_HEADER",
    *,
    default: str = "x-apisports-key",
) -> str:
    load_project_env()
    return os.getenv(env_var, default).strip() or default


def get_api_football_host(
    env_var: str = "API_FOOTBALL_HOST",
    *,
    default: str = DEFAULT_API_FOOTBALL_HOST,
) -> str:
    load_project_env()
    return os.getenv(env_var, default).strip() or default


def fetch_sportmonks_expected_lineups_by_team_ids(
    team_ids: Iterable[int | str],
    *,
    api_token: str,
    include: Iterable[str] = DEFAULT_SPORTMONKS_EXPECTED_LINEUPS_INCLUDES,
    per_page: int | None = None,
    request_pause_seconds: float = 0.0,
) -> dict[str, Any]:
    payloads: list[dict[str, Any]] = []
    resolved_team_ids: list[int] = []

    for raw_team_id in team_ids:
        team_id = int(raw_team_id)
        resolved_team_ids.append(team_id)
        payloads.extend(
            _fetch_sportmonks_expected_lineup_pages_for_team(
                team_id,
                api_token=api_token,
                include=include,
                per_page=per_page,
                request_pause_seconds=request_pause_seconds,
            )
        )

    combined_data: list[Any] = []
    for payload in payloads:
        combined_data.extend(payload.get("data") or [])

    return {
        "data": combined_data,
        "meta": {
            "provider": "sportmonks",
            "team_ids": resolved_team_ids,
            "request_count": len(payloads),
        },
    }


def fetch_sportmonks_teams_by_search(
    search_query: str,
    *,
    api_token: str,
    include: Iterable[str] = DEFAULT_SPORTMONKS_TEAM_SEARCH_INCLUDES,
    per_page: int | None = None,
) -> dict[str, Any]:
    request_url = _build_sportmonks_team_search_url(
        search_query,
        api_token=api_token,
        include=include,
        per_page=per_page,
    )
    payload = _read_json_response(request_url)
    if not isinstance(payload, dict):
        raise ValueError("Sportmonks team-search response must be a JSON object.")
    return payload


def fetch_sportmonks_team_search_candidates(
    searches: Iterable[dict[str, Any] | tuple[str, str]],
    *,
    api_token: str,
    include: Iterable[str] = DEFAULT_SPORTMONKS_TEAM_SEARCH_INCLUDES,
    per_page: int | None = None,
    request_pause_seconds: float = 0.0,
) -> dict[str, Any]:
    query_payloads: list[dict[str, Any]] = []
    target_teams: list[str] = []
    search_queries: list[str] = []

    for raw_search in searches:
        if isinstance(raw_search, dict):
            target_team = str(raw_search.get("target_team", "")).strip()
            search_query = str(raw_search.get("search_query", "")).strip()
        else:
            target_team = str(raw_search[0]).strip()
            search_query = str(raw_search[1]).strip()

        if target_team == "" or search_query == "":
            continue

        payload = fetch_sportmonks_teams_by_search(
            search_query,
            api_token=api_token,
            include=include,
            per_page=per_page,
        )
        query_payloads.append(
            {
                "target_team": target_team,
                "search_query": search_query,
                "response": payload,
            }
        )
        target_teams.append(target_team)
        search_queries.append(search_query)
        if request_pause_seconds > 0:
            sleep(request_pause_seconds)

    return {
        "queries": query_payloads,
        "meta": {
            "provider": "sportmonks",
            "target_teams": list(dict.fromkeys(target_teams)),
            "search_queries": search_queries,
            "request_count": len(query_payloads),
        },
    }


def fetch_api_football_teams_by_search(
    search_query: str,
    *,
    api_key: str,
    api_key_header: str = "x-apisports-key",
    api_host: str = DEFAULT_API_FOOTBALL_HOST,
) -> dict[str, Any]:
    request_url = _build_api_football_team_search_url(search_query)
    payload = _read_json_response(
        request_url,
        headers=_build_api_football_headers(
            api_key,
            api_key_header=api_key_header,
            api_host=api_host,
        ),
    )
    if not isinstance(payload, dict):
        raise ValueError("API-Football team-search response must be a JSON object.")
    return payload


def fetch_api_football_team_search_candidates(
    searches: Iterable[dict[str, Any] | tuple[str, str]],
    *,
    api_key: str,
    api_key_header: str = "x-apisports-key",
    api_host: str = DEFAULT_API_FOOTBALL_HOST,
    request_pause_seconds: float = 0.0,
) -> dict[str, Any]:
    query_payloads: list[dict[str, Any]] = []
    target_teams: list[str] = []
    search_queries: list[str] = []

    for raw_search in searches:
        if isinstance(raw_search, dict):
            target_team = str(raw_search.get("target_team", "")).strip()
            search_query = str(raw_search.get("search_query", "")).strip()
        else:
            target_team = str(raw_search[0]).strip()
            search_query = str(raw_search[1]).strip()

        if target_team == "" or search_query == "":
            continue

        payload = fetch_api_football_teams_by_search(
            search_query,
            api_key=api_key,
            api_key_header=api_key_header,
            api_host=api_host,
        )
        query_payloads.append(
            {
                "target_team": target_team,
                "search_query": search_query,
                "response": payload,
            }
        )
        target_teams.append(target_team)
        search_queries.append(search_query)
        if request_pause_seconds > 0:
            sleep(request_pause_seconds)

    return {
        "queries": query_payloads,
        "meta": {
            "provider": "api_football",
            "target_teams": list(dict.fromkeys(target_teams)),
            "search_queries": search_queries,
            "request_count": len(query_payloads),
        },
    }


def save_sportmonks_expected_lineups_json(
    destination: str | Path,
    *,
    team_ids: Iterable[int | str],
    api_token: str,
    include: Iterable[str] = DEFAULT_SPORTMONKS_EXPECTED_LINEUPS_INCLUDES,
    per_page: int | None = None,
    request_pause_seconds: float = 0.0,
) -> Path:
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    payload = fetch_sportmonks_expected_lineups_by_team_ids(
        team_ids,
        api_token=api_token,
        include=include,
        per_page=per_page,
        request_pause_seconds=request_pause_seconds,
    )
    destination_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return destination_path


def save_sportmonks_expected_lineups_outputs(
    *,
    raw_destination: str | Path,
    csv_destination: str | Path,
    team_ids: Iterable[int | str],
    api_token: str,
    include: Iterable[str] = DEFAULT_SPORTMONKS_EXPECTED_LINEUPS_INCLUDES,
    per_page: int | None = None,
    request_pause_seconds: float = 0.0,
) -> tuple[Path, Path]:
    payload = fetch_sportmonks_expected_lineups_by_team_ids(
        team_ids,
        api_token=api_token,
        include=include,
        per_page=per_page,
        request_pause_seconds=request_pause_seconds,
    )
    raw_path = _write_json_payload(raw_destination, payload)
    csv_path = _write_sportmonks_expected_lineups_csv_payload(csv_destination, payload)
    return raw_path, csv_path


def save_sportmonks_team_search_candidates_json(
    destination: str | Path,
    *,
    searches: Iterable[dict[str, Any] | tuple[str, str]],
    api_token: str,
    include: Iterable[str] = DEFAULT_SPORTMONKS_TEAM_SEARCH_INCLUDES,
    per_page: int | None = None,
    request_pause_seconds: float = 0.0,
) -> Path:
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    payload = fetch_sportmonks_team_search_candidates(
        searches,
        api_token=api_token,
        include=include,
        per_page=per_page,
        request_pause_seconds=request_pause_seconds,
    )
    destination_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return destination_path


def save_api_football_team_search_candidates_json(
    destination: str | Path,
    *,
    searches: Iterable[dict[str, Any] | tuple[str, str]],
    api_key: str,
    api_key_header: str = "x-apisports-key",
    api_host: str = DEFAULT_API_FOOTBALL_HOST,
    request_pause_seconds: float = 0.0,
) -> Path:
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    payload = fetch_api_football_team_search_candidates(
        searches,
        api_key=api_key,
        api_key_header=api_key_header,
        api_host=api_host,
        request_pause_seconds=request_pause_seconds,
    )
    destination_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return destination_path


def save_sportmonks_expected_lineups_csv(
    destination: str | Path,
    *,
    team_ids: Iterable[int | str],
    api_token: str,
    include: Iterable[str] = DEFAULT_SPORTMONKS_EXPECTED_LINEUPS_INCLUDES,
    per_page: int | None = None,
    request_pause_seconds: float = 0.0,
) -> Path:
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    payload = fetch_sportmonks_expected_lineups_by_team_ids(
        team_ids,
        api_token=api_token,
        include=include,
        per_page=per_page,
        request_pause_seconds=request_pause_seconds,
    )
    frame = standardize_sportmonks_expected_lineups_payload(payload)
    frame.to_csv(destination_path, index=False)
    return destination_path


def fetch_api_football_injuries(
    *,
    api_key: str,
    league: int | None = None,
    season: int | None = None,
    fixture: int | None = None,
    team: int | None = None,
    player: int | None = None,
    date: str | None = None,
    timezone: str | None = None,
    page: int | None = None,
    api_key_header: str = "x-apisports-key",
    api_host: str = DEFAULT_API_FOOTBALL_HOST,
) -> dict[str, Any]:
    payloads = _fetch_api_football_injury_pages(
        api_key=api_key,
        league=league,
        season=season,
        fixture=fixture,
        team=team,
        player=player,
        date=date,
        timezone=timezone,
        page=page,
        api_key_header=api_key_header,
        api_host=api_host,
    )

    combined_response: list[Any] = []
    for payload in payloads:
        combined_response.extend(payload.get("response") or [])

    return {
        "get": "injuries",
        "response": combined_response,
        "paging": {
            "current": 1 if combined_response else 0,
            "total": len(payloads),
        },
        "parameters": {
            key: value
            for key, value in {
                "league": league,
                "season": season,
                "fixture": fixture,
                "team": team,
                "player": player,
                "date": date,
                "timezone": timezone,
            }.items()
            if value is not None
        },
    }


def fetch_api_football_injuries_by_team_ids(
    team_ids: Iterable[int | str],
    *,
    api_key: str,
    league: int | None = None,
    season: int | None = None,
    fixture: int | None = None,
    date: str | None = None,
    timezone: str | None = None,
    page: int | None = None,
    api_key_header: str = "x-apisports-key",
    api_host: str = DEFAULT_API_FOOTBALL_HOST,
) -> dict[str, Any]:
    combined_response: list[Any] = []
    resolved_team_ids: list[int] = []

    for raw_team_id in team_ids:
        team_id = int(raw_team_id)
        resolved_team_ids.append(team_id)
        payload = fetch_api_football_injuries(
            api_key=api_key,
            league=league,
            season=season,
            fixture=fixture,
            team=team_id,
            player=None,
            date=date,
            timezone=timezone,
            page=page,
            api_key_header=api_key_header,
            api_host=api_host,
        )
        combined_response.extend(payload.get("response") or [])

    return {
        "get": "injuries",
        "response": combined_response,
        "paging": {
            "current": 1 if combined_response else 0,
            "total": len(resolved_team_ids),
        },
        "parameters": {
            key: value
            for key, value in {
                "league": league,
                "season": season,
                "fixture": fixture,
                "date": date,
                "timezone": timezone,
            }.items()
            if value is not None
        },
        "meta": {
            "provider": "api_football",
            "team_ids": resolved_team_ids,
            "request_count": len(resolved_team_ids),
        },
    }


def save_api_football_injuries_json(
    destination: str | Path,
    *,
    api_key: str,
    league: int | None = None,
    season: int | None = None,
    fixture: int | None = None,
    team: int | None = None,
    player: int | None = None,
    date: str | None = None,
    timezone: str | None = None,
    page: int | None = None,
    api_key_header: str = "x-apisports-key",
    api_host: str = DEFAULT_API_FOOTBALL_HOST,
) -> Path:
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    payload = fetch_api_football_injuries(
        api_key=api_key,
        league=league,
        season=season,
        fixture=fixture,
        team=team,
        player=player,
        date=date,
        timezone=timezone,
        page=page,
        api_key_header=api_key_header,
        api_host=api_host,
    )
    destination_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return destination_path


def save_api_football_injuries_outputs(
    *,
    raw_destination: str | Path,
    csv_destination: str | Path,
    api_key: str,
    league: int | None = None,
    season: int | None = None,
    fixture: int | None = None,
    team: int | None = None,
    player: int | None = None,
    date: str | None = None,
    timezone: str | None = None,
    page: int | None = None,
    api_key_header: str = "x-apisports-key",
    api_host: str = DEFAULT_API_FOOTBALL_HOST,
) -> tuple[Path, Path]:
    payload = fetch_api_football_injuries(
        api_key=api_key,
        league=league,
        season=season,
        fixture=fixture,
        team=team,
        player=player,
        date=date,
        timezone=timezone,
        page=page,
        api_key_header=api_key_header,
        api_host=api_host,
    )
    raw_path = _write_json_payload(raw_destination, payload)
    csv_path = _write_api_football_injuries_csv_payload(csv_destination, payload)
    return raw_path, csv_path


def save_api_football_injuries_by_team_ids_json(
    destination: str | Path,
    *,
    team_ids: Iterable[int | str],
    api_key: str,
    league: int | None = None,
    season: int | None = None,
    fixture: int | None = None,
    date: str | None = None,
    timezone: str | None = None,
    page: int | None = None,
    api_key_header: str = "x-apisports-key",
    api_host: str = DEFAULT_API_FOOTBALL_HOST,
) -> Path:
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    payload = fetch_api_football_injuries_by_team_ids(
        team_ids,
        api_key=api_key,
        league=league,
        season=season,
        fixture=fixture,
        date=date,
        timezone=timezone,
        page=page,
        api_key_header=api_key_header,
        api_host=api_host,
    )
    destination_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return destination_path


def save_api_football_injuries_by_team_ids_outputs(
    *,
    raw_destination: str | Path,
    csv_destination: str | Path,
    team_ids: Iterable[int | str],
    api_key: str,
    league: int | None = None,
    season: int | None = None,
    fixture: int | None = None,
    date: str | None = None,
    timezone: str | None = None,
    page: int | None = None,
    api_key_header: str = "x-apisports-key",
    api_host: str = DEFAULT_API_FOOTBALL_HOST,
) -> tuple[Path, Path]:
    payload = fetch_api_football_injuries_by_team_ids(
        team_ids,
        api_key=api_key,
        league=league,
        season=season,
        fixture=fixture,
        date=date,
        timezone=timezone,
        page=page,
        api_key_header=api_key_header,
        api_host=api_host,
    )
    raw_path = _write_json_payload(raw_destination, payload)
    csv_path = _write_api_football_injuries_csv_payload(csv_destination, payload)
    return raw_path, csv_path


def save_api_football_injuries_csv(
    destination: str | Path,
    *,
    api_key: str,
    league: int | None = None,
    season: int | None = None,
    fixture: int | None = None,
    team: int | None = None,
    player: int | None = None,
    date: str | None = None,
    timezone: str | None = None,
    page: int | None = None,
    api_key_header: str = "x-apisports-key",
    api_host: str = DEFAULT_API_FOOTBALL_HOST,
) -> Path:
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    payload = fetch_api_football_injuries(
        api_key=api_key,
        league=league,
        season=season,
        fixture=fixture,
        team=team,
        player=player,
        date=date,
        timezone=timezone,
        page=page,
        api_key_header=api_key_header,
        api_host=api_host,
    )
    frame = standardize_api_football_injuries_payload(payload)
    frame.to_csv(destination_path, index=False)
    return destination_path


def save_api_football_injuries_by_team_ids_csv(
    destination: str | Path,
    *,
    team_ids: Iterable[int | str],
    api_key: str,
    league: int | None = None,
    season: int | None = None,
    fixture: int | None = None,
    date: str | None = None,
    timezone: str | None = None,
    page: int | None = None,
    api_key_header: str = "x-apisports-key",
    api_host: str = DEFAULT_API_FOOTBALL_HOST,
) -> Path:
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    payload = fetch_api_football_injuries_by_team_ids(
        team_ids,
        api_key=api_key,
        league=league,
        season=season,
        fixture=fixture,
        date=date,
        timezone=timezone,
        page=page,
        api_key_header=api_key_header,
        api_host=api_host,
    )
    frame = standardize_api_football_injuries_payload(payload)
    frame.to_csv(destination_path, index=False)
    return destination_path


def _write_json_payload(
    destination: str | Path,
    payload: dict[str, Any],
) -> Path:
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return destination_path


def _write_sportmonks_expected_lineups_csv_payload(
    destination: str | Path,
    payload: dict[str, Any],
) -> Path:
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    frame = standardize_sportmonks_expected_lineups_payload(payload)
    frame.to_csv(destination_path, index=False)
    return destination_path


def _write_api_football_injuries_csv_payload(
    destination: str | Path,
    payload: dict[str, Any],
) -> Path:
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    frame = standardize_api_football_injuries_payload(payload)
    frame.to_csv(destination_path, index=False)
    return destination_path


def download_url_to_path(
    destination: str | Path,
    *,
    source_url: str,
    headers: dict[str, str] | None = None,
    overwrite: bool = False,
) -> Path:
    destination_path = Path(destination)
    if destination_path.exists() and not overwrite:
        return destination_path

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    request = Request(source_url, headers=headers or {})
    with urlopen(request) as response, destination_path.open("wb") as output_file:
        shutil.copyfileobj(response, output_file)
    return destination_path


def _fetch_sportmonks_expected_lineup_pages_for_team(
    team_id: int,
    *,
    api_token: str,
    include: Iterable[str],
    per_page: int | None,
    request_pause_seconds: float,
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    page = 1
    while True:
        request_url = _build_sportmonks_expected_lineup_team_url(
            team_id,
            api_token=api_token,
            include=include,
            page=page,
            per_page=per_page,
        )
        payload = _read_json_response(request_url)
        payloads.append(payload)
        if not _sportmonks_payload_has_next_page(payload):
            break
        page += 1
        if request_pause_seconds > 0:
            sleep(request_pause_seconds)
    return payloads


def _fetch_api_football_injury_pages(
    *,
    api_key: str,
    league: int | None,
    season: int | None,
    fixture: int | None,
    team: int | None,
    player: int | None,
    date: str | None,
    timezone: str | None,
    page: int | None,
    api_key_header: str,
    api_host: str,
) -> list[dict[str, Any]]:
    headers = _build_api_football_headers(
        api_key,
        api_key_header=api_key_header,
        api_host=api_host,
    )
    payloads: list[dict[str, Any]] = []
    current_page = page or 1

    while True:
        request_url = _build_api_football_injuries_url(
            league=league,
            season=season,
            fixture=fixture,
            team=team,
            player=player,
            date=date,
            timezone=timezone,
            page=current_page,
        )
        payload = _read_json_response(request_url, headers=headers)
        payloads.append(payload)
        if page is not None or not _api_football_payload_has_next_page(payload):
            break
        current_page += 1
    return payloads


def _build_sportmonks_expected_lineup_team_url(
    team_id: int,
    *,
    api_token: str,
    include: Iterable[str],
    page: int | None,
    per_page: int | None,
) -> str:
    query_params = {
        "api_token": api_token,
        "include": ",".join([item.strip() for item in include if str(item).strip()]),
        "page": page,
        "per_page": per_page,
    }
    encoded = urlencode({key: value for key, value in query_params.items() if value is not None})
    return f"{SPORTMONKS_FOOTBALL_API_BASE_URL}/expected-lineups/teams/{team_id}?{encoded}"


def _build_sportmonks_team_search_url(
    search_query: str,
    *,
    api_token: str,
    include: Iterable[str],
    per_page: int | None,
) -> str:
    query_params = {
        "api_token": api_token,
        "include": ",".join([item.strip() for item in include if str(item).strip()]),
        "per_page": per_page,
    }
    encoded = urlencode({key: value for key, value in query_params.items() if value is not None})
    escaped_query = quote(search_query.strip(), safe="")
    return f"{SPORTMONKS_FOOTBALL_API_BASE_URL}/teams/search/{escaped_query}?{encoded}"


def _build_api_football_injuries_url(
    *,
    league: int | None,
    season: int | None,
    fixture: int | None,
    team: int | None,
    player: int | None,
    date: str | None,
    timezone: str | None,
    page: int | None,
) -> str:
    query_params = {
        key: value
        for key, value in {
            "league": league,
            "season": season,
            "fixture": fixture,
            "team": team,
            "player": player,
            "date": date,
            "timezone": timezone,
            "page": page,
        }.items()
        if value is not None
    }
    encoded = urlencode(query_params)
    base_url = f"{API_FOOTBALL_BASE_URL}/injuries"
    return base_url if encoded == "" else f"{base_url}?{encoded}"


def _build_api_football_team_search_url(search_query: str) -> str:
    cleaned_query = search_query.strip()
    lookup_key = "search"
    lookup_value = cleaned_query
    if ":" in cleaned_query:
        prefix, value = cleaned_query.split(":", 1)
        normalized_prefix = prefix.strip().lower()
        if normalized_prefix in {"search", "code", "country"}:
            lookup_key = normalized_prefix
            lookup_value = value.strip()
    encoded = urlencode({lookup_key: lookup_value})
    return f"{API_FOOTBALL_BASE_URL}/teams?{encoded}"


def _build_api_football_headers(
    api_key: str,
    *,
    api_key_header: str,
    api_host: str,
) -> dict[str, str]:
    headers = {
        api_key_header: api_key,
        "Accept": "application/json",
    }
    if api_key_header.lower() == "x-rapidapi-key":
        headers["x-rapidapi-host"] = api_host
    return headers


def _sportmonks_payload_has_next_page(payload: dict[str, Any]) -> bool:
    pagination = payload.get("pagination")
    if not isinstance(pagination, dict):
        pagination = ((payload.get("meta") or {}).get("pagination")) or {}

    has_more = pagination.get("has_more")
    if has_more is not None:
        return bool(has_more)

    next_page = pagination.get("next_page")
    if next_page not in (None, ""):
        return True

    current_page = pagination.get("current_page") or pagination.get("currentPage")
    last_page = pagination.get("last_page") or pagination.get("lastPage")
    if current_page is not None and last_page is not None:
        return int(current_page) < int(last_page)
    return False


def _api_football_payload_has_next_page(payload: dict[str, Any]) -> bool:
    paging = payload.get("paging") or {}
    current_page = paging.get("current")
    total_pages = paging.get("total")
    if current_page is None or total_pages is None:
        return False
    return int(current_page) < int(total_pages)


def _read_json_response(
    request_url: str,
    *,
    headers: dict[str, str] | None = None,
    max_rate_limit_retries: int = 2,
    base_rate_limit_delay_seconds: float = 1.0,
) -> dict[str, Any]:
    request = Request(request_url, headers=headers or {})
    attempts = 0
    while True:
        try:
            with urlopen(request) as response:
                return json.load(response)
        except HTTPError as exc:
            if exc.code != 429 or attempts >= max_rate_limit_retries:
                raise
            retry_after = None
            if exc.headers is not None:
                retry_after = exc.headers.get("Retry-After")
            delay_seconds = _resolve_rate_limit_delay_seconds(
                retry_after,
                base_delay_seconds=base_rate_limit_delay_seconds,
                attempts=attempts,
            )
            sleep(delay_seconds)
            attempts += 1


def _resolve_rate_limit_delay_seconds(
    retry_after: str | None,
    *,
    base_delay_seconds: float,
    attempts: int,
) -> float:
    if retry_after not in (None, ""):
        try:
            return max(float(retry_after), 0.0)
        except (TypeError, ValueError):
            pass
    return max(base_delay_seconds * (2**attempts), 0.0)
