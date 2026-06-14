from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from .env import load_project_env
from .international_results import canonicalize_tournament_name, standardize_international_results

FOOTBALL_DATA_API_BASE_URL = "https://api.football-data.org/v4"
FOOTBALL_DATA_DEFAULT_COMPETITION_CODES = (
    "WC",
    "QCAF",
    "QAFC",
    "QUFA",
    "QOFC",
    "QCBL",
    "QCCF",
    "EC",
    "CA",
)


def get_football_data_api_token(env_var: str = "FOOTBALL_DATA_API_TOKEN") -> str:
    load_project_env()
    token = os.getenv(env_var)
    if not token:
        raise RuntimeError(
            f"Missing {env_var}. Create a football-data.org token and export it before downloading."
        )
    return token


def fetch_football_data_competition_matches(
    competition_code: str,
    *,
    api_token: str,
    season: int | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    status: str | None = "FINISHED",
) -> dict[str, Any]:
    query_params = {
        key: value
        for key, value in {
            "season": season,
            "dateFrom": date_from,
            "dateTo": date_to,
            "status": status,
        }.items()
        if value is not None
    }
    request_url = (
        f"{FOOTBALL_DATA_API_BASE_URL}/competitions/{competition_code}/matches"
        + (f"?{urlencode(query_params)}" if query_params else "")
    )
    request = Request(request_url, headers={"X-Auth-Token": api_token})
    with urlopen(request) as response:
        return json.load(response)


def fetch_football_data_matches(
    *,
    api_token: str,
    competition_codes: Iterable[str] = FOOTBALL_DATA_DEFAULT_COMPETITION_CODES,
    season: int | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    status: str | None = "FINISHED",
) -> pd.DataFrame:
    payloads = []
    for competition_code in competition_codes:
        payloads.append(
            fetch_football_data_competition_matches(
                competition_code,
                api_token=api_token,
                season=season,
                date_from=date_from,
                date_to=date_to,
                status=status,
            )
        )
    return standardize_football_data_payloads(payloads)


def standardize_football_data_payloads(payloads: Iterable[dict[str, Any]]) -> pd.DataFrame:
    raw_columns = [
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tournament",
        "city",
        "country",
        "neutral",
        "source_match_id",
        "source_competition_code",
        "source_competition_name",
        "source_last_updated_utc",
    ]
    raw_rows: list[dict[str, Any]] = []
    for payload in payloads:
        competition = payload.get("competition") or {}
        competition_code = str(competition.get("code", ""))
        competition_name = canonicalize_tournament_name(str(competition.get("name", "")))

        for match in payload.get("matches", []):
            home_team = ((match.get("homeTeam") or {}).get("name")) or ""
            away_team = ((match.get("awayTeam") or {}).get("name")) or ""
            if not home_team or not away_team:
                continue

            score = match.get("score") or {}
            regular_time = score.get("regularTime") or {}
            full_time = score.get("fullTime") or {}
            home_goals = regular_time.get("home")
            away_goals = regular_time.get("away")
            if home_goals is None or away_goals is None:
                home_goals = full_time.get("home")
                away_goals = full_time.get("away")
            if home_goals is None or away_goals is None:
                continue

            raw_rows.append(
                {
                    "date": str(match["utcDate"])[:10],
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_score": int(home_goals),
                    "away_score": int(away_goals),
                    "tournament": competition_name,
                    "city": match.get("venue"),
                    "country": (payload.get("area") or {}).get("name"),
                    "neutral": infer_neutral_site_from_competition_code(competition_code),
                    "source_match_id": match.get("id"),
                    "source_competition_code": competition_code,
                    "source_competition_name": competition.get("name"),
                    "source_last_updated_utc": match.get("lastUpdated"),
                }
            )

    standardized = standardize_international_results(
        pd.DataFrame.from_records(raw_rows, columns=raw_columns)
    )
    standardized["source"] = "football_data_api"
    return standardized


def save_football_data_matches_csv(
    destination: str | Path,
    *,
    api_token: str,
    competition_codes: Iterable[str] = FOOTBALL_DATA_DEFAULT_COMPETITION_CODES,
    season: int | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    status: str | None = "FINISHED",
) -> Path:
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe = fetch_football_data_matches(
        api_token=api_token,
        competition_codes=competition_codes,
        season=season,
        date_from=date_from,
        date_to=date_to,
        status=status,
    )
    dataframe.to_csv(destination_path, index=False)
    return destination_path


def infer_neutral_site_from_competition_code(competition_code: str) -> bool:
    return competition_code in {"WC", "EC", "CA"}
