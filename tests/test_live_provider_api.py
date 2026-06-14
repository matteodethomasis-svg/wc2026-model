from __future__ import annotations

import io
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse

import pandas as pd

from wc2026_model.data import live_provider_api


def test_fetch_sportmonks_expected_lineups_by_team_ids_combines_paginated_pages(
    monkeypatch,
) -> None:
    responses = {
        (500, 1): {
            "data": [
                {
                    "id": 11,
                    "fixture_id": 7001,
                    "player_name": "Mike Maignan",
                    "team_id": 500,
                    "participant": {"id": 500, "name": "France"},
                    "type": {"name": "Starting XI"},
                }
            ],
            "pagination": {"current_page": 1, "last_page": 2},
        },
        (500, 2): {
            "data": [
                {
                    "id": 12,
                    "fixture_id": 7001,
                    "player_name": "Bradley Barcola",
                    "team_id": 500,
                    "participant": {"id": 500, "name": "France"},
                    "type": {"name": "Bench"},
                }
            ],
            "pagination": {"current_page": 2, "last_page": 2},
        },
    }
    seen_requests: list[tuple[int, int, str]] = []

    def fake_read_json_response(request_url: str, *, headers=None):
        parsed = urlparse(request_url)
        query = parse_qs(parsed.query)
        team_id = int(parsed.path.rstrip("/").split("/")[-1])
        page = int(query.get("page", ["1"])[0])
        include = query.get("include", [""])[0]
        seen_requests.append((team_id, page, include))
        return responses[(team_id, page)]

    monkeypatch.setattr(live_provider_api, "_read_json_response", fake_read_json_response)

    payload = live_provider_api.fetch_sportmonks_expected_lineups_by_team_ids(
        [500],
        api_token="secret-token",
        include=("fixture", "participant", "type"),
    )

    assert len(payload["data"]) == 2
    assert payload["meta"]["team_ids"] == [500]
    assert seen_requests == [
        (500, 1, "fixture,participant,type"),
        (500, 2, "fixture,participant,type"),
    ]


def test_fetch_sportmonks_team_search_candidates_collects_query_metadata(monkeypatch) -> None:
    seen_paths: list[str] = []

    def fake_read_json_response(request_url: str, *, headers=None):
        parsed = urlparse(request_url)
        seen_paths.append(parsed.path)
        return {
            "data": [
                {
                    "id": 500,
                    "name": "France",
                    "type": "national",
                    "gender": "male",
                    "country": {"id": 2, "name": "France"},
                }
            ]
        }

    monkeypatch.setattr(live_provider_api, "_read_json_response", fake_read_json_response)

    payload = live_provider_api.fetch_sportmonks_team_search_candidates(
        [{"target_team": "France", "search_query": "France"}],
        api_token="secret-token",
    )

    assert payload["meta"]["provider"] == "sportmonks"
    assert payload["meta"]["target_teams"] == ["France"]
    assert payload["queries"][0]["target_team"] == "France"
    assert payload["queries"][0]["search_query"] == "France"
    assert payload["queries"][0]["response"]["data"][0]["id"] == 500
    assert seen_paths == ["/v3/football/teams/search/France"]


def test_fetch_api_football_team_search_candidates_collects_query_metadata(monkeypatch) -> None:
    seen_paths: list[str] = []
    seen_headers: list[dict[str, str] | None] = []

    def fake_read_json_response(request_url: str, *, headers=None):
        parsed = urlparse(request_url)
        seen_paths.append(parsed.path + ("?" + parsed.query if parsed.query else ""))
        seen_headers.append(headers)
        return {
            "response": [
                {
                    "team": {
                        "id": 2,
                        "name": "France",
                        "code": "FRA",
                        "country": "France",
                        "national": True,
                    }
                }
            ]
        }

    monkeypatch.setattr(live_provider_api, "_read_json_response", fake_read_json_response)

    payload = live_provider_api.fetch_api_football_team_search_candidates(
        [{"target_team": "France", "search_query": "France"}],
        api_key="secret-key",
        api_key_header="x-apisports-key",
    )

    assert payload["meta"]["provider"] == "api_football"
    assert payload["meta"]["target_teams"] == ["France"]
    assert payload["queries"][0]["target_team"] == "France"
    assert payload["queries"][0]["search_query"] == "France"
    assert payload["queries"][0]["response"]["response"][0]["team"]["id"] == 2
    assert seen_headers[0]["x-apisports-key"] == "secret-key"
    assert seen_paths == ["/teams?search=France"]


def test_fetch_api_football_injuries_paginates_and_builds_headers(monkeypatch) -> None:
    responses = {
        1: {
            "response": [
                {
                    "team": {"id": 2, "name": "France"},
                    "player": {"id": 101, "name": "Mike Maignan", "type": "Injury"},
                }
            ],
            "paging": {"current": 1, "total": 2},
        },
        2: {
            "response": [
                {
                    "team": {"id": 3, "name": "Norway"},
                    "player": {"id": 202, "name": "Martin Odegaard", "type": "Doubtful"},
                }
            ],
            "paging": {"current": 2, "total": 2},
        },
    }
    seen_headers: list[dict[str, str] | None] = []
    seen_pages: list[int] = []

    def fake_read_json_response(request_url: str, *, headers=None):
        parsed = urlparse(request_url)
        query = parse_qs(parsed.query)
        page = int(query.get("page", ["1"])[0])
        seen_pages.append(page)
        seen_headers.append(headers)
        return responses[page]

    monkeypatch.setattr(live_provider_api, "_read_json_response", fake_read_json_response)

    payload = live_provider_api.fetch_api_football_injuries(
        api_key="secret-key",
        league=1,
        season=2026,
        api_key_header="x-apisports-key",
    )

    assert len(payload["response"]) == 2
    assert seen_pages == [1, 2]
    assert seen_headers[0]["x-apisports-key"] == "secret-key"
    assert seen_headers[0]["Accept"] == "application/json"


def test_fetch_api_football_injuries_by_team_ids_combines_team_queries(monkeypatch) -> None:
    seen_team_ids: list[int] = []

    def fake_fetch_api_football_injuries(**kwargs):
        team_id = int(kwargs["team"])
        seen_team_ids.append(team_id)
        return {
            "response": [
                {
                    "team": {"id": team_id, "name": f"Team {team_id}"},
                    "player": {"id": 1000 + team_id, "name": f"Player {team_id}", "type": "Injury"},
                }
            ]
        }

    monkeypatch.setattr(
        live_provider_api,
        "fetch_api_football_injuries",
        fake_fetch_api_football_injuries,
    )

    payload = live_provider_api.fetch_api_football_injuries_by_team_ids(
        [2, 3],
        api_key="secret-key",
        league=1,
        season=2026,
    )

    assert seen_team_ids == [2, 3]
    assert len(payload["response"]) == 2
    assert payload["meta"]["team_ids"] == [2, 3]
    assert payload["meta"]["request_count"] == 2


def test_build_api_football_headers_supports_rapidapi_host() -> None:
    headers = live_provider_api._build_api_football_headers(
        "secret-key",
        api_key_header="x-rapidapi-key",
        api_host="v3.football.api-sports.io",
    )

    assert headers["x-rapidapi-key"] == "secret-key"
    assert headers["x-rapidapi-host"] == "v3.football.api-sports.io"


def test_build_api_football_injuries_url_omits_missing_query_params() -> None:
    request_url = live_provider_api._build_api_football_injuries_url(
        league=1,
        season=2026,
        fixture=None,
        team=2,
        player=None,
        date="2026-06-16",
        timezone=None,
        page=3,
    )

    parsed = urlparse(request_url)
    query = parse_qs(parsed.query)

    assert parsed.path.endswith("/injuries")
    assert query == {
        "league": ["1"],
        "season": ["2026"],
        "team": ["2"],
        "date": ["2026-06-16"],
        "page": ["3"],
    }


def test_build_api_football_team_search_url_encodes_query() -> None:
    request_url = live_provider_api._build_api_football_team_search_url("Bosnia and Herzegovina")
    parsed = urlparse(request_url)
    query = parse_qs(parsed.query)

    assert parsed.path.endswith("/teams")
    assert query == {"search": ["Bosnia and Herzegovina"]}


def test_get_api_football_api_key_loads_project_env(monkeypatch) -> None:
    monkeypatch.delenv("API_FOOTBALL_API_KEY", raising=False)

    def fake_load_project_env():
        monkeypatch.setenv("API_FOOTBALL_API_KEY", "loaded-from-dotenv")

    monkeypatch.setattr(live_provider_api, "load_project_env", fake_load_project_env)

    assert live_provider_api.get_api_football_api_key() == "loaded-from-dotenv"


def test_read_json_response_retries_after_429(monkeypatch) -> None:
    calls: list[str] = []
    seen_delays: list[float] = []

    class FakeResponse(io.StringIO):
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self.close()
            return False

    def fake_urlopen(request):
        calls.append(request.full_url)
        if len(calls) == 1:
            raise HTTPError(
                request.full_url,
                429,
                "Too Many Requests",
                {"Retry-After": "0"},
                None,
            )
        return FakeResponse('{"response": [{"team": {"id": 2}}]}')

    monkeypatch.setattr(live_provider_api, "urlopen", fake_urlopen)
    monkeypatch.setattr(live_provider_api, "sleep", lambda seconds: seen_delays.append(seconds))

    payload = live_provider_api._read_json_response("https://example.com/teams?search=France")

    assert len(calls) == 2
    assert seen_delays == [0.0]
    assert payload["response"][0]["team"]["id"] == 2


def test_save_api_football_injuries_csv_writes_normalized_frame(tmp_path, monkeypatch) -> None:
    payload = {
        "response": [
            {
                "team": {"id": 2, "name": "France"},
                "player": {
                    "id": 101,
                    "name": "Mike Maignan",
                    "type": "Injury",
                    "reason": "Hamstring",
                },
                "fixture": {"id": 5001, "date": "2026-06-16T19:00:00+00:00"},
            }
        ]
    }

    monkeypatch.setattr(
        live_provider_api,
        "fetch_api_football_injuries",
        lambda **_: payload,
    )

    destination = tmp_path / "injuries.csv"
    live_provider_api.save_api_football_injuries_csv(destination, api_key="secret-key")
    frame = pd.read_csv(destination)

    assert frame.shape[0] == 1
    assert frame.loc[0, "team"] == "France"
    assert frame.loc[0, "availability_status"] == "unavailable"


def test_save_sportmonks_expected_lineups_outputs_fetches_once_and_writes_both_files(
    tmp_path,
    monkeypatch,
) -> None:
    payload = {
        "data": [
            {
                "id": 11,
                "fixture_id": 19347797,
                "player_id": 101,
                "team_id": 500,
                "player_name": "Mike Maignan",
                "participant": {"id": 500, "name": "France"},
                "fixture": {"id": 19347797, "starting_at": "2026-06-16T19:00:00Z"},
                "type": {"id": 77614, "name": "Starting XI"},
            }
        ]
    }
    seen_calls: list[list[int]] = []

    def fake_fetch(team_ids, **kwargs):
        seen_calls.append(list(team_ids))
        return payload

    monkeypatch.setattr(
        live_provider_api,
        "fetch_sportmonks_expected_lineups_by_team_ids",
        fake_fetch,
    )

    raw_path, csv_path = live_provider_api.save_sportmonks_expected_lineups_outputs(
        raw_destination=tmp_path / "lineups.json",
        csv_destination=tmp_path / "lineups.csv",
        team_ids=[500],
        api_token="secret-token",
    )

    assert seen_calls == [[500]]
    assert raw_path.exists()
    assert csv_path.exists()
    frame = pd.read_csv(csv_path)
    assert frame.loc[0, "team"] == "France"


def test_save_api_football_injuries_by_team_ids_outputs_fetches_once_and_writes_both_files(
    tmp_path,
    monkeypatch,
) -> None:
    payload = {
        "response": [
            {
                "team": {"id": 2, "name": "France"},
                "player": {
                    "id": 101,
                    "name": "Mike Maignan",
                    "type": "Injury",
                    "reason": "Hamstring",
                },
                "fixture": {"id": 5001, "date": "2026-06-16T19:00:00+00:00"},
            }
        ]
    }
    seen_calls: list[list[int]] = []

    def fake_fetch(team_ids, **kwargs):
        seen_calls.append(list(team_ids))
        return payload

    monkeypatch.setattr(
        live_provider_api,
        "fetch_api_football_injuries_by_team_ids",
        fake_fetch,
    )

    raw_path, csv_path = live_provider_api.save_api_football_injuries_by_team_ids_outputs(
        raw_destination=tmp_path / "injuries.json",
        csv_destination=tmp_path / "injuries.csv",
        team_ids=[2],
        api_key="secret-key",
    )

    assert seen_calls == [[2]]
    assert raw_path.exists()
    assert csv_path.exists()
    frame = pd.read_csv(csv_path)
    assert frame.loc[0, "team"] == "France"


def test_save_api_football_injuries_by_team_ids_csv_writes_normalized_frame(
    tmp_path,
    monkeypatch,
) -> None:
    payload = {
        "response": [
            {
                "team": {"id": 2, "name": "France"},
                "player": {
                    "id": 101,
                    "name": "Mike Maignan",
                    "type": "Injury",
                    "reason": "Hamstring",
                },
                "fixture": {"id": 5001, "date": "2026-06-16T19:00:00+00:00"},
            },
            {
                "team": {"id": 3, "name": "Norway"},
                "player": {
                    "id": 102,
                    "name": "Martin Odegaard",
                    "type": "Doubtful",
                    "reason": "Knock",
                },
                "fixture": {"id": 5002, "date": "2026-06-16T22:00:00+00:00"},
            },
        ]
    }

    monkeypatch.setattr(
        live_provider_api,
        "fetch_api_football_injuries_by_team_ids",
        lambda *_, **__: payload,
    )

    destination = tmp_path / "injuries_by_team.csv"
    live_provider_api.save_api_football_injuries_by_team_ids_csv(
        destination,
        team_ids=[2, 3],
        api_key="secret-key",
    )
    frame = pd.read_csv(destination)

    assert frame.shape[0] == 2
    assert sorted(frame["team"].tolist()) == ["France", "Norway"]
