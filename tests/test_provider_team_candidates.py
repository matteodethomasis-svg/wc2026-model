from __future__ import annotations

import pandas as pd

from wc2026_model.data import (
    apply_provider_team_match_suggestions,
    build_api_football_team_search_queries,
    build_provider_team_match_suggestions,
    build_team_search_queries,
    score_provider_team_candidates,
    standardize_api_football_team_search_payload,
    standardize_sportmonks_team_search_payload,
)


def test_build_team_search_queries_adds_known_world_cup_aliases() -> None:
    queries = build_team_search_queries("Turkey")

    assert queries == ["Turkey", "Türkiye", "Turkiye"]


def test_build_api_football_team_search_queries_adds_code_fallback() -> None:
    queries = build_api_football_team_search_queries("United States")

    assert queries == ["United States", "USA", "United States of America", "code:USA"]


def test_standardize_sportmonks_team_search_payload_extracts_query_context() -> None:
    payload = {
        "queries": [
            {
                "target_team": "Turkey",
                "search_query": "TÃ¼rkiye",
                "response": {
                    "data": [
                        {
                            "id": 825,
                            "name": "Turkey",
                            "short_code": "TUR",
                            "type": "national",
                            "gender": "male",
                            "country": {"id": 792, "name": "Turkey"},
                            "last_played_at": "2026-06-12 20:00:00",
                        }
                    ]
                },
            }
        ]
    }

    frame = standardize_sportmonks_team_search_payload(payload)

    assert frame.shape[0] == 1
    assert frame.loc[0, "target_team"] == "Turkey"
    assert frame.loc[0, "search_query"] == "TÃ¼rkiye"
    assert frame.loc[0, "team_id"] == 825
    assert frame.loc[0, "candidate_name"] == "Turkey"
    assert frame.loc[0, "candidate_type"] == "national"
    assert frame.loc[0, "source"] == "sportmonks_team_search"


def test_standardize_api_football_team_search_payload_extracts_query_context() -> None:
    payload = {
        "queries": [
            {
                "target_team": "France",
                "search_query": "France",
                "response": {
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
                },
            }
        ]
    }

    frame = standardize_api_football_team_search_payload(payload)

    assert frame.shape[0] == 1
    assert frame.loc[0, "target_team"] == "France"
    assert frame.loc[0, "search_query"] == "France"
    assert frame.loc[0, "team_id"] == 2
    assert frame.loc[0, "candidate_name"] == "France"
    assert frame.loc[0, "candidate_short_code"] == "FRA"
    assert frame.loc[0, "candidate_type"] == "national"
    assert frame.loc[0, "candidate_country_name"] == "France"
    assert frame.loc[0, "source"] == "api_football_team_search"


def test_build_provider_team_match_suggestions_prefers_senior_national_team() -> None:
    registry = pd.DataFrame(
        [
            {"group": "I", "slot": 1, "team": "France"},
            {"group": "I", "slot": 2, "team": "Turkey"},
        ]
    )
    candidates = pd.DataFrame(
        [
            {
                "provider": "sportmonks",
                "target_team": "France",
                "search_query": "France",
                "team_id": 100,
                "candidate_name": "France",
                "candidate_type": "national",
                "candidate_gender": "male",
                "candidate_placeholder": False,
            },
            {
                "provider": "sportmonks",
                "target_team": "France",
                "search_query": "France",
                "team_id": 101,
                "candidate_name": "France Women",
                "candidate_type": "national",
                "candidate_gender": "female",
                "candidate_placeholder": False,
            },
            {
                "provider": "sportmonks",
                "target_team": "Turkey",
                "search_query": "Turkey",
                "team_id": 200,
                "candidate_name": "Turkey U21",
                "candidate_type": "national",
                "candidate_gender": "male",
                "candidate_placeholder": False,
            },
            {
                "provider": "sportmonks",
                "target_team": "Turkey",
                "search_query": "TÃ¼rkiye",
                "team_id": 201,
                "candidate_name": "Turkey",
                "candidate_type": "national",
                "candidate_gender": "male",
                "candidate_placeholder": False,
            },
        ]
    )

    ranked = score_provider_team_candidates(candidates)
    suggestions = build_provider_team_match_suggestions(
        registry,
        ranked,
        provider="sportmonks",
    )

    france = suggestions.loc[suggestions["team"] == "France"].iloc[0]
    turkey = suggestions.loc[suggestions["team"] == "Turkey"].iloc[0]

    assert bool(france["selected"]) is True
    assert france["provider_team_id"] == 100
    assert france["provider_team_name"] == "France"
    assert france["selection_status"] == "auto_selected"
    assert bool(turkey["selected"]) is True
    assert turkey["provider_team_id"] == 201
    assert turkey["provider_team_name"] == "Turkey"


def test_build_provider_team_match_suggestions_uses_api_football_code_queries() -> None:
    registry = pd.DataFrame(
        [
            {"group": "D", "slot": 1, "team": "United States"},
            {"group": "D", "slot": 4, "team": "Turkey"},
        ]
    )
    candidates = pd.DataFrame(
        [
            {
                "provider": "api_football",
                "target_team": "United States",
                "search_query": "code:USA",
                "team_id": 2384,
                "candidate_name": "USA",
                "candidate_short_code": "USA",
                "candidate_type": "national",
                "candidate_gender": "",
                "candidate_placeholder": False,
            },
            {
                "provider": "api_football",
                "target_team": "Turkey",
                "search_query": "code:TUR",
                "team_id": 777,
                "candidate_name": "Türkiye",
                "candidate_short_code": "TUR",
                "candidate_type": "national",
                "candidate_gender": "",
                "candidate_placeholder": False,
            },
            {
                "provider": "api_football",
                "target_team": "Turkey",
                "search_query": "code:TUR",
                "team_id": 1539,
                "candidate_name": "Turkmenistan",
                "candidate_short_code": "TUR",
                "candidate_type": "national",
                "candidate_gender": "",
                "candidate_placeholder": False,
            },
        ]
    )

    ranked = score_provider_team_candidates(candidates)
    suggestions = build_provider_team_match_suggestions(
        registry,
        ranked,
        provider="api_football",
    )

    united_states = suggestions.loc[suggestions["team"] == "United States"].iloc[0]
    turkey = suggestions.loc[suggestions["team"] == "Turkey"].iloc[0]

    assert bool(united_states["selected"]) is True
    assert united_states["provider_team_id"] == 2384
    assert united_states["provider_team_name"] == "USA"
    assert bool(turkey["selected"]) is True
    assert turkey["provider_team_id"] == 777
    assert turkey["provider_team_name"] == "Türkiye"


def test_apply_provider_team_match_suggestions_updates_registry_notes() -> None:
    registry = pd.DataFrame(
        [
            {
                "group": "I",
                "slot": 1,
                "team": "France",
                "sportmonks_team_id": None,
                "sportmonks_team_name": "",
                "api_football_team_id": None,
                "api_football_team_name": "",
                "notes": "",
            }
        ]
    )
    suggestions = pd.DataFrame(
        [
            {
                "team": "France",
                "provider": "sportmonks",
                "provider_team_id": 100,
                "provider_team_name": "France",
                "selected": True,
                "match_score": 7.5,
                "score_margin": 3.2,
                "search_query": "France",
                "selection_status": "auto_selected",
                "selection_reason": "score=7.50; margin=3.20",
            }
        ]
    )

    updated = apply_provider_team_match_suggestions(
        registry,
        suggestions,
        provider="sportmonks",
    )

    assert updated.loc[0, "sportmonks_team_id"] == 100
    assert updated.loc[0, "sportmonks_team_name"] == "France"
    assert "sportmonks_auto_selected" in updated.loc[0, "notes"]
