"""World Cup 2026 tournament simulation helpers."""

from .played_results import load_played_group_results
from .simulation import (
    build_group_stage_schedule,
    load_round_of_32_lookup,
    rank_group_standings,
    rank_third_place_teams,
    resolve_round_of_32_matchups,
    sample_scoreline_from_matrix,
    simulate_world_cup_2026,
)

__all__ = [
    "build_group_stage_schedule",
    "load_played_group_results",
    "load_round_of_32_lookup",
    "rank_group_standings",
    "rank_third_place_teams",
    "resolve_round_of_32_matchups",
    "sample_scoreline_from_matrix",
    "simulate_world_cup_2026",
]
