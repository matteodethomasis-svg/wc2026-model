"""Feature engineering helpers."""

from .elo import (
    EloConfig,
    augment_with_pre_match_elo,
    build_latest_elo_ratings,
    goal_difference_multiplier,
    infer_match_importance,
)
from .form import RecentFormConfig, augment_with_pre_match_form_features
from .international_context import (
    CONFEDERATIONS,
    attach_confederation_features,
    attach_fixture_h2h_features,
    augment_with_pre_match_h2h_features,
    get_team_confederation,
)
from .squad_strength import (
    aggregate_team_squad_strength,
    attach_team_strength_ratings,
    build_squad_player_club_elo_frame,
    normalize_club_name,
)
from .world_cup_xg import WorldCupXGConfig, augment_with_pre_match_xg_features
from .world_cup_xg import attach_latest_team_xg_features, build_latest_team_xg_snapshot

__all__ = [
    "EloConfig",
    "RecentFormConfig",
    "WorldCupXGConfig",
    "CONFEDERATIONS",
    "aggregate_team_squad_strength",
    "attach_confederation_features",
    "attach_fixture_h2h_features",
    "augment_with_pre_match_elo",
    "augment_with_pre_match_form_features",
    "augment_with_pre_match_h2h_features",
    "augment_with_pre_match_xg_features",
    "attach_latest_team_xg_features",
    "attach_team_strength_ratings",
    "build_latest_team_xg_snapshot",
    "build_latest_elo_ratings",
    "build_squad_player_club_elo_frame",
    "get_team_confederation",
    "goal_difference_multiplier",
    "infer_match_importance",
    "normalize_club_name",
]
