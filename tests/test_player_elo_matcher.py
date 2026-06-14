import importlib.util
from pathlib import Path

import pandas as pd

# The matcher lives in a script (scripts/), load it directly.
_spec = importlib.util.spec_from_file_location(
    "build_world_cup_player_elo_strength",
    Path(__file__).resolve().parents[1] / "scripts" / "build_world_cup_player_elo_strength.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def _index():
    player_elo = pd.DataFrame(
        [
            {"player_name": "O. Dembélé", "elo": 2069.0, "current_team": "Paris Saint Germain"},
            {"player_name": "M. Dembélé", "elo": 1698.5, "current_team": "Al-Ettifaq"},
            {"player_name": "M. Maignan", "elo": 1969.9, "current_team": "AC Milan"},
        ]
    )
    return _mod._build_player_elo_index(player_elo)


def test_matches_unique_surname_initial() -> None:
    elo, method = _mod._match_player_elo("Mike Maignan", "Milan", _index())
    assert elo == 1969.9
    assert method in {"name", "name+club"}


def test_disambiguates_homonyms_by_club() -> None:
    # Two "Dembélé" with initial O./M.; full name "Ousmane Dembélé" -> O. at PSG.
    elo, method = _mod._match_player_elo("Ousmane Dembélé", "Paris Saint-Germain", _index())
    assert elo == 2069.0


def test_unmatched_returns_none() -> None:
    elo, method = _mod._match_player_elo("Nonexistent Player", "Nowhere FC", _index())
    assert elo is None
    assert method == "unmatched"


def test_surname_and_initial_normalization_handles_accents() -> None:
    assert _mod._surname_key("Ousmane Dembélé") == "dembele"
    assert _mod._first_initial("Ousmane Dembélé") == "o"
