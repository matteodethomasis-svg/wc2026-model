from __future__ import annotations

import os

from wc2026_model.data import football_data_api
from wc2026_model.data.env import load_project_env


def test_load_project_env_reads_key_value_pairs(tmp_path, monkeypatch) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "# comment",
                "API_FOOTBALL_API_KEY=test-key",
                'API_FOOTBALL_HOST="custom.host"',
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("API_FOOTBALL_API_KEY", raising=False)
    monkeypatch.delenv("API_FOOTBALL_HOST", raising=False)

    loaded_path = load_project_env(env_path, override=True)

    assert loaded_path == env_path.resolve()
    assert os.environ["API_FOOTBALL_API_KEY"] == "test-key"
    assert os.environ["API_FOOTBALL_HOST"] == "custom.host"


def test_get_football_data_api_token_loads_project_env(monkeypatch) -> None:
    monkeypatch.delenv("FOOTBALL_DATA_API_TOKEN", raising=False)

    def fake_load_project_env():
        monkeypatch.setenv("FOOTBALL_DATA_API_TOKEN", "loaded-football-data-token")

    monkeypatch.setattr(football_data_api, "load_project_env", fake_load_project_env)

    assert football_data_api.get_football_data_api_token() == "loaded-football-data-token"
