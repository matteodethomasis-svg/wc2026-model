from __future__ import annotations

import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_ENV_PATH = _PROJECT_ROOT / ".env"
_LOADED_ENV_PATHS: set[Path] = set()


def load_project_env(
    env_path: str | Path | None = None,
    *,
    override: bool = False,
) -> Path | None:
    resolved_path = Path(env_path) if env_path is not None else _DEFAULT_ENV_PATH
    resolved_path = resolved_path.resolve()
    if not resolved_path.exists():
        return None
    if resolved_path in _LOADED_ENV_PATHS and not override:
        return resolved_path

    for raw_line in resolved_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env_key = key.strip()
        env_value = _strip_wrapping_quotes(value.strip())
        if env_key == "":
            continue
        if override or env_key not in os.environ:
            os.environ[env_key] = env_value

    _LOADED_ENV_PATHS.add(resolved_path)
    return resolved_path


def _strip_wrapping_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value
