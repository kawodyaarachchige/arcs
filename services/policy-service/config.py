"""Load the same YAML file the RL trainer uses so one place owns the numbers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from arcs_rl.config import load_config, validate_config_keys


def default_config_path() -> Path:
    """Prefer `ARCS_CONFIG`, else repo `configs/arcs.default.yaml` next to this service."""
    env = os.environ.get("ARCS_CONFIG")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    # services/policy-service/config.py -> repo root is parents[2]
    repo_root = here.parents[2]
    return repo_root / "configs" / "arcs.default.yaml"


def load_arcs_root_config(path: Path | None = None) -> dict[str, Any]:
    """Parse YAML and validate required sections (fail fast on typos)."""
    p = default_config_path() if path is None else Path(path)
    data = load_config(p)
    validate_config_keys(data)
    return data
