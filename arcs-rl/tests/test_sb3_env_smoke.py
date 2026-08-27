"""Smoke test: Stable-Baselines3 accepts our env API."""

from __future__ import annotations

from pathlib import Path

import pytest

from arcs_rl.config import load_config, validate_config_keys
from arcs_rl.envs import ArcsMicroserviceEnv

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "arcs.default.yaml"


def test_sb3_check_env_dqn() -> None:
    pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.env_checker import check_env

    data = load_config(DEFAULT_CONFIG)
    validate_config_keys(data)
    env = ArcsMicroserviceEnv(data, algorithm="dqn")
    check_env(env, warn=True)
