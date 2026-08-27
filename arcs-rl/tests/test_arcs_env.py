"""Tests for the Gymnasium microservice environment."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from arcs_rl.config import load_config, validate_config_keys
from arcs_rl.envs import ArcsMicroserviceEnv
from arcs_rl.envs.discrete_actions import discrete_layout_from_config, encode_discrete_action

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "arcs.default.yaml"


def test_env_dqn_deterministic_with_seed() -> None:
    data = load_config(DEFAULT_CONFIG)
    validate_config_keys(data)

    def run() -> tuple[list[float], list[float]]:
        env = ArcsMicroserviceEnv(data, algorithm="dqn")
        _obs0, _ = env.reset(seed=123)
        rewards: list[float] = []
        obs_hashes: list[float] = []
        layout = discrete_layout_from_config(data["policy"], data["action"]["dqn"])
        action = encode_discrete_action(1, 1.0, 2000.0, layout)
        for _ in range(5):
            obs, r, term, trunc, info = env.step(action)
            rewards.append(float(r))
            obs_hashes.append(float(np.sum(obs)))
            assert not term and not trunc
            assert "reward_breakdown" in info
        return rewards, obs_hashes

    r1, o1 = run()
    r2, o2 = run()
    assert r1 == r2
    assert o1 == o2


def test_env_ppo_step_runs() -> None:
    data = load_config(DEFAULT_CONFIG)
    validate_config_keys(data)
    env = ArcsMicroserviceEnv(data, algorithm="ppo")
    obs, _ = env.reset(seed=7)
    assert obs.shape == (12,)
    a = env.action_space.sample()
    obs2, r, term, trunc, info = env.step(a)
    assert obs2.shape == (12,)
    assert isinstance(r, float)
    assert not term and not trunc
    assert "timeout_ms" in info
