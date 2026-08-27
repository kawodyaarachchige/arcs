"""Tests for DQN (MultiDiscrete) and PPO (Box) action helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from arcs_rl.config import load_config
from arcs_rl.envs.continuous_actions import decode_continuous_action, make_ppo_box_space
from arcs_rl.envs.discrete_actions import (
    decode_discrete_action,
    discrete_layout_from_config,
    encode_discrete_action,
    make_multi_discrete_space,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "arcs.default.yaml"


def test_discrete_round_trip() -> None:
    data = load_config(DEFAULT_CONFIG)
    layout = discrete_layout_from_config(data["policy"], data["action"]["dqn"])
    space = make_multi_discrete_space(layout)
    a = encode_discrete_action(2, 1.5, 500.0, layout)
    assert space.contains(a)
    r, b, t = decode_discrete_action(a, layout)
    assert r == 2
    assert b == 1.5
    assert t == 500.0


def test_discrete_clamps_bad_indices() -> None:
    data = load_config(DEFAULT_CONFIG)
    layout = discrete_layout_from_config(data["policy"], data["action"]["dqn"])
    bad = np.array([999, -1, 0], dtype=np.int64)
    r, b, t = decode_discrete_action(bad, layout)
    assert layout.retry_min <= r <= layout.retry_max
    assert b in layout.backoff_multipliers
    assert t in layout.timeout_ms_bins


def test_ppo_decode_clips() -> None:
    data = load_config(DEFAULT_CONFIG)
    policy = data["policy"]
    space = make_ppo_box_space(policy)
    # Retry fraction; backoff below min (clips); timeout below min (clips).
    v = np.array([-1.0, 0.1, 50.0], dtype=np.float32)
    r, b, t = decode_continuous_action(v, policy)
    assert policy["retry"]["min"] <= r <= policy["retry"]["max"]
    assert b == float(policy["backoff"]["min_multiplier"])
    assert t == float(policy["timeout_ms"]["min"])
    sample = space.sample()
    assert space.contains(sample)
