"""Tests for the default reward function."""

from __future__ import annotations

from pathlib import Path

from arcs_rl.config import load_config
from arcs_rl.rewards.default import compute_reward

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "arcs.default.yaml"


def test_compute_reward_numeric_example() -> None:
    data = load_config(DEFAULT_CONFIG)
    reward_cfg = data["reward"]
    total, parts = compute_reward(
        success_rate=1.0,
        avg_latency_ms=1000.0,
        retry_overhead=0.5,
        overload_penalty=0.2,
        cascade_active=False,
        reward_cfg=reward_cfg,
    )
    w = reward_cfg["weights"]
    expected = (
        w["success_rate"] * 1.0
        - w["latency"] * 1.0
        - w["retry_overhead"] * 0.5
        - w["overload_penalty"] * 0.2
    )
    assert abs(total - expected) < 1e-9
    assert parts["term_success"] == w["success_rate"] * 1.0


def test_cascade_subtracts_scale() -> None:
    data = load_config(DEFAULT_CONFIG)
    reward_cfg = data["reward"]
    base, _ = compute_reward(
        success_rate=1.0,
        avg_latency_ms=0.0,
        retry_overhead=0.0,
        overload_penalty=0.0,
        cascade_active=False,
        reward_cfg=reward_cfg,
    )
    with_cascade, parts = compute_reward(
        success_rate=1.0,
        avg_latency_ms=0.0,
        retry_overhead=0.0,
        overload_penalty=0.0,
        cascade_active=True,
        reward_cfg=reward_cfg,
    )
    assert with_cascade == base - float(reward_cfg["cascade_penalty_scale"])
    assert parts["term_cascade"] == float(reward_cfg["cascade_penalty_scale"])
