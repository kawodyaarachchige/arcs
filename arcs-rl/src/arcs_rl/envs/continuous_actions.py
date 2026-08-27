"""
PPO-friendly actions: a short vector of real numbers for retry shape, backoff, and timeout.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium.spaces import Box


def make_ppo_box_space(policy_cfg: dict[str, Any]) -> Box:
    """
    Build a Box aligned with `policy.backoff` and `policy.timeout_ms`.

    Dimension 0 is the retry fraction in [0, 1]. Dimensions 1–2 are backoff and timeout in
    their natural units (multiplier and milliseconds).
    """
    b_min = float(policy_cfg["backoff"]["min_multiplier"])
    b_max = float(policy_cfg["backoff"]["max_multiplier"])
    t_min = float(policy_cfg["timeout_ms"]["min"])
    t_max = float(policy_cfg["timeout_ms"]["max"])
    low = np.array([0.0, b_min, t_min], dtype=np.float32)
    high = np.array([1.0, b_max, t_max], dtype=np.float32)
    return Box(low=low, high=high, dtype=np.float32)


def decode_continuous_action(
    vec: np.ndarray,
    policy_cfg: dict[str, Any],
) -> tuple[int, float, float]:
    """
    Turn the agent’s Box sample into integer retry plus continuous backoff and timeout.

    Values are clipped to policy bounds so tiny numeric drift from the policy net stays safe.
    """
    r_min = int(policy_cfg["retry"]["min"])
    r_max = int(policy_cfg["retry"]["max"])
    b_min = float(policy_cfg["backoff"]["min_multiplier"])
    b_max = float(policy_cfg["backoff"]["max_multiplier"])
    t_min = float(policy_cfg["timeout_ms"]["min"])
    t_max = float(policy_cfg["timeout_ms"]["max"])

    v = np.asarray(vec, dtype=np.float64).reshape(-1)
    if v.size != 3:
        msg = f"PPO action must have length 3, got shape {vec.shape}"
        raise ValueError(msg)

    r_frac = float(np.clip(v[0], 0.0, 1.0))
    span = max(0, r_max - r_min)
    # Map 0 → min retries, 1 → max retries (inclusive).
    retry = r_min + int(round(r_frac * span)) if span > 0 else r_min
    retry = int(np.clip(retry, r_min, r_max))

    backoff = float(np.clip(v[1], b_min, b_max))
    timeout_ms = float(np.clip(v[2], t_min, t_max))
    return retry, backoff, timeout_ms
