"""
DQN-friendly actions: three small integers pick retry count, backoff strength, and timeout.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from gymnasium.spaces import MultiDiscrete


@dataclass(frozen=True)
class DiscreteActionLayout:
    """Fixed sizes and lookup tables for encoding and decoding."""

    retry_min: int
    retry_max: int
    backoff_multipliers: tuple[float, ...]
    timeout_ms_bins: tuple[float, ...]


def discrete_layout_from_config(
    policy_cfg: dict[str, Any],
    dqn_cfg: dict[str, Any],
) -> DiscreteActionLayout:
    """Build a layout from the `policy` and `action.dqn` sections of the YAML file."""
    r_min = int(policy_cfg["retry"]["min"])
    r_max = int(policy_cfg["retry"]["max"])
    bo = tuple(float(x) for x in dqn_cfg["backoff_multipliers"])
    to = tuple(float(x) for x in dqn_cfg["timeout_ms_bins"])
    return DiscreteActionLayout(
        retry_min=r_min,
        retry_max=r_max,
        backoff_multipliers=bo,
        timeout_ms_bins=to,
    )


def make_multi_discrete_space(layout: DiscreteActionLayout) -> MultiDiscrete:
    """Gymnasium space with one discrete slot per knob (retry, backoff bin, timeout bin)."""
    n_retry = layout.retry_max - layout.retry_min + 1
    nvec = np.array(
        [n_retry, len(layout.backoff_multipliers), len(layout.timeout_ms_bins)],
        dtype=np.int64,
    )
    return MultiDiscrete(nvec)


def decode_discrete_action(
    action: np.ndarray,
    layout: DiscreteActionLayout,
) -> tuple[int, float, float]:
    """
    Turn the agent’s three indices into real retry / backoff / timeout numbers.

    Indices outside the grid are clamped so bad inputs cannot crash training.
    """
    a = np.asarray(action, dtype=np.int64).reshape(-1)
    if a.size != 3:
        msg = f"DQN action must have length 3, got shape {action.shape}"
        raise ValueError(msg)
    n_r = layout.retry_max - layout.retry_min + 1
    ir = int(np.clip(a[0], 0, n_r - 1))
    ib = int(np.clip(a[1], 0, len(layout.backoff_multipliers) - 1))
    it = int(np.clip(a[2], 0, len(layout.timeout_ms_bins) - 1))
    retry = layout.retry_min + ir
    backoff = layout.backoff_multipliers[ib]
    timeout_ms = layout.timeout_ms_bins[it]
    return retry, backoff, timeout_ms


def encode_discrete_action(
    retry: int,
    backoff: float,
    timeout_ms: float,
    layout: DiscreteActionLayout,
) -> np.ndarray:
    """
    Pick the closest grid point to a concrete policy tuple (handy for tests and debugging).

    Retry must already be an integer in range; backoff and timeout snap to the nearest bin.
    """
    r = int(np.clip(retry, layout.retry_min, layout.retry_max))
    ir = r - layout.retry_min
    ib = int(np.argmin([abs(backoff - b) for b in layout.backoff_multipliers]))
    it = int(np.argmin([abs(timeout_ms - t) for t in layout.timeout_ms_bins]))
    return np.array([ir, ib, it], dtype=np.int64)
