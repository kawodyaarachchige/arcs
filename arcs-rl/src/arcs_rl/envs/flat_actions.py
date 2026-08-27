"""
Stable-Baselines3's DQN only accepts a single Discrete action, not MultiDiscrete.

Pack the three small integers (retry bin, backoff bin, timeout bin) into one flat index and
unpack again when stepping the real environment. Same math as row-major numbering in a 3-D grid.
"""

from __future__ import annotations

import numpy as np
from gymnasium.spaces import MultiDiscrete


def flat_dim(nvec: np.ndarray | tuple[int, ...]) -> int:
    """How many distinct flat actions exist for this MultiDiscrete shape."""
    v = np.asarray(nvec, dtype=np.int64)
    return int(np.prod(v))


def multi_to_flat(nvec: np.ndarray | tuple[int, ...], components: np.ndarray) -> int:
    """Turn one MultiDiscrete sample into a single integer in [0, flat_dim)."""
    v = np.asarray(nvec, dtype=np.int64)
    c = np.asarray(components, dtype=np.int64).reshape(-1)
    if c.shape[0] != v.shape[0]:
        msg = "component count must match nvec length"
        raise ValueError(msg)
    idx = 0
    m = 1
    for i in range(len(v) - 1, -1, -1):
        if not (0 <= c[i] < v[i]):
            msg = f"component {i}={c[i]} out of range for nvec[i]={v[i]}"
            raise ValueError(msg)
        idx += int(c[i]) * m
        m *= int(v[i])
    return idx


def flat_to_multi(nvec: np.ndarray | tuple[int, ...], flat: int) -> np.ndarray:
    """Turn a flat index back into the MultiDiscrete vector the inner env expects."""
    v = np.asarray(nvec, dtype=np.int64)
    k = int(flat)
    out = np.zeros(len(v), dtype=np.int64)
    for i in range(len(v) - 1, -1, -1):
        out[i] = k % int(v[i])
        k //= int(v[i])
    return out.astype(np.int32, copy=False)


def nvec_from_space(action_space: MultiDiscrete) -> np.ndarray:
    """Read branch sizes from the Gymnasium space."""
    return np.asarray(action_space.nvec, dtype=np.int64)
