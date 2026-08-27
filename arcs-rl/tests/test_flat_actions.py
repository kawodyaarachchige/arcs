"""Round-trip tests for flat MultiDiscrete encoding (DQN compatibility)."""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium.spaces import MultiDiscrete

from arcs_rl.envs.flat_actions import flat_dim, flat_to_multi, multi_to_flat


@pytest.mark.parametrize(
    "nvec",
    [
        np.array([6, 6, 7], dtype=np.int64),
        np.array([2, 3, 4], dtype=np.int64),
    ],
)
def test_flat_roundtrip(nvec: np.ndarray) -> None:
    md = MultiDiscrete(nvec)
    for _ in range(50):
        sample = md.sample()
        k = multi_to_flat(nvec, sample)
        assert 0 <= k < flat_dim(nvec)
        back = flat_to_multi(nvec, k)
        assert np.array_equal(back, sample)
