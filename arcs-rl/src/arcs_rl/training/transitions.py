"""
Helpers to turn plain Python dicts into numpy rows for the replay store.

Useful later when you import rows from HTTP logs or tracing systems: keep the same field names
here so training code stays stable.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def row_dict_to_numpy(
    row: dict[str, Any],
    *,
    state_dim: int,
    action_shape: tuple[int, ...],
    action_dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]:
    """
    Convert one transition dict into the tuple `ReplayStorage.append` expects.

    Keys: obs, next_obs, action (same shape as training), reward, done.
    """
    obs = np.asarray(row["obs"], dtype=np.float32).reshape(state_dim)
    next_obs = np.asarray(row["next_obs"], dtype=np.float32).reshape(state_dim)
    action = np.asarray(row["action"], dtype=action_dtype).reshape(action_shape)
    reward = float(row["reward"])
    done = bool(row["done"])
    return obs, next_obs, action, reward, done
