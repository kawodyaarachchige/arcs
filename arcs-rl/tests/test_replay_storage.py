"""Tests for on-disk numpy replay shards and index rotation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from arcs_rl.config import load_config, validate_config_keys
from arcs_rl.envs import ArcsMicroserviceEnv
from arcs_rl.envs.wrappers import FlattenMultiDiscreteActions
from arcs_rl.replay_storage import open_storage

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "arcs.default.yaml"


def test_append_flush_and_load_roundtrip(tmp_path: Path) -> None:
    data = load_config(DEFAULT_CONFIG)
    data["replay_buffer"]["path"] = str(tmp_path / "rb")
    data["replay_buffer"]["transitions_per_shard"] = 4
    data["replay_buffer"]["max_shards"] = 10
    validate_config_keys(data)

    env = ArcsMicroserviceEnv(data, algorithm="dqn")
    wenv = FlattenMultiDiscreteActions(env)
    store = open_storage(tmp_path / "rb", data, env=env)
    obs, _ = wenv.reset(seed=0)
    for i in range(10):
        a = int(wenv.action_space.sample())
        obs2, _, _, _, _ = wenv.step(a)
        store.append(obs, obs2, np.array([a], dtype=np.int32), 0.1 * i, False)
        obs = obs2
    store.flush()

    arrs = store.load_arrays()
    assert arrs["obs"].shape == (10, 12)
    assert arrs["actions"].shape == (10, 1)
    assert arrs["rewards"].shape == (10,)
    assert store.total_transitions == 10


def test_rotation_drops_oldest_shard(tmp_path: Path) -> None:
    data = load_config(DEFAULT_CONFIG)
    data["replay_buffer"]["path"] = str(tmp_path / "rb")
    data["replay_buffer"]["transitions_per_shard"] = 3
    data["replay_buffer"]["max_shards"] = 2
    validate_config_keys(data)

    env = ArcsMicroserviceEnv(data, algorithm="dqn")
    wenv = FlattenMultiDiscreteActions(env)
    store = open_storage(tmp_path / "rb", data, env=env)
    obs, _ = wenv.reset(seed=1)
    # Nine transitions in shards of three would need three files; cap is two, so the oldest shard
    # is deleted and only the last six rows remain on disk.
    for _ in range(9):
        a = int(wenv.action_space.sample())
        obs2, _, _, _, _ = wenv.step(a)
        store.append(obs, obs2, np.array([a], dtype=np.int32), 0.0, False)
        obs = obs2
    store.flush()

    shard_files = sorted((tmp_path / "rb").glob("shard_*.npz"))
    assert len(shard_files) <= 2
    arrs = store.load_arrays()
    assert arrs["obs"].shape[0] == 6


def test_rejects_algorithm_mismatch(tmp_path: Path) -> None:
    data = load_config(DEFAULT_CONFIG)
    data["replay_buffer"]["path"] = str(tmp_path / "rb")
    validate_config_keys(data)

    env = ArcsMicroserviceEnv(data, algorithm="dqn")
    open_storage(tmp_path / "rb", data, env=env)

    data2 = load_config(DEFAULT_CONFIG)
    data2["replay_buffer"]["path"] = str(tmp_path / "rb")
    data2["action"]["algorithm"] = "ppo"
    validate_config_keys(data2)
    env2 = ArcsMicroserviceEnv(data2, algorithm="ppo")
    with pytest.raises(ValueError, match="algorithm"):
        open_storage(tmp_path / "rb", data2, env=env2)
