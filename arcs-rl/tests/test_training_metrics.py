"""Tests for optional Prometheus export from replay storage and training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from prometheus_client import CollectorRegistry, generate_latest

from arcs_rl.config import load_config, validate_config_keys, validate_prometheus_config
from arcs_rl.envs import ArcsMicroserviceEnv
from arcs_rl.envs.wrappers import FlattenMultiDiscreteActions
from arcs_rl.monitoring.training_metrics import TrainingMetricsExporter
from arcs_rl.replay_storage import open_storage

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "arcs.default.yaml"


def test_validate_prometheus_config_accepts_default() -> None:
    data = load_config(DEFAULT_CONFIG)
    validate_prometheus_config(data["prometheus"])


def test_validate_prometheus_config_rejects_bad_port() -> None:
    data = load_config(DEFAULT_CONFIG)
    data["prometheus"]["scrape_port"] = 70000
    with pytest.raises(ValueError, match="65535"):
        validate_prometheus_config(data["prometheus"])


def test_replay_metrics_track_flushes_and_rotation(tmp_path: Path) -> None:
    data = load_config(DEFAULT_CONFIG)
    data["replay_buffer"]["path"] = str(tmp_path / "rb")
    data["replay_buffer"]["transitions_per_shard"] = 3
    data["replay_buffer"]["max_shards"] = 2
    validate_config_keys(data)

    reg = CollectorRegistry()
    exporter = TrainingMetricsExporter(data, registry=reg)

    env = ArcsMicroserviceEnv(data, algorithm="dqn")
    wenv = FlattenMultiDiscreteActions(env)
    store = open_storage(tmp_path / "rb", data, env=env, metrics_exporter=exporter)
    obs, _ = wenv.reset(seed=0)
    for _ in range(9):
        a = int(wenv.action_space.sample())
        obs2, _, _, _, _ = wenv.step(a)
        store.append(obs, obs2, np.array([a], dtype=np.int32), 0.0, False)
        obs = obs2
    store.flush()

    text = generate_latest(reg).decode("utf-8")
    assert "arcs_replay_shard_flushes_total 3.0" in text
    assert "arcs_replay_shards_removed_total 1.0" in text
