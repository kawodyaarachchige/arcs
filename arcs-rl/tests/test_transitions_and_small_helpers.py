"""Small helpers: replay row conversion, device pick, metrics callback, training metrics wiring."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from arcs_rl.config import load_config, validate_config_keys
from arcs_rl.monitoring.training_metrics import (
    TrainingMetricsExporter,
    maybe_start_training_metrics,
)
from arcs_rl.training._device import resolve_training_device
from arcs_rl.training.metrics_callback import TrainingMetricsCallback
from arcs_rl.training.transitions import row_dict_to_numpy

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "arcs.default.yaml"


def test_row_dict_to_numpy_shapes() -> None:
    """Replay import uses fixed shapes so storage stays compatible with training."""
    row = {
        "obs": [0.1] * 12,
        "next_obs": [0.2] * 12,
        "action": [3],
        "reward": 1.5,
        "done": False,
    }
    obs, next_obs, action, reward, done = row_dict_to_numpy(
        row,
        state_dim=12,
        action_shape=(1,),
        action_dtype=np.dtype("int32"),
    )
    assert obs.shape == (12,)
    assert next_obs.shape == (12,)
    assert action.shape == (1,)
    assert reward == 1.5
    assert done is False


def test_training_metrics_callback_records_reward() -> None:
    exporter = MagicMock()
    cb = TrainingMetricsCallback(exporter, algorithm="DQN")
    cb.locals = {"rewards": [0.5, 1.5]}
    assert cb._on_step() is True
    exporter.record_env_step.assert_called_once()

    cb2 = TrainingMetricsCallback(exporter, algorithm="DQN")
    cb2.locals = {}
    assert cb2._on_step() is True


def test_resolve_training_device_explicit() -> None:
    assert resolve_training_device("cpu") == "cpu"
    assert resolve_training_device("cuda") == "cuda"


def test_resolve_training_device_auto_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    fake = MagicMock()
    fake.cuda.is_available.return_value = False
    monkeypatch.setitem(sys.modules, "torch", fake)
    assert resolve_training_device("auto") == "cpu"


def test_resolve_training_device_auto_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    orig = builtins.__import__

    def selective_import(name: str, *args: object, **kwargs: object) -> object:
        # Only block the top-level torch import used by resolve_training_device.
        if name == "torch":
            raise ImportError("blocked for test")
        return orig(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", selective_import)
    assert resolve_training_device("auto") == "cpu"


def test_maybe_start_training_metrics_off() -> None:
    data = load_config(DEFAULT_CONFIG)
    validate_config_keys(data)
    assert data["prometheus"]["enabled"] is False
    assert maybe_start_training_metrics(data) is None


def test_maybe_start_training_metrics_on(monkeypatch: pytest.MonkeyPatch) -> None:
    import arcs_rl.monitoring.training_metrics as tm

    calls: list[tuple[int, str]] = []

    def fake_start(port: int, addr: str = "0.0.0.0", registry: object | None = None) -> None:
        calls.append((port, addr))

    monkeypatch.setattr(tm, "start_http_server", fake_start)
    data = load_config(DEFAULT_CONFIG)
    data["prometheus"]["enabled"] = True
    validate_config_keys(data)
    out = maybe_start_training_metrics(data)
    assert out is not None
    assert calls


def test_training_metrics_exporter_sync_and_steps(tmp_path: Path) -> None:
    from prometheus_client import CollectorRegistry

    data = load_config(DEFAULT_CONFIG)
    data["replay_buffer"]["path"] = str(tmp_path / "rb")
    validate_config_keys(data)
    reg = CollectorRegistry()
    ex = TrainingMetricsExporter(data, registry=reg)
    storage = MagicMock()
    storage.total_transitions = 10
    storage.pending_count = 2
    ex.sync_replay_from_storage(storage)
    ex.on_replay_flush()
    ex.on_replay_shard_removed()
    ex.record_env_step(algorithm="dqn", reward=1.0)
    ex.record_env_step(algorithm="dqn", reward=-1.0)


def test_training_metrics_start_http_server(monkeypatch: pytest.MonkeyPatch) -> None:
    from prometheus_client import CollectorRegistry

    import arcs_rl.monitoring.training_metrics as tm

    monkeypatch.setattr(tm, "start_http_server", lambda *a, **k: None)
    data = load_config(DEFAULT_CONFIG)
    data["prometheus"]["enabled"] = True
    validate_config_keys(data)
    reg = CollectorRegistry()
    ex = TrainingMetricsExporter(data, registry=reg)
    ex.start_http_server(data)
