"""Torch inference loader and forward pass (uses the same tiny offline run as other smoke tests)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from arcs_rl.config import load_config, validate_config_keys
from arcs_rl.envs.arcs_env import ArcsMicroserviceEnv
from arcs_rl.export.torch_export import export_torchscript
from arcs_rl.inference.runtime import load_torch_inference_or_none
from arcs_rl.training.offline import run_offline_training

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "arcs.default.yaml"


def test_inference_disabled_returns_none() -> None:
    data = load_config(DEFAULT_CONFIG)
    validate_config_keys(data)
    assert data["serving"]["inference_enabled"] is False
    assert load_torch_inference_or_none(data) is None


def test_inference_forward_from_torchscript(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("stable_baselines3")

    data = load_config(DEFAULT_CONFIG)
    data["replay_buffer"]["path"] = str(tmp_path / "rb")
    data["replay_buffer"]["min_transitions"] = 32
    data["replay_buffer"]["transitions_per_shard"] = 16
    data["replay_buffer"]["max_shards"] = 20
    data["training"]["offline"]["gradient_steps"] = 3
    data["training"]["offline"]["batch_size"] = 8
    data["training"]["offline"]["fill_rollout_max_steps"] = 200
    data["training"]["checkpoint_dir"] = str(tmp_path / "ck")
    data["training"]["dqn"]["buffer_size"] = 64
    validate_config_keys(data)

    sb3_zip = run_offline_training(data, run_name="inf")
    ts_path = tmp_path / "m.ts"
    export_torchscript(sb3_zip, ts_path, algorithm="dqn")

    data_inf = load_config(DEFAULT_CONFIG)
    data_inf["serving"]["inference_enabled"] = True
    data_inf["serving"]["torchscript_path"] = str(ts_path)
    data_inf["serving"]["fail_if_model_missing"] = True
    validate_config_keys(data_inf)

    env = ArcsMicroserviceEnv(data_inf)
    obs, _ = env.reset()
    rt = load_torch_inference_or_none(data_inf)
    assert rt is not None
    act = rt.suggested_action(np.asarray(obs, dtype=np.float32))
    assert 0 <= act.retry <= data_inf["policy"]["retry"]["max"]
