"""Fast integration tests for offline / online training (tiny budgets)."""

from __future__ import annotations

from pathlib import Path

import pytest

from arcs_rl.config import load_config, validate_config_keys
from arcs_rl.export.torch_export import export_torchscript
from arcs_rl.training.offline import run_offline_training
from arcs_rl.training.online import run_online_training

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "arcs.default.yaml"


@pytest.fixture
def tiny_dqn_config(tmp_path: Path) -> dict:
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
    return data


def test_offline_dqn_smoke(tiny_dqn_config: dict, tmp_path: Path) -> None:
    out = run_offline_training(tiny_dqn_config, run_name="sm")
    assert out.is_file()
    ts = tmp_path / "m.ts"
    export_torchscript(out, ts, algorithm="dqn")
    assert ts.is_file()
    assert ts.with_suffix(".json").is_file()


def test_online_dqn_smoke(tmp_path: Path) -> None:
    pytest.importorskip("stable_baselines3")
    data = load_config(DEFAULT_CONFIG)
    data["replay_buffer"]["path"] = str(tmp_path / "rb")
    data["training"]["checkpoint_dir"] = str(tmp_path / "ck")
    data["training"]["online"]["total_timesteps"] = 32
    data["training"]["dqn"]["buffer_size"] = 128
    validate_config_keys(data)
    out = run_online_training(data, run_name="dq")
    assert out.is_file()


def test_offline_ppo_rejected(tmp_path: Path) -> None:
    pytest.importorskip("stable_baselines3")
    data = load_config(DEFAULT_CONFIG)
    data["action"]["algorithm"] = "ppo"
    data["replay_buffer"]["path"] = str(tmp_path / "rb")
    validate_config_keys(data)
    with pytest.raises(ValueError, match="DQN"):
        run_offline_training(data, run_name="x")


def test_online_ppo_smoke(tmp_path: Path) -> None:
    pytest.importorskip("stable_baselines3")
    data = load_config(DEFAULT_CONFIG)
    data["action"]["algorithm"] = "ppo"
    data["replay_buffer"]["path"] = str(tmp_path / "rb")
    data["training"]["checkpoint_dir"] = str(tmp_path / "ck")
    data["training"]["online"]["total_timesteps"] = 64
    data["training"]["ppo"]["n_steps"] = 32
    data["training"]["ppo"]["batch_size"] = 16
    validate_config_keys(data)
    out = run_online_training(data, run_name="pp")
    assert out.is_file()
