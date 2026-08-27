"""Edge cases for TorchScript export: metadata files, filename hints, and error messages."""

from __future__ import annotations

from pathlib import Path

import pytest

from arcs_rl.config import load_config, validate_config_keys
from arcs_rl.export.torch_export import (
    export_onnx_if_available,
    export_torchscript,
    write_model_sidecar,
)
from arcs_rl.training.online import run_online_training

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "arcs.default.yaml"


def test_write_model_sidecar_json(tmp_path: Path) -> None:
    p = tmp_path / "m.json"
    write_model_sidecar(
        p,
        algorithm="dqn",
        observation_dim=12,
        action_summary={"kind": "test"},
    )
    text = p.read_text(encoding="utf-8")
    assert '"algorithm": "dqn"' in text
    assert "12" in text


def test_export_torchscript_requires_algorithm_hint(tmp_path: Path) -> None:
    pytest.importorskip("stable_baselines3")
    z = tmp_path / "mystery.zip"
    z.write_bytes(b"not a real zip")
    with pytest.raises(ValueError, match="Pass algorithm"):
        export_torchscript(z, tmp_path / "o.ts")


def test_export_torchscript_infers_dqn_from_filename(tmp_path: Path) -> None:
    pytest.importorskip("stable_baselines3")
    pytest.importorskip("torch")
    data = load_config(DEFAULT_CONFIG)
    data["replay_buffer"]["path"] = str(tmp_path / "rb")
    data["training"]["checkpoint_dir"] = str(tmp_path / "ck")
    data["training"]["online"]["total_timesteps"] = 32
    data["training"]["dqn"]["buffer_size"] = 128
    validate_config_keys(data)
    z = run_online_training(data, run_name="nm")
    hinted = tmp_path / "run_dqn_final.zip"
    hinted.write_bytes(z.read_bytes())
    out = tmp_path / "tr.ts"
    export_torchscript(hinted, out)
    assert out.is_file()


def test_export_torchscript_unsupported_algorithm(tmp_path: Path) -> None:
    pytest.importorskip("stable_baselines3")
    data = load_config(DEFAULT_CONFIG)
    data["replay_buffer"]["path"] = str(tmp_path / "rb")
    data["training"]["checkpoint_dir"] = str(tmp_path / "ck")
    data["training"]["online"]["total_timesteps"] = 32
    data["training"]["dqn"]["buffer_size"] = 128
    validate_config_keys(data)
    z = run_online_training(data, run_name="zz")
    with pytest.raises(ValueError, match="Unsupported algorithm"):
        export_torchscript(z, tmp_path / "x.ts", algorithm="sac")


def test_export_torchscript_ppo_path(tmp_path: Path) -> None:
    pytest.importorskip("stable_baselines3")
    data = load_config(DEFAULT_CONFIG)
    data["action"]["algorithm"] = "ppo"
    data["replay_buffer"]["path"] = str(tmp_path / "rb")
    data["training"]["checkpoint_dir"] = str(tmp_path / "ck")
    data["training"]["online"]["total_timesteps"] = 64
    data["training"]["ppo"]["n_steps"] = 32
    data["training"]["ppo"]["batch_size"] = 16
    validate_config_keys(data)
    z = run_online_training(data, run_name="ppo")
    out = tmp_path / "ppo.ts"
    export_torchscript(z, out, algorithm="ppo")
    assert out.is_file()
    assert out.with_suffix(".json").is_file()


def test_export_onnx_if_available_returns_none_without_onnx(tmp_path: Path) -> None:
    pytest.importorskip("stable_baselines3")
    data = load_config(DEFAULT_CONFIG)
    data["replay_buffer"]["path"] = str(tmp_path / "rb")
    data["training"]["checkpoint_dir"] = str(tmp_path / "ck")
    data["training"]["online"]["total_timesteps"] = 32
    data["training"]["dqn"]["buffer_size"] = 128
    validate_config_keys(data)
    z = run_online_training(data, run_name="on")
    # When onnx is not installed, helper quietly skips ONNX export.
    assert export_onnx_if_available(z, tmp_path / "m.onnx", algorithm="dqn") is None
