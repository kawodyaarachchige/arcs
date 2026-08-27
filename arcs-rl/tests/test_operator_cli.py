"""Tests for operator-facing CLI commands (validate config, eval)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from arcs_rl.cli import main_eval, main_validate_config
from arcs_rl.config import load_config, validate_config_keys
from arcs_rl.training.online import run_online_training

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "arcs.default.yaml"


def test_validate_config_ok(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["arcs-validate-config", str(DEFAULT_CONFIG)])
    main_validate_config()
    out = capsys.readouterr().out
    assert "OK" in out


def test_validate_config_missing_keys(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("version: 1\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["arcs-validate-config", str(bad)])
    with pytest.raises(SystemExit) as ei:
        main_validate_config()
    assert ei.value.code == 1


def test_eval_smoke(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    pytest.importorskip("stable_baselines3")
    data = load_config(DEFAULT_CONFIG)
    data["replay_buffer"]["path"] = str(tmp_path / "rb")
    data["training"]["checkpoint_dir"] = str(tmp_path / "ck")
    data["training"]["online"]["total_timesteps"] = 64
    data["training"]["dqn"]["buffer_size"] = 128
    validate_config_keys(data)
    out_zip = run_online_training(data, run_name="ev")
    assert out_zip.is_file()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "arcs-eval",
            "-c",
            str(DEFAULT_CONFIG),
            "-m",
            str(out_zip),
            "-n",
            "2",
        ],
    )
    main_eval()
    text = capsys.readouterr().out
    assert "Mean return:" in text


def test_eval_verbose_prints_paths(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    pytest.importorskip("stable_baselines3")
    data = load_config(DEFAULT_CONFIG)
    data["replay_buffer"]["path"] = str(tmp_path / "rb")
    data["training"]["checkpoint_dir"] = str(tmp_path / "ck")
    data["training"]["online"]["total_timesteps"] = 64
    data["training"]["dqn"]["buffer_size"] = 128
    validate_config_keys(data)
    out_zip = run_online_training(data, run_name="evv")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "arcs-eval",
            "-v",
            "-c",
            str(DEFAULT_CONFIG),
            "-m",
            str(out_zip),
            "-n",
            "1",
        ],
    )
    main_eval()
    text = capsys.readouterr().out
    assert "Config:" in text
    assert "Mean return:" in text


def test_benchmark_repo_root_found() -> None:
    from arcs_rl.cli import _find_benchmark_repo_root

    root = _find_benchmark_repo_root()
    assert root is not None
    assert (root / "benchmarks" / "run_experiment.py").is_file()


def test_main_benchmark_missing_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    from arcs_rl import cli as cli_mod

    monkeypatch.setattr(cli_mod, "_find_benchmark_repo_root", lambda: None)
    monkeypatch.setattr(sys, "argv", ["arcs-benchmark"])
    with pytest.raises(SystemExit) as ei:
        cli_mod.main_benchmark()
    assert ei.value.code == 2
