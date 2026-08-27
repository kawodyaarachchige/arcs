"""Exercise CLI entry points so typos in argparse wiring fail in tests, not in production."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from arcs_rl.cli import main_benchmark, main_export, main_train
from arcs_rl.config import load_config, validate_config_keys
from arcs_rl.training.online import run_online_training

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "arcs.default.yaml"


def test_main_train_offline_smoke(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    pytest.importorskip("stable_baselines3")
    data = load_config(DEFAULT_CONFIG)
    data["replay_buffer"]["path"] = str(tmp_path / "rb")
    data["replay_buffer"]["min_transitions"] = 32
    data["replay_buffer"]["transitions_per_shard"] = 16
    data["replay_buffer"]["max_shards"] = 20
    data["training"]["offline"]["gradient_steps"] = 2
    data["training"]["offline"]["batch_size"] = 8
    data["training"]["offline"]["fill_rollout_max_steps"] = 100
    data["training"]["checkpoint_dir"] = str(tmp_path / "ck")
    data["training"]["dqn"]["buffer_size"] = 64
    validate_config_keys(data)
    cfg_path = tmp_path / "cfg.yaml"
    import yaml

    cfg_path.write_text(yaml.safe_dump(data), encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["arcs-train", "offline", "-c", str(cfg_path), "-n", "cli"])
    main_train()
    out = capsys.readouterr().out
    assert "Saved model:" in out


def test_main_train_online_smoke(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    pytest.importorskip("stable_baselines3")
    data = load_config(DEFAULT_CONFIG)
    data["replay_buffer"]["path"] = str(tmp_path / "rb")
    data["training"]["checkpoint_dir"] = str(tmp_path / "ck2")
    data["training"]["online"]["total_timesteps"] = 32
    data["training"]["dqn"]["buffer_size"] = 128
    validate_config_keys(data)
    cfg_path = tmp_path / "cfg2.yaml"
    import yaml

    cfg_path.write_text(yaml.safe_dump(data), encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["arcs-train", "online", "-c", str(cfg_path)])
    main_train()
    assert "Saved model:" in capsys.readouterr().out


def test_main_export_torch_smoke(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    pytest.importorskip("stable_baselines3")
    data = load_config(DEFAULT_CONFIG)
    data["replay_buffer"]["path"] = str(tmp_path / "rb")
    data["training"]["checkpoint_dir"] = str(tmp_path / "ck")
    data["training"]["online"]["total_timesteps"] = 32
    data["training"]["dqn"]["buffer_size"] = 128
    validate_config_keys(data)
    z = run_online_training(data, run_name="ex")
    out_ts = tmp_path / "out.ts"
    monkeypatch.setattr(
        sys,
        "argv",
        ["arcs-export", "torch", str(z), str(out_ts), "-a", "dqn"],
    )
    main_export()
    assert "Wrote" in capsys.readouterr().out
    assert out_ts.is_file()


def test_main_benchmark_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, object] = {}

    def fake_run(*a: object, **k: object) -> None:
        called["yes"] = True

    import arcs_rl.cli as cli_mod

    monkeypatch.setattr(cli_mod.runpy, "run_module", fake_run)
    bench = REPO_ROOT / "benchmarks/config/default.yaml"
    monkeypatch.setattr(sys, "argv", ["arcs-benchmark", "-c", str(bench), "--arms", "static"])
    main_benchmark()
    assert called.get("yes") is True


def test_main_benchmark_default_config_path(monkeypatch: pytest.MonkeyPatch) -> None:
    import arcs_rl.cli as cli_mod

    monkeypatch.setattr(cli_mod.runpy, "run_module", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", ["arcs-benchmark", "--arms", "static"])
    main_benchmark()
