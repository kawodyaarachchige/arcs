"""Failure paths for saved-model evaluation (missing file, unknown algorithm)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from arcs_rl.config import load_config, validate_config_keys
from arcs_rl.evaluation.sb3_eval import evaluate_saved_model

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "arcs.default.yaml"


def test_evaluate_saved_model_missing_file(tmp_path: Path) -> None:
    pytest.importorskip("stable_baselines3")
    data = load_config(DEFAULT_CONFIG)
    validate_config_keys(data)
    missing = tmp_path / "nope.zip"
    with pytest.raises(FileNotFoundError, match="Model file not found"):
        evaluate_saved_model(data, missing, n_eval_episodes=1)


def test_evaluate_saved_model_bad_algorithm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("stable_baselines3")
    from arcs_rl.evaluation import sb3_eval as se

    # Avoid building a real env: invalid algorithm would fail earlier in make_vec_env.
    vec = MagicMock()
    monkeypatch.setattr(se, "make_vec_env", lambda *a, **k: vec)

    data = load_config(DEFAULT_CONFIG)
    validate_config_keys(data)
    data["action"]["algorithm"] = "not_real"
    p = tmp_path / "fake.zip"
    p.write_bytes(b"x")
    with pytest.raises(ValueError, match="Unsupported action.algorithm"):
        evaluate_saved_model(data, p, n_eval_episodes=1)
    vec.close.assert_called_once()
