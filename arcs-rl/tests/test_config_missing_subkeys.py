"""Missing nested keys should raise clear errors (helps catch typos in hand-edited YAML)."""

from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from arcs_rl.config import load_config, validate_config_keys

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "arcs.default.yaml"


def _base() -> dict[str, Any]:
    return copy.deepcopy(load_config(DEFAULT_CONFIG))


@pytest.mark.parametrize(
    ("mutator", "snippet"),
    [
        (lambda d: d["prometheus"].pop("enabled", None), "prometheus missing keys"),
        (lambda d: d["serving"].pop("grpc_port", None), "serving missing keys"),
        (lambda d: d["replay_buffer"].pop("path", None), "replay_buffer missing keys"),
        (lambda d: d["training"].pop("device", None), "training missing keys"),
        (lambda d: d["training"]["dqn"].pop("buffer_size", None), "training.dqn missing keys"),
        (lambda d: d["training"]["ppo"].pop("batch_size", None), "training.ppo missing keys"),
        (
            lambda d: d["training"]["online"].pop("total_timesteps", None),
            "training.online missing keys",
        ),
        (
            lambda d: d["training"]["offline"].pop("gradient_steps", None),
            "training.offline missing keys",
        ),
        (lambda d: d["observation"].pop("feature_order", None), "observation missing keys"),
        (lambda d: d["reward"].pop("weights", None), "reward missing keys"),
        (lambda d: d["reward"]["weights"].pop("latency", None), "reward.weights missing keys"),
        (lambda d: d["action"].pop("dqn", None), "action missing keys"),
        (lambda d: d["safeguards"].pop("max_retries", None), "safeguards missing keys"),
    ],
)
def test_validate_config_missing_nested_keys(
    mutator: Callable[[dict[str, Any]], None],
    snippet: str,
) -> None:
    data = _base()
    mutator(data)
    with pytest.raises(ValueError, match=snippet):
        validate_config_keys(data)
