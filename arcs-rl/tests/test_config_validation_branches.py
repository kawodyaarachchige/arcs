"""
Extra config validation tests: copy the default YAML, break one field, assert a clear error.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from arcs_rl.config import (
    load_config,
    validate_action_config,
    validate_config_keys,
    validate_observation_config,
    validate_prometheus_config,
    validate_reward_config,
    validate_safeguards_config,
    validate_serving_config,
    validate_training_config,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "arcs.default.yaml"


def _deepcopy_config() -> dict[str, Any]:
    return copy.deepcopy(load_config(DEFAULT_CONFIG))


def test_load_config_root_must_be_mapping(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("- not a mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_config(p)


def test_effective_training_seed_prefers_training_seed() -> None:
    from arcs_rl.config import effective_training_seed

    d = _deepcopy_config()
    d["training"]["seed"] = 99
    assert effective_training_seed(d) == 99


def test_effective_training_seed_falls_back_to_determinism() -> None:
    from arcs_rl.config import effective_training_seed

    d = _deepcopy_config()
    d["training"]["seed"] = None
    d["determinism"]["seed"] = 12
    assert effective_training_seed(d) == 12


@pytest.mark.parametrize(
    ("mutator", "match", "exc"),
    [
        (lambda d: d.pop("version", None), "Missing required", ValueError),
        (lambda d: d["prometheus"].__setitem__("enabled", "yes"), "true or false", ValueError),
        (lambda d: d["prometheus"].__setitem__("bind_address", ""), "non-empty string", ValueError),
        (lambda d: d["prometheus"].__setitem__("scrape_port", 99999), "65535", ValueError),
        (
            lambda d: d["prometheus"].__setitem__("metrics_path", "metrics"),
            "starting with",
            ValueError,
        ),
        (
            lambda d: d["prometheus"].__setitem__("metric_prefix", "   "),
            "non-empty string",
            ValueError,
        ),
        (lambda d: d["serving"].__setitem__("grpc_port", 0), "65535", ValueError),
        (lambda d: d["serving"].__setitem__("device", "tpu"), "cpu", ValueError),
        (
            lambda d: d["serving"].__setitem__("torchscript_path", 123),
            "null or a string",
            ValueError,
        ),
        (lambda d: d["replay_buffer"].__setitem__("shard_format", "lmdb"), "numpy", ValueError),
        (lambda d: d["replay_buffer"].__setitem__("min_transitions", 0), ">= 1", ValueError),
        (lambda d: d["replay_buffer"].__setitem__("transitions_per_shard", 0), ">= 1", ValueError),
        (lambda d: d["replay_buffer"].__setitem__("max_shards", 0), ">= 1", ValueError),
        (lambda d: d["training"].__setitem__("device", "fpga"), "auto", ValueError),
        (lambda d: d["training"].__setitem__("learning_rate", 0.0), "> 0", ValueError),
        (lambda d: d["training"].__setitem__("seed", -1), "non-negative", ValueError),
        (
            lambda d: d["training"]["dqn"].__setitem__("exploration_fraction", 2.0),
            "0 and 1",
            ValueError,
        ),
        (lambda d: d["training"]["dqn"].__setitem__("buffer_size", 0), ">= 1", ValueError),
        (lambda d: d["training"]["ppo"].__setitem__("n_steps", 0), ">= 1", ValueError),
        (lambda d: d["training"]["ppo"].__setitem__("batch_size", 0), ">= 1", ValueError),
        (lambda d: d["training"]["online"].__setitem__("total_timesteps", 0), ">= 1", ValueError),
        (lambda d: d["training"]["offline"].__setitem__("gradient_steps", 0), ">= 1", ValueError),
        (lambda d: d["training"]["offline"].__setitem__("batch_size", 0), ">= 1", ValueError),
        (
            lambda d: d["training"]["offline"].__setitem__("fill_rollout_max_steps", 0),
            ">= 1",
            ValueError,
        ),
        (
            lambda d: d["observation"].__setitem__("missing_value_strategy", "ignore"),
            "neutral",
            ValueError,
        ),
        (
            lambda d: d["observation"]["normalization"].__setitem__("mode", "weird"),
            "min_max",
            ValueError,
        ),
        (
            lambda d: d["reward"].__setitem__("cascade_error_rate_threshold", 2.0),
            "between 0 and 1",
            ValueError,
        ),
        (lambda d: d["action"].__setitem__("algorithm", "a2c"), "dqn", ValueError),
        (lambda d: d["policy"]["retry"].__setitem__("min", 9), "<= policy.retry.max", ValueError),
        (lambda d: d["safeguards"].__setitem__("max_retries", 99), "policy.retry.max", ValueError),
        (
            lambda d: d["safeguards"].__setitem__("max_policy_changes_per_route_per_minute", 0),
            "> 0",
            ValueError,
        ),
        (
            lambda d: d["safeguards"].__setitem__(
                "circuit_breaker_clear_error_rate_threshold",
                0.99,
            ),
            "hysteresis",
            ValueError,
        ),
        (lambda d: d["safeguards"].__setitem__("route_state_ttl_seconds", 0), "> 0", ValueError),
    ],
)
def test_validate_config_keys_rejects_bad_values(
    mutator: Callable[[dict[str, Any]], None],
    match: str,
    exc: type[Exception],
) -> None:
    data = _deepcopy_config()
    mutator(data)
    with pytest.raises(exc, match=match):
        validate_config_keys(data)


def test_validate_reward_weight_type() -> None:
    data = _deepcopy_config()
    data["reward"]["weights"]["success_rate"] = "big"
    with pytest.raises(TypeError, match="number"):
        validate_config_keys(data)


def test_validate_reward_cascade_scale_type() -> None:
    data = _deepcopy_config()
    data["reward"]["cascade_penalty_scale"] = None
    with pytest.raises(TypeError, match="number"):
        validate_config_keys(data)


def test_validate_action_dqn_backoff_outside_policy() -> None:
    data = _deepcopy_config()
    data["action"]["dqn"]["backoff_multipliers"] = [0.1]
    with pytest.raises(ValueError, match="outside"):
        validate_config_keys(data)


def test_validate_action_dqn_timeout_bins_unsorted() -> None:
    data = _deepcopy_config()
    data["action"]["dqn"]["timeout_ms_bins"] = [5000, 100, 10000]
    with pytest.raises(ValueError, match="sorted"):
        validate_config_keys(data)


def test_validate_action_dqn_backoff_element_type() -> None:
    data = _deepcopy_config()
    data["action"]["dqn"]["backoff_multipliers"] = ["x"]
    with pytest.raises(TypeError, match="number"):
        validate_config_keys(data)


def test_validate_action_ppo_bounds_false() -> None:
    data = _deepcopy_config()
    data["action"]["ppo"]["use_policy_bounds"] = False
    with pytest.raises(ValueError, match="not supported"):
        validate_config_keys(data)


def test_validate_safeguards_backoff_cover_policy() -> None:
    data = _deepcopy_config()
    data["safeguards"]["backoff_multiplier_bounds"] = [1.0, 2.0]
    with pytest.raises(ValueError, match="cover policy"):
        validate_config_keys(data)


def test_validate_safeguards_timeout_cover_policy() -> None:
    data = _deepcopy_config()
    data["safeguards"]["timeout_ms_bounds"] = [500.0, 9000.0]
    with pytest.raises(ValueError, match="cover policy"):
        validate_config_keys(data)


def test_validate_observation_feature_order_wrong() -> None:
    data = _deepcopy_config()
    fo = list(data["observation"]["feature_order"])
    fo[0], fo[1] = fo[1], fo[0]
    data["observation"]["feature_order"] = fo
    with pytest.raises(ValueError, match="FEATURE_NAMES"):
        validate_config_keys(data)


def test_validate_observation_feature_order_bad_length() -> None:
    data = _deepcopy_config()
    data["observation"]["feature_order"] = data["observation"]["feature_order"][:5]
    with pytest.raises(ValueError, match="12"):
        validate_config_keys(data)


def test_validate_training_determinism_missing_seed() -> None:
    data = _deepcopy_config()
    del data["determinism"]["seed"]
    with pytest.raises(ValueError, match="determinism"):
        validate_training_config(data["training"], data.get("determinism", {}))


def test_validate_prometheus_not_mapping() -> None:
    with pytest.raises(ValueError, match="mapping"):
        validate_prometheus_config("bad")  # type: ignore[arg-type]


def test_validate_serving_not_mapping() -> None:
    with pytest.raises(ValueError, match="mapping"):
        validate_serving_config([])  # type: ignore[arg-type]


def test_validate_observation_not_mapping() -> None:
    with pytest.raises(ValueError, match="mapping"):
        validate_observation_config("x")  # type: ignore[arg-type]


def test_validate_reward_not_mapping() -> None:
    with pytest.raises(ValueError, match="mapping"):
        validate_reward_config(1)  # type: ignore[arg-type]


def test_validate_action_not_mapping() -> None:
    data = _deepcopy_config()
    with pytest.raises(ValueError, match="mapping"):
        validate_action_config("no", data["policy"])  # type: ignore[arg-type]


def test_validate_safeguards_not_mapping() -> None:
    data = _deepcopy_config()
    with pytest.raises(ValueError, match="mapping"):
        validate_safeguards_config([], data["policy"])  # type: ignore[arg-type]


def test_validate_action_dqn_not_mapping() -> None:
    data = _deepcopy_config()
    data["action"]["dqn"] = "nope"
    with pytest.raises(ValueError, match="mapping"):
        validate_config_keys(data)


def test_validate_action_ppo_not_mapping() -> None:
    data = _deepcopy_config()
    data["action"]["ppo"] = []
    with pytest.raises(ValueError, match="mapping"):
        validate_config_keys(data)


def test_validate_training_subblocks_not_mapping() -> None:
    data = _deepcopy_config()
    data["training"]["dqn"] = "bad"
    with pytest.raises(ValueError, match="mapping"):
        validate_config_keys(data)


def test_round_trip_yaml_dump_load(tmp_path: Path) -> None:
    """Round-trip through YAML should still validate (catches type surprises)."""
    data = _deepcopy_config()
    p = tmp_path / "copy.yaml"
    p.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    again = load_config(p)
    validate_config_keys(again)
