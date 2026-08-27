"""Tests that the default YAML file loads and has the right shape (sections and types)."""

from __future__ import annotations

from pathlib import Path

import pytest

from arcs_rl.config import (
    load_config,
    validate_action_config,
    validate_config_keys,
    validate_reward_config,
    validate_safeguards_config,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "arcs.default.yaml"


def test_default_config_loads_and_has_required_keys() -> None:
    data = load_config(DEFAULT_CONFIG)
    validate_config_keys(data)
    assert data["version"] == 1
    assert data["determinism"]["seed"] == 42
    assert data["policy"]["retry"]["max"] == 5
    assert data["replay_buffer"]["min_transitions"] == 100_000
    assert data["replay_buffer"]["transitions_per_shard"] == 5000
    assert data["training"]["device"] == "auto"
    assert data["training"]["offline"]["gradient_steps"] == 10000
    assert data["simulation"]["telemetry_export_interval_ms"] == 500
    assert data["simulation"]["max_request_delay_ms"] == 500
    assert data["simulation"]["compose_host_ports"]["echo"] == 18001
    assert data["simulation"]["compose_host_ports"]["policy_service"] == 18080
    assert data["simulation"]["compose_host_ports"]["policy_grpc"] == 15051
    assert data["simulation"]["compose_host_ports"]["envoy"] == 10000
    assert data["simulation"]["compose_host_ports"]["grafana"] == 3000
    assert data["simulation"]["compose_host_ports"]["telemetry_bridge"] == 18085
    assert data["observation_bridge"]["enabled"] is False
    assert "base_url" in data["observation_bridge"]
    assert data["serving"]["grpc_port"] == 50051
    assert data["serving"]["inference_enabled"] is False
    assert data["observation"]["latency_window_s"] == 10
    assert data["observation"]["error_window_s"] == 30
    assert data["observation"]["reference_rps"] == 10000
    assert data["observation"]["missing_value_strategy"] == "neutral"
    assert data["observation"]["normalization"]["mode"] == "min_max"
    assert len(data["observation"]["feature_order"]) == 12
    assert data["action"]["algorithm"] == "dqn"
    assert "backoff_multipliers" in data["action"]["dqn"]
    assert data["action"]["ppo"]["use_policy_bounds"] is True
    assert "cascade_error_rate_threshold" in data["reward"]
    assert data["reward"]["cascade_error_rate_threshold"] == 0.35
    assert data["safeguards"]["circuit_breaker_clear_error_rate_threshold"] == 0.35
    assert data["safeguards"]["route_state_ttl_seconds"] == 3600
    assert data["prometheus"]["enabled"] is False
    assert data["prometheus"]["scrape_port"] == 9092


def test_safeguards_rejects_bad_hysteresis() -> None:
    data = load_config(DEFAULT_CONFIG)
    data["safeguards"]["circuit_breaker_clear_error_rate_threshold"] = 0.6
    with pytest.raises(ValueError, match="hysteresis"):
        validate_safeguards_config(data["safeguards"], data["policy"])


def test_load_config_rejects_non_mapping_root(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_config(p)


def test_action_rejects_unsorted_timeout_bins() -> None:
    data = load_config(DEFAULT_CONFIG)
    data["action"]["dqn"]["timeout_ms_bins"] = [500, 100, 2000]
    with pytest.raises(ValueError, match="sorted"):
        validate_action_config(data["action"], data["policy"])


def test_action_rejects_backoff_outside_policy() -> None:
    data = load_config(DEFAULT_CONFIG)
    data["action"]["dqn"]["backoff_multipliers"] = [0.1, 1.0, 3.0]
    with pytest.raises(ValueError, match="outside"):
        validate_action_config(data["action"], data["policy"])


def test_reward_rejects_bad_cascade_threshold() -> None:
    data = load_config(DEFAULT_CONFIG)
    data["reward"]["cascade_error_rate_threshold"] = 1.5
    with pytest.raises(ValueError, match="between"):
        validate_reward_config(data["reward"])
