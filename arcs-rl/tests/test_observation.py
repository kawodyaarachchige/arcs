"""Tests for the 12-number state vector: windows, norms, missing data, and config wiring."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from arcs_rl.config import load_config, validate_config_keys
from arcs_rl.observation import (
    STATE_DIM,
    aggregator_from_config,
    build_state_vector,
)
from arcs_rl.observation.normalization import min_max_unit, percentile_linear, z_score_to_unit
from arcs_rl.observation.windows import TimedFloatBuffer, TimedRequestBuffer

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "arcs.default.yaml"


def test_state_dim_and_default_config_wiring() -> None:
    root = load_config(DEFAULT_CONFIG)
    validate_config_keys(root)
    agg = aggregator_from_config(root)
    v = build_state_vector(agg, now=100.0)
    assert v.shape == (STATE_DIM,)
    assert np.all(np.isfinite(v))
    assert np.all(v >= 0.0) and np.all(v <= 1.0)


def test_latency_window_percentiles() -> None:
    root = load_config(DEFAULT_CONFIG)
    agg = aggregator_from_config(root)
    # Five latencies inside a 10 s window ending at t=100.
    for i, ms in enumerate([10.0, 20.0, 30.0, 40.0, 50.0]):
        agg.record_latency_ms(95.0 + i * 0.1, ms)
    v = agg.build_vector(now=100.0)
    raw50 = percentile_linear(np.array([10.0, 20.0, 30.0, 40.0, 50.0]), 50.0)
    raw99 = percentile_linear(np.array([10.0, 20.0, 30.0, 40.0, 50.0]), 99.0)
    assert raw50 == pytest.approx(30.0)
    assert raw50 is not None and raw99 is not None
    # min_max norm with latency_ms_upper from default YAML (10000)
    exp50 = min_max_unit(raw50, 0.0, 10000.0, 0.0, 1.0)
    exp99 = min_max_unit(raw99, 0.0, 10000.0, 0.0, 1.0)
    assert v[3] == pytest.approx(exp50)
    assert v[4] == pytest.approx(exp99)


def test_error_rate_thirty_second_window() -> None:
    root = load_config(DEFAULT_CONFIG)
    agg = aggregator_from_config(root)
    agg.record_request(200.0, ok=False)
    agg.record_request(201.0, ok=False)
    agg.record_request(202.0, ok=True)
    v = agg.build_vector(now=210.0)
    assert v[5] == pytest.approx(min_max_unit(2.0 / 3.0, 0.0, 1.0, 0.0, 1.0))


def test_old_samples_drop_out_of_window() -> None:
    buf = TimedFloatBuffer(window_s=10.0)
    buf.append(0.0, 1.0)
    buf.append(100.0, 2.0)
    assert buf.values(now=100.0).tolist() == [2.0]
    assert buf.values(now=105.0).tolist() == [2.0]


def test_nan_latency_ignored_and_slot_missing_filled() -> None:
    root = load_config(DEFAULT_CONFIG)
    agg = aggregator_from_config(root)
    agg.record_latency_ms(50.0, float("nan"))
    v = agg.build_vector(now=50.0)
    # No latency samples → neutral strategy → 0.5 for p50/p99 slots
    assert v[3] == pytest.approx(0.5)
    assert v[4] == pytest.approx(0.5)


def test_pessimistic_missing_raises_latency_slots() -> None:
    root = load_config(DEFAULT_CONFIG)
    root = copy.deepcopy(root)
    root["observation"] = copy.deepcopy(root["observation"])
    root["observation"]["missing_value_strategy"] = "pessimistic"
    agg = aggregator_from_config(root)
    v = agg.build_vector(now=10.0)
    assert v[3] == pytest.approx(1.0)
    assert v[4] == pytest.approx(1.0)


def test_normalization_clips_above_upper_bound() -> None:
    out = min_max_unit(50000.0, 0.0, 10000.0, 0.0, 1.0)
    assert out == pytest.approx(1.0)


def test_determinism_same_sequence() -> None:
    root = load_config(DEFAULT_CONFIG)
    a1 = aggregator_from_config(root)
    a2 = aggregator_from_config(root)
    for t in (10.0, 11.0, 12.0):
        a1.record_latency_ms(t, 100.0 + t)
        a2.record_latency_ms(t, 100.0 + t)
        a1.record_request(t, ok=t != 11.0)
        a2.record_request(t, ok=t != 11.0)
        a1.record_load(t, cpu=0.2, memory=0.3, queue_depth=10.0)
        a2.record_load(t, cpu=0.2, memory=0.3, queue_depth=10.0)
        a1.record_global_rps(t, 500.0)
        a2.record_global_rps(t, 500.0)
    a1.set_policy_context(retry_count=2, backoff_multiplier=1.5, timeout_ms=3000.0)
    a2.set_policy_context(retry_count=2, backoff_multiplier=1.5, timeout_ms=3000.0)
    np.testing.assert_array_almost_equal(a1.build_vector(now=20.0), a2.build_vector(now=20.0))


def test_policy_context_scales_into_vector() -> None:
    root = load_config(DEFAULT_CONFIG)
    agg = aggregator_from_config(root)
    agg.set_policy_context(retry_count=0, backoff_multiplier=0.5, timeout_ms=100.0)
    # Fill minimal telemetry so missing logic does not dominate.
    agg.record_load(1.0, cpu=0.0, memory=0.0, queue_depth=0.0)
    agg.record_global_rps(1.0, 0.0)
    v = agg.build_vector(now=2.0)
    assert v[6] == pytest.approx(0.0)
    assert v[7] == pytest.approx(0.0)
    assert v[8] == pytest.approx(0.0)


def test_z_score_mode_uses_yaml_stats() -> None:
    root = load_config(DEFAULT_CONFIG)
    root = copy.deepcopy(root)
    root["observation"]["normalization"]["mode"] = "z_score"
    agg = aggregator_from_config(root)
    agg.record_latency_ms(10.0, 200.0)
    v = agg.build_vector(now=15.0)
    expected = z_score_to_unit(200.0, 200.0, 150.0, 0.0, 1.0)
    assert v[3] == pytest.approx(expected)
    assert v[4] == pytest.approx(expected)


def test_build_vector_without_now_uses_latest_sample_time() -> None:
    root = load_config(DEFAULT_CONFIG)
    agg = aggregator_from_config(root)
    agg.record_latency_ms(42.0, 80.0)
    v = agg.build_vector()
    v2 = agg.build_vector(now=42.0)
    np.testing.assert_array_almost_equal(v, v2)


def test_timed_request_buffer_error_rate_none_when_empty() -> None:
    buf = TimedRequestBuffer(window_s=30.0)
    assert buf.error_rate(now=10.0) is None


def test_reserved_slots_are_zero() -> None:
    root = load_config(DEFAULT_CONFIG)
    agg = aggregator_from_config(root)
    v = agg.build_vector(now=0.0)
    assert v[10] == 0.0
    assert v[11] == 0.0


def test_invalid_feature_order_rejected(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    root = load_config(DEFAULT_CONFIG)
    bad = copy.deepcopy(root)
    bad["observation"]["feature_order"] = list(reversed(bad["observation"]["feature_order"]))
    import yaml

    p.write_text(yaml.safe_dump(bad), encoding="utf-8")
    data = load_config(p)
    with pytest.raises(ValueError, match="feature_order"):
        validate_config_keys(data)
