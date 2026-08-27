"""Tests for feeding StateAggregator from Prometheus-style instant numbers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from arcs_rl.config import load_config, validate_config_keys
from arcs_rl.observation.aggregator import aggregator_from_config
from arcs_rl.observation.prom_feed import apply_instant_metrics, policy_context_from_arcs_root

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "arcs.default.yaml"


def test_apply_instant_metrics_builds_twelve_d_vector() -> None:
    root = load_config(DEFAULT_CONFIG)
    validate_config_keys(root)
    agg = aggregator_from_config(root)
    apply_instant_metrics(
        agg,
        now=1_000_000.0,
        p50_ms=80.0,
        p99_ms=240.0,
        error_rate=0.1,
        rps=500.0,
        cpu=0.2,
        memory=0.3,
        queue_depth=2.0,
        policy=policy_context_from_arcs_root(root),
    )
    vec = agg.build_vector(1_000_000.0)
    assert vec.shape == (12,)
    assert np.all(np.isfinite(vec))
    assert float(vec.min()) >= 0.0
    assert float(vec.max()) <= 1.0


def test_policy_context_reads_yaml_defaults() -> None:
    root = load_config(DEFAULT_CONFIG)
    validate_config_keys(root)
    ctx = policy_context_from_arcs_root(root)
    assert ctx.retry_count == int(root["policy"]["retry"]["default"])
