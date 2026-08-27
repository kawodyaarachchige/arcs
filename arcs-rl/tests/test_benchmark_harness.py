"""Unit tests for the benchmark harness (Prometheus parsing, YAML, manifests)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from benchmarks.harness.bench_config import load_benchmark_config
from benchmarks.harness.manifest import try_git_sha, utc_now_iso, write_manifest
from benchmarks.harness.prometheus_api import parse_scalar_value

from arcs_rl.config import load_config, validate_config_keys

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_parse_scalar_value_vector() -> None:
    sample = {
        "status": "success",
        "data": {
            "resultType": "vector",
            "result": [
                {
                    "metric": {"job": "test"},
                    "value": [1710000000.0, "0.042"],
                },
            ],
        },
    }
    assert parse_scalar_value(sample) == pytest.approx(0.042)


def test_parse_scalar_value_empty() -> None:
    assert parse_scalar_value({"status": "success", "data": {"result": []}}) is None


def test_load_default_benchmark_config() -> None:
    p = REPO_ROOT / "benchmarks" / "config" / "default.yaml"
    cfg = load_benchmark_config(p)
    assert cfg["prometheus"]["base_url"].startswith("http")
    assert len(cfg["prometheus_queries"]) >= 1


def test_benchmark_static_yaml_matches_arcs_schema() -> None:
    """The static benchmark file must stay a full valid ARCS root config."""
    p = REPO_ROOT / "configs" / "benchmark-static.yaml"
    data = load_config(p)
    validate_config_keys(data)


def test_benchmark_adaptive_yaml_matches_arcs_schema() -> None:
    p = REPO_ROOT / "configs" / "benchmark-adaptive.yaml"
    data = load_config(p)
    validate_config_keys(data)
    assert data["serving"]["inference_enabled"] is True


def test_write_manifest_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "m.json"
    write_manifest(p, {"a": 1, "b": "x"})
    out = json.loads(p.read_text(encoding="utf-8"))
    assert out["a"] == 1


def test_utc_now_iso_format() -> None:
    s = utc_now_iso()
    assert "T" in s
    assert s.endswith("+00:00")


def test_try_git_sha_returns_string_or_none() -> None:
    sha = try_git_sha(REPO_ROOT)
    assert sha is None or (isinstance(sha, str) and len(sha) >= 4)


def test_instant_query_uses_urllib() -> None:
    from benchmarks.harness import prometheus_api

    class FakeResp:
        def __enter__(self) -> FakeResp:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return b'{"status":"success","data":{"resultType":"vector","result":[]}}'

    def fake_urlopen(*_a: object, **_k: object) -> FakeResp:
        return FakeResp()

    from arcs_rl.monitoring import prometheus_query as pq

    with patch.object(pq.urllib.request, "urlopen", fake_urlopen):
        data = prometheus_api.instant_query("http://localhost:9090", "up")
    assert data["status"] == "success"
