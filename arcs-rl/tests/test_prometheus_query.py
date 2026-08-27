"""Unit tests for the shared Prometheus HTTP helper."""

from __future__ import annotations

from unittest.mock import patch

from arcs_rl.monitoring.prometheus_query import instant_query, parse_scalar_value


def test_parse_scalar_value_empty_result() -> None:
    data = {"status": "success", "data": {"result": []}}
    assert parse_scalar_value(data) is None


def test_parse_scalar_value_success() -> None:
    data = {
        "status": "success",
        "data": {
            "result": [
                {
                    "metric": {"job": "x"},
                    "value": [1234567890.0, "42.5"],
                },
            ],
        },
    }
    assert parse_scalar_value(data) == 42.5


def test_instant_query_passes_headers() -> None:
    captured: dict[str, object] = {}

    class FakeResp:
        def __enter__(self) -> FakeResp:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return b'{"status":"success","data":{"result":[]}}'

    def fake_urlopen(req: object, **_k: object) -> FakeResp:
        captured["req"] = req
        return FakeResp()

    with patch("arcs_rl.monitoring.prometheus_query.urllib.request.urlopen", fake_urlopen):
        instant_query(
            "http://localhost:9090",
            "up",
            extra_headers={"Authorization": "Bearer test"},
        )
    req = captured["req"]
    assert hasattr(req, "headers")
    assert req.headers.get("Authorization") == "Bearer test"  # type: ignore[attr-defined]
