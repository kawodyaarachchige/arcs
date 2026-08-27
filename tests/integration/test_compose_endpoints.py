"""
End-to-end checks against the Docker Compose stack (Envoy, policy, sample apps).

These only run when you opt in (see conftest): they start containers and need a few minutes.
"""

from __future__ import annotations

import json
import urllib.request

import pytest


@pytest.mark.integration
def test_policy_health_live(compose_stack: None) -> None:
    with urllib.request.urlopen("http://127.0.0.1:18080/health/live", timeout=10) as r:
        assert r.status == 200


@pytest.mark.integration
def test_echo_through_envoy_ext_authz(compose_stack: None) -> None:
    """
    Traffic hits Envoy first; Envoy asks the policy service, then forwards to echo.

    We only check that we get HTTP 200 and a body — proving the authz hop completed.
    """
    req = urllib.request.Request(
        "http://127.0.0.1:10000/echo?message=integration",
        headers={
            "x-arcs-route": "/echo",
            "x-arcs-error-rate": "0.01",
            "x-arcs-trace-id": "itest-1",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        assert r.status == 200
        body = r.read().decode("utf-8")
    data = json.loads(body)
    assert "message" in data or "echo" in body.lower()


@pytest.mark.integration
def test_prometheus_targets_metric(compose_stack: None) -> None:
    """Prometheus should be scraping something (proves metrics path is alive)."""
    url = "http://127.0.0.1:9090/api/v1/query?query=up"
    with urllib.request.urlopen(url, timeout=15) as r:
        payload = json.loads(r.read().decode("utf-8"))
    assert payload.get("status") == "success"
    assert "data" in payload


@pytest.mark.integration
def test_direct_echo_port(compose_stack: None) -> None:
    with urllib.request.urlopen("http://127.0.0.1:18001/health/live", timeout=10) as r:
        assert r.status == 200
