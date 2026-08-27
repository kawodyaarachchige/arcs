#!/usr/bin/env python3
"""
Quick manual check that the policy service clamps values and exposes Prometheus text.

Run after `docker compose ... up policy-service` (or point ARCS_POLICY_URL at any reachable base URL).

Example:
  ARCS_POLICY_URL=http://127.0.0.1:18080 python scripts/safeguard_smoke_test.py
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request


def _get(url: str) -> tuple[int, bytes]:
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def _post_json(url: str, payload: dict) -> tuple[int, bytes]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        return resp.status, resp.read()


def main() -> int:
    base = os.environ.get("ARCS_POLICY_URL", "http://127.0.0.1:18080").rstrip("/")
    code, health = _get(f"{base}/health/live")
    if code != 200 or health.decode("utf-8").strip() != "ok":
        print("health check failed:", code, health[:200])
        return 1

    code, metrics = _get(f"{base}/metrics")
    if code != 200 or b"arcs_policy_decisions_total" not in metrics:
        print("metrics scrape failed:", code, metrics[:200])
        return 1

    payload = {
        "route": "/smoke",
        "error_rate": 0.0,
        "suggested": {"retry": 999, "backoff_multiplier": 999.0, "timeout_ms": 1.0},
        "trace_id": "smoke-test",
    }
    code, body = _post_json(f"{base}/decide", payload)
    if code != 200:
        print("decide failed:", code, body[:200])
        return 1
    data = json.loads(body.decode("utf-8"))
    if data["retry"] > 5:
        print("retry not clamped:", data)
        return 1
    if data["timeout_ms"] < 100:
        print("timeout not clamped:", data)
        return 1
    print("OK:", data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
