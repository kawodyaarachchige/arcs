"""
Call Prometheus HTTP instant queries without extra dependencies (stdlib only).

The benchmark harness and the Streamlit operator view both use this so query behavior stays
identical.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


def instant_query(
    base_url: str,
    expr: str,
    *,
    timeout_s: float = 30.0,
    extra_headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Run GET /api/v1/query and return the full JSON object.

    Pass ``extra_headers`` for auth, e.g. ``{"Authorization": "Bearer ..."}`` (never commit tokens).
    """
    q = urllib.parse.urlencode({"query": expr})
    url = f"{base_url.rstrip('/')}/api/v1/query?{q}"
    headers = dict(extra_headers) if extra_headers else {}
    req = urllib.request.Request(url, method="GET", headers=headers)
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


def parse_scalar_value(data: dict[str, Any]) -> float | None:
    """
    Pull one numeric value out of an instant-query response.

    Returns None if Prometheus has no point yet (common right after startup).
    """
    if data.get("status") != "success":
        return None
    result = data.get("data", {}).get("result")
    if not result:
        return None
    first = result[0]
    value_pair = first.get("value")
    if not value_pair or len(value_pair) < 2:
        return None
    raw = value_pair[1]
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None
