"""Talk to Prometheus over HTTP (instant queries only — no extra Python deps)."""

from __future__ import annotations

from arcs_rl.monitoring.prometheus_query import instant_query, parse_scalar_value

__all__ = ["instant_query", "parse_scalar_value"]
