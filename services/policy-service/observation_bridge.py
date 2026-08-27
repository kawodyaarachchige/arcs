"""
Optional HTTP call to the telemetry bridge when Envoy did not send a 12-number observation.

We only do this when TorchScript inference is enabled, because that mode needs a real vector.
"""

from __future__ import annotations

import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

logger = logging.getLogger("arcs.policy.observation_bridge")


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _effective_bridge_block(root: dict[str, Any]) -> dict[str, Any]:
    """Merge YAML with optional Kubernetes-style env overrides (full service DNS name)."""
    raw = root.get("observation_bridge")
    block: dict[str, Any] = dict(raw) if isinstance(raw, dict) else {}
    if _env_truthy("ARCS_OBSERVATION_BRIDGE_ENABLED"):
        block["enabled"] = True
    url = os.environ.get("ARCS_OBSERVATION_BRIDGE_URL", "").strip()
    if url:
        block["base_url"] = url.rstrip("/")
    return block


def maybe_fetch_observation(
    root: dict[str, Any],
    inference: Any,
    route: str,
    observation: list[float] | None,
) -> list[float] | None:
    """
    Return the caller’s observation if present; else try the telemetry bridge when inference is on.

    On network errors we log a warning and return None so the caller can surface a clear error.
    """
    if observation is not None and len(observation) == 12:
        return observation

    block = _effective_bridge_block(root)
    if not block.get("enabled"):
        return observation

    inf_on = bool(root["serving"]["inference_enabled"]) and inference is not None
    if not inf_on:
        return observation

    base = str(block.get("base_url", "")).rstrip("/")
    if not base:
        return observation

    timeout = float(block.get("timeout_seconds", 1.0))
    q = urllib.parse.urlencode({"route": route})
    url = f"{base}/v1/state?{q}"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8").strip()
    except urllib.error.HTTPError as e:
        logger.warning("observation bridge HTTP %s for %s", e.code, url)
        return observation
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        logger.warning("observation bridge unreachable (%s): %s", type(e).__name__, e)
        return observation

    parts = [p.strip() for p in raw.split(",") if p.strip() != ""]
    try:
        floats = [float(p) for p in parts]
    except ValueError:
        logger.warning("observation bridge returned non-numeric body")
        return observation
    if len(floats) != 12:
        logger.warning("observation bridge returned %s numbers, need 12", len(floats))
        return observation
    return floats
