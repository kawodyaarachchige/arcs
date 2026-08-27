"""
Shared path for HTTP and gRPC: build a suggested triple, run safeguards, return the final answer.

Keeping this in one module avoids the REST and gRPC APIs drifting apart.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from arcs_rl.inference.runtime import TorchInferenceRuntime
from arcs_rl.safeguards import PolicyAction, SafeguardEngine
from observation_bridge import maybe_fetch_observation


def metric_route_label(route: str) -> str:
    """Shorten routes so Prometheus labels stay readable."""
    r = route.strip()
    if len(r) > 64:
        return f"{r[:61]}..."
    return r or "unknown"


@dataclass(frozen=True)
class DecideOut:
    retry: int
    backoff_multiplier: float
    timeout_ms: float
    override_reasons: list[str]
    frozen_active: bool
    trace_id: str | None
    suggested_snapshot: dict[str, float]


def default_suggested_from_config(root: dict[str, Any]) -> PolicyAction:
    """YAML defaults when the caller sends no triple (used by the Envoy helper path)."""
    pol = root["policy"]
    return PolicyAction(
        retry=int(pol["retry"]["default"]),
        backoff_multiplier=float(pol["backoff"]["default_multiplier"]),
        timeout_ms=float(pol["timeout_ms"]["default"]),
    )


def suggested_from_inputs(
    root: dict[str, Any],
    inference: TorchInferenceRuntime | None,
    *,
    suggested_retry: float | None,
    suggested_backoff: float | None,
    suggested_timeout_ms: float | None,
    observation: list[float] | None,
    allow_default_suggested: bool,
) -> PolicyAction:
    """
    Turn observation and/or caller suggestions into one PolicyAction before safeguards.

    `allow_default_suggested` is True only for Envoy: we may fall back to YAML defaults.
    For normal `/decide` JSON calls, keep it False so older clients must still pass a triple unless
    they send a full 12-number observation while inference is enabled.
    """
    obs = observation
    has_obs = obs is not None and len(obs) == 12
    has_full_suggested = (
        suggested_retry is not None
        and suggested_backoff is not None
        and suggested_timeout_ms is not None
    )
    if obs is not None and len(obs) not in (0, 12):
        msg = "observation must be empty or have exactly 12 numbers"
        raise ValueError(msg)

    if inference is not None and has_obs:
        return inference.suggested_action(np.asarray(obs, dtype=np.float32))

    if has_full_suggested:
        return PolicyAction(
            retry=int(suggested_retry),
            backoff_multiplier=float(suggested_backoff),
            timeout_ms=float(suggested_timeout_ms),
        )

    if allow_default_suggested:
        return default_suggested_from_config(root)

    inf_on = bool(root["serving"]["inference_enabled"]) and inference is not None
    if inf_on:
        msg = "Inference needs a 12-number observation, or a full suggested triple."
        raise ValueError(msg)

    msg = "Send a suggested triple, or a 12-number observation while inference is enabled."
    raise ValueError(msg)


def run_decide(
    *,
    root: dict[str, Any],
    engine: SafeguardEngine,
    inference: TorchInferenceRuntime | None,
    route: str,
    error_rate: float,
    suggested_retry: float | None,
    suggested_backoff: float | None,
    suggested_timeout_ms: float | None,
    observation: list[float] | None,
    trace_id: str | None,
    allow_default_suggested: bool,
) -> DecideOut:
    """Apply safeguards and package the response."""
    observation = maybe_fetch_observation(root, inference, route, observation)
    suggested = suggested_from_inputs(
        root,
        inference,
        suggested_retry=suggested_retry,
        suggested_backoff=suggested_backoff,
        suggested_timeout_ms=suggested_timeout_ms,
        observation=observation,
        allow_default_suggested=allow_default_suggested,
    )
    now = time.monotonic()
    decision = engine.decide(
        route=route,
        suggested=suggested,
        error_rate=error_rate,
        now_monotonic_s=now,
    )
    return DecideOut(
        retry=decision.action.retry,
        backoff_multiplier=decision.action.backoff_multiplier,
        timeout_ms=decision.action.timeout_ms,
        override_reasons=list(decision.override_reasons),
        frozen_active=decision.frozen_active,
        trace_id=trace_id,
        suggested_snapshot={
            "retry": float(suggested.retry),
            "backoff_multiplier": float(suggested.backoff_multiplier),
            "timeout_ms": float(suggested.timeout_ms),
        },
    )
