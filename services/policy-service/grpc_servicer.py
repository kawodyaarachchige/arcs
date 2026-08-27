"""
gRPC implementation of arcs.policy.v1.Policy — same decisions as the HTTP /decide route.

Metadata: trace ids may appear as `traceparent` (W3C) or `x-request-id` (common on gateways).
"""

from __future__ import annotations

import time
from typing import Any

import grpc

from arcs.policy.v1 import policy_pb2, policy_pb2_grpc
from decide_logic import metric_route_label, run_decide
from logging_utils import log_decision
from metrics import DECISION_LATENCY_SECONDS, DECISIONS_TOTAL, FREEZE_ACTIVE, OVERRIDES_TOTAL


def _trace_from_metadata(metadata: Any) -> str | None:
    """Pick the first useful id from gRPC metadata for log correlation."""
    if not metadata:
        return None
    lower: dict[str, Any] = {}
    for k, v in metadata:
        key = k.decode("utf-8") if isinstance(k, bytes) else str(k)
        lower[key.lower()] = v
    for key in ("traceparent", "x-request-id", "x-arcs-trace-id"):
        if key in lower:
            raw = lower[key]
            if isinstance(raw, bytes):
                return raw.decode("utf-8", errors="replace")
            return str(raw)
    return None


class PolicyServicer(policy_pb2_grpc.PolicyServicer):
    """Delegates to the same safeguard engine and optional TorchScript as FastAPI."""

    def __init__(self, *, root: dict[str, Any], runtime: Any) -> None:
        self._root = root
        self._runtime = runtime

    def Decide(
        self,
        request: policy_pb2.DecideRequest,
        context: grpc.ServicerContext,
    ) -> policy_pb2.DecideResponse:
        t0 = time.perf_counter()
        md = context.invocation_metadata()
        trace = request.trace_id or _trace_from_metadata(md) or None

        if request.HasField("suggested"):
            s = request.suggested
            sr = float(s.retry)
            sb = float(s.backoff_multiplier)
            st = float(s.timeout_ms)
        else:
            sr = sb = st = None
        obs = list(request.observation) if request.observation else None

        try:
            out = run_decide(
                root=self._root,
                engine=self._runtime.engine,
                inference=self._runtime.inference,
                route=request.route,
                error_rate=float(request.error_rate),
                suggested_retry=sr,
                suggested_backoff=sb,
                suggested_timeout_ms=st,
                observation=obs,
                trace_id=trace,
                allow_default_suggested=False,
            )
        except ValueError as e:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))

        elapsed = time.perf_counter() - t0
        DECISION_LATENCY_SECONDS.observe(elapsed)

        mroute = metric_route_label(request.route)
        DECISIONS_TOTAL.labels(mroute).inc()
        for reason in out.override_reasons:
            OVERRIDES_TOTAL.labels(mroute, reason).inc()
        FREEZE_ACTIVE.labels(mroute).set(1.0 if out.frozen_active else 0.0)

        payload: dict[str, Any] = {
            "route": request.route,
            "error_rate": float(request.error_rate),
            "suggested": out.suggested_snapshot,
            "final": {
                "retry": out.retry,
                "backoff_multiplier": out.backoff_multiplier,
                "timeout_ms": out.timeout_ms,
            },
            "override_reasons": out.override_reasons,
            "frozen_active": out.frozen_active,
            "trace_id": trace,
            "transport": "grpc",
        }
        log_decision(payload)

        return policy_pb2.DecideResponse(
            retry=out.retry,
            backoff_multiplier=out.backoff_multiplier,
            timeout_ms=out.timeout_ms,
            override_reasons=out.override_reasons,
            frozen_active=out.frozen_active,
        )
