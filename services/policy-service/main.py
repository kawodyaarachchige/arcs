"""
Policy service: turns a suggested retry/backoff/timeout into a safe final answer.

HTTP and gRPC share the same logic. TorchScript uses the 12-D observation when YAML enables it.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Any

import grpc
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from arcs.policy.v1 import policy_pb2_grpc
from arcs_rl.inference.runtime import load_torch_inference_or_none
from arcs_rl.safeguards import SafeguardConfig, SafeguardEngine
from config import load_arcs_root_config
from decide_logic import metric_route_label, run_decide
from grpc_servicer import PolicyServicer
from logging_utils import log_decision
from metrics import DECISION_LATENCY_SECONDS, DECISIONS_TOTAL, FREEZE_ACTIVE, OVERRIDES_TOTAL
from schemas import DecideRequest, DecideResponse

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("arcs.policy")


class _Runtime(BaseModel):
    """Loaded YAML, safeguard engine, and optional neural net for observations."""

    model_config = {"arbitrary_types_allowed": True}
    engine: SafeguardEngine
    inference: Any
    root: dict[str, Any]


def _start_grpc_server(runtime: _Runtime, port: int) -> grpc.Server:
    """Background thread runs this blocking server so FastAPI and gRPC share one process."""
    server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=8))
    policy_pb2_grpc.add_PolicyServicer_to_server(
        PolicyServicer(root=runtime.root, runtime=runtime),
        server,
    )
    listen = f"[::]:{port}"
    server.add_insecure_port(listen)

    def _run() -> None:
        server.start()
        server.wait_for_termination()

    threading.Thread(target=_run, name="arcs-policy-grpc", daemon=True).start()
    logger.info("gRPC policy listening on %s", listen)
    return server


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load YAML once; build the engine and optional TorchScript; start gRPC in a side thread."""
    root = load_arcs_root_config()
    cfg = SafeguardConfig.from_arcs_config(root)
    inference = load_torch_inference_or_none(root)
    app.state.runtime = _Runtime(engine=SafeguardEngine(cfg), inference=inference, root=root)
    port = int(root["serving"]["grpc_port"])
    app.state.grpc_server = _start_grpc_server(app.state.runtime, port)
    yield


app = FastAPI(
    title="ARCS Policy Service",
    version="0.2.0",
    lifespan=lifespan,
)


def _request_has_arcs_envoy_headers(request: Request) -> bool:
    """True when Envoy (or a client) sent the headers ext_authz forwards for a check."""
    h = request.headers
    return h.get("x-arcs-route") is not None or h.get("x-arcs-error-rate") is not None


def _build_ext_authz_response(request: Request, runtime: _Runtime) -> Response:
    """Shared ext_authz implementation for /envoy/ext_authz and Envoy's client-path checks."""
    h = request.headers
    route = h.get("x-arcs-route") or request.url.path or "/"
    err_raw = h.get("x-arcs-error-rate")
    error_rate = _parse_float_header(err_raw)
    if error_rate is None:
        error_rate = 0.0
    trace_id = h.get("x-arcs-trace-id") or h.get("traceparent")

    obs_raw = h.get("x-arcs-obs")
    observation: list[float] | None = None
    if obs_raw:
        parts = [p.strip() for p in obs_raw.split(",")]
        try:
            observation = [float(p) for p in parts if p != ""]
        except ValueError:
            observation = None
        if observation is not None and len(observation) != 12:
            return Response(status_code=400, content=b"bad x-arcs-obs")

    sr = _parse_float_header(h.get("x-arcs-suggested-retry"))
    sb = _parse_float_header(h.get("x-arcs-suggested-backoff"))
    st = _parse_float_header(h.get("x-arcs-suggested-timeout-ms"))

    t0 = time.perf_counter()
    try:
        out = run_decide(
            root=runtime.root,
            engine=runtime.engine,
            inference=runtime.inference,
            route=route,
            error_rate=error_rate,
            suggested_retry=sr,
            suggested_backoff=sb,
            suggested_timeout_ms=st,
            observation=observation,
            trace_id=trace_id,
            allow_default_suggested=True,
        )
    except ValueError as e:
        return Response(status_code=400, content=str(e).encode("utf-8"))
    elapsed = time.perf_counter() - t0
    DECISION_LATENCY_SECONDS.observe(elapsed)

    mroute = metric_route_label(route)
    DECISIONS_TOTAL.labels(mroute).inc()
    for reason in out.override_reasons:
        OVERRIDES_TOTAL.labels(mroute, reason).inc()
    FREEZE_ACTIVE.labels(mroute).set(1.0 if out.frozen_active else 0.0)

    payload: dict[str, Any] = {
        "route": route,
        "error_rate": error_rate,
        "suggested": out.suggested_snapshot,
        "final": {
            "retry": out.retry,
            "backoff_multiplier": out.backoff_multiplier,
            "timeout_ms": out.timeout_ms,
        },
        "override_reasons": out.override_reasons,
        "frozen_active": out.frozen_active,
        "trace_id": trace_id,
        "transport": "envoy_ext_authz",
    }
    log_decision(payload)

    headers = {
        "x-arcs-retry": str(out.retry),
        "x-arcs-backoff-mult": str(out.backoff_multiplier),
        "x-arcs-timeout-ms": str(out.timeout_ms),
        "x-arcs-frozen": "1" if out.frozen_active else "0",
    }
    if trace_id:
        headers["x-arcs-trace-id"] = trace_id
    return Response(status_code=200, headers=headers)


@app.middleware("http")
async def envoy_ext_authz_client_path_compat(request: Request, call_next):
    """
    Envoy http ext_authz calls policy-service with the *client* URL path (/echo, /health/live, …),
    not /envoy/ext_authz. Treat those as auth checks when ARCS headers are present.
    """
    path = request.url.path
    if path == "/envoy/ext_authz":
        return await call_next(request)
    if path.startswith("/decide") or path == "/metrics":
        return await call_next(request)
    if not _request_has_arcs_envoy_headers(request):
        return await call_next(request)
    runtime: _Runtime = request.app.state.runtime
    return _build_ext_authz_response(request, runtime)


@app.get("/health/live", response_class=PlainTextResponse)
def health_live() -> str:
    """Simple liveness probe for Docker/Kubernetes."""
    return "ok"


@app.get("/metrics")
def metrics() -> PlainTextResponse:
    """Prometheus text format (same idea as other services, but local scrape here)."""
    data = generate_latest()
    return PlainTextResponse(content=data.decode("utf-8"), media_type=CONTENT_TYPE_LATEST)


@app.post("/decide", response_model=DecideResponse)
def decide(body: DecideRequest) -> DecideResponse:
    """Apply safeguards and return the final triple plus override reasons."""
    runtime: _Runtime = app.state.runtime
    t0 = time.perf_counter()
    sug = body.suggested
    try:
        out = run_decide(
            root=runtime.root,
            engine=runtime.engine,
            inference=runtime.inference,
            route=body.route,
            error_rate=body.error_rate,
            suggested_retry=float(sug.retry) if sug is not None else None,
            suggested_backoff=float(sug.backoff_multiplier) if sug is not None else None,
            suggested_timeout_ms=float(sug.timeout_ms) if sug is not None else None,
            observation=body.observation,
            trace_id=body.trace_id,
            allow_default_suggested=False,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    elapsed = time.perf_counter() - t0
    DECISION_LATENCY_SECONDS.observe(elapsed)

    mroute = metric_route_label(body.route)
    DECISIONS_TOTAL.labels(mroute).inc()
    for reason in out.override_reasons:
        OVERRIDES_TOTAL.labels(mroute, reason).inc()
    FREEZE_ACTIVE.labels(mroute).set(1.0 if out.frozen_active else 0.0)

    payload: dict[str, Any] = {
        "route": body.route,
        "error_rate": body.error_rate,
        "suggested": out.suggested_snapshot,
        "final": {
            "retry": out.retry,
            "backoff_multiplier": out.backoff_multiplier,
            "timeout_ms": out.timeout_ms,
        },
        "override_reasons": out.override_reasons,
        "frozen_active": out.frozen_active,
        "trace_id": body.trace_id,
        "transport": "http",
    }
    log_decision(payload)

    return DecideResponse(
        retry=out.retry,
        backoff_multiplier=out.backoff_multiplier,
        timeout_ms=out.timeout_ms,
        override_reasons=out.override_reasons,
        frozen_active=out.frozen_active,
    )


def _parse_float_header(raw: str | None) -> float | None:
    if raw is None or raw.strip() == "":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


@app.api_route("/envoy/ext_authz", methods=["GET", "POST"])
def envoy_ext_authz(request: Request) -> Response:
    """
    Envoy calls this before forwarding to the app. We return HTTP 200 and headers the proxy can
    attach to the upstream request (retry / backoff / timeout), plus tracing ids when present.
    """
    runtime: _Runtime = app.state.runtime
    return _build_ext_authz_response(request, runtime)


@app.get("/")
def root() -> dict[str, str]:
    """Tiny hint for humans poking the service in a browser."""
    return {
        "service": "arcs-policy",
        "health": "/health/live",
        "decide": "POST /decide",
        "grpc": "arcs.policy.v1.Policy/Decide",
        "envoy": "/envoy/ext_authz",
    }
