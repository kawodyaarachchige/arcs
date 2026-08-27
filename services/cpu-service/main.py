"""
CPU-style service: each request does a fixed amount of math work (hashing).

We never allow “do math forever” in one request — that could freeze the machine.
Keeping a hard upper bound is fair to us and to anyone else on the same pc.
"""

from __future__ import annotations

import hashlib
import os
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from pydantic import BaseModel, Field

SERVICE_NAME = os.environ.get("OTEL_SERVICE_NAME", "arcs-cpu-service")
OTLP_ENDPOINT = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel-collector:4317")
# No single request may ask for more loop iterations than this (change via env if your machine is faster/slower).
_MAX_ITERATIONS = int(os.environ.get("MAX_CPU_ITERATIONS", "5000000"))
_DEFAULT_ITERATIONS = int(os.environ.get("DEFAULT_CPU_ITERATIONS", "200000"))


def _setup_telemetry() -> None:
    resource = Resource.create(
        {
            "service.name": SERVICE_NAME,
            "deployment.environment": "synthetic-testbed",
        }
    )
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=OTLP_ENDPOINT, insecure=True)),
    )
    trace.set_tracer_provider(provider)
    reader = PeriodicExportingMetricReader(
        OTLPMetricExporter(endpoint=OTLP_ENDPOINT, insecure=True),
        export_interval_millis=500,
    )
    metrics.set_meter_provider(MeterProvider(resource=resource, metric_readers=[reader]))


@asynccontextmanager
async def lifespan(app: FastAPI):
    _setup_telemetry()
    meter = metrics.get_meter(__name__)
    app.state.request_count = meter.create_counter(
        "arcs.http.requests",
        description="Total HTTP requests to the CPU microservice.",
    )
    app.state.latency_ms = meter.create_histogram(
        "arcs.http.latency.ms",
        unit="ms",
        description="Request duration including CPU work units.",
    )
    app.state.cpu_units = meter.create_histogram(
        "arcs.cpu.hash_iterations",
        description="How many hash iterations ran (proxy for load).",
    )
    yield


app = FastAPI(title="ARCS CPU Service", version="0.1.0", lifespan=lifespan)
FastAPIInstrumentor.instrument_app(app)


@app.middleware("http")
async def observe(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    app.state.request_count.add(
        1,
        {
            "http.route": request.url.path,
            "http.status_code": str(response.status_code),
        },
    )
    app.state.latency_ms.record(elapsed_ms, {"http.route": request.url.path})
    return response


@app.get("/health/live")
async def live() -> dict[str, str]:
    return {"status": "live"}


@app.get("/health/ready")
async def ready() -> dict[str, str]:
    return {"status": "ready"}


class ComputeBody(BaseModel):
    """Body for POST /compute: how many hash rounds to run (must stay below the server max)."""

    iterations: int = Field(default=_DEFAULT_ITERATIONS, ge=1)


def _run_hash_chain(iterations: int) -> bytes:
    """Do the same work every time (no randomness) — good for comparing runs."""
    x = b"arcs-cpu-seed"
    for _ in range(iterations):
        x = hashlib.sha256(x).digest()
    return x


@app.post("/compute")
async def compute(body: ComputeBody) -> dict[str, Any]:
    """
    Run hashing for `iterations` rounds. If you ask for more than the max, you get an error (we don’t silently trim).
    """
    if body.iterations > _MAX_ITERATIONS:
        raise HTTPException(
            status_code=400,
            detail=f"iterations must be <= {_MAX_ITERATIONS} (lab safety cap)",
        )
    start = time.perf_counter()
    digest = _run_hash_chain(body.iterations)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    app.state.cpu_units.record(body.iterations, {"path": "/compute"})
    return {
        "iterations": body.iterations,
        "elapsed_ms": round(elapsed_ms, 3),
        "digest_prefix": digest[:8].hex(),
    }


@app.get("/compute")
async def compute_get(iterations: int = _DEFAULT_ITERATIONS) -> dict[str, Any]:
    """Same as POST /compute but easy to try from a browser or curl with ?iterations=..."""
    if iterations > _MAX_ITERATIONS or iterations < 1:
        raise HTTPException(
            status_code=400,
            detail=f"iterations must be in [1, {_MAX_ITERATIONS}]",
        )
    body = ComputeBody(iterations=iterations)
    return await compute(body)
