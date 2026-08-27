"""
Aggregator: one service that calls several other URLs and bundles the answers.

Think of one person texting a few friends at once and summarizing who replied and how long it took.
That pattern shows up a lot in real “microservice” systems.
"""

from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

SERVICE_NAME = os.environ.get("OTEL_SERVICE_NAME", "arcs-aggregator-service")
OTLP_ENDPOINT = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel-collector:4317")
# Comma-separated URLs — point straight at echo/cpu, or through Toxiproxy if we want fake network problems (see benchmarks/scenarios).
_DEFAULT_DOWNSTREAM = "http://echo-service:8080/health/ready,http://cpu-service:8080/health/ready"
DOWNSTREAM_URLS = [
    u.strip() for u in os.environ.get("DOWNSTREAM_URLS", _DEFAULT_DOWNSTREAM).split(",") if u.strip()
]
_HTTP_TIMEOUT_S = float(os.environ.get("AGGREGATOR_HTTP_TIMEOUT_S", "5.0"))


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
    # So outgoing HTTP calls show up in the same trace when possible.
    HTTPXClientInstrumentor().instrument()


@asynccontextmanager
async def lifespan(app: FastAPI):
    _setup_telemetry()
    meter = metrics.get_meter(__name__)
    app.state.request_count = meter.create_counter(
        "arcs.http.requests",
        description="Aggregator inbound requests.",
    )
    app.state.latency_ms = meter.create_histogram(
        "arcs.http.latency.ms",
        unit="ms",
        description="Aggregator handler latency.",
    )
    app.state.fanout_results = meter.create_counter(
        "arcs.aggregator.downstream.calls",
        description="Downstream calls completed (label by status).",
    )
    app.state.client = httpx.AsyncClient(timeout=_HTTP_TIMEOUT_S)
    yield
    await app.state.client.aclose()


app = FastAPI(title="ARCS Aggregator Service", version="0.1.0", lifespan=lifespan)
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
    """We don’t actually ping downstreams here — just confirms the app is up."""
    if not DOWNSTREAM_URLS:
        return {"status": "ready", "note": "no downstreams configured"}
    return {"status": "ready"}


@app.get("/aggregate")
async def aggregate() -> dict[str, Any]:
    """
    Hit every configured URL at the same time and return a short report per URL.
    If one URL fails, we still see the others — that’s what “partial failure” looks like.
    """
    client: httpx.AsyncClient = app.state.client
    results: list[dict[str, Any]] = []

    async def one(url: str) -> dict[str, Any]:
        start = time.perf_counter()
        try:
            r = await client.get(url)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            ok = r.is_success
            app.state.fanout_results.add(1, {"url": url, "ok": str(ok)})
            return {
                "url": url,
                "status_code": r.status_code,
                "elapsed_ms": round(elapsed_ms, 3),
                "ok": ok,
            }
        except httpx.RequestError as e:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            app.state.fanout_results.add(1, {"url": url, "ok": "false"})
            return {
                "url": url,
                "error": type(e).__name__,
                "detail": str(e),
                "elapsed_ms": round(elapsed_ms, 3),
                "ok": False,
            }

    results = await asyncio.gather(*[one(u) for u in DOWNSTREAM_URLS])
    oks = sum(1 for r in results if r.get("ok"))
    return {
        "downstreams": len(DOWNSTREAM_URLS),
        "success_count": oks,
        "results": results,
    }
