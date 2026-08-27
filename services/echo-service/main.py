"""
Echo service: you send text (or JSON), it sends it back — like shouting into a canyon.

We cap how long a request can “pretend to be slow” so nobody overloads a shared computer by accident.
Numbers about requests (counts, timing) are sent to OpenTelemetry so Prometheus can graph them later.
"""

from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Query, Request
from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

SERVICE_NAME = os.environ.get("OTEL_SERVICE_NAME", "arcs-echo-service")
OTLP_ENDPOINT = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel-collector:4317")
# Never sleep longer than this many milliseconds on purpose (matches the kind of slow network we simulate elsewhere).
_MAX_DELAY_MS = int(os.environ.get("MAX_REQUEST_DELAY_MS", "500"))


def _setup_telemetry() -> None:
    """Send traces and metrics to the OpenTelemetry collector (local dev uses simple gRPC, not encrypted)."""
    resource = Resource.create(
        {
            "service.name": SERVICE_NAME,
            # So graphs show this is practice traffic, not real customers.
            "deployment.environment": "synthetic-testbed",
        }
    )
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(
        BatchSpanProcessor(
            OTLPSpanExporter(endpoint=OTLP_ENDPOINT, insecure=True),
        )
    )
    trace.set_tracer_provider(provider)

    # Push metric updates about this often (milliseconds) so graphs stay fairly fresh.
    reader = PeriodicExportingMetricReader(
        OTLPMetricExporter(endpoint=OTLP_ENDPOINT, insecure=True),
        export_interval_millis=500,
    )
    metrics.set_meter_provider(MeterProvider(resource=resource, metric_readers=[reader]))


@asynccontextmanager
async def lifespan(app: FastAPI):
    _setup_telemetry()
    # Count requests and record how long they take (in addition to FastAPI’s built-in tracing).
    meter = metrics.get_meter(__name__)
    app.state.request_count = meter.create_counter(
        "arcs.http.requests",
        description="Total HTTP requests handled by this synthetic echo service.",
    )
    app.state.latency_ms = meter.create_histogram(
        "arcs.http.latency.ms",
        unit="ms",
        description="End-to-end request time inside the service (not including network outside).",
    )
    yield
    # When the process exits, the SDK tries to flush anything still buffered.


app = FastAPI(
    title="ARCS Echo Service",
    version="0.1.0",
    lifespan=lifespan,
)
FastAPIInstrumentor.instrument_app(app)


@app.middleware("http")
async def count_requests(request: Request, call_next):
    """Count each request and record how long it took."""
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
    app.state.latency_ms.record(
        elapsed_ms,
        {"http.route": request.url.path},
    )
    return response


@app.get("/health/live")
async def live() -> dict[str, str]:
    """Simple “is the program running?” check (used by orchestrators like Kubernetes)."""
    return {"status": "live"}


@app.get("/health/ready")
async def ready() -> dict[str, str]:
    """“Ready to take traffic?” — this service doesn’t wait on a database, so ready ≈ live."""
    return {"status": "ready"}


@app.get("/echo")
async def echo_get(
    message: str = Query(default="hello", max_length=512),
    delay_ms: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    """
    Return your message. Optional delay_ms adds a pause (capped) so you can practice slow responses safely.
    """
    d = min(delay_ms, _MAX_DELAY_MS)
    if d > 0:
        await asyncio.sleep(d / 1000.0)
    return {"message": message, "delay_ms_applied": d}


@app.post("/echo")
async def echo_post(body: dict[str, Any], delay_ms: int = Query(default=0, ge=0)) -> dict[str, Any]:
    """Return your JSON body back; delay_ms works the same as on GET /echo."""
    d = min(delay_ms, _MAX_DELAY_MS)
    if d > 0:
        await asyncio.sleep(d / 1000.0)
    return {"echo": body, "delay_ms_applied": d}
