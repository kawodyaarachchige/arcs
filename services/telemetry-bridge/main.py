"""
HTTP service: build the 12-number observation vector the RL stack uses.

We read live Prometheus metrics and optional Kafka JSON. The policy service may call /v1/state when
Envoy did not send x-arcs-obs.

Prometheus already holds latency-style signals from OpenTelemetry; we push them through the same
StateAggregator math as training so the model sees a consistent state.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, generate_latest

from arcs_rl.config import load_config, validate_config_keys
from arcs_rl.monitoring.prometheus_query import instant_query, parse_scalar_value
from arcs_rl.observation.aggregator import StateAggregator, aggregator_from_config
from arcs_rl.observation.prom_feed import apply_instant_metrics, policy_context_from_arcs_root

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("arcs.telemetry_bridge")

KAFKA_EVENTS_TOTAL = Counter(
    "arcs_telemetry_bridge_kafka_events_total",
    "Kafka messages processed by the telemetry bridge",
    ["topic", "status"],
)

# One aggregator per route label so several paths can stay separated if you need that later.
_AGG_LOCK = threading.Lock()
_AGG_BY_ROUTE: dict[str, StateAggregator] = {}
_ROOT: dict[str, Any] | None = None


def _config_path() -> Path:
    env = os.environ.get("TELEMETRY_BRIDGE_ARCS_CONFIG") or os.environ.get("ARCS_CONFIG")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    return here.parents[2] / "configs" / "arcs.default.yaml"


def _prom_base() -> str:
    return os.environ.get("PROMETHEUS_BASE_URL", "http://prometheus:9090").rstrip("/")


def _kafka_bootstrap() -> str | None:
    raw = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "").strip()
    return raw or None


def _kafka_topic() -> str:
    return os.environ.get("KAFKA_TOPIC_ARCS_TELEMETRY", "arcs.telemetry.v1")


def _load_root() -> dict[str, Any]:
    global _ROOT
    if _ROOT is None:
        data = load_config(_config_path())
        validate_config_keys(data)
        _ROOT = data
    return _ROOT


def _get_agg(route: str) -> StateAggregator:
    r = route.strip() or "/"
    with _AGG_LOCK:
        if r not in _AGG_BY_ROUTE:
            _AGG_BY_ROUTE[r] = aggregator_from_config(_load_root())
        return _AGG_BY_ROUTE[r]


def _poll_prometheus_into(agg: StateAggregator, route: str) -> None:
    """Pull a few recording-rule metrics and push them into the aggregator for this route."""
    base = _prom_base()
    p50 = parse_scalar_value(
        instant_query(base, "arcs:http_server_duration:p50_10m", timeout_s=5.0),
    )
    p99 = parse_scalar_value(
        instant_query(base, "arcs:http_server_duration:p99_10m", timeout_s=5.0),
    )
    rps = parse_scalar_value(
        instant_query(base, "arcs:policy:decisions_per_second", timeout_s=5.0),
    )
    ctx = policy_context_from_arcs_root(_load_root())
    apply_instant_metrics(
        agg,
        p50_ms=p50,
        p99_ms=p99,
        error_rate=None,
        rps=rps,
        cpu=None,
        memory=None,
        queue_depth=None,
        policy=ctx,
    )


def _apply_kafka_event(agg: StateAggregator, event: dict[str, Any]) -> None:
    """
    Apply one JSON object from Kafka.

    Expected shape (version 1):
      { "schema": "arcs.telemetry.v1", "route": "/bench", "latency_ms": 120.0, "ok": true }
    """
    if str(event.get("schema", "")) != "arcs.telemetry.v1":
        return
    now = time.time()
    lat = event.get("latency_ms")
    ok = event.get("ok")
    if lat is not None:
        with contextlib.suppress(TypeError, ValueError):
            agg.record_latency_ms(now, float(lat))
    if ok is not None:
        agg.record_request(now, ok=bool(ok))


def _kafka_consumer_loop() -> None:
    bootstrap = _kafka_bootstrap()
    if not bootstrap:
        return
    try:
        from kafka import KafkaConsumer
    except ImportError:
        logger.warning("kafka-python-ng not available; Kafka mode disabled")
        return

    topic = _kafka_topic()
    logger.info("Kafka consumer starting on %s topic=%s", bootstrap, topic)
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap.split(","),
        value_deserializer=lambda b: json.loads(b.decode("utf-8")),
        group_id=os.environ.get("KAFKA_GROUP_ID", "arcs-telemetry-bridge"),
        auto_offset_reset="earliest",
        enable_auto_commit=True,
    )
    while True:
        try:
            pack = consumer.poll(timeout_ms=2000)
            if not pack:
                continue
            for _tp, messages in pack.items():
                for msg in messages:
                    try:
                        event = msg.value
                        if not isinstance(event, dict):
                            raise TypeError("event not a dict")
                        route = str(event.get("route") or "/")
                        agg = _get_agg(route)
                        with _AGG_LOCK:
                            _apply_kafka_event(agg, event)
                        KAFKA_EVENTS_TOTAL.labels(topic=topic, status="ok").inc()
                    except (TypeError, ValueError, json.JSONDecodeError) as e:
                        logger.debug("skip kafka message: %s", e)
                        KAFKA_EVENTS_TOTAL.labels(topic=topic, status="error").inc()
        except Exception:
            logger.exception("kafka consumer loop error; sleeping before retry")
            time.sleep(2.0)


app = FastAPI(title="ARCS Telemetry Bridge", version="0.1.0")


@app.on_event("startup")
def _startup() -> None:
    _load_root()
    if _kafka_bootstrap():
        threading.Thread(target=_kafka_consumer_loop, name="kafka-consumer", daemon=True).start()


@app.get("/health/live", response_class=PlainTextResponse)
def health_live() -> str:
    return "ok"


@app.get("/metrics")
def metrics() -> PlainTextResponse:
    data = generate_latest()
    return PlainTextResponse(content=data.decode("utf-8"), media_type=CONTENT_TYPE_LATEST)


@app.get("/v1/state")
def v1_state(route: str = "/") -> Response:
    """
    Return the 12-number observation as plain comma-separated floats (same order as training).

    We always refresh Prometheus-backed slices on each call so the vector tracks the live cluster.
    """
    agg = _get_agg(route)
    with _AGG_LOCK:
        try:
            _poll_prometheus_into(agg, route)
        except OSError as e:
            logger.warning("prometheus poll failed: %s", e)
        vec = agg.build_vector()
    body = ",".join(str(float(x)) for x in vec.tolist())
    return Response(content=body, media_type="text/plain")


@app.post("/v1/kafka-test-publish")
def kafka_test_publish(payload: dict[str, Any]) -> dict[str, str]:
    """
    Dev-only helper: push one JSON event through the same parser as Kafka (no broker required).

    Useful for tests and demos; protect with auth if this port is ever public.
    """
    route = str(payload.get("route") or "/")
    agg = _get_agg(route)
    with _AGG_LOCK:
        _apply_kafka_event(agg, payload)
    return {"status": "accepted", "route": route}
