"""
Feed a StateAggregator using numbers that came straight from Prometheus (or any instant query).

We reuse the same buffers and build_vector() as the simulator so training and serving stay aligned.
The bridge service polls Prometheus, pushes a few synthetic samples at “now”, then reads the vector.
"""

from __future__ import annotations

import math
import time
from typing import Any

from arcs_rl.observation.aggregator import PolicyContext, StateAggregator


def _upper_latency_ms(lo: float, p99_ms: float | None) -> float:
    """Pick a high-end latency sample; fall back to lo when p99 is missing or below lo."""
    if p99_ms is None or not math.isfinite(p99_ms):
        return lo
    p99f = float(p99_ms)
    return p99f if p99f >= lo else lo


def apply_instant_metrics(
    agg: StateAggregator,
    *,
    now: float | None = None,
    p50_ms: float | None = None,
    p99_ms: float | None = None,
    error_rate: float | None = None,
    rps: float | None = None,
    cpu: float | None = None,
    memory: float | None = None,
    queue_depth: float | None = None,
    policy: PolicyContext | None = None,
) -> None:
    """
    Push one snapshot into the aggregator at time `now` (seconds, wall clock is fine).

    Latency: we add a handful of fake samples between p50 and p99 so percentiles look sensible
    when both ends are present. If only one side exists, we repeat that value a few times.
    """
    t = time.time() if now is None else float(now)
    if policy is not None:
        agg.set_policy_context(
            retry_count=policy.retry_count,
            backoff_multiplier=policy.backoff_multiplier,
            timeout_ms=policy.timeout_ms,
        )

    if p50_ms is not None and math.isfinite(p50_ms) and p50_ms >= 0:
        lo = float(p50_ms)
        hi = _upper_latency_ms(lo, p99_ms)
        # Spread a few points across the range so p50/p99 inside the window are not identical noise.
        for i in range(5):
            frac = i / 4.0
            agg.record_latency_ms(t + i * 1e-6, lo + (hi - lo) * frac)

    if error_rate is not None and math.isfinite(error_rate):
        er = max(0.0, min(1.0, float(error_rate)))
        # Approximate error_rate with a tiny batch of Bernoulli-style outcomes (deterministic-ish).
        n = 20
        errors = int(round(er * n))
        for i in range(n):
            agg.record_request(t + 2e-6 * i, ok=(i >= errors))

    if rps is not None and math.isfinite(rps) and rps >= 0:
        agg.record_global_rps(t, float(rps))

    cu = 0.5 if cpu is None or not math.isfinite(cpu) else float(cpu)
    mu = 0.5 if memory is None or not math.isfinite(memory) else float(memory)
    qd = 0.0 if queue_depth is None or not math.isfinite(queue_depth) else float(queue_depth)
    agg.record_load(
        t,
        cpu=max(0.0, min(1.0, cu)),
        memory=max(0.0, min(1.0, mu)),
        queue_depth=max(0.0, qd),
    )


def policy_context_from_arcs_root(root: dict[str, Any]) -> PolicyContext:
    """Build PolicyContext from the policy section of the same YAML the policy service loads."""
    pol = root["policy"]
    return PolicyContext(
        retry_count=int(pol["retry"]["default"]),
        backoff_multiplier=float(pol["backoff"]["default_multiplier"]),
        timeout_ms=float(pol["timeout_ms"]["default"]),
    )
