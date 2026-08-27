"""
Names and order of the twelve numbers in the state vector.

Each slot ties back to the product spec: load, latency, errors, current policy knobs,
traffic level, and two spare slots for later ideas (e.g. how many downstream services).
The order here must match YAML `observation.feature_order` so training and deployment
stay in sync.
"""

from __future__ import annotations

# Fixed size required by the formal spec.
STATE_DIM = 12

# Index -> name. Position i in the vector is always FEATURE_NAMES[i].
FEATURE_NAMES: tuple[str, ...] = (
    "cpu_util",  # 0 — how busy the CPU is (normalized).
    "memory_util",  # 1 — how full memory is (normalized).
    "queue_depth",  # 2 — how backed up waiting work is (normalized).
    "latency_p50_ms",  # 3 — typical recent latency (median over the latency window).
    "latency_p99_ms",  # 4 — tail latency (roughly “almost worst case” in the window).
    "error_rate",  # 5 — fraction of requests that failed in the error window.
    "retry_count",  # 6 — how many retries the policy is using right now (scaled).
    "backoff_multiplier",  # 7 — how aggressive backoff is (scaled).
    "timeout_ms",  # 8 — how long we wait before giving up (scaled).
    "global_rps",  # 9 — how hard the system is being hit overall (scaled).
    "reserved_0",  # 10 — unused for now; keep zero unless you define it later.
    "reserved_1",  # 11 — same as reserved_0.
)
