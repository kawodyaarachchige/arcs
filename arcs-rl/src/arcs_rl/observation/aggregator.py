"""
Builds the 12-number state vector from recent samples and the current policy settings.

Callers push measurements over time, then call `build_vector` at a decision time.
No randomness: same history and same `now` always produce the same vector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from arcs_rl.observation.features import STATE_DIM
from arcs_rl.observation.normalization import (
    min_max_unit,
    percentile_linear,
    z_score_to_unit,
)
from arcs_rl.observation.windows import TimedFloatBuffer, TimedRequestBuffer


@dataclass
class PolicyContext:
    """Retry/backoff/timeout the proxy is using for this route right now."""

    retry_count: int = 0
    backoff_multiplier: float = 1.0
    timeout_ms: float = 2000.0


class StateAggregator:
    """
    Collects telemetry and policy context, then outputs a fixed-length numpy vector.

    See `FEATURE_NAMES` for what each index means.
    """

    def __init__(self, observation_cfg: dict[str, Any], policy_cfg: dict[str, Any]) -> None:
        self._obs = observation_cfg
        self._policy = policy_cfg
        self._norm = observation_cfg["normalization"]
        self._missing = str(observation_cfg["missing_value_strategy"])
        lat_w = float(observation_cfg["latency_window_s"])
        err_w = float(observation_cfg["error_window_s"])
        # Latency percentiles use the shorter window from the spec.
        self._latency_buf = TimedFloatBuffer(window_s=lat_w)
        # Rolling means for load metrics use the same window so everything lines up.
        self._cpu_buf = TimedFloatBuffer(window_s=lat_w)
        self._mem_buf = TimedFloatBuffer(window_s=lat_w)
        self._queue_buf = TimedFloatBuffer(window_s=lat_w)
        self._rps_buf = TimedFloatBuffer(window_s=lat_w)
        self._request_buf = TimedRequestBuffer(window_s=err_w)
        self.policy_context = PolicyContext()
        self._clip_lo = float(self._norm["clip_min"])
        self._clip_hi = float(self._norm["clip_max"])
        self._mode = str(self._norm["mode"])
        self._lat_upper = float(self._norm["latency_ms_upper"])
        self._queue_cap = float(self._norm["queue_depth_cap"])
        self._ref_rps = float(observation_cfg["reference_rps"])
        self._z = self._norm.get("z_score") or {}

    def set_policy_context(
        self,
        *,
        retry_count: int | None = None,
        backoff_multiplier: float | None = None,
        timeout_ms: float | None = None,
    ) -> None:
        """Update the policy slice of the vector (what the mesh is doing today)."""
        if retry_count is not None:
            self.policy_context.retry_count = int(retry_count)
        if backoff_multiplier is not None:
            self.policy_context.backoff_multiplier = float(backoff_multiplier)
        if timeout_ms is not None:
            self.policy_context.timeout_ms = float(timeout_ms)

    def record_latency_ms(self, t: float, latency_ms: float) -> None:
        """Record one request’s latency so we can compute median and tail latency."""
        self._latency_buf.append(t, latency_ms)

    def record_request(self, t: float, *, ok: bool) -> None:
        """Record one attempt: ok=False means it counted as an error for error_rate."""
        self._request_buf.record(t, not ok)

    def record_load(self, t: float, *, cpu: float, memory: float, queue_depth: float) -> None:
        """
        Record resource usage. cpu and memory should be fractions between 0 and 1 (e.g. 0.73 = 73%).
        queue_depth is a raw count; we divide by `queue_depth_cap` from config.
        """
        self._cpu_buf.append(t, cpu)
        self._mem_buf.append(t, memory)
        self._queue_buf.append(t, queue_depth)

    def record_global_rps(self, t: float, rps: float) -> None:
        """How many requests per second across the system (or your chosen scope)."""
        self._rps_buf.append(t, rps)

    def raw_error_rate(self, now: float | None = None) -> float | None:
        """
        Fraction of failed attempts in the error window (0–1), or None if there is no data yet.

        This is the real error rate before normalization, useful for cascade-style penalties.
        """
        t_now = self._default_now() if now is None else float(now)
        return self._request_buf.error_rate(t_now)

    def _norm_latency_pair(self, p50: float | None, p99: float | None) -> tuple[float, float]:
        """Turn two latency numbers into normalized slots, handling missing data."""

        def one(raw: float | None) -> float:
            if raw is None:
                return self._missing_scalar(pessimistic_high=True)
            if self._mode == "z_score":
                m = float(self._z.get("latency_ms_mean", 200.0))
                s = float(self._z.get("latency_ms_std", 150.0))
                return z_score_to_unit(raw, m, s, self._clip_lo, self._clip_hi)
            return min_max_unit(raw, 0.0, self._lat_upper, self._clip_lo, self._clip_hi)

        return one(p50), one(p99)

    def _norm_scalar_min_max(
        self,
        raw: float | None,
        lo: float,
        hi: float,
        *,
        pessimistic_high: bool,
    ) -> float:
        if raw is None:
            return self._missing_scalar(pessimistic_high=pessimistic_high)
        if self._mode == "z_score":
            # z_score only defined in YAML for a subset; fall back to min_max for others.
            return min_max_unit(raw, lo, hi, self._clip_lo, self._clip_hi)
        return min_max_unit(raw, lo, hi, self._clip_lo, self._clip_hi)

    def _missing_scalar(self, *, pessimistic_high: bool) -> float:
        if self._missing == "pessimistic":
            return self._clip_hi if pessimistic_high else self._clip_lo
        # neutral: middle of the output range
        return (self._clip_lo + self._clip_hi) / 2.0

    def _norm_error_rate(self, rate: float | None) -> float:
        if rate is None:
            return self._missing_scalar(pessimistic_high=True)
        if self._mode == "z_score":
            m = float(self._z.get("error_rate_mean", 0.05))
            s = float(self._z.get("error_rate_std", 0.1))
            return z_score_to_unit(rate, m, s, self._clip_lo, self._clip_hi)
        return min_max_unit(rate, 0.0, 1.0, self._clip_lo, self._clip_hi)

    def _norm_global_rps(self, rps: float | None) -> float:
        if rps is None:
            return self._missing_scalar(pessimistic_high=True)
        if self._mode == "z_score":
            m = float(self._z.get("global_rps_mean", 1000.0))
            s = float(self._z.get("global_rps_std", 500.0))
            return z_score_to_unit(rps, m, s, self._clip_lo, self._clip_hi)
        return min_max_unit(rps, 0.0, self._ref_rps, self._clip_lo, self._clip_hi)

    def build_vector(self, now: float | None = None) -> np.ndarray:
        """
        Produce the 12-dimensional state at time `now` (seconds, same clock as samples).

        If `now` is None, use the latest sample time seen in any buffer so unit tests
        can skip managing a clock until they call `flush` logic.
        """
        t_now = self._default_now() if now is None else float(now)
        lat_vals = self._latency_buf.values(t_now)
        p50 = percentile_linear(lat_vals, 50.0)
        p99 = percentile_linear(lat_vals, 99.0)
        n_p50, n_p99 = self._norm_latency_pair(p50, p99)

        cpu_raw = self._cpu_buf.mean(t_now)
        mem_raw = self._mem_buf.mean(t_now)
        q_raw = self._queue_buf.mean(t_now)
        n_cpu = self._norm_scalar_min_max(cpu_raw, 0.0, 1.0, pessimistic_high=True)
        n_mem = self._norm_scalar_min_max(mem_raw, 0.0, 1.0, pessimistic_high=True)
        # Queue is always scaled by a fixed cap (simple and stable for both norm modes).
        n_queue = (
            self._missing_scalar(pessimistic_high=True)
            if q_raw is None
            else min_max_unit(
                q_raw,
                0.0,
                self._queue_cap,
                self._clip_lo,
                self._clip_hi,
            )
        )

        err = self._request_buf.error_rate(t_now)
        n_err = self._norm_error_rate(err)

        r_min = int(self._policy["retry"]["min"])
        r_max = int(self._policy["retry"]["max"])
        b_min = float(self._policy["backoff"]["min_multiplier"])
        b_max = float(self._policy["backoff"]["max_multiplier"])
        t_min = float(self._policy["timeout_ms"]["min"])
        t_max = float(self._policy["timeout_ms"]["max"])

        n_retry = min_max_unit(
            float(self.policy_context.retry_count),
            float(r_min),
            float(r_max),
            self._clip_lo,
            self._clip_hi,
        )
        n_back = min_max_unit(
            self.policy_context.backoff_multiplier,
            b_min,
            b_max,
            self._clip_lo,
            self._clip_hi,
        )
        n_tmo = min_max_unit(
            self.policy_context.timeout_ms,
            t_min,
            t_max,
            self._clip_lo,
            self._clip_hi,
        )

        rps_raw = self._rps_buf.mean(t_now)
        n_rps = self._norm_global_rps(rps_raw)

        vec = np.array(
            [
                n_cpu,
                n_mem,
                n_queue,
                n_p50,
                n_p99,
                n_err,
                n_retry,
                n_back,
                n_tmo,
                n_rps,
                0.0,
                0.0,
            ],
            dtype=np.float64,
        )
        assert vec.shape == (STATE_DIM,), vec.shape
        # Final safety: never hand NaN or Inf to downstream RL code.
        nan_fill = self._missing_scalar(pessimistic_high=True)
        vec = np.nan_to_num(
            vec,
            nan=nan_fill,
            posinf=self._clip_hi,
            neginf=self._clip_lo,
        )
        vec = np.clip(vec, self._clip_lo, self._clip_hi)
        return vec

    def _default_now(self) -> float:
        """Latest timestamp among buffered data, or 0.0 if everything is empty."""
        times: list[float] = []
        for buf in (
            self._latency_buf,
            self._cpu_buf,
            self._mem_buf,
            self._queue_buf,
            self._rps_buf,
            self._request_buf,
        ):
            ts = buf.latest_timestamp()
            if ts is not None:
                times.append(ts)
        return max(times) if times else 0.0


def aggregator_from_config(root: dict[str, Any]) -> StateAggregator:
    """Build an aggregator from the full parsed YAML root (needs `observation` + `policy`)."""
    return StateAggregator(observation_cfg=root["observation"], policy_cfg=root["policy"])


def build_state_vector(agg: StateAggregator, now: float | None = None) -> np.ndarray:
    """Small helper so imports read nicely: same as `agg.build_vector(now)`."""
    return agg.build_vector(now)
