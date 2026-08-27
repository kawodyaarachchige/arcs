"""
This module is plain Python with no FastAPI imports so tests can check the math quickly.

"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class PolicyAction:
    """One retry/backoff/timeout triple (what Envoy or a client library would use)."""

    retry: int
    backoff_multiplier: float
    timeout_ms: float


@dataclass(frozen=True, slots=True)
class SafeguardConfig:
    """Loaded from YAML `safeguards:` plus defaults from `policy:` for first-time routes."""

    max_retries: int
    backoff_min: float
    backoff_max: float
    timeout_ms_min: float
    timeout_ms_max: float
    max_policy_changes_per_route_per_minute: int
    # When error_rate is at or above this, we “freeze” and keep the last healthy action.
    circuit_breaker_error_rate_threshold: float
    # When error_rate is at or below this, we can leave freeze mode
    # (must be < threshold for hysteresis).
    circuit_breaker_clear_error_rate_threshold: float
    # Drop per-route memory if we have not seen the route in this many seconds
    # (stops unbounded growth).
    route_state_ttl_seconds: float
    # Safe starting point when we have never seen a healthy decision for a route yet.
    default_retry: int
    default_backoff_multiplier: float
    default_timeout_ms: float

    @classmethod
    def from_arcs_config(cls, root: dict[str, Any]) -> SafeguardConfig:
        """Build from the top-level ARCS YAML mapping (same file the RL trainer uses)."""
        sg = root["safeguards"]
        pol = root["policy"]
        return cls(
            max_retries=int(sg["max_retries"]),
            backoff_min=float(sg["backoff_multiplier_bounds"][0]),
            backoff_max=float(sg["backoff_multiplier_bounds"][1]),
            timeout_ms_min=float(sg["timeout_ms_bounds"][0]),
            timeout_ms_max=float(sg["timeout_ms_bounds"][1]),
            max_policy_changes_per_route_per_minute=int(
                sg["max_policy_changes_per_route_per_minute"]
            ),
            circuit_breaker_error_rate_threshold=float(sg["circuit_breaker_error_rate_threshold"]),
            circuit_breaker_clear_error_rate_threshold=float(
                sg["circuit_breaker_clear_error_rate_threshold"]
            ),
            route_state_ttl_seconds=float(sg["route_state_ttl_seconds"]),
            default_retry=int(pol["retry"]["default"]),
            default_backoff_multiplier=float(pol["backoff"]["default_multiplier"]),
            default_timeout_ms=float(pol["timeout_ms"]["default"]),
        )


@dataclass(slots=True)
class RouteSafeguardState:
    """
    Memory for one route (for example `/api/checkout`).
    """

    last_applied: PolicyAction | None = None
    # Snapshot from the last time the route looked “healthy enough” to learn from.
    last_safe: PolicyAction | None = None
    frozen: bool = False
    # Timestamps (seconds, monotonic preferred) of moments the applied triple changed.
    change_times: deque[float] = field(default_factory=deque)
    last_seen_monotonic_s: float = 0.0


@dataclass(frozen=True, slots=True)
class SafeguardDecision:
    """What we return to the caller: final numbers + why we touched them."""

    action: PolicyAction
    override_reasons: tuple[str, ...]
    frozen_active: bool


def _clamp_action(cfg: SafeguardConfig, action: PolicyAction) -> tuple[PolicyAction, list[str]]:
    """Force retry/backoff/timeout into configured min/max ranges."""
    reasons: list[str] = []
    r = int(action.retry)
    if r < 0:
        reasons.append("retry_clamped_non_negative")
        r = 0
    if r > cfg.max_retries:
        reasons.append("retry_clamped_to_max")
        r = cfg.max_retries

    b = float(action.backoff_multiplier)
    if b < cfg.backoff_min:
        reasons.append("backoff_clamped_to_min")
        b = cfg.backoff_min
    elif b > cfg.backoff_max:
        reasons.append("backoff_clamped_to_max")
        b = cfg.backoff_max

    t = float(action.timeout_ms)
    if t < cfg.timeout_ms_min:
        reasons.append("timeout_clamped_to_min")
        t = cfg.timeout_ms_min
    elif t > cfg.timeout_ms_max:
        reasons.append("timeout_clamped_to_max")
        t = cfg.timeout_ms_max

    out = PolicyAction(retry=r, backoff_multiplier=b, timeout_ms=t)
    return out, reasons


def _default_action(cfg: SafeguardConfig) -> PolicyAction:
    """Conservative starting point when we have no history yet."""
    return PolicyAction(
        retry=cfg.default_retry,
        backoff_multiplier=cfg.default_backoff_multiplier,
        timeout_ms=cfg.default_timeout_ms,
    )


# While frozen under high error, do not keep burning high retry counts from last_safe.
_FREEZE_MAX_RETRY = 1


def _cap_retry_for_freeze(action: PolicyAction) -> tuple[PolicyAction, list[str]]:
    """Cap retries to at most 1 so freeze sheds load instead of repeating last_safe knocks."""
    if action.retry <= _FREEZE_MAX_RETRY:
        return action, []
    capped = PolicyAction(
        retry=_FREEZE_MAX_RETRY,
        backoff_multiplier=action.backoff_multiplier,
        timeout_ms=action.timeout_ms,
    )
    return capped, ["circuit_breaker_freeze_cap_retry"]


def _actions_equal(a: PolicyAction, b: PolicyAction) -> bool:
    return (
        a.retry == b.retry
        and abs(a.backoff_multiplier - b.backoff_multiplier) < 1e-9
        and abs(a.timeout_ms - b.timeout_ms) < 1e-9
    )


def _prune_changes(window_s: float, now_s: float, change_times: deque[float]) -> None:
    """Drop change timestamps older than one rolling minute window."""
    cutoff = now_s - window_s
    while change_times and change_times[0] < cutoff:
        change_times.popleft()


class SafeguardEngine:
    """
    Applies clamps, rate limits, and freeze rules.

    Keep this class small: callers pass monotonic time so tests do not depend on the real clock.
    """

    def __init__(self, cfg: SafeguardConfig) -> None:
        self._cfg = cfg
        self._routes: dict[str, RouteSafeguardState] = {}

    def _evict_stale_routes(self, now_mono_s: float) -> None:
        """Forget routes that have been quiet for a long time (bounded memory)."""
        ttl = self._cfg.route_state_ttl_seconds
        if ttl <= 0:
            return
        dead: list[str] = []
        for route, st in self._routes.items():
            if now_mono_s - st.last_seen_monotonic_s > ttl:
                dead.append(route)
        for r in dead:
            del self._routes[r]

    def decide(
        self,
        *,
        route: str,
        suggested: PolicyAction,
        error_rate: float,
        now_monotonic_s: float | None = None,
    ) -> SafeguardDecision:
        """
        Main entry: suggested action from the model, error_rate in [0, 1], route id for per-route
        state.

        `now_monotonic_s` should be `time.monotonic()` in prod; tests inject fixed values.
        """
        now = time.monotonic() if now_monotonic_s is None else float(now_monotonic_s)
        self._evict_stale_routes(now)

        st = self._routes.get(route)
        if st is None:
            st = RouteSafeguardState()
            self._routes[route] = st
        st.last_seen_monotonic_s = now

        cfg = self._cfg
        clamped, clamp_reasons = _clamp_action(cfg, suggested)
        reasons: list[str] = list(clamp_reasons)

        # If already in freeze mode, only exit when error looks clearly better (hysteresis).
        if st.frozen:
            if error_rate <= cfg.circuit_breaker_clear_error_rate_threshold:
                st.frozen = False
                reasons.append("circuit_breaker_cleared")
            else:
                # Still recovering: hold last safe dials but cap retries (shed load).
                hold = st.last_safe if st.last_safe is not None else _default_action(cfg)
                held, hold_reasons = _clamp_action(cfg, hold)
                reasons.extend(hold_reasons)
                held, cap_reasons = _cap_retry_for_freeze(held)
                reasons.extend(cap_reasons)
                reasons.append("circuit_breaker_still_frozen_hold_last_safe")
                st.last_applied = held
                return SafeguardDecision(
                    action=held,
                    override_reasons=tuple(reasons),
                    frozen_active=True,
                )

        # Fresh trip: error looks bad enough that stop trusting the model for a while.
        if error_rate >= cfg.circuit_breaker_error_rate_threshold:
            st.frozen = True
            hold = st.last_safe if st.last_safe is not None else _default_action(cfg)
            held, hold_reasons = _clamp_action(cfg, hold)
            reasons.extend(hold_reasons)
            held, cap_reasons = _cap_retry_for_freeze(held)
            reasons.extend(cap_reasons)
            reasons.append("circuit_breaker_freeze_hold_last_safe")
            st.last_applied = held
            return SafeguardDecision(
                action=held,
                override_reasons=tuple(reasons),
                frozen_active=True,
            )

        # Healthy enough to consider new behavior.
        # Rate limit *changes* to the applied policy triple (per route, rolling 60 seconds).
        window_s = 60.0
        _prune_changes(window_s, now, st.change_times)

        candidate = clamped
        if st.last_applied is not None and not _actions_equal(candidate, st.last_applied):
            # Would this push us over the per-minute change budget?
            if len(st.change_times) >= cfg.max_policy_changes_per_route_per_minute:
                reasons.append("rate_limited_policy_changes")
                candidate = st.last_applied
            else:
                st.change_times.append(now)

        st.last_applied = candidate
        st.last_safe = candidate
        return SafeguardDecision(
            action=candidate,
            override_reasons=tuple(reasons),
            frozen_active=False,
        )
