"""Property-based checks that safeguards never return impossible numbers."""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from arcs_rl.safeguards.rules import PolicyAction, SafeguardConfig, SafeguardEngine


def _engine() -> SafeguardEngine:
    cfg = SafeguardConfig(
        max_retries=5,
        backoff_min=0.5,
        backoff_max=3.0,
        timeout_ms_min=100.0,
        timeout_ms_max=10000.0,
        max_policy_changes_per_route_per_minute=1000,
        circuit_breaker_error_rate_threshold=0.99,
        circuit_breaker_clear_error_rate_threshold=0.01,
        route_state_ttl_seconds=1e9,
        default_retry=2,
        default_backoff_multiplier=1.0,
        default_timeout_ms=2000.0,
    )
    return SafeguardEngine(cfg)


@given(
    retry=st.integers(min_value=-50, max_value=50),
    backoff=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    timeout=st.floats(min_value=-5000.0, max_value=50000.0, allow_nan=False, allow_infinity=False),
    err=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    t=st.floats(min_value=0.0, max_value=1e9, allow_nan=False, allow_infinity=False),
)
def test_outputs_stay_inside_bounds(
    retry: int,
    backoff: float,
    timeout: float,
    err: float,
    t: float,
) -> None:
    eng = _engine()
    d = eng.decide(
        route="hypo",
        suggested=PolicyAction(retry, backoff, timeout),
        error_rate=err,
        now_monotonic_s=t,
    )
    assert 0 <= d.action.retry <= 5
    assert 0.5 <= d.action.backoff_multiplier <= 3.0
    assert 100.0 <= d.action.timeout_ms <= 10000.0


@given(
    err=st.sampled_from([0.0, 0.1, 0.2]),
    t=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
)
def test_healthy_suggestions_stable_route(
    err: float,
    t: float,
) -> None:
    eng = _engine()
    ok = PolicyAction(retry=2, backoff_multiplier=1.0, timeout_ms=2000.0)
    d = eng.decide(route="stable", suggested=ok, error_rate=err, now_monotonic_s=t)
    # Not a hard guarantee (rate limit history matters), but for a fresh route this should be clean.
    assert d.frozen_active is False
