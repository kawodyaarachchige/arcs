"""Deterministic tests for the hybrid safeguard engine (clamps, freeze, rate limits)."""

from __future__ import annotations

from arcs_rl.safeguards.rules import (
    PolicyAction,
    SafeguardConfig,
    SafeguardEngine,
)


def _minimal_cfg(**overrides: object) -> SafeguardConfig:
    base = {
        "max_retries": 5,
        "backoff_min": 0.5,
        "backoff_max": 3.0,
        "timeout_ms_min": 100.0,
        "timeout_ms_max": 10000.0,
        "max_policy_changes_per_route_per_minute": 3,
        "circuit_breaker_error_rate_threshold": 0.9,
        "circuit_breaker_clear_error_rate_threshold": 0.2,
        "route_state_ttl_seconds": 3600.0,
        "default_retry": 2,
        "default_backoff_multiplier": 1.0,
        "default_timeout_ms": 2000.0,
    }
    base.update(overrides)
    return SafeguardConfig(**base)


def test_clamps_retry_timeout_backoff() -> None:
    eng = SafeguardEngine(_minimal_cfg())
    d = eng.decide(
        route="r1",
        suggested=PolicyAction(retry=99, backoff_multiplier=50.0, timeout_ms=2.0),
        error_rate=0.0,
        now_monotonic_s=0.0,
    )
    assert d.action.retry == 5
    assert d.action.backoff_multiplier == 3.0
    assert d.action.timeout_ms == 100.0
    assert "retry_clamped_to_max" in d.override_reasons
    assert "backoff_clamped_to_max" in d.override_reasons
    assert "timeout_clamped_to_min" in d.override_reasons


def test_freeze_holds_last_safe_until_clear() -> None:
    eng = SafeguardEngine(
        _minimal_cfg(
            circuit_breaker_error_rate_threshold=0.5,
            circuit_breaker_clear_error_rate_threshold=0.1,
        )
    )
    first = eng.decide(
        route="checkout",
        suggested=PolicyAction(retry=1, backoff_multiplier=1.0, timeout_ms=500.0),
        error_rate=0.0,
        now_monotonic_s=1.0,
    )
    assert first.frozen_active is False
    assert first.action.retry == 1

    bad = eng.decide(
        route="checkout",
        suggested=PolicyAction(retry=4, backoff_multiplier=2.0, timeout_ms=800.0),
        error_rate=0.9,
        now_monotonic_s=2.0,
    )
    assert bad.frozen_active is True
    # Last safe was retry=1; freeze keeps dials but still caps retry ≤ 1.
    assert bad.action.retry == 1
    assert bad.action.backoff_multiplier == first.action.backoff_multiplier
    assert bad.action.timeout_ms == first.action.timeout_ms

    mid = eng.decide(
        route="checkout",
        suggested=PolicyAction(retry=0, backoff_multiplier=0.5, timeout_ms=100.0),
        error_rate=0.3,
        now_monotonic_s=3.0,
    )
    assert mid.frozen_active is True
    assert mid.action.retry == 1

    ok = eng.decide(
        route="checkout",
        suggested=PolicyAction(retry=3, backoff_multiplier=1.5, timeout_ms=900.0),
        error_rate=0.05,
        now_monotonic_s=4.0,
    )
    assert ok.frozen_active is False
    assert ok.action.retry == 3


def test_freeze_caps_retry_when_last_safe_was_high() -> None:
    eng = SafeguardEngine(
        _minimal_cfg(
            circuit_breaker_error_rate_threshold=0.5,
            circuit_breaker_clear_error_rate_threshold=0.1,
        )
    )
    healthy = eng.decide(
        route="pay",
        suggested=PolicyAction(retry=3, backoff_multiplier=1.5, timeout_ms=1000.0),
        error_rate=0.0,
        now_monotonic_s=1.0,
    )
    assert healthy.action.retry == 3

    frozen = eng.decide(
        route="pay",
        suggested=PolicyAction(retry=5, backoff_multiplier=2.0, timeout_ms=2000.0),
        error_rate=1.0,
        now_monotonic_s=2.0,
    )
    assert frozen.frozen_active is True
    assert frozen.action.retry == 1
    assert "circuit_breaker_freeze_cap_retry" in frozen.override_reasons
    assert "circuit_breaker_freeze_hold_last_safe" in frozen.override_reasons


def test_rate_limit_blocks_rapid_changes() -> None:
    # One allowed change per rolling minute → the third distinct policy attempt should be blocked.
    eng = SafeguardEngine(_minimal_cfg(max_policy_changes_per_route_per_minute=1))
    route = "api"
    t = 10.0
    a1 = eng.decide(
        route=route,
        suggested=PolicyAction(0, 1.0, 1000.0),
        error_rate=0.0,
        now_monotonic_s=t,
    )
    assert "rate_limited_policy_changes" not in a1.override_reasons

    a2 = eng.decide(
        route=route,
        suggested=PolicyAction(1, 1.0, 1000.0),
        error_rate=0.0,
        now_monotonic_s=t + 1.0,
    )
    assert a2.action.retry == 1

    a3 = eng.decide(
        route=route,
        suggested=PolicyAction(2, 1.0, 1000.0),
        error_rate=0.0,
        now_monotonic_s=t + 2.0,
    )
    assert "rate_limited_policy_changes" in a3.override_reasons
    assert a3.action.retry == 1


def test_ttl_evicts_route_state() -> None:
    eng = SafeguardEngine(_minimal_cfg(route_state_ttl_seconds=5.0))
    eng.decide(
        route="gone",
        suggested=PolicyAction(1, 1.0, 1000.0),
        error_rate=0.0,
        now_monotonic_s=0.0,
    )
    assert "gone" in eng._routes  # noqa: SLF001 - intentional white-box check

    eng.decide(
        route="other",
        suggested=PolicyAction(1, 1.0, 1000.0),
        error_rate=0.0,
        now_monotonic_s=100.0,
    )
    assert "gone" not in eng._routes  # noqa: SLF001
