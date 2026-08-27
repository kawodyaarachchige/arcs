"""
Turn raw metrics into numbers roughly in [0, 1] so the learning code gets stable inputs.

We support two styles from config: min_max (scale between a low and high) and z_score
(compare to a typical mean and spread, then squash into range).
"""

from __future__ import annotations

import numpy as np


def clip_to_range(x: float, lo: float, hi: float) -> float:
    """Force a number to sit between lo and hi (inclusive)."""
    return float(np.clip(x, lo, hi))


def min_max_unit(v: float, v_min: float, v_max: float, clip_lo: float, clip_hi: float) -> float:
    """Linear scale from [v_min, v_max] into [clip_lo, clip_hi], then clip."""
    if v_max <= v_min:
        return clip_to_range((clip_lo + clip_hi) / 2.0, clip_lo, clip_hi)
    u = (v - v_min) / (v_max - v_min)
    out = clip_lo + u * (clip_hi - clip_lo)
    return clip_to_range(out, clip_lo, clip_hi)


def z_score_to_unit(
    v: float,
    mean: float,
    std: float,
    clip_lo: float,
    clip_hi: float,
) -> float:
    """
    How many standard deviations away from `mean`, squashed to about [0, 1].

    Values beyond ±3σ are treated as ±3σ so we do not explode when something weird happens.
    """
    s = max(float(std), 1e-9)
    t = (float(v) - float(mean)) / s
    t = float(np.clip(t, -3.0, 3.0))
    u = (t + 3.0) / 6.0
    out = clip_lo + u * (clip_hi - clip_lo)
    return clip_to_range(out, clip_lo, clip_hi)


def percentile_linear(values: np.ndarray, q: float) -> float | None:
    """Return the q-th percentile (q between 0 and 100), or None if there is no data."""
    if values.size == 0:
        return None
    return float(np.percentile(values, q, method="linear"))
