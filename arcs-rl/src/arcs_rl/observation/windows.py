"""
Time-based buffers: we only keep samples newer than a sliding cutoff.

That way percentiles and error rates always match “the last N seconds” from the spec,
and the same input sequence always yields the same result (no randomness).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TimedFloatBuffer:
    """Stores (time, value) pairs and drops anything older than `window_s` seconds."""

    window_s: float
    _items: list[tuple[float, float]] = field(default_factory=list)

    def append(self, t: float, value: float) -> None:
        # Skip bad numbers so they never poison percentiles.
        if not np.isfinite(value):
            return
        self._items.append((t, float(value)))
        self._prune(t)

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_s
        if self._items and self._items[0][0] < cutoff:
            self._items = [x for x in self._items if x[0] >= cutoff]

    def values(self, now: float) -> np.ndarray:
        """All values still inside the window, in insertion order."""
        self._prune(now)
        if not self._items:
            return np.array([], dtype=np.float64)
        return np.array([v for _, v in self._items], dtype=np.float64)

    def mean(self, now: float) -> float | None:
        """Average of values in the window, or None if there are none."""
        arr = self.values(now)
        if arr.size == 0:
            return None
        return float(np.mean(arr))

    def latest_timestamp(self) -> float | None:
        """Time of the newest sample, if any."""
        if not self._items:
            return None
        return float(self._items[-1][0])


@dataclass
class TimedRequestBuffer:
    """
    One record per request attempt: did it error?

    Error rate is errors divided by attempts inside the window.
    """

    window_s: float
    _items: list[tuple[float, bool]] = field(default_factory=list)

    def record(self, t: float, is_error: bool) -> None:
        self._items.append((t, is_error))
        self._prune(t)

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_s
        if self._items and self._items[0][0] < cutoff:
            self._items = [x for x in self._items if x[0] >= cutoff]

    def error_rate(self, now: float) -> float | None:
        """Fraction of failed attempts, or None if there were no attempts."""
        self._prune(now)
        if not self._items:
            return None
        errors = sum(1 for _, err in self._items if err)
        return errors / len(self._items)

    def latest_timestamp(self) -> float | None:
        if not self._items:
            return None
        return float(self._items[-1][0])
