"""
Turn training and replay numbers into Prometheus metrics (only when you enable this in YAML).

Keep label sets tiny on purpose: lots of different label values would make Prometheus heavy
and harder to run on a laptop.
"""

from __future__ import annotations

from typing import Any

from prometheus_client import REGISTRY, CollectorRegistry, Counter, Gauge, start_http_server


def _metric_name(prefix: str, suffix: str) -> str:
    """Build one legal metric name from the YAML prefix plus a short suffix."""
    base = prefix.rstrip("_")
    return f"{base}_{suffix}"


class TrainingMetricsExporter:
    """
    Holds gauges and counters for replay files and training progress.

    Call :meth:`start_http_server` once so Prometheus (or Grafana) can scrape the process.
    """

    def __init__(
        self,
        config: dict[str, Any],
        *,
        registry: CollectorRegistry | None = None,
    ) -> None:
        pc = config["prometheus"]
        self._prefix = str(pc["metric_prefix"])
        p = self._prefix
        # Tests pass a fresh registry so metrics never clash across test cases in one process.
        self._registry = registry if registry is not None else REGISTRY

        # Transitions already written to .npz files (excludes rows still waiting in RAM).
        self._replay_stored = Gauge(
            _metric_name(p, "replay_stored_transitions"),
            "Transitions already written into .npz shards on disk.",
            registry=self._registry,
        )
        # How many transitions are waiting in memory for the next shard write.
        self._replay_pending = Gauge(
            _metric_name(p, "replay_pending_transitions"),
            "Transitions buffered in RAM before the next flush to disk.",
            registry=self._registry,
        )
        # Counts each time we finish writing one shard file (helps spot write churn).
        self._replay_flushes = Counter(
            _metric_name(p, "replay_shard_flushes_total"),
            "How many shard files were written.",
            registry=self._registry,
        )
        # When the rolling window deletes an old shard to make room, we count it here.
        self._replay_removed = Counter(
            _metric_name(p, "replay_shards_removed_total"),
            "How many old shard files were deleted to stay under max_shards.",
            registry=self._registry,
        )
        # One label only: dqn or ppo — keeps cardinality at two time series.
        self._actions = Counter(
            _metric_name(p, "training_actions_total"),
            "Environment steps taken during learning (one algorithm label).",
            labelnames=("algorithm",),
            registry=self._registry,
        )
        # Smoothed reward so a graph is readable; not used for math inside the learner.
        self._reward_ema = Gauge(
            _metric_name(p, "training_reward_moving_avg"),
            "Exponential moving average of recent step rewards (for dashboards).",
            registry=self._registry,
        )
        self._reward_ema_value: float | None = None
        # How strongly new rewards pull the average (higher = react faster, more jitter).
        self._reward_ema_alpha = 0.02

    def start_http_server(self, config: dict[str, Any]) -> None:
        """Listen for HTTP scrapes (same idea as the policy service /metrics endpoint)."""
        pc = config["prometheus"]
        port = int(pc["scrape_port"])
        addr = str(pc["bind_address"])
        # prometheus_client serves /metrics on this port in a background thread.
        start_http_server(port, addr=addr, registry=self._registry)

    def sync_replay_from_storage(self, storage: Any) -> None:
        """Push current replay file counts to Prometheus."""
        self._replay_stored.set(storage.total_transitions)
        self._replay_pending.set(storage.pending_count)

    def on_replay_flush(self) -> None:
        """Call after writing a shard file."""
        self._replay_flushes.inc()

    def on_replay_shard_removed(self) -> None:
        """Call when an old shard file is deleted."""
        self._replay_removed.inc()

    def record_env_step(self, *, algorithm: str, reward: float) -> None:
        """Count one action and update the reward moving average."""
        algo = algorithm.lower()
        self._actions.labels(algorithm=algo).inc()
        if self._reward_ema_value is None:
            self._reward_ema_value = float(reward)
        else:
            a = self._reward_ema_alpha
            self._reward_ema_value = (1.0 - a) * self._reward_ema_value + a * float(reward)
        self._reward_ema.set(self._reward_ema_value)


def maybe_start_training_metrics(config: dict[str, Any]) -> TrainingMetricsExporter | None:
    """
    If YAML turns metrics on, build the exporter and open the HTTP scrape port.

    If metrics stay off, return None so training stays quiet (good for CI and one-off scripts).
    """
    if not bool(config["prometheus"]["enabled"]):
        return None
    out = TrainingMetricsExporter(config)
    out.start_http_server(config)
    return out
