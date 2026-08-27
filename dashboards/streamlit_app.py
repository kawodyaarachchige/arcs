"""
Read-only operator view: Prometheus snapshots plus optional replay shard stats.

Run from the repository root (``arcs-rl`` on ``PYTHONPATH`` or installed editable), for example::

    streamlit run dashboards/streamlit_app.py

This UI does not change policies, restart services, or send traffic. It only reads metrics and
local files.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import streamlit as st

# Repo package: keep imports lazy-friendly for tests that patch sys.modules.
from arcs_rl.config import load_config
from arcs_rl.monitoring.prometheus_query import instant_query, parse_scalar_value

# Policy service metrics use fixed names (see services/policy-service/metrics.py).
_POLICY_QUERIES: tuple[tuple[str, str], ...] = (
    ("sum(arcs_policy_decisions_total)", "Total policy decisions (all routes)"),
    ("sum(arcs_safeguard_overrides_total)", "Safeguard overrides (all routes/reasons)"),
    ("max(arcs_policy_freeze_active)", "Circuit-breaker freeze (1 = frozen somewhere)"),
)


def _headers_from_env(token_env: str) -> dict[str, str] | None:
    """
    If the user names an env var and it is set, attach a Bearer token (never hard-code secrets).
    """
    name = (token_env or "").strip()
    if not name:
        return None
    token = os.environ.get(name, "").strip()
    if not token:
        return None
    return {"Authorization": f"Bearer {token}"}


def _training_metric_exprs(metric_prefix: str) -> tuple[tuple[str, str], ...]:
    """
    Build PromQL for training/replay gauges using the same prefix as ``prometheus.metric_prefix``.

    Names match ``TrainingMetricsExporter`` in arcs_rl (prefix strips a trailing underscore first).
    """
    base = str(metric_prefix).rstrip("_")
    pfx = f"{base}_"
    return (
        (f"{pfx}training_reward_moving_avg", "Training reward (smoothed)"),
        (f"{pfx}replay_stored_transitions", "Replay transitions on disk"),
        (f"{pfx}replay_pending_transitions", "Replay rows waiting in RAM"),
    )


def _show_prometheus_panel(
    base_url: str,
    headers: dict[str, str] | None,
    metric_prefix: str | None,
) -> None:
    st.subheader("Prometheus (instant queries)")
    st.caption("Values are whatever Prometheus last scraped; empty means no series yet.")

    rows: list[tuple[str, str, str]] = []
    queries = list(_POLICY_QUERIES)
    if metric_prefix:
        queries.extend(_training_metric_exprs(metric_prefix))

    for expr, label in queries:
        try:
            raw = instant_query(base_url, expr, extra_headers=headers)
            val = parse_scalar_value(raw)
            disp = "—" if val is None else f"{val:.6g}"
        except OSError as e:
            disp = f"error: {e}"
        rows.append((label, expr, disp))

    st.table(
        {
            "Metric": [r[0] for r in rows],
            "Query": [r[1] for r in rows],
            "Value": [r[2] for r in rows],
        },
    )


def _replay_shard_summary(npz_path: Path) -> None:
    """Load one compressed shard and show shapes and simple column means (read-only)."""
    data = np.load(npz_path)
    st.write(f"**File:** `{npz_path}`")
    for key in sorted(data.files):
        arr = data[key]
        flat = np.asarray(arr, dtype=np.float64).ravel()
        mean = float(flat.mean()) if flat.size else float("nan")
        st.write(f"- `{key}`: shape {arr.shape}, mean **{mean:.6g}**")


def main() -> None:
    st.set_page_config(page_title="ARCS operator view", layout="wide")
    st.title("ARCS operator view")
    st.caption("Read-only: metrics and local replay files. No cluster control.")

    with st.sidebar:
        st.header("Connection")
        prom_url = st.text_input("Prometheus base URL", value="http://127.0.0.1:9090")
        token_env = st.text_input(
            "Bearer token env var (optional)",
            value="",
            help=(
                "Name an environment variable that holds a secret token; "
                "we never store it in the app."
            ),
        )
        st.header("Training metrics prefix")
        arcs_yaml = st.text_input(
            "Path to arcs YAML (optional)",
            value="",
            help="If set, we read prometheus.metric_prefix for training/replay metric names.",
        )
        manual_prefix = st.text_input(
            "Or type metric prefix manually",
            value="arcs_",
            help="Same value as prometheus.metric_prefix in your config (often arcs_).",
        )

    metric_prefix: str | None = None
    if arcs_yaml.strip():
        try:
            cfg = load_config(Path(arcs_yaml.strip()))
            metric_prefix = str(cfg["prometheus"]["metric_prefix"])
            st.success(f"Loaded prefix from YAML: `{metric_prefix}`")
        except (OSError, ValueError) as e:
            st.warning(f"Could not load YAML: {e}")
            metric_prefix = manual_prefix.strip() or None
    else:
        metric_prefix = manual_prefix.strip() or None

    headers = _headers_from_env(token_env)
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    st.info(f"Data source: **{prom_url}** — snapshot time: **{now}**")

    _show_prometheus_panel(prom_url, headers, metric_prefix)

    st.subheader("Replay shard preview (optional)")
    st.caption(
        "Pick a `.npz` file written by training replay storage. "
        "Nothing here is uploaded or modified."
    )
    shard_dir = st.text_input("Directory containing `.npz` shards", value="")
    if shard_dir.strip():
        root = Path(shard_dir.strip()).expanduser()
        if root.is_dir():
            shards = sorted(root.glob("*.npz"))
            if not shards:
                st.warning("No `.npz` files in that directory.")
            else:
                choice = st.selectbox("Shard file", options=shards, format_func=lambda p: p.name)
                if choice is not None:
                    _replay_shard_summary(choice)
        else:
            st.error("That path is not a directory.")


if __name__ == "__main__":
    main()
