"""Load and lightly validate benchmarks/config/*.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_benchmark_config(path: Path) -> dict[str, Any]:
    """Read a benchmark YAML file and return the root mapping."""
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        msg = f"Benchmark config root must be a mapping, got {type(data).__name__}"
        raise ValueError(msg)
    _validate_minimal(data)
    return data


def _validate_minimal(data: dict[str, Any]) -> None:
    """Fail fast on missing keys so runs do not half-start."""
    required = (
        "output_dir",
        "prometheus",
        "load",
        "prometheus_queries",
        "post_load_settle_seconds",
    )
    missing = [k for k in required if k not in data]
    if missing:
        msg = f"Benchmark config missing keys: {sorted(missing)}"
        raise ValueError(msg)
    if "base_url" not in data["prometheus"]:
        msg = "prometheus.base_url is required"
        raise ValueError(msg)
    load = data["load"]
    for k in ("target_url", "warmup_seconds", "duration_seconds", "route_header"):
        if k not in load:
            msg = f"load.{k} is required"
            raise ValueError(msg)
    ex = str(load.get("executor", "ramping-vus")).strip()
    if ex == "constant-arrival-rate":
        for k in ("rate", "time_unit", "pre_allocated_vus", "max_vus"):
            if k not in load:
                msg = f"load.{k} is required when executor is constant-arrival-rate"
                raise ValueError(msg)
    else:
        if "stages" not in load:
            msg = "load.stages is required when executor is ramping-vus"
            raise ValueError(msg)
