"""
Run k6 against Envoy, then snapshot a few Prometheus metrics.

Usage (from repository root):
  PYTHONPATH=. python -m benchmarks.run_experiment --config benchmarks/config/default.yaml

Before you run: start the Docker stack, point policy-service at the right YAML for each arm,
and restart the container between static vs adaptive. This script does not change Docker for you —
it only records which files belong to which arm so the report stays honest.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
from pathlib import Path
from typing import Any

import yaml

from benchmarks.harness.bench_config import load_benchmark_config
from benchmarks.harness.manifest import (
    try_git_sha,
    try_submodule_commit,
    utc_now_iso,
    write_manifest,
)
from benchmarks.harness.prometheus_api import instant_query, parse_scalar_value

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_yaml_raw(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        out = yaml.safe_load(f)
    if not isinstance(out, dict):
        msg = f"Expected a mapping in {path}"
        raise ValueError(msg)
    return out


def _build_k6_env(
    arm: str,
    load_cfg: dict[str, Any],
    obs_list: list[float] | None,
) -> dict[str, str]:
    """Map benchmark YAML + arm name into environment variables k6 reads."""
    env = os.environ.copy()
    env["TARGET_URL"] = str(load_cfg["target_url"])
    env["ARC_ROUTE"] = str(load_cfg["route_header"])
    env["ARC_ERROR_RATE"] = str(load_cfg.get("error_rate", 0.0))
    env["K6_EXECUTOR"] = str(load_cfg.get("executor", "ramping-vus")).strip()

    if arm == "adaptive" and obs_list is not None:
        # Fixed synthetic state vector — good for repeatability; not live cluster telemetry.
        env["ARC_OBS"] = ",".join(str(float(x)) for x in obs_list)
    else:
        env["ARC_OBS"] = ""

    ex = env["K6_EXECUTOR"]
    if ex == "constant-arrival-rate":
        env["K6_RATE"] = str(int(load_cfg["rate"]))
        env["K6_TIME_UNIT"] = str(load_cfg["time_unit"])
        env["K6_DURATION"] = f"{int(load_cfg['duration_seconds'])}s"
        env["K6_PRE_ALLOCATED_VUS"] = str(int(load_cfg["pre_allocated_vus"]))
        env["K6_MAX_VUS"] = str(int(load_cfg["max_vus"]))
    else:
        stages = load_cfg["stages"]
        env["K6_STAGES_JSON"] = json.dumps(stages)

    return env


def _run_k6(
    script_path: Path,
    summary_path: Path,
    log_path: Path,
    env: dict[str, str],
) -> int:
    """Run k6 and write JSON summary + captured stdout/stderr. Returns the process exit code."""
    k6 = shutil.which("k6")
    if not k6:
        print(
            "k6 not found on PATH. Install: https://k6.io/docs/getting-started/installation/",
            file=sys.stderr,
        )
        return 127

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        k6,
        "run",
        f"--summary-export={summary_path}",
        str(script_path),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_f:
        proc = subprocess.run(
            cmd,
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            cwd=REPO_ROOT,
            check=False,
        )
    return int(proc.returncode)


def _append_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Append one row per Prometheus query for this arm."""
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file()
    fieldnames = ("utc_iso", "arm", "query_name", "expr", "value")
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        for row in rows:
            w.writerow(row)


def run(
    config_path: Path,
    arms: list[str],
) -> Path:
    """Execute selected arms and return the output directory for this run."""
    cfg = load_benchmark_config(config_path)
    cfg_snapshot = _load_yaml_raw(config_path)

    run_id = utc_now_iso().replace(":", "").replace("+00:00", "Z")
    out_dir = (REPO_ROOT / cfg["output_dir"] / run_id).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    prom_base = str(cfg["prometheus"]["base_url"])
    load_cfg = cfg["load"]
    settle = float(cfg["post_load_settle_seconds"])
    queries = cfg["prometheus_queries"]
    policy_cfgs = cfg.get("policy_configs", {})
    obs = load_cfg.get("observation")
    obs_list: list[float] | None = (
        [float(x) for x in obs] if isinstance(obs, list) and len(obs) == 12 else None
    )

    script_path = REPO_ROOT / "benchmarks" / "k6" / "envoy_load.js"

    manifest_common: dict[str, Any] = {
        "kind": "arcs_benchmark_run",
        "started_utc": utc_now_iso(),
        "repo_root": str(REPO_ROOT),
        "git_sha_short": try_git_sha(REPO_ROOT),
        "deathstarbench_submodule_commit_short": try_submodule_commit(
            REPO_ROOT,
            "third_party/deathstarbench",
        ),
        "benchmark_config_path": str(config_path.resolve()),
        "benchmark_config": cfg_snapshot,
        "policy_configs_documentation": policy_cfgs,
        "note": (
            "Swap ARCS_CONFIG on policy-service and restart it between arms when comparing "
            "static YAML defaults vs TorchScript. This runner does not restart Docker for you."
        ),
    }
    metrics_csv = out_dir / "metrics.csv"
    arms_completed: list[str] = []

    for arm in arms:
        arm = arm.strip().lower()
        if arm not in ("static", "adaptive"):
            msg = f"Unknown arm {arm!r} (use static or adaptive)"
            raise ValueError(msg)

        t_arm_start = utc_now_iso()
        warmup = float(load_cfg["warmup_seconds"])
        if warmup > 0:
            time.sleep(warmup)

        env = _build_k6_env(arm, load_cfg, obs_list)
        summary_json = out_dir / f"k6_summary_{arm}.json"
        k6_log = out_dir / f"k6_log_{arm}.txt"
        code = _run_k6(script_path, summary_json, k6_log, env)

        if settle > 0:
            time.sleep(settle)

        rows: list[dict[str, Any]] = []
        ts = utc_now_iso()
        for q in queries:
            name = str(q["name"])
            expr = str(q["expr"])
            try:
                data = instant_query(prom_base, expr)
                val = parse_scalar_value(data)
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                rows.append(
                    {
                        "utc_iso": ts,
                        "arm": arm,
                        "query_name": name,
                        "expr": expr,
                        "value": f"ERROR:{type(e).__name__}",
                    },
                )
                continue

            rows.append(
                {
                    "utc_iso": ts,
                    "arm": arm,
                    "query_name": name,
                    "expr": expr,
                    "value": "" if val is None else str(val),
                },
            )

        _append_metrics_csv(metrics_csv, rows)

        arm_manifest = {
            **manifest_common,
            "arm": arm,
            "arm_started_utc": t_arm_start,
            "k6_exit_code": code,
            "k6_summary_path": str(summary_json.relative_to(REPO_ROOT)),
            "k6_log_path": str(k6_log.relative_to(REPO_ROOT)),
        }
        write_manifest(out_dir / f"manifest_{arm}.json", arm_manifest)
        arms_completed.append(arm)

    final = {
        **manifest_common,
        "finished_utc": utc_now_iso(),
        "output_dir": str(out_dir),
        "arms_completed": arms_completed,
        "metrics_csv": str(metrics_csv.relative_to(REPO_ROOT)),
    }
    write_manifest(out_dir / "manifest.json", final)

    print(f"Wrote results under: {out_dir}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run k6 load test + Prometheus snapshots.")
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "config" / "default.yaml",
        help="Path to benchmark YAML.",
    )
    parser.add_argument(
        "--arms",
        type=str,
        default="both",
        help="Comma list: static, adaptive, or both (runs in that order).",
    )
    args = parser.parse_args()

    raw = args.arms.lower().strip()
    if raw == "both":
        arms_list = ["static", "adaptive"]
    else:
        arms_list = [a.strip() for a in raw.split(",") if a.strip()]

    try:
        run(args.config.resolve(), arms_list)
    except urllib.error.URLError as e:
        print(f"Prometheus request failed: {e}", file=sys.stderr)
        raise SystemExit(1) from e
    except KeyboardInterrupt:
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
