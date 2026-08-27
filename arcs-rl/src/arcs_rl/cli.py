"""
Command-line entry points: train, export, validate config, and evaluate saved models.

Keep flags thin; all defaults still live in the YAML file.
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


def _find_benchmark_repo_root() -> Path | None:
    """Locate the repository root that contains ``benchmarks/run_experiment.py``."""
    start = Path(__file__).resolve()
    for base in [start.parent, *start.parents]:
        candidate = base / "benchmarks" / "run_experiment.py"
        if candidate.is_file():
            return base
    return None


def _build_train_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="arcs-train",
        description="Train DQN or PPO with Stable-Baselines3.",
    )
    sub = p.add_subparsers(dest="command", required=True)
    off = sub.add_parser(
        "offline",
        help="Fill replay data if needed, then run DQN offline training.",
    )
    off.add_argument("-c", "--config", type=Path, required=True, help="Path to arcs YAML config.")
    off.add_argument(
        "-n",
        "--run-name",
        default="offline_run",
        help="Subfolder name under training.checkpoint_dir.",
    )
    on = sub.add_parser("online", help="Run online RL against the mock environment.")
    on.add_argument("-c", "--config", type=Path, required=True, help="Path to arcs YAML config.")
    on.add_argument(
        "-n",
        "--run-name",
        default="online_run",
        help="Subfolder name under training.checkpoint_dir.",
    )
    return p


def _build_export_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="arcs-export",
        description="Export a saved SB3 zip to TorchScript.",
    )
    sub = p.add_subparsers(dest="command", required=True)
    ts = sub.add_parser("torch", help="Write TorchScript plus a JSON sidecar.")
    ts.add_argument("model_zip", type=Path, help="Path to the .zip from training.")
    ts.add_argument("out", type=Path, help="Output path for the .ts file.")
    ts.add_argument(
        "-a",
        "--algorithm",
        choices=("dqn", "ppo"),
        default=None,
        help="Override when the file name does not contain dqn/ppo.",
    )
    return p


def _build_validate_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        prog="arcs-validate-config",
        description="Load an ARCS YAML file and exit 0 if every required section is present.",
    )


def _build_eval_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="arcs-eval",
        description=(
            "Evaluate a saved SB3 .zip on the mock microservice simulator (mean episode reward). "
            "This does not measure gRPC policy-server or Envoy latency."
        ),
    )
    p.add_argument("-c", "--config", type=Path, required=True, help="Path to arcs YAML config.")
    p.add_argument(
        "-m",
        "--model",
        type=Path,
        required=True,
        help="Path to the .zip produced by arcs-train.",
    )
    p.add_argument(
        "-n",
        "--episodes",
        type=int,
        default=5,
        help="How many full episodes to average (default: 5).",
    )
    p.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions from the policy instead of the greedy/deterministic choice.",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print paths and settings before the score.",
    )
    return p


def main_validate_config() -> None:
    parser = _build_validate_parser()
    parser.add_argument(
        "config",
        type=Path,
        help="Path to arcs YAML config.",
    )
    args = parser.parse_args()
    from arcs_rl.config import load_config, validate_config_keys

    try:
        data = load_config(args.config)
        validate_config_keys(data)
    except (OSError, ValueError) as e:
        print(f"Config invalid: {e}", file=sys.stderr)
        raise SystemExit(1) from e
    print(f"OK: {args.config} passes schema checks.")


def main_eval() -> None:
    parser = _build_eval_parser()
    args = parser.parse_args()
    from arcs_rl.config import load_config, validate_config_keys
    from arcs_rl.evaluation.sb3_eval import evaluate_saved_model

    data = load_config(args.config)
    validate_config_keys(data)
    if args.verbose:
        print(f"Config: {args.config.resolve()}")
        print(f"Model:  {args.model.resolve()}")
        print(f"Episodes: {args.episodes}, stochastic={args.stochastic}")
    mean_r, std_r = evaluate_saved_model(
        data,
        args.model,
        n_eval_episodes=args.episodes,
        deterministic=not args.stochastic,
    )
    print(f"Mean return: {mean_r:.4f} (+/- {std_r:.4f})")


def main_benchmark() -> None:
    """
    Delegate to ``benchmarks.run_experiment``: k6 + Prometheus snapshots.

    Expects a full source tree (with ``benchmarks/``). Does not start containers or install k6.
    """
    parser = argparse.ArgumentParser(
        prog="arcs-benchmark",
        description=(
            "Run the k6 load test and snapshot Prometheus metrics. Requires k6 on PATH, a "
            "reachable Prometheus in the benchmark YAML, and (for real traffic) the Docker stack "
            "already up. This tool only orchestrates the harness; it does not start services "
            "for you."
        ),
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=None,
        help="Benchmark YAML (default: benchmarks/config/default.yaml next to the repo root).",
    )
    parser.add_argument(
        "--arms",
        type=str,
        default="both",
        help="Comma list: static, adaptive, or both (same as run_experiment).",
    )
    args = parser.parse_args()
    root = _find_benchmark_repo_root()
    if root is None:
        print(
            "Could not find benchmarks/run_experiment.py — use a full ARCS checkout with "
            "benchmarks/.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    bench_cfg = args.config
    if bench_cfg is None:
        bench_cfg = (root / "benchmarks" / "config" / "default.yaml").resolve()
    else:
        bench_cfg = bench_cfg.resolve()

    root_s = str(root)
    prepended = root_s not in sys.path
    if prepended:
        sys.path.insert(0, root_s)
    old_argv = sys.argv.copy()
    try:
        sys.argv = [
            "run_experiment",
            "-c",
            str(bench_cfg),
            "--arms",
            args.arms,
        ]
        runpy.run_module("benchmarks.run_experiment", run_name="__main__")
    finally:
        sys.argv = old_argv
        if prepended:
            sys.path.remove(root_s)


def main_train() -> None:
    parser = _build_train_parser()
    args = parser.parse_args()
    from arcs_rl.config import load_config, validate_config_keys

    data = load_config(args.config)
    validate_config_keys(data)
    if args.command == "offline":
        from arcs_rl.training.offline import run_offline_training

        out = run_offline_training(data, run_name=args.run_name)
        print(f"Saved model: {out}")
    else:
        from arcs_rl.training.online import run_online_training

        out = run_online_training(data, run_name=args.run_name)
        print(f"Saved model: {out}")


def main_export() -> None:
    parser = _build_export_parser()
    args = parser.parse_args()
    if args.command == "torch":
        from arcs_rl.export.torch_export import export_torchscript

        path = export_torchscript(args.model_zip, args.out, algorithm=args.algorithm)
        print(f"Wrote {path} and metadata JSON.")


if __name__ == "__main__":
    main_train()
