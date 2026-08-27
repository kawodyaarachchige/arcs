"""
Turn metrics.csv from a benchmark run into a simple bar chart (optional).

Matplotlib - Import it only when run this script.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def _read_latest_by_arm(path: Path) -> dict[str, dict[str, float]]:
    """Group rows by arm and query_name; keep the last numeric value for each pair."""
    by_arm: dict[str, dict[str, float]] = defaultdict(dict)
    with path.open(encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            arm = row.get("arm", "")
            qn = row.get("query_name", "")
            raw = row.get("value", "")
            if raw.startswith("ERROR:") or raw == "":
                continue
            try:
                val = float(raw)
            except ValueError:
                continue
            by_arm[arm][qn] = val
    return dict(by_arm)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot benchmark metrics.csv (needs matplotlib).")
    parser.add_argument("metrics_csv", type=Path, help="Path to metrics.csv from a run.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="PNG path (default: same folder as CSV, name overview.png).",
    )
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit(
            "matplotlib is not installed. Install dev tools or: pip install matplotlib",
        ) from e

    data = _read_latest_by_arm(args.metrics_csv)
    if not data:
        raise SystemExit("No numeric rows found in CSV.")

    out = args.output or (args.metrics_csv.parent / "overview.png")

    # One subplot per metric name so different units never share a y-axis.
    names = sorted({k for arm in data.values() for k in arm})
    n = len(names)
    fig, axes = plt.subplots(n, 1, figsize=(8, 2.5 * max(1, n)), squeeze=False)
    arms = sorted(data.keys())
    for i, metric in enumerate(names):
        ax = axes[i][0]
        vals = [data[a].get(metric, 0.0) for a in arms]
        ax.bar(arms, vals)
        ax.set_title(metric)
        ax.set_ylabel("value")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
