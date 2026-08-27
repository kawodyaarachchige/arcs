"""Write JSON run manifests so results stay traceable (what config, when, which git revision)."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    """Return an ISO-8601 timestamp in UTC (timezone-aware)."""
    return datetime.now(UTC).isoformat()


def try_submodule_commit(repo_root: Path, submodule_path: str) -> str | None:
    """Return short git commit for a submodule checkout, if present."""
    sub = repo_root / submodule_path
    if not sub.is_dir():
        return None
    try:
        out = subprocess.run(
            ["git", "-C", str(sub), "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
        if out.returncode != 0:
            return None
        line = (out.stdout or "").strip()
        return line or None
    except OSError:
        return None


def try_git_sha(repo_root: Path) -> str | None:
    """Return short git commit hash if we are inside a git checkout, else None."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
        if out.returncode != 0:
            return None
        line = (out.stdout or "").strip()
        return line or None
    except OSError:
        return None


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    """Write pretty-printed JSON (UTF-8) for humans and scripts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True)
    path.write_text(text + "\n", encoding="utf-8")
