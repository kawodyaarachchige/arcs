"""Manifest helpers stay safe when optional submodules are missing."""

from __future__ import annotations

from pathlib import Path

from benchmarks.harness.manifest import try_submodule_commit

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_try_submodule_commit_returns_none_or_short_sha() -> None:
    sha = try_submodule_commit(REPO_ROOT, "third_party/deathstarbench")
    assert sha is None or (isinstance(sha, str) and 4 <= len(sha) <= 40)
