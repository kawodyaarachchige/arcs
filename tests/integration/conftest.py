"""
Shared rules for integration tests (the ones that talk to real Docker services).

They stay off by default so a normal `pytest` run stays quick and does not need Docker.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSE_FILE = REPO_ROOT / "infra" / "docker" / "docker-compose.yml"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip integration tests unless the operator opted in (or pytest -m integration forces)."""
    markexpr = getattr(config.option, "markexpr", "") or ""
    if markexpr and "integration" in markexpr:
        return
    skip_no_env = pytest.mark.skip(
        reason="Set ARCS_INTEGRATION=1 to run integration tests, or run: pytest -m integration",
    )
    env_ok = os.environ.get("ARCS_INTEGRATION", "").lower() in ("1", "true", "yes")
    for item in items:
        if "integration" in item.keywords and not env_ok:
            item.add_marker(skip_no_env)


@pytest.fixture(scope="module")
def compose_stack() -> None:
    """Bring the full Compose testbed up once; tear it down after the module finishes."""
    if shutil.which("docker") is None:
        pytest.skip("docker not on PATH")

    # CI can start Compose once in the workflow, then run these tests against the live stack.
    reuse = os.environ.get("ARCS_COMPOSE_ALREADY_UP", "").lower() in ("1", "true", "yes")
    if reuse:
        yield
        return

    subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE_FILE), "up", "-d", "--build"],
        cwd=REPO_ROOT,
        check=True,
        timeout=600,
    )

    # Wait until the same checks as the nightly workflow: echo + aggregator respond.
    deadline = time.time() + 180.0
    ok = False
    while time.time() < deadline:
        try:
            r1 = subprocess.run(
                ["curl", "-sf", "http://127.0.0.1:18001/health/live"],
                capture_output=True,
                timeout=5,
            )
            r2 = subprocess.run(
                ["curl", "-sf", "http://127.0.0.1:18003/aggregate"],
                capture_output=True,
                timeout=10,
            )
            if r1.returncode == 0 and r2.returncode == 0:
                ok = True
                break
        except (OSError, subprocess.TimeoutExpired):
            pass
        time.sleep(2.0)

    if not ok:
        subprocess.run(
            ["docker", "compose", "-f", str(COMPOSE_FILE), "logs", "--tail", "80"],
            cwd=REPO_ROOT,
        )
        pytest.fail("Compose stack did not become healthy in time")

    yield

    subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE_FILE), "down", "-v"],
        cwd=REPO_ROOT,
        timeout=120,
    )
