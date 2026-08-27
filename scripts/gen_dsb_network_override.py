#!/usr/bin/env python3
"""
Regenerate infra/docker/docker-compose.dsb.override.yml from upstream DeathStarBench socialNetwork compose.

Run from the repository root after updating the submodule:

  python3 scripts/gen_dsb_network_override.py

This keeps every DSB service on the same Docker network as ARCS (arcs_arcs) without hand-editing 25+ names.
"""

from __future__ import annotations

from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
UPSTREAM = REPO / "third_party/deathstarbench/socialNetwork/docker-compose.yml"
OUT = REPO / "infra/docker/docker-compose.dsb.override.yml"

ENVOY_PATCH = """
  # When you merge this file for DeathStarBench, Envoy reads envoy-dsb.yaml and forwards to nginx-thrift.
  envoy:
    volumes:
      - ../envoy/envoy-dsb.yaml:/etc/envoy/envoy.yaml:ro
    depends_on:
      policy-service:
        condition: service_healthy
      nginx-thrift:
        condition: service_started
""".strip("\n")


def main() -> None:
    data = yaml.safe_load(UPSTREAM.read_text(encoding="utf-8"))
    services = sorted(data.get("services", {}).keys())
    lines = [
        "# Merged after upstream socialNetwork/docker-compose.yml: every DSB service joins the ARCS network.",
        "# The network name arcs_arcs matches Docker Compose project \"arcs\" + network key \"arcs\" "
        "from infra/docker/docker-compose.yml.",
        "# Regenerate with: python3 scripts/gen_dsb_network_override.py",
        "networks:",
        "  arcs:",
        "    external: true",
        "    name: arcs_arcs",
        "",
        "services:",
    ]
    for s in services:
        lines.append(f"  {s}:")
        lines.append("    networks:")
        lines.append("      - arcs")
    lines.append("")
    for line in ENVOY_PATCH.split("\n"):
        lines.append(line)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT} ({len(services)} DSB services + envoy patch)")


if __name__ == "__main__":
    main()
