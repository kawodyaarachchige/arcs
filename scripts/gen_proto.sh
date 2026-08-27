#!/usr/bin/env bash
# Regenerate Python gRPC stubs from protos/. Run from repo root after editing .proto files.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT="$ROOT/arcs-rl/src"
mkdir -p "$OUT"
python3 -m grpc_tools.protoc \
  -I "$ROOT/protos" \
  --python_out="$OUT" \
  --grpc_python_out="$OUT" \
  --pyi_out="$OUT" \
  arcs/policy/v1/policy.proto
touch "$OUT/arcs/__init__.py" "$OUT/arcs/policy/__init__.py" "$OUT/arcs/policy/v1/__init__.py"
