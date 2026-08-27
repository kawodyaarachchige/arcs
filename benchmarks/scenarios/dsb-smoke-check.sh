#!/usr/bin/env bash
# Quick checks after `make compose-up-dsb`: Envoy answers and DeathStarBench nginx is reachable.
# Does not run load tests. Use synthetic x-arcs headers only — no real user data.
set -euo pipefail
curl -sf -o /dev/null -H 'x-arcs-route: /dsb-smoke' -H 'x-arcs-error-rate: 0' \
  'http://127.0.0.1:10000/' || {
  echo "Envoy → DSB smoke failed (is compose-up-dsb running and nginx-thrift healthy?)" >&2
  exit 1
}
echo "dsb-smoke-check: OK"
