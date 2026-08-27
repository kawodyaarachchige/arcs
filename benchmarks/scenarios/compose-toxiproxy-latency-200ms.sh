#!/usr/bin/env sh
# Adds 200 ms extra delay on the echo path through Toxiproxy (good range to try: about 50–500 ms).
# Needs: Docker Compose stack running and the "echo" proxy created (see toxiproxy-init in compose).

set -eu
TOXIPROXY_API="${TOXIPROXY_API:-http://127.0.0.1:8474}"

curl -sf -X POST "${TOXIPROXY_API}/proxies/echo/toxics" \
  -H 'Content-Type: application/json' \
  -d '{"name":"latency","type":"latency","attributes":{"latency":200}}'

echo "Latency toxic applied to proxy 'echo'."
