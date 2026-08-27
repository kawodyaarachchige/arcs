#!/usr/bin/env sh
# Removes all toxics from the echo proxy (return to “clean” network path).

set -eu
TOXIPROXY_API="${TOXIPROXY_API:-http://127.0.0.1:8474}"

for toxic in latency down timeout slow_close; do
  curl -sf -X DELETE "${TOXIPROXY_API}/proxies/echo/toxics/${toxic}" || true
done

echo "Cleared known toxics on proxy 'echo' (ignore errors if a toxic was absent)."
