#!/usr/bin/env bash
# Example: sign an exported TorchScript file with cosign (install cosign separately).
# Usage: COSIGN_PASSWORD=... ./scripts/sign_model_example.sh /path/to/model.pt
set -euo pipefail
FILE="${1:?path to .pt file}"
if ! command -v cosign >/dev/null 2>&1; then
  echo "Install cosign from https://docs.sigstore.dev/cosign/installation/ first." >&2
  exit 1
fi
cosign sign-blob --yes --output-signature "${FILE}.sig" --bundle "${FILE}.bundle" "${FILE}"
echo "Wrote ${FILE}.sig and ${FILE}.bundle — verify before loading in production."
