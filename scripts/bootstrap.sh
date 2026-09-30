#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "This bootstrap targets macOS arm64 (Apple Silicon); got $(uname -s)/$(uname -m)." >&2
  exit 1
fi

uv sync --group dev --locked
uv run python -m scripts.bootstrap_lib
