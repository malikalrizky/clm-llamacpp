#!/usr/bin/env bash
# Start llama-server (embeddings) then clm-serve playground.
# llama.cpp b11272: --alias sets the OpenAI model name for /v1/embeddings.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SERVER="$ROOT/bin/llama-server"
GGUF_FILE="${CLM_GGUF_FILE:-Qwen3-8B-Q4_K_M.gguf}"
GGUF="$ROOT/models/$GGUF_FILE"
EMB_URL="http://127.0.0.1:8090"
EMB_MODEL="${CLM_EMB_MODEL:-qwen3-8b}"

if [[ ! -x "$SERVER" ]]; then
  echo "missing $SERVER — run ./scripts/bootstrap.sh first" >&2
  exit 1
fi
if [[ ! -f "$GGUF" ]]; then
  echo "missing $GGUF — run ./scripts/bootstrap.sh first" >&2
  exit 1
fi

LLAMA_PID=""
cleanup() {
  if [[ -n "${LLAMA_PID}" ]] && kill -0 "$LLAMA_PID" 2>/dev/null; then
    kill "$LLAMA_PID" 2>/dev/null || true
    wait "$LLAMA_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

"$SERVER" \
  -m "$GGUF" \
  --host 127.0.0.1 \
  --port 8090 \
  --embedding \
  --pooling last \
  --ctx-size 2048 \
  --ubatch-size 2048 \
  --batch-size 2048 \
  --parallel 1 \
  --embd-normalize 2 \
  --alias "$EMB_MODEL" \
  &
LLAMA_PID=$!
echo "llama-server pid=$LLAMA_PID"

uv run python -m scripts.wait_for_embedder "$EMB_URL" 180

uv run clm-serve \
  --host 127.0.0.1 \
  --port 8700 \
  --emb-url "${EMB_URL}/v1/embeddings" \
  --emb-model "$EMB_MODEL"
