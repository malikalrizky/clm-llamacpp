"""Pinned download artifacts for local CLM + llama.cpp bootstrap."""

from __future__ import annotations

import os

LLAMA_RELEASE = "b11272"
LLAMA_ASSET = "llama-b11272-bin-macos-arm64.tar.gz"
LLAMA_URL = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_RELEASE}/{LLAMA_ASSET}"
LLAMA_SHA256 = "1ef6db9f1913725a9a7522f1719e987c88329f266e906270436b23d662985a20"

GGUF_REPO = "Qwen/Qwen3-8B-GGUF"
GGUF_FILE = os.environ.get("CLM_GGUF_FILE", "Qwen3-8B-Q4_K_M.gguf")
GGUF_REVISION = "7c41481f57cb95916b40956ab2f0b139b296d974"
GGUF_SHA256 = "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785"

EMB_HOST = "127.0.0.1"
EMB_PORT = 8090
CLM_PORT = 8700
EMB_MODEL = "qwen3-8b"
CTX_SIZE = 2048
