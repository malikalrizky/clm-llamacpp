# CLM + llama.cpp (UV)

Local typed System One playground for [Contrastive-LM/CLM](https://github.com/Contrastive-LM/CLM) on **macOS Apple Silicon**, using **llama.cpp** for embeddings and **UV** for Python. No vLLM.

## Requirements

- macOS arm64 (Apple Silicon)
- [UV](https://docs.astral.sh/uv/)
- ~6+ GB free disk (Qwen3-8B Q4_K_M GGUF ≈ 4.7 GB) and enough RAM/unified memory to load it

## Setup

```bash
uv sync --group dev
./scripts/bootstrap.sh   # downloads llama-server (b11272), GGUF, CLM head
./scripts/serve.sh       # :8090 embeddings + :8700 playground/API
```

Open [http://127.0.0.1:8700/](http://127.0.0.1:8700/).

Smoke (with serve running):

```bash
uv run python scripts/smoke_system_one.py
```

## Notes

- Encoder is llama.cpp (`--pooling last`, ctx 2048). Scores may not match upstream vLLM numbers if pooling or tokenization differ; if rankings look flat/near-uniform, check pooling flags first.
- Smaller/faster GGUF: `CLM_GGUF_FILE=… ./scripts/bootstrap.sh` (file must exist under `Qwen/Qwen3-8B-GGUF`).
- First load of the 8B GGUF is slow and memory-heavy.
- Python package is vendored under `vendor/clm` (see `vendor/UPSTREAM.md`).

## Tests

```bash
uv run pytest -v
```
