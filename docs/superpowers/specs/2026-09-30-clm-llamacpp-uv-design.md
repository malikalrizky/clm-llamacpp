# CLM + llama.cpp + UV (local playground)

Date: 2026-09-30  
Status: draft for review

## Intent

Run [Contrastive-LM/CLM](https://github.com/Contrastive-LM/CLM) locally as a **playground + typed System One API** (`noul` / `choice` / `score`), with:

- **Encoder:** llama.cpp only (no vLLM)
- **Python deps:** UV only (no pip)
- **Machine:** this Mac, Apple Silicon, Metal

Success: open `http://127.0.0.1:8700/`, ask typed questions, get structured answers. Fine-tuning and upstream evals are out of scope for v1.

## Approach

Thin vendor of upstream `src/clm` into this repo. Drop `vllm` from package dependencies. Bootstrap downloads a pinned macOS arm64 `llama-server` binary and a Qwen3-8B GGUF. `clm-serve` talks to llama.cpp over OpenAI-compatible `/v1/embeddings`.

Rejected alternatives:

- Homebrew for llama.cpp — harder to pin, not UV-managed
- Git submodule of full upstream — pulls train/eval/vLLM surface area
- Published `contrastive-lm` + UV override — possible, but vendoring makes pooling/API patches local

## Layout

```
clm-test/
  pyproject.toml          # UV project; local clm package; no vllm
  uv.lock
  README.md
  scripts/
    bootstrap.sh          # llama-server + GGUF (+ head if needed)
    serve.sh              # llama-server + clm-serve
    smoke_system_one.py   # optional local API smoke
  vendor/clm/             # vendored upstream serving package
  models/                 # gitignored; GGUF
  bin/                    # gitignored; llama-server
  docs/superpowers/specs/ # this design
```

Python dependencies for serve: `torch`, `fastapi`, `uvicorn`, `numpy`, `requests`. Optional: `huggingface_hub` for head/GGUF fetch. No `vllm`.

## Bootstrap

`scripts/bootstrap.sh` is idempotent:

1. Require macOS arm64; exit with a clear message otherwise.
2. Download a **pinned** Metal-capable `llama-server` release asset into `bin/llama-server`. Skip when present and checksum matches.
3. Download a **pinned** Qwen3-8B GGUF into `models/` (default mid quant, e.g. `Q4_K_M`; override via env for smaller/faster quants).
4. Ensure the CLM reference projection head is available (`clm-download` or equivalent into cache / `heads/`).

Python env: `uv sync` only. No pip.

Exact release URLs, asset names, and checksums are chosen at implementation time and pinned in the script (not left as “latest”).

## Runtime

`scripts/serve.sh` starts two processes:

1. **`bin/llama-server`** on `:8090`
   - Metal
   - OpenAI-compatible `/v1/embeddings`
   - GGUF from `models/`
   - Context and embedding pooling flags pinned to match CLM’s reference setup as closely as the chosen llama.cpp build allows (target: last-token / EOS-style pooling, max tokens **2048**)

2. **`uv run clm-serve`** on `:8700`
   - `--emb-url http://127.0.0.1:8090/v1/embeddings`
   - `--emb-model` set to the model name llama-server exposes
   - Playground UI enabled (default)
   - Projection heads on CPU or MPS if torch provides it; **encoder stays in llama.cpp**

`serve.sh` waits until the embedder responds before starting `clm-serve`.

## Errors

- Bootstrap fails fast on wrong arch, failed download, or checksum mismatch.
- If the embedder is down, `clm-serve` returns **502** (upstream behavior). Startup order + health wait reduce that during normal use.
- README notes first-load latency and memory pressure on 8B; point users at a lower-quant override.

## Testing

- Manual: playground — one `noul`, one `choice`, one `score`.
- Optional: `uv run python scripts/smoke_system_one.py` against `:8700`, assert structured answer fields exist.
- Do not claim numerical parity with upstream vLLM until measured. README states pooling/API mismatch as the main quality risk.

## Risks

| Risk | Mitigation |
|------|------------|
| llama.cpp pooling ≠ vLLM last-token pooling | Pin flags; smoke fixed prompts; patch vendored `clm` or server flags if rankings look flat/wrong |
| Upstream hard-depends on vLLM | Vendor serving code; omit vLLM from `pyproject.toml` |
| Binary / GGUF URLs drift | Pin versions + checksums in `bootstrap.sh` |
| Syncing upstream fixes | Manual re-vendor of `vendor/clm` when needed |

## Out of scope (v1)

- Fine-tuning, training data, eval harnesses
- Linux / NVIDIA / CPU-only Mac paths (may work later; not designed here)
- `llama-cpp-python` in-process embeddings
- Matching upstream latency or accuracy numbers

## Implementation notes (for the plan)

1. Vendor upstream `src/clm` serving tree; strip or ignore non-serve entrypoints that assume vLLM.
2. Author UV `pyproject.toml` + lockfile without vLLM.
3. Write `bootstrap.sh` / `serve.sh` with pinned assets.
4. README: `uv sync` → bootstrap → serve → open playground.
5. Smoke script + a short “if scores look wrong, check pooling” note.

No product code until this spec is approved and an implementation plan exists.
