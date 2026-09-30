# CLM + llama.cpp + UV Local Playground Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Local typed System One playground (`noul` / `choice` / `score`) on macOS Apple Silicon using vendored CLM, UV-only Python deps, and llama.cpp as the only encoder.

**Architecture:** Vendor upstream `src/clm` into `vendor/clm` without vLLM. Bootstrap downloads a pinned Metal `llama-server` and Qwen3-8B GGUF. `serve.sh` runs llama-server (`:8090` embeddings, `--pooling last`) then `clm-serve` (`:8700`) pointed at that embedder. Small Python helpers under `scripts/` own arch/checksum/download logic so bootstrap is testable.

**Tech Stack:** Python ≥3.10, UV, torch (heads only), FastAPI/uvicorn, requests/numpy, llama.cpp `llama-server` (Metal), Qwen3-8B GGUF, CLM reference head from Hugging Face.

**Spec:** `docs/superpowers/specs/2026-09-30-clm-llamacpp-uv-design.md`

## Global Constraints

- Encoder: llama.cpp only — no vLLM installed or imported
- Python deps: UV only — no pip
- Target: macOS Apple Silicon (arm64) with Metal
- Ports: embedder `:8090`, CLM playground/API `:8700`
- Max tokens / ctx: **2048**
- Default GGUF: `Qwen/Qwen3-8B-GGUF` → `Qwen3-8B-Q4_K_M.gguf` (override via env)
- Pinned llama.cpp asset: release **b11272**, file `llama-b11272-bin-macos-arm64.tar.gz`, sha256 `1ef6db9f1913725a9a7522f1719e987c88329f266e906270436b23d662985a20`
- Upstream CLM pin: git tree `bb42c6c5bf914fd449bed2f6ca65be80602cb1f7` (main at design time)
- Out of scope: fine-tuning, evals, Linux/NVIDIA, `llama-cpp-python`

## Review Focus

- Non-arm64 or non-Darwin host → `bootstrap.sh` exits non-zero with a clear message (Task 4)
- Wrong or corrupted `llama-server` tarball → checksum mismatch exits before install (Task 4)
- Embedder not ready → `serve.sh` waits/fails; does not start `clm-serve` blind (Task 5)
- llama.cpp float embeddings vs upstream `encoding_format=base64` → vendored `Embedder._fetch` accepts float lists and does not require `truncate_prompt_tokens` (Task 3)
- Wrong pooling → smoke ranking is not near-uniform on a fixed easy triple (Task 6)

## File Structure

| Path | Responsibility |
|------|----------------|
| `pyproject.toml` / `uv.lock` | UV project; package root `vendor/`; deps without vLLM; scripts `clm-serve`, `clm-download` |
| `vendor/clm/**` | Vendored serving package (upstream `src/clm` at pinned SHA) |
| `vendor/UPSTREAM.md` | Upstream URL + SHA + what was omitted |
| `scripts/pins.py` | Pinned URLs, filenames, sha256, default ports/models |
| `scripts/bootstrap_lib.py` | `require_macos_arm64()`, `sha256_file()`, `download_file()`, extract helpers |
| `scripts/bootstrap.sh` | Thin shell wrapper calling UV + bootstrap_lib |
| `scripts/serve.sh` | Start llama-server then `uv run clm-serve` |
| `scripts/smoke_system_one.py` | Hit `:8700` with noul/choice/score (+ rank non-uniformity check) |
| `tests/test_bootstrap_lib.py` | Arch gate + checksum behavior |
| `tests/test_embedder_compat.py` | Embedder request/response shape for llama.cpp |
| `.gitignore` | `bin/`, `models/`, `.venv/`, `uv.lock` optional keep, caches, heads |
| `README.md` | `uv sync` → bootstrap → serve → open playground; pooling caveat |
| `bin/`, `models/` | Gitignored artifacts from bootstrap |

---

### Task 1: UV project skeleton (no vLLM)

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `vendor/clm/__init__.py` (minimal placeholder until Task 2)
- Test: `tests/test_project_constraints.py`

**Interfaces:**
- Consumes: nothing
- Produces: installable project name `clm-test` (or `contrastive-lm-local`); console scripts wired after Task 2; package discovery `where = ["vendor"]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_project_constraints.py
from pathlib import Path

def test_pyproject_omits_vllm():
    text = Path("pyproject.toml").read_text()
    assert "vllm" not in text.lower()

def test_required_deps_listed():
    text = Path("pyproject.toml").read_text().lower()
    for dep in ("torch", "fastapi", "uvicorn", "numpy", "requests"):
        assert dep in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_project_constraints.py -v`  
Expected: FAIL (missing `pyproject.toml` or deps)

- [ ] **Step 3: Implement `pyproject.toml` + `.gitignore`**

`pyproject.toml` requirements:
- `requires-python = ">=3.10"`
- dependencies exactly: `numpy>=1.24`, `requests>=2.28`, `torch>=2.1`, `fastapi>=0.100`, `uvicorn>=0.23`, `huggingface_hub>=0.20` (for head/GGUF fetch) — **no vllm, no pyarrow**
- `[tool.setuptools.packages.find] where = ["vendor"]`
- `[project.scripts]` `clm-serve = "clm.server:main"`, `clm-download = "clm.heads:download_main"` (will work after Task 2)
- Placeholder `vendor/clm/__init__.py` with `__version__ = "0.1.0"` so import works

`.gitignore`: `bin/`, `models/`, `.venv/`, `__pycache__/`, `*.pt`, `.cache/`, `checkpoints/`

- [ ] **Step 4: Run test to verify it passes**

Run: `uv sync && uv run pytest tests/test_project_constraints.py -v`  
Expected: PASS; `uv.lock` created; lockfile must not contain package name `vllm`

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock .gitignore vendor/clm/__init__.py tests/test_project_constraints.py
git commit -m "chore: scaffold UV project without vLLM"
```

---

### Task 2: Vendor upstream `src/clm`

**Files:**
- Create: `vendor/clm/{__init__.py,client.py,schema.py,embedder.py,engine.py,heads.py,cache.py,server.py,py.typed,static/*}`
- Create: `vendor/UPSTREAM.md`
- Test: `tests/test_vendor_import.py`

**Interfaces:**
- Consumes: Task 1 package layout
- Produces: `from clm import CLMClient, Noul, Choice, Score`; `clm-serve` / `clm-download` entry points

- [ ] **Step 1: Write the failing test**

```python
def test_clm_public_exports():
    from clm import CLMClient, Noul, Choice, Score
    assert CLMClient is not None
    assert Noul(instructions="x").instructions == "x"

def test_console_scripts_resolvable():
    from clm.server import main as serve_main
    from clm.heads import download_main
    assert callable(serve_main) and callable(download_main)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_vendor_import.py -v`  
Expected: FAIL on missing modules

- [ ] **Step 3: Copy upstream serving tree**

```bash
# from a temp clone at bb42c6c5bf914fd449bed2f6ca65be80602cb1f7
# copy only src/clm/** into vendor/clm/
```

Write `vendor/UPSTREAM.md` with repo URL, SHA, and note that train/eval/vLLM scripts were not vendored. Do not edit behavior yet (Task 3 patches embedder).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_vendor_import.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add vendor/ tests/test_vendor_import.py
git commit -m "chore: vendor CLM serving package at bb42c6c"
```

---

### Task 3: Embedder compatibility with llama.cpp

**Files:**
- Modify: `vendor/clm/embedder.py` (`Embedder._fetch`)
- Test: `tests/test_embedder_compat.py`

**Interfaces:**
- Consumes: `Embedder(url, model=..., max_tokens=2048)`
- Produces: `_fetch` that (1) sends `encoding_format: "float"` by default for llama.cpp, (2) omits `truncate_prompt_tokens` unless env `CLM_TRUNCATE_PROMPT_TOKENS=1`, (3) still accepts base64 string embeddings if returned, (4) L2-normalizes as today

- [ ] **Step 1: Write the failing test**

```python
import base64
import json
import numpy as np
from clm.embedder import Embedder, l2

class FakeResp:
    def __init__(self, payload, status=200):
        self.status_code = status
        self._payload = payload
        self.text = json.dumps(payload)
    def json(self):
        return self._payload

def test_fetch_sends_float_encoding_and_parses_float_list(monkeypatch):
    captured = {}
    emb = Embedder(url="http://example/v1/embeddings", model="qwen3-8b")
    def fake_post(url, json=None, timeout=None):
        captured["body"] = json
        vec = [0.0, 3.0, 4.0]  # L2 -> [0, 0.6, 0.8]
        return FakeResp({"data": [{"index": 0, "embedding": vec}], "usage": {"prompt_tokens": 2}})
    monkeypatch.setattr(emb.session, "post", fake_post)
    out, tokens = emb._fetch(["hi"])
    assert captured["body"]["encoding_format"] == "float"
    assert "truncate_prompt_tokens" not in captured["body"]
    assert tokens == 2
    np.testing.assert_allclose(out[0], l2(np.array([0.0, 3.0, 4.0], dtype=np.float32)))

def test_fetch_still_accepts_base64(monkeypatch):
    emb = Embedder(url="http://example/v1/embeddings")
    raw = np.array([1.0, 0.0, 0.0], dtype=np.float32).tobytes()
    def fake_post(url, json=None, timeout=None):
        return FakeResp({"data": [{"index": 0, "embedding": base64.b64encode(raw).decode()}], "usage": {}})
    monkeypatch.setattr(emb.session, "post", fake_post)
    out, _ = emb._fetch(["hi"])
    np.testing.assert_allclose(out[0], np.array([1.0, 0.0, 0.0], dtype=np.float32))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_embedder_compat.py -v`  
Expected: FAIL (`encoding_format` still `base64` and/or truncate present)

- [ ] **Step 3: Implement `Embedder._fetch` changes in `vendor/clm/embedder.py`**

Keep request URL and L2 behavior. Default body: `model`, `input`, `encoding_format="float"`. Only add `truncate_prompt_tokens` when explicitly enabled. Parse embedding as float list or base64 float32 buffer.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_embedder_compat.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add vendor/clm/embedder.py tests/test_embedder_compat.py
git commit -m "fix: make Embedder work with llama.cpp float embeddings"
```

---

### Task 4: Bootstrap pins + downloads

**Files:**
- Create: `scripts/pins.py`
- Create: `scripts/bootstrap_lib.py`
- Create: `scripts/bootstrap.sh`
- Test: `tests/test_bootstrap_lib.py`

**Interfaces:**
- Consumes: nothing from prior tasks except UV env for `huggingface_hub` / `clm-download`
- Produces:
  - `scripts.pins.LLAMA_RELEASE = "b11272"`
  - `scripts.pins.LLAMA_ASSET = "llama-b11272-bin-macos-arm64.tar.gz"`
  - `scripts.pins.LLAMA_URL` / `LLAMA_SHA256` as above
  - `scripts.pins.GGUF_REPO = "Qwen/Qwen3-8B-GGUF"`, `GGUF_FILE = "Qwen3-8B-Q4_K_M.gguf"` (env `CLM_GGUF_FILE` overrides)
  - `require_macos_arm64() -> None` raises `SystemExit` with message containing `macOS` and `arm64` on mismatch
  - `sha256_file(path: Path) -> str`
  - `download_file(url: str, dest: Path, expected_sha256: str | None) -> Path`
  - `ensure_llama_server(bin_dir: Path) -> Path` → `bin/llama-server` executable
  - `ensure_gguf(models_dir: Path) -> Path`
  - `ensure_clm_head() -> str` path via `clm.heads.download()`
  - `bootstrap.sh` calls `uv run python -m scripts.bootstrap_lib` (or `python scripts/bootstrap_lib.py`)

- [ ] **Step 1: Write the failing test**

```python
import pytest
from pathlib import Path
from scripts.bootstrap_lib import require_macos_arm64, sha256_file, download_file

def test_require_macos_arm64_rejects_linux(monkeypatch):
    monkeypatch.setattr("scripts.bootstrap_lib.platform.system", lambda: "Linux")
    monkeypatch.setattr("scripts.bootstrap_lib.platform.machine", lambda: "arm64")
    with pytest.raises(SystemExit) as ei:
        require_macos_arm64()
    assert "macOS" in str(ei.value).lower() or "darwin" in str(ei.value).lower()

def test_sha256_file(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"abc")
    assert sha256_file(p) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"

def test_download_file_checksum_mismatch(tmp_path, monkeypatch):
    dest = tmp_path / "out.bin"
    def fake_urlretrieve(url, filename):
        Path(filename).write_bytes(b"nope")
    monkeypatch.setattr("scripts.bootstrap_lib.urllib.request.urlretrieve", fake_urlretrieve)
    with pytest.raises(SystemExit):
        download_file("http://example/x", dest, expected_sha256="00" * 32)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_bootstrap_lib.py -v`  
Expected: FAIL (module missing)

- [ ] **Step 3: Implement `pins.py`, `bootstrap_lib.py`, `bootstrap.sh`**

`ensure_llama_server`: download tarball to `bin/` cache, verify sha256, extract, locate `llama-server` binary inside the archive, install to `bin/llama-server`, `chmod +x`. Skip download when `bin/llama-server` exists and optional marker hash file matches.

`ensure_gguf`: download via `huggingface_hub.hf_hub_download` into `models/` (or curl the resolve URL). Default file `Qwen3-8B-Q4_K_M.gguf`; env override for smaller quants later.

`ensure_clm_head`: call `clm.heads.download()`.

`bootstrap.sh`: `set -euo pipefail`; `cd` repo root; `uv sync`; `uv run python -m scripts.bootstrap_lib`.

Make `scripts` importable: empty `scripts/__init__.py` or run as file with path hacks — prefer package-style `python -m scripts.bootstrap_lib` with `scripts/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_bootstrap_lib.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/ tests/test_bootstrap_lib.py
git commit -m "feat: bootstrap llama-server, GGUF, and CLM head"
```

---

### Task 5: `serve.sh` runtime

**Files:**
- Create: `scripts/serve.sh`
- Create: `scripts/wait_for_embedder.py`
- Test: `tests/test_wait_for_embedder.py`

**Interfaces:**
- Consumes: `bin/llama-server`, GGUF path from pins/env, `uv run clm-serve`
- Produces:
  - `wait_for_embedder(base_url: str, timeout_s: float) -> None` polls `GET {base}/v1/models` until 200 or raises `SystemExit`
  - `serve.sh` starts:
    ```
    bin/llama-server -m models/<gguf> --host 127.0.0.1 --port 8090 \
      --embedding --pooling last --ctx-size 2048 --embd-normalize 2
    ```
    then after wait:
    ```
    uv run clm-serve --emb-url http://127.0.0.1:8090/v1/embeddings --emb-model qwen3-8b
    ```
  - Pass `--alias qwen3-8b` / `-a qwen3-8b` to llama-server if the build supports naming the model to match `--emb-model`; otherwise set `--emb-model` to whatever `/v1/models` returns (document the chosen flag in the script comment once verified against b11272)
  - Trap EXIT to kill llama-server child

- [ ] **Step 1: Write the failing test**

```python
import pytest
from scripts.wait_for_embedder import wait_for_embedder

def test_wait_succeeds_when_models_ok(monkeypatch):
    class R:
        status_code = 200
    monkeypatch.setattr("scripts.wait_for_embedder.requests.get", lambda *a, **k: R())
    wait_for_embedder("http://127.0.0.1:8090", timeout_s=1)

def test_wait_exits_on_timeout(monkeypatch):
    def boom(*a, **k):
        raise ConnectionError("nope")
    monkeypatch.setattr("scripts.wait_for_embedder.requests.get", boom)
    with pytest.raises(SystemExit):
        wait_for_embedder("http://127.0.0.1:8090", timeout_s=0.2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_wait_for_embedder.py -v`  
Expected: FAIL

- [ ] **Step 3: Implement `wait_for_embedder.py` and `serve.sh`**

Keep flags aligned with Review Focus (pooling last, ctx 2048). Log PIDs. Do not start `clm-serve` until wait succeeds.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_wait_for_embedder.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/serve.sh scripts/wait_for_embedder.py tests/test_wait_for_embedder.py
git commit -m "feat: serve llama-server embeddings and clm-serve playground"
```

---

### Task 6: Smoke script + README

**Files:**
- Create: `scripts/smoke_system_one.py`
- Create: `README.md`
- Test: `tests/test_smoke_helpers.py` (pure helpers only — live smoke is manual/integration)

**Interfaces:**
- Consumes: running server at `CLM_BASE_URL` default `http://127.0.0.1:8700`
- Produces:
  - `assert_structured_answers(payload: dict) -> None` checks noul/choice/score fields
  - `assert_rank_not_uniform(ranked: list[dict], min_top_prob: float = 0.6) -> None`
  - `smoke_system_one.py` main: POST `/v1/systemone` with one of each question type; POST `/v1/rank` with tides/moon triple from upstream README; exit 0 only if asserts pass
  - README: prerequisites (arm64 Mac), `uv sync`, `./scripts/bootstrap.sh`, `./scripts/serve.sh`, open `http://127.0.0.1:8700/`, smoke command, note that scores may diverge from vLLM if pooling mismatches, env `CLM_GGUF_FILE` for smaller quants, memory/first-load caveat

- [ ] **Step 1: Write the failing test**

```python
import pytest
from scripts.smoke_system_one import assert_structured_answers, assert_rank_not_uniform

def test_assert_structured_answers_ok():
    assert_structured_answers({
        "answers": {
            "u": {"noul": 0.4},
            "d": {"choice": "billing", "probabilities": {"billing": 0.9, "tech": 0.1}},
            "f": {"score": 1.5, "probabilities": {"0": 0.1, "1": 0.4, "2": 0.5}},
        }
    })

def test_assert_rank_rejects_uniform():
    with pytest.raises(AssertionError):
        assert_rank_not_uniform([
            {"candidate": "a", "prob": 0.34},
            {"candidate": "b", "prob": 0.33},
            {"candidate": "c", "prob": 0.33},
        ], min_top_prob=0.6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_smoke_helpers.py -v`  
Expected: FAIL

- [ ] **Step 3: Implement smoke helpers, CLI, README**

Match upstream playground question shapes. Rank prompt: context/question empty or simple; answers = Moon / Photosynthesis / Earth round (same as upstream README).

- [ ] **Step 4: Run unit tests**

Run: `uv run pytest tests/ -v`  
Expected: all PASS

- [ ] **Step 5: Manual integration (after bootstrap on the Mac)**

Run:
```bash
./scripts/bootstrap.sh   # large download; once
./scripts/serve.sh       # separate terminal
uv run python scripts/smoke_system_one.py
```
Expected: smoke exit 0; playground loads at `:8700`

- [ ] **Step 6: Commit**

```bash
git add scripts/smoke_system_one.py tests/test_smoke_helpers.py README.md
git commit -m "docs: README and smoke checks for local playground"
```

---

## Self-review notes

- Spec coverage: layout, UV/no vLLM, vendor, bootstrap pins, serve, errors/wait, smoke, README pooling risk — each has a task.
- Pinned llama.cpp b11272 + sha256 and GGUF defaults are explicit for Task 4/5.
- Embedder float/truncate mismatch covered in Task 3 + Review Focus.
- No train/eval/vLLM paths introduced.
