"""Bootstrap helpers: arch gate, checksums, downloads for llama-server / GGUF / head."""

from __future__ import annotations

import hashlib
import os
import platform
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from scripts import pins


def require_macos_arm64() -> None:
    system = platform.system()
    machine = platform.machine()
    if system != "Darwin" or machine not in ("arm64", "aarch64"):
        raise SystemExit(
            f"This bootstrap targets macOS arm64 (Apple Silicon); got {system}/{machine}."
        )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest: Path, expected_sha256: str | None = None) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, tmp)
    if expected_sha256:
        got = sha256_file(tmp)
        if got != expected_sha256:
            tmp.unlink(missing_ok=True)
            raise SystemExit(f"checksum mismatch for {dest.name}: got {got}, want {expected_sha256}")
    tmp.replace(dest)
    return dest


def ensure_llama_server(bin_dir: Path) -> Path:
    bin_dir.mkdir(parents=True, exist_ok=True)
    server = bin_dir / "llama-server"
    marker = bin_dir / f".{pins.LLAMA_ASSET}.sha256"
    dylib = bin_dir / "libllama-server-impl.dylib"
    if (
        server.is_file()
        and dylib.is_file()
        and marker.is_file()
        and marker.read_text().strip() == pins.LLAMA_SHA256
    ):
        print(f"Already present: {server}")
        return server

    tarball = bin_dir / pins.LLAMA_ASSET
    download_file(pins.LLAMA_URL, tarball, expected_sha256=pins.LLAMA_SHA256)

    with tempfile.TemporaryDirectory(prefix="llama-extract-") as td:
        with tarfile.open(tarball, "r:gz") as tf:
            tf.extractall(td)
        # Release layout: llama-b11272/{llama-server, *.dylib, ...}
        roots = [p for p in Path(td).iterdir() if p.is_dir()]
        src_root = roots[0] if len(roots) == 1 else Path(td)
        for path in src_root.iterdir():
            dest = bin_dir / path.name
            if path.is_dir():
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(path, dest)
            else:
                shutil.copy2(path, dest)
                if path.name == "llama-server":
                    dest.chmod(dest.stat().st_mode | 0o111)
        if not server.is_file():
            raise SystemExit(f"llama-server not found after extracting {tarball.name}")
        if not dylib.is_file():
            raise SystemExit(f"libllama-server-impl.dylib missing after extracting {tarball.name}")

    marker.write_text(pins.LLAMA_SHA256 + "\n")
    print(f"Installed {server}")
    return server


def ensure_gguf(models_dir: Path) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    dest = models_dir / pins.GGUF_FILE
    if dest.is_file() and dest.stat().st_size > 0:
        got = sha256_file(dest)
        if got == pins.GGUF_SHA256:
            print(f"Already present: {dest}")
            return dest
        print(f"checksum mismatch for existing {dest.name}; re-downloading")
        dest.unlink()
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise SystemExit("huggingface_hub required to download GGUF") from e
    print(f"Downloading {pins.GGUF_REPO}/{pins.GGUF_FILE}@{pins.GGUF_REVISION}")
    path = hf_hub_download(
        pins.GGUF_REPO,
        pins.GGUF_FILE,
        revision=pins.GGUF_REVISION,
        local_dir=str(models_dir),
    )
    path = Path(path)
    if path.resolve() != dest.resolve() and path.is_file():
        if not dest.exists():
            shutil.copy2(path, dest)
    if not dest.is_file():
        raise SystemExit(f"GGUF missing after download: {dest}")
    got = sha256_file(dest)
    if got != pins.GGUF_SHA256:
        raise SystemExit(f"checksum mismatch for {dest.name}: got {got}, want {pins.GGUF_SHA256}")
    return dest


def ensure_clm_head() -> str:
    from clm.heads import download

    path = download()
    print(f"CLM head: {path}")
    return path


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    root = Path(__file__).resolve().parents[1]
    require_macos_arm64()
    ensure_llama_server(root / "bin")
    ensure_gguf(root / "models")
    ensure_clm_head()
    print("Bootstrap complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
