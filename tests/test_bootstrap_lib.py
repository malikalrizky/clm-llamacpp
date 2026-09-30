import pytest
from pathlib import Path
from scripts.bootstrap_lib import require_macos_arm64, sha256_file, download_file

def test_require_macos_arm64_rejects_linux(monkeypatch):
    monkeypatch.setattr("scripts.bootstrap_lib.platform.system", lambda: "Linux")
    monkeypatch.setattr("scripts.bootstrap_lib.platform.machine", lambda: "arm64")
    with pytest.raises(SystemExit) as ei:
        require_macos_arm64()
    assert "macos" in str(ei.value).lower() or "darwin" in str(ei.value).lower()

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

def test_ensure_llama_server_installs_dylibs(tmp_path, monkeypatch):
    """Extracted install must keep Metal/server dylibs next to the binary."""
    from scripts import bootstrap_lib, pins
    # Pretend tarball already downloaded with checksum
    asset = tmp_path / pins.LLAMA_ASSET
    import tarfile, io, os
    inner = tmp_path / "build"
    inner.mkdir()
    (inner / "llama-server").write_bytes(b"#!/bin/sh\n")
    (inner / "llama-server").chmod(0o755)
    (inner / "libllama-server-impl.dylib").write_bytes(b"dylib")
    with tarfile.open(asset, "w:gz") as tf:
        tf.add(inner / "llama-server", arcname="llama-b11272/llama-server")
        tf.add(inner / "libllama-server-impl.dylib", arcname="llama-b11272/libllama-server-impl.dylib")
    monkeypatch.setattr(pins, "LLAMA_URL", "http://example/x")
    def fake_download(url, dest, expected_sha256=None):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(asset.read_bytes())
        return dest
    monkeypatch.setattr(bootstrap_lib, "download_file", fake_download)
    bin_dir = tmp_path / "bin"
    server = bootstrap_lib.ensure_llama_server(bin_dir)
    assert server.is_file()
    assert (bin_dir / "libllama-server-impl.dylib").is_file()
