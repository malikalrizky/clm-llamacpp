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
