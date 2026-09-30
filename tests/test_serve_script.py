from pathlib import Path

def test_serve_sh_binds_localhost_and_sets_ubatch():
    text = Path("scripts/serve.sh").read_text()
    assert "--host 127.0.0.1" in text
    assert "clm-serve" in text
    # clm-serve must also bind localhost
    assert "--host 127.0.0.1" in text.split("clm-serve", 1)[1] or text.count("--host 127.0.0.1") >= 2
    assert "-ub 2048" in text or "--ubatch-size 2048" in text
    assert "-b 2048" in text or "--batch-size 2048" in text
    assert "--parallel 1" in text or "-np 1" in text
    assert "exec uv run clm-serve" not in text

def test_bootstrap_sh_checks_arch_before_uv_sync():
    text = Path("scripts/bootstrap.sh").read_text()
    arch_pos = text.find("arm64")
    sync_pos = text.find("uv sync")
    assert arch_pos != -1 and sync_pos != -1 and arch_pos < sync_pos
    assert "--locked" in text
