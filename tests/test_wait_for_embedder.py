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
