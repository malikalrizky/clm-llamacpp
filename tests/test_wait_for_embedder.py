import pytest
from scripts.wait_for_embedder import wait_for_embedder

def test_wait_succeeds_when_health_ok(monkeypatch):
    seen = {}
    class R:
        status_code = 200
    def fake_get(url, timeout=None):
        seen["url"] = url
        return R()
    monkeypatch.setattr("scripts.wait_for_embedder.requests.get", fake_get)
    wait_for_embedder("http://127.0.0.1:8090", timeout_s=1)
    assert seen["url"].endswith("/health")

def test_wait_exits_on_timeout(monkeypatch):
    def boom(*a, **k):
        raise ConnectionError("nope")
    monkeypatch.setattr("scripts.wait_for_embedder.requests.get", boom)
    with pytest.raises(SystemExit):
        wait_for_embedder("http://127.0.0.1:8090", timeout_s=0.2)
