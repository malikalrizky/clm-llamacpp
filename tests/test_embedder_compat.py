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
