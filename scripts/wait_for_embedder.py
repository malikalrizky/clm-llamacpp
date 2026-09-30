"""Poll llama-server until /v1/models responds."""

from __future__ import annotations

import sys
import time

import requests


def wait_for_embedder(base_url: str, timeout_s: float = 120.0) -> None:
    url = base_url.rstrip("/") + "/v1/models"
    deadline = time.monotonic() + timeout_s
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            r = requests.get(url, timeout=2.0)
            if r.status_code == 200:
                return
            last_err = RuntimeError(f"status {r.status_code}")
        except Exception as e:  # noqa: BLE001 — poll until timeout
            last_err = e
        time.sleep(0.2)
    raise SystemExit(f"embedder not ready at {url} within {timeout_s}s: {last_err}")


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    base = argv[0] if argv else "http://127.0.0.1:8090"
    timeout = float(argv[1]) if len(argv) > 1 else 120.0
    wait_for_embedder(base, timeout)
    print(f"embedder ready: {base}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
