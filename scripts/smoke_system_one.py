"""Smoke helpers and CLI for local CLM playground."""

from __future__ import annotations

import os
import sys
from typing import Any

import requests


def assert_structured_answers(payload: dict[str, Any]) -> None:
    answers = payload.get("answers") or {}
    if "u" in answers:
        assert "noul" in answers["u"], answers["u"]
    if "d" in answers:
        assert "choice" in answers["d"] and "probabilities" in answers["d"], answers["d"]
    if "f" in answers:
        assert "score" in answers["f"] and "probabilities" in answers["f"], answers["f"]
    # generic: at least one typed answer when keys unknown
    if not answers:
        raise AssertionError("no answers in payload")
    for _k, v in answers.items():
        if not isinstance(v, dict):
            raise AssertionError(f"answer not an object: {v!r}")
        if not any(x in v for x in ("noul", "choice", "score")):
            raise AssertionError(f"missing typed fields: {v!r}")


def assert_rank_not_uniform(ranked: list[dict[str, Any]], min_top_prob: float = 0.6) -> None:
    if not ranked:
        raise AssertionError("empty ranked list")
    top = float(ranked[0].get("prob", 0.0))
    if top < min_top_prob:
        raise AssertionError(f"top prob {top} < {min_top_prob}; ranking looks flat/wrong pooling?")


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    base = os.environ.get("CLM_BASE_URL", "http://127.0.0.1:8700").rstrip("/")
    session = requests.Session()

    so = {
        "state": "Customer: my invoice was charged twice and nobody answers the phone!",
        "questions": {
            "urgency": {"type": "noul", "instructions": "Is this urgent?"},
            "department": {
                "type": "choice",
                "instructions": "Which team should handle this?",
                "criteria": {
                    "billing": "Charges, invoices, refunds",
                    "technical": "Bugs and outages",
                },
            },
            "frustration": {
                "type": "score",
                "instructions": "How frustrated is the customer?",
                "criteria": ["Calm", "Frustrated", "Very angry"],
            },
        },
    }
    r = session.post(f"{base}/v1/systemone", json=so, timeout=300)
    r.raise_for_status()
    payload = r.json()
    assert_structured_answers(payload)
    print("systemone ok:", {k: list(v) for k, v in payload.get("answers", {}).items()})

    rank_body = {
        "context": "",
        "question": "What causes tides on Earth?",
        "answers": [
            "The Moon's gravitational pull.",
            "Photosynthesis in plants.",
            "Because the Earth is round.",
        ],
    }
    rr = session.post(f"{base}/v1/rank", json=rank_body, timeout=300)
    rr.raise_for_status()
    ranked = rr.json().get("ranked") or []
    assert_rank_not_uniform(ranked, min_top_prob=0.6)
    print("rank ok:", ranked)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
