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

def test_assert_rank_requires_moon_winner():
    with pytest.raises(AssertionError):
        assert_rank_not_uniform([
            {"candidate": "Photosynthesis in plants.", "prob": 0.9},
            {"candidate": "The Moon's gravitational pull.", "prob": 0.05},
            {"candidate": "Because the Earth is round.", "prob": 0.05},
        ], min_top_prob=0.6, expected_substring="Moon")
