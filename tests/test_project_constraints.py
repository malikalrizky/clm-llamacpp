from pathlib import Path

def test_pyproject_omits_vllm():
    text = Path("pyproject.toml").read_text()
    assert "vllm" not in text.lower()

def test_required_deps_listed():
    text = Path("pyproject.toml").read_text().lower()
    for dep in ("torch", "fastapi", "uvicorn", "numpy", "requests"):
        assert dep in text
