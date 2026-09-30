from pathlib import Path
import re

def _dependency_lines() -> list[str]:
    text = Path("pyproject.toml").read_text()
    m = re.search(r"dependencies\s*=\s*\[(.*?)\]", text, re.S)
    assert m, "dependencies list missing"
    return re.findall(r'"([^"]+)"', m.group(1))

def test_pyproject_omits_vllm_dependency():
    deps = _dependency_lines()
    assert not any(re.match(r"(?i)vllm(\s|$|[><=!])", d) for d in deps)

def test_required_deps_listed():
    text = Path("pyproject.toml").read_text().lower()
    for dep in ("torch", "fastapi", "uvicorn", "numpy", "requests"):
        assert dep in text
