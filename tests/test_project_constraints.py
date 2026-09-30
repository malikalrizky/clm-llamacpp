from pathlib import Path
import re
import tomllib

def test_pyproject_omits_vllm_dependency():
    data = tomllib.loads(Path("pyproject.toml").read_text())
    deps = data["project"]["dependencies"]
    assert not any(re.match(r"(?i)vllm(\s|$|[><=!])", d) for d in deps)

def test_required_deps_listed():
    text = Path("pyproject.toml").read_text().lower()
    for dep in ("torch", "fastapi", "uvicorn", "numpy", "requests"):
        assert dep in text
