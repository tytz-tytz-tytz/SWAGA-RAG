from pathlib import Path


def test_pyproject_has_python_and_dev_test_dependency():
    root = Path(__file__).resolve().parents[1]
    content = (root / "pyproject.toml").read_text(encoding="utf-8")

    assert 'requires-python = ">=3.12,<3.13"' in content
    assert "[project.optional-dependencies]" in content
    assert 'pytest==8.0.0' in content


def test_bootstrap_script_has_required_steps():
    root = Path(__file__).resolve().parents[1]
    content = (root / "scripts" / "bootstrap_env.ps1").read_text(encoding="utf-8")

    assert "Python312" in content
    assert "-m venv" in content
    assert '-m pip install -e ".[dev]"' in content
    assert "Get-PreferredPython" in content


def test_env_example_exists_and_has_keys():
    root = Path(__file__).resolve().parents[1]
    env_example = root / ".env.example"
    assert env_example.exists(), ".env.example is required for judge/API setup"

    content = env_example.read_text(encoding="utf-8")
    assert "OPENAI_API_KEY" in content
    assert "COMETAPI_API_KEY" in content
