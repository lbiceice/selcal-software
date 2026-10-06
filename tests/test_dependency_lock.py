from __future__ import annotations

import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest
from _documentation import public_documentation_text

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LOCKFILE = REPOSITORY_ROOT / "uv.lock"


def _project_package(lock: dict[str, object]) -> dict[str, object]:
    packages = lock.get("package")
    assert isinstance(packages, list)
    matches = [
        package
        for package in packages
        if isinstance(package, dict) and package.get("name") == "selcal"
    ]
    assert len(matches) == 1
    return matches[0]


def _dependency_names(value: object) -> set[str]:
    assert isinstance(value, list)
    names: set[str] = set()
    for dependency in value:
        assert isinstance(dependency, dict)
        name = dependency.get("name")
        assert isinstance(name, str)
        names.add(name)
    return names


def test_lockfile_records_runtime_and_all_declared_optional_surfaces() -> None:
    assert LOCKFILE.is_file(), "generate uv.lock from the current pyproject.toml"
    lock = tomllib.loads(LOCKFILE.read_text(encoding="utf-8"))

    assert lock["version"] == 1
    assert lock["requires-python"] == ">=3.11"

    project = _project_package(lock)
    assert project["version"] == "0.1.0"
    assert project["source"] == {"editable": "."}
    assert _dependency_names(project["dependencies"]) == {"numpy"}

    optional = project.get("optional-dependencies")
    assert isinstance(optional, dict)
    assert set(optional) == {"benchmark", "dev", "docs"}
    assert _dependency_names(optional["benchmark"]) == {"matplotlib"}
    assert _dependency_names(optional["docs"]) == {"sphinx"}
    assert _dependency_names(optional["dev"]) == {
        "build",
        "mypy",
        "pip",
        "pytest",
        "pytest-cov",
        "ruff",
    }


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv is not installed")
def test_lockfile_is_current_for_pyproject() -> None:
    completed = subprocess.run(
        ["uv", "lock", "--check"],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        errors="replace",
    )
    assert completed.returncode == 0, completed.stderr


def test_readme_documents_exact_locked_environment_commands() -> None:
    readme = public_documentation_text(REPOSITORY_ROOT)
    assert "uv sync --locked --all-extras" in readme
    assert "uv lock --check" in readme
    assert "uv run --locked --all-extras python -m pytest" in readme
