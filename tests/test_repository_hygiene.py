from __future__ import annotations

import json
from pathlib import Path

import pytest
from _platform_support import symlink_or_skip

from scripts.audit_repository_hygiene import AuditFinding, audit_repository, main

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MACHINE_PREFIX = "/" + "Users/alice/"


def _write_minimal_release_surface(root: Path) -> None:
    package = root / "src" / "selcal"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8", newline="\n")
    (root / "README.md").write_text(
        "# Example\n\n```python\nprint('result')\n```\n", encoding="utf-8", newline="\n"
    )
    (root / "pyproject.toml").write_text(
        "[project]\nname = 'selcal'\n", encoding="utf-8", newline="\n"
    )


def test_current_release_surface_passes_repository_hygiene_gate() -> None:
    assert audit_repository(REPOSITORY_ROOT) == ()


def test_production_python_debug_call_is_rejected(tmp_path: Path) -> None:
    _write_minimal_release_surface(tmp_path)
    source = tmp_path / "src" / "selcal" / "core.py"
    source.write_text("def run() -> None:\n    print('debug')\n", encoding="utf-8", newline="\n")

    assert audit_repository(tmp_path) == (
        AuditFinding(
            rule="PYTHON_DEBUG_CALL",
            path="src/selcal/core.py",
            line=2,
            detail="print",
        ),
    )


def test_machine_local_absolute_path_is_rejected(tmp_path: Path) -> None:
    _write_minimal_release_surface(tmp_path)
    source = tmp_path / "src" / "selcal" / "core.py"
    source.write_text(
        f'DATA = "{MACHINE_PREFIX}private/input.csv"\n', encoding="utf-8", newline="\n"
    )

    assert audit_repository(tmp_path) == (
        AuditFinding(
            rule="ABSOLUTE_MACHINE_PATH",
            path="src/selcal/core.py",
            line=1,
            detail=MACHINE_PREFIX,
        ),
    )


def test_credential_shaped_material_is_rejected_without_echoing_secret(
    tmp_path: Path,
) -> None:
    _write_minimal_release_surface(tmp_path)
    secret = "sk-" + "A" * 32
    (tmp_path / "README.md").write_text(
        f'api_{"key"} = "{secret}"\n', encoding="utf-8", newline="\n"
    )

    assert audit_repository(tmp_path) == (
        AuditFinding(
            rule="CREDENTIAL_MATERIAL",
            path="README.md",
            line=1,
            detail="OPENAI_STYLE_TOKEN",
        ),
    )
    assert secret not in repr(audit_repository(tmp_path))


def test_symlink_in_release_surface_is_rejected_without_following_it(tmp_path: Path) -> None:
    _write_minimal_release_surface(tmp_path)
    outside = tmp_path / "outside.py"
    outside.write_text("VALUE = 2\n", encoding="utf-8", newline="\n")
    link = tmp_path / "src" / "selcal" / "linked.py"
    symlink_or_skip(link, outside)
    assert audit_repository(tmp_path) == (
        AuditFinding(
            rule="SYMLINK_IN_RELEASE_SURFACE",
            path="src/selcal/linked.py",
            line=0,
            detail="not followed",
        ),
    )


@pytest.mark.parametrize("directory", ["superpowers", "status", "benchmarks"])
def test_unshipped_review_evidence_is_preserved_outside_release_surface(
    tmp_path: Path,
    directory: str,
) -> None:
    _write_minimal_release_surface(tmp_path)
    internal = tmp_path / "docs" / directory / "review.md"
    internal.parent.mkdir(parents=True)
    original = f"{MACHINE_PREFIX}internal-only\n".encode()
    internal.write_bytes(original)

    assert audit_repository(tmp_path) == ()
    assert internal.read_bytes() == original


def test_explicitly_packaged_status_document_is_audited(tmp_path: Path) -> None:
    _write_minimal_release_surface(tmp_path)
    status = tmp_path / "docs" / "status" / "readiness.md"
    status.parent.mkdir(parents=True)
    status.write_text(
        f"worktree: {MACHINE_PREFIX}private/checkout\n", encoding="utf-8", newline="\n"
    )
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8") + "\n[tool.setuptools.data-files]\n"
        '"share/selcal/docs/status" = ["docs/status/readiness.md"]\n',
        encoding="utf-8",
        newline="\n",
    )

    assert audit_repository(tmp_path) == (
        AuditFinding(
            rule="ABSOLUTE_MACHINE_PATH",
            path="docs/status/readiness.md",
            line=1,
            detail=MACHINE_PREFIX,
        ),
    )


def test_dependency_lock_is_part_of_portable_release_surface(tmp_path: Path) -> None:
    _write_minimal_release_surface(tmp_path)
    (tmp_path / "uv.lock").write_text(
        f'source = "{MACHINE_PREFIX}private/wheelhouse"\n', encoding="utf-8", newline="\n"
    )

    assert audit_repository(tmp_path) == (
        AuditFinding(
            rule="ABSOLUTE_MACHINE_PATH",
            path="uv.lock",
            line=1,
            detail=MACHINE_PREFIX,
        ),
    )


def test_generated_documentation_sources_are_part_of_release_surface(
    tmp_path: Path,
) -> None:
    _write_minimal_release_surface(tmp_path)
    api_doc = tmp_path / "docs" / "api" / "usage.rst"
    api_doc.parent.mkdir(parents=True)
    api_doc.write_text(
        f"Developer checkout: {MACHINE_PREFIX}private/checkout\n",
        encoding="utf-8",
        newline="\n",
    )

    assert audit_repository(tmp_path) == (
        AuditFinding(
            rule="ABSOLUTE_MACHINE_PATH",
            path="docs/api/usage.rst",
            line=1,
            detail=MACHINE_PREFIX,
        ),
    )


def test_benchmark_receipts_are_part_of_portable_release_surface(tmp_path: Path) -> None:
    _write_minimal_release_surface(tmp_path)
    receipt = tmp_path / "docs" / "benchmarks" / "in_memory_standard_20260831.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text(
        '{"checkout": "' + MACHINE_PREFIX + 'private/checkout"}\n',
        encoding="utf-8",
        newline="\n",
    )

    assert audit_repository(tmp_path) == (
        AuditFinding(
            rule="ABSOLUTE_MACHINE_PATH",
            path="docs/benchmarks/in_memory_standard_20260831.json",
            line=1,
            detail=MACHINE_PREFIX,
        ),
    )


def test_benchmark_vector_figures_are_part_of_portable_release_surface(
    tmp_path: Path,
) -> None:
    _write_minimal_release_surface(tmp_path)
    figure = tmp_path / "docs" / "benchmarks" / "in_memory_standard_20260831_figure.svg"
    figure.parent.mkdir(parents=True)
    figure.write_text(
        f"<svg><text>{MACHINE_PREFIX}private/checkout</text></svg>\n",
        encoding="utf-8",
        newline="\n",
    )

    assert audit_repository(tmp_path) == (
        AuditFinding(
            rule="ABSOLUTE_MACHINE_PATH",
            path="docs/benchmarks/in_memory_standard_20260831_figure.svg",
            line=1,
            detail=MACHINE_PREFIX,
        ),
    )


@pytest.mark.parametrize("relative", ["src/selcal", "examples", "docs/api", "tests"])
def test_symlinked_release_directory_is_rejected_without_reading_target(
    tmp_path: Path,
    relative: str,
) -> None:
    _write_minimal_release_surface(tmp_path)
    outside = tmp_path / "private-evidence"
    outside.mkdir()
    (outside / "private.py").write_text("VALUE = 1\n", encoding="utf-8", newline="\n")
    link = tmp_path / relative
    if link.is_dir():
        (link / "__init__.py").unlink()
        link.rmdir()
    link.parent.mkdir(parents=True, exist_ok=True)
    symlink_or_skip(link, outside, target_is_directory=True)
    assert audit_repository(tmp_path) == (
        AuditFinding("SYMLINK_IN_RELEASE_SURFACE", relative, 0, "not followed"),
    )


@pytest.mark.parametrize("relative", ["tests/test_new.py", "examples/workflow/input.csv"])
def test_shipped_test_and_csv_inputs_are_audited(tmp_path: Path, relative: str) -> None:
    _write_minimal_release_surface(tmp_path)
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'VALUE = "{MACHINE_PREFIX}private/input"\n', encoding="utf-8", newline="\n")

    assert audit_repository(tmp_path) == (
        AuditFinding("ABSOLUTE_MACHINE_PATH", relative, 1, MACHINE_PREFIX),
    )


def test_command_line_result_is_machine_readable_and_redacts_secret(
    tmp_path: Path,
    capsys,
) -> None:
    _write_minimal_release_surface(tmp_path)
    secret = "ghp_" + "B" * 32
    (tmp_path / "README.md").write_text(f'token = "{secret}"\n', encoding="utf-8", newline="\n")

    assert main([str(tmp_path)]) == 1
    output = capsys.readouterr().out
    payload = json.loads(output)

    assert payload == {
        "findings": [
            {
                "detail": "GITHUB_STYLE_TOKEN",
                "line": 1,
                "path": "README.md",
                "rule": "CREDENTIAL_MATERIAL",
            }
        ],
        "repository": str(tmp_path.resolve()),
        "status": "FAIL",
    }
    assert secret not in output
