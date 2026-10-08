from __future__ import annotations

import argparse
import ast
import json
import os
import re
import stat
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

# This inventory is checked for exact equality with real sdist members. Internal
# evidence remains in the checkout; it is not silently treated as portable data.
REPOSITORY_ONLY_TEST_FILES = frozenset(
    {
        "tests/test_dependency_licence_evidence.py",
        "tests/test_scientific_code_smell_audit.py",
        "tests/test_verifier_complexity_slice_v1.py",
        "tests/_verifier_structure_gate_v1.py",
        "tests/fixtures/verifier_structure_gate_v1.json",
        "tests/test_internal_design_review.py",
        "tests/task10/test_authority_amendment_scope_v1.py",
        "tests/task10/test_authority_review_package_v2.py",
        "tests/task10/test_authority_review_package_v3.py",
        "tests/task10/test_registry_cross_representation_v1.py",
        "tests/test_manuscript_latex_builder.py",
        "tests/test_manuscript_number_binding.py",
        "tests/test_windows_runner_delivery.py",
        "tests/test_windows_package_delivery.py",
        "tests/test_m6_pearson_pair_comparison.py",
        "tests/test_build_public_release.py",
    }
)
REPOSITORY_ONLY_SCRIPT_FILES = frozenset(
    {
        "scripts/build_public_release.py",
        "scripts/build_windows_package.py",
    }
)
_PUBLIC_BENCHMARK_FILES = (
    "in_memory_standard_20260831.json",
    "in_memory_standard_20260831_figure.svg",
    "in_memory_standard_20260831_figure.png",
    "in_memory_standard_20260908.json",
    "in_memory_standard_20260908_figure.svg",
    "in_memory_standard_20260908_figure.png",
    "in_memory_standard_20260922.json",
    "in_memory_standard_20260922_figure.svg",
    "in_memory_standard_20260922_figure.png",
    "in_memory_standard_20260928.json",
    "in_memory_standard_20260928_r5.json",
    "in_memory_standard_20260929_f2a.json",
    "in_memory_standard_20260929_f2b1.json",
    "in_memory_standard_20260929_f2b2.json",
    "in_memory_standard_20260929_f2c.json",
    "in_memory_standard_20260929_f2c_r1.json",
    "in_memory_standard_20260929_f3.json",
    "in_memory_standard_20260929_f4b1.json",
    "in_memory_standard_20260929_f4b2a.json",
    "in_memory_standard_20260929_f4b2a_r2.json",
    "in_memory_standard_20260929_f4b2b.json",
    "in_memory_standard_20261002_r9.json",
    "in_memory_standard_20261002_r10.json",
    "in_memory_standard_20261003_alg.json",
    "in_memory_standard_20261004_r13.json",
    "in_memory_standard_20261004_r14.json",
    "in_memory_standard_20261006_r16.json",
    "in_memory_standard_20261006_r16b.json",
    "in_memory_standard_20261006_r17.json",
    "in_memory_standard_20261008_r21.json",
)
_MACHINE_PATH_PATTERNS = (
    re.compile(r"/(?:Users|home)/[^/\s\"']+/"),
    re.compile(r"/" + r"var/folders/"),
    re.compile(r"/" + r"tmp/"),
    re.compile(r"[A-Za-z]:\\Users\\[^\\\s\"']+\\"),
)
_CREDENTIAL_PATTERNS = (
    ("AWS_ACCESS_KEY", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("GITHUB_STYLE_TOKEN", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("OPENAI_STYLE_TOKEN", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
    (
        "PRIVATE_KEY_BLOCK",
        re.compile(r"-----BEGIN (?:RSA|EC|OPENSSH|DSA) PRIVATE KEY-----"),
    ),
    (
        "CREDENTIAL_ASSIGNMENT",
        re.compile(
            r"(?i)\b(?:api[_-]?key|password|secret|access[_-]?token|auth[_-]?token)"
            r"\s*[:=]\s*[\"'][^\"'\s]{8,}[\"']"
        ),
    ),
)


@dataclass(frozen=True, order=True)
class AuditFinding:
    rule: str
    path: str
    line: int
    detail: str


def _relative(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _first_symlink(root: Path, path: Path) -> Path | None:
    current = root
    for part in path.relative_to(root).parts:
        current = current / part
        if current.is_symlink():
            return current
    return None


def _release_surface_paths(root: Path) -> tuple[Path, ...]:
    """Public sources from MANIFEST.in plus exact wheel data-file declarations.

    This is an explicit SelCal inventory, not a MANIFEST interpreter. A change to
    either packaging declaration must also pass the real-sdist equality check.
    Symlinks are returned as findings and never traversed during discovery.
    """
    candidates: set[Path] = set()
    exact = {
        ".gitignore",
        ".github/workflows/ci.yml",
        "Dockerfile",
        ".dockerignore",
        "compose.yaml",
        "compose.ui.yaml",
        "MANIFEST.in",
        "README.md",
        "CHANGELOG.md",
        "CITATION.cff",
        "pyproject.toml",
        "uv.lock",
        "LICENSE.txt",
        "Licence.txt",
        "scripts/windows_check.ps1",
        "scripts/windows_native_capture.ps1",
        "scripts/windows_evidence_archive.ps1",
    }
    exact.update("docs/benchmarks/" + name for name in _PUBLIC_BENCHMARK_FILES)
    config = root / "pyproject.toml"
    if config.is_file() and _first_symlink(root, config) is None:
        data_files = (
            tomllib.loads(_read_utf8_regular_file(config))
            .get("tool", {})
            .get("setuptools", {})
            .get("data-files", {})
        )
        for paths in data_files.values():
            for relative in paths:
                path = root / relative
                if Path(relative).is_absolute() or ".." in Path(relative).parts:
                    raise ValueError("packaged data-file path must be checkout-relative")
                candidates.add(_first_symlink(root, path) or path)
    for relative in exact:
        path = root / relative
        link = _first_symlink(root, path)
        if link is not None or path.exists():
            candidates.add(link or path)

    trees = {
        "src/selcal": {".py", ".html", ".css", ".js", ".csv", ".json"},
        "examples": {".py", ".csv", ".json"},
        "docs/api": {".rst", ".py"},
        "docs/guide": {".md", ".png"},
        "scripts": {".py"},
        "tests": {".py", ".json", ".csv"},
    }
    for relative, suffixes in trees.items():
        surface = root / relative
        if (link := _first_symlink(root, surface)) is not None:
            candidates.add(link)
            continue
        for directory, subdirs, names in os.walk(surface, followlinks=False):
            parent = Path(directory)
            for name in subdirs[:]:
                path = parent / name
                if name in {"__pycache__", "_build"}:
                    subdirs.remove(name)
                elif path.is_symlink():
                    candidates.add(path)
                    subdirs.remove(name)
            for name in names:
                path = parent / name
                if (
                    path.suffix in suffixes
                    and _relative(root, path) not in REPOSITORY_ONLY_TEST_FILES
                    and _relative(root, path) not in REPOSITORY_ONLY_SCRIPT_FILES
                ):
                    candidates.add(path)
    return tuple(sorted(candidates, key=lambda path: _relative(root, path)))


def _read_utf8_regular_file(path: Path) -> str:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError("release-surface entry is not a regular file")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    return b"".join(chunks).decode("utf-8")


def _text_findings(relative: str, text: str) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        for pattern in _MACHINE_PATH_PATTERNS:
            match = pattern.search(line)
            if match is not None:
                findings.append(
                    AuditFinding(
                        rule="ABSOLUTE_MACHINE_PATH",
                        path=relative,
                        line=line_number,
                        detail=match.group(),
                    )
                )
                break
        for credential_name, pattern in _CREDENTIAL_PATTERNS:
            if pattern.search(line) is not None:
                findings.append(
                    AuditFinding(
                        rule="CREDENTIAL_MATERIAL",
                        path=relative,
                        line=line_number,
                        detail=credential_name,
                    )
                )
                break
    return findings


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name) and call.func.id in {"breakpoint", "print"}:
        return call.func.id
    if (
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "set_trace"
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "pdb"
    ):
        return "pdb.set_trace"
    return None


def _production_python_findings(relative: str, text: str) -> list[AuditFinding]:
    try:
        tree = ast.parse(text)
    except SyntaxError as error:
        return [
            AuditFinding(
                rule="PYTHON_PARSE_ERROR",
                path=relative,
                line=error.lineno or 0,
                detail=error.msg,
            )
        ]

    findings: list[AuditFinding] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and (name := _call_name(node)) is not None:
            findings.append(
                AuditFinding(
                    rule="PYTHON_DEBUG_CALL",
                    path=relative,
                    line=node.lineno,
                    detail=name,
                )
            )
    return findings


def audit_repository(repository_root: Path) -> tuple[AuditFinding, ...]:
    root = repository_root.resolve(strict=True)
    findings: list[AuditFinding] = []
    for path in _release_surface_paths(root):
        relative = _relative(root, path)
        if path.is_symlink():
            findings.append(
                AuditFinding(
                    rule="SYMLINK_IN_RELEASE_SURFACE",
                    path=relative,
                    line=0,
                    detail="not followed",
                )
            )
            continue
        if path.suffix == ".png":
            continue
        try:
            text = _read_utf8_regular_file(path)
        except (OSError, UnicodeDecodeError) as error:
            findings.append(
                AuditFinding(
                    rule="UNREADABLE_RELEASE_SURFACE_FILE",
                    path=relative,
                    line=0,
                    detail=type(error).__name__,
                )
            )
            continue
        findings.extend(_text_findings(relative, text))
        if relative.startswith("src/selcal/") and path.suffix == ".py":
            findings.extend(_production_python_findings(relative, text))
    return tuple(sorted(findings))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit SelCal's release surface for local paths, debug calls and secrets."
    )
    parser.add_argument("repository", nargs="?", default=".")
    arguments = parser.parse_args(argv)
    repository = Path(arguments.repository).resolve(strict=True)
    findings = audit_repository(repository)
    payload = {
        "findings": [asdict(finding) for finding in findings],
        "repository": str(repository),
        "status": "FAIL" if findings else "PASS",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
