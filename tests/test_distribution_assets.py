"""Inspect real build outputs without writing build metadata into the checkout.

Repository review tests remain collected in a checkout; their exact exclusions
and dependency reasons are declared in MANIFEST.in and checked against the sdist.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path

import pytest
from _platform_support import mkfifo_or_skip, symlink_or_skip

from scripts.audit_repository_hygiene import _release_surface_paths, audit_repository

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ONLY_TEST_FILES = {
    "tests/test_dependency_licence_evidence.py": "internal dependency evidence and binding",
    "tests/test_scientific_code_smell_audit.py": "internal historical receipts and readiness",
    "tests/test_verifier_complexity_slice_v1.py": "historical verifier receipt and design hashes",
    "tests/_verifier_structure_gate_v1.py": "helper for the historical verifier receipt gate",
    "tests/fixtures/verifier_structure_gate_v1.json": "fixture for the historical verifier gate",
    "tests/test_internal_design_review.py": "internal trusted-call design assertions",
    "tests/task10/test_authority_amendment_scope_v1.py": "internal authority amendment package",
    "tests/task10/test_authority_review_package_v2.py": "internal authority review package v2",
    "tests/task10/test_authority_review_package_v3.py": "internal authority review package v3",
    "tests/task10/test_registry_cross_representation_v1.py": "internal authority registry review",
    "tests/test_manuscript_latex_builder.py": (
        "draft manuscript and private repository status checks"
    ),
    "tests/test_manuscript_number_binding.py": (
        "unpublished draft headline evidence and numeric bindings"
    ),
    "tests/test_windows_runner_delivery.py": "internal Windows delivery runner and bundle checks",
    "tests/test_windows_package_delivery.py": "internal Windows delivery runner and bundle checks",
    "tests/test_m6_pearson_pair_comparison.py": "development-only research preparation script",
    "tests/test_build_public_release.py": (
        "internal release builder and release-inventory assertions"
    ),
}


def _copy_distribution_source(root: Path, source: Path) -> None:
    ignore_generated = shutil.ignore_patterns(
        ".git",
        ".venv",
        "__pycache__",
        "*.pyc",
        "*.egg-info",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        "build",
        "dist",
        "_build",
    )

    def ignore(directory: str, names: list[str]) -> set[str]:
        excluded = ignore_generated(directory, names)
        if Path(directory) == root / "docs/status":
            # Retain ordinary private documents to exercise MANIFEST pruning.
            # Historical negative fixtures are not distribution build inputs.
            excluded.add("evidence")
        return excluded

    # Do not dereference public links into apparently safe regular files.
    shutil.copytree(root, source, ignore=ignore, symlinks=True)


@pytest.mark.parametrize("kind", ["regular", "dangling", "cycle", "fifo"])
def test_distribution_copy_excludes_only_private_evidence(tmp_path: Path, kind: str) -> None:
    root = tmp_path / "checkout"
    evidence = root / "docs/status/evidence"
    evidence.mkdir(parents=True)
    private = evidence / "negative-fixture"
    if kind == "regular":
        private.write_text("private evidence\n", encoding="utf-8")
    elif kind == "fifo":
        mkfifo_or_skip(private)
    else:
        target = Path("missing") if kind == "dangling" else Path(".")
        symlink_or_skip(private, target, target_is_directory=kind == "cycle")
    original = private.lstat()
    retained = (
        "docs/status/retained.md",
        "docs/status/evidence-other/retained.md",
        "docs/superpowers/retained.md",
        "docs/api/evidence/retained.rst",
        "src/selcal/retained.py",
        "tests/fixtures/evidence/retained.json",
        next(iter(REPOSITORY_ONLY_TEST_FILES)),
    )
    for name in retained:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("retained\n", encoding="utf-8")
    source = tmp_path / "source"
    _copy_distribution_source(root, source)
    assert not (source / "docs/status/evidence").exists()
    assert private.lstat() == original
    for name in retained:
        assert (source / name).read_bytes() == (root / name).read_bytes(), name


@pytest.mark.parametrize("directory", [False, True])
def test_distribution_copy_preserves_public_links_for_rejection(
    tmp_path: Path,
    directory: bool,
) -> None:
    root = tmp_path / "checkout"
    public = root / "docs/api" if directory else root / "docs/api/usage.rst"
    public.parent.mkdir(parents=True)
    target = tmp_path / "target"
    if directory:
        target.mkdir()
        (target / "usage.rst").write_text("public text\n", encoding="utf-8")
    else:
        target.write_text("public text\n", encoding="utf-8")
    symlink_or_skip(public, target, target_is_directory=directory)
    source = tmp_path / "source"
    _copy_distribution_source(root, source)
    relative = public.relative_to(root)
    assert (source / relative).is_symlink()
    assert [(row.rule, row.path) for row in audit_repository(source)] == [
        ("SYMLINK_IN_RELEASE_SURFACE", relative.as_posix()),
    ]


def test_distribution_copy_does_not_silently_skip_public_fifo(tmp_path: Path) -> None:
    root = tmp_path / "checkout"
    public = root / "examples/evidence/series.csv"
    public.parent.mkdir(parents=True)
    mkfifo_or_skip(public)
    with pytest.raises(shutil.Error, match="named pipe"):
        _copy_distribution_source(root, tmp_path / "source")


def _add_distribution_prune_sentinels(source: Path) -> None:
    for relative in ("docs/status", "docs/superpowers"):
        private = source / relative / "distribution-prune-sentinel.py"
        for parent in (source / "docs", private.parent):
            if parent.is_symlink():
                raise ValueError(f"symlink private build-input directory: {parent}")
        private.parent.mkdir(parents=True, exist_ok=True)
        with private.open("x", encoding="utf-8") as handle:
            handle.write("# private build input; must not ship\n")


@pytest.mark.parametrize("fault", ["docs_link", "parent_link", "file_link", "existing_file"])
def test_distribution_sentinels_do_not_overwrite_or_follow_links(
    tmp_path: Path,
    fault: str,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "distribution-prune-sentinel.py"
    target.write_text("must remain unchanged\n", encoding="utf-8")
    if fault in {"docs_link", "parent_link"}:
        link = source / ("docs" if fault == "docs_link" else "docs/status")
        link.parent.mkdir(parents=True, exist_ok=True)
        # In the ancestor case, status also points to the protected file's parent.
        if fault == "docs_link":
            symlink_or_skip(outside / "status", outside, target_is_directory=True)
        symlink_or_skip(link, outside, target_is_directory=True)
    else:
        private = source / "docs/status/distribution-prune-sentinel.py"
        private.parent.mkdir(parents=True)
        if fault == "file_link":
            symlink_or_skip(private, target)
        else:
            private.write_text("existing input\n", encoding="utf-8")
    with pytest.raises((ValueError, FileExistsError)):
        _add_distribution_prune_sentinels(source)
    assert target.read_text(encoding="utf-8") == "must remain unchanged\n"
    if fault == "existing_file":
        assert private.read_text(encoding="utf-8") == "existing input\n"


@pytest.fixture(scope="module")
def distributions(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path]:
    workspace = tmp_path_factory.mktemp("distribution-assets")
    source = workspace / "source"
    _copy_distribution_source(ROOT, source)
    assert audit_repository(source) == ()
    # These inputs must exist even when this test runs from an unpacked sdist.
    _add_distribution_prune_sentinels(source)
    output = workspace / "dist"
    completed = subprocess.run(
        [sys.executable, "-m", "build", "--outdir", str(output), str(source)],
        cwd=workspace,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    sdists = list(output.glob("*.tar.gz"))
    wheels = list(output.glob("*.whl"))
    assert len(sdists) == len(wheels) == 1
    return source, sdists[0], wheels[0]


def _sdist_files(path: Path) -> dict[str, bytes]:
    with tarfile.open(path, "r:gz") as archive:
        files = {}
        for member in archive.getmembers():
            assert member.isfile() or member.isdir(), f"nonregular sdist member: {member.name}"
            if member.isfile():
                content = archive.extractfile(member)
                assert content is not None
                files[member.name.partition("/")[2]] = content.read()
        return files


def test_sdist_contains_documented_user_assets(distributions: tuple[Path, Path, Path]) -> None:
    source, sdist, _ = distributions
    required = {
        ".gitignore",
        "README.md",
        "pyproject.toml",
        "uv.lock",
        "examples/basic_selection_aware_calibration.py",
        "examples/binned_nette_block_shuffle.py",
        "examples/fail_closed_not_evaluable.py",
        "examples/save_and_read_result.py",
        "scripts/benchmark_in_memory.py",
        "scripts/plot_in_memory_benchmark.py",
        "scripts/m6_pearson_diagnostic.py",
        "scripts/compare_pearson_diagnostic.py",
        "tests/test_public_quality_commands.py",
        "docs/api/conf.py",
        "docs/api/index.rst",
        "docs/api/architecture.rst",
        "docs/api/performance.rst",
        "docs/api/public_api.rst",
        "docs/api/usage.rst",
        "docs/benchmarks/in_memory_standard_20260831.json",
        "docs/benchmarks/in_memory_standard_20260831_figure.svg",
        "docs/benchmarks/in_memory_standard_20260908.json",
        "docs/benchmarks/in_memory_standard_20260908_figure.svg",
        "docs/benchmarks/in_memory_standard_20260908_figure.png",
        "docs/benchmarks/in_memory_standard_20260922.json",
        "docs/benchmarks/in_memory_standard_20260922_figure.svg",
        "docs/benchmarks/in_memory_standard_20260922_figure.png",
        "docs/benchmarks/in_memory_standard_20260928.json",
        "docs/benchmarks/in_memory_standard_20260928_r5.json",
        "docs/benchmarks/in_memory_standard_20260929_f2a.json",
        "docs/benchmarks/in_memory_standard_20260929_f2b1.json",
        "docs/benchmarks/in_memory_standard_20260929_f2b2.json",
        "docs/benchmarks/in_memory_standard_20260929_f2c.json",
        "docs/benchmarks/in_memory_standard_20260929_f2c_r1.json",
        "docs/benchmarks/in_memory_standard_20260929_f3.json",
        "docs/benchmarks/in_memory_standard_20260929_f4b1.json",
        "docs/benchmarks/in_memory_standard_20260929_f4b2a.json",
        "docs/benchmarks/in_memory_standard_20260929_f4b2a_r2.json",
        "docs/benchmarks/in_memory_standard_20260929_f4b2b.json",
        "docs/benchmarks/in_memory_standard_20261002_r9.json",
        "docs/benchmarks/in_memory_standard_20261002_r10.json",
        "docs/benchmarks/in_memory_standard_20261003_alg.json",
        "docs/benchmarks/in_memory_standard_20261004_r13.json",
        "docs/benchmarks/in_memory_standard_20261006_r16.json",
        "docs/benchmarks/in_memory_standard_20261006_r16b.json",
        "docs/benchmarks/in_memory_standard_20261006_r17.json",
        "docs/benchmarks/in_memory_standard_20261008_r21.json",
    }
    readme = (source / "README.md").read_text(encoding="utf-8")
    required.update(re.findall(r"(?:examples|scripts)/[A-Za-z0-9_./-]+\.py", readme))
    required.update(
        path.relative_to(source).as_posix()
        for path in (source / "examples" / "workflow").glob("*")
        if path.suffix in {".csv", ".json"} and path.is_file()
    )
    files = _sdist_files(sdist)
    assert not (missing := required - files.keys()), f"sdist missing user assets: {sorted(missing)}"
    for name in required:
        assert files[name] == (source / name).read_bytes(), name


def test_public_metadata_does_not_claim_unverified_os_or_undecided_licence() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert "Operating System :: OS Independent" not in project["classifiers"]

    origin = (ROOT / "docs/provenance/SOURCE_ORIGIN.tsv").read_text(encoding="utf-8")
    assert "SELCAL_LICENCE_DECISION_NOT_DUE" not in origin
    assert "BSD-3-Clause_rights_confirmed_by_author_20261005" in origin

    boundary = (ROOT / "docs/architecture/m0_m2_boundary.md").read_text(encoding="utf-8")
    assert "Current reconciliation:" not in boundary
    assert "as of 2026-08-31" in boundary


def test_sdist_preserves_source_and_test_inputs(distributions: tuple[Path, Path, Path]) -> None:
    source, sdist, _ = distributions
    required = {
        path.relative_to(source).as_posix()
        for directory in (source / "src" / "selcal", source / "tests")
        for path in directory.rglob("*")
        if path.is_file() and path.suffix in {".py", ".json", ".csv", ".html", ".css", ".js"}
    } - REPOSITORY_ONLY_TEST_FILES.keys()
    files = _sdist_files(sdist)
    assert not (missing := required - files.keys()), (
        f"sdist missing source/test files: {sorted(missing)}"
    )
    for name in required:
        assert files[name] == (source / name).read_bytes(), name


def test_ui_assets_are_exact_in_both_distributions(distributions: tuple[Path, Path, Path]) -> None:
    source, sdist, wheel = distributions
    expected = {"index.html", "app.js", "style.css", "series.csv", "pearson.json"}
    source_files = _sdist_files(sdist)
    with zipfile.ZipFile(wheel) as archive:
        actual = {
            name.removeprefix("selcal/web/")
            for name in archive.namelist()
            if name.startswith("selcal/web/")
        }
        assert actual == expected
        for name in expected:
            content = (source / "src" / "selcal" / "web" / name).read_bytes()
            assert archive.read("selcal/web/" + name) == content
            assert source_files["src/selcal/web/" + name] == content


def test_sdist_contains_the_citation_referenced_by_readme(
    distributions: tuple[Path, Path, Path],
) -> None:
    source, sdist, _ = distributions
    citation = (source / "CITATION.cff").read_bytes()
    assert citation
    assert "CITATION.cff" in (source / "README.md").read_text(encoding="utf-8")
    assert _sdist_files(sdist)["CITATION.cff"] == citation


@pytest.mark.parametrize("name", [
    "windows_check.ps1", "windows_native_capture.ps1", "windows_evidence_archive.ps1",
])
def test_sdist_includes_runner_and_helpers_required_by_shipped_tests(
    distributions: tuple[Path, Path, Path], name: str,
) -> None:
    source, sdist, _ = distributions
    files = _sdist_files(sdist)
    assert "tests/test_windows_check_workpaths.py" in files
    relative = "scripts/" + name
    assert relative in files, f"sdist ships runner tests without their dependency: {relative}"
    assert files[relative] == (source / relative).read_bytes()


def test_distributions_exclude_internal_reviews_and_generated_outputs(
    distributions: tuple[Path, Path, Path],
) -> None:
    source, sdist, wheel = distributions
    for relative in ("docs/status", "docs/superpowers"):
        assert (source / relative / "distribution-prune-sentinel.py").is_file()
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist()) | _sdist_files(sdist).keys()
    for name in names:
        assert "docs/superpowers/" not in name, name
        assert "docs/status/" not in name, name
        assert "docs/api/_build/" not in name, name
        assert "__pycache__/" not in name, name
        assert not name.endswith((".pyc", ".pyo", ".DS_Store")), name


def test_repository_only_tests_have_explicit_exclusions_and_remain_in_checkout(
    distributions: tuple[Path, Path, Path],
) -> None:
    source, sdist, _ = distributions
    manifest = (source / "MANIFEST.in").read_text(encoding="utf-8")
    files = _sdist_files(sdist)
    for name, reason in REPOSITORY_ONLY_TEST_FILES.items():
        assert f"# {reason}\nexclude {name}\n" in manifest, name
        assert name not in files, name
        if not (ROOT / name).exists():
            assert not (ROOT / ".git").exists(), f"missing checkout evidence: {name}"
        else:
            assert (source / name).read_bytes() == (ROOT / name).read_bytes(), name


def test_sdist_file_inventory_exactly_matches_audited_release_surface(
    distributions: tuple[Path, Path, Path],
) -> None:
    source, sdist, _ = distributions
    files = _sdist_files(sdist)
    # Setuptools-generated metadata is validated by the build and wheel checks.
    generated = {"PKG-INFO", "setup.cfg"} | {
        "src/selcal.egg-info/" + name
        for name in (
            "PKG-INFO",
            "SOURCES.txt",
            "dependency_links.txt",
            "entry_points.txt",
            "requires.txt",
            "top_level.txt",
        )
    }
    selected = {path.relative_to(source).as_posix() for path in _release_surface_paths(source)}
    assert files.keys() - generated == selected
    assert audit_repository(source) == ()


@pytest.mark.parametrize("fault", ["machine_path", "credential", "symlink", "symlink_directory"])
def test_real_sdist_public_asset_faults_are_rejected(
    distributions: tuple[Path, Path, Path],
    tmp_path: Path,
    fault: str,
) -> None:
    _, sdist, _ = distributions
    files = _sdist_files(sdist)
    for name, content in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    assert audit_repository(tmp_path) == ()
    public = tmp_path / "docs/api/usage.rst"
    assert public.read_bytes() == files["docs/api/usage.rst"]
    expected_path = "docs/api/usage.rst"
    if fault == "symlink_directory":
        public.parent.rename(tmp_path / "private-api-copy")
        symlink_or_skip(public.parent, tmp_path / "private-api-copy", target_is_directory=True)
        expected = "SYMLINK_IN_RELEASE_SURFACE"
        expected_path = "docs/api"
    elif fault == "symlink":
        public.unlink()
        symlink_or_skip(public, tmp_path / "README.md")
        expected = "SYMLINK_IN_RELEASE_SURFACE"
    else:
        payload = (
            "/" + "Users/example/private/checkout" if fault == "machine_path" else "sk-" + "X" * 32
        )
        public.write_text(payload + "\n", encoding="utf-8", newline="\n")
        expected = "ABSOLUTE_MACHINE_PATH" if fault == "machine_path" else "CREDENTIAL_MATERIAL"
    assert [(row.rule, row.path) for row in audit_repository(tmp_path)] == [
        (expected, expected_path),
    ]


@pytest.mark.parametrize("link_type", [tarfile.SYMTYPE, tarfile.LNKTYPE])
def test_sdist_member_inventory_rejects_archive_links(
    distributions: tuple[Path, Path, Path],
    tmp_path: Path,
    link_type: bytes,
) -> None:
    _, sdist, _ = distributions
    linked = tmp_path / "linked.tar.gz"
    with tarfile.open(sdist, "r:gz") as original, tarfile.open(linked, "w:gz") as mutated:
        for member in original.getmembers():
            mutated.addfile(member, original.extractfile(member) if member.isfile() else None)
        link = tarfile.TarInfo("selcal-0.1.0.dev0/docs/api/linked.rst")
        link.type = link_type
        link.linkname = "usage.rst"
        mutated.addfile(link)
    with pytest.raises(AssertionError, match="nonregular sdist member"):
        _sdist_files(linked)


def test_wheel_from_sdist_preserves_runtime_and_declared_documentation(
    distributions: tuple[Path, Path, Path],
) -> None:
    source, _, wheel = distributions
    pyproject = tomllib.loads((source / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]
    data_prefix = f"{project['name']}-{project['version']}.data/data/"
    with zipfile.ZipFile(wheel) as archive:
        for path in (source / "src" / "selcal").rglob("*.py"):
            name = path.relative_to(source / "src").as_posix()
            assert archive.read(name) == path.read_bytes(), name
        for destination, paths in pyproject["tool"]["setuptools"]["data-files"].items():
            for relative in paths:
                name = data_prefix + destination + "/" + Path(relative).name
                assert archive.read(name) == (source / relative).read_bytes(), name
