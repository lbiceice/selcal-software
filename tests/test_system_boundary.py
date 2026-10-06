from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path

from _documentation import public_documentation_text

import selcal
from scripts.audit_repository_hygiene import _release_surface_paths
from selcal.contracts import PlanRequest, ResolvedScientificPlan
from selcal.resolution import PlanResolution, resolve_plan

FORBIDDEN_IMPORTS = {"idtxl", "jpype", "pyjnius", "rpy2", "streamlit", "gradio"}
CONCRETE_ADAPTER_NAMES = {
    "equal_width_binned_nette_v1",
    "lagged_pearson_v1",
    "circular_shift_v1",
    "block_shuffle_v1",
}

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
PRODUCTION_ROOT = REPOSITORY_ROOT / "src" / "selcal"
ARCHITECTURE_BOUNDARY = REPOSITORY_ROOT / "docs" / "architecture" / "m0_m2_boundary.md"
P3_BOUNDARY = REPOSITORY_ROOT / "docs" / "provenance" / "P3_READ_ONLY_BOUNDARY.md"
LEGACY_IDENTITY = "P3-NetTE-repro@d0af7ad06f3159e4b228f44da3c453d1b3786a0e"
DESIGN_AUTHORITIES = (
    "2026-08-25-p3-selcal-stage3-design.md",
    "2026-08-26-p3-selcal-multi-adapter-calibration-audit-design.md",
)
REQUIRED_ARCHITECTURE_ROWS = (
    "| Unresolved request → registered contracts | Implemented and locally tested |",
    "| v1 scientific hash identity | Golden vectors preserved locally |",
    "| Binned NetTE statistic | Implemented and locally tested |",
    "| Lagged Pearson statistic | Implemented and locally tested |",
    "| Circular-shift/block-shuffle preservation contracts | "
    "M0-M2 implementation candidate; final verification pending |",
    "| Full surrogate reselection calibration | "
    "M0-M2 implementation candidate; final verification pending |",
    "| Resolution-bound result verifier | "
    "M0-M2 implementation candidate; final verification pending |",
    "| Independent finite-state algorithm oracle | "
    "Internal implementation candidate; not top-level public API |",
    "| M6 calibration and comparator evidence | HOLD / not executed |",
    "| SoftwareX scientific impact | HOLD |",
)


def _imported_top_level_modules(source: str) -> set[str]:
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.partition(".")[0])
    return imported


def _dependency_name(dependency: str) -> str:
    match = re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", dependency)
    assert match is not None
    return re.sub(r"[-_.]+", "-", match.group()).lower()


def _concrete_adapter_names_in(text: str) -> set[str]:
    return {name for name in CONCRETE_ADAPTER_NAMES if name in text}


def _package_python_files(source_root: Path, include: list[str]) -> tuple[Path, ...]:
    assert include == ["selcal*"]
    package_root = source_root / "selcal"
    return tuple(
        sorted(
            path
            for path in package_root.rglob("*.py")
            if "__pycache__" not in path.parts and not path.is_symlink()
        )
    )


def test_production_tree_enforces_clean_room_firewall() -> None:
    assert SOURCE_ROOT.is_dir()
    assert PRODUCTION_ROOT.is_dir()

    pyproject = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    package_find = pyproject["tool"]["setuptools"]["packages"]["find"]
    assert package_find.get("where") == ["src"]
    assert package_find.get("include") == ["selcal*"]
    source_files = _package_python_files(SOURCE_ROOT, package_find["include"])
    assert source_files
    assert all(b"P3-NetTE-repro" not in path.read_bytes() for path in source_files)

    imported = set().union(
        *(
            _imported_top_level_modules(path.read_text(encoding="utf-8"))
            for path in source_files
            if path.suffix == ".py"
        )
    )
    assert not imported.intersection(FORBIDDEN_IMPORTS)

    unexpected_sibling_packages = [
        path.name
        for path in SOURCE_ROOT.iterdir()
        if path.is_dir() and path.name != "selcal" and (path / "__init__.py").is_file()
    ]
    assert not unexpected_sibling_packages

    project = pyproject["project"]

    declared_dependencies = list(project.get("dependencies", []))
    for dependency_group in project.get("optional-dependencies", {}).values():
        declared_dependencies.extend(dependency_group)

    declared_names = {_dependency_name(dependency) for dependency in declared_dependencies}
    assert not declared_names.intersection(FORBIDDEN_IMPORTS)

    data_files = pyproject["tool"]["setuptools"].get("data-files")
    assert data_files == {
        "share/selcal/docs/architecture": [
            "docs/architecture/m0_m2_boundary.md",
            "docs/architecture/null_reselection_v2_design.md",
            "docs/architecture/in_memory_execution_budget_profile_20260828.csv",
        ],
        "share/selcal/docs/provenance": [
            "docs/provenance/P3_READ_ONLY_BOUNDARY.md",
            "docs/provenance/SOURCE_ORIGIN.tsv",
        ],
    }


def test_clean_room_source_discovery_excludes_cache_and_metadata_trees(
    tmp_path: Path,
) -> None:
    package = tmp_path / "src" / "selcal"
    package.mkdir(parents=True)
    real_source = package / "core.py"
    real_source.write_text("VALUE = 1\n", encoding="utf-8", newline="\n")
    cache = package / "__pycache__"
    cache.mkdir()
    (cache / "cached.py").write_text("P3-NetTE-repro\n", encoding="utf-8", newline="\n")
    metadata = tmp_path / "src" / "selcal.egg-info"
    metadata.mkdir()
    (metadata / "generated.py").write_text("P3-NetTE-repro\n", encoding="utf-8", newline="\n")
    (package / "core.pyc").write_bytes(b"P3-NetTE-repro")

    assert _package_python_files(tmp_path / "src", ["selcal*"]) == (real_source,)


def test_core_contract_and_hash_modules_do_not_name_concrete_adapters() -> None:
    for module_name in ("contracts.py", "canonical.py"):
        module_text = (PRODUCTION_ROOT / module_name).read_text(encoding="utf-8")
        assert not _concrete_adapter_names_in(module_text)


def test_concrete_adapter_guard_reports_each_whole_name() -> None:
    for adapter_name in CONCRETE_ADAPTER_NAMES:
        synthetic_text = f"prefix {adapter_name} suffix"
        assert _concrete_adapter_names_in(synthetic_text) == {adapter_name}


def test_top_level_public_surface_is_minimal_and_resolves_exact_objects() -> None:
    assert set(selcal.__all__) == {
        "InvalidNullForInferenceError",
        "PlanMigrationV1ToV2",
        "PlanRequest",
        "PlanRequestV2",
        "PlanResolution",
        "PlanResolutionV2",
        "ResolvedScientificPlan",
        "ResolvedScientificPlanV2",
        "calibrate_selected_family",
        "migrate_plan_v1_to_v2",
        "resolve_plan",
        "resolve_plan_v2",
        "verify_calibration_result",
    }
    assert selcal.PlanRequest is PlanRequest
    assert selcal.PlanResolution is PlanResolution
    assert selcal.ResolvedScientificPlan is ResolvedScientificPlan
    assert selcal.resolve_plan is resolve_plan
    assert selcal.__version__ == "0.1.0"
    assert not hasattr(selcal, "AdapterRegistry")


def test_shipped_text_is_portable() -> None:
    for path in _release_surface_paths(REPOSITORY_ROOT):
        assert not path.is_symlink(), path
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        assert "/" + "Users/" not in text, path
        assert "vin" + "cent" not in text.casefold(), path


def test_portable_provenance_uses_logical_authorities_and_legacy_identity() -> None:
    architecture_text = ARCHITECTURE_BOUNDARY.read_text(encoding="utf-8")
    p3_boundary_text = P3_BOUNDARY.read_text(encoding="utf-8")

    assert all(authority in architecture_text for authority in DESIGN_AUTHORITIES)
    assert LEGACY_IDENTITY in p3_boundary_text
    assert "/" + "Users/" not in architecture_text
    assert "/" + "Users/" not in p3_boundary_text


def test_architecture_capability_table_is_truthful() -> None:
    architecture_text = ARCHITECTURE_BOUNDARY.read_text(encoding="utf-8")

    assert all(row in architecture_text for row in REQUIRED_ARCHITECTURE_ROWS)
    for line in architecture_text.splitlines():
        if line.startswith("|") and "full-reselection calibration" in line.casefold():
            assert "implementation candidate" in line.casefold()
            assert "final verification pending" in line.casefold()


def test_readme_states_implemented_and_absent_product_boundaries() -> None:
    readme_text = public_documentation_text(REPOSITORY_ROOT)

    assert "M0-M2 IMPLEMENTATION CANDIDATE / FINAL VERIFICATION PENDING" in readme_text
    assert "Current implementation:" in readme_text
    assert (
        "SCIENTIFIC EVIDENCE: BOUNDED SYNTHETIC STUDIES EXECUTED / "
        "GENERAL IMPACT AND INDEPENDENT-USER BENEFIT HOLD"
    ) in readme_text
    # Licence is chosen; public source release, DOI and manuscript submission remain open.
    assert "LICENSE: BSD-3-CLAUSE" in readme_text
    assert "DOI: PENDING" in readme_text
    assert "MANUSCRIPT: IN PREPARATION, NOT SUBMITTED" in readme_text
    assert (
        "LOCAL RESUME: REPLAY_BEFORE_CONTINUE / EVIDENCE BUNDLES: CONTENT-CHECKED / "
        "UI: PARTIAL_UI_BASIC_LOOP"
    ) in readme_text
    assert (
        "input/validate/run/resume/result-content-check loop plus reports, evidence export\n"
        "and checked downloads. F4C integration is locally accepted on macOS; container,\n"
        "renewed native Windows and release gates remain separate."
        in readme_text
    )
    assert "selcal export run.sqlite evidence --max-bytes 8388608" in readme_text
    assert "selcal verify-export evidence --max-bytes 8388608" in readme_text
    assert "PUBLIC RELEASE PENDING" in readme_text
    assert "RESUME, EVIDENCE BUNDLES, UI: NOT DUE" not in readme_text
    assert "have not had a new native Windows execution" in readme_text
    assert "No license has been selected" not in readme_text
    assert "release surface is public" not in readme_text
    assert "SelCal is not a causal-edge inference product." in readme_text
