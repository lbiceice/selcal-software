from __future__ import annotations

import re
import subprocess
import sys
import tomllib
from pathlib import Path
from _documentation import public_documentation_text

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DOCS_SOURCE = REPOSITORY_ROOT / "docs" / "api"


def test_sphinx_documentation_builds_without_warnings_and_lists_public_api(
    tmp_path: Path,
) -> None:
    pyproject = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text("utf-8"))
    assert pyproject["project"]["optional-dependencies"]["docs"] == ["sphinx>=8.2.3,<9"]

    output = tmp_path / "html"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "sphinx",
            "--fail-on-warning",
            "--keep-going",
            "-b",
            "html",
            str(DOCS_SOURCE),
            str(output),
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert (output / "index.html").is_file()
    assert (output / "architecture.html").is_file()
    assert (output / "performance.html").is_file()
    assert (output / "usage.html").is_file()

    architecture = (output / "architecture.html").read_text(encoding="utf-8")
    for architecture_marker in (
        "Component boundaries",
        "selcal.contracts_v2",
        "selcal.calibration_v2",
        "Terminal persistence and a thin CLI are implemented",
    ):
        assert architecture_marker in architecture

    performance = (output / "performance.html").read_text(encoding="utf-8")
    for performance_marker in (
        "Benchmark contract",
        "benchmark_in_memory.py --preset standard",
        "in_memory_standard_20260922_figure.svg",
        "single-process characterization",
        "tracemalloc",
    ):
        assert performance_marker in performance

    usage = (output / "usage.html").read_text(encoding="utf-8")
    for usage_marker in (
        "basic_selection_aware_calibration.py",
        "binned_nette_block_shuffle.py",
        "fail_closed_not_evaluable.py",
        "NOT_EVALUABLE",
        "not scientific validation",
    ):
        assert usage_marker in usage

    public_api = (output / "public_api.html").read_text(encoding="utf-8")
    for public_name in (
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
        "run_files",
        "read_workflow",
        "report_record",
        "export_record",
        "verify_export",
    ):
        assert public_name in public_api


def test_readme_documents_reproducible_api_reference_build() -> None:
    readme = public_documentation_text(REPOSITORY_ROOT)

    assert 'python -m pip install ".[docs]"' in readme
    assert (
        "python -m sphinx --fail-on-warning --keep-going -b html docs/api docs/api/_build/html"
    ) in readme
    assert "docs/api/_build/html/index.html" in readme


def test_generated_api_reference_is_excluded_from_version_control() -> None:
    ignore_rules = (REPOSITORY_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()

    assert "/docs/api/_build/" in ignore_rules


def test_evidence_export_documents_scientific_status_and_content_only_scope() -> None:
    for path in (REPOSITORY_ROOT / "docs/guide/reference.md", DOCS_SOURCE / "usage.rst"):
        content = path.read_text(encoding="utf-8")
        assert "selcal export run.sqlite evidence --max-bytes " in content
        assert "selcal verify-export evidence --max-bytes " in content
        for marker in ("manifest", "aggregate", "exit 7", "historical", "fresh destination"):
            assert marker in content


def test_documented_export_budgets_fit_the_shipped_pearson_example(tmp_path: Path) -> None:
    budgets = []
    for path in (REPOSITORY_ROOT / "docs/guide/reference.md", DOCS_SOURCE / "usage.rst"):
        content = path.read_text(encoding="utf-8")
        export = re.findall(r"selcal export run\.sqlite evidence --max-bytes (\d+)", content)
        verify = re.findall(r"selcal verify-export evidence --max-bytes (\d+)", content)
        assert len(export) == len(verify) == 1
        assert export == verify, f"{path.name}: export and verify budgets differ"
        budgets.append(export[0])
    assert budgets[0] == budgets[1], "README and usage export budgets differ"

    def run(*args: str) -> None:
        completed = subprocess.run(
            [sys.executable, "-I", "-B", "-m", "selcal", *args],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=30,
            check=False,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr

    # One shipped synthetic record is shared by both documentation examples.
    run(
        "run",
        str(REPOSITORY_ROOT / "examples/workflow/series.csv"),
        str(REPOSITORY_ROOT / "examples/workflow/pearson.json"),
        "run.sqlite",
        "--max-bytes",
        budgets[0],
    )
    record_before = (tmp_path / "run.sqlite").read_bytes()
    for index, budget in enumerate(budgets):
        destination = f"evidence-{index}"
        run("export", "run.sqlite", destination, "--max-bytes", budget)
        run("verify-export", destination, "--max-bytes", budget)
    assert (tmp_path / "run.sqlite").read_bytes() == record_before


def test_usage_capability_summary_separates_export_from_open_gates() -> None:
    content = " ".join((DOCS_SOURCE / "usage.rst").read_text(encoding="utf-8").split())
    assert "Full evidence bundles and UI remain open." not in content
    assert "``export`` and ``verify-export`` are implemented" in content
    assert "PARTIAL_UI_BASIC_LOOP" in content
    assert "UI resume and record content checks are implemented" in content
    assert "UI reports, export and downloads are implemented" in content
    assert "Availability is advisory, not verification or download authorization" in content
    assert (
        "Neither recovery nor portable export has received new native Windows validation"
    ) in content
    assert "older Windows results do not validate these changes" in content
    assert (
        "Neither service proves historical execution authenticity, scientific validity"
    ) in content


def test_usage_and_architecture_do_not_call_implemented_export_unfinished() -> None:
    for filename in ("usage.rst", "architecture.rst"):
        content = " ".join((DOCS_SOURCE / filename).read_text(encoding="utf-8").split())
        assert "Full evidence bundles and UI remain open." not in content
        assert (
            "Full evidence bundles, general scientific validation, a user interface, "
            "and a public release remain unfinished."
        ) not in content
        assert "Portable evidence" in content
    architecture = " ".join((DOCS_SOURCE / "architecture.rst").read_text(encoding="utf-8").split())
    assert ("General scientific validation and a public release remain unfinished.") in architecture


def test_basic_ui_instructions_state_immutable_jobs_and_stop_boundary():
    for path in (REPOSITORY_ROOT / "docs/guide/reference.md", DOCS_SOURCE / "usage.rst"):
        content = path.read_text(encoding="utf-8")
        for text in (
            "selcal ui --workspace",
            "127.0.0.1",
            "Ctrl-C",
            "immutable",
            "PARTIAL_UI_BASIC_LOOP",
        ):
            assert text in content


def test_public_api_ui_summary_matches_implemented_controls_and_open_gates():
    content = " ".join((DOCS_SOURCE / "public_api.rst").read_text(encoding="utf-8").split())
    assert "UI recovery and evidence controls remain pending" not in content
    assert (
        "UI resume, record content checks, reports, evidence export and downloads are implemented"
        in content
    )
    assert "PARTIAL_UI_BASIC_LOOP" in content
    assert "Full integration and release validation remain separate gates" in content
    assert "not historical execution authentication or scientific validation" in content


def test_checkpoint_examples_use_the_example_compatible_storage_budget():
    for path in (REPOSITORY_ROOT / "docs/guide/reference.md", DOCS_SOURCE / "usage.rst"):
        content = path.read_text(encoding="utf-8")
        checkpoint_lines = [
            line.strip()
            for line in content.splitlines()
            if line.strip().startswith("selcal ")
            and ("--checkpoint run-checkpoint" in line or "selcal resume run-checkpoint" in line)
        ]
        assert len(checkpoint_lines) == 2
        assert all("--max-bytes 8388608" in line for line in checkpoint_lines)
