from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from _documentation import public_documentation_text

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BASIC_EXAMPLE = REPOSITORY_ROOT / "examples" / "basic_selection_aware_calibration.py"
FAIL_CLOSED_EXAMPLE = REPOSITORY_ROOT / "examples" / "fail_closed_not_evaluable.py"
BLOCK_SHUFFLE_EXAMPLE = REPOSITORY_ROOT / "examples" / "binned_nette_block_shuffle.py"
README = REPOSITORY_ROOT / "README.md"


def _run_basic_example() -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(BASIC_EXAMPLE)],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=15,
        check=False,
    )


def _run_fail_closed_example() -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(FAIL_CLOSED_EXAMPLE)],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=15,
        check=False,
    )


def _run_block_shuffle_example() -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(BLOCK_SHUFFLE_EXAMPLE)],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=15,
        check=False,
    )


def test_basic_example_runs_as_a_user_facing_process_with_exact_output() -> None:
    first = _run_basic_example()
    second = _run_basic_example()

    assert first.returncode == second.returncode == 0
    assert first.stderr == second.stderr == ""
    assert first.stdout == second.stdout
    assert json.loads(first.stdout) == {
        "decision_statistic": 0.981980506062,
        "exceedance_count": 1,
        "failure_count": 0,
        "p_value": 0.2,
        "planned_replicates": 9,
        "reject_null": False,
        "schema": "selcal.example.basic-selection-aware-calibration.v1",
        "scientific_plan_sha256": (
            "02548ce6fdc6f84e3912bc566b8a665d131b5e67816b44fbc1e5476fb006463b"
        ),
        "selected_candidate": 2,
        "status": "complete",
    }


def test_readme_routes_users_to_install_and_run_the_exact_example() -> None:
    text = public_documentation_text(REPOSITORY_ROOT)

    assert "## Install from source" in text
    assert "python -m pip install ." in text
    assert "## Run the deterministic example" in text
    assert "python examples/basic_selection_aware_calibration.py" in text
    assert (
        "This example is an executable contract demonstration, not M6 scientific evidence."
        in text
    )
    assert "## Inspect fail-closed behavior" in text
    assert "python examples/fail_closed_not_evaluable.py" in text
    assert "NOT_EVALUABLE is not evidence of no effect or non-significance." in text
    assert "## Exercise binned NetTE with block shuffle" in text
    assert "python examples/binned_nette_block_shuffle.py" in text
    assert "This broadens executable feature coverage; it is not a comparator or M6 result." in text


def test_fail_closed_example_reports_not_evaluable_without_a_p_value() -> None:
    first = _run_fail_closed_example()
    second = _run_fail_closed_example()

    assert first.returncode == second.returncode == 0
    assert first.stderr == second.stderr == ""
    assert first.stdout == second.stdout
    assert json.loads(first.stdout) == {
        "diagnostics": ["fewer_than_two_blocks_v2"],
        "failure_stage": "null_bind",
        "observed_result_count": 0,
        "p_value": None,
        "planned_replicates": 7,
        "reject_null": None,
        "replicate_count": 0,
        "schema": "selcal.example.fail-closed-not-evaluable.v1",
        "scientific_plan_sha256": (
            "20f75a7830a9492ca9b2de606325383a35e9a735c589e3c475caf2f1b9701a00"
        ),
        "status": "not_evaluable",
    }


def test_binned_nette_block_shuffle_example_runs_with_exact_output() -> None:
    first = _run_block_shuffle_example()
    second = _run_block_shuffle_example()

    assert first.returncode == second.returncode == 0
    assert first.stderr == second.stderr == ""
    assert first.stdout == second.stdout
    assert json.loads(first.stdout) == {
        "decision_statistic": 0.17328679514,
        "exceedance_count": 3,
        "failure_count": 0,
        "p_value": 0.4,
        "planned_replicates": 9,
        "reject_null": False,
        "replicate_count": 9,
        "schema": "selcal.example.binned-nette-block-shuffle.v1",
        "scientific_plan_sha256": (
            "e05be39cfa07814d97813b09aa887a26403abd9a881a3969616d29d89590405d"
        ),
        "selected_candidate": 1,
        "status": "complete",
    }
