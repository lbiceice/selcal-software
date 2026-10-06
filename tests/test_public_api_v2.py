from __future__ import annotations

import ast
import csv
import inspect
import json
import os
import pickle
import subprocess
import sys
import textwrap
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from _documentation import public_documentation_text

import selcal
import selcal.calibration_v2 as calibration_v2
import selcal.contracts_v2 as contracts_v2
from selcal.calibration_v2 import verify_calibration_result
from selcal.contracts import PlanRequest, SeriesPair
from selcal.contracts_v2 import (
    PlanRequestV2,
    PlanVersionError,
    ResolvedScientificPlanV2,
)
from selcal.inference import InvalidNullForInferenceError, calibrate_selected_family
from selcal.migration_v1_to_v2 import PlanMigrationV1ToV2, migrate_plan_v1_to_v2
from selcal.resolution import resolve_plan
from selcal.resolution_v2 import PlanResolutionV2, resolve_plan_v2

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXECUTION_BUDGET_PROFILE = (
    REPOSITORY_ROOT / "docs/architecture/in_memory_execution_budget_profile_20260828.csv"
)
EXECUTION_BUDGET_PROFILE_FIELDS = (
    "id",
    "null",
    "statistic",
    "B",
    "C",
    "N",
    "q",
    "S",
    "bins",
    "BC",
    "BS",
    "work",
    "trace_peak_B",
    "rss_delta_B",
    "wall_s",
    "status",
)

HISTORICAL_V1_EXPORTS = {
    "PlanRequest",
    "PlanResolution",
    "ResolvedScientificPlan",
    "resolve_plan",
}
APPROVED_V2_EXPORTS = {
    "PlanRequestV2",
    "ResolvedScientificPlanV2",
    "PlanResolutionV2",
    "PlanMigrationV1ToV2",
    "resolve_plan_v2",
    "migrate_plan_v1_to_v2",
    "calibrate_selected_family",
    "verify_calibration_result",
    # R11-01: the public analysis entry raises this for nulls without a validity guarantee.
    "InvalidNullForInferenceError",
}
FORBIDDEN_TOP_LEVEL_EXPORTS = {
    "_RESOLUTION_SEAL",
    "_V2_RESOLUTION_SEAL",
    "_MIGRATION_SEAL",
    "_PREPARATION_SEAL",
    "_prepare_calibration",
    "_execute_replicates",
    "_STATISTIC_REGISTRY",
    "_NULL_REGISTRY",
    "_STATISTIC_REGISTRY_V2",
    "_NULL_REGISTRY_V2",
    "AdapterRegistry",
    "ResolvedAdaptersV2",
    "_BoundCircularShiftV2",
    "_BoundBlockShuffleV2",
    "_BoundLaggedPearsonAdapter",
    "_BoundBinnedNetTEAdapter",
    "_apply_state",
    "_apply_block_order",
    "circular_token_sha256",
    "block_token_sha256",
    "ExactOracleResultV0",
    "exact_state_oracle_v0",
}


def _pair() -> SeriesPair:
    return SeriesPair(
        source=np.asarray((0.0, 3.0, 1.0, 2.0, 4.0, 5.0), dtype=np.float64),
        target=np.asarray((0.0, 1.0, 3.0, 2.0, 5.0, 4.0), dtype=np.float64),
    )


def _historical_checkpoint_payload() -> dict[str, object]:
    boundary = (REPOSITORY_ROOT / "docs/architecture/m0_m2_boundary.md").read_text(
        encoding="utf-8"
    )
    heading = "## 2026-08-27 SoftwareX contract refresh"
    remainder = boundary.split(heading, maxsplit=1)[1]
    assert "\n## " in remainder, "historical checkpoint must have an explicit section bound"
    section = remainder.split("\n## ", maxsplit=1)[0]
    payload_text = section.split("```json", maxsplit=1)[1].split("```", maxsplit=1)[0]
    payload = json.loads(payload_text)
    assert type(payload) is dict
    return cast(dict[str, object], payload)


def _v1_request() -> PlanRequest:
    return PlanRequest(
        candidates=(1, 2),
        statistic_name="lagged_pearson_v1",
        statistic_params={},
        selection_rule="max_upper",
        null_name="circular_shift_v1",
        null_params={"min_shift": 1},
        replicates=3,
        alpha=0.05,
        tie_tolerance=1e-12,
        root_seed=17,
        failure_policy="fail_closed_v1",
    )


def _v2_request() -> PlanRequestV2:
    return PlanRequestV2(
        candidates=(1, 2),
        statistic_name="lagged_pearson_v1",
        statistic_params={},
        selection_rule="max_upper",
        null_name="circular_shift_v2",
        null_params={"min_shift": 1},
        replicates=3,
        alpha=0.05,
        tie_tolerance=1e-12,
        root_seed=17,
    )


def test_top_level_public_allowlist_is_exact_and_resolves_canonical_objects() -> None:
    expected = HISTORICAL_V1_EXPORTS | APPROVED_V2_EXPORTS
    expected_objects: dict[str, object] = {
        "PlanRequestV2": PlanRequestV2,
        "ResolvedScientificPlanV2": ResolvedScientificPlanV2,
        "PlanResolutionV2": PlanResolutionV2,
        "PlanMigrationV1ToV2": PlanMigrationV1ToV2,
        "resolve_plan_v2": resolve_plan_v2,
        "migrate_plan_v1_to_v2": migrate_plan_v1_to_v2,
        "calibrate_selected_family": calibrate_selected_family,
        "verify_calibration_result": verify_calibration_result,
        "InvalidNullForInferenceError": InvalidNullForInferenceError,
    }

    assert set(selcal.__all__) == expected
    for name, expected_object in expected_objects.items():
        assert getattr(selcal, name, None) is expected_object


def test_verify_calibration_result_retains_picklable_public_function_identity() -> None:
    verifier = verify_calibration_result

    assert verifier.__module__ == "selcal.contracts_v2"
    assert verifier.__name__ == "verify_calibration_result"
    assert verifier.__qualname__ == "verify_calibration_result"
    assert str(inspect.signature(verifier)) == (
        "(result: 'object', resolution: 'object', /) -> 'None'"
    )
    assert all(
        parameter.kind is inspect.Parameter.POSITIONAL_ONLY
        for parameter in inspect.signature(verifier).parameters.values()
    )
    assert verifier.__annotations__ == {
        "result": "object",
        "resolution": "object",
        "return": "None",
    }
    assert verifier.__doc__
    assert verifier.__defaults__ is None
    assert verifier.__kwdefaults__ is None
    assert pickle.loads(pickle.dumps(verifier)) is verifier
    assert selcal.verify_calibration_result is verifier
    assert calibration_v2.verify_calibration_result is verifier
    assert contracts_v2.verify_calibration_result is verifier


def test_resolve_plan_v2_retains_module_level_public_function_identity() -> None:
    assert resolve_plan_v2.__module__ == "selcal.resolution_v2"
    assert resolve_plan_v2.__name__ == "resolve_plan_v2"
    assert resolve_plan_v2.__qualname__ == "resolve_plan_v2"
    assert str(inspect.signature(resolve_plan_v2)) == (
        "(request: 'PlanRequestV2', /) -> 'PlanResolutionV2'"
    )
    assert resolve_plan_v2.__annotations__ == {
        "request": "PlanRequestV2",
        "return": "PlanResolutionV2",
    }
    assert resolve_plan_v2.__doc__ is not None
    assert "Resolve one exact v2 request" in resolve_plan_v2.__doc__
    assert "built-in v2 registries" in resolve_plan_v2.__doc__


def test_resolve_plan_v2_pickle_round_trip_preserves_public_object_identity() -> None:
    assert pickle.loads(pickle.dumps(resolve_plan_v2)) is resolve_plan_v2


def test_resolve_plan_v2_public_identity_is_stable_in_fresh_process(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPOSITORY_ROOT / "src")
    for variable_name in tuple(environment):
        if variable_name == "COVERAGE_PROCESS_START" or variable_name.startswith(
            ("COV_CORE_", "COVERAGE_")
        ):
            environment.pop(variable_name)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import inspect
                import pickle
                from selcal.resolution_v2 import resolve_plan_v2

                failures = []
                if resolve_plan_v2.__module__ != "selcal.resolution_v2":
                    failures.append(("module", resolve_plan_v2.__module__))
                if resolve_plan_v2.__name__ != "resolve_plan_v2":
                    failures.append(("name", resolve_plan_v2.__name__))
                if resolve_plan_v2.__qualname__ != "resolve_plan_v2":
                    failures.append(("qualname", resolve_plan_v2.__qualname__))
                signature = str(inspect.signature(resolve_plan_v2))
                expected_signature = "(request: 'PlanRequestV2', /) -> 'PlanResolutionV2'"
                if signature != expected_signature:
                    failures.append(("signature", signature))
                expected_annotations = {
                    "request": "PlanRequestV2",
                    "return": "PlanResolutionV2",
                }
                if resolve_plan_v2.__annotations__ != expected_annotations:
                    failures.append(("annotations", resolve_plan_v2.__annotations__))
                if (
                    not resolve_plan_v2.__doc__
                    or "Resolve one exact v2 request" not in resolve_plan_v2.__doc__
                ):
                    failures.append(("doc", resolve_plan_v2.__doc__))
                try:
                    restored = pickle.loads(pickle.dumps(resolve_plan_v2))
                except Exception as error:
                    failures.append(("pickle", type(error).__name__, str(error)))
                else:
                    if restored is not resolve_plan_v2:
                        failures.append(("pickle_identity", restored))
                assert failures == [], failures
                """
            ),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=10,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_internal_execution_oracle_registry_and_transform_objects_stay_private() -> None:
    assert FORBIDDEN_TOP_LEVEL_EXPORTS.isdisjoint(selcal.__all__)
    assert not {name for name in FORBIDDEN_TOP_LEVEL_EXPORTS if hasattr(selcal, name)}


def test_only_resolution_v2_is_directly_executable_and_v1_is_typed_rejection() -> None:
    public_calibrator = cast(
        Callable[[SeriesPair, PlanResolutionV2], object],
        selcal.calibrate_selected_family,
    )
    v1_request = _v1_request()
    v1_resolution = resolve_plan(v1_request)
    v2_request = _v2_request()
    v2_resolution = resolve_plan_v2(v2_request)

    for inert in (v1_request, v1_resolution.plan, v2_request, v2_resolution.plan):
        assert not any(hasattr(inert, name) for name in ("calibrate", "execute", "run"))

    for foreign in (v1_request, v1_resolution.plan, v1_resolution, v2_request, v2_resolution.plan):
        with pytest.raises(PlanVersionError, match="exact PlanResolutionV2"):
            public_calibrator(_pair(), cast(PlanResolutionV2, foreign))

    assert tuple(inspect.signature(public_calibrator).parameters) == ("pair", "resolution")


def test_package_data_includes_both_boundary_and_approved_v2_architecture() -> None:
    pyproject = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    data_files = pyproject["tool"]["setuptools"]["data-files"]

    assert data_files["share/selcal/docs/architecture"] == [
        "docs/architecture/m0_m2_boundary.md",
        "docs/architecture/null_reselection_v2_design.md",
        "docs/architecture/in_memory_execution_budget_profile_20260828.csv",
    ]
    assert data_files["share/selcal/docs/provenance"] == [
        "docs/provenance/P3_READ_ONLY_BOUNDARY.md",
        "docs/provenance/SOURCE_ORIGIN.tsv",
    ]
    assert not {key for key in data_files if key.startswith("docs/")}


def test_in_memory_execution_budget_profile_is_complete_and_recomputable() -> None:
    assert EXECUTION_BUDGET_PROFILE.is_file()
    with EXECUTION_BUDGET_PROFILE.open(encoding="utf-8", newline="") as profile_file:
        reader = csv.DictReader(profile_file)
        assert tuple(reader.fieldnames or ()) == EXECUTION_BUDGET_PROFILE_FIELDS
        rows = tuple(reader)

    assert len(rows) == 32
    assert len({row["id"] for row in rows}) == 32
    assert {row["status"] for row in rows} == {"complete"}
    null_statistic_pairs: set[tuple[str, str]] = set()
    binned_bins: set[int] = set()
    parsed: list[dict[str, int]] = []
    for row in rows:
        values = {name: int(row[name]) for name in ("B", "C", "N", "S", "BC", "BS", "work")}
        parsed.append(values)
        assert values["BC"] == values["B"] * values["C"]
        assert values["BS"] == values["B"] * values["S"]
        assert values["work"] == values["B"] * (
            values["C"] * values["N"] + values["S"] + values["N"]
        )
        assert float(row["wall_s"]) > 0.0
        assert int(row["trace_peak_B"]) > 0
        assert int(row["rss_delta_B"]) > 0
        null_statistic_pairs.add((row["null"], row["statistic"]))
        if row["null"] == "circular_shift_v2":
            assert row["q"] == ""
            assert values["S"] == 1
        else:
            assert row["null"] == "block_shuffle_v2"
            assert int(row["q"]) == values["S"]
        if row["statistic"] == "equal_width_binned_nette_v1":
            binned_bins.add(int(row["bins"]))
        else:
            assert row["statistic"] == "lagged_pearson_v1"
            assert row["bins"] == ""

    assert null_statistic_pairs == {
        (null_name, statistic_name)
        for null_name in ("circular_shift_v2", "block_shuffle_v2")
        for statistic_name in ("lagged_pearson_v1", "equal_width_binned_nette_v1")
    }
    assert {32, 256} <= binned_bins
    assert any(values["B"] == 1_000 for values in parsed)
    assert any(values["BC"] == 5_000 for values in parsed)
    assert max(values["BS"] for values in parsed) >= 81_920
    assert max(values["work"] for values in parsed) >= 3_072_500


def test_budget_design_cites_transcribed_profile_and_preserves_measurement_limits() -> None:
    design = (REPOSITORY_ROOT / "docs/architecture/null_reselection_v2_design.md").read_text(
        encoding="utf-8"
    )
    normalized = " ".join(design.split()).casefold()

    assert "in_memory_execution_budget_profile_20260828.csv" in design
    assert "COORDINATOR_TRANSCRIBED_FROM_SUBAGENT_TOOL_OUTPUT" in design
    assert "8f86c5a" in design
    assert "apple m4 pro" in normalized
    assert "cpython 3.12.10" in normalized
    assert "numpy 2.2.6" in normalized
    assert "original stdout" in normalized and "here-doc" in normalized
    assert "not preserved as independent artifacts" in normalized
    assert "single observation per point" in normalized
    assert "ru_maxrss" in normalized and "high-water" in normalized
    assert "tracemalloc" in normalized and "native allocations" in normalized
    assert "variance" in normalized and "confidence interval" in normalized
    assert "transcribed evidence table, not raw data" in normalized


def test_fresh_wheel_install_exposes_namespaced_docs_through_distribution_metadata(
    tmp_path: Path,
) -> None:
    wheel_directory = tmp_path / "wheel"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--outdir",
            str(wheel_directory),
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=30,
    )
    wheels = tuple(wheel_directory.glob("*.whl"))
    assert len(wheels) == 1

    install_root = tmp_path / "installed"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-compile",
            "--no-deps",
            "--target",
            str(install_root),
            str(wheels[0]),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=30,
    )
    script = """
import csv
import io
from importlib import metadata
from pathlib import Path
import sys

distribution = metadata.distribution('selcal')
record = distribution.read_text('RECORD')
assert record is not None
recorded = tuple(row[0] for row in csv.reader(io.StringIO(record)))
data_root = Path(sys.argv[1])
for suffix in (
    'share/selcal/docs/architecture/m0_m2_boundary.md',
    'share/selcal/docs/architecture/null_reselection_v2_design.md',
    'share/selcal/docs/architecture/in_memory_execution_budget_profile_20260828.csv',
    'share/selcal/docs/provenance/P3_READ_ONLY_BOUNDARY.md',
    'share/selcal/docs/provenance/SOURCE_ORIGIN.tsv',
):
    matches = tuple(entry for entry in recorded if entry.replace('\\\\', '/').endswith(suffix))
    assert len(matches) == 1, (suffix, matches)
    assert (data_root / suffix).is_file(), (data_root, suffix)
design = (data_root / 'share/selcal/docs/architecture/null_reselection_v2_design.md').read_text(
    encoding='utf-8'
)
assert 'greater than 256 bits' in design
assert 'integer magnitude exceeds safe diagnostic envelope' in design
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(install_root)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    for variable_name in tuple(environment):
        if variable_name == "COVERAGE_PROCESS_START" or variable_name.startswith(
            ("COV_CORE_", "COVERAGE_")
        ):
            environment.pop(variable_name)
    subprocess.run(
        [sys.executable, "-c", script, str(install_root)],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=10,
    )


@pytest.mark.parametrize(
    "relative_path",
    ("README.md", "docs/architecture/m0_m2_boundary.md"),
)
def test_package_facing_text_uses_candidate_status_and_retains_every_hold(
    relative_path: str,
) -> None:
    text = (
        public_documentation_text(REPOSITORY_ROOT)
        if relative_path == "README.md"
        else (REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8")
    )
    upper = text.upper()

    assert "M0-M2 IMPLEMENTATION CANDIDATE / FINAL VERIFICATION PENDING" in upper
    if relative_path == "README.md":
        assert (
            "SCIENTIFIC EVIDENCE: BOUNDED SYNTHETIC STUDIES EXECUTED / "
            "GENERAL IMPACT AND INDEPENDENT-USER BENEFIT HOLD"
        ) in upper
    else:
        assert "M6 SCIENTIFIC IMPACT" in upper and "HOLD" in upper
    for downstream in ("UI", "RELEASE", "LICENSE", "DOI", "MANUSCRIPT"):
        assert downstream in upper
    if relative_path == "README.md":
        assert "LOCAL RESUME: REPLAY_BEFORE_CONTINUE" in upper
        assert "EVIDENCE BUNDLES: CONTENT-CHECKED / UI: PARTIAL_UI_BASIC_LOOP" in upper
        assert "SELCAL EXPORT RUN.SQLITE EVIDENCE --MAX-BYTES 8388608" in upper
        assert "SELCAL VERIFY-EXPORT EVIDENCE --MAX-BYTES 8388608" in upper
        assert "PUBLIC RELEASE PENDING" in upper
        assert "DOI: PENDING" in upper and "NOT SUBMITTED" in upper
    else:
        assert "NOT DUE" in upper
        assert "HISTORICAL RECORD" in upper and "SUPERSEDED" in upper
    for forbidden_claim in (
        "M0-M2 IN-MEMORY CONTRACT LOCALLY VERIFIED",
        "SCIENTIFICALLY CALIBRATED",
        "SOFTWAREX READY",
        "UPLOAD READY",
        "MINOR REVISION LIKELY",
    ):
        assert forbidden_claim not in upper


def test_top_level_surface_has_no_downstream_product_operations() -> None:
    forbidden_operations = {
        "save",
        "load",
        "resume",
        "report",
        "launch_ui",
        "publish",
        "release",
        "deposit_doi",
        "build_manuscript",
        "submit_softwarex",
    }
    assert forbidden_operations.isdisjoint(selcal.__all__)
    assert not {name for name in forbidden_operations if hasattr(selcal, name)}


def test_dated_softwarex_refresh_is_historical_and_current_gates_are_reconciled() -> None:
    payload = _historical_checkpoint_payload()
    assert payload["status"] == "historical_superseded_checkpoint"
    assert payload["closed_as_current_blockers"] == [
        "v2_design_written_approval",
        "full_reselection_implementation",
    ]
    gate_status = payload["current_gate_status"]
    assert type(gate_status) is dict
    assert set(gate_status) >= {
        "m6_impact_and_comparators",
        "public_release",
        "license",
        "research_data",
        "clean_install_and_documentation",
        "rights_and_author_facts",
        "live_portal",
    }
    assert all(
        type(status) is str and status.startswith("HOLD")
        for gate, status in gate_status.items()
        if gate != "m0_m2"
    )


def test_historical_checkpoint_has_bounded_machine_readable_status_fields() -> None:
    payload = _historical_checkpoint_payload()

    assert set(payload) == {
        "schema",
        "as_of",
        "status",
        "contract_validation",
        "closed_as_current_blockers",
        "current_gate_status",
    }
    assert payload["schema"] == "selcal.softwarex-contract-checkpoint.v1"
    assert payload["as_of"] == "2026-08-27"
    assert payload["status"] == "historical_superseded_checkpoint"
    assert payload["contract_validation"] == "HOLD"
    assert payload["closed_as_current_blockers"] == [
        "v2_design_written_approval",
        "full_reselection_implementation",
    ]
    assert payload["current_gate_status"] == {
        "m0_m2": "IMPLEMENTATION_CANDIDATE_FINAL_VERIFICATION_PENDING",
        "m6_impact_and_comparators": "HOLD",
        "public_release": "HOLD_NOT_DUE",
        "license": "HOLD_NOT_DUE",
        "research_data": "HOLD_NOT_DUE",
        "clean_install_and_documentation": "HOLD_NOT_DUE",
        "rights_and_author_facts": "HOLD_NOT_DUE",
        "live_portal": "HOLD_NOT_DUE",
    }


def test_readme_v2_example_is_self_contained_and_executable(tmp_path: Path) -> None:
    readme = public_documentation_text(REPOSITORY_ROOT)
    section = readme.split("The candidate v2 in-memory execution path is:", maxsplit=1)[1]
    code = section.split("```python", maxsplit=1)[1].split("```", maxsplit=1)[0]
    wheel_directory = tmp_path / "wheel"
    subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(wheel_directory)],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=30,
    )
    wheels = tuple(wheel_directory.glob("*.whl"))
    assert len(wheels) == 1
    install_root = tmp_path / "installed"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-compile",
            "--no-deps",
            "--target",
            str(install_root),
            str(wheels[0]),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=30,
    )
    script = f"""
from pathlib import Path
import sys
import selcal

install_root = Path(sys.argv[1]).resolve()
assert Path(selcal.__file__).resolve().is_relative_to(install_root)
{code}
assert result.planned_replicates <= 9
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(install_root)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    for variable_name in tuple(environment):
        if variable_name == "COVERAGE_PROCESS_START" or variable_name.startswith(
            ("COV_CORE_", "COVERAGE_")
        ):
            environment.pop(variable_name)
    subprocess.run(
        [sys.executable, "-c", script, str(install_root)],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=10,
    )


def test_readme_example_test_uses_fresh_installed_artifact_with_timeout() -> None:
    source = textwrap.dedent(
        inspect.getsource(test_readme_v2_example_is_self_contained_and_executable)
    )
    tree = ast.parse(source)
    subprocess_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "subprocess"
        and node.func.attr == "run"
    ]
    direct_exec_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "exec"
    ]

    assert subprocess_calls
    assert all(
        any(keyword.arg == "timeout" for keyword in call.keywords)
        for call in subprocess_calls
    )
    assert not direct_exec_calls
    assert "--target" in source
    assert "PYTHONPATH" in source
    assert "selcal.__file__" in source


def test_readme_fresh_install_subprocess_scrubs_parent_coverage_controls() -> None:
    source = textwrap.dedent(
        inspect.getsource(test_readme_v2_example_is_self_contained_and_executable)
    )

    assert "COVERAGE_PROCESS_START" in source
    assert "COV_CORE_" in source
    assert ".pop(" in source


def test_packaged_design_documents_the_oversized_integer_exception() -> None:
    design = (REPOSITORY_ROOT / "docs/architecture/null_reselection_v2_design.md").read_text(
        encoding="utf-8"
    )

    assert "greater than 256 bits" in design
    assert "before product or decimal materialization" in design
    assert "integer magnitude exceeds safe diagnostic envelope" in design


def test_package_text_distinguishes_plan_representability_from_executor_budget() -> None:
    readme = public_documentation_text(REPOSITORY_ROOT)
    boundary = (REPOSITORY_ROOT / "docs/architecture/m0_m2_boundary.md").read_text(
        encoding="utf-8"
    )
    design = (REPOSITORY_ROOT / "docs/architecture/null_reselection_v2_design.md").read_text(
        encoding="utf-8"
    )

    for text in (readme, boundary):
        assert "resolve successfully" in text.casefold()
        assert "does not imply" in text.casefold()
        assert "in-memory executor budget" in text.casefold()
    assert "representability cap" in design.casefold()
    assert "IN_MEMORY_EXECUTION_BUDGET_EXCEEDED_V2" in design
    assert "32-point pressure profile" in design
    assert "reference environment" in design.casefold()
    assert "adjacent-bin" in design.casefold()
    assert "HOLD" in design
