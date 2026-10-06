from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import pytest
from _platform_support import requires_source_checkout

ROOT = Path(__file__).resolve().parents[1]


def driver():
    path = ROOT / "scripts/compare_pearson_diagnostic.py"
    assert path.is_file(), "actual-public-API comparison driver is absent"
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location("compare_pearson_diagnostic", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


@pytest.mark.parametrize("case", ["reselection", "labelled_blocks", "failed_candidates"])
def test_actual_public_api_replay(case):
    out = driver().run_case(case)
    assert out["scope"] == "CONDITIONAL_PUBLIC_STATE_REPLAY_NOT_RNG_VALIDATION"
    assert out["comparison_status"] == "AGREEMENT", out["mismatches"]
    assert out["checked_observed_candidates"] == 2
    assert out["checked_replicates"] == 9
    assert [row["replicate_id"] for row in out["replicates"]] == list(range(9))
    assert all(len(row["candidate_comparisons"]) == 2 for row in out["replicates"])
    if case == "failed_candidates":
        assert out["reference_summary"]["F"] == out["production_summary"]["F"] == 4
        assert out["reference_summary"]["E"] == out["production_summary"]["E"] == 4
        assert out["reference_summary"]["p"] is out["production_summary"]["p"] is None
        assert out["reference_summary"]["bounds"] == (0.5, 0.9)


def test_small_scalar_residual_never_excuses_tail_mismatch():
    mod = driver()
    assert mod.tail_comparison(1.0, 1.0, 1.0, float.fromhex("0x1.fffffffffffffp-1")) == {
        "reference_indicator": True,
        "production_indicator": False,
        "mismatch": "TAIL_INDICATOR_DISAGREEMENT",
    }


def test_reference_runs_with_product_and_numpy_imports_blocked():
    path = ROOT / "scripts/m6_pearson_diagnostic.py"
    imports = ast.walk(ast.parse(path.read_text(encoding="utf-8")))
    roots = []
    for node in imports:
        if isinstance(node, ast.Import):
            roots.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            roots.append(node.module.split(".")[0])
    assert set(roots) <= {"__future__", "math", "itertools"}
    code = """
import importlib.abc, importlib.util, json, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('selcal', 'numpy', 'scipy'):
            raise RuntimeError('forbidden reference dependency: '+fullname)
sys.meta_path.insert(0, Block())
spec = importlib.util.spec_from_file_location('ref', sys.argv[1])
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)
r = ref.exact((0.,3.,1.,2.,4.,5.), (0.,1.,3.,2.,5.,4.), (1,2))
print(json.dumps({'p':r['p_exact'], 'count':len(r['states'])}))
"""
    run = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code, str(path)],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=15,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout) == {"p": 0.5, "count": 6}


def projected(record, **changes):
    values = {field.name: getattr(record, field.name) for field in fields(record)}
    values.update(changes)
    return SimpleNamespace(**values)


def test_real_result_decision_mutation_is_detected(monkeypatch):
    mod = driver()
    original = mod.calibrate_selected_family

    def changed(*args):
        actual = original(*args)
        return projected(actual, reject_null=not actual.reject_null)

    monkeypatch.setattr(mod, "calibrate_selected_family", changed)
    out = mod.run_case("reselection")
    assert out["comparison_status"] == "DISAGREEMENT"
    assert "REJECT_NULL_DISAGREEMENT" in out["mismatches"]


def test_real_failure_status_and_stage_mutations_are_detected(monkeypatch):
    from selcal.contracts import ReplicateStatus, RunStatus

    mod = driver()
    original = mod.calibrate_selected_family

    def changed(*args):
        actual = original(*args)
        rows = tuple(
            projected(row, status=ReplicateStatus.COMPLETE, failure_stage=None)
            for row in actual.replicates
        )
        return projected(actual, status=RunStatus.COMPLETE, failure_stage=None, replicates=rows)

    monkeypatch.setattr(mod, "calibrate_selected_family", changed)
    out = mod.run_case("failed_candidates")
    assert out["comparison_status"] == "DISAGREEMENT"
    assert "RUN_STATUS_DISAGREEMENT" in out["mismatches"]
    assert "RUN_FAILURE_STAGE_DISAGREEMENT" in out["mismatches"]
    assert out["mismatches"].count("REPLICATE_STATUS_DISAGREEMENT") == 4
    assert out["mismatches"].count("REPLICATE_FAILURE_STAGE_DISAGREEMENT") == 4


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True])
def test_nonfinite_or_nonfloat_production_estimate_cannot_pass(monkeypatch, value):
    mod = driver()
    original = mod.calibrate_selected_family

    def changed(*args):
        actual = original(*args)
        rows = (actual.observed_results[0], projected(actual.observed_results[1], estimate=value))
        return projected(actual, observed_results=rows)

    monkeypatch.setattr(mod, "calibrate_selected_family", changed)
    out = mod.run_case("reselection")
    assert out["comparison_status"] == "DISAGREEMENT"
    assert "INVALID_PRODUCTION_ESTIMATE" in out["mismatches"]
    json.dumps(out, allow_nan=False)


def test_malformed_token_retains_failed_position_and_later_rows(monkeypatch):
    mod = driver()
    original = mod.calibrate_selected_family

    def changed(*args):
        actual = original(*args)
        rows = (projected(actual.replicates[0], transform_token=None), *actual.replicates[1:])
        return projected(actual, replicates=rows)

    monkeypatch.setattr(mod, "calibrate_selected_family", changed)
    out = mod.run_case("reselection")
    assert out["comparison_status"] == "DISAGREEMENT"
    assert "INVALID_TRANSFORM_TOKEN" in out["mismatches"]
    assert len(out["replicates"]) == 9
    assert out["replicates"][0]["replicate_id"] == 0
    assert out["reference_summary"] is None


@requires_source_checkout
def test_cli_writes_report_and_never_overwrites(tmp_path):
    output = tmp_path / "diagnostic.json"
    cmd = [
        sys.executable,
        "-B",
        str(ROOT / "scripts/compare_pearson_diagnostic.py"),
        "--output",
        str(output),
    ]
    run = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
        errors="replace", timeout=60)
    assert run.returncode == 0, run.stderr
    before = output.read_bytes()
    report = json.loads(before)
    assert report["scientific_study_executed"] is False
    assert len(report["cases"]) == 3
    assert all(c["comparison_status"] == "AGREEMENT" for c in report["cases"])
    again = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
        errors="replace", timeout=60)
    assert again.returncode != 0
    assert output.read_bytes() == before


def test_public_call_failure_does_not_discard_other_cases(monkeypatch):
    mod = driver()
    assert callable(getattr(mod, "run_diagnostics", None)), "per-case failure custody absent"
    original = mod.run_case
    seen = []

    def fail_one(case):
        seen.append(case)
        if case == "labelled_blocks":
            raise ValueError("bounded diagnostic public-call failure")
        return original(case)

    monkeypatch.setattr(mod, "run_case", fail_one)
    rows = mod.run_diagnostics()
    assert seen == list(mod.CASES)
    assert [r["comparison_status"] for r in rows] == ["AGREEMENT", "EXECUTION_FAILURE", "AGREEMENT"]
    assert rows[1]["exception_type"] == "ValueError"
    assert rows[1]["mismatches"] == ["PUBLIC_DIAGNOSTIC_EXECUTION_FAILURE"]


@pytest.mark.parametrize("value", [float("nan"), float("-inf"), True, 2.0])
def test_invalid_production_maximum_is_retained_as_disagreement(monkeypatch, value):
    mod = driver()
    original = mod.calibrate_selected_family

    def changed(*args):
        actual = original(*args)
        selection = projected(actual.observed_selection, decision_statistic=value)
        return projected(actual, observed_selection=selection)

    monkeypatch.setattr(mod, "calibrate_selected_family", changed)
    out = mod.run_case("reselection")
    assert out["comparison_status"] == "DISAGREEMENT"
    assert "INVALID_PRODUCTION_MAXIMUM" in out["mismatches"]
    json.dumps(out, allow_nan=False)
