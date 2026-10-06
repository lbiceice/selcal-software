"""Disclosed orbit regression fixtures, not new scientific evidence."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import pytest
from _platform_support import requires_source_checkout
from test_m6_pearson_diagnostic_public import ROOT, driver, projected

ORBIT_CASES = [
    (f"orbit_{'_'.join(map(str, candidates))}_{rule}", candidates, rule)
    for candidates in ((2,), (1, 2), (1, 2, 3, 4))
    for rule in ("max_upper", "max_absolute")
]
SOURCE = tuple(float((i * i + 3 * i + 7) % 67) for i in range(64))
TARGET = tuple(SOURCE[(i - 2) % 64] for i in range(64))


def orbit_driver():
    mod = driver()
    assert hasattr(mod, "ORBIT_CASES"), "the six disclosed orbit fixtures are absent"
    assert list(mod.ORBIT_CASES) == [case for case, _, _ in ORBIT_CASES]
    return mod


@pytest.fixture(scope="module")
def orbit_runs():
    mod = orbit_driver()
    return {case: mod.run_case(case) for case, _, _ in ORBIT_CASES}


@pytest.mark.parametrize("case,candidates,rule", ORBIT_CASES)
def test_every_orbit_position_is_recountable(orbit_runs, case, candidates, rule):
    out = orbit_runs[case]
    assert out["scope"] == "CONDITIONAL_PUBLIC_STATE_REPLAY_NOT_RNG_VALIDATION"
    assert out["comparison_status"] == "AGREEMENT", out["mismatches"]
    assert out["mismatches"] == []
    assert out["input"]["source_hex"] == [v.hex() for v in SOURCE]
    assert out["input"]["target_hex"] == [v.hex() for v in TARGET]
    assert out["request"] == dict(
        candidates=candidates,
        statistic_name="lagged_pearson_v1",
        statistic_params={},
        selection_rule=rule,
        null_name="circular_shift_v2",
        null_params={"min_shift": 1},
        replicates=199,
        alpha=0.05,
        tie_tolerance=0.0,
        root_seed=17,
    )
    table = out["reference_exact"]
    assert table["status"] == "COMPLETE_TABLE"
    assert table["denominator"] == 64
    assert table["failure_count"] == 0
    assert [r["state_id"] for r in table["states"]] == list(range(64))
    assert [r["state"] for r in table["states"]] == list(range(64))
    assert [r["is_identity"] for r in table["states"]] == [True] + [False] * 63
    observed = out["observed"]
    assert observed["state"] == observed["state_id"] == 0
    assert observed["is_identity"] is True
    assert observed["candidate_comparisons"] == out["observed_comparison"]
    ref_a = observed["reference_selection"]["decision_statistic"]
    prod_a = float.fromhex(observed["production_selection"]["maximum_hex"])
    # The target is the source shifted by two, so the value is 1 in exact arithmetic. The
    # independent reference and production use different arithmetic: both give 1.0 on macOS,
    # while on Linux production gives 1 - 4e-16 (docs/status/evidence/linux_20260923). Both
    # must lie within 4 ULP of 1; the decision-level agreement is checked above.
    assert abs(ref_a - 1.0) <= 4 * math.ulp(1.0)
    assert abs(prod_a - 1.0) <= 4 * math.ulp(1.0)
    exact_tail = sum(r["selection"]["decision_statistic"] >= ref_a for r in table["states"])
    # Count all 64 labels, including identity; this is the exact floor C/N.
    assert exact_tail == table["numerator"] == len(candidates)
    assert table["p_exact"] == len(candidates) / 64
    assert sum(r["exceeds_observed"] for r in table["states"]) == exact_tail
    assert out["checked_observed_candidates"] == len(candidates)
    assert out["checked_replicates"] == len(out["replicates"]) == 199
    assert [r["replicate_id"] for r in out["replicates"]] == list(range(199))
    ref_indicators, prod_indicators = [], []
    negative_maxima = 0
    for row in [observed, *out["replicates"]]:
        assert row["mismatches"] == []
        assert row["is_identity"] is (row["state"] == 0)
        assert row["state_id"] == row["state"]
        comparisons = row["candidate_comparisons"]
        assert len(comparisons) == len(candidates)
        assert [r["candidate_id"] for r in comparisons] == list(candidates)
        assert [r["production_candidate_id"] for r in comparisons] == list(candidates)
        assert all(
            r["support_n"] == r["production_support_n"] == 64 - max(candidates) for r in comparisons
        )
        assert all(
            r["reference_failure"] is None and r["production_validity"] == "valid"
            for r in comparisons
        )
        assert all(r["mismatches"] == [] for r in comparisons)
        estimates = [float.fromhex(r["production_hex"]) for r in comparisons]
        scores = estimates if rule == "max_upper" else list(map(abs, estimates))
        assert [float.fromhex(r["production_score_hex"]) for r in comparisons] == scores
        assert [float.fromhex(r["expected_production_score_hex"]) for r in comparisons] == scores
        prod_max = float.fromhex(row["production_selection"]["maximum_hex"])
        assert prod_max == max(scores)
        if rule == "max_absolute" and prod_max > max(estimates):
            negative_maxima += 1
        ref_max = row["reference_selection"]["decision_statistic"]
        assert row["tail"]["reference_indicator"] is (ref_max >= ref_a)
        assert row["tail"]["production_indicator"] is (prod_max >= prod_a)
        if row is not observed:
            assert set(row["token_comparison"]) == {
                "schema",
                "null_name",
                "null_parameter_sha256",
                "semantic_input_sha256",
                "scientific_plan_sha256",
                "bound_null_owner_sha256",
                "state_schema",
                "is_identity",
            }
            assert all(check["matches"] for check in row["token_comparison"].values())
            assert row["reference_status"] == row["production_status"] == "complete"
            assert row["reference_failure_stage"] is row["production_failure_stage"] is None
            ref_indicators.append(ref_max >= ref_a)
            prod_indicators.append(prod_max >= prod_a)
    if rule == "max_absolute":
        assert negative_maxima > 0, "exercise a negative candidate attaining the absolute maximum"
    assert ref_indicators == prod_indicators
    e = sum(ref_indicators)
    for summary in (out["reference_summary"], out["production_summary"]):
        assert (summary["B"], summary["E"], summary["F"]) == (199, e, 0)
        assert summary["p"] == (1 + e) / 200
        assert summary["reject_null"] is (summary["p"] <= 0.05)
    assert list(out["reference_summary"]["indicators"]) == ref_indicators
    json.dumps(out, allow_nan=False)


def test_complete_request_reaches_actual_public_api_and_binds_identity(monkeypatch):
    from selcal.canonical import semantic_input_sha256
    from selcal.canonical_v2 import scientific_plan_v2_sha256

    mod = orbit_driver()
    original = mod.calibrate_selected_family
    captured = []

    def observe_call(pair, resolution):
        actual = original(pair, resolution)
        captured.append((pair, resolution.plan, actual))
        return actual

    monkeypatch.setattr(mod, "calibrate_selected_family", observe_call)
    outputs = mod.run_diagnostics(suite="orbit")
    assert len(captured) == len(outputs) == 6
    for out, (pair, plan, actual) in zip(outputs, captured, strict=True):
        assert all(check["matches"] for check in out["identity_comparison"].values())
        assert tuple(pair.source) == SOURCE and tuple(pair.target) == TARGET
        assert set(out["request"]) == {field.name for field in fields(mod.PlanRequestV2)}
        for name, value in out["request"].items():
            assert value == getattr(plan, name)
        assert out["semantic_input_sha256"] == semantic_input_sha256(pair, plan.candidates)
        assert out["scientific_plan_sha256"] == scientific_plan_v2_sha256(plan)
        assert out["semantic_input_sha256"] == actual.semantic_input_sha256
        assert out["scientific_plan_sha256"] == actual.scientific_plan_sha256


def test_finite_schedule_below_exact_floor_is_not_a_disagreement(orbit_runs):
    below = [
        out
        for out in orbit_runs.values()
        if out["production_summary"]["p"] < out["reference_exact"]["p_exact"]
    ]
    assert below, "the disclosed finite schedules exercise p below C/N"
    assert all(out["comparison_status"] == "AGREEMENT" for out in below)


def test_absolute_internal_maximum_check_detects_joint_signed_maximum_corruption():
    mod = orbit_driver()
    ref = mod.reference.scan(
        (0.0, 1.0, 2.0, 3.0, 4.0), (4.0, 3.0, 2.0, 1.0, 0.0), (1, 2), "max_absolute", 0.0
    )
    results = tuple(
        SimpleNamespace(
            candidate_id=r["candidate_id"],
            support_n=r["support_n"],
            estimate=r["estimate"],
            selection_score=abs(r["estimate"]),
            validity=SimpleNamespace(value="valid"),
            diagnostics=(),
        )
        for r in ref["candidate_records"]
    )
    selection = SimpleNamespace(**ref["selection"])
    _, mismatches = mod._compare_scan(ref, results, selection, rule="max_absolute")
    assert mismatches == []
    # Both reported maxima agree with each other, but violate max_absolute.
    corrupt = dict(ref, selection=dict(ref["selection"], decision_statistic=-1.0))
    selection.decision_statistic = -1.0
    _, mismatches = mod._compare_scan(corrupt, results, selection, rule="max_absolute")
    assert "PRODUCTION_MAXIMUM_INTERNAL_INCONSISTENCY" in mismatches
    assert "MAXIMUM_DISAGREEMENT" not in mismatches


@requires_source_checkout
def test_orbit_cli_writes_complete_report_and_never_overwrites(tmp_path):
    output = tmp_path / "orbit.json"
    cmd = [
        sys.executable,
        "-B",
        str(ROOT / "scripts/compare_pearson_diagnostic.py"),
        "--suite",
        "orbit",
        "--output",
        str(output),
    ]
    run = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
        errors="replace", timeout=45)
    assert run.returncode == 0, run.stderr
    before = output.read_bytes()
    report = json.loads(before)
    assert report["suite"] == "orbit"
    assert report["status"] == "LOCAL_DIAGNOSTIC_ONLY"
    assert report["scientific_study_executed"] is report["external_validation"] is False
    assert report["submission_ready"] is False
    assert len(report["cases"]) == 6
    assert all(len(c["replicates"]) == 199 for c in report["cases"])
    assert report["sources_before"] == report["sources_after"] == report["sources"]
    for source in report["sources"]:
        data = Path(source["path"]).read_bytes()
        assert source["bytes"] == len(data)
        assert source["sha256"] == hashlib.sha256(data).hexdigest()
    again = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
        errors="replace", timeout=45)
    assert again.returncode != 0
    assert output.read_bytes() == before


@requires_source_checkout
def test_orbit_cli_exits_nonzero_and_retains_disagreement_details(monkeypatch, tmp_path):
    mod = orbit_driver()
    original = mod.calibrate_selected_family

    def corrupt_call(*args):
        actual = original(*args)
        return projected(actual, reject_null=not actual.reject_null)

    output = tmp_path / "disagreement.json"
    monkeypatch.setattr(mod, "calibrate_selected_family", corrupt_call)
    monkeypatch.setattr(sys, "argv", ["diagnostic", "--suite", "orbit", "--output", str(output)])
    assert mod.main() == 1
    cases = json.loads(output.read_text(encoding="utf-8"))["cases"]
    assert len(cases) == 6
    assert all(c["comparison_status"] == "DISAGREEMENT" for c in cases)
    assert all("REJECT_NULL_DISAGREEMENT" in c["mismatches"] for c in cases)
    assert all(len(c["replicates"]) == 199 for c in cases)


@requires_source_checkout
def test_orbit_source_drift_is_still_rejected(monkeypatch, tmp_path):
    mod = orbit_driver()
    before = mod.source_inventory()
    after = [dict(before[0], sha256="0" * 64), *before[1:]]
    inventories = iter((before, after))
    monkeypatch.setattr(mod, "source_inventory", lambda: next(inventories))
    output = tmp_path / "drift.json"
    monkeypatch.setattr(sys, "argv", ["diagnostic", "--suite", "orbit", "--output", str(output)])
    with pytest.raises(RuntimeError, match="source bytes changed"):
        mod.main()
    assert not output.exists()


@pytest.mark.parametrize("position", ["observed", "replicate"])
@pytest.mark.parametrize("score", [-9.0, float("nan"), float("inf"), True, None])
def test_selection_score_corruption_is_detected_and_retained(monkeypatch, position, score):
    mod = driver()
    original = mod.calibrate_selected_family

    def changed(*args):
        actual = original(*args)
        if position == "observed":
            changed_results = (
                projected(actual.observed_results[0], selection_score=score),
                *actual.observed_results[1:],
            )
            return projected(actual, observed_results=changed_results)
        first = actual.replicates[0]
        changed_results = (
            projected(first.statistic_results[0], selection_score=score),
            *first.statistic_results[1:],
        )
        return projected(
            actual,
            replicates=(
                projected(first, statistic_results=changed_results),
                *actual.replicates[1:],
            ),
        )

    monkeypatch.setattr(mod, "calibrate_selected_family", changed)
    out = mod.run_case("orbit_1_2_max_absolute")
    assert out["comparison_status"] == "DISAGREEMENT"
    assert "SELECTION_SCORE_DISAGREEMENT" in out["mismatches"]
    row = out["observed"] if position == "observed" else out["replicates"][0]
    comparison = row["candidate_comparisons"][0]
    assert "SELECTION_SCORE_DISAGREEMENT" in comparison["mismatches"]
    assert comparison["production_score_hex"] == (score.hex() if type(score) is float else None)
    if type(score) is bool:
        assert comparison["production_score_value"] is score
    assert len(out["replicates"]) == 199
    assert out["replicates"][-1]["mismatches"] == []
    json.dumps(out, allow_nan=False)


@pytest.mark.parametrize("valid_candidate", [False, True])
def test_failed_scan_requires_all_candidate_scores_to_be_none(monkeypatch, valid_candidate):
    mod = driver()
    original = mod.calibrate_selected_family

    def changed(*args):
        actual = original(*args)
        rows = list(actual.replicates)
        index = next(i for i, r in enumerate(rows) if r.selection is None)
        failed = rows[index]
        results = list(failed.statistic_results)
        candidate = next(
            i for i, r in enumerate(results) if (r.validity.value == "valid") is valid_candidate
        )
        results[candidate] = projected(results[candidate], selection_score=0.0)
        rows[index] = projected(failed, statistic_results=tuple(results))
        return projected(actual, replicates=tuple(rows))

    monkeypatch.setattr(mod, "calibrate_selected_family", changed)
    out = mod.run_case("failed_candidates")
    assert out["comparison_status"] == "DISAGREEMENT"
    assert "SELECTION_SCORE_DISAGREEMENT" in out["mismatches"]
    assert out["production_summary"]["F"] == 4
    assert out["reference_summary"]["bounds"] == (0.5, 0.9)
    assert len(out["replicates"]) == 9


@pytest.mark.parametrize(
    "field,value",
    [
        ("alpha", 0.99),
        ("alpha", float("nan")),
        ("alpha", True),
        ("semantic_input_sha256", "0" * 64),
        ("scientific_plan_sha256", "0" * 64),
    ],
)
def test_returned_run_identity_must_match_actual_input_and_request(monkeypatch, field, value):
    mod = driver()
    original = mod.calibrate_selected_family

    def changed(*args):
        return projected(original(*args), **{field: value})

    monkeypatch.setattr(mod, "calibrate_selected_family", changed)
    out = mod.run_case("orbit_1_2_max_absolute")
    assert out["comparison_status"] == "DISAGREEMENT"
    assert "RUN_" + field.upper() + "_DISAGREEMENT" in out["mismatches"]
    assert out["identity_comparison"][field]["matches"] is False
    if field == "alpha":
        assert out["identity_comparison"][field]["expected"] == 0.05
    else:
        assert out["identity_comparison"][field]["production"] == value
        assert out["identity_comparison"][field]["expected"] != value
    assert len(out["replicates"]) == 199
    assert all(row["mismatches"] == [] for row in out["replicates"])
    json.dumps(out, allow_nan=False)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "wrong"),
        ("null_name", "block_shuffle_v2"),
        ("null_parameter_sha256", "0" * 64),
        ("semantic_input_sha256", "0" * 64),
        ("scientific_plan_sha256", "0" * 64),
        ("bound_null_owner_sha256", "0" * 64),
    ],
)
def test_emitted_token_identity_is_bound_and_retained(monkeypatch, field, value):
    mod = driver()
    original = mod.calibrate_selected_family

    def changed(*args):
        actual = original(*args)
        first = actual.replicates[0]
        token = projected(first.transform_token, **{field: value})
        return projected(
            actual, replicates=(projected(first, transform_token=token), *actual.replicates[1:])
        )

    monkeypatch.setattr(mod, "calibrate_selected_family", changed)
    out = mod.run_case("orbit_1_2_max_absolute")
    assert out["comparison_status"] == "DISAGREEMENT"
    mismatch = "TOKEN_" + field.upper() + "_DISAGREEMENT"
    assert mismatch in out["mismatches"]
    assert mismatch in out["replicates"][0]["mismatches"]
    check = out["replicates"][0]["token_comparison"][field]
    assert check["production"] == value and check["matches"] is False
    assert check["expected"] != value
    assert len(out["replicates"]) == 199
    assert out["replicates"][-1]["mismatches"] == []
    json.dumps(out, allow_nan=False)


def test_matching_corrupted_run_and_token_hashes_cannot_supply_the_expected_identity(monkeypatch):
    mod = driver()
    original = mod.calibrate_selected_family

    def changed(*args):
        actual = original(*args)
        rows = tuple(
            projected(
                row, transform_token=projected(row.transform_token, semantic_input_sha256="0" * 64)
            )
            for row in actual.replicates
        )
        return projected(actual, semantic_input_sha256="0" * 64, replicates=rows)

    monkeypatch.setattr(mod, "calibrate_selected_family", changed)
    out = mod.run_case("orbit_1_2_max_absolute")
    assert out["comparison_status"] == "DISAGREEMENT"
    assert "RUN_SEMANTIC_INPUT_SHA256_DISAGREEMENT" in out["mismatches"]
    assert out["mismatches"].count("TOKEN_SEMANTIC_INPUT_SHA256_DISAGREEMENT") == 199
    assert len(out["replicates"]) == 199


def test_identity_and_position_disagreements_survive_a_malformed_state(monkeypatch):
    mod = driver()
    original = mod.calibrate_selected_family

    def changed(*args):
        actual = original(*args)
        first = actual.replicates[0]
        token = projected(first.transform_token, state=None, null_name="block_shuffle_v2")
        return projected(
            actual,
            replicates=(
                projected(first, replicate_id=999, transform_token=token),
                *actual.replicates[1:],
            ),
        )

    monkeypatch.setattr(mod, "calibrate_selected_family", changed)
    out = mod.run_case("orbit_1_2_max_absolute")
    expected = {
        "REPLICATE_ID_DISAGREEMENT",
        "TOKEN_NULL_NAME_DISAGREEMENT",
        "INVALID_TRANSFORM_TOKEN",
    }
    assert expected <= set(out["mismatches"])
    assert expected <= set(out["replicates"][0]["mismatches"])
    assert out["replicates"][0]["token_comparison"]["null_name"]["production"] == "block_shuffle_v2"
    assert len(out["replicates"]) == 199
    assert out["replicates"][-1]["mismatches"] == []


def test_emitted_state_schema_is_bound(monkeypatch):
    mod = driver()
    original = mod.calibrate_selected_family

    def changed(*args):
        actual = original(*args)
        first = actual.replicates[0]
        token = projected(
            first.transform_token, state=projected(first.transform_token.state, schema="wrong")
        )
        return projected(
            actual, replicates=(projected(first, transform_token=token), *actual.replicates[1:])
        )

    monkeypatch.setattr(mod, "calibrate_selected_family", changed)
    out = mod.run_case("orbit_1_2_max_absolute")
    assert "TOKEN_STATE_SCHEMA_DISAGREEMENT" in out["mismatches"]
    assert out["replicates"][0]["token_comparison"]["state_schema"]["production"] == "wrong"
    assert len(out["replicates"]) == 199
