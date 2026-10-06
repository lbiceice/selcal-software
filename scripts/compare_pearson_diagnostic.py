"""Run disclosed, bounded Pearson diagnostics through the actual public API.

The comparison is conditional on emitted labelled states, not an RNG check.
No result from this driver is a confirmatory study or an external human return.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from datetime import UTC, datetime
from pathlib import Path

import m6_pearson_diagnostic as reference
import numpy as np

import selcal
from selcal import PlanRequestV2, calibrate_selected_family, resolve_plan_v2
from selcal.canonical import semantic_input_sha256
from selcal.canonical_v2 import scientific_plan_v2_sha256
from selcal.contracts import SeriesPair

CASES = {
    "reselection": (
        (0.0, 3.0, 1.0, 2.0, 4.0, 5.0),
        (0.0, 1.0, 3.0, 2.0, 5.0, 4.0),
        "circular_shift_v2",
        1,
    ),
    "labelled_blocks": (
        (0.0, 1.0, 0.0, 1.0, 2.0, 3.0),
        (0.0, 2.0, 1.0, 4.0, 3.0, 5.0),
        "block_shuffle_v2",
        2,
    ),
    "failed_candidates": (
        (0.0, 0.0, 1.0, 2.0, 0.0, 0.0),
        (0.0, 1.0, 2.0, 3.0, 4.0, 5.0),
        "circular_shift_v2",
        1,
    ),
}
ORBIT_CASES = {
    f"orbit_{'_'.join(map(str, candidates))}_{rule}": (candidates, rule)
    for candidates in ((2,), (1, 2), (1, 2, 3, 4))
    for rule in ("max_upper", "max_absolute")
}
TOLERANCE = 64 * math.ulp(1.0)


def tail_comparison(ref_observed, ref_state, prod_observed, prod_state):
    ref_indicator = None if ref_state is None or ref_observed is None else ref_state >= ref_observed
    prod_indicator = (
        None if prod_state is None or prod_observed is None else prod_state >= prod_observed
    )
    return dict(
        reference_indicator=ref_indicator,
        production_indicator=prod_indicator,
        mismatch=None if ref_indicator is prod_indicator else "TAIL_INDICATOR_DISAGREEMENT",
    )


def _compare_scan(reference_scan, results, selection, *, rule="max_upper"):
    mismatches = []
    comparisons = []
    if len(results) != len(reference_scan["candidate_records"]):
        mismatches.append("CANDIDATE_COUNT_DISAGREEMENT")
    for ref, prod in zip(reference_scan["candidate_records"], results, strict=False):
        failures = []
        if ref["candidate_id"] != prod.candidate_id or ref["support_n"] != prod.support_n:
            failures.append("CANDIDATE_SUPPORT_DISAGREEMENT")
        expected_valid = ref["failure_reason"] is None
        if expected_valid != (prod.validity.value == "valid"):
            failures.append("VALIDITY_DISAGREEMENT")
        if not expected_valid:
            expected_code = "SELCAL_LAGGED_PEARSON_" + ref["failure_reason"]
            if expected_code not in prod.diagnostics:
                failures.append("FAILURE_REASON_DISAGREEMENT")
        rv, pv = ref["estimate"], prod.estimate
        valid_production_number = type(pv) is float and math.isfinite(pv) and abs(pv) <= 1.0
        if pv is not None and not valid_production_number:
            failures.append("INVALID_PRODUCTION_ESTIMATE")
        residual = None if rv is None or not valid_production_number else abs(rv - pv)
        if ((rv is None) != (pv is None)) or (residual is not None and residual > TOLERANCE):
            failures.append("ESTIMATE_DISAGREEMENT")
        # A failed scan remains raw: even its valid candidates must have no score.
        scored_scan = reference_scan["selection"] is not None
        expected_score = (
            (abs(pv) if rule == "max_absolute" else pv)
            if scored_scan and valid_production_number
            else None
        )
        reference_score = (abs(rv) if rule == "max_absolute" else rv) if scored_scan else None
        score = prod.selection_score
        valid_score = type(score) is float and math.isfinite(score)
        if score is not None and not valid_score:
            failures.append("INVALID_PRODUCTION_SELECTION_SCORE")
        if (
            type(score) is not type(expected_score)
            or score != expected_score
            or (score is not None and not valid_score)
        ):
            failures.append("SELECTION_SCORE_DISAGREEMENT")
        comparisons.append(
            dict(
                candidate_id=ref["candidate_id"],
                support_n=ref["support_n"],
                production_candidate_id=prod.candidate_id,
                production_support_n=prod.support_n,
                production_validity=prod.validity.value,
                reference_score_hex=None if reference_score is None else reference_score.hex(),
                expected_production_score_hex=(
                    None if expected_score is None else expected_score.hex()
                ),
                production_score_hex=score.hex() if type(score) is float else None,
                production_score_value=score if type(score) in (bool, int, str) else None,
                production_score_type=type(score).__name__,
                reference_hex=None if rv is None else rv.hex(),
                production_hex=pv.hex() if valid_production_number else None,
                production_value_kind=(
                    "NULL"
                    if pv is None
                    else "FINITE_CORRELATION"
                    if valid_production_number
                    else "INVALID_FLOAT"
                    if type(pv) is float
                    else "NON_FLOAT"
                ),
                residual_hex=None if residual is None else residual.hex(),
                reference_failure=ref["failure_reason"],
                production_diagnostics=prod.diagnostics,
                mismatches=failures,
            )
        )
        mismatches.extend(failures)
    expected = reference_scan["selection"]
    if (expected is None) != (selection is None):
        mismatches.append("SELECTION_VALIDITY_DISAGREEMENT")
    if selection is not None and expected is not None:
        maximum = selection.decision_statistic
        if type(maximum) is not float or not math.isfinite(maximum) or abs(maximum) > 1.0:
            mismatches.append("INVALID_PRODUCTION_MAXIMUM")
        for field in ("selected_index", "selected_candidate", "tied_candidates"):
            if expected[field] != getattr(selection, field):
                mismatches.append("SELECTION_" + field.upper() + "_DISAGREEMENT")
        if abs(expected["decision_statistic"] - selection.decision_statistic) > TOLERANCE:
            mismatches.append("MAXIMUM_DISAGREEMENT")
        if not all(
            type(r.estimate) is float and math.isfinite(r.estimate) for r in results
        ) or selection.decision_statistic != max(
            abs(r.estimate) if rule == "max_absolute" else r.estimate for r in results
        ):
            mismatches.append("PRODUCTION_MAXIMUM_INTERNAL_INCONSISTENCY")
    return comparisons, mismatches


def _production_selection(selection):
    if selection is None:
        return None
    maximum = selection.decision_statistic
    valid = type(maximum) is float and math.isfinite(maximum) and abs(maximum) <= 1.0
    return dict(
        selected_candidate=selection.selected_candidate,
        selected_index=selection.selected_index,
        tied_candidates=selection.tied_candidates,
        maximum_hex=maximum.hex() if valid else None,
        maximum_value_kind="FINITE_CORRELATION" if valid else "INVALID_MAXIMUM",
    )


def _compare_identity(expected, production, prefix):
    """Retain both sides of the existing scalar identity fields, including bad types."""
    comparisons, mismatches = {}, []
    for field, expected_value in expected.items():
        value = production.get(field)
        matches = type(value) is type(expected_value) and value == expected_value
        comparisons[field] = dict(
            expected=expected_value,
            production=(
                value
                if (
                    type(value) in (str, bool, int)
                    or (type(value) is float and math.isfinite(value))
                )
                else None
            ),
            production_hex=value.hex() if type(value) is float else None,
            production_type=type(value).__name__,
            matches=matches,
        )
        if not matches:
            mismatches.append(prefix + "_" + field.upper() + "_DISAGREEMENT")
    return comparisons, mismatches


def run_case(case: str) -> dict:
    """Compare one fixed development fixture without modifying its result."""
    if case in ORBIT_CASES:
        candidates, rule = ORBIT_CASES[case]
        x = tuple(float((i * i + 3 * i + 7) % 67) for i in range(64))
        y = tuple(x[(i - 2) % 64] for i in range(64))
        null, parameter, replicates, tolerance = "circular_shift_v2", 1, 199, 0.0
    else:
        x, y, null, parameter = CASES[case]
        candidates, rule, replicates, tolerance = (1, 2), "max_upper", 9, 1e-12
    null_params = {"min_shift" if null == "circular_shift_v2" else "block_length": parameter}
    request_parameters = dict(
        candidates=candidates,
        statistic_name="lagged_pearson_v1",
        statistic_params={},
        selection_rule=rule,
        null_name=null,
        null_params=null_params,
        replicates=replicates,
        alpha=0.05,
        tie_tolerance=tolerance,
        root_seed=17,
    )
    return run_pair(case, x, y, request_parameters)


def run_pair(case: str, x, y, request_parameters: dict) -> dict:
    """Compare a supplied bounded pair against the independent labelled table."""
    request = PlanRequestV2(**request_parameters)
    candidates, rule = request.candidates, request.selection_rule.value
    null, null_params = request.null_name, dict(request.null_params)
    parameter = null_params["min_shift" if null == "circular_shift_v2" else "block_length"]
    replicates, tolerance = request.replicates, request.tie_tolerance
    x, y = tuple(float(v) for v in x), tuple(float(v) for v in y)
    pair = SeriesPair(
        source=np.asarray(x, dtype=np.float64), target=np.asarray(y, dtype=np.float64)
    )
    resolution = resolve_plan_v2(request)
    expected_run = dict(
        alpha=request.alpha,
        semantic_input_sha256=semantic_input_sha256(pair, resolution.plan.candidates),
        scientific_plan_sha256=scientific_plan_v2_sha256(resolution.plan),
    )
    # Bind the resolved null to independently computed input/plan identities.
    bound = resolution.adapters.null_model.bind(
        pair,
        semantic_input_sha256=expected_run["semantic_input_sha256"],
        scientific_plan_sha256=expected_run["scientific_plan_sha256"],
    ).bound
    identity_token = bound.identity_token()
    expected_token = {
        field: getattr(identity_token, field)
        for field in (
            "schema",
            "null_name",
            "null_parameter_sha256",
            "semantic_input_sha256",
            "scientific_plan_sha256",
            "bound_null_owner_sha256",
        )
    }
    expected_token["state_schema"] = identity_token.state.schema
    actual = calibrate_selected_family(pair, resolution)
    identity_comparison, identity_failures = _compare_identity(
        expected_run, {field: getattr(actual, field, None) for field in expected_run}, "RUN"
    )
    table = reference.exact(
        x,
        y,
        candidates,
        rule=rule,
        tolerance=tolerance,
        null_name=null,
        null_parameter=parameter,
    )
    by_state = {row["state"]: row for row in table["states"]}
    observed = table["states"][0]
    ref_a = observed["selection"]["decision_statistic"] if observed["selection"] else None
    prod_a = actual.observed_selection.decision_statistic if actual.observed_selection else None
    observed_comparison, mismatches = _compare_scan(
        observed, actual.observed_results, actual.observed_selection, rule=rule
    )
    observed_record = dict(
        state_id=observed["state_id"],
        state=observed["state"],
        is_identity=True,
        candidate_comparisons=observed_comparison,
        reference_selection=observed["selection"],
        production_selection=_production_selection(actual.observed_selection),
        tail=tail_comparison(ref_a, ref_a, prod_a, prod_a),
        mismatches=list(mismatches),
    )
    mismatches.extend(identity_failures)
    expected_attempts = replicates if ref_a is not None else 0
    if len(actual.replicates) != expected_attempts or actual.planned_replicates != replicates:
        mismatches.append("PLANNED_REPLICATE_COUNT_DISAGREEMENT")
    rows, maxima = [], []
    for expected_id, replicate in enumerate(actual.replicates):
        row_failures = []
        if replicate.replicate_id != expected_id:
            row_failures.append("REPLICATE_ID_DISAGREEMENT")
        token = replicate.transform_token
        state_object = getattr(token, "state", None)
        token_values = {field: getattr(token, field, None) for field in expected_token}
        token_values["state_schema"] = getattr(state_object, "schema", None)
        token_comparison, token_failures = _compare_identity(expected_token, token_values, "TOKEN")
        row_failures.extend(token_failures)
        state = getattr(
            state_object, "shift" if null == "circular_shift_v2" else "block_order", None
        )
        valid_state = (
            type(state) is int
            if null == "circular_shift_v2"
            else type(state) is tuple and all(type(label) is int for label in state)
        )
        if not valid_state or type(getattr(token, "is_identity", None)) is not bool:
            row_failures.append("INVALID_TRANSFORM_TOKEN")
            mismatches.extend(row_failures)
            rows.append(
                dict(
                    replicate_id=replicate.replicate_id,
                    state=None,
                    token_comparison=token_comparison,
                    candidate_comparisons=[],
                    mismatches=row_failures,
                )
            )
            continue
        if state not in by_state:
            row_failures.append("STATE_MEMBERSHIP_DISAGREEMENT")
            mismatches.extend(row_failures)
            rows.append(
                dict(
                    replicate_id=replicate.replicate_id,
                    state=state,
                    token_comparison=token_comparison,
                    mismatches=row_failures,
                )
            )
            continue
        ref = by_state[state]
        identity_flag, identity_flag_failures = _compare_identity(
            {"is_identity": ref["is_identity"]}, {"is_identity": token.is_identity}, "TOKEN"
        )
        token_comparison.update(identity_flag)
        row_failures.extend(identity_flag_failures)
        expected_status = "complete" if ref["selection"] is not None else "analytic_failure"
        expected_stage = None if ref["selection"] is not None else "statistic_scan"
        actual_stage = None if replicate.failure_stage is None else replicate.failure_stage.value
        if replicate.status.value != expected_status:
            row_failures.append("REPLICATE_STATUS_DISAGREEMENT")
        if actual_stage != expected_stage:
            row_failures.append("REPLICATE_FAILURE_STAGE_DISAGREEMENT")
        if replicate.transform_token.is_identity != ref["is_identity"]:
            row_failures.append("IDENTITY_DISAGREEMENT")
        candidate_comparisons, local = _compare_scan(
            ref, replicate.statistic_results, replicate.selection, rule=rule
        )
        row_failures.extend(local)
        ref_max = ref["selection"]["decision_statistic"] if ref["selection"] else None
        prod_max = replicate.selection.decision_statistic if replicate.selection else None
        tails = tail_comparison(ref_a, ref_max, prod_a, prod_max)
        if tails["mismatch"]:
            row_failures.append(tails["mismatch"])
        if ref["is_identity"] and prod_max != prod_a:
            row_failures.append("IDENTITY_MAXIMUM_DISAGREEMENT")
        maxima.append(ref_max)
        rows.append(
            dict(
                replicate_id=replicate.replicate_id,
                state_id=ref["state_id"],
                state=state,
                is_identity=replicate.transform_token.is_identity,
                token_comparison=token_comparison,
                reference_status=expected_status,
                production_status=replicate.status.value,
                reference_failure_stage=expected_stage,
                production_failure_stage=actual_stage,
                candidate_comparisons=candidate_comparisons,
                tail=tails,
                reference_selection=ref["selection"],
                production_selection=_production_selection(replicate.selection),
                mismatches=row_failures,
            )
        )
        mismatches.extend(row_failures)
    summary = (
        reference.summarize_tail(ref_a, tuple(maxima))
        if ref_a is not None and len(maxima) == actual.planned_replicates and maxima
        else None
    )
    if ref_a is None:
        # The independent observed scan fails before any replicate is attempted.
        # B remains the requested budget; zero E/F are not completed null draws.
        summary = dict(B=replicates, E=0, F=0, indicators=(), bounds=None, p=None)
    prod_summary = dict(
        B=actual.planned_replicates,
        E=actual.exceedance_count,
        F=actual.failure_count,
        p=actual.p_value,
        bounds=None
        if actual.failure_count == 0
        else (actual.exceedance_bound_low, actual.exceedance_bound_high),
    )
    raw_bounds = {
        field: getattr(actual, field) for field in ("exceedance_bound_low", "exceedance_bound_high")
    }
    prod_summary["raw_bounds"] = {
        field: dict(
            production=(
                value
                if type(value) in (str, bool, int)
                or (type(value) is float and math.isfinite(value))
                else None
            ),
            production_hex=value.hex() if type(value) is float else None,
            production_type=type(value).__name__,
        )
        for field, value in raw_bounds.items()
    }
    if summary is not None:
        for field in ("B", "E", "F", "p"):
            if (
                type(summary[field]) is not type(prod_summary[field])
                or summary[field] != prod_summary[field]
            ):
                mismatches.append("SUMMARY_" + field.upper() + "_DISAGREEMENT")
        expected_bounds = summary["bounds"] if summary["F"] else (None, None)
        bound_comparison, bound_failures = _compare_identity(
            dict(zip(raw_bounds, expected_bounds, strict=True)), raw_bounds, "RAW_BOUND"
        )
        prod_summary["raw_bounds"] = bound_comparison
        if bound_failures:
            mismatches.append("FAILURE_BOUND_DISAGREEMENT")
        expected_status = "complete" if ref_a is not None and summary["F"] == 0 else "not_evaluable"
        expected_stage = (
            "observed_statistic_scan"
            if ref_a is None
            else None if summary["F"] == 0 else "replicate_execution"
        )
        expected_decision = None if summary["p"] is None else summary["p"] <= request.alpha
        if actual.status.value != expected_status:
            mismatches.append("RUN_STATUS_DISAGREEMENT")
        if (None if actual.failure_stage is None else actual.failure_stage.value) != expected_stage:
            mismatches.append("RUN_FAILURE_STAGE_DISAGREEMENT")
        if actual.reject_null is not expected_decision:
            mismatches.append("REJECT_NULL_DISAGREEMENT")
        summary.update(
            status=expected_status, failure_stage=expected_stage, reject_null=expected_decision
        )
    else:
        mismatches.append("OBSERVED_OR_SCHEDULE_UNEVALUABLE")
    prod_summary.update(
        status=actual.status.value,
        failure_stage=None if actual.failure_stage is None else actual.failure_stage.value,
        reject_null=actual.reject_null,
    )
    return dict(
        case=case,
        scope="CONDITIONAL_PUBLIC_STATE_REPLAY_NOT_RNG_VALIDATION",
        comparison_status="AGREEMENT" if not mismatches else "DISAGREEMENT",
        checked_observed_candidates=len(observed_comparison),
        checked_replicates=len(rows),
        input=dict(
            source_hex=[x.hex() for x in x],
            target_hex=[v.hex() for v in y],
            candidates=candidates,
            selection_rule=rule,
            null_name=null,
            null_params=null_params,
            B=replicates,
            alpha=request.alpha,
            seed=request.root_seed,
            tie_tolerance=tolerance,
        ),
        request=request_parameters,
        identity_comparison=identity_comparison,
        semantic_input_sha256=actual.semantic_input_sha256,
        scientific_plan_sha256=actual.scientific_plan_sha256,
        observed_comparison=observed_comparison,
        observed=observed_record,
        replicates=rows,
        reference_exact=table,
        reference_summary=summary,
        production_summary=prod_summary,
        production_status=actual.status.value,
        mismatches=mismatches,
    )


def source_inventory():
    root = Path(__file__).resolve().parents[1]
    if Path(selcal.__file__).resolve() != root / "src/selcal/__init__.py":
        raise RuntimeError("loaded SelCal is not the scoped worktree")
    paths = sorted((root / "src/selcal").rglob("*.py"))
    paths += [Path(__file__).resolve(), Path(reference.__file__).resolve()]
    return [
        dict(path=str(p), bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        for p in paths
    ]


def run_diagnostics(suite="basic"):
    """Keep one outcome per disclosed case, even when a public call raises."""
    if suite not in ("basic", "orbit"):
        raise ValueError("unknown diagnostic suite")
    cases = []
    for case in CASES if suite == "basic" else ORBIT_CASES:
        try:
            cases.append(run_case(case))
        except Exception as error:
            # Do not catch interruption/SystemExit or serialize arbitrary exception payloads.
            cases.append(
                dict(
                    case=case,
                    comparison_status="EXECUTION_FAILURE",
                    exception_type=type(error).__name__[:120],
                    mismatches=["PUBLIC_DIAGNOSTIC_EXECUTION_FAILURE"],
                )
            )
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("basic", "orbit"), default="basic")
    parser.add_argument(
        "--output", type=Path, help="new local JSON file; existing files are never overwritten"
    )
    args = parser.parse_args()
    before = source_inventory()
    cases = run_diagnostics(suite=args.suite)
    after = source_inventory()
    if before != after:
        raise RuntimeError("source bytes changed during diagnostic")
    report = dict(
        status="LOCAL_DIAGNOSTIC_ONLY",
        suite=args.suite,
        checked_at_utc=datetime.now(UTC).isoformat(),
        python=platform.python_version(),
        numpy=np.__version__,
        sources=after,
        sources_before=before,
        sources_after=after,
        cases=cases,
        scientific_study_executed=False,
        external_validation=False,
        submission_ready=False,
    )
    encoded = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
        print(
            json.dumps(
                dict(
                    output=str(args.output.resolve()),
                    cases={c["case"]: c["comparison_status"] for c in cases},
                )
            )
        )
    else:
        print(encoded, end="")
    return 0 if all(c["comparison_status"] == "AGREEMENT" for c in cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
