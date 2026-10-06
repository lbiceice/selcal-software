from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from enum import StrEnum

import numpy as np
import pytest

import selcal.calibration_v2 as calibration_v2
import selcal.contracts_v2 as contracts_v2
from selcal.contracts import (
    ReplicateStatus,
    RunStatus,
    SelectionResult,
    StatisticResult,
    Validity,
)
from selcal.contracts_v2 import (
    BlockShuffleStateV2,
    CalibrationResult,
    CircularShiftStateV2,
    NullTransformToken,
    ReplicateFailureStage,
    ReplicateOutcome,
    RunFailureStage,
    V2IntegrityError,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64


def token(
    *,
    shift: int = 0,
    null_parameter_sha256: str = SHA_C,
    semantic_input_sha256: str = SHA_A,
    scientific_plan_sha256: str = SHA_B,
    bound_null_owner_sha256: str = SHA_D,
) -> NullTransformToken:
    return NullTransformToken(
        schema="selcal.null-transform-token.v2",
        null_name="circular_shift_v2",
        null_parameter_sha256=null_parameter_sha256,
        semantic_input_sha256=semantic_input_sha256,
        scientific_plan_sha256=scientific_plan_sha256,
        bound_null_owner_sha256=bound_null_owner_sha256,
        is_identity=shift == 0,
        state=CircularShiftStateV2(
            schema="selcal.circular-shift-state.v2",
            shift=shift,
        ),
    )


def alternate_null_token() -> NullTransformToken:
    return NullTransformToken(
        schema="selcal.null-transform-token.v2",
        null_name="block_shuffle_v2",
        null_parameter_sha256=SHA_C,
        semantic_input_sha256=SHA_A,
        scientific_plan_sha256=SHA_B,
        bound_null_owner_sha256=SHA_D,
        is_identity=False,
        state=BlockShuffleStateV2(
            schema="selcal.block-shuffle-state.v2",
            block_order=(1, 0),
        ),
    )


class ForeignReplicateFailureStage(StrEnum):
    STATISTIC_SCAN = "statistic_scan"


class ForeignRunFailureStage(StrEnum):
    REPLICATE_EXECUTION = "replicate_execution"


def statistic(
    candidate: int,
    *,
    estimate: float | None,
    score: float | None,
    validity: Validity,
) -> StatisticResult:
    return StatisticResult(
        candidate_id=candidate,
        estimate=estimate,
        selection_score=score,
        support_n=10,
        validity=validity,
        diagnostics=(),
        backend_identity="test-backend-v1",
        preprocessing_identity="test-preprocessing-v1",
    )


def scored_vector(first: float = 3.0, second: float = 1.0) -> tuple[StatisticResult, ...]:
    return (
        statistic(1, estimate=first, score=first, validity=Validity.VALID),
        statistic(2, estimate=second, score=second, validity=Validity.VALID),
    )


def raw_failure_vector() -> tuple[StatisticResult, ...]:
    return (
        statistic(1, estimate=2.0, score=None, validity=Validity.VALID),
        statistic(
            2,
            estimate=None,
            score=None,
            validity=Validity.ANALYTIC_FAILURE,
        ),
    )


def selection(
    *,
    candidate: int = 1,
    index: int = 0,
    decision: float = 3.0,
    ties: tuple[int, ...] = (1,),
) -> SelectionResult:
    return SelectionResult(
        selected_candidate=candidate,
        selected_index=index,
        decision_statistic=decision,
        tied_candidates=ties,
    )


def complete_outcome(
    replicate_id: int,
    *,
    decision: float,
    selected_candidate: int = 1,
) -> ReplicateOutcome:
    if selected_candidate == 1:
        results = scored_vector(decision, 1.0)
        selected_index = 0
    else:
        results = scored_vector(1.0, decision)
        selected_index = 1
    return ReplicateOutcome(
        replicate_id=replicate_id,
        seed_digest_sha256=f"{replicate_id + 1:064x}",
        status=ReplicateStatus.COMPLETE,
        failure_stage=None,
        transform_token=token(shift=replicate_id),
        statistic_results=results,
        selection=selection(
            candidate=selected_candidate,
            index=selected_index,
            decision=decision,
            ties=(selected_candidate,),
        ),
        diagnostics=(),
    )


def failed_outcome(replicate_id: int) -> ReplicateOutcome:
    return ReplicateOutcome(
        replicate_id=replicate_id,
        seed_digest_sha256=f"{replicate_id + 1:064x}",
        status=ReplicateStatus.ANALYTIC_FAILURE,
        failure_stage=ReplicateFailureStage.STATISTIC_SCAN,
        transform_token=token(shift=replicate_id),
        statistic_results=raw_failure_vector(),
        selection=None,
        diagnostics=("candidate 2 failed",),
    )


def complete_calibration_values() -> dict[str, object]:
    return {
        "status": RunStatus.COMPLETE,
        "failure_stage": None,
        "semantic_input_sha256": SHA_A,
        "scientific_plan_sha256": SHA_B,
        "planned_replicates": 2,
        "alpha": 0.05,
        "observed_results": scored_vector(3.0, 1.0),
        "observed_selection": selection(decision=3.0),
        "replicates": (
            complete_outcome(0, decision=4.0),
            complete_outcome(1, decision=2.0, selected_candidate=2),
        ),
        "exceedance_count": 1,
        "failure_count": 0,
        "p_value": (1 + 1) / (2 + 1),
        "exceedance_bound_low": None,
        "exceedance_bound_high": None,
        "reject_null": False,
        "diagnostics": (),
    }


def make_complete_calibration(**changes: object) -> CalibrationResult:
    values = complete_calibration_values()
    values.update(changes)
    return CalibrationResult(**values)  # type: ignore[arg-type]


class DerivedStatisticResult(StatisticResult):
    pass


def derived_statistic() -> DerivedStatisticResult:
    return DerivedStatisticResult(
        candidate_id=1,
        estimate=1.0,
        selection_score=1.0,
        support_n=10,
        validity=Validity.VALID,
        diagnostics=(),
        backend_identity="test-backend-v1",
        preprocessing_identity="test-preprocessing-v1",
    )


def test_complete_replicate_is_a_strict_nonhashable_value() -> None:
    outcome = complete_outcome(0, decision=4.0)
    assert outcome == complete_outcome(0, decision=4.0)
    with pytest.raises(FrozenInstanceError):
        outcome.replicate_id = 1  # type: ignore[misc]
    with pytest.raises(TypeError):
        hash(outcome)


@pytest.mark.parametrize(
    "changes",
    [
        {"replicate_id": True},
        {"replicate_id": np.int64(0)},
        {"replicate_id": -1},
        {"seed_digest_sha256": "A" * 64},
        {"seed_digest_sha256": "a" * 63},
        {"status": "complete"},
        {"failure_stage": ReplicateFailureStage.STATISTIC_SCAN},
        {"failure_stage": ForeignReplicateFailureStage.STATISTIC_SCAN},
        {"transform_token": object()},
        {"statistic_results": list(scored_vector(4.0, 1.0))},
        {"statistic_results": ()},
        {"statistic_results": tuple(reversed(scored_vector(4.0, 1.0)))},
        {"statistic_results": (scored_vector()[0], scored_vector()[0])},
        {"statistic_results": (derived_statistic(), scored_vector()[1])},
        {"statistic_results": raw_failure_vector()},
        {"statistic_results": scored_vector(4.0, 1.0), "selection": None},
        {"statistic_results": scored_vector(4.0, 1.0), "selection": selection(decision=3.0)},
        {
            "statistic_results": scored_vector(4.0, 1.0),
            "selection": selection(candidate=2, index=0, decision=4.0, ties=(2,)),
        },
        {
            "statistic_results": scored_vector(4.0, 1.0),
            "selection": selection(candidate=1, index=2, decision=4.0, ties=(1,)),
        },
        {
            "statistic_results": scored_vector(4.0, 1.0),
            "selection": selection(candidate=1, index=0, decision=4.0, ties=(1, 3)),
        },
        {
            "statistic_results": scored_vector(4.0, 4.0),
            "selection": selection(candidate=2, index=1, decision=4.0, ties=(1, 2)),
        },
        {"diagnostics": ["x"]},
        {"diagnostics": (1,)},
    ],
)
def test_complete_replicate_rejects_each_self_contained_contradiction(
    changes: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "replicate_id": 0,
        "seed_digest_sha256": "1" * 64,
        "status": ReplicateStatus.COMPLETE,
        "failure_stage": None,
        "transform_token": token(),
        "statistic_results": scored_vector(4.0, 1.0),
        "selection": selection(decision=4.0),
        "diagnostics": (),
    }
    values.update(changes)
    with pytest.raises(V2IntegrityError):
        ReplicateOutcome(**values)  # type: ignore[arg-type]


def test_tied_complete_replicate_uses_family_max_not_representative_score() -> None:
    results = scored_vector(2.999, 3.0)
    outcome = ReplicateOutcome(
        replicate_id=0,
        seed_digest_sha256="1" * 64,
        status=ReplicateStatus.COMPLETE,
        failure_stage=None,
        transform_token=token(),
        statistic_results=results,
        selection=selection(candidate=1, index=0, decision=3.0, ties=(1, 2)),
        diagnostics=(),
    )
    assert outcome.selection is not None
    assert outcome.selection.decision_statistic == 3.0


@pytest.mark.parametrize(
    "changes",
    [
        {"status": "analytic_failure"},
        {"failure_stage": None},
        {"failure_stage": "statistic_scan"},
        {"failure_stage": ForeignReplicateFailureStage.STATISTIC_SCAN},
        {"selection": selection(decision=3.0)},
        {"statistic_results": ()},
        {
            "statistic_results": (
                statistic(1, estimate=1.0, score=None, validity=Validity.VALID),
                statistic(2, estimate=2.0, score=None, validity=Validity.VALID),
            )
        },
        {
            "statistic_results": (
                statistic(1, estimate=1.0, score=1.0, validity=Validity.VALID),
                raw_failure_vector()[1],
            )
        },
        {"statistic_results": tuple(reversed(raw_failure_vector()))},
    ],
)
def test_failed_replicate_requires_a_full_raw_failure_vector(
    changes: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "replicate_id": 0,
        "seed_digest_sha256": "1" * 64,
        "status": ReplicateStatus.ANALYTIC_FAILURE,
        "failure_stage": ReplicateFailureStage.STATISTIC_SCAN,
        "transform_token": token(),
        "statistic_results": raw_failure_vector(),
        "selection": None,
        "diagnostics": (),
    }
    values.update(changes)
    with pytest.raises(V2IntegrityError):
        ReplicateOutcome(**values)  # type: ignore[arg-type]


def test_complete_calibration_reconstructs_exact_b_math() -> None:
    result = make_complete_calibration()
    assert result.exceedance_count == 1
    assert result.p_value == 2 / 3
    assert result.reject_null is False
    assert result.exceedance_bound_low is None
    assert result.exceedance_bound_high is None
    with pytest.raises(TypeError):
        hash(result)


def test_complete_calibration_counts_equality_and_rejects_at_equal_alpha() -> None:
    result = make_complete_calibration(
        alpha=2 / 3,
        replicates=(
            complete_outcome(0, decision=3.0),
            complete_outcome(1, decision=2.0, selected_candidate=2),
        ),
        exceedance_count=1,
        p_value=2 / 3,
        reject_null=True,
    )
    assert result.exceedance_count == 1
    assert result.p_value == result.alpha == 2 / 3
    assert result.reject_null is True
    assert result.exceedance_bound_low is None
    assert result.exceedance_bound_high is None


def test_task8c_plan_bound_verifier_is_contract_owned_and_wired_to_calibration() -> None:
    verifier = getattr(contracts_v2, "verify_calibration_result", None)
    assert callable(verifier), "Task 8C must add the public plan-bound result verifier"
    assert getattr(calibration_v2, "verify_calibration_result", None) is verifier


@pytest.mark.parametrize(
    "changes",
    [
        {"status": "complete"},
        {"failure_stage": RunFailureStage.REPLICATE_EXECUTION},
        {"failure_stage": "replicate_execution"},
        {"failure_stage": ForeignRunFailureStage.REPLICATE_EXECUTION},
        {"semantic_input_sha256": "A" * 64},
        {"scientific_plan_sha256": "b" * 63},
        {"planned_replicates": True},
        {"planned_replicates": np.int64(2)},
        {"planned_replicates": 0},
        {"planned_replicates": 1_000_001},
        {"alpha": 1},
        {"alpha": np.float64(0.05)},
        {"alpha": float("nan")},
        {"alpha": 0.0},
        {"alpha": 1.0},
        {"observed_results": list(scored_vector())},
        {"observed_results": tuple(reversed(scored_vector()))},
        {"observed_results": raw_failure_vector()},
        {"observed_selection": None},
        {"replicates": list(complete_calibration_values()["replicates"])},
        {"replicates": (complete_outcome(1, decision=4.0), complete_outcome(0, decision=2.0))},
        {"replicates": (complete_outcome(0, decision=4.0), complete_outcome(0, decision=2.0))},
        {"replicates": (complete_outcome(0, decision=4.0),)},
        {"replicates": (complete_outcome(0, decision=4.0), object())},
        {"replicates": (complete_outcome(0, decision=4.0), failed_outcome(1))},
        {"exceedance_count": True},
        {"exceedance_count": np.int64(1)},
        {"exceedance_count": 0},
        {"failure_count": 1},
        {"p_value": np.float64(2 / 3)},
        {"p_value": 0.5},
        {"p_value": None},
        {"exceedance_bound_low": 2 / 3},
        {"exceedance_bound_high": 2 / 3},
        {"reject_null": np.bool_(False)},
        {"reject_null": True},
        {"reject_null": None},
        {"diagnostics": ["x"]},
    ],
)
def test_complete_calibration_rejects_every_contradiction(
    changes: dict[str, object],
) -> None:
    with pytest.raises(V2IntegrityError):
        make_complete_calibration(**changes)


def test_complete_calibration_rejects_candidate_vector_drift() -> None:
    drifted = ReplicateOutcome(
        replicate_id=1,
        seed_digest_sha256="2" * 64,
        status=ReplicateStatus.COMPLETE,
        failure_stage=None,
        transform_token=token(shift=1),
        statistic_results=(
            statistic(1, estimate=2.0, score=2.0, validity=Validity.VALID),
            statistic(3, estimate=1.0, score=1.0, validity=Validity.VALID),
        ),
        selection=selection(decision=2.0),
        diagnostics=(),
    )
    with pytest.raises(V2IntegrityError):
        make_complete_calibration(replicates=(complete_outcome(0, decision=4.0), drifted))


def foreign_token(case: str) -> NullTransformToken:
    if case == "semantic_input_sha256":
        return token(shift=1, semantic_input_sha256=SHA_E)
    if case == "scientific_plan_sha256":
        return token(shift=1, scientific_plan_sha256=SHA_E)
    if case == "null_parameter_sha256":
        return token(shift=1, null_parameter_sha256=SHA_E)
    if case == "bound_null_owner_sha256":
        return token(shift=1, bound_null_owner_sha256=SHA_E)
    if case == "null_name":
        return alternate_null_token()
    raise AssertionError(f"unsupported test case: {case}")


@pytest.mark.parametrize("run_kind", ["complete", "replicate_failure"])
@pytest.mark.parametrize(
    "ownership_field",
    [
        "semantic_input_sha256",
        "scientific_plan_sha256",
        "null_parameter_sha256",
        "bound_null_owner_sha256",
        "null_name",
    ],
)
def test_calibration_rejects_foreign_replicate_token_ownership(
    run_kind: str,
    ownership_field: str,
) -> None:
    values = complete_calibration_values() if run_kind == "complete" else replicate_failure_values()
    replicates = values["replicates"]
    assert type(replicates) is tuple
    first, second = replicates
    assert type(second) is ReplicateOutcome
    values["replicates"] = (
        first,
        replace(second, transform_token=foreign_token(ownership_field)),
    )
    with pytest.raises(V2IntegrityError):
        CalibrationResult(**values)  # type: ignore[arg-type]


def test_null_bind_not_evaluable_has_no_scans_or_counts() -> None:
    result = CalibrationResult(
        status=RunStatus.NOT_EVALUABLE,
        failure_stage=RunFailureStage.NULL_BIND,
        semantic_input_sha256=SHA_A,
        scientific_plan_sha256=SHA_B,
        planned_replicates=2,
        alpha=0.05,
        observed_results=(),
        observed_selection=None,
        replicates=(),
        exceedance_count=0,
        failure_count=0,
        p_value=None,
        exceedance_bound_low=None,
        exceedance_bound_high=None,
        reject_null=None,
        diagnostics=("shift space empty",),
    )
    assert result.failure_stage is RunFailureStage.NULL_BIND


@pytest.mark.parametrize(
    "changes",
    [
        {"status": RunStatus.COMPLETE},
        {"failure_stage": None},
        {"observed_results": raw_failure_vector()},
        {"observed_selection": selection()},
        {"replicates": (complete_outcome(0, decision=4.0),)},
        {"exceedance_count": 1},
        {"failure_count": 1},
        {"p_value": 0.5},
        {"exceedance_bound_low": 0.5, "exceedance_bound_high": 0.5},
        {"reject_null": False},
    ],
)
def test_null_bind_result_rejects_any_evaluated_state(changes: dict[str, object]) -> None:
    values: dict[str, object] = {
        "status": RunStatus.NOT_EVALUABLE,
        "failure_stage": RunFailureStage.NULL_BIND,
        "semantic_input_sha256": SHA_A,
        "scientific_plan_sha256": SHA_B,
        "planned_replicates": 2,
        "alpha": 0.05,
        "observed_results": (),
        "observed_selection": None,
        "replicates": (),
        "exceedance_count": 0,
        "failure_count": 0,
        "p_value": None,
        "exceedance_bound_low": None,
        "exceedance_bound_high": None,
        "reject_null": None,
        "diagnostics": (),
    }
    values.update(changes)
    with pytest.raises(V2IntegrityError):
        CalibrationResult(**values)  # type: ignore[arg-type]


def test_observed_failure_retains_the_full_raw_vector_and_stops() -> None:
    result = CalibrationResult(
        status=RunStatus.NOT_EVALUABLE,
        failure_stage=RunFailureStage.OBSERVED_STATISTIC_SCAN,
        semantic_input_sha256=SHA_A,
        scientific_plan_sha256=SHA_B,
        planned_replicates=2,
        alpha=0.05,
        observed_results=raw_failure_vector(),
        observed_selection=None,
        replicates=(),
        exceedance_count=0,
        failure_count=0,
        p_value=None,
        exceedance_bound_low=None,
        exceedance_bound_high=None,
        reject_null=None,
        diagnostics=("candidate 2 failed",),
    )
    assert len(result.observed_results) == 2


@pytest.mark.parametrize(
    "changes",
    [
        {"failure_stage": RunFailureStage.NULL_BIND},
        {"observed_results": ()},
        {"observed_results": scored_vector()},
        {"observed_results": tuple(reversed(raw_failure_vector()))},
        {"observed_selection": selection()},
        {"replicates": (failed_outcome(0),)},
        {"exceedance_count": 1},
        {"failure_count": 1},
        {"p_value": 0.5},
        {"exceedance_bound_low": 0.5, "exceedance_bound_high": 0.5},
        {"reject_null": False},
    ],
)
def test_observed_failure_rejects_any_downstream_work(changes: dict[str, object]) -> None:
    values: dict[str, object] = {
        "status": RunStatus.NOT_EVALUABLE,
        "failure_stage": RunFailureStage.OBSERVED_STATISTIC_SCAN,
        "semantic_input_sha256": SHA_A,
        "scientific_plan_sha256": SHA_B,
        "planned_replicates": 2,
        "alpha": 0.05,
        "observed_results": raw_failure_vector(),
        "observed_selection": None,
        "replicates": (),
        "exceedance_count": 0,
        "failure_count": 0,
        "p_value": None,
        "exceedance_bound_low": None,
        "exceedance_bound_high": None,
        "reject_null": None,
        "diagnostics": (),
    }
    values.update(changes)
    with pytest.raises(V2IntegrityError):
        CalibrationResult(**values)  # type: ignore[arg-type]


def replicate_failure_values() -> dict[str, object]:
    return {
        "status": RunStatus.NOT_EVALUABLE,
        "failure_stage": RunFailureStage.REPLICATE_EXECUTION,
        "semantic_input_sha256": SHA_A,
        "scientific_plan_sha256": SHA_B,
        "planned_replicates": 2,
        "alpha": 0.05,
        "observed_results": scored_vector(3.0, 1.0),
        "observed_selection": selection(decision=3.0),
        "replicates": (
            complete_outcome(0, decision=4.0),
            failed_outcome(1),
        ),
        "exceedance_count": 1,
        "failure_count": 1,
        "p_value": None,
        "exceedance_bound_low": (1 + 1) / (2 + 1),
        "exceedance_bound_high": (1 + 1 + 1) / (2 + 1),
        "reject_null": None,
        "diagnostics": ("one replicate failed",),
    }


def test_replicate_failure_retains_exact_b_and_exact_bounds() -> None:
    result = CalibrationResult(**replicate_failure_values())  # type: ignore[arg-type]
    assert len(result.replicates) == result.planned_replicates
    assert result.failure_count == 1
    assert result.exceedance_bound_low == 2 / 3
    assert result.exceedance_bound_high == 1.0


@pytest.mark.parametrize(
    "changes",
    [
        {"status": RunStatus.COMPLETE},
        {"failure_stage": None},
        {"observed_results": raw_failure_vector()},
        {"observed_selection": None},
        {"replicates": (complete_outcome(0, decision=4.0), complete_outcome(1, decision=2.0))},
        {"replicates": (failed_outcome(1), complete_outcome(0, decision=4.0))},
        {"replicates": (failed_outcome(0),)},
        {"exceedance_count": 0},
        {"failure_count": 0},
        {"failure_count": 2},
        {"p_value": 2 / 3},
        {"exceedance_bound_low": None},
        {"exceedance_bound_high": None},
        {"exceedance_bound_low": 0.5},
        {"exceedance_bound_high": 0.9},
        {"reject_null": False},
    ],
)
def test_replicate_failure_rejects_every_math_contradiction(
    changes: dict[str, object],
) -> None:
    values = replicate_failure_values()
    values.update(changes)
    with pytest.raises(V2IntegrityError):
        CalibrationResult(**values)  # type: ignore[arg-type]
