from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
import pytest

import selcal.calibration_v2 as calibration_v2
import selcal.contracts_v2 as contracts_v2
from selcal.contracts import (
    ReplicateStatus,
    RunStatus,
    SeriesPair,
    StatisticResult,
    Validity,
)
from selcal.contracts_v2 import (
    CalibrationResult,
    NullTransformToken,
    PlanRequestV2,
    ReplicateFailureStage,
    RunFailureStage,
    V2IntegrityError,
)
from selcal.nulls.circular_shift_v2 import _BoundCircularShiftV2
from selcal.resolution_v2 import PlanResolutionV2, resolve_plan_v2
from selcal.statistics.lagged_pearson import _BoundLaggedPearsonAdapter

ScorePair = tuple[float, float]
ScriptedScan = ScorePair | None


def _pair() -> SeriesPair:
    return SeriesPair(
        source=np.asarray([0, 1, 2, 3, 4, 5], dtype=np.float64),
        target=np.asarray([1, 0, 2, 5, 3, 4], dtype=np.float64),
    )


def _resolution(*, replicates: int = 3, alpha: float = 0.75) -> PlanResolutionV2:
    return resolve_plan_v2(
        PlanRequestV2(
            candidates=(1, 2),
            statistic_name="lagged_pearson_v1",
            statistic_params={},
            selection_rule="max_upper",
            null_name="circular_shift_v2",
            null_params={"min_shift": 1},
            replicates=replicates,
            alpha=alpha,
            tie_tolerance=1e-12,
            root_seed=17,
        )
    )


def _install_scripted_run(
    monkeypatch: pytest.MonkeyPatch,
    scans: tuple[ScriptedScan, ...],
) -> list[int]:
    """Install one observed scan followed by exactly one scan per planned replicate."""

    assert len(scans) >= 2
    scripted_indices = tuple(range(1, len(scans)))
    original_sample = _BoundCircularShiftV2.sample_token
    sampled_replicates = 0

    class ScriptedIndexCapability:
        def __init__(self, index: int) -> None:
            self._index = index

        def randbelow(self, bound: int, /) -> int:
            assert 0 <= self._index < bound
            return self._index

    def scripted_sample(
        self: _BoundCircularShiftV2,
        random: object,
        /,
    ) -> NullTransformToken:
        del random
        nonlocal sampled_replicates
        replicate_id = sampled_replicates
        sampled_replicates += 1
        assert replicate_id < len(scripted_indices)
        return original_sample(self, ScriptedIndexCapability(scripted_indices[replicate_id]))

    # Slice 2 transitional null-kernel seam. It preserves the real callback's
    # token construction and range checks without reopening production RNG.
    monkeypatch.setattr(_BoundCircularShiftV2, "sample_token", scripted_sample)
    calls: list[int] = []

    def scripted_evaluate(
        bound: _BoundLaggedPearsonAdapter,
        supplied: SeriesPair,
        /,
    ) -> tuple[StatisticResult, ...]:
        del supplied
        call_id = len(calls)
        calls.append(call_id)
        assert call_id < len(scans), "calibration performed an unplanned statistic scan"
        scripted = scans[call_id]
        if scripted is None:
            return tuple(
                StatisticResult(
                    candidate_id=candidate,
                    estimate=None,
                    selection_score=None,
                    support_n=bound.observed_length - max(bound.candidates),
                    validity=Validity.ANALYTIC_FAILURE,
                    diagnostics=("scripted_task8c_analytical_failure",),
                    backend_identity=bound.backend_identity,
                    preprocessing_identity=bound.preprocessing_identity,
                )
                for candidate in bound.candidates
            )
        return tuple(
            StatisticResult(
                candidate_id=candidate,
                estimate=estimate,
                selection_score=None,
                support_n=bound.observed_length - max(bound.candidates),
                validity=Validity.VALID,
                diagnostics=(),
                backend_identity=bound.backend_identity,
                preprocessing_identity=bound.preprocessing_identity,
            )
            for candidate, estimate in zip(bound.candidates, scripted, strict=True)
        )

    monkeypatch.setattr(_BoundLaggedPearsonAdapter, "evaluate_all", scripted_evaluate)
    return calls


def _calibrator() -> Callable[[SeriesPair, PlanResolutionV2], CalibrationResult]:
    candidate = getattr(calibration_v2, "calibrate_selected_family", None)
    assert callable(candidate), "Task 8C must add calibrate_selected_family(pair, resolution, /)"
    return cast(Callable[[SeriesPair, PlanResolutionV2], CalibrationResult], candidate)


def test_complete_finalization_uses_inclusive_equality_and_exact_plus_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # observed A=3; surrogate family maxima are 4, 3, and 2.
    calls = _install_scripted_run(
        monkeypatch,
        ((3.0, 1.0), (2.0, 4.0), (3.0, 0.0), (1.0, 2.0)),
    )
    exact = _resolution(alpha=0.75)

    result = _calibrator()(_pair(), exact)

    assert result.status is RunStatus.COMPLETE
    assert result.failure_stage is None
    assert result.planned_replicates == 3
    assert len(calls) == 1 + result.planned_replicates
    assert result.observed_selection is not None
    assert result.observed_selection.decision_statistic == 3.0
    assert tuple(
        outcome.selection.decision_statistic
        for outcome in result.replicates
        if outcome.selection is not None
    ) == (4.0, 3.0, 2.0)
    assert result.exceedance_count == 2  # 4 >= 3 and equality 3 >= 3.
    assert result.p_value == (1 + 2) / (3 + 1) == 0.75
    assert result.reject_null is True  # rejection is also inclusive at p == alpha.
    assert result.exceedance_bound_low is None
    assert result.exceedance_bound_high is None


def test_replicate_failure_retains_exact_b_and_uses_the_planned_denominator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # One complete exceedance, one analytical failure, and one complete nonexceedance.
    calls = _install_scripted_run(
        monkeypatch,
        ((3.0, 1.0), (2.0, 4.0), None, (1.0, 2.0)),
    )
    exact = _resolution(alpha=0.5)

    result = _calibrator()(_pair(), exact)

    assert result.status is RunStatus.NOT_EVALUABLE
    assert result.failure_stage is RunFailureStage.REPLICATE_EXECUTION
    assert len(calls) == 1 + exact.plan.replicates
    assert tuple(outcome.replicate_id for outcome in result.replicates) == (0, 1, 2)
    assert all(type(outcome.transform_token) is NullTransformToken for outcome in result.replicates)
    assert all(
        tuple(item.candidate_id for item in outcome.statistic_results) == (1, 2)
        for outcome in result.replicates
    )
    assert tuple(outcome.status for outcome in result.replicates) == (
        ReplicateStatus.COMPLETE,
        ReplicateStatus.ANALYTIC_FAILURE,
        ReplicateStatus.COMPLETE,
    )
    failed = result.replicates[1]
    assert failed.failure_stage is ReplicateFailureStage.STATISTIC_SCAN
    assert failed.selection is None
    assert len(failed.statistic_results) == len(exact.plan.candidates)
    assert result.exceedance_count == 1
    assert result.failure_count == 1
    assert result.p_value is None
    assert result.reject_null is None
    assert result.exceedance_bound_low == (1 + 1) / (3 + 1) == 0.5
    assert result.exceedance_bound_high == (1 + 1 + 1) / (3 + 1) == 0.75
    successful_replicates = result.planned_replicates - result.failure_count
    assert result.exceedance_bound_low != (
        (1 + result.exceedance_count) / (successful_replicates + 1)
    )


def test_full_reselection_gives_three_quarters_not_observed_candidate_half(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scans: tuple[ScorePair, ...] = (
        (3.0, 1.0),
        (2.0, 4.0),
        (3.0, 0.0),
        (1.0, 2.0),
    )
    _install_scripted_run(monkeypatch, scans)
    result = _calibrator()(_pair(), _resolution())

    assert result.observed_selection is not None
    observed_index = result.observed_selection.selected_index
    surrogate_family_maxima = tuple(max(scores) for scores in scans[1:])
    observed_candidate_scores = tuple(scores[observed_index] for scores in scans[1:])
    full_reselection_p = (
        1
        + sum(
            score >= result.observed_selection.decision_statistic
            for score in surrogate_family_maxima
        )
    ) / len(scans)
    observed_candidate_reuse_p = (
        1
        + sum(
            score >= result.observed_results[observed_index].selection_score
            for score in observed_candidate_scores
        )
    ) / len(scans)

    assert tuple(
        outcome.selection.selected_candidate
        for outcome in result.replicates
        if outcome.selection is not None
    ) == (2, 1, 2)
    assert result.p_value == full_reselection_p == 3 / 4
    assert observed_candidate_reuse_p == 1 / 2


def test_constructor_valid_cross_module_verifier_forgery_is_resolution_inconsistent() -> None:
    exact = _resolution()
    result = _calibrator()(_pair(), exact)
    original = result.observed_results[0]
    inconsistent = StatisticResult(
        candidate_id=original.candidate_id,
        estimate=original.estimate,
        selection_score=original.selection_score,
        support_n=original.support_n,
        validity=original.validity,
        diagnostics=original.diagnostics,
        backend_identity="constructor-valid-foreign-backend",
        preprocessing_identity=original.preprocessing_identity,
    )
    forged = CalibrationResult(
        status=result.status,
        failure_stage=result.failure_stage,
        semantic_input_sha256=result.semantic_input_sha256,
        scientific_plan_sha256=result.scientific_plan_sha256,
        planned_replicates=result.planned_replicates,
        alpha=result.alpha,
        observed_results=(inconsistent, *result.observed_results[1:]),
        observed_selection=result.observed_selection,
        replicates=result.replicates,
        exceedance_count=result.exceedance_count,
        failure_count=result.failure_count,
        p_value=result.p_value,
        exceedance_bound_low=result.exceedance_bound_low,
        exceedance_bound_high=result.exceedance_bound_high,
        reject_null=result.reject_null,
        diagnostics=result.diagnostics,
    )

    assert type(forged) is CalibrationResult
    with pytest.raises(V2IntegrityError, match="implementation identity drifted"):
        contracts_v2.verify_calibration_result(forged, exact)
