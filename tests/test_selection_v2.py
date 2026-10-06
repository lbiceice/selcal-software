from __future__ import annotations

from dataclasses import FrozenInstanceError
from enum import StrEnum

import numpy as np
import pytest

from selcal.contracts import SelectionRule, StatisticResult, Validity
from selcal.contracts_v2 import V2IntegrityError
from selcal.selection import select_candidate
from selcal.selection_v2 import select_family_v2


def raw_result(candidate_id: int, estimate: float) -> StatisticResult:
    return StatisticResult(
        candidate_id=candidate_id,
        estimate=estimate,
        selection_score=None,
        support_n=24,
        validity=Validity.VALID,
        diagnostics=(f"candidate-{candidate_id}",),
        backend_identity="test-backend-v2",
        preprocessing_identity="test-preprocessing-v2",
    )


def analytic_failure(candidate_id: int) -> StatisticResult:
    return StatisticResult(
        candidate_id=candidate_id,
        estimate=None,
        selection_score=None,
        support_n=24,
        validity=Validity.ANALYTIC_FAILURE,
        diagnostics=("analytic-failure",),
        backend_identity="test-backend-v2",
        preprocessing_identity="test-preprocessing-v2",
    )


def corrupt(record: StatisticResult, field: str, value: object) -> StatisticResult:
    object.__setattr__(record, field, value)
    return record


def test_v2_tolerance_selects_canonical_representative_but_keeps_true_family_maximum() -> None:
    original = (raw_result(1, 0.999), raw_result(2, 1.0))

    selection, scored = select_family_v2(original, (1, 2), "max_upper", 0.01)

    assert selection.selected_candidate == 1
    assert selection.selected_index == 0
    assert selection.tied_candidates == (1, 2)
    assert selection.decision_statistic == 1.0
    assert tuple(item.selection_score for item in scored) == (0.999, 1.0)
    assert tuple(item.selection_score for item in original) == (None, None)
    assert all(updated is not source for updated, source in zip(scored, original, strict=True))


def test_v1_tolerance_characterization_keeps_representative_score_as_decision() -> None:
    selection, scored = select_candidate(
        (raw_result(1, 0.999), raw_result(2, 1.0)),
        "max_upper",
        0.01,
    )

    assert selection.selected_candidate == 1
    assert selection.decision_statistic == 0.999
    assert tuple(item.selection_score for item in scored) == (0.999, 1.0)


def test_v2_max_absolute_preserves_signed_estimate_and_reports_absolute_maximum() -> None:
    original = (raw_result(1, -0.8), raw_result(2, 0.5))

    selection, scored = select_family_v2(
        original,
        (1, 2),
        SelectionRule.MAX_ABSOLUTE,
        0.0,
    )

    assert selection.selected_candidate == 1
    assert selection.decision_statistic == 0.8
    assert scored[0].estimate == -0.8
    assert scored[0].selection_score == 0.8
    with pytest.raises(FrozenInstanceError):
        scored[0].selection_score = 1.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("results", "expected_candidates"),
    [
        ((raw_result(1, 0.1),), (1, 2)),
        ((raw_result(1, 0.1), raw_result(2, 0.2), raw_result(3, 0.3)), (1, 2)),
        ((raw_result(1, 0.1), raw_result(1, 0.2)), (1, 2)),
        ((raw_result(2, 0.2), raw_result(1, 0.1)), (1, 2)),
    ],
    ids=["missing", "extra", "duplicate", "reordered"],
)
def test_v2_rejects_any_candidate_vector_identity_drift(
    results: tuple[StatisticResult, ...],
    expected_candidates: tuple[int, ...],
) -> None:
    with pytest.raises(V2IntegrityError, match="candidate"):
        select_family_v2(results, expected_candidates, "max_upper", 0.0)


def test_v2_rejects_pre_scored_raw_results() -> None:
    prescored = StatisticResult(
        candidate_id=1,
        estimate=0.4,
        selection_score=0.4,
        support_n=24,
        validity=Validity.VALID,
        diagnostics=("pre-scored",),
        backend_identity="test-backend-v2",
        preprocessing_identity="test-preprocessing-v2",
    )

    with pytest.raises(V2IntegrityError, match=r"raw|selection_score"):
        select_family_v2((prescored,), (1,), "max_upper", 0.0)


def test_v2_rejects_analytic_failure_before_scoring() -> None:
    with pytest.raises(V2IntegrityError, match="VALID"):
        select_family_v2((analytic_failure(1),), (1,), "max_upper", 0.0)


@pytest.mark.parametrize(
    "expected_candidates",
    [
        [1, 2],
        (2, 1),
        (1, 1),
        (np.int64(1), 2),
        (True, 2),
        (0, 2),
    ],
)
def test_v2_expected_candidates_must_be_an_exact_canonical_tuple(
    expected_candidates: object,
) -> None:
    with pytest.raises(V2IntegrityError, match=r"expected_candidates|canonical"):
        select_family_v2(
            (raw_result(1, 0.1), raw_result(2, 0.2)),
            expected_candidates,  # type: ignore[arg-type]
            "max_upper",
            0.0,
        )


def test_v2_rejects_result_container_and_record_subclasses() -> None:
    class ResultTuple(tuple[StatisticResult, ...]):
        pass

    class DerivedStatisticResult(StatisticResult):
        pass

    derived = DerivedStatisticResult(
        candidate_id=1,
        estimate=0.2,
        selection_score=None,
        support_n=24,
        validity=Validity.VALID,
        diagnostics=("derived",),
        backend_identity="test-backend-v2",
        preprocessing_identity="test-preprocessing-v2",
    )

    with pytest.raises(V2IntegrityError, match="exact tuple"):
        select_family_v2(ResultTuple((raw_result(1, 0.2),)), (1,), "max_upper", 0.0)
    with pytest.raises(V2IntegrityError, match="exact StatisticResult"):
        select_family_v2((derived,), (1,), "max_upper", 0.0)
    with pytest.raises(V2IntegrityError, match="exact tuple"):
        select_family_v2([raw_result(1, 0.2)], (1,), "max_upper", 0.0)  # type: ignore[arg-type]


class ForeignRule(StrEnum):
    MAX_UPPER = "max_upper"


class StringSubclass(str):
    pass


@pytest.mark.parametrize(
    "rule",
    [ForeignRule.MAX_UPPER, StringSubclass("max_upper"), "minimum", True, None],
)
def test_v2_rejects_foreign_subclass_unsupported_and_bool_rules(rule: object) -> None:
    with pytest.raises(V2IntegrityError, match="selection rule"):
        select_family_v2((raw_result(1, 0.2),), (1,), rule, 0.0)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "tie_tolerance",
    [0, True, np.float64(0.0), -0.1, float("nan"), float("inf"), -float("inf")],
)
def test_v2_tolerance_must_be_an_exact_nonnegative_finite_builtin_float(
    tie_tolerance: object,
) -> None:
    with pytest.raises(V2IntegrityError, match="tie_tolerance"):
        select_family_v2(
            (raw_result(1, 0.2),),
            (1,),
            "max_upper",
            tie_tolerance,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("estimate", [float("nan"), float("inf"), -float("inf")])
def test_v2_rejects_corrupted_nonfinite_estimates(estimate: float) -> None:
    record = corrupt(raw_result(1, 0.2), "estimate", estimate)

    with pytest.raises(V2IntegrityError, match="finite estimate"):
        select_family_v2((record,), (1,), "max_upper", 0.0)
