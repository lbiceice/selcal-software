from __future__ import annotations

from dataclasses import FrozenInstanceError
from fractions import Fraction

import numpy as np
import pytest

from selcal.contracts import SelectionRule, StatisticResult, Validity
from selcal.selection import select_candidate


def result(
    candidate_id: int,
    estimate: float,
    *,
    selection_score: float | None = None,
) -> StatisticResult:
    return StatisticResult(
        candidate_id=candidate_id,
        estimate=estimate,
        selection_score=selection_score,
        support_n=24,
        validity=Validity.VALID,
        diagnostics=(f"candidate-{candidate_id}",),
        backend_identity="test-backend-v1",
        preprocessing_identity="test-preprocessing-v1",
    )


def corrupt(record: StatisticResult, field: str, value: object) -> StatisticResult:
    object.__setattr__(record, field, value)
    return record


def test_max_upper_recomputes_scores_and_uses_first_candidate_within_tolerance() -> None:
    original = (
        result(1, 0.7, selection_score=999.0),
        result(2, 0.7000000000005, selection_score=-999.0),
        result(3, -0.9, selection_score=9999.0),
    )

    selection, scored = select_candidate(original, "max_upper", 1e-12)

    assert tuple(item.selection_score for item in scored) == pytest.approx(
        (0.7, 0.7000000000005, -0.9)
    )
    assert selection.selected_candidate == 1
    assert selection.selected_index == 0
    assert selection.tied_candidates == (1, 2)
    assert selection.decision_statistic == pytest.approx(0.7)


def test_max_absolute_selects_greatest_magnitude_and_reports_absolute_decision() -> None:
    selection, scored = select_candidate(
        (result(1, 0.7), result(2, -0.9), result(3, 0.2)),
        SelectionRule.MAX_ABSOLUTE,
        0.0,
    )

    assert tuple(item.selection_score for item in scored) == pytest.approx((0.7, 0.9, 0.2))
    assert selection.selected_candidate == 2
    assert selection.selected_index == 1
    assert selection.tied_candidates == (2,)
    assert selection.decision_statistic == pytest.approx(0.9)


def test_exact_tie_selects_first_canonical_candidate() -> None:
    selection, _ = select_candidate(
        (result(1, 0.4), result(2, 0.4), result(3, 0.1)),
        "max_upper",
        0.0,
    )

    assert selection.selected_candidate == 1
    assert selection.selected_index == 0
    assert selection.tied_candidates == (1, 2)


def test_tolerance_boundary_is_included_and_just_outside_is_excluded() -> None:
    selection, _ = select_candidate(
        (result(1, 0.75), result(2, 0.749), result(3, 0.65), result(4, 0.649)),
        "max_upper",
        0.1,
    )

    assert selection.tied_candidates == (1, 2, 3)


def test_zero_tolerance_includes_exact_ties_only() -> None:
    selection, _ = select_candidate(
        (result(1, 1.0), result(2, 1.0), result(3, np.nextafter(1.0, 0.0))),
        "max_upper",
        0,
    )

    assert selection.tied_candidates == (1, 2)


def test_all_negative_max_upper_selects_least_negative() -> None:
    selection, scored = select_candidate(
        (result(1, -3.0), result(2, -0.2), result(3, -1.0)),
        "max_upper",
        0.0,
    )

    assert tuple(item.selection_score for item in scored) == (-3.0, -0.2, -1.0)
    assert selection.selected_candidate == 2
    assert selection.decision_statistic == -0.2


def test_all_negative_max_absolute_selects_greatest_magnitude() -> None:
    selection, scored = select_candidate(
        (result(1, -3.0), result(2, -0.2), result(3, -1.0)),
        "max_absolute",
        0.0,
    )

    assert tuple(item.selection_score for item in scored) == (3.0, 0.2, 1.0)
    assert selection.selected_candidate == 1
    assert selection.decision_statistic == 3.0


@pytest.mark.parametrize("rule", [SelectionRule.MAX_UPPER, SelectionRule.MAX_ABSOLUTE])
def test_signed_zero_is_normalized_stably(rule: SelectionRule) -> None:
    selection, scored = select_candidate(
        (result(1, -0.0), result(2, 0.0)),
        rule,
        0.0,
    )

    assert selection.selected_candidate == 1
    assert selection.decision_statistic == 0.0
    assert not np.signbit(selection.decision_statistic)
    assert all(not np.signbit(item.selection_score) for item in scored)


def test_empty_result_vector_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        select_candidate((), "max_upper", 0.0)


@pytest.mark.parametrize("container_kind", ["list", "generator", "tuple-subclass"])
def test_result_vector_requires_an_exact_tuple_container(container_kind: str) -> None:
    base = (result(1, 0.2),)
    if container_kind == "list":
        invalid = list(base)
    elif container_kind == "generator":
        invalid = (item for item in base)
    else:

        class TupleSubclass(tuple[StatisticResult, ...]):
            pass

        invalid = TupleSubclass(base)

    with pytest.raises(ValueError, match="exact tuple"):
        select_candidate(invalid, "max_upper", 0.0)  # type: ignore[arg-type]


def test_non_statistic_result_element_is_rejected() -> None:
    with pytest.raises(ValueError, match="StatisticResult"):
        select_candidate((result(1, 0.2), "not-a-result"), "max_upper", 0.0)  # type: ignore[arg-type]


def test_statistic_result_subclass_is_rejected_before_hostile_replace() -> None:
    class HostileStatisticResult(StatisticResult):
        def __init__(
            self,
            candidate_id: int,
            estimate: float | None,
            selection_score: float | None,
            support_n: int,
            validity: Validity,
            diagnostics: tuple[str, ...],
            backend_identity: str,
            preprocessing_identity: str,
        ) -> None:
            super().__init__(
                candidate_id=candidate_id,
                estimate=estimate,
                selection_score=selection_score,
                support_n=support_n,
                validity=validity,
                diagnostics=diagnostics,
                backend_identity=backend_identity,
                preprocessing_identity=preprocessing_identity,
            )
            if selection_score is not None:
                object.__setattr__(self, "candidate_id", 999)
                object.__setattr__(self, "selection_score", -999.0)

    hostile = HostileStatisticResult(
        candidate_id=1,
        estimate=0.2,
        selection_score=None,
        support_n=24,
        validity=Validity.VALID,
        diagnostics=("hostile-replace",),
        backend_identity="test-backend-v1",
        preprocessing_identity="test-preprocessing-v1",
    )

    with pytest.raises(ValueError, match="exact StatisticResult"):
        select_candidate((hostile,), "max_upper", 0.0)

    assert hostile.candidate_id == 1
    assert hostile.selection_score is None


@pytest.mark.parametrize(
    "results",
    [
        (result(2, 0.2), result(1, 0.3)),
        (result(1, 0.2), result(1, 0.3)),
    ],
    ids=["unsorted", "duplicate"],
)
def test_candidate_vector_must_be_strictly_ascending_and_unique(
    results: tuple[StatisticResult, ...],
) -> None:
    with pytest.raises(ValueError, match="canonical candidate IDs"):
        select_candidate(results, "max_upper", 0.0)


@pytest.mark.parametrize("candidate_id", [True, np.int64(1), 0, -1])
def test_candidate_identity_must_be_a_canonical_builtin_integer(candidate_id: object) -> None:
    invalid = corrupt(result(1, 0.2), "candidate_id", candidate_id)

    with pytest.raises(ValueError, match="canonical candidate IDs"):
        select_candidate((invalid,), "max_upper", 0.0)


def test_analytic_failure_result_is_rejected() -> None:
    failure = StatisticResult(
        candidate_id=1,
        estimate=None,
        selection_score=None,
        support_n=24,
        validity=Validity.ANALYTIC_FAILURE,
        diagnostics=("analytic-failure",),
        backend_identity="test-backend-v1",
        preprocessing_identity="test-preprocessing-v1",
    )

    with pytest.raises(ValueError, match="VALID"):
        select_candidate((failure,), "max_upper", 0.0)


@pytest.mark.parametrize("estimate", [None, float("nan"), float("inf"), -float("inf")])
def test_missing_or_nonfinite_estimate_is_rejected(estimate: object) -> None:
    invalid = corrupt(result(1, 0.2), "estimate", estimate)

    with pytest.raises(ValueError, match="finite estimate"):
        select_candidate((invalid,), "max_upper", 0.0)


def test_scoring_cannot_alias_a_mutable_identity_from_an_adversarial_record() -> None:
    mutable_identity = ["mutable-backend"]
    invalid = corrupt(result(1, 0.2), "backend_identity", mutable_identity)

    with pytest.raises(ValueError, match="backend_identity"):
        select_candidate((invalid,), "max_upper", 0.0)


@pytest.mark.parametrize("rule", ["minimum", "MAX_UPPER", "", 7, None, True])
def test_invalid_selection_rule_is_rejected(rule: object) -> None:
    with pytest.raises(ValueError, match="selection rule"):
        select_candidate((result(1, 0.2),), rule, 0.0)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "tolerance",
    [-1.0, float("nan"), float("inf"), -float("inf"), True, "0", None],
)
def test_invalid_tie_tolerance_is_rejected(tolerance: object) -> None:
    with pytest.raises(ValueError, match="tie_tolerance"):
        select_candidate((result(1, 0.2),), "max_upper", tolerance)  # type: ignore[arg-type]


def test_overflowing_real_tolerance_is_normalized_to_value_error() -> None:
    overflowing = Fraction(10**400, 1)

    with pytest.raises(ValueError, match="tie_tolerance"):
        select_candidate((result(1, 0.2),), "max_upper", overflowing)  # type: ignore[arg-type]


def test_overflowing_real_estimate_is_normalized_to_value_error() -> None:
    invalid = corrupt(result(1, 0.2), "estimate", Fraction(10**400, 1))

    with pytest.raises(ValueError, match="finite estimate"):
        select_candidate((invalid,), "max_upper", 0.0)


def test_each_real_is_converted_to_float_exactly_once() -> None:
    class StatefulReal(Fraction):
        conversions: int

        def __new__(cls, value: Fraction) -> StatefulReal:
            instance = super().__new__(cls, value)
            instance.conversions = 0
            return instance

        def __float__(self) -> float:
            self.conversions += 1
            if self.conversions > 1:
                raise RuntimeError("real converted more than once")
            return super().__float__()

    estimate = StatefulReal(Fraction(3, 4))
    tolerance = StatefulReal(Fraction(1, 10))
    adversarial = corrupt(result(1, 0.2), "estimate", estimate)

    selection, scored = select_candidate(
        (adversarial,),
        "max_upper",
        tolerance,  # type: ignore[arg-type]
    )

    assert estimate.conversions == 1
    assert tolerance.conversions == 1
    assert selection.decision_statistic == 0.75
    assert scored[0].selection_score == 0.75


def test_outputs_are_new_frozen_records_and_inputs_remain_unchanged() -> None:
    original = (
        result(1, 0.3, selection_score=41.0),
        result(2, 0.6, selection_score=-17.0),
    )
    before = tuple(original)

    first_selection, first_scored = select_candidate(original, "max_upper", 0.0)
    second_selection, second_scored = select_candidate(original, "max_upper", 0.0)

    assert original == before
    assert tuple(item.selection_score for item in original) == (41.0, -17.0)
    assert first_scored is not original
    assert all(
        updated is not source for updated, source in zip(first_scored, original, strict=True)
    )
    assert first_selection == second_selection
    assert first_scored == second_scored
    with pytest.raises(FrozenInstanceError):
        first_scored[0].selection_score = 1.0  # type: ignore[misc]


def test_scoring_preserves_record_fields_vector_length_and_candidate_identity() -> None:
    original = (result(1, 0.3), result(4, 0.6), result(9, -0.8))

    _, scored = select_candidate(original, "max_absolute", 0.0)

    assert len(scored) == len(original)
    assert tuple(item.candidate_id for item in scored) == (1, 4, 9)
    for source, updated in zip(original, scored, strict=True):
        assert updated.candidate_id == source.candidate_id
        assert updated.estimate == source.estimate
        assert updated.support_n == source.support_n
        assert updated.validity is source.validity
        assert updated.diagnostics == source.diagnostics
        assert updated.backend_identity == source.backend_identity
        assert updated.preprocessing_identity == source.preprocessing_identity
