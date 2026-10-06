from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError, fields
from types import MappingProxyType

import numpy as np
import pytest

from selcal.contracts import (
    CalibrationResult,
    PlanRequest,
    ReplicateOutcome,
    ReplicateStatus,
    RunStatus,
    SelectionResult,
    SelectionRule,
    SeriesPair,
    StatisticResult,
    Validity,
    canonical_candidates,
)


def valid_plan(**changes: object) -> PlanRequest:
    values = {
        "candidates": (3, 1, 2),
        "statistic_name": "equal_width_binned_nette_v1",
        "statistic_params": {"bins": 3},
        "selection_rule": "max_upper",
        "null_name": "circular_shift_v1",
        "null_params": {"min_shift": 1},
        "replicates": 9,
        "alpha": 0.05,
        "tie_tolerance": 1e-12,
        "root_seed": 17,
        "failure_policy": "fail_closed_v1",
    }
    values.update(changes)
    return PlanRequest(**values)  # type: ignore[arg-type]


def test_plan_canonicalizes_candidate_order() -> None:
    assert valid_plan().candidates == (1, 2, 3)


def test_canonical_candidate_helper_uses_the_same_identity() -> None:
    assert canonical_candidates([7, 2, 5]) == (2, 5, 7)


@pytest.mark.parametrize("bad", [(1, 1), (0, 1), (-1, 2), (True, 2), ()])
def test_invalid_candidates_fail(bad: object) -> None:
    with pytest.raises(ValueError, match="candidates"):
        valid_plan(candidates=bad)


@pytest.mark.parametrize("bad", [(1, 2.0), (1, "2"), (1, None)])
def test_non_integer_candidates_fail(bad: object) -> None:
    with pytest.raises(ValueError, match="candidates"):
        valid_plan(candidates=bad)


@pytest.mark.parametrize("bad", [0, -1, True, 1.5])
def test_invalid_replicates_fail(bad: object) -> None:
    with pytest.raises(ValueError, match="replicates"):
        valid_plan(replicates=bad)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.1, float("nan"), float("inf"), True])
def test_invalid_alpha_fails(bad: object) -> None:
    with pytest.raises(ValueError, match="alpha"):
        valid_plan(alpha=bad)


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf"), True, "0"])
def test_invalid_tie_tolerance_fails(bad: object) -> None:
    with pytest.raises(ValueError, match="tie_tolerance"):
        valid_plan(tie_tolerance=bad)


@pytest.mark.parametrize("bad", [-1, True, 1.5])
def test_invalid_root_seed_fails(bad: object) -> None:
    with pytest.raises(ValueError, match="root_seed"):
        valid_plan(root_seed=bad)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"selection_rule": "minimum"}, "selection_rule"),
        ({"statistic_name": "unknown"}, "statistic_name"),
        ({"null_name": "unknown"}, "null_name"),
        ({"failure_policy": "continue"}, "failure_policy"),
    ],
)
def test_malformed_contract_names_and_common_values_fail(
    change: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        valid_plan(**change)


def test_unknown_but_versioned_contract_names_are_valid_requests() -> None:
    plan = valid_plan(
        statistic_name="unknown_statistic_v12",
        statistic_params={"future": [1, 2]},
        null_name="unknown_null_v2",
        null_params={"future": {"enabled": True}},
    )

    assert plan.statistic_name == "unknown_statistic_v12"
    assert plan.null_name == "unknown_null_v2"


@pytest.mark.parametrize(
    ("field_name", "allowed_value"),
    [
        ("statistic_name", "equal_width_binned_nette_v1"),
        ("null_name", "circular_shift_v1"),
        ("failure_policy", "fail_closed_v1"),
    ],
)
def test_plan_names_reject_hash_consistent_equality_spoofing(
    field_name: str,
    allowed_value: str,
) -> None:
    class EqualitySpoof(str):
        def __eq__(self, other: object) -> bool:
            return other == allowed_value

        def __ne__(self, other: object) -> bool:
            return not self == other

        def __hash__(self) -> int:
            return hash(allowed_value)

    spoof = EqualitySpoof("unsupported-spoof")
    assert str(spoof) == "unsupported-spoof"
    assert spoof == allowed_value
    assert hash(spoof) == hash(allowed_value)

    with pytest.raises(ValueError, match=field_name):
        valid_plan(**{field_name: spoof})


def test_supported_selection_rules_are_normalized_to_enums() -> None:
    assert valid_plan().selection_rule is SelectionRule.MAX_UPPER
    assert (
        valid_plan(selection_rule=SelectionRule.MAX_ABSOLUTE).selection_rule
        is SelectionRule.MAX_ABSOLUTE
    )


def test_selection_rule_enum_has_exact_stable_values() -> None:
    assert {member.value for member in SelectionRule} == {"max_upper", "max_absolute"}


@pytest.mark.parametrize(
    "params",
    [
        {np.str_("bins"): 3},
        {"bins": np.int64(3)},
        {"bins": float("nan")},
        {"bins": (3,)},
    ],
)
def test_plan_parameter_boundary_rejects_nonexact_json(params: dict[object, object]) -> None:
    with pytest.raises(ValueError):
        valid_plan(statistic_params=params)


def test_plan_parameter_mappings_are_copied_and_frozen() -> None:
    statistic_params = {"bins": 3}
    null_params = {"min_shift": 1}
    plan = valid_plan(statistic_params=statistic_params, null_params=null_params)

    statistic_params["bins"] = 99
    null_params["min_shift"] = 99

    assert plan.statistic_params == {"bins": 3}
    assert plan.null_params == {"min_shift": 1}
    assert isinstance(plan.statistic_params, MappingProxyType)
    assert isinstance(plan.null_params, MappingProxyType)
    with pytest.raises(TypeError):
        plan.statistic_params["bins"] = 4  # type: ignore[index]


def test_series_pair_is_copied_and_read_only() -> None:
    source = np.array([1.0, 2.0, 3.0])
    pair = SeriesPair(source=source, target=np.array([4.0, 5.0, 6.0]))
    source[0] = 99.0
    assert pair.source[0] == 1.0
    with pytest.raises(ValueError):
        pair.source[0] = 2.0


def test_series_pair_buffer_writeability_cannot_be_reenabled() -> None:
    source = np.array([1.0, 2.0, 3.0])
    pair = SeriesPair(source=source, target=np.array([4.0, 5.0, 6.0]))

    with pytest.raises(ValueError):
        pair.source.setflags(write=True)

    source[0] = 99.0
    assert pair.source[0] == 1.0
    assert not pair.source.flags.writeable
    with pytest.raises(ValueError):
        pair.source[0] = 2.0


def test_series_pair_coerces_both_arrays_to_finite_float64_vectors() -> None:
    target = np.array([4, 5, 6], dtype=np.int32)
    pair = SeriesPair(source=np.array([1, 2, 3]), target=target)
    target[0] = 99

    assert pair.source.dtype == np.float64
    assert pair.target.dtype == np.float64
    assert pair.source.ndim == pair.target.ndim == 1
    assert np.isfinite(pair.source).all()
    assert np.isfinite(pair.target).all()
    assert not pair.source.flags.writeable
    assert not pair.target.flags.writeable
    assert pair.target[0] == 4.0


def test_series_pair_normalizes_only_numeric_zero_to_positive_zero() -> None:
    smallest_subnormal = np.nextafter(0.0, 1.0)
    pair = SeriesPair(
        source=np.array([-0.0, smallest_subnormal]),
        target=np.array([+0.0, -smallest_subnormal]),
    )

    assert not np.signbit(pair.source[0])
    assert not np.signbit(pair.target[0])
    assert pair.source[1] == smallest_subnormal
    assert pair.target[1] == -smallest_subnormal
    assert np.signbit(pair.target[1])


@pytest.mark.parametrize(
    ("source", "target", "message"),
    [
        (np.array([]), np.array([]), "non-empty"),
        (np.ones((2, 2)), np.ones((2, 2)), "one-dimensional"),
        (np.array([1.0, np.nan]), np.array([1.0, 2.0]), "finite"),
        (np.array([1.0, 2.0]), np.array([1.0, np.inf]), "finite"),
        (np.array([1.0]), np.array([1.0, 2.0]), "equal shape"),
    ],
)
def test_invalid_series_pair_fails(
    source: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    target: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        SeriesPair(source=source, target=target)


def test_series_pair_equality_is_array_aware_and_explicitly_unhashable() -> None:
    pair = SeriesPair(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    equal_pair = SeriesPair(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    different_pair = SeriesPair(np.array([1.0, 9.0]), np.array([3.0, 4.0]))

    assert pair == equal_pair
    assert pair != different_pair
    assert SeriesPair.__hash__ is None
    with pytest.raises(TypeError, match="unhashable"):
        hash(pair)


def test_result_numerics_normalize_only_signed_zero() -> None:
    smallest_subnormal = np.nextafter(0.0, 1.0)
    result = statistic_result(estimate=-0.0, selection_score=smallest_subnormal)
    selection = selection_result(decision_statistic=-0.0)

    assert result.estimate == 0.0
    assert not np.signbit(result.estimate)
    assert result.selection_score == smallest_subnormal
    assert selection.decision_statistic == 0.0
    assert not np.signbit(selection.decision_statistic)


def statistic_result(**changes: object) -> StatisticResult:
    values = {
        "candidate_id": 2,
        "estimate": 0.4,
        "selection_score": 0.5,
        "support_n": 12,
        "validity": Validity.VALID,
        "diagnostics": ("stable",),
        "backend_identity": "numpy-v1",
        "preprocessing_identity": "raw-v1",
    }
    values.update(changes)
    return StatisticResult(**values)  # type: ignore[arg-type]


def analytic_failure_statistic() -> StatisticResult:
    return statistic_result(
        estimate=None,
        selection_score=None,
        validity=Validity.ANALYTIC_FAILURE,
        diagnostics=("analytic failure",),
    )


def selection_result(**changes: object) -> SelectionResult:
    values = {
        "selected_candidate": 2,
        "selected_index": 1,
        "decision_statistic": 0.5,
        "tied_candidates": (2, 1),
    }
    values.update(changes)
    return SelectionResult(**values)  # type: ignore[arg-type]


def replicate_outcome(**changes: object) -> ReplicateOutcome:
    values = {
        "replicate_id": 0,
        "seed": 17,
        "status": ReplicateStatus.COMPLETE,
        "statistic_results": (
            statistic_result(candidate_id=1),
            statistic_result(candidate_id=2),
        ),
        "selection": selection_result(),
        "transform_token": {"offsets": [1, {"seed": 17}], "applied": True},
        "diagnostics": ("complete",),
    }
    values.update(changes)
    return ReplicateOutcome(**values)  # type: ignore[arg-type]


def calibration_result(**changes: object) -> CalibrationResult:
    values = {
        "status": RunStatus.COMPLETE,
        "semantic_input_sha256": "a" * 64,
        "scientific_plan_sha256": "b" * 64,
        "observed_results": (
            statistic_result(candidate_id=1),
            statistic_result(candidate_id=2),
        ),
        "observed_selection": selection_result(),
        "replicates": (replicate_outcome(),),
        "exceedance_count": 0,
        "failure_count": 0,
        "p_value": 0.3,
        "exceedance_bound_low": 0.3,
        "exceedance_bound_high": 0.3,
        "reject_null": False,
        "diagnostics": ("complete",),
    }
    values.update(changes)
    return CalibrationResult(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("factory", [valid_plan, replicate_outcome, calibration_result])
def test_mapping_bearing_contracts_are_structurally_equal_and_explicitly_unhashable(
    factory: Callable[[], object],
) -> None:
    first = factory()
    second = factory()

    assert first == second
    assert type(first).__hash__ is None
    with pytest.raises(TypeError, match="unhashable"):
        hash(first)


def test_plan_outcomes_and_results_preserve_structural_inequality() -> None:
    assert valid_plan() != valid_plan(root_seed=18)
    assert statistic_result() != StatisticResult(
        candidate_id=2,
        estimate=0.6,
        selection_score=0.5,
        support_n=12,
        validity=Validity.VALID,
        diagnostics=("stable",),
        backend_identity="numpy-v1",
        preprocessing_identity="raw-v1",
    )
    assert replicate_outcome() != ReplicateOutcome(
        replicate_id=1,
        seed=17,
        status=ReplicateStatus.COMPLETE,
        statistic_results=(
            statistic_result(candidate_id=1),
            statistic_result(candidate_id=2),
        ),
        selection=selection_result(),
        transform_token={"offsets": [1, {"seed": 17}], "applied": True},
        diagnostics=("complete",),
    )
    assert calibration_result() == calibration_result()


@pytest.mark.parametrize(
    "changes",
    [
        {"candidate_id": True},
        {"candidate_id": 0},
        {"candidate_id": -1},
        {"candidate_id": 1.5},
        {"support_n": True},
        {"support_n": 0},
        {"support_n": -1},
        {"support_n": 1.5},
        {"estimate": None},
        {"estimate": True},
        {"estimate": float("nan")},
        {"estimate": float("inf")},
        {"selection_score": True},
        {"selection_score": float("nan")},
        {"selection_score": float("inf")},
        {"diagnostics": ("valid", 3)},
    ],
)
def test_invalid_valid_statistic_result_fails(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        statistic_result(**changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"estimate": 0.4, "selection_score": None},
        {"estimate": None, "selection_score": 0.5},
    ],
)
def test_contradictory_analytic_failure_statistic_fails(
    changes: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="ANALYTIC_FAILURE"):
        statistic_result(validity=Validity.ANALYTIC_FAILURE, **changes)


def test_statistic_result_normalizes_validity_and_allows_deferred_selection_score() -> None:
    valid = statistic_result(validity="valid", selection_score=None)
    failure = analytic_failure_statistic()

    assert valid.validity is Validity.VALID
    assert valid.selection_score is None
    assert failure.validity is Validity.ANALYTIC_FAILURE


@pytest.mark.parametrize("field_name", ["backend_identity", "preprocessing_identity"])
@pytest.mark.parametrize("bad_identity", [[], {}, None, ""])
def test_statistic_result_rejects_invalid_identity_strings(
    field_name: str,
    bad_identity: object,
) -> None:
    with pytest.raises(ValueError, match=field_name):
        statistic_result(**{field_name: bad_identity})


def test_statistic_result_normalizes_benign_identity_string_subclasses() -> None:
    class BenignString(str):
        pass

    result = statistic_result(
        backend_identity=BenignString("backend-v2"),
        preprocessing_identity=BenignString("preprocessing-v2"),
    )

    assert result.backend_identity == "backend-v2"
    assert type(result.backend_identity) is str
    assert result.preprocessing_identity == "preprocessing-v2"
    assert type(result.preprocessing_identity) is str


@pytest.mark.parametrize(
    "changes",
    [
        {"selected_candidate": True},
        {"selected_candidate": 0},
        {"selected_index": True},
        {"selected_index": -1},
        {"selected_index": 1.5},
        {"decision_statistic": True},
        {"decision_statistic": float("nan")},
        {"decision_statistic": float("inf")},
        {"tied_candidates": ()},
        {"tied_candidates": (3,)},
    ],
)
def test_invalid_selection_result_fails(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        selection_result(**changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"replicate_id": True},
        {"replicate_id": -1},
        {"replicate_id": 1.5},
        {"seed": True},
        {"seed": -1},
        {"seed": 1.5},
        {"selection": None},
        {"statistic_results": (analytic_failure_statistic(),)},
        {"diagnostics": ("complete", 3)},
    ],
)
def test_invalid_complete_replicate_outcome_fails(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        replicate_outcome(**changes)


def test_complete_replicate_requires_nonempty_statistic_results() -> None:
    with pytest.raises(ValueError, match="COMPLETE"):
        replicate_outcome(statistic_results=())


@pytest.mark.parametrize(
    "selection",
    [
        selection_result(selected_index=2),
        selection_result(selected_candidate=2, selected_index=0, tied_candidates=(2,)),
    ],
)
def test_complete_replicate_selection_must_match_vector(
    selection: SelectionResult,
) -> None:
    with pytest.raises(ValueError, match="COMPLETE"):
        replicate_outcome(selection=selection)


@pytest.mark.parametrize(
    ("candidate_ids", "selection"),
    [
        ((2, 1), selection_result(selected_candidate=2, selected_index=0, tied_candidates=(2,))),
        ((1, 1), selection_result(selected_candidate=1, selected_index=0, tied_candidates=(1,))),
    ],
)
def test_complete_replicate_requires_canonical_candidate_vector(
    candidate_ids: tuple[int, int], selection: SelectionResult
) -> None:
    results = tuple(
        statistic_result(candidate_id=candidate_id) for candidate_id in candidate_ids
    )
    with pytest.raises(ValueError, match="COMPLETE"):
        replicate_outcome(statistic_results=results, selection=selection)


def test_complete_replicate_ties_must_be_in_candidate_vector() -> None:
    with pytest.raises(ValueError, match="COMPLETE"):
        replicate_outcome(
            selection=selection_result(tied_candidates=(2, 3)),
        )


def test_complete_replicate_allows_deferred_selection_scores() -> None:
    outcome = replicate_outcome(
        statistic_results=(
            statistic_result(candidate_id=1, selection_score=None),
            statistic_result(candidate_id=2, selection_score=None),
        )
    )

    assert all(result.selection_score is None for result in outcome.statistic_results)


def test_analytic_failure_replicate_rejects_selection() -> None:
    with pytest.raises(ValueError, match="ANALYTIC_FAILURE"):
        replicate_outcome(
            status=ReplicateStatus.ANALYTIC_FAILURE,
            selection=selection_result(),
            statistic_results=(),
        )


@pytest.mark.parametrize("candidate_ids", [(2, 1), (1, 1)])
def test_analytic_failure_replicate_retains_all_valid_invalid_vector(
    candidate_ids: tuple[int, int],
) -> None:
    statistic_results = tuple(
        statistic_result(candidate_id=candidate_id) for candidate_id in candidate_ids
    )

    outcome = replicate_outcome(
        status=ReplicateStatus.ANALYTIC_FAILURE,
        selection=None,
        statistic_results=statistic_results,
    )

    assert tuple(result.candidate_id for result in outcome.statistic_results) == candidate_ids


def test_analytic_failure_replicate_allows_null_or_statistic_failure_shapes() -> None:
    null_failure = replicate_outcome(
        status="analytic_failure", selection=None, statistic_results=()
    )
    statistic_failure = replicate_outcome(
        status=ReplicateStatus.ANALYTIC_FAILURE,
        selection=None,
        statistic_results=(statistic_result(), analytic_failure_statistic()),
    )

    assert null_failure.status is ReplicateStatus.ANALYTIC_FAILURE
    assert statistic_failure.statistic_results[-1].validity is Validity.ANALYTIC_FAILURE


@pytest.mark.parametrize(
    "changes",
    [
        {"exceedance_count": True},
        {"exceedance_count": -1},
        {"exceedance_count": 1.5},
        {"failure_count": True},
        {"failure_count": -1},
        {"failure_count": 1.5},
        {"exceedance_count": 2},
        {"diagnostics": ("complete", 3)},
    ],
)
def test_invalid_calibration_scalars_and_counts_fail(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        calibration_result(**changes)


def test_calibration_failure_count_must_match_present_replicates() -> None:
    failed_replicate = replicate_outcome(
        status=ReplicateStatus.ANALYTIC_FAILURE,
        selection=None,
        statistic_results=(),
    )
    with pytest.raises(ValueError, match="failure_count"):
        calibration_result(
            status=RunStatus.NOT_EVALUABLE,
            replicates=(failed_replicate,),
            failure_count=0,
            p_value=None,
            exceedance_bound_low=None,
            exceedance_bound_high=None,
            reject_null=None,
        )


@pytest.mark.parametrize(
    ("field_name", "bad"),
    [
        ("p_value", True),
        ("p_value", -0.1),
        ("p_value", 1.1),
        ("p_value", float("nan")),
        ("p_value", float("inf")),
        ("exceedance_bound_low", True),
        ("exceedance_bound_low", -0.1),
        ("exceedance_bound_low", float("nan")),
        ("exceedance_bound_high", 1.1),
        ("exceedance_bound_high", float("inf")),
    ],
)
def test_invalid_calibration_probabilities_fail(field_name: str, bad: object) -> None:
    with pytest.raises(ValueError):
        calibration_result(**{field_name: bad})


@pytest.mark.parametrize(
    "changes",
    [
        {"exceedance_bound_low": None},
        {"exceedance_bound_high": None},
        {"exceedance_bound_low": 0.8, "exceedance_bound_high": 0.2},
    ],
)
def test_calibration_bounds_must_form_an_ordered_pair(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="bound"):
        calibration_result(**changes)


def test_complete_calibration_requires_nonempty_observed_results() -> None:
    with pytest.raises(ValueError, match="COMPLETE"):
        calibration_result(observed_results=())


def test_complete_calibration_requires_nonempty_replicates() -> None:
    with pytest.raises(ValueError, match="COMPLETE"):
        calibration_result(replicates=())


def test_complete_calibration_observed_selection_must_match_vector() -> None:
    with pytest.raises(ValueError, match="COMPLETE"):
        calibration_result(
            observed_selection=selection_result(
                selected_candidate=2,
                selected_index=0,
                tied_candidates=(2,),
            )
        )


def test_complete_calibration_replicate_vector_must_match_observed_vector() -> None:
    mismatched_replicate = replicate_outcome(
        statistic_results=(
            statistic_result(candidate_id=1),
            statistic_result(candidate_id=3),
        ),
        selection=selection_result(
            selected_candidate=3,
            selected_index=1,
            tied_candidates=(3,),
        ),
    )
    with pytest.raises(ValueError, match="COMPLETE"):
        calibration_result(replicates=(mismatched_replicate,))


@pytest.mark.parametrize("bad", ["false", 0])
def test_complete_calibration_reject_null_must_be_exact_bool(bad: object) -> None:
    with pytest.raises(ValueError, match="COMPLETE"):
        calibration_result(reject_null=bad)


@pytest.mark.parametrize(
    "changes",
    [
        {"observed_selection": None},
        {"observed_results": (analytic_failure_statistic(),)},
        {
            "replicates": (
                replicate_outcome(
                    status=ReplicateStatus.ANALYTIC_FAILURE,
                    selection=None,
                    statistic_results=(),
                ),
            ),
            "failure_count": 1,
        },
        {"p_value": None},
        {"exceedance_bound_low": None, "exceedance_bound_high": None},
        {"reject_null": None},
        {"exceedance_bound_low": 0.2, "exceedance_bound_high": 0.4},
    ],
)
def test_contradictory_complete_calibration_state_fails(
    changes: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="COMPLETE"):
        calibration_result(**changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"p_value": 0.3, "reject_null": None},
        {"p_value": None, "reject_null": False},
    ],
)
def test_contradictory_not_evaluable_calibration_state_fails(
    changes: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="NOT_EVALUABLE"):
        calibration_result(
            status=RunStatus.NOT_EVALUABLE,
            exceedance_bound_low=None,
            exceedance_bound_high=None,
            **changes,
        )


def test_not_evaluable_calibration_allows_observation_or_replicate_failure() -> None:
    observation_failure = calibration_result(
        status=RunStatus.NOT_EVALUABLE,
        observed_results=(analytic_failure_statistic(),),
        observed_selection=None,
        replicates=(),
        exceedance_count=0,
        failure_count=0,
        p_value=None,
        exceedance_bound_low=None,
        exceedance_bound_high=None,
        reject_null=None,
    )
    failed_replicate = replicate_outcome(
        status=ReplicateStatus.ANALYTIC_FAILURE,
        selection=None,
        statistic_results=(),
    )
    replicate_failure = calibration_result(
        status="not_evaluable",
        replicates=(failed_replicate,),
        exceedance_count=0,
        failure_count=1,
        p_value=None,
        exceedance_bound_low=0.0,
        exceedance_bound_high=1.0,
        reject_null=None,
    )

    assert observation_failure.status is RunStatus.NOT_EVALUABLE
    assert observation_failure.replicates == ()
    assert replicate_failure.failure_count == 1


def test_not_evaluable_retains_noncanonical_observed_vector() -> None:
    result = calibration_result(
        status=RunStatus.NOT_EVALUABLE,
        observed_results=(
            statistic_result(candidate_id=2),
            statistic_result(candidate_id=1),
        ),
        observed_selection=None,
        replicates=(),
        exceedance_count=0,
        failure_count=0,
        p_value=None,
        exceedance_bound_low=None,
        exceedance_bound_high=None,
        reject_null=None,
    )

    assert tuple(item.candidate_id for item in result.observed_results) == (2, 1)


@pytest.mark.parametrize(
    ("contract", "expected"),
    [
        (
            StatisticResult,
            (
                "candidate_id",
                "estimate",
                "selection_score",
                "support_n",
                "validity",
                "diagnostics",
                "backend_identity",
                "preprocessing_identity",
            ),
        ),
        (
            SelectionResult,
            ("selected_candidate", "selected_index", "decision_statistic", "tied_candidates"),
        ),
        (
            ReplicateOutcome,
            (
                "replicate_id",
                "seed",
                "status",
                "statistic_results",
                "selection",
                "transform_token",
                "diagnostics",
            ),
        ),
        (
            CalibrationResult,
            (
                "status",
                "semantic_input_sha256",
                "scientific_plan_sha256",
                "observed_results",
                "observed_selection",
                "replicates",
                "exceedance_count",
                "failure_count",
                "p_value",
                "exceedance_bound_low",
                "exceedance_bound_high",
                "reject_null",
                "diagnostics",
            ),
        ),
    ],
)
def test_result_contracts_contain_exactly_the_specified_fields(
    contract: type[object], expected: tuple[str, ...]
) -> None:
    assert tuple(field.name for field in fields(contract)) == expected


@pytest.mark.parametrize(
    "instance",
    [statistic_result(), selection_result(), replicate_outcome(), calibration_result()],
)
def test_result_contracts_are_frozen(instance: object) -> None:
    field_name = fields(type(instance))[0].name
    with pytest.raises(FrozenInstanceError):
        setattr(instance, field_name, None)


def test_result_contracts_normalize_relevant_tuples_and_candidate_identity() -> None:
    result = statistic_result()
    selection = SelectionResult(2, 0, 0.5, [2])  # type: ignore[arg-type]
    replicate = ReplicateOutcome(
        0,
        17,
        ReplicateStatus.COMPLETE,
        [result],  # type: ignore[arg-type]
        selection,
        None,
        ["complete"],  # type: ignore[arg-type]
    )
    calibration = CalibrationResult(
        RunStatus.COMPLETE,
        "a" * 64,
        "b" * 64,
        [result],  # type: ignore[arg-type]
        selection,
        [replicate],  # type: ignore[arg-type]
        0,
        0,
        0.3,
        0.3,
        0.3,
        False,
        ["complete"],  # type: ignore[arg-type]
    )

    assert selection.tied_candidates == (2,)
    assert replicate.statistic_results == (result,)
    assert replicate.diagnostics == ("complete",)
    assert calibration.observed_results == (result,)
    assert calibration.replicates == (replicate,)
    assert calibration.diagnostics == ("complete",)


def test_replicate_transform_token_is_recursively_copied_and_frozen() -> None:
    token = {"offsets": [1, {"seed": 17}], "applied": True}
    outcome = ReplicateOutcome(
        replicate_id=0,
        seed=17,
        status=ReplicateStatus.COMPLETE,
        statistic_results=(
            statistic_result(candidate_id=1),
            statistic_result(candidate_id=2),
        ),
        selection=selection_result(),
        transform_token=token,
        diagnostics=(),
    )

    token["offsets"][1]["seed"] = 99  # type: ignore[index]

    assert outcome.transform_token is not None
    assert outcome.transform_token["offsets"][1]["seed"] == 17  # type: ignore[index]
    with pytest.raises(TypeError):
        outcome.transform_token["new"] = 1  # type: ignore[index]
    with pytest.raises(TypeError):
        outcome.transform_token["offsets"][1]["seed"] = 2  # type: ignore[index]


def test_recursive_json_normalizes_accepted_string_subclasses_to_builtin_str() -> None:
    class StringSubclass(str):
        pass

    outcome = replicate_outcome(
        transform_token={
            StringSubclass("outer"): [
                np.str_("value"),
                {np.str_("nested"): StringSubclass("text")},
            ]
        }
    )

    assert outcome.transform_token is not None
    outer_key = next(iter(outcome.transform_token))
    outer_value = outcome.transform_token[outer_key]
    assert type(outer_key) is str
    assert isinstance(outer_value, tuple)
    assert type(outer_value[0]) is str
    assert isinstance(outer_value[1], MappingProxyType)
    nested_key = next(iter(outer_value[1]))
    assert type(nested_key) is str
    assert type(outer_value[1][nested_key]) is str


def test_status_enums_have_stable_values_needed_by_frozen_results() -> None:
    assert {member.value for member in RunStatus} == {
        "complete",
        "not_evaluable",
    }
    assert {member.value for member in Validity} == {"valid", "analytic_failure"}
    assert {member.value for member in ReplicateStatus} == {
        "complete",
        "analytic_failure",
    }
