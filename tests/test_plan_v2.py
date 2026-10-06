from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from types import MappingProxyType

import numpy as np
import pytest

from selcal.contracts import PlanRequest, SelectionRule
from selcal.contracts_v2 import PlanRequestV2


def valid_request_v2(**changes: object) -> PlanRequestV2:
    values: dict[str, object] = {
        "candidates": (3, 1, 2),
        "statistic_name": "lagged_pearson_v1",
        "statistic_params": {},
        "selection_rule": "max_upper",
        "null_name": "circular_shift_v2",
        "null_params": {"min_shift": 1},
        "replicates": 9,
        "alpha": 0.05,
        "tie_tolerance": 1e-12,
        "root_seed": 17,
    }
    values.update(changes)
    return PlanRequestV2(**values)  # type: ignore[arg-type]


def test_v2_request_is_a_physically_separate_exact_contract() -> None:
    v2 = valid_request_v2()
    v1 = PlanRequest(
        candidates=(3, 1, 2),
        statistic_name="lagged_pearson_v1",
        statistic_params={},
        selection_rule="max_upper",
        null_name="circular_shift_v1",
        null_params={"min_shift": 1},
        replicates=9,
        alpha=0.05,
        tie_tolerance=1e-12,
        root_seed=17,
        failure_policy="fail_closed_v1",
    )

    assert type(v2) is PlanRequestV2
    assert not isinstance(v2, PlanRequest)
    assert v2 != v1
    assert v1 != v2
    assert not hasattr(v2, "__dict__")
    with pytest.raises(TypeError, match="unhashable"):
        hash(v2)
    with pytest.raises(FrozenInstanceError):
        v2.root_seed = 18  # type: ignore[misc]


def test_v2_request_canonicalizes_candidates_and_selection_rule() -> None:
    request = valid_request_v2(
        candidates=(9, 2, 5), selection_rule=SelectionRule.MAX_ABSOLUTE
    )

    assert request.candidates == (2, 5, 9)
    assert request.selection_rule is SelectionRule.MAX_ABSOLUTE


@pytest.mark.parametrize(
    "candidates",
    [(), (1, 1), (0, 1), (-1, 1), (True, 1), (1.5, 2), None],
)
def test_v2_request_rejects_invalid_candidate_collections(candidates: object) -> None:
    with pytest.raises(ValueError, match="candidates"):
        valid_request_v2(candidates=candidates)


def test_v2_request_deep_freezes_exact_json_parameters_without_aliasing() -> None:
    statistic_params = {"outer": [{"x": 1.5}, True, None]}
    null_params = {"min_shift": 1, "labels": ["a", "b"]}

    request = valid_request_v2(
        statistic_params=statistic_params,
        null_params=MappingProxyType(null_params),
    )
    statistic_params["outer"] = []
    null_params["labels"] = []

    assert type(request.statistic_params) is MappingProxyType
    assert type(request.null_params) is MappingProxyType
    assert request.statistic_params == {"outer": (MappingProxyType({"x": 1.5}), True, None)}
    assert request.null_params == {"min_shift": 1, "labels": ("a", "b")}
    with pytest.raises(TypeError):
        request.null_params["min_shift"] = 2  # type: ignore[index]


@pytest.mark.parametrize("value", [np.int64(1), np.float64(1.0), np.bool_(True)])
@pytest.mark.parametrize("field", ["statistic_params", "null_params"])
def test_v2_request_rejects_numpy_scalars_in_parameters(
    field: str, value: object
) -> None:
    with pytest.raises(ValueError, match="exact JSON"):
        valid_request_v2(**{field: {"value": value}})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("statistic_name", "lagged_pearson"),
        ("statistic_name", "Lagged_Pearson_v1"),
        ("statistic_name", 1),
        ("null_name", "circular_shift"),
        ("null_name", "circular-shift-v2"),
        ("null_name", None),
        ("selection_rule", "minimum"),
        ("selection_rule", "MAX_UPPER"),
        ("selection_rule", True),
    ],
)
def test_v2_request_rejects_invalid_contract_names_and_selection_rules(
    field: str, value: object
) -> None:
    with pytest.raises(ValueError):
        valid_request_v2(**{field: value})


@pytest.mark.parametrize(
    "root_seed",
    [True, np.int64(1), -1, 2**64, 0.0, "1", None],
)
def test_v2_root_seed_requires_exact_bounded_builtin_int(root_seed: object) -> None:
    with pytest.raises(ValueError, match="root_seed"):
        valid_request_v2(root_seed=root_seed)


@pytest.mark.parametrize("root_seed", [0, 2**64 - 1])
def test_v2_root_seed_accepts_both_boundaries(root_seed: int) -> None:
    assert valid_request_v2(root_seed=root_seed).root_seed == root_seed


@pytest.mark.parametrize(
    "replicates",
    [True, np.int64(1), 0, 1_000_001, 1.0, "1", None],
)
def test_v2_replicates_requires_exact_bounded_builtin_int(replicates: object) -> None:
    with pytest.raises(ValueError, match="replicates"):
        valid_request_v2(replicates=replicates)


@pytest.mark.parametrize("replicates", [1, 1_000_000])
def test_v2_replicates_accepts_both_boundaries(replicates: int) -> None:
    assert valid_request_v2(replicates=replicates).replicates == replicates


@pytest.mark.parametrize(
    "alpha",
    [
        True,
        1,
        np.float64(0.05),
        0.0,
        1.0,
        -0.1,
        1.1,
        float("nan"),
        float("inf"),
        -float("inf"),
        "0.05",
        None,
    ],
)
def test_v2_alpha_requires_finite_open_interval_builtin_float(alpha: object) -> None:
    with pytest.raises(ValueError, match="alpha"):
        valid_request_v2(alpha=alpha)


@pytest.mark.parametrize(
    "tie_tolerance",
    [
        True,
        0,
        np.float64(0.0),
        -0.1,
        float("nan"),
        float("inf"),
        -float("inf"),
        "0",
        None,
    ],
)
def test_v2_tie_tolerance_requires_nonnegative_finite_builtin_float(
    tie_tolerance: object,
) -> None:
    with pytest.raises(ValueError, match="tie_tolerance"):
        valid_request_v2(tie_tolerance=tie_tolerance)


@pytest.mark.parametrize("tie_tolerance", [0.0, 1e-12])
def test_v2_tie_tolerance_accepts_nonnegative_builtin_float(
    tie_tolerance: float,
) -> None:
    assert valid_request_v2(tie_tolerance=tie_tolerance).tie_tolerance == tie_tolerance


def test_v2_request_exposes_the_frozen_plan_contract_constants() -> None:
    request = valid_request_v2()

    assert request.schema == "selcal.scientific-plan.v2"
    assert request.decision_contract == "family_max_with_canonical_tie_label_v2"
    assert request.failure_contract == "exact_b_fail_closed_v2"
    assert (
        request.rng_contract
        == "sha256_framed_plan_rid_stream_to_pcg64_raw64_v1"
    )
    assert request.common_support == "max_candidate_lag_v1"
    assert request.calibration_contract == "full_reselection_global_mc_plus_one_v2"
    assert {field.name for field in fields(PlanRequestV2)} == {
        "candidates",
        "statistic_name",
        "statistic_params",
        "selection_rule",
        "null_name",
        "null_params",
        "replicates",
        "alpha",
        "tie_tolerance",
        "root_seed",
    }


def test_v2_request_equality_is_exact_and_value_based() -> None:
    first = valid_request_v2(
        candidates=(3, 2, 1),
        statistic_params={"b": [True, {"x": 1.0}], "a": 1},
    )
    second = valid_request_v2(
        candidates=(1, 2, 3),
        statistic_params={"a": 1, "b": [True, {"x": 1.0}]},
    )

    assert first == second
    assert first != valid_request_v2(root_seed=18)


@pytest.mark.parametrize("field", ["statistic_params", "null_params"])
@pytest.mark.parametrize(
    ("left", "right"),
    [
        ({"value": True}, {"value": 1}),
        ({"value": 1}, {"value": 1.0}),
        ({"outer": [{"value": True}]}, {"outer": [{"value": 1}]}),
        ({"outer": [{"value": 1}]}, {"outer": [{"value": 1.0}]}),
    ],
    ids=["bool-int", "int-float", "nested-bool-int", "nested-int-float"],
)
def test_v2_request_equality_preserves_exact_json_scalar_identity(
    field: str,
    left: dict[str, object],
    right: dict[str, object],
) -> None:
    assert valid_request_v2(**{field: left}) != valid_request_v2(**{field: right})
