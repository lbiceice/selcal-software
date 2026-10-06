from __future__ import annotations

import math
import warnings
from collections.abc import Iterator, Mapping
from dataclasses import FrozenInstanceError
from operator import setitem
from types import MappingProxyType

import numpy as np
import pytest

import selcal.statistics as statistics
import selcal.statistics.lagged_pearson as lagged_pearson_module
from selcal.contracts import AnalyticFailure, SelectionRule, SeriesPair, Validity
from selcal.parameters import freeze_exact_json_mapping
from selcal.statistics.base import BoundStatisticAdapter, StatisticAdapter
from selcal.statistics.lagged_pearson import (
    LaggedPearsonAdapter,
    _BoundLaggedPearsonAdapter,
    _clamp_correlation,
)


class _CustomMapping(Mapping[str, object]):
    def __getitem__(self, key: str) -> object:
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(())

    def __len__(self) -> int:
        return 0


class _DictSubclass(dict[str, object]):
    pass


class _TupleSubclass(tuple[int, ...]):
    pass


class _StringSubclass(str):
    pass


def _pair(source: list[float], target: list[float]) -> SeriesPair:
    return SeriesPair(
        source=np.asarray(source, dtype=np.float64),
        target=np.asarray(target, dtype=np.float64),
    )


def _manual_pearson(source: list[float], target: list[float]) -> float:
    """Independent Python-sum oracle, separate from the NumPy implementation."""

    source_mean = sum(source) / len(source)
    target_mean = sum(target) / len(target)
    source_centered = [value - source_mean for value in source]
    target_centered = [value - target_mean for value in target]
    numerator = sum(
        source_value * target_value
        for source_value, target_value in zip(source_centered, target_centered, strict=True)
    )
    source_ss = sum(value * value for value in source_centered)
    target_ss = sum(value * value for value in target_centered)
    return numerator / math.sqrt(source_ss * target_ss)


def _manual_scaled_pearson(source: list[float], target: list[float]) -> float:
    """Independent overflow-safe oracle using Python scaling and ``math.fsum``."""

    source_scale = max(abs(value) for value in source)
    target_scale = max(abs(value) for value in target)
    source_scaled = [value / source_scale for value in source]
    target_scaled = [value / target_scale for value in target]
    source_mean = math.fsum(source_scaled) / len(source_scaled)
    target_mean = math.fsum(target_scaled) / len(target_scaled)
    source_centered = [value - source_mean for value in source_scaled]
    target_centered = [value - target_mean for value in target_scaled]
    numerator = math.fsum(
        source_value * target_value
        for source_value, target_value in zip(source_centered, target_centered, strict=True)
    )
    source_norm = math.sqrt(math.fsum(value * value for value in source_centered))
    target_norm = math.sqrt(math.fsum(value * value for value in target_centered))
    return numerator / (source_norm * target_norm)


def _manual_common_support_estimates(
    pair: SeriesPair, candidates: tuple[int, ...]
) -> tuple[float, ...]:
    start = max(candidates)
    future = [float(value) for value in pair.target[start:]]
    return tuple(
        _manual_pearson(
            [float(value) for value in pair.source[start - candidate : -candidate]],
            future,
        )
        for candidate in candidates
    )


def _fixture() -> SeriesPair:
    return _pair(
        [0, 1, 4, 2, 5, 3, 7, 6],
        [8, 6, 9, -4, -2, -5, -3, -7],
    )


def _direct_bound(**overrides: object) -> _BoundLaggedPearsonAdapter:
    values: dict[str, object] = {
        "observed_length": 8,
        "candidates": (1, 3),
    }
    values.update(overrides)
    return _BoundLaggedPearsonAdapter(**values)  # type: ignore[arg-type]


def test_adapter_protocol_name_and_empty_immutable_parameters() -> None:
    adapter = LaggedPearsonAdapter()
    second = LaggedPearsonAdapter.from_parameters({})

    assert isinstance(adapter, StatisticAdapter)
    assert adapter.name == "lagged_pearson_v1"
    assert adapter.parameters == {}
    assert isinstance(adapter.parameters, MappingProxyType)
    assert adapter.parameters is not second.parameters
    with pytest.raises(TypeError):
        setitem(adapter.parameters, "hidden", 1)
    parameter_attribute = "_parameters"
    with pytest.raises(FrozenInstanceError):
        setattr(adapter, parameter_attribute, MappingProxyType({"hidden": 1}))


def test_adapter_is_publicly_exported_without_removing_existing_exports() -> None:
    assert statistics.LaggedPearsonAdapter is LaggedPearsonAdapter
    assert statistics.__all__ == [
        "BinnedNetTEAdapter",
        "BoundStatisticAdapter",
        "LaggedPearsonAdapter",
        "StatisticAdapter",
        "conditional_mutual_information",
    ]


def test_from_parameters_accepts_exact_empty_dict_and_trusted_mapping_proxy() -> None:
    trusted = freeze_exact_json_mapping({}, name="test parameters")

    from_dict = LaggedPearsonAdapter.from_parameters({})
    from_proxy = LaggedPearsonAdapter.from_parameters(trusted)

    assert from_dict.parameters == from_proxy.parameters == {}
    assert isinstance(from_dict.parameters, MappingProxyType)
    assert isinstance(from_proxy.parameters, MappingProxyType)


@pytest.mark.parametrize(
    "parameters",
    [
        {"extra": 1},
        [],
        None,
        _CustomMapping(),
        _DictSubclass(),
        {"$float64": 1},
        {"value": np.int64(1)},
        {"value": (1,)},
    ],
)
def test_from_parameters_rejects_nonempty_nonmapping_and_nonexact_payloads(
    parameters: object,
) -> None:
    with pytest.raises(ValueError, match=r"parameter|dict|MappingProxyType|reserved|JSON"):
        LaggedPearsonAdapter.from_parameters(parameters)


def test_from_parameters_rejects_cycles_through_the_existing_hardened_freezer() -> None:
    cyclic: dict[str, object] = {}
    cyclic["self"] = cyclic

    with pytest.raises(ValueError, match="cyclic"):
        LaggedPearsonAdapter.from_parameters(cyclic)


def test_bind_canonicalizes_candidates_and_freezes_exact_bound_state() -> None:
    pair = _fixture()
    bound = LaggedPearsonAdapter().bind(pair, (3, 1))

    assert isinstance(bound, BoundStatisticAdapter)
    assert bound.name == "lagged_pearson_v1"
    assert bound.observed_length == 8
    assert type(bound.observed_length) is int
    assert bound.candidates == (1, 3)
    assert type(bound.candidates) is tuple
    with pytest.raises(FrozenInstanceError):
        bound.observed_length = 9  # type: ignore[misc]


def test_bind_requires_series_pair_and_at_least_two_common_support_samples() -> None:
    pair = _pair([0, 1, 2, 3], [3, 2, 1, 0])

    with pytest.raises(ValueError, match=r"SeriesPair"):
        LaggedPearsonAdapter().bind(object(), (1,))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"at least 2|support"):
        LaggedPearsonAdapter().bind(pair, (1, 3))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"observed_length": True}, "observed_length"),
        ({"observed_length": np.int64(8)}, "observed_length"),
        ({"observed_length": 8.0}, "observed_length"),
        ({"observed_length": 4}, "at least 2"),
        ({"candidates": [1, 3]}, "canonical"),
        ({"candidates": _TupleSubclass((1, 3))}, "canonical"),
        ({"candidates": (3, 1)}, "canonical"),
        ({"candidates": (np.int64(1), 3)}, "canonical"),
        ({"candidates": (True, 3)}, "canonical"),
        ({"candidates": (1, 1)}, "canonical"),
    ],
)
def test_direct_bound_construction_fails_closed_on_malformed_state(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _direct_bound(**overrides)


def test_evaluate_matches_independent_oracle_on_frozen_max_lag_common_support() -> None:
    pair = _fixture()
    expected = _manual_common_support_estimates(pair, (1, 3))
    bound = LaggedPearsonAdapter().bind(pair, (3, 1))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        results = bound.evaluate(pair, (1, 3), SelectionRule.MAX_UPPER)

    assert tuple(result.candidate_id for result in results) == (1, 3)
    assert tuple(result.estimate for result in results) == pytest.approx(expected, abs=1e-15)
    assert tuple(result.selection_score for result in results) == pytest.approx(
        expected, abs=1e-15
    )
    assert {result.support_n for result in results} == {5}
    assert {result.validity for result in results} == {Validity.VALID}
    assert {result.diagnostics for result in results} == {("SELCAL_LAGGED_PEARSON_OK",)}


def test_selection_rule_changes_only_score_and_preserves_negative_signed_estimates() -> None:
    pair = _fixture()
    bound = LaggedPearsonAdapter().bind(pair, (1, 3))

    upper = bound.evaluate(pair, (1, 3), "max_upper")
    absolute = bound.evaluate(pair, (1, 3), "max_absolute")

    upper_estimates = tuple(result.estimate for result in upper)
    absolute_estimates = tuple(result.estimate for result in absolute)
    assert upper_estimates == absolute_estimates
    assert upper_estimates[0] == pytest.approx(-1.0, abs=1e-15)
    assert upper[0].selection_score == pytest.approx(-1.0, abs=1e-15)
    assert absolute[0].selection_score == pytest.approx(1.0, abs=1e-15)
    assert tuple(result.selection_score for result in absolute) == pytest.approx(
        tuple(abs(value) for value in upper_estimates if value is not None),
        abs=1e-15,
    )


def test_source_zero_variance_is_visible_per_candidate() -> None:
    pair = _pair(
        [0, 1, 5, 5, 5, 5, 5, 9],
        [8, 7, 6, 1, 3, 2, 5, 4],
    )
    results = (
        LaggedPearsonAdapter()
        .bind(pair, (1, 3))
        .evaluate(pair, (1, 3), SelectionRule.MAX_UPPER)
    )

    assert results[0].validity is Validity.ANALYTIC_FAILURE
    assert results[0].estimate is results[0].selection_score is None
    assert results[0].diagnostics == ("SELCAL_LAGGED_PEARSON_ZERO_SOURCE_VARIANCE",)
    assert results[1].validity is Validity.VALID
    assert results[1].diagnostics == ("SELCAL_LAGGED_PEARSON_OK",)


def test_target_zero_variance_fails_every_candidate() -> None:
    pair = _pair(
        [0, 1, 4, 2, 5, 3, 7, 6],
        [9, 8, 7, 2, 2, 2, 2, 2],
    )
    results = (
        LaggedPearsonAdapter()
        .bind(pair, (1, 3))
        .evaluate(pair, (1, 3), SelectionRule.MAX_ABSOLUTE)
    )

    assert {result.validity for result in results} == {Validity.ANALYTIC_FAILURE}
    assert {result.estimate for result in results} == {None}
    assert {result.selection_score for result in results} == {None}
    assert {result.diagnostics for result in results} == {
        ("SELCAL_LAGGED_PEARSON_ZERO_TARGET_VARIANCE",)
    }


def test_mutated_nonfinite_pair_is_not_misreported_as_analytic_failure() -> None:
    pair = _fixture()
    bound = LaggedPearsonAdapter().bind(pair, (1, 3))
    corrupted_target = pair.target.copy()
    corrupted_target[4] = np.nan
    object.__setattr__(pair, "target", corrupted_target)


    with pytest.raises(ValueError, match="finite"):
        bound.evaluate(pair, (1, 3), SelectionRule.MAX_UPPER)


@pytest.mark.parametrize("error_type", [TypeError, ValueError, OverflowError])
def test_common_support_implementation_errors_are_not_analytic_failures(
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[Exception],
) -> None:
    pair = _fixture()
    bound = LaggedPearsonAdapter().bind(pair, (1, 3))

    def fail_common_support(*args: object, **kwargs: object) -> object:
        raise error_type("injected common-support implementation failure")

    monkeypatch.setattr(lagged_pearson_module, "common_support", fail_common_support)

    with pytest.raises(error_type, match="implementation failure"):
        bound.evaluate(pair, (1, 3), SelectionRule.MAX_UPPER)


@pytest.mark.parametrize("error_type", [TypeError, ValueError, OverflowError])
def test_candidate_implementation_errors_are_not_analytic_failures(
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[Exception],
) -> None:
    pair = _fixture()
    bound = LaggedPearsonAdapter().bind(pair, (1, 3))

    def fail_candidate(*args: object, **kwargs: object) -> float:
        raise error_type("injected candidate implementation failure")

    monkeypatch.setattr(lagged_pearson_module, "_lagged_pearson", fail_candidate)

    with pytest.raises(error_type, match="implementation failure"):
        bound.evaluate(pair, (1, 3), SelectionRule.MAX_UPPER)


def test_floating_point_failure_remains_an_explicit_analytic_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair = _fixture()
    bound = LaggedPearsonAdapter().bind(pair, (1, 3))

    def fail_candidate(*args: object, **kwargs: object) -> float:
        raise FloatingPointError("injected floating-point failure")

    monkeypatch.setattr(lagged_pearson_module, "_lagged_pearson", fail_candidate)

    results = bound.evaluate(pair, (1, 3), SelectionRule.MAX_UPPER)

    assert {result.validity for result in results} == {Validity.ANALYTIC_FAILURE}
    assert {result.estimate for result in results} == {None}
    assert {result.selection_score for result in results} == {None}
    assert {result.diagnostics for result in results} == {
        ("SELCAL_LAGGED_PEARSON_NUMERIC_FAILURE",)
    }


def test_correlation_boundary_clamps_only_tiny_float64_overshoot() -> None:
    epsilon = float(np.finfo(np.float64).eps)

    assert _clamp_correlation(1.0 + 16.0 * epsilon) == 1.0
    assert _clamp_correlation(-1.0 - 16.0 * epsilon) == -1.0
    with pytest.raises(AnalyticFailure, match="bound"):
        _clamp_correlation(1.0 + 64.0 * epsilon)
    with pytest.raises(AnalyticFailure, match="finite"):
        _clamp_correlation(float("nan"))


def test_extreme_finite_scale_is_warning_free_finite_and_bounded() -> None:
    pair = _pair(
        [1e308, -1e308, 5e307, -5e307, 2.5e307, -2.5e307, 7.5e307, -7.5e307],
        [-9e307, 9e307, -4e307, 4e307, -2e307, 2e307, -7e307, 7e307],
    )
    bound = LaggedPearsonAdapter().bind(pair, (3, 1))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        results = bound.evaluate(pair, (1, 3), SelectionRule.MAX_UPPER)

    assert {result.support_n for result in results} == {5}
    assert {result.validity for result in results} == {Validity.VALID}
    assert all(
        result.estimate is not None
        and math.isfinite(result.estimate)
        and -1.0 <= result.estimate <= 1.0
        for result in results
    )


def test_source_mixed_max_and_subnormal_scale_is_valid_under_public_evaluate() -> None:
    maximum = float(np.finfo(np.float64).max)
    subnormal = float(np.finfo(np.float64).smallest_subnormal)
    pair = _pair(
        [maximum, subnormal, -maximum, 0.0],
        [0.0, 1.0, 2.0, 4.0],
    )
    expected = _manual_scaled_pearson(
        [float(value) for value in pair.source[:3]],
        [float(value) for value in pair.target[1:]],
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = LaggedPearsonAdapter().bind(pair, (1,)).evaluate(
            pair, (1,), SelectionRule.MAX_UPPER
        )[0]

    assert result.validity is Validity.VALID
    assert result.estimate is not None
    assert math.isfinite(result.estimate)
    assert -1.0 <= result.estimate <= 1.0
    assert result.estimate == pytest.approx(expected, abs=1e-15)
    assert result.support_n == 3


def test_target_mixed_max_and_subnormal_scale_is_valid_under_public_evaluate() -> None:
    maximum = float(np.finfo(np.float64).max)
    subnormal = float(np.finfo(np.float64).smallest_subnormal)
    pair = _pair(
        [1.0, 2.0, 4.0, 0.0],
        [0.0, maximum, subnormal, -maximum],
    )
    expected = _manual_scaled_pearson(
        [float(value) for value in pair.source[:3]],
        [float(value) for value in pair.target[1:]],
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = LaggedPearsonAdapter().bind(pair, (1,)).evaluate(
            pair, (1,), SelectionRule.MAX_UPPER
        )[0]

    assert result.validity is Validity.VALID
    assert result.estimate is not None
    assert math.isfinite(result.estimate)
    assert -1.0 <= result.estimate <= 1.0
    assert result.estimate == pytest.approx(expected, abs=1e-15)
    assert result.support_n == 3


def test_evaluation_is_deterministic_and_does_not_mutate_pair_bytes() -> None:
    pair = _fixture()
    bound = LaggedPearsonAdapter().bind(pair, (3, 1))
    source_before = pair.source.tobytes()
    target_before = pair.target.tobytes()

    first = bound.evaluate(pair, (1, 3), SelectionRule.MAX_UPPER)
    second = bound.evaluate(pair, (1, 3), SelectionRule.MAX_UPPER)

    assert first == second
    assert pair.source.tobytes() == source_before
    assert pair.target.tobytes() == target_before


def test_backend_and_preprocessing_identities_are_exact_stable_and_nonempty() -> None:
    bound = LaggedPearsonAdapter().bind(_fixture(), (1, 3))

    assert bound.backend_identity == (
        f"selcal.lagged_pearson.v1|numpy={np.__version__}|units=pearson_correlation"
    )
    assert bound.preprocessing_identity == (
        "no_hidden_transform|formula_two_pass_centered_after_power_of_two_scaling"
        "|correctly_rounded_sums|common_support_max_lag"
    )


@pytest.mark.parametrize(
    ("pair", "candidates", "rule", "message"),
    [
        (_pair([0, 1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1, 0]), (1, 3), "max_upper", "length"),
        (_fixture(), [1, 3], "max_upper", "canonical"),
        (_fixture(), _TupleSubclass((1, 3)), "max_upper", "canonical"),
        (_fixture(), (3, 1), "max_upper", "canonical"),
        (_fixture(), (1, 2), "max_upper", "bound candidates"),
        (_fixture(), (np.int64(1), 3), "max_upper", "canonical"),
        (_fixture(), (True, 3), "max_upper", "canonical"),
        (_fixture(), (1, 3), "minimum", "selection rule"),
        (_fixture(), (1, 3), _StringSubclass("max_upper"), "selection rule"),
        (_fixture(), (1, 3), object(), "selection rule"),
    ],
)
def test_evaluate_rejects_pair_candidate_and_rule_drift(
    pair: SeriesPair,
    candidates: object,
    rule: object,
    message: str,
) -> None:
    bound = LaggedPearsonAdapter().bind(_fixture(), (1, 3))

    with pytest.raises(ValueError, match=message):
        bound.evaluate(pair, candidates, rule)  # type: ignore[arg-type]


def test_evaluate_requires_a_series_pair() -> None:
    bound = LaggedPearsonAdapter().bind(_fixture(), (1, 3))

    with pytest.raises(ValueError, match="SeriesPair"):
        bound.evaluate(object(), (1, 3), SelectionRule.MAX_UPPER)  # type: ignore[arg-type]
