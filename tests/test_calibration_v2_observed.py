from __future__ import annotations

import inspect
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import FunctionType

import numpy as np
import pytest

import selcal.calibration_v2 as calibration_v2
from selcal.calibration_v2 import _prepare_calibration, _PreparedCalibration
from selcal.canonical import semantic_input_sha256
from selcal.canonical_v2 import scientific_plan_v2_sha256
from selcal.contracts import PlanRequest, SeriesPair, Validity
from selcal.contracts_v2 import PlanRequestV2, PlanVersionError, V2IntegrityError
from selcal.nulls.circular_shift_v2 import CircularShiftNullV2, _BoundCircularShiftV2
from selcal.resolution import resolve_plan
from selcal.resolution_v2 import PlanResolutionV2, resolve_plan_v2
from selcal.selection_v2 import select_family_v2 as real_select_family_v2
from selcal.statistics.lagged_pearson import (
    LaggedPearsonAdapter,
    _BoundLaggedPearsonAdapter,
)

_SEALED_REPLICATE_EXECUTOR = calibration_v2._execute_replicates


def pair(source: list[float], target: list[float]) -> SeriesPair:
    return SeriesPair(
        source=np.asarray(source, dtype=np.float64),
        target=np.asarray(target, dtype=np.float64),
    )


def valid_pair() -> SeriesPair:
    return pair(
        [0, 1, 4, 2, 5, 3, 7, 6],
        [8, 6, 9, -4, -2, -5, -3, -7],
    )


def resolution(*, min_shift: int = 1) -> PlanResolutionV2:
    return resolve_plan_v2(
        PlanRequestV2(
            candidates=(1, 3),
            statistic_name="lagged_pearson_v1",
            statistic_params={},
            selection_rule="max_upper",
            null_name="circular_shift_v2",
            null_params={"min_shift": min_shift},
            replicates=7,
            alpha=0.05,
            tie_tolerance=1e-12,
            root_seed=17,
        )
    )


@contextmanager
def _trace_capsule_stream_creation(calls: dict[str, int]) -> Iterator[None]:
    closure = inspect.getclosurevars(_SEALED_REPLICATE_EXECUTOR).nonlocals
    create_stream = closure.get("create_random_stream")
    assert type(create_stream) is FunctionType

    def profile(frame: object, event: str, arg: object) -> None:
        del arg
        if event == "call" and frame.f_code is create_stream.__code__:  # type: ignore[attr-defined]
            calls["rng"] += 1

    previous_profile = sys.getprofile()
    sys.setprofile(profile)
    try:
        yield
    finally:
        sys.setprofile(previous_profile)


def _install_call_spies(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, int]:
    calls = {
        "statistic_bind": 0,
        "null_bind": 0,
        "evaluate_all": 0,
        "selection": 0,
        "rng": 0,
        "sample_token": 0,
        "apply_token": 0,
        "replicates": 0,
    }
    statistic_bind = LaggedPearsonAdapter.bind
    null_bind = CircularShiftNullV2.bind
    evaluate_all = _BoundLaggedPearsonAdapter.evaluate_all
    selection = real_select_family_v2
    sample_token = _BoundCircularShiftV2.sample_token
    apply_token = _BoundCircularShiftV2.apply

    def spy_statistic_bind(
        self: LaggedPearsonAdapter,
        observed_pair: SeriesPair,
        candidates: tuple[int, ...],
    ) -> _BoundLaggedPearsonAdapter:
        calls["statistic_bind"] += 1
        return statistic_bind(self, observed_pair, candidates)

    def spy_null_bind(
        self: CircularShiftNullV2,
        observed_pair: SeriesPair,
        *,
        semantic_input_sha256: str,
        scientific_plan_sha256: str,
    ) -> object:
        calls["null_bind"] += 1
        return null_bind(
            self,
            observed_pair,
            semantic_input_sha256=semantic_input_sha256,
            scientific_plan_sha256=scientific_plan_sha256,
        )

    def spy_evaluate_all(
        self: _BoundLaggedPearsonAdapter,
        observed_pair: SeriesPair,
        /,
    ) -> object:
        calls["evaluate_all"] += 1
        return evaluate_all(self, observed_pair)

    def spy_selection(*args: object, **kwargs: object) -> object:
        calls["selection"] += 1
        return selection(*args, **kwargs)  # type: ignore[arg-type]

    def spy_sample_token(self: _BoundCircularShiftV2, random: object, /) -> object:
        calls["sample_token"] += 1
        return sample_token(self, random)  # type: ignore[arg-type]

    def spy_apply_token(self: _BoundCircularShiftV2, token: object, /) -> object:
        calls["apply_token"] += 1
        return apply_token(self, token)  # type: ignore[arg-type]

    def bomb_replicates(*args: object, **kwargs: object) -> object:
        del args, kwargs
        calls["replicates"] += 1
        raise AssertionError("Task 8A must not execute replicates")

    monkeypatch.setattr(LaggedPearsonAdapter, "bind", spy_statistic_bind)
    monkeypatch.setattr(CircularShiftNullV2, "bind", spy_null_bind)
    monkeypatch.setattr(_BoundLaggedPearsonAdapter, "evaluate_all", spy_evaluate_all)
    monkeypatch.setattr(calibration_v2, "select_family_v2", spy_selection)
    monkeypatch.setattr(_BoundCircularShiftV2, "sample_token", spy_sample_token)
    monkeypatch.setattr(_BoundCircularShiftV2, "apply", spy_apply_token)
    monkeypatch.setattr(
        calibration_v2,
        "_execute_replicates",
        bomb_replicates,
        raising=False,
    )
    return calls


@pytest.mark.parametrize("foreign", [object(), None, 3])
def test_prepare_requires_an_exact_v2_resolution(foreign: object) -> None:
    with pytest.raises(PlanVersionError, match="exact PlanResolutionV2"):
        _prepare_calibration(valid_pair(), foreign)  # type: ignore[arg-type]


def test_prepare_rejects_a_v1_resolution_before_binding() -> None:
    v1 = resolve_plan(
        PlanRequest(
            candidates=(1, 3),
            statistic_name="lagged_pearson_v1",
            statistic_params={},
            selection_rule="max_upper",
            null_name="circular_shift_v1",
            null_params={"min_shift": 1},
            replicates=7,
            alpha=0.05,
            tie_tolerance=1e-12,
            root_seed=17,
            failure_policy="fail_closed_v1",
        )
    )
    with pytest.raises(PlanVersionError, match="exact PlanResolutionV2"):
        _prepare_calibration(valid_pair(), v1)  # type: ignore[arg-type]


def test_prepare_rehashes_the_resolver_owned_plan_before_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact = resolution()
    bind_calls = 0
    original_bind = LaggedPearsonAdapter.bind

    def spy_bind(
        self: LaggedPearsonAdapter,
        observed_pair: SeriesPair,
        candidates: tuple[int, ...],
    ) -> object:
        nonlocal bind_calls
        bind_calls += 1
        return original_bind(self, observed_pair, candidates)

    monkeypatch.setattr(LaggedPearsonAdapter, "bind", spy_bind)
    object.__setattr__(exact.plan, "root_seed", 18)

    with pytest.raises(V2IntegrityError, match="drifted"):
        _prepare_calibration(valid_pair(), exact)
    assert bind_calls == 0


def test_disabled_null_binds_both_adapters_once_and_terminates_without_scans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_call_spies(monkeypatch)
    observed = pair([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
    exact = resolution(min_shift=3)

    with _trace_capsule_stream_creation(calls):
        result = _prepare_calibration(observed, exact)

    assert result.status.value == "not_evaluable"
    assert result.failure_stage.value == "null_bind"
    assert result.semantic_input_sha256 == semantic_input_sha256(observed, exact.plan.candidates)
    assert result.scientific_plan_sha256 == scientific_plan_v2_sha256(exact.plan)
    assert result.planned_replicates == exact.plan.replicates
    assert result.alpha == exact.plan.alpha
    assert result.observed_results == ()
    assert result.observed_selection is None
    assert result.replicates == ()
    assert result.exceedance_count == result.failure_count == 0
    assert result.p_value is None
    assert result.exceedance_bound_low is None
    assert result.exceedance_bound_high is None
    assert result.reject_null is None
    assert result.diagnostics == ("shift_space_empty_v2",)
    assert calls == {
        "statistic_bind": 1,
        "null_bind": 1,
        "evaluate_all": 0,
        "selection": 0,
        "rng": 0,
        "sample_token": 0,
        "apply_token": 0,
        "replicates": 0,
    }


def test_enabled_observed_path_scans_and_selects_once_without_mutating_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_call_spies(monkeypatch)
    observed = valid_pair()
    source_before = observed.source.tobytes(order="C")
    target_before = observed.target.tobytes(order="C")
    exact = resolution()

    with _trace_capsule_stream_creation(calls):
        prepared = _prepare_calibration(observed, exact)

    assert type(prepared) is _PreparedCalibration
    assert prepared.resolution is exact
    assert prepared.observed_pair == observed
    assert prepared.observed_pair is not observed
    assert prepared.semantic_input_sha256 == semantic_input_sha256(observed, exact.plan.candidates)
    assert prepared.scientific_plan_sha256 == scientific_plan_v2_sha256(exact.plan)
    assert tuple(result.candidate_id for result in prepared.observed_results) == (1, 3)
    assert all(result.validity is Validity.VALID for result in prepared.observed_results)
    assert all(result.selection_score is not None for result in prepared.observed_results)
    assert prepared.observed_selection.decision_statistic == max(
        result.selection_score
        for result in prepared.observed_results
        if result.selection_score is not None
    )
    assert prepared.bound_statistic.candidates == exact.plan.candidates
    assert prepared.bound_null.observed_length == observed.source.size
    assert prepared.statistic_snapshot.backend_identity == (
        prepared.bound_statistic.backend_identity
    )
    assert prepared.null_snapshot.null_parameter_sha256 == (
        prepared.bound_null.null_parameter_sha256  # type: ignore[attr-defined]
    )
    assert prepared.diagnostics == ()
    assert observed.source.tobytes(order="C") == source_before
    assert observed.target.tobytes(order="C") == target_before
    assert not observed.source.flags.writeable
    assert not observed.target.flags.writeable
    assert calls == {
        "statistic_bind": 1,
        "null_bind": 1,
        "evaluate_all": 1,
        "selection": 1,
        "rng": 0,
        "sample_token": 0,
        "apply_token": 0,
        "replicates": 0,
    }


def test_prepare_rejects_non_series_pair_before_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bind_calls = 0
    original_bind: Callable[..., object] = LaggedPearsonAdapter.bind

    def spy_bind(*args: object, **kwargs: object) -> object:
        nonlocal bind_calls
        bind_calls += 1
        return original_bind(*args, **kwargs)

    monkeypatch.setattr(LaggedPearsonAdapter, "bind", spy_bind)
    with pytest.raises(V2IntegrityError, match="exact SeriesPair"):
        _prepare_calibration(object(), resolution())  # type: ignore[arg-type]
    assert bind_calls == 0


def test_observed_preparation_supports_registered_binned_nette_and_block_shuffle() -> None:
    observed = pair(
        [0, 1, 0, 1, 2, 1, 2, 0, 2, 1],
        [1, 0, 1, 2, 1, 0, 2, 1, 2, 0],
    )
    exact = resolve_plan_v2(
        PlanRequestV2(
            candidates=(1, 2),
            statistic_name="equal_width_binned_nette_v1",
            statistic_params={"bins": 3},
            selection_rule="max_absolute",
            null_name="block_shuffle_v2",
            null_params={"block_length": 2},
            replicates=7,
            alpha=0.05,
            tie_tolerance=1e-12,
            root_seed=17,
        )
    )

    prepared = _prepare_calibration(observed, exact)

    assert type(prepared) is _PreparedCalibration
    assert prepared.bound_statistic.name == "equal_width_binned_nette_v1"
    assert prepared.bound_null.name == "block_shuffle_v2"
    assert tuple(result.candidate_id for result in prepared.observed_results) == (1, 2)
    assert all(result.selection_score is not None for result in prepared.observed_results)
