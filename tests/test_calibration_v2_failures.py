from __future__ import annotations

import gc
import inspect
import sys
from collections.abc import Iterator, Mapping
from copy import copy
from dataclasses import fields, replace
from types import MappingProxyType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

import selcal.calibration_v2 as calibration_v2
import selcal.nulls.block_shuffle_v2 as block_shuffle_v2
import selcal.resolution_v2 as resolution_v2
from selcal.calibration_v2 import (
    _prepare_calibration,
    _reverify_active_preparation,
    _reverify_run_identity,
    _snapshot_bound_null,
    _snapshot_bound_statistic,
    _validate_null_bind_snapshot,
    _validate_raw_observed_vector,
)
from selcal.canonical import semantic_input_sha256
from selcal.canonical_v2 import scientific_plan_v2_sha256
from selcal.contracts import RunStatus, SeriesPair, StatisticResult, Validity
from selcal.contracts_v2 import (
    CalibrationResult,
    NullBindStatus,
    NullDisabledReason,
    PlanRequestV2,
    RunFailureStage,
    V2IntegrityError,
)
from selcal.nulls.block_shuffle_v2 import _BoundBlockShuffleV2
from selcal.nulls.circular_shift_v2 import CircularShiftNullV2, _BoundCircularShiftV2
from selcal.nulls.executable_base import NullBindResult
from selcal.nulls.owned_transform_v2 import (
    _owned_pair_content_guard_sha256,
    owner_digest,
)
from selcal.resolution_v2 import resolve_plan_v2
from selcal.selection_v2 import select_family_v2 as real_select_family_v2
from selcal.statistics.binned_nette import BinnedNetTEAdapter, _BoundBinnedNetTEAdapter
from selcal.statistics.lagged_pearson import (
    LaggedPearsonAdapter,
    _BoundLaggedPearsonAdapter,
)


def pair(source: list[float], target: list[float]) -> SeriesPair:
    return SeriesPair(
        source=np.asarray(source, dtype=np.float64),
        target=np.asarray(target, dtype=np.float64),
    )


def resolution():
    return resolve_plan_v2(
        PlanRequestV2(
            candidates=(1, 2),
            statistic_name="lagged_pearson_v1",
            statistic_params={},
            selection_rule="max_upper",
            null_name="circular_shift_v2",
            null_params={"min_shift": 1},
            replicates=5,
            alpha=0.05,
            tie_tolerance=1e-12,
            root_seed=17,
        )
    )


def _ownership_records() -> dict[int, Any]:
    require = resolution_v2._require_resolver_owned_resolution_v2
    ownership_lookup = inspect.getclosurevars(require).nonlocals["ownership_lookup"]
    records_get = inspect.getclosurevars(ownership_lookup).nonlocals["records_get"]
    records = getattr(records_get, "__self__", None)
    assert type(records) is dict
    return records


def valid_observed() -> SeriesPair:
    return pair([0, 1, 4, 2, 5, 3], [4, 1, 3, 0, 5, 2])


def binned_observed() -> SeriesPair:
    return pair(
        [0, 1, 0, 1, 2, 1, 2, 0, 2, 1],
        [1, 0, 1, 2, 1, 0, 2, 1, 2, 0],
    )


def binned_resolution():
    return resolve_plan_v2(
        PlanRequestV2(
            candidates=(1, 2),
            statistic_name="equal_width_binned_nette_v1",
            statistic_params={"bins": 3},
            selection_rule="max_absolute",
            null_name="block_shuffle_v2",
            null_params={"block_length": 2},
            replicates=5,
            alpha=0.05,
            tie_tolerance=1e-12,
            root_seed=17,
        )
    )


def resolution_with_null(null_name: str, null_params: dict[str, int]):
    return resolve_plan_v2(
        PlanRequestV2(
            candidates=(1, 2),
            statistic_name="lagged_pearson_v1",
            statistic_params={},
            selection_rule="max_upper",
            null_name=null_name,
            null_params=null_params,
            replicates=5,
            alpha=0.05,
            tie_tolerance=1e-12,
            root_seed=17,
        )
    )


def statistic_snapshot_fields() -> dict[str, object]:
    bound = LaggedPearsonAdapter().bind(valid_observed(), (1, 2))
    return {
        "name": bound.name,
        "parameters": bound.parameters,
        "candidates": bound.candidates,
        "backend_identity": bound.backend_identity,
        "preprocessing_identity": bound.preprocessing_identity,
        "evaluate_all": bound.evaluate_all,
    }


def null_snapshot_context() -> tuple[dict[str, object], object, SeriesPair, str, str]:
    exact = resolution()
    observed = valid_observed()
    semantic_digest = semantic_input_sha256(observed, exact.plan.candidates)
    plan_digest = scientific_plan_v2_sha256(exact.plan)
    bind = exact.adapters.null_model.bind(
        observed,
        semantic_input_sha256=semantic_digest,
        scientific_plan_sha256=plan_digest,
    )
    assert bind.bound is not None
    bound = bind.bound
    fields = {
        "name": bound.name,
        "parameters": bound.parameters,
        "observed_length": bound.observed_length,
        "total_state_count": bound.total_state_count,
        "null_parameter_sha256": bound.null_parameter_sha256,  # type: ignore[attr-defined]
        "bound_null_owner_sha256": bound.bound_null_owner_sha256,  # type: ignore[attr-defined]
        "_semantic_input_sha256": bound._semantic_input_sha256,  # type: ignore[attr-defined]
        "_scientific_plan_sha256": bound._scientific_plan_sha256,  # type: ignore[attr-defined]
    }
    return fields, exact, observed, semantic_digest, plan_digest


def test_observed_analytical_failure_retains_complete_raw_vector_and_stops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = pair([1, 1, 1, 1, 1, 1], [0, 1, 3, 2, 5, 4])
    exact = resolution()
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
    statistic_bind = type(exact.adapters.statistic).bind
    null_bind = type(exact.adapters.null_model).bind
    evaluate_all = _BoundLaggedPearsonAdapter.evaluate_all
    sample_token = _BoundCircularShiftV2.sample_token
    apply_token = _BoundCircularShiftV2.apply

    def spy_statistic_bind(self: object, *args: object, **kwargs: object) -> object:
        calls["statistic_bind"] += 1
        return statistic_bind(self, *args, **kwargs)  # type: ignore[arg-type]

    def spy_null_bind(self: object, *args: object, **kwargs: object) -> object:
        calls["null_bind"] += 1
        return null_bind(self, *args, **kwargs)  # type: ignore[arg-type]

    def spy_evaluate_all(self: object, supplied: SeriesPair, /) -> object:
        calls["evaluate_all"] += 1
        return evaluate_all(self, supplied)  # type: ignore[arg-type]

    def bomb_selection(*args: object, **kwargs: object) -> object:
        del args, kwargs
        calls["selection"] += 1
        raise AssertionError("failed observed vectors must not be selected")

    def spy_sample_token(self: object, random: object, /) -> object:
        calls["sample_token"] += 1
        return sample_token(self, random)  # type: ignore[arg-type]

    def spy_apply_token(self: object, token: object, /) -> object:
        calls["apply_token"] += 1
        return apply_token(self, token)  # type: ignore[arg-type]

    def bomb_replicates(*args: object, **kwargs: object) -> object:
        del args, kwargs
        calls["replicates"] += 1
        raise AssertionError("observed failure must not execute replicates")

    monkeypatch.setattr(type(exact.adapters.statistic), "bind", spy_statistic_bind)
    monkeypatch.setattr(type(exact.adapters.null_model), "bind", spy_null_bind)
    monkeypatch.setattr(_BoundLaggedPearsonAdapter, "evaluate_all", spy_evaluate_all)
    monkeypatch.setattr(calibration_v2, "select_family_v2", bomb_selection)
    monkeypatch.setattr(_BoundCircularShiftV2, "sample_token", spy_sample_token)
    monkeypatch.setattr(_BoundCircularShiftV2, "apply", spy_apply_token)
    executor_closure = inspect.getclosurevars(calibration_v2._execute_replicates).nonlocals
    create_stream = executor_closure.get("create_random_stream")
    assert inspect.isfunction(create_stream)

    def profile(frame: object, event: str, arg: object) -> None:
        del arg
        if event == "call" and frame.f_code is create_stream.__code__:  # type: ignore[attr-defined]
            calls["rng"] += 1

    monkeypatch.setattr(
        calibration_v2,
        "_execute_replicates",
        bomb_replicates,
        raising=False,
    )

    previous_profile = sys.getprofile()
    sys.setprofile(profile)
    try:
        result = _prepare_calibration(observed, exact)
    finally:
        sys.setprofile(previous_profile)

    assert type(result) is CalibrationResult
    assert result.status is RunStatus.NOT_EVALUABLE
    assert result.failure_stage is RunFailureStage.OBSERVED_STATISTIC_SCAN
    assert tuple(item.candidate_id for item in result.observed_results) == (1, 2)
    assert len(result.observed_results) == len(exact.plan.candidates)
    assert any(item.validity is Validity.ANALYTIC_FAILURE for item in result.observed_results)
    assert all(item.selection_score is None for item in result.observed_results)
    assert result.observed_selection is None
    assert result.replicates == ()
    assert result.exceedance_count == result.failure_count == 0
    assert result.p_value is None
    assert result.exceedance_bound_low is None
    assert result.exceedance_bound_high is None
    assert result.reject_null is None
    assert result.diagnostics == ("observed_statistic_scan_analytical_failure_v2",)
    assert calls == {
        "statistic_bind": 1,
        "null_bind": 1,
        "evaluate_all": 1,
        "selection": 0,
        "rng": 0,
        "sample_token": 0,
        "apply_token": 0,
        "replicates": 0,
    }


def test_generic_observed_runtime_error_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def explode(self: object, supplied: SeriesPair, /) -> object:
        del self, supplied
        raise RuntimeError("infrastructure exploded")

    monkeypatch.setattr(_BoundLaggedPearsonAdapter, "evaluate_all", explode)
    with pytest.raises(RuntimeError, match="infrastructure exploded"):
        _prepare_calibration(
            pair([0, 1, 4, 2, 5, 3], [4, 1, 3, 0, 5, 2]),
            resolution(),
        )


def test_observed_malformed_candidate_vector_is_an_integrity_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = _BoundLaggedPearsonAdapter.evaluate_all

    def malformed(
        self: _BoundLaggedPearsonAdapter,
        supplied: SeriesPair,
        /,
    ) -> object:
        return original(self, supplied)[:-1]

    monkeypatch.setattr(_BoundLaggedPearsonAdapter, "evaluate_all", malformed)
    with pytest.raises(Exception, match="candidate vector") as caught:
        _prepare_calibration(
            pair([0, 1, 4, 2, 5, 3], [4, 1, 3, 0, 5, 2]),
            resolution(),
        )
    assert type(caught.value).__name__ == "V2IntegrityError"


class _RaisingIdentity:
    @property
    def name(self) -> str:
        raise ValueError("bad descriptor")


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"name": "wrong_statistic_v2"}, "name"),
        ({"parameters": 1}, "parameters must be a mapping"),
        ({"parameters": {"bad": object()}}, "parameters are invalid"),
        ({"parameters": {"bad": 1}}, "parameters do not match"),
        ({"candidates": [1, 2]}, "candidates"),
        ({"candidates": (1, 3)}, "candidates"),
        ({"backend_identity": 1}, "backend_identity"),
        ({"backend_identity": ""}, "backend_identity"),
        ({"preprocessing_identity": 1}, "preprocessing_identity"),
        ({"preprocessing_identity": ""}, "preprocessing_identity"),
        ({"evaluate_all": None}, "evaluate_all"),
    ],
)
def test_bound_statistic_snapshot_fails_closed(
    changes: dict[str, object],
    match: str,
) -> None:
    exact = resolution()
    fields = statistic_snapshot_fields()
    fields.update(changes)
    with pytest.raises(V2IntegrityError, match=match):
        _snapshot_bound_statistic(
            SimpleNamespace(**fields),
            exact,
            observed_pair=valid_observed(),
        )


@pytest.mark.parametrize("malformed", [object(), _RaisingIdentity()])
def test_bound_statistic_snapshot_normalizes_structural_errors(
    malformed: object,
) -> None:
    with pytest.raises(V2IntegrityError, match="identity snapshot"):
        _snapshot_bound_statistic(
            malformed,
            resolution(),
            observed_pair=valid_observed(),
        )


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"name": "wrong_null_v2"}, "name"),
        ({"parameters": 1}, "parameters must be a mapping"),
        ({"parameters": {"bad": object()}}, "parameters are invalid"),
        ({"parameters": {"min_shift": 2}}, "parameters do not match"),
        ({"observed_length": "6"}, "observed length"),
        ({"observed_length": 7}, "observed length"),
        ({"total_state_count": "6"}, "total state count"),
        ({"total_state_count": 0}, "total state count"),
        ({"null_parameter_sha256": "a" * 64}, "parameter digest"),
        ({"_semantic_input_sha256": "b" * 64}, "input or plan identity"),
        ({"_scientific_plan_sha256": "b" * 64}, "input or plan identity"),
        ({"bound_null_owner_sha256": "c" * 64}, "owner digest"),
    ],
)
def test_bound_null_snapshot_fails_closed(
    changes: dict[str, object],
    match: str,
) -> None:
    fields, exact, observed, semantic_digest, plan_digest = null_snapshot_context()
    fields.update(changes)
    with pytest.raises(V2IntegrityError, match=match):
        _snapshot_bound_null(
            SimpleNamespace(**fields),
            exact,  # type: ignore[arg-type]
            observed_length=int(observed.source.size),
            semantic_digest=semantic_digest,
            plan_digest=plan_digest,
            observed_pair=observed,
        )


@pytest.mark.parametrize("malformed", [object(), _RaisingIdentity()])
def test_bound_null_snapshot_normalizes_structural_errors(malformed: object) -> None:
    _, exact, observed, semantic_digest, plan_digest = null_snapshot_context()
    with pytest.raises(V2IntegrityError, match="identity snapshot"):
        _snapshot_bound_null(
            malformed,
            exact,  # type: ignore[arg-type]
            observed_length=int(observed.source.size),
            semantic_digest=semantic_digest,
            plan_digest=plan_digest,
            observed_pair=observed,
        )


def test_observed_vector_rejects_non_tuple_and_non_result_entries() -> None:
    exact = resolution()
    bound = LaggedPearsonAdapter().bind(valid_observed(), exact.plan.candidates)
    _, snapshot = _snapshot_bound_statistic(
        bound,
        exact,
        observed_pair=valid_observed(),
    )
    for malformed in ([], (object(),)):
        with pytest.raises(V2IntegrityError, match="exact result tuple"):
            _validate_raw_observed_vector(malformed, snapshot, exact)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda results: (replace(results[0], selection_score=1.0), results[1]), "unscored"),
        (
            lambda results: (replace(results[0], backend_identity="foreign"), results[1]),
            "implementation identity",
        ),
        (
            lambda results: (
                replace(results[0], preprocessing_identity="foreign"),
                results[1],
            ),
            "implementation identity",
        ),
    ],
)
def test_observed_vector_rejects_scoring_and_implementation_drift(
    mutation: object,
    match: str,
) -> None:
    exact = resolution()
    observed = valid_observed()
    bound = LaggedPearsonAdapter().bind(observed, exact.plan.candidates)
    _, snapshot = _snapshot_bound_statistic(bound, exact, observed_pair=observed)
    raw = bound.evaluate_all(observed)
    malformed = mutation(raw)  # type: ignore[operator]
    with pytest.raises(V2IntegrityError, match=match):
        _validate_raw_observed_vector(malformed, snapshot, exact)


def test_observed_vector_rejects_mutated_valid_estimate_and_validity() -> None:
    exact = resolution()
    observed = valid_observed()
    bound = LaggedPearsonAdapter().bind(observed, exact.plan.candidates)
    _, snapshot = _snapshot_bound_statistic(bound, exact, observed_pair=observed)
    for field, value, match in (
        ("estimate", float("nan"), "finite estimate"),
        ("validity", "foreign", "validity is unsupported"),
    ):
        raw = bound.evaluate_all(observed)
        object.__setattr__(raw[0], field, value)
        with pytest.raises(V2IntegrityError, match=match):
            _validate_raw_observed_vector(raw, snapshot, exact)


def test_prepare_rejects_non_exact_null_bind_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact = resolution()
    monkeypatch.setattr(type(exact.adapters.null_model), "bind", lambda *args, **kwargs: object())
    with pytest.raises(V2IntegrityError, match="exact NullBindResult"):
        _prepare_calibration(valid_observed(), exact)


def test_prepare_rejects_enabled_null_bind_without_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact = resolution()
    observed = valid_observed()
    semantic_digest = semantic_input_sha256(observed, exact.plan.candidates)
    plan_digest = scientific_plan_v2_sha256(exact.plan)
    malformed = exact.adapters.null_model.bind(
        observed,
        semantic_input_sha256=semantic_digest,
        scientific_plan_sha256=plan_digest,
    )
    assert type(malformed) is NullBindResult
    assert malformed.status is NullBindStatus.ENABLED
    object.__setattr__(malformed, "bound", None)
    monkeypatch.setattr(
        type(exact.adapters.null_model),
        "bind",
        lambda *args, **kwargs: malformed,
    )
    with pytest.raises(V2IntegrityError, match="retain a bound null"):
        _prepare_calibration(observed, exact)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("status", "enabled", "status"),
        ("disabled_reason", NullDisabledReason.SHIFT_SPACE_EMPTY, "disabled_reason"),
    ],
)
def test_prepare_revalidates_enabled_null_bind_fields(
    field: str,
    value: object,
    match: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact = resolution()
    observed = valid_observed()
    malformed = exact.adapters.null_model.bind(
        observed,
        semantic_input_sha256=semantic_input_sha256(observed, exact.plan.candidates),
        scientific_plan_sha256=scientific_plan_v2_sha256(exact.plan),
    )
    object.__setattr__(malformed, field, value)
    monkeypatch.setattr(
        type(exact.adapters.null_model),
        "bind",
        lambda *args, **kwargs: malformed,
    )
    with pytest.raises(V2IntegrityError, match=match):
        _prepare_calibration(observed, exact)


@pytest.mark.parametrize("field", ["bound", "disabled_reason"])
def test_prepare_revalidates_disabled_null_bind_fields(
    field: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact = resolve_plan_v2(
        PlanRequestV2(
            candidates=(1, 3),
            statistic_name="lagged_pearson_v1",
            statistic_params={},
            selection_rule="max_upper",
            null_name="circular_shift_v2",
            null_params={"min_shift": 3},
            replicates=5,
            alpha=0.05,
            tie_tolerance=1e-12,
            root_seed=17,
        )
    )
    observed = pair([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
    disabled = exact.adapters.null_model.bind(
        observed,
        semantic_input_sha256=semantic_input_sha256(observed, exact.plan.candidates),
        scientific_plan_sha256=scientific_plan_v2_sha256(exact.plan),
    )
    assert disabled.status is NullBindStatus.DISABLED
    if field == "bound":
        object.__setattr__(disabled, field, object())
    else:
        object.__setattr__(disabled, field, None)
    monkeypatch.setattr(
        type(exact.adapters.null_model),
        "bind",
        lambda *args, **kwargs: disabled,
    )
    with pytest.raises(V2IntegrityError, match=field):
        _prepare_calibration(observed, exact)


def test_prepare_rechecks_plan_identity_after_null_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact = resolution()
    original_bind = type(exact.adapters.null_model).bind

    def drifting_bind(self: object, *args: object, **kwargs: object) -> object:
        result = original_bind(self, *args, **kwargs)  # type: ignore[arg-type]
        object.__setattr__(exact.plan, "root_seed", 99)
        return result

    monkeypatch.setattr(type(exact.adapters.null_model), "bind", drifting_bind)
    with pytest.raises(V2IntegrityError, match="drifted"):
        _prepare_calibration(valid_observed(), exact)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("candidate_id", True, "candidate vector"),
        ("estimate", 1.0, "analytical failure.*estimate"),
    ],
)
def test_observed_failure_vector_rejects_mutated_internal_fields(
    field: str,
    value: object,
    match: str,
) -> None:
    exact = resolution()
    observed = pair([1, 1, 1, 1, 1, 1], [0, 1, 3, 2, 5, 4])
    bound = LaggedPearsonAdapter().bind(observed, exact.plan.candidates)
    _, snapshot = _snapshot_bound_statistic(bound, exact, observed_pair=observed)
    raw = bound.evaluate_all(observed)
    assert raw[0].validity is Validity.ANALYTIC_FAILURE
    object.__setattr__(raw[0], field, value)
    with pytest.raises(V2IntegrityError, match=match):
        _validate_raw_observed_vector(raw, snapshot, exact)


@pytest.mark.parametrize("diagnostics", [[], (1,)])
def test_null_bind_snapshot_requires_an_exact_string_diagnostic_tuple(
    diagnostics: object,
) -> None:
    bind = NullBindResult(
        status=NullBindStatus.DISABLED,
        bound=None,
        disabled_reason=NullDisabledReason.SHIFT_SPACE_EMPTY,
        diagnostics=(),
    )
    object.__setattr__(bind, "diagnostics", diagnostics)
    with pytest.raises(V2IntegrityError, match="diagnostics"):
        _validate_null_bind_snapshot(bind)


def test_run_identity_reverification_checks_supplied_hash_anchors() -> None:
    exact = resolution()
    observed = valid_observed()
    semantic_digest = semantic_input_sha256(observed, exact.plan.candidates)
    plan_digest = scientific_plan_v2_sha256(exact.plan)
    with pytest.raises(V2IntegrityError, match="plan identity"):
        _reverify_run_identity(
            observed,
            exact,
            semantic_digest=semantic_digest,
            plan_digest="a" * 64,
        )
    with pytest.raises(V2IntegrityError, match="input identity"):
        _reverify_run_identity(
            observed,
            exact,
            semantic_digest="b" * 64,
            plan_digest=plan_digest,
        )


def test_statistic_backend_drift_during_scan_is_rejected_against_initial_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_evaluate = _BoundLaggedPearsonAdapter.evaluate_all

    def drifting_evaluate(
        self: _BoundLaggedPearsonAdapter,
        supplied: SeriesPair,
        /,
    ) -> object:
        monkeypatch.setattr(
            _BoundLaggedPearsonAdapter,
            "backend_identity",
            property(lambda instance: "drifted-backend-v2"),
        )
        return original_evaluate(self, supplied)

    monkeypatch.setattr(_BoundLaggedPearsonAdapter, "evaluate_all", drifting_evaluate)
    with pytest.raises(V2IntegrityError, match=r"statistic.*drift"):
        _prepare_calibration(valid_observed(), resolution())


def test_null_parameter_digest_drift_during_observed_scan_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact = resolution()
    original_bind = type(exact.adapters.null_model).bind
    original_evaluate = _BoundLaggedPearsonAdapter.evaluate_all
    captured_bound: list[object] = []

    def capture_bind(self: object, *args: object, **kwargs: object) -> object:
        result = original_bind(self, *args, **kwargs)  # type: ignore[arg-type]
        assert result.bound is not None
        captured_bound.append(result.bound)
        return result

    def drifting_evaluate(
        self: _BoundLaggedPearsonAdapter,
        supplied: SeriesPair,
        /,
    ) -> object:
        object.__setattr__(captured_bound[0], "_null_parameter_sha256", "a" * 64)
        return original_evaluate(self, supplied)

    monkeypatch.setattr(type(exact.adapters.null_model), "bind", capture_bind)
    monkeypatch.setattr(_BoundLaggedPearsonAdapter, "evaluate_all", drifting_evaluate)
    with pytest.raises(V2IntegrityError, match=r"null.*drift|parameter digest"):
        _prepare_calibration(valid_observed(), exact)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("support_n", True, "support_n"),
        ("diagnostics", (object(),), "diagnostics"),
    ],
)
def test_observed_failure_revalidates_support_and_diagnostics(
    field: str,
    value: object,
    match: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_evaluate = _BoundLaggedPearsonAdapter.evaluate_all

    def malformed_failure(
        self: _BoundLaggedPearsonAdapter,
        supplied: SeriesPair,
        /,
    ) -> object:
        results = original_evaluate(self, supplied)
        object.__setattr__(results[0], field, value)
        return results

    monkeypatch.setattr(_BoundLaggedPearsonAdapter, "evaluate_all", malformed_failure)
    observed = pair([1, 1, 1, 1, 1, 1], [0, 1, 3, 2, 5, 4])
    with pytest.raises(V2IntegrityError, match=match):
        _prepare_calibration(observed, resolution())


def test_damaged_exact_series_pair_is_normalized_to_integrity_error() -> None:
    observed = valid_observed()
    object.__setattr__(observed, "source", object())
    with pytest.raises(V2IntegrityError, match="input snapshot"):
        _prepare_calibration(observed, resolution())


def test_second_statistic_identity_value_error_is_normalized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = valid_observed()
    bound = LaggedPearsonAdapter().bind(observed, (1, 2))
    raw = bound.evaluate_all(observed)
    getter_calls = 0

    def stateful_backend(self: object) -> str:
        del self
        nonlocal getter_calls
        getter_calls += 1
        if getter_calls == 1:
            return raw[0].backend_identity
        raise ValueError("secondary identity read failed")

    monkeypatch.setattr(
        _BoundLaggedPearsonAdapter,
        "backend_identity",
        property(stateful_backend),
    )
    monkeypatch.setattr(
        _BoundLaggedPearsonAdapter,
        "evaluate_all",
        lambda self, supplied: raw,
    )
    with pytest.raises(V2IntegrityError, match="identity snapshot"):
        _prepare_calibration(observed, resolution())


@pytest.mark.parametrize("error", [RuntimeError("runtime"), OSError("io"), AssertionError("bug")])
def test_second_statistic_identity_nonstructural_errors_propagate(
    error: BaseException,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = valid_observed()
    bound = LaggedPearsonAdapter().bind(observed, (1, 2))
    raw = bound.evaluate_all(observed)
    getter_calls = 0

    def stateful_backend(self: object) -> str:
        del self
        nonlocal getter_calls
        getter_calls += 1
        if getter_calls == 1:
            return raw[0].backend_identity
        raise error

    monkeypatch.setattr(
        _BoundLaggedPearsonAdapter,
        "backend_identity",
        property(stateful_backend),
    )
    monkeypatch.setattr(
        _BoundLaggedPearsonAdapter,
        "evaluate_all",
        lambda self, supplied: raw,
    )
    with pytest.raises(type(error), match=str(error)):
        _prepare_calibration(observed, resolution())


def test_disabled_terminal_reverifies_bound_statistic_before_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact = resolve_plan_v2(
        PlanRequestV2(
            candidates=(1, 3),
            statistic_name="lagged_pearson_v1",
            statistic_params={},
            selection_rule="max_upper",
            null_name="circular_shift_v2",
            null_params={"min_shift": 3},
            replicates=5,
            alpha=0.05,
            tie_tolerance=1e-12,
            root_seed=17,
        )
    )
    original_null_bind = type(exact.adapters.null_model).bind

    def drifting_null_bind(self: object, *args: object, **kwargs: object) -> object:
        result = original_null_bind(self, *args, **kwargs)  # type: ignore[arg-type]
        monkeypatch.setattr(
            _BoundLaggedPearsonAdapter,
            "backend_identity",
            property(lambda instance: "drifted-before-terminal-v2"),
        )
        return result

    monkeypatch.setattr(type(exact.adapters.null_model), "bind", drifting_null_bind)
    observed = pair([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
    with pytest.raises(V2IntegrityError, match=r"statistic.*drift"):
        _prepare_calibration(observed, exact)


def test_selection_completion_reverifies_bound_statistic_before_prepared_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_select = calibration_v2.select_family_v2

    def drifting_selection(*args: object, **kwargs: object) -> object:
        result = original_select(*args, **kwargs)  # type: ignore[arg-type]
        monkeypatch.setattr(
            _BoundLaggedPearsonAdapter,
            "preprocessing_identity",
            property(lambda instance: "drifted-after-selection-v2"),
        )
        return result

    monkeypatch.setattr(calibration_v2, "select_family_v2", drifting_selection)
    with pytest.raises(V2IntegrityError, match=r"statistic.*drift"):
        _prepare_calibration(valid_observed(), resolution())


def test_null_live_snapshot_drift_is_compared_to_initial_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact = resolution()
    original_bind = type(exact.adapters.null_model).bind
    original_evaluate = _BoundLaggedPearsonAdapter.evaluate_all
    captured_bound: list[object] = []

    def capture_bind(self: object, *args: object, **kwargs: object) -> object:
        result = original_bind(self, *args, **kwargs)  # type: ignore[arg-type]
        assert result.bound is not None
        captured_bound.append(result.bound)
        return result

    def drifting_evaluate(
        self: _BoundLaggedPearsonAdapter,
        supplied: SeriesPair,
        /,
    ) -> object:
        state_count = captured_bound[0].total_state_count  # type: ignore[attr-defined]
        object.__setattr__(captured_bound[0], "_total_state_count", state_count + 1)
        return original_evaluate(self, supplied)

    monkeypatch.setattr(type(exact.adapters.null_model), "bind", capture_bind)
    monkeypatch.setattr(_BoundLaggedPearsonAdapter, "evaluate_all", drifting_evaluate)
    with pytest.raises(V2IntegrityError, match=r"null.*drift"):
        _prepare_calibration(valid_observed(), exact)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("backend_identity", 1),
        ("backend_identity", ""),
        ("preprocessing_identity", 1),
        ("preprocessing_identity", ""),
    ],
)
def test_observed_result_identities_must_be_exact_nonempty_strings(
    field: str,
    value: object,
) -> None:
    exact = resolution()
    observed = valid_observed()
    bound = LaggedPearsonAdapter().bind(observed, exact.plan.candidates)
    _, snapshot = _snapshot_bound_statistic(bound, exact, observed_pair=observed)
    raw = bound.evaluate_all(observed)
    object.__setattr__(raw[0], field, value)
    with pytest.raises(V2IntegrityError, match="identity is invalid"):
        _validate_raw_observed_vector(raw, snapshot, exact)


def test_run_identity_normalizes_structural_semantic_rehash_failure() -> None:
    exact = resolution()
    observed = valid_observed()
    semantic_digest = semantic_input_sha256(observed, exact.plan.candidates)
    plan_digest = scientific_plan_v2_sha256(exact.plan)
    object.__setattr__(observed, "target", object())
    with pytest.raises(V2IntegrityError, match="run identity snapshot"):
        _reverify_run_identity(
            observed,
            exact,
            semantic_digest=semantic_digest,
            plan_digest=plan_digest,
        )


def test_active_preparation_requires_bound_null_and_snapshot_as_a_pair() -> None:
    prepared = _prepare_calibration(valid_observed(), resolution())
    assert type(prepared) is calibration_v2._PreparedCalibration
    null_bind = NullBindResult(
        status=NullBindStatus.ENABLED,
        bound=prepared.bound_null,
        disabled_reason=None,
        diagnostics=prepared.diagnostics,
    )
    null_bind_snapshot = _validate_null_bind_snapshot(null_bind)
    with pytest.raises(V2IntegrityError, match="must be paired"):
        _reverify_active_preparation(
            prepared.observed_pair,
            prepared.resolution,
            semantic_digest=prepared.semantic_input_sha256,
            plan_digest=prepared.scientific_plan_sha256,
            bound_statistic=prepared.bound_statistic,
            statistic_snapshot=prepared.statistic_snapshot,
            null_bind=null_bind,
            null_bind_snapshot=null_bind_snapshot,
            bound_null=prepared.bound_null,
            null_snapshot=None,
        )


@pytest.mark.parametrize("diagnostics", [["mutated"], ("mutated",)])
def test_null_bind_diagnostics_drift_during_scan_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    diagnostics: object,
) -> None:
    exact = resolution()
    original_bind = type(exact.adapters.null_model).bind
    original_evaluate = _BoundLaggedPearsonAdapter.evaluate_all
    captured_bind: list[NullBindResult] = []

    def capture_bind(self: object, *args: object, **kwargs: object) -> object:
        result = original_bind(self, *args, **kwargs)  # type: ignore[arg-type]
        captured_bind.append(result)
        return result

    def drifting_evaluate(
        self: _BoundLaggedPearsonAdapter,
        supplied: SeriesPair,
        /,
    ) -> object:
        object.__setattr__(captured_bind[0], "diagnostics", diagnostics)
        return original_evaluate(self, supplied)

    monkeypatch.setattr(type(exact.adapters.null_model), "bind", capture_bind)
    monkeypatch.setattr(_BoundLaggedPearsonAdapter, "evaluate_all", drifting_evaluate)
    with pytest.raises(V2IntegrityError, match=r"null bind.*drift|diagnostics"):
        _prepare_calibration(valid_observed(), exact)


@pytest.mark.parametrize("field", ["status", "bound", "disabled_reason", "diagnostics"])
def test_null_bind_missing_slot_at_initial_snapshot_is_normalized(
    field: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact = resolution()
    observed = valid_observed()
    damaged = exact.adapters.null_model.bind(
        observed,
        semantic_input_sha256=semantic_input_sha256(observed, exact.plan.candidates),
        scientific_plan_sha256=scientific_plan_v2_sha256(exact.plan),
    )
    object.__delattr__(damaged, field)
    monkeypatch.setattr(
        type(exact.adapters.null_model),
        "bind",
        lambda *args, **kwargs: damaged,
    )
    with pytest.raises(V2IntegrityError, match="null bind identity snapshot"):
        _prepare_calibration(observed, exact)


@pytest.mark.parametrize("field", ["status", "bound", "disabled_reason", "diagnostics"])
def test_null_bind_missing_slot_at_scan_checkpoint_is_normalized(
    field: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact = resolution()
    original_bind = type(exact.adapters.null_model).bind
    original_evaluate = _BoundLaggedPearsonAdapter.evaluate_all
    captured_bind: list[NullBindResult] = []

    def capture_bind(self: object, *args: object, **kwargs: object) -> object:
        result = original_bind(self, *args, **kwargs)  # type: ignore[arg-type]
        captured_bind.append(result)
        return result

    def damaging_evaluate(
        self: _BoundLaggedPearsonAdapter,
        supplied: SeriesPair,
        /,
    ) -> object:
        object.__delattr__(captured_bind[0], field)
        return original_evaluate(self, supplied)

    monkeypatch.setattr(type(exact.adapters.null_model), "bind", capture_bind)
    monkeypatch.setattr(_BoundLaggedPearsonAdapter, "evaluate_all", damaging_evaluate)
    with pytest.raises(V2IntegrityError, match="null bind identity snapshot"):
        _prepare_calibration(valid_observed(), exact)


@pytest.mark.parametrize(
    "error",
    [RuntimeError("runtime"), OSError("io"), AssertionError("bug")],
)
def test_null_bind_snapshot_nonstructural_field_errors_propagate(
    error: BaseException,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bind = NullBindResult(
        status=NullBindStatus.DISABLED,
        bound=None,
        disabled_reason=NullDisabledReason.SHIFT_SPACE_EMPTY,
        diagnostics=(),
    )

    def fail_diagnostics(self: object) -> object:
        raise error

    monkeypatch.setattr(NullBindResult, "diagnostics", property(fail_diagnostics))
    with pytest.raises(type(error), match=str(error)):
        _validate_null_bind_snapshot(bind)


@pytest.mark.parametrize("diagnostics", [["list"], (object(),)])
def test_prepared_calibration_requires_exact_string_tuple_diagnostics(
    diagnostics: object,
) -> None:
    prepared = _prepare_calibration(valid_observed(), resolution())
    assert type(prepared) is calibration_v2._PreparedCalibration
    object.__setattr__(prepared, "diagnostics", diagnostics)
    require_owned = getattr(calibration_v2, "_require_preparer_owned_prepared", None)
    assert callable(require_owned), "preparer ownership gate must exist"
    with pytest.raises(V2IntegrityError, match="diagnostics"):
        require_owned(prepared)


class _FailingItemsMapping(Mapping[str, object]):
    def __init__(self, error: BaseException) -> None:
        self._error = error

    def __getitem__(self, key: str) -> object:
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(())

    def __len__(self) -> int:
        return 0

    def items(self) -> object:
        raise self._error


class _SecondItemsFailureMapping(Mapping[str, object]):
    def __init__(self, error: BaseException | None = None) -> None:
        self.items_calls = 0
        self._error = error or AttributeError("second items read failed")

    def __getitem__(self, key: str) -> object:
        if key == "min_shift":
            return 1
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(("min_shift",))

    def __len__(self) -> int:
        return 1

    def items(self) -> object:
        self.items_calls += 1
        if self.items_calls == 1:
            return {"min_shift": 1}.items()
        raise self._error


@pytest.mark.parametrize("role", ["statistic", "null"])
def test_bound_parameter_items_attribute_error_is_normalized(
    role: str,
) -> None:
    if role == "statistic":
        fields = statistic_snapshot_fields()
        fields["parameters"] = _FailingItemsMapping(AttributeError("items failed"))
        with pytest.raises(V2IntegrityError, match="parameters"):
            _snapshot_bound_statistic(
                SimpleNamespace(**fields),
                resolution(),
                observed_pair=valid_observed(),
            )
        return

    fields, exact, observed, semantic_digest, plan_digest = null_snapshot_context()
    fields["parameters"] = _FailingItemsMapping(AttributeError("items failed"))
    with pytest.raises(V2IntegrityError, match="parameters"):
        _snapshot_bound_null(
            SimpleNamespace(**fields),
            exact,  # type: ignore[arg-type]
            observed_length=int(observed.source.size),
            semantic_digest=semantic_digest,
            plan_digest=plan_digest,
            observed_pair=observed,
        )


def test_null_parameter_digest_second_items_failure_is_normalized() -> None:
    fields, exact, observed, semantic_digest, plan_digest = null_snapshot_context()
    parameters = _SecondItemsFailureMapping()
    fields["parameters"] = parameters
    with pytest.raises(V2IntegrityError, match="parameters"):
        _snapshot_bound_null(
            SimpleNamespace(**fields),
            exact,  # type: ignore[arg-type]
            observed_length=int(observed.source.size),
            semantic_digest=semantic_digest,
            plan_digest=plan_digest,
            observed_pair=observed,
        )
    assert parameters.items_calls == 2


@pytest.mark.parametrize("role", ["statistic", "null"])
@pytest.mark.parametrize("error", [RuntimeError("runtime"), OSError("io"), AssertionError("bug")])
def test_bound_parameter_items_nonstructural_errors_propagate(
    role: str,
    error: BaseException,
) -> None:
    if role == "statistic":
        fields = statistic_snapshot_fields()
        fields["parameters"] = _FailingItemsMapping(error)
        with pytest.raises(type(error), match=str(error)):
            _snapshot_bound_statistic(
                SimpleNamespace(**fields),
                resolution(),
                observed_pair=valid_observed(),
            )
        return

    fields, exact, observed, semantic_digest, plan_digest = null_snapshot_context()
    fields["parameters"] = _FailingItemsMapping(error)
    with pytest.raises(type(error), match=str(error)):
        _snapshot_bound_null(
            SimpleNamespace(**fields),
            exact,  # type: ignore[arg-type]
            observed_length=int(observed.source.size),
            semantic_digest=semantic_digest,
            plan_digest=plan_digest,
            observed_pair=observed,
        )


@pytest.mark.parametrize("error", [RuntimeError("runtime"), OSError("io"), AssertionError("bug")])
def test_null_parameter_digest_second_nonstructural_error_propagates(
    error: BaseException,
) -> None:
    fields, exact, observed, semantic_digest, plan_digest = null_snapshot_context()
    parameters = _SecondItemsFailureMapping(error)
    fields["parameters"] = parameters
    with pytest.raises(type(error), match=str(error)):
        _snapshot_bound_null(
            SimpleNamespace(**fields),
            exact,  # type: ignore[arg-type]
            observed_length=int(observed.source.size),
            semantic_digest=semantic_digest,
            plan_digest=plan_digest,
            observed_pair=observed,
        )
    assert parameters.items_calls == 2


def test_disabled_path_reverifies_active_state_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact = resolve_plan_v2(
        PlanRequestV2(
            candidates=(1, 3),
            statistic_name="lagged_pearson_v1",
            statistic_params={},
            selection_rule="max_upper",
            null_name="circular_shift_v2",
            null_params={"min_shift": 3},
            replicates=5,
            alpha=0.05,
            tie_tolerance=1e-12,
            root_seed=17,
        )
    )
    original_reverify = calibration_v2._reverify_active_preparation
    reverify_calls = 0

    def spy_reverify(*args: object, **kwargs: object) -> None:
        nonlocal reverify_calls
        reverify_calls += 1
        original_reverify(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(calibration_v2, "_reverify_active_preparation", spy_reverify)
    result = _prepare_calibration(
        pair([0, 1, 2, 3, 4], [4, 3, 2, 1, 0]),
        exact,
    )
    assert type(result) is CalibrationResult
    assert result.failure_stage is RunFailureStage.NULL_BIND
    assert reverify_calls == 1


def test_pearson_observed_length_drift_after_scan_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_evaluate = _BoundLaggedPearsonAdapter.evaluate_all

    def drifting_evaluate(
        self: _BoundLaggedPearsonAdapter,
        supplied: SeriesPair,
        /,
    ) -> object:
        results = original_evaluate(self, supplied)
        object.__setattr__(self, "observed_length", self.observed_length + 1)
        return results

    monkeypatch.setattr(_BoundLaggedPearsonAdapter, "evaluate_all", drifting_evaluate)
    with pytest.raises(V2IntegrityError, match=r"statistic.*drift"):
        _prepare_calibration(valid_observed(), resolution())


@pytest.mark.parametrize(
    "field",
    ["observed_length", "source_edges", "target_edges", "bind_diagnostics"],
)
def test_binned_execution_state_drift_after_scan_is_rejected(
    field: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_evaluate = _BoundBinnedNetTEAdapter.evaluate_all

    def drifting_evaluate(
        self: _BoundBinnedNetTEAdapter,
        supplied: SeriesPair,
        /,
    ) -> object:
        results = original_evaluate(self, supplied)
        if field == "observed_length":
            replacement: object = self.observed_length + 1
        elif field == "bind_diagnostics":
            replacement = ("SELCAL_BINNED_NETTE_CONSTANT_SOURCE",)
        else:
            replacement = np.asarray(getattr(self, field), dtype=np.float64) + 0.125
            replacement.setflags(write=False)  # type: ignore[union-attr]
        object.__setattr__(self, field, replacement)
        return results

    monkeypatch.setattr(_BoundBinnedNetTEAdapter, "evaluate_all", drifting_evaluate)
    with pytest.raises(V2IntegrityError, match=r"statistic.*drift"):
        _prepare_calibration(binned_observed(), binned_resolution())


def test_statistic_evaluate_operation_drift_after_scan_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_evaluate = _BoundLaggedPearsonAdapter.evaluate_all

    def replacement_evaluate(
        self: _BoundLaggedPearsonAdapter,
        supplied: SeriesPair,
        /,
    ) -> object:
        return original_evaluate(self, supplied)

    def swapping_evaluate(
        self: _BoundLaggedPearsonAdapter,
        supplied: SeriesPair,
        /,
    ) -> object:
        results = original_evaluate(self, supplied)
        monkeypatch.setattr(
            _BoundLaggedPearsonAdapter,
            "evaluate_all",
            replacement_evaluate,
        )
        return results

    monkeypatch.setattr(_BoundLaggedPearsonAdapter, "evaluate_all", swapping_evaluate)
    with pytest.raises(V2IntegrityError, match=r"statistic.*drift|evaluate_all"):
        _prepare_calibration(valid_observed(), resolution())


@pytest.mark.parametrize(
    ("null_name", "null_params"),
    [
        ("circular_shift_v2", {"min_shift": 1}),
        ("block_shuffle_v2", {"block_length": 2}),
    ],
)
def test_bound_null_observed_owner_state_drift_after_scan_is_rejected(
    null_name: str,
    null_params: dict[str, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact = resolution_with_null(null_name, null_params)
    original_bind = type(exact.adapters.null_model).bind
    original_evaluate = _BoundLaggedPearsonAdapter.evaluate_all
    captured_bound: list[object] = []

    def capture_bind(self: object, *args: object, **kwargs: object) -> object:
        result = original_bind(self, *args, **kwargs)  # type: ignore[arg-type]
        assert result.bound is not None
        captured_bound.append(result.bound)
        return result

    def drifting_evaluate(
        self: _BoundLaggedPearsonAdapter,
        supplied: SeriesPair,
        /,
    ) -> object:
        results = original_evaluate(self, supplied)
        foreign = pair([5, 4, 3, 2, 1, 0], [0, 2, 4, 1, 3, 5])
        object.__setattr__(captured_bound[0], "_observed_pair", foreign)
        object.__setattr__(
            captured_bound[0],
            "_observed_content_guard_sha256",
            _owned_pair_content_guard_sha256(foreign),
        )
        return results

    monkeypatch.setattr(type(exact.adapters.null_model), "bind", capture_bind)
    monkeypatch.setattr(_BoundLaggedPearsonAdapter, "evaluate_all", drifting_evaluate)
    with pytest.raises(V2IntegrityError, match=r"null.*drift"):
        _prepare_calibration(valid_observed(), exact)


@pytest.mark.parametrize(
    ("null_name", "null_params", "bound_type"),
    [
        ("circular_shift_v2", {"min_shift": 1}, _BoundCircularShiftV2),
        ("block_shuffle_v2", {"block_length": 2}, _BoundBlockShuffleV2),
    ],
)
@pytest.mark.parametrize("method_name", ["identity_token", "sample_token", "apply"])
def test_bound_null_operation_drift_after_scan_is_rejected(
    null_name: str,
    null_params: dict[str, int],
    bound_type: type[object],
    method_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact = resolution_with_null(null_name, null_params)
    original_evaluate = _BoundLaggedPearsonAdapter.evaluate_all

    def replacement_operation(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("replacement operation must never execute in Task 8A")

    def swapping_evaluate(
        self: _BoundLaggedPearsonAdapter,
        supplied: SeriesPair,
        /,
    ) -> object:
        results = original_evaluate(self, supplied)
        monkeypatch.setattr(bound_type, method_name, replacement_operation)
        return results

    monkeypatch.setattr(_BoundLaggedPearsonAdapter, "evaluate_all", swapping_evaluate)
    with pytest.raises(V2IntegrityError, match=r"null.*drift|operation"):
        _prepare_calibration(valid_observed(), exact)


@pytest.mark.parametrize(
    "attack",
    [
        "foreign_selection",
        "list_scored",
        "candidate",
        "estimate",
        "score",
        "support",
        "diagnostics",
        "decision",
        "tie",
        "selected_index",
    ],
)
def test_prepare_independently_verifies_the_selector_output_weld(
    attack: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def malicious_selector(*args: object, **kwargs: object) -> object:
        selection, scored = real_select_family_v2(*args, **kwargs)  # type: ignore[arg-type]
        if attack == "foreign_selection":
            return object(), scored
        if attack == "list_scored":
            return selection, list(scored)
        if attack in {"candidate", "estimate", "score", "support", "diagnostics"}:
            field, value = {
                "candidate": ("candidate_id", 99),
                "estimate": ("estimate", float(scored[0].estimate or 0.0) + 0.25),
                "score": (
                    "selection_score",
                    float(scored[0].selection_score or 0.0) + 0.25,
                ),
                "support": ("support_n", scored[0].support_n + 1),
                "diagnostics": ("diagnostics", ("selector-mutated",)),
            }[attack]
            object.__setattr__(scored[0], field, value)
            return selection, scored
        field, value = {
            "decision": ("decision_statistic", selection.decision_statistic + 0.25),
            "tie": ("tied_candidates", exact_tuple_without_first(selection.tied_candidates)),
            "selected_index": ("selected_index", len(scored)),
        }[attack]
        object.__setattr__(selection, field, value)
        return selection, scored

    monkeypatch.setattr(calibration_v2, "select_family_v2", malicious_selector)
    with pytest.raises(V2IntegrityError, match=r"selection|selector|scored|observed"):
        _prepare_calibration(valid_observed(), resolution())


def exact_tuple_without_first(values: tuple[int, ...]) -> tuple[int, ...]:
    if len(values) > 1:
        return values[1:]
    return (99,)


def test_prepared_retains_the_exact_raw_observed_vector() -> None:
    prepared = _prepare_calibration(valid_observed(), resolution())
    assert type(prepared) is calibration_v2._PreparedCalibration
    assert hasattr(prepared, "raw_observed_results")
    raw = prepared.raw_observed_results  # type: ignore[attr-defined]
    assert type(raw) is tuple
    assert all(type(result) is StatisticResult for result in raw)
    assert tuple(result.candidate_id for result in raw) == prepared.resolution.plan.candidates
    assert all(result.selection_score is None for result in raw)


def require_preparer_owned(value: object) -> object:
    gate = getattr(calibration_v2, "_require_preparer_owned_prepared", None)
    assert callable(gate), "preparer ownership gate must exist"
    return gate(value)


def test_legitimate_prepared_state_has_process_local_preparer_ownership() -> None:
    prepared = _prepare_calibration(valid_observed(), resolution())
    assert require_preparer_owned(prepared) is prepared


def test_prepared_rejects_direct_construction_without_the_private_seal() -> None:
    prepared = _prepare_calibration(valid_observed(), resolution())
    assert type(prepared) is calibration_v2._PreparedCalibration
    kwargs = {
        field.name: getattr(prepared, field.name)
        for field in fields(type(prepared))
        if field.init
    }
    with pytest.raises(V2IntegrityError, match=r"preparer|created"):
        type(prepared)(**kwargs)


def test_preparer_ownership_rejects_replace_copy_and_object_new() -> None:
    prepared = _prepare_calibration(valid_observed(), resolution())
    assert type(prepared) is calibration_v2._PreparedCalibration
    with pytest.raises(V2IntegrityError, match=r"preparer|created"):
        replace(prepared)
    for candidate in (
        copy(prepared),
        object.__new__(calibration_v2._PreparedCalibration),
    ):
        with pytest.raises(V2IntegrityError, match=r"preparer|owned|snapshot"):
            require_preparer_owned(candidate)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("diagnostics", ("drifted",)),
        ("semantic_input_sha256", "a" * 64),
        ("scientific_plan_sha256", "b" * 64),
    ],
)
def test_preparer_ownership_rejects_prepared_field_drift(
    field: str,
    value: object,
) -> None:
    prepared = _prepare_calibration(valid_observed(), resolution())
    assert type(prepared) is calibration_v2._PreparedCalibration
    object.__setattr__(prepared, field, value)
    with pytest.raises(V2IntegrityError, match=r"preparer|owned|snapshot|diagnostics"):
        require_preparer_owned(prepared)


def test_evaluate_all_is_resolved_once_and_the_frozen_callable_is_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_evaluate = _BoundLaggedPearsonAdapter.evaluate_all

    class CountingDescriptor:
        def __init__(self) -> None:
            self.bind_calls = 0

        def __get__(self, instance: object, owner: type[object]) -> object:
            self.bind_calls += 1
            return original_evaluate.__get__(instance, owner)

    descriptor = CountingDescriptor()
    monkeypatch.setattr(_BoundLaggedPearsonAdapter, "evaluate_all", descriptor)
    prepared = _prepare_calibration(valid_observed(), resolution())
    assert type(prepared) is calibration_v2._PreparedCalibration
    assert descriptor.bind_calls == 1


def test_evaluate_all_property_is_rejected_without_running_its_getter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    getter_calls = 0
    original_evaluate = _BoundLaggedPearsonAdapter.evaluate_all

    def stateful_getter(self: object) -> object:
        nonlocal getter_calls
        getter_calls += 1
        if getter_calls == 1:
            return original_evaluate.__get__(self, type(self))
        raise ValueError("second evaluate_all read")

    monkeypatch.setattr(
        _BoundLaggedPearsonAdapter,
        "evaluate_all",
        property(stateful_getter),
    )
    with pytest.raises(V2IntegrityError, match="evaluate_all"):
        _prepare_calibration(valid_observed(), resolution())
    assert getter_calls == 0


@pytest.mark.parametrize(
    "error",
    [
        AttributeError("missing"),
        TypeError("type"),
        ValueError("value"),
        OverflowError("overflow"),
    ],
)
def test_evaluate_all_descriptor_structural_errors_are_normalized(
    error: BaseException,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingDescriptor:
        def __get__(self, instance: object, owner: type[object]) -> object:
            del instance, owner
            raise error

    monkeypatch.setattr(_BoundLaggedPearsonAdapter, "evaluate_all", FailingDescriptor())
    with pytest.raises(V2IntegrityError, match="evaluate_all"):
        _prepare_calibration(valid_observed(), resolution())


@pytest.mark.parametrize(
    "error",
    [RuntimeError("runtime"), OSError("io"), AssertionError("bug")],
)
def test_evaluate_all_descriptor_nonstructural_errors_propagate(
    error: BaseException,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingDescriptor:
        def __get__(self, instance: object, owner: type[object]) -> object:
            del instance, owner
            raise error

    monkeypatch.setattr(_BoundLaggedPearsonAdapter, "evaluate_all", FailingDescriptor())
    with pytest.raises(type(error), match=str(error)):
        _prepare_calibration(valid_observed(), resolution())


def test_prepared_private_seal_does_not_bypass_field_validation() -> None:
    prepared = _prepare_calibration(valid_observed(), resolution())
    assert type(prepared) is calibration_v2._PreparedCalibration
    kwargs = {
        field.name: getattr(prepared, field.name)
        for field in fields(type(prepared))
        if field.init
    }
    kwargs["diagnostics"] = ["bad"]
    with pytest.raises(V2IntegrityError, match="diagnostics"):
        type(prepared)(
            **kwargs,
            seal=calibration_v2._PREPARATION_SEAL,
        )


def test_result_snapshot_normalizes_a_missing_slot() -> None:
    prepared = _prepare_calibration(valid_observed(), resolution())
    assert type(prepared) is calibration_v2._PreparedCalibration
    object.__delattr__(prepared.raw_observed_results[0], "support_n")
    with pytest.raises(V2IntegrityError, match="snapshot"):
        require_preparer_owned(prepared)


def test_scored_result_rejects_a_nonfinite_selection_score() -> None:
    prepared = _prepare_calibration(valid_observed(), resolution())
    assert type(prepared) is calibration_v2._PreparedCalibration
    object.__setattr__(prepared.observed_results[0], "selection_score", float("nan"))
    with pytest.raises(V2IntegrityError, match="selection score"):
        require_preparer_owned(prepared)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("selected_candidate", True, "candidate"),
        ("selected_index", True, "index"),
        ("decision_statistic", float("nan"), "decision"),
        ("tied_candidates", [1], "tied"),
    ],
)
def test_selection_snapshot_rejects_mutated_fields(
    field: str,
    value: object,
    match: str,
) -> None:
    prepared = _prepare_calibration(valid_observed(), resolution())
    assert type(prepared) is calibration_v2._PreparedCalibration
    object.__setattr__(prepared.observed_selection, field, value)
    with pytest.raises(V2IntegrityError, match=match):
        require_preparer_owned(prepared)


def test_selection_snapshot_normalizes_a_missing_slot() -> None:
    prepared = _prepare_calibration(valid_observed(), resolution())
    assert type(prepared) is calibration_v2._PreparedCalibration
    object.__delattr__(prepared.observed_selection, "selected_index")
    with pytest.raises(V2IntegrityError, match="selection snapshot"):
        require_preparer_owned(prepared)


def test_prepared_result_vectors_remain_exact_tuples() -> None:
    prepared = _prepare_calibration(valid_observed(), resolution())
    assert type(prepared) is calibration_v2._PreparedCalibration
    object.__setattr__(prepared, "raw_observed_results", list(prepared.raw_observed_results))
    with pytest.raises(V2IntegrityError, match="exact StatisticResult tuple"):
        require_preparer_owned(prepared)


@pytest.mark.parametrize("attack", ["raw_mutation", "short_scored"])
def test_selector_rejects_vector_level_weld_attacks(
    attack: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def malicious_selector(*args: object, **kwargs: object) -> object:
        selection, scored = real_select_family_v2(*args, **kwargs)  # type: ignore[arg-type]
        raw = args[0]
        assert type(raw) is tuple
        if attack == "raw_mutation":
            object.__setattr__(raw[0], "support_n", raw[0].support_n + 1)
            return selection, scored
        return selection, scored[:-1]

    monkeypatch.setattr(calibration_v2, "select_family_v2", malicious_selector)
    with pytest.raises(V2IntegrityError, match=r"mutated|does not match"):
        _prepare_calibration(valid_observed(), resolution())


def test_selector_rejects_plan_rule_drift_before_prepared_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact = resolution()

    def drifting_selector(*args: object, **kwargs: object) -> object:
        result = real_select_family_v2(*args, **kwargs)  # type: ignore[arg-type]
        object.__setattr__(exact.plan, "selection_rule", "max_upper")
        return result

    monkeypatch.setattr(calibration_v2, "select_family_v2", drifting_selector)
    with pytest.raises(V2IntegrityError, match="selection rule"):
        _prepare_calibration(valid_observed(), exact)


def test_selector_validator_rejects_an_empty_family() -> None:
    with pytest.raises(V2IntegrityError, match="empty observed family"):
        calibration_v2._validate_selector_output(
            (),
            (),
            calibration_v2.SelectionResult(
                selected_candidate=1,
                selected_index=0,
                decision_statistic=0.0,
                tied_candidates=(1,),
            ),
            (),
            resolution().plan.candidates,
            True,
            resolution().plan.tie_tolerance,
        )


@pytest.mark.parametrize("attack", ["wrong_operation", "missing_operation_slot"])
def test_preparer_ownership_rejects_frozen_operation_damage(attack: str) -> None:
    prepared = _prepare_calibration(valid_observed(), resolution())
    assert type(prepared) is calibration_v2._PreparedCalibration
    execution = prepared.statistic_snapshot.execution
    if attack == "wrong_operation":
        object.__setattr__(execution, "evaluate_all", object())
    else:
        object.__delattr__(execution.evaluate_all, "callable")
    with pytest.raises(V2IntegrityError, match=r"operation|snapshot"):
        require_preparer_owned(prepared)


@pytest.mark.parametrize("attack", ["component", "execution"])
def test_preparer_ownership_rejects_a_replaced_component_snapshot(attack: str) -> None:
    prepared = _prepare_calibration(valid_observed(), resolution())
    assert type(prepared) is calibration_v2._PreparedCalibration
    if attack == "component":
        object.__setattr__(prepared, "statistic_snapshot", object())
    else:
        object.__setattr__(prepared.statistic_snapshot, "execution", object())
    with pytest.raises(V2IntegrityError, match="snapshot"):
        require_preparer_owned(prepared)


def test_prepared_signature_normalizes_a_missing_component() -> None:
    prepared = _prepare_calibration(valid_observed(), resolution())
    assert type(prepared) is calibration_v2._PreparedCalibration
    object.__delattr__(prepared, "statistic_snapshot")
    with pytest.raises(V2IntegrityError, match="snapshot"):
        require_preparer_owned(prepared)


def test_preparer_ownership_requires_an_exact_prepared_value() -> None:
    with pytest.raises(V2IntegrityError, match="exact preparer-owned"):
        require_preparer_owned(object())


def test_prepared_capture_rejects_a_drifted_null_bind_association() -> None:
    prepared = _prepare_calibration(valid_observed(), resolution())
    assert type(prepared) is calibration_v2._PreparedCalibration
    object.__setattr__(
        prepared.null_bind_snapshot,
        "status",
        NullBindStatus.DISABLED,
    )
    with pytest.raises(V2IntegrityError, match="null bind association"):
        calibration_v2._capture_prepared_ownership(prepared)


def test_selector_return_container_must_be_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(calibration_v2, "select_family_v2", lambda *args: object())
    with pytest.raises(V2IntegrityError, match="two-item tuple"):
        _prepare_calibration(valid_observed(), resolution())


@pytest.mark.parametrize("role", ["statistic", "null"])
def test_registered_bound_snapshot_requires_the_exact_implementation(role: str) -> None:
    exact = resolution()
    with pytest.raises(V2IntegrityError, match="exact registered implementation"):
        if role == "statistic":
            resolution_v2._snapshot_registered_bound_statistic_v2(
                exact,
                object(),
                expected_pair=valid_observed(),
            )
        else:
            resolution_v2._snapshot_registered_bound_null_v2(
                exact,
                object(),
                expected_pair=valid_observed(),
            )


@pytest.mark.parametrize("attack", ["missing", "noncallable"])
def test_registered_statistic_operation_resolution_fails_closed(
    attack: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact = resolution()
    observed = valid_observed()
    bound = LaggedPearsonAdapter().bind(observed, exact.plan.candidates)
    if attack == "missing":
        monkeypatch.delattr(_BoundLaggedPearsonAdapter, "evaluate_all")
        match = "operation snapshot"
    else:
        monkeypatch.setattr(_BoundLaggedPearsonAdapter, "evaluate_all", object())
        match = "callable"
    with pytest.raises(V2IntegrityError, match=match):
        resolution_v2._snapshot_registered_bound_statistic_v2(
            exact,
            bound,
            expected_pair=observed,
        )


@pytest.mark.parametrize("role", ["pearson", "binned"])
def test_bound_statistic_execution_snapshot_normalizes_missing_state(role: str) -> None:
    if role == "pearson":
        exact = resolution()
        observed = valid_observed()
        bound: object = LaggedPearsonAdapter().bind(
            observed,
            exact.plan.candidates,
        )
        object.__delattr__(bound, "observed_length")
    else:
        exact = binned_resolution()
        observed = binned_observed()
        bound = exact.adapters.statistic.bind(
            observed,
            exact.plan.candidates,
        )
        object.__delattr__(bound, "bind_diagnostics")
    with pytest.raises(V2IntegrityError, match="execution snapshot"):
        resolution_v2._snapshot_registered_bound_statistic_v2(
            exact,
            bound,
            expected_pair=observed,
        )


def test_binned_execution_snapshot_requires_exact_edge_arrays() -> None:
    exact = binned_resolution()
    observed = binned_observed()
    bound = exact.adapters.statistic.bind(
        observed,
        exact.plan.candidates,
    )
    object.__setattr__(bound, "source_edges", object())
    with pytest.raises(V2IntegrityError, match="exact ndarray"):
        resolution_v2._snapshot_registered_bound_statistic_v2(
            exact,
            bound,
            expected_pair=observed,
        )


@pytest.mark.parametrize(
    ("null_name", "null_params", "field"),
    [
        ("circular_shift_v2", {"min_shift": 1}, "_min_shift"),
        ("block_shuffle_v2", {"block_length": 2}, "_block_length"),
        (
            "circular_shift_v2",
            {"min_shift": 1},
            "_observed_content_guard_sha256",
        ),
    ],
)
def test_bound_null_execution_snapshot_normalizes_missing_state(
    null_name: str,
    null_params: dict[str, int],
    field: str,
) -> None:
    exact = resolution_with_null(null_name, null_params)
    observed = valid_observed()
    bind = exact.adapters.null_model.bind(
        observed,
        semantic_input_sha256=semantic_input_sha256(observed, exact.plan.candidates),
        scientific_plan_sha256=scientific_plan_v2_sha256(exact.plan),
    )
    assert bind.bound is not None
    object.__delattr__(bind.bound, field)
    with pytest.raises(V2IntegrityError, match="execution snapshot"):
        resolution_v2._snapshot_registered_bound_null_v2(
            exact,
            bind.bound,
            expected_pair=observed,
        )


def test_bound_null_execution_snapshot_rejects_a_foreign_pair_type() -> None:
    exact = resolution()
    observed = valid_observed()
    bind = exact.adapters.null_model.bind(
        observed,
        semantic_input_sha256=semantic_input_sha256(observed, exact.plan.candidates),
        scientific_plan_sha256=scientific_plan_v2_sha256(exact.plan),
    )
    assert bind.bound is not None
    object.__setattr__(bind.bound, "_observed_pair", object())
    with pytest.raises(V2IntegrityError, match="exact SeriesPair"):
        resolution_v2._snapshot_registered_bound_null_v2(
            exact,
            bind.bound,
            expected_pair=observed,
        )


def test_bound_null_execution_snapshot_rejects_guard_drift() -> None:
    exact = resolution()
    observed = valid_observed()
    bind = exact.adapters.null_model.bind(
        observed,
        semantic_input_sha256=semantic_input_sha256(observed, exact.plan.candidates),
        scientific_plan_sha256=scientific_plan_v2_sha256(exact.plan),
    )
    assert bind.bound is not None
    object.__setattr__(bind.bound, "_observed_content_guard_sha256", "a" * 64)
    with pytest.raises(V2IntegrityError, match="content guard"):
        resolution_v2._snapshot_registered_bound_null_v2(
            exact,
            bind.bound,
            expected_pair=observed,
        )


def test_binned_bind_must_derive_its_edges_from_this_observed_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact = binned_resolution()
    original_bind = BinnedNetTEAdapter.bind
    foreign = pair(
        [-100, -80, -60, -40, -20, 0, 20, 40, 60, 100],
        [100, 60, 40, 20, 0, -20, -40, -60, -80, -100],
    )

    def foreign_bind(
        self: BinnedNetTEAdapter,
        observed_pair: SeriesPair,
        candidates: tuple[int, ...],
    ) -> object:
        del observed_pair
        return original_bind(self, foreign, candidates)

    monkeypatch.setattr(BinnedNetTEAdapter, "bind", foreign_bind)
    with pytest.raises(V2IntegrityError, match=r"statistic.*observed|binding|edges"):
        _prepare_calibration(binned_observed(), exact)


@pytest.mark.parametrize(
    ("null_name", "null_params"),
    [
        ("circular_shift_v2", {"min_shift": 1}),
        ("block_shuffle_v2", {"block_length": 2}),
    ],
)
def test_bound_null_must_retain_this_observed_pair_at_initial_bind(
    null_name: str,
    null_params: dict[str, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact = resolution_with_null(null_name, null_params)
    original_bind = type(exact.adapters.null_model).bind
    foreign = pair([5, 4, 3, 2, 1, 0], [0, 2, 4, 1, 3, 5])

    def foreign_bind(
        self: object,
        observed_pair: SeriesPair,
        *,
        semantic_input_sha256: str,
        scientific_plan_sha256: str,
    ) -> object:
        del observed_pair
        return original_bind(
            self,
            foreign,
            semantic_input_sha256=semantic_input_sha256,
            scientific_plan_sha256=scientific_plan_sha256,
        )

    monkeypatch.setattr(type(exact.adapters.null_model), "bind", foreign_bind)
    with pytest.raises(V2IntegrityError, match=r"null.*observed|binding|pair"):
        _prepare_calibration(valid_observed(), exact)


def test_pearson_same_length_foreign_bind_has_no_hidden_input_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = valid_observed()
    exact = resolution()
    baseline = _prepare_calibration(observed, exact)
    assert type(baseline) is calibration_v2._PreparedCalibration
    baseline_raw = tuple(result.estimate for result in baseline.raw_observed_results)
    original_bind = LaggedPearsonAdapter.bind
    foreign = pair([5, 4, 3, 2, 1, 0], [0, 2, 4, 1, 3, 5])

    def foreign_bind(
        self: LaggedPearsonAdapter,
        observed_pair: SeriesPair,
        candidates: tuple[int, ...],
    ) -> object:
        del observed_pair
        return original_bind(self, foreign, candidates)

    monkeypatch.setattr(LaggedPearsonAdapter, "bind", foreign_bind)
    prepared = _prepare_calibration(observed, exact)
    assert type(prepared) is calibration_v2._PreparedCalibration
    assert tuple(result.estimate for result in prepared.raw_observed_results) == baseline_raw


def _execution_case(
    role: str,
) -> tuple[object, SeriesPair, str, bool]:
    if role == "pearson_length":
        return resolution(), valid_observed(), "observed_length", False
    if role == "binned_bins":
        return binned_resolution(), binned_observed(), "bins", False
    if role == "binned_length":
        return binned_resolution(), binned_observed(), "observed_length", False
    if role == "circular_min_shift":
        return resolution(), valid_observed(), "_min_shift", True
    if role == "block_length":
        return (
            resolution_with_null("block_shuffle_v2", {"block_length": 2}),
            valid_observed(),
            "_block_length",
            True,
        )
    if role == "block_count":
        return (
            resolution_with_null("block_shuffle_v2", {"block_length": 2}),
            valid_observed(),
            "_block_count",
            True,
        )
    raise AssertionError(f"unknown execution role: {role}")


def _non_builtin_integer(value: int, kind: str) -> object:
    if kind == "bool":
        return bool(value)
    if kind == "float":
        return float(value)
    if kind == "numpy":
        return np.int64(value)
    raise AssertionError(f"unknown integer kind: {kind}")


@pytest.mark.parametrize(
    "role",
    [
        "pearson_length",
        "binned_bins",
        "binned_length",
        "circular_min_shift",
        "block_length",
        "block_count",
    ],
)
@pytest.mark.parametrize("kind", ["bool", "float", "numpy"])
def test_initial_bound_execution_integers_must_be_exact(
    role: str,
    kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact, observed, field, is_null = _execution_case(role)
    adapter = exact.adapters.null_model if is_null else exact.adapters.statistic  # type: ignore[attr-defined]
    original_bind = type(adapter).bind

    def drifting_bind(self: object, *args: object, **kwargs: object) -> object:
        result = original_bind(self, *args, **kwargs)  # type: ignore[arg-type]
        bound = result.bound if is_null else result
        assert bound is not None
        current = getattr(bound, field)
        object.__setattr__(bound, field, _non_builtin_integer(current, kind))
        return result

    monkeypatch.setattr(type(adapter), "bind", drifting_bind)
    with pytest.raises(V2IntegrityError, match=r"execution|statistic|null|binding|snapshot"):
        _prepare_calibration(observed, exact)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "role",
    [
        "pearson_length",
        "binned_bins",
        "binned_length",
        "circular_min_shift",
        "block_length",
        "block_count",
    ],
)
@pytest.mark.parametrize("kind", ["bool", "float", "numpy"])
def test_prepared_execution_integer_type_drift_is_rejected(
    role: str,
    kind: str,
) -> None:
    exact, observed, field, is_null = _execution_case(role)
    prepared = _prepare_calibration(observed, exact)  # type: ignore[arg-type]
    assert type(prepared) is calibration_v2._PreparedCalibration
    bound = prepared.bound_null if is_null else prepared.bound_statistic
    current = getattr(bound, field)
    object.__setattr__(bound, field, _non_builtin_integer(current, kind))
    with pytest.raises(V2IntegrityError, match=r"drift|execution|snapshot"):
        require_preparer_owned(prepared)


@pytest.mark.parametrize(
    ("null_name", "null_params", "field", "replacement"),
    [
        ("circular_shift_v2", {"min_shift": 1}, "_min_shift", 2),
        ("circular_shift_v2", {"min_shift": 1}, "_total_state_count", 7),
        ("block_shuffle_v2", {"block_length": 2}, "_block_length", 1),
        ("block_shuffle_v2", {"block_length": 2}, "_block_count", 1),
        ("block_shuffle_v2", {"block_length": 2}, "_total_state_count", 7),
    ],
)
def test_initial_null_execution_state_must_match_plan_and_derived_invariants(
    null_name: str,
    null_params: dict[str, int],
    field: str,
    replacement: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact = resolution_with_null(null_name, null_params)
    original_bind = type(exact.adapters.null_model).bind

    def drifting_bind(self: object, *args: object, **kwargs: object) -> object:
        result = original_bind(self, *args, **kwargs)  # type: ignore[arg-type]
        assert result.bound is not None
        object.__setattr__(result.bound, field, replacement)
        return result

    monkeypatch.setattr(type(exact.adapters.null_model), "bind", drifting_bind)
    with pytest.raises(V2IntegrityError, match=r"null|execution|state|parameter"):
        _prepare_calibration(valid_observed(), exact)


class _DescriptorGetAttributeFailure:
    def __init__(self, error: BaseException) -> None:
        self._error = error

    def __getattribute__(self, name: str) -> object:
        if name == "__get__":
            raise object.__getattribute__(self, "_error")
        return object.__getattribute__(self, name)

    def __get__(self, instance: object, owner: type[object]) -> object:
        del instance, owner
        raise AssertionError("descriptor binding must not be reached")


@pytest.mark.parametrize(
    ("method_name", "bound_type"),
    [
        ("evaluate_all", _BoundLaggedPearsonAdapter),
        ("identity_token", _BoundCircularShiftV2),
        ("sample_token", _BoundCircularShiftV2),
        ("apply", _BoundCircularShiftV2),
    ],
)
@pytest.mark.parametrize("error_type", [TypeError, ValueError, OverflowError])
def test_descriptor_get_attribute_structural_errors_are_normalized(
    method_name: str,
    bound_type: type[object],
    error_type: type[BaseException],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact = resolution()
    descriptor = _DescriptorGetAttributeFailure(error_type("descriptor access"))
    if bound_type is _BoundLaggedPearsonAdapter:
        monkeypatch.setattr(bound_type, method_name, descriptor)
    else:
        original_bind = type(exact.adapters.null_model).bind

        def install_after_bind(self: object, *args: object, **kwargs: object) -> object:
            result = original_bind(self, *args, **kwargs)  # type: ignore[arg-type]
            monkeypatch.setattr(bound_type, method_name, descriptor)
            return result

        monkeypatch.setattr(type(exact.adapters.null_model), "bind", install_after_bind)
    with pytest.raises(V2IntegrityError, match=r"operation|snapshot|descriptor"):
        _prepare_calibration(valid_observed(), exact)


@pytest.mark.parametrize(
    ("method_name", "bound_type"),
    [
        ("evaluate_all", _BoundLaggedPearsonAdapter),
        ("identity_token", _BoundCircularShiftV2),
        ("sample_token", _BoundCircularShiftV2),
        ("apply", _BoundCircularShiftV2),
    ],
)
@pytest.mark.parametrize("error_type", [RuntimeError, OSError, AssertionError])
def test_descriptor_get_attribute_nonstructural_errors_propagate(
    method_name: str,
    bound_type: type[object],
    error_type: type[BaseException],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact = resolution()
    descriptor = _DescriptorGetAttributeFailure(error_type("descriptor access"))
    if bound_type is _BoundLaggedPearsonAdapter:
        monkeypatch.setattr(bound_type, method_name, descriptor)
    else:
        original_bind = type(exact.adapters.null_model).bind

        def install_after_bind(self: object, *args: object, **kwargs: object) -> object:
            result = original_bind(self, *args, **kwargs)  # type: ignore[arg-type]
            monkeypatch.setattr(bound_type, method_name, descriptor)
            return result

        monkeypatch.setattr(type(exact.adapters.null_model), "bind", install_after_bind)
    with pytest.raises(error_type, match="descriptor access"):
        _prepare_calibration(valid_observed(), exact)


def _manual_pearson_vector(
    bound: _BoundLaggedPearsonAdapter,
    *,
    backend_identity: str,
    validity: Validity,
) -> tuple[StatisticResult, ...]:
    estimate = None if validity is Validity.ANALYTIC_FAILURE else 0.25
    diagnostics = ("analytical_failure",) if estimate is None else ()
    return tuple(
        StatisticResult(
            candidate_id=candidate,
            estimate=estimate,
            selection_score=None,
            support_n=bound.observed_length - max(bound.candidates),
            validity=validity,
            diagnostics=diagnostics,
            backend_identity=backend_identity,
            preprocessing_identity=bound.preprocessing_identity,
        )
        for candidate in bound.candidates
    )


def test_analytical_failure_return_rechecks_raw_after_the_last_live_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_backend = _BoundLaggedPearsonAdapter.backend_identity
    captured_raw: list[tuple[StatisticResult, ...]] = []
    reads = 0

    def manual_failure(
        self: _BoundLaggedPearsonAdapter,
        supplied: SeriesPair,
        /,
    ) -> tuple[StatisticResult, ...]:
        del supplied
        results = _manual_pearson_vector(
            self,
            backend_identity=original_backend.__get__(self, type(self)),
            validity=Validity.ANALYTIC_FAILURE,
        )
        captured_raw.append(results)
        return results

    def stateful_backend(self: _BoundLaggedPearsonAdapter) -> str:
        nonlocal reads
        reads += 1
        if reads == 4:
            object.__setattr__(captured_raw[0][0], "diagnostics", ("mutated",))
        return original_backend.__get__(self, type(self))

    monkeypatch.setattr(_BoundLaggedPearsonAdapter, "evaluate_all", manual_failure)
    monkeypatch.setattr(
        _BoundLaggedPearsonAdapter,
        "backend_identity",
        property(stateful_backend),
    )
    with pytest.raises(V2IntegrityError, match=r"diagnostics|observed|snapshot"):
        _prepare_calibration(valid_observed(), resolution())


def test_observed_failure_branch_uses_the_frozen_raw_validity_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_backend = _BoundLaggedPearsonAdapter.backend_identity
    original_validity = inspect.getattr_static(StatisticResult, "validity")

    def stateful_validity(self: StatisticResult) -> Validity:
        callers = tuple(frame.function for frame in inspect.stack()[1:3])
        if callers in {
            ("<genexpr>", "_prepare_calibration"),
            ("<genexpr>", "_validate_raw_failure_vector"),
        }:
            return Validity.ANALYTIC_FAILURE
        return original_validity.__get__(self, type(self))

    def set_validity(self: StatisticResult, value: Validity) -> None:
        original_validity.__set__(self, value)

    def manual_valid(
        self: _BoundLaggedPearsonAdapter,
        supplied: SeriesPair,
        /,
    ) -> tuple[StatisticResult, ...]:
        del supplied
        results = _manual_pearson_vector(
            self,
            backend_identity=original_backend.__get__(self, type(self)),
            validity=Validity.VALID,
        )
        monkeypatch.setattr(
            StatisticResult,
            "validity",
            property(stateful_validity, set_validity),
        )
        return results

    monkeypatch.setattr(_BoundLaggedPearsonAdapter, "evaluate_all", manual_valid)
    outcome = _prepare_calibration(valid_observed(), resolution())
    monkeypatch.setattr(StatisticResult, "validity", original_validity)
    assert type(outcome) is calibration_v2._PreparedCalibration
    assert all(
        snapshot[4] is Validity.VALID
        for snapshot in calibration_v2._result_vector_snapshot(
            outcome.raw_observed_results,
            subject="retained raw observed vector",
        )
    )


def test_prepared_return_rechecks_scored_results_after_the_last_live_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_backend = _BoundLaggedPearsonAdapter.backend_identity
    captured_scored: list[tuple[StatisticResult, ...]] = []
    reads = 0

    def manual_valid(
        self: _BoundLaggedPearsonAdapter,
        supplied: SeriesPair,
        /,
    ) -> tuple[StatisticResult, ...]:
        del supplied
        return _manual_pearson_vector(
            self,
            backend_identity=original_backend.__get__(self, type(self)),
            validity=Validity.VALID,
        )

    def capture_selector(*args: object, **kwargs: object) -> object:
        selection, scored = real_select_family_v2(*args, **kwargs)  # type: ignore[arg-type]
        captured_scored.append(scored)
        return selection, scored

    def stateful_backend(self: _BoundLaggedPearsonAdapter) -> str:
        nonlocal reads
        reads += 1
        if reads == 6:
            object.__setattr__(captured_scored[0][0], "selection_score", 999.0)
        return original_backend.__get__(self, type(self))

    monkeypatch.setattr(_BoundLaggedPearsonAdapter, "evaluate_all", manual_valid)
    monkeypatch.setattr(calibration_v2, "select_family_v2", capture_selector)
    monkeypatch.setattr(
        _BoundLaggedPearsonAdapter,
        "backend_identity",
        property(stateful_backend),
    )
    with pytest.raises(V2IntegrityError, match=r"scored|selection|snapshot|weld"):
        _prepare_calibration(valid_observed(), resolution())


def test_post_bind_missing_plan_field_is_normalized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact = resolution()
    original_bind = LaggedPearsonAdapter.bind

    def deleting_bind(
        self: LaggedPearsonAdapter,
        observed_pair: SeriesPair,
        candidates: tuple[int, ...],
    ) -> object:
        bound = original_bind(self, observed_pair, candidates)
        object.__delattr__(exact.plan, "statistic_name")
        return bound

    monkeypatch.setattr(LaggedPearsonAdapter, "bind", deleting_bind)
    with pytest.raises(V2IntegrityError, match=r"plan|resolution|snapshot"):
        _prepare_calibration(valid_observed(), exact)


def test_post_bind_invalid_binned_scalar_is_normalized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact = binned_resolution()
    original_bind = BinnedNetTEAdapter.bind

    def invalid_bind(
        self: BinnedNetTEAdapter,
        observed_pair: SeriesPair,
        candidates: tuple[int, ...],
    ) -> object:
        bound = original_bind(self, observed_pair, candidates)
        object.__setattr__(bound, "bins", "x")
        return bound

    monkeypatch.setattr(BinnedNetTEAdapter, "bind", invalid_bind)
    with pytest.raises(V2IntegrityError, match=r"statistic|execution|snapshot"):
        _prepare_calibration(binned_observed(), exact)


def test_prepared_null_identity_field_drift_reaches_the_live_comparison() -> None:
    prepared = _prepare_calibration(valid_observed(), resolution())
    assert type(prepared) is calibration_v2._PreparedCalibration
    object.__setattr__(prepared.null_snapshot, "name", "foreign_null_v2")
    with pytest.raises(V2IntegrityError, match=r"null.*drift"):
        require_preparer_owned(prepared)


def test_prepared_final_signature_is_compared_with_the_registered_signature() -> None:
    prepared = _prepare_calibration(valid_observed(), resolution())
    assert type(prepared) is calibration_v2._PreparedCalibration
    diagnostics = ("jointly-drifted",)
    object.__setattr__(prepared, "diagnostics", diagnostics)
    object.__setattr__(prepared.null_bind_snapshot, "diagnostics", diagnostics)
    with pytest.raises(V2IntegrityError, match=r"snapshot.*drift"):
        require_preparer_owned(prepared)


@pytest.mark.parametrize("role", ["statistic", "null"])
def test_registered_bound_snapshot_normalizes_missing_plan_fields(role: str) -> None:
    exact = resolution()
    observed = valid_observed()
    if role == "statistic":
        bound = exact.adapters.statistic.bind(observed, exact.plan.candidates)
        object.__delattr__(exact.plan, "statistic_name")
        with pytest.raises(V2IntegrityError, match="registry snapshot"):
            resolution_v2._snapshot_registered_bound_statistic_v2(
                exact,
                bound,
                expected_pair=observed,
            )
        return

    bind = exact.adapters.null_model.bind(
        observed,
        semantic_input_sha256=semantic_input_sha256(observed, exact.plan.candidates),
        scientific_plan_sha256=scientific_plan_v2_sha256(exact.plan),
    )
    assert bind.bound is not None
    object.__delattr__(exact.plan, "null_name")
    with pytest.raises(V2IntegrityError, match="registry snapshot"):
        resolution_v2._snapshot_registered_bound_null_v2(
            exact,
            bind.bound,
            expected_pair=observed,
        )


@pytest.mark.parametrize("role", ["statistic", "null"])
@pytest.mark.parametrize(
    "registration_change",
    ["in_place_snapshotter", "entry_clone", "entry_snapshotter_and_bound_type"],
)
def test_prepared_gate_uses_resolver_frozen_bound_registration_metadata(
    role: str,
    registration_change: str,
) -> None:
    prepared = _prepare_calibration(valid_observed(), resolution())
    assert type(prepared) is calibration_v2._PreparedCalibration
    if role == "statistic":
        registrations = resolution_v2._STATISTIC_REGISTRATIONS_V2
        name = prepared.resolution.plan.statistic_name
        field = "bound_statistic_snapshot"
        bound = prepared.bound_statistic
        if registration_change != "entry_clone":
            object.__setattr__(bound, "observed_length", bound.observed_length + 1)
    else:
        registrations = resolution_v2._NULL_REGISTRATIONS_V2
        name = prepared.resolution.plan.null_name
        field = "bound_null_snapshot"
        bound = prepared.bound_null
        if registration_change != "entry_clone":
            object.__setattr__(bound, "_min_shift", bound._min_shift + 1)  # type: ignore[attr-defined]
    registration = registrations[name]
    original_snapshotter = getattr(registration, field)
    referents = gc.get_referents(registrations)
    assert len(referents) == 1 and type(referents[0]) is dict
    backing = referents[0]

    def replay_initial(
        *args: object,
        **kwargs: object,
    ) -> object:
        del kwargs
        initial = args[-1]
        assert initial is not None
        return initial

    replacement = replace(
        registration,
        bound_implementation_type=(
            type(bound)
            if registration_change == "entry_snapshotter_and_bound_type"
            else registration.bound_implementation_type
        ),
        bound_statistic_snapshot=(
            replay_initial
            if role == "statistic"
            and registration_change == "entry_snapshotter_and_bound_type"
            else registration.bound_statistic_snapshot
        ),
        bound_null_snapshot=(
            replay_initial
            if role == "null"
            and registration_change == "entry_snapshotter_and_bound_type"
            else registration.bound_null_snapshot
        ),
    )
    changed_in_place = False
    try:
        if registration_change == "in_place_snapshotter":
            try:
                object.__setattr__(registration, field, replay_initial)
                changed_in_place = True
            except AttributeError:
                pass
        else:
            backing[name] = replacement
        with pytest.raises(V2IntegrityError, match=r"registration|resolution|drift"):
            require_preparer_owned(prepared)
    finally:
        if changed_in_place:
            object.__setattr__(registration, field, original_snapshotter)
        if registration_change != "in_place_snapshotter":
            backing[name] = registration


@pytest.mark.parametrize("role", ["statistic", "null"])
def test_resolver_rejects_registration_drift_from_live_identity_getter(
    role: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if role == "statistic":
        registration = resolution_v2._STATISTIC_REGISTRATIONS_V2[
            "lagged_pearson_v1"
        ]
        field = "bound_statistic_snapshot"
        adapter_type = LaggedPearsonAdapter
        expected_name = "lagged_pearson_v1"
    else:
        registration = resolution_v2._NULL_REGISTRATIONS_V2["circular_shift_v2"]
        field = "bound_null_snapshot"
        adapter_type = CircularShiftNullV2
        expected_name = "circular_shift_v2"
    original_snapshotter = getattr(registration, field)

    def replay_after_drift(
        bound: object,
        expected_pair: object,
        expected_parameters: object,
        initial: object,
    ) -> object:
        if initial is None:
            assert original_snapshotter is not None
            snapshot = original_snapshotter(
                bound,
                expected_pair,
                expected_parameters,
                initial,
            )
            field_name = "observed_length" if role == "statistic" else "_min_shift"
            object.__setattr__(bound, field_name, getattr(bound, field_name) + 1)
            return snapshot
        return initial

    def drifting_name(self: object) -> str:
        del self
        object.__setattr__(registration, field, replay_after_drift)
        return expected_name

    monkeypatch.setattr(adapter_type, "name", property(drifting_name))
    try:
        with pytest.raises(V2IntegrityError, match=r"registration|metadata|drift"):
            resolution()
    finally:
        object.__setattr__(registration, field, original_snapshotter)


@pytest.mark.parametrize("role", ["statistic", "null"])
def test_resolver_rejects_registration_deletion_from_live_identity_getter(
    role: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if role == "statistic":
        registrations = resolution_v2._STATISTIC_REGISTRATIONS_V2
        name = "lagged_pearson_v1"
        adapter_type = LaggedPearsonAdapter
    else:
        registrations = resolution_v2._NULL_REGISTRATIONS_V2
        name = "circular_shift_v2"
        adapter_type = CircularShiftNullV2
    referents = gc.get_referents(registrations)
    assert len(referents) == 1 and type(referents[0]) is dict
    backing = referents[0]
    registration = backing[name]

    def deleting_name(self: object) -> str:
        del self
        backing.pop(name, None)
        return name

    monkeypatch.setattr(adapter_type, "name", property(deleting_name))
    try:
        with pytest.raises(V2IntegrityError, match=r"registered.*metadata"):
            resolution()
    finally:
        backing[name] = registration


def test_registration_freezer_rejects_nonexact_and_invalid_entries() -> None:
    registration = resolution_v2._STATISTIC_REGISTRATIONS_V2[
        "lagged_pearson_v1"
    ]
    with pytest.raises(V2IntegrityError, match="exact registry entry"):
        resolution_v2._freeze_adapter_registration(object(), subject="registration")
    with pytest.raises(V2IntegrityError, match="metadata is invalid"):
        resolution_v2._freeze_adapter_registration(
            replace(registration, factory=object()),
            subject="registration",
        )


def test_canonical_registration_lookup_rejects_an_unknown_name() -> None:
    registration = resolution_v2._STATISTIC_REGISTRATIONS_V2[
        "lagged_pearson_v1"
    ]
    with pytest.raises(V2IntegrityError, match="no canonical"):
        resolution_v2._require_canonical_adapter_registration(
            "unknown_statistic_v2",
            registration,
            role="statistic",
        )


@pytest.mark.parametrize("role", ["statistic", "null"])
@pytest.mark.parametrize("stage", ["resolve", "live", "pure", "prepared"])
def test_v2_registry_factory_identity_is_bound_to_module_initialization(
    role: str,
    stage: str,
) -> None:
    registry = (
        resolution_v2._STATISTIC_REGISTRY_V2
        if role == "statistic"
        else resolution_v2._NULL_REGISTRY_V2
    )
    name = "lagged_pearson_v1" if role == "statistic" else "circular_shift_v2"
    referents = gc.get_referents(registry._factories)
    assert len(referents) == 1 and type(referents[0]) is dict
    backing = referents[0]
    original_factory = backing[name]
    exact = None
    prepared = None
    if stage != "resolve":
        exact = resolution()
        if stage == "prepared":
            prepared = _prepare_calibration(valid_observed(), exact)

    def exact_builtin_wrapper(parameters: object) -> object:
        return original_factory(parameters)

    backing[name] = exact_builtin_wrapper
    try:
        with pytest.raises(V2IntegrityError, match=r"registry|factory|drift"):
            if stage == "resolve":
                resolution()
            elif stage == "live":
                resolution_v2._require_resolver_owned_resolution_v2(exact)
            elif stage == "pure":
                resolution_v2._require_resolver_owned_resolution_v2_pure(exact)
            else:
                require_preparer_owned(prepared)
    finally:
        backing[name] = original_factory


@pytest.mark.parametrize("role", ["statistic", "null"])
@pytest.mark.parametrize("stage", ["resolve", "pure"])
@pytest.mark.parametrize(
    "mutation",
    ["registry_object", "mapping_object", "entry_added", "entry_deleted"],
)
def test_v2_registry_structure_is_bound_to_module_initialization(
    role: str,
    stage: str,
    mutation: str,
) -> None:
    registry_name = (
        "_STATISTIC_REGISTRY_V2" if role == "statistic" else "_NULL_REGISTRY_V2"
    )
    registry = getattr(resolution_v2, registry_name)
    name = "lagged_pearson_v1" if role == "statistic" else "circular_shift_v2"
    original_mapping = registry._factories
    referents = gc.get_referents(original_mapping)
    assert len(referents) == 1 and type(referents[0]) is dict
    backing = referents[0]
    original_factory = backing[name]
    exact = resolution() if stage == "pure" else None
    try:
        if mutation == "registry_object":
            setattr(
                resolution_v2,
                registry_name,
                type(registry)(tuple(original_mapping.items())),
            )
        elif mutation == "mapping_object":
            object.__setattr__(registry, "_factories", MappingProxyType(dict(backing)))
        elif mutation == "entry_added":
            backing["foreign_v2"] = original_factory
        else:
            backing.pop(name)
        with pytest.raises(V2IntegrityError, match=r"registry|factory|drift"):
            if stage == "resolve":
                resolution()
            else:
                resolution_v2._require_resolver_owned_resolution_v2_pure(exact)
    finally:
        setattr(resolution_v2, registry_name, registry)
        object.__setattr__(registry, "_factories", original_mapping)
        backing.pop("foreign_v2", None)
        backing[name] = original_factory


@pytest.mark.parametrize("role", ["statistic", "null"])
@pytest.mark.parametrize("restore_during_factory", [False, True])
def test_resolver_rejects_a_foreign_registry_before_its_factory_runs(
    role: str,
    restore_during_factory: bool,
) -> None:
    registry_name = (
        "_STATISTIC_REGISTRY_V2" if role == "statistic" else "_NULL_REGISTRY_V2"
    )
    original_registry = getattr(resolution_v2, registry_name)
    name = "lagged_pearson_v1" if role == "statistic" else "circular_shift_v2"
    original_factory = original_registry._factories[name]
    factory_calls = 0

    def foreign_factory(parameters: object) -> object:
        nonlocal factory_calls
        factory_calls += 1
        if restore_during_factory:
            setattr(resolution_v2, registry_name, original_registry)
        return original_factory(parameters)

    setattr(
        resolution_v2,
        registry_name,
        type(original_registry)(((name, foreign_factory),)),
    )
    try:
        with pytest.raises(V2IntegrityError, match=r"registry|factory|drift"):
            resolution()
        assert factory_calls == 0
    finally:
        setattr(resolution_v2, registry_name, original_registry)


@pytest.mark.parametrize("role", ["statistic", "null"])
def test_foreign_registry_integrity_precedes_invalid_factory_parameters(
    role: str,
) -> None:
    registry_name = (
        "_STATISTIC_REGISTRY_V2" if role == "statistic" else "_NULL_REGISTRY_V2"
    )
    original_registry = getattr(resolution_v2, registry_name)
    name = "lagged_pearson_v1" if role == "statistic" else "circular_shift_v2"
    original_factory = original_registry._factories[name]
    factory_calls = 0

    def foreign_factory(parameters: object) -> object:
        nonlocal factory_calls
        factory_calls += 1
        return original_factory(parameters)

    setattr(
        resolution_v2,
        registry_name,
        type(original_registry)(((name, foreign_factory),)),
    )
    request = PlanRequestV2(
        candidates=(1, 2),
        statistic_name="lagged_pearson_v1",
        statistic_params={"bad": 1} if role == "statistic" else {},
        selection_rule="max_upper",
        null_name="circular_shift_v2",
        null_params={"bad": 1} if role == "null" else {"min_shift": 1},
        replicates=5,
        alpha=0.05,
        tie_tolerance=1e-12,
        root_seed=17,
    )
    try:
        with pytest.raises(V2IntegrityError, match=r"registry|factory|drift"):
            resolution_v2.resolve_plan_v2(request)
        assert factory_calls == 0
    finally:
        setattr(resolution_v2, registry_name, original_registry)


def test_registry_prebuild_gate_rejects_an_unknown_foreign_factory_without_calling_it(
) -> None:
    original_registry = resolution_v2._STATISTIC_REGISTRY_V2
    factory_calls = 0

    def foreign_factory(parameters: object) -> object:
        del parameters
        nonlocal factory_calls
        factory_calls += 1
        return object()

    resolution_v2._STATISTIC_REGISTRY_V2 = type(original_registry)(
        (("unknown_statistic_v2", foreign_factory),)
    )
    try:
        with pytest.raises(V2IntegrityError, match=r"registry|factory|drift"):
            resolution_v2.resolve_plan_v2(
                PlanRequestV2(
                    candidates=(1, 2),
                    statistic_name="unknown_statistic_v2",
                    statistic_params={},
                    selection_rule="max_upper",
                    null_name="circular_shift_v2",
                    null_params={"min_shift": 1},
                    replicates=5,
                    alpha=0.05,
                    tie_tolerance=1e-12,
                    root_seed=17,
                )
            )
        assert factory_calls == 0
    finally:
        resolution_v2._STATISTIC_REGISTRY_V2 = original_registry


def test_verified_adapter_helpers_require_the_exact_registered_types() -> None:
    with pytest.raises(V2IntegrityError, match="exact registered implementation"):
        resolution_v2._verified_statistic(
            object(),
            expected_name="lagged_pearson_v1",
            expected_params=MappingProxyType({}),
        )
    with pytest.raises(V2IntegrityError, match="exact registered implementation"):
        resolution_v2._verified_null(
            object(),
            expected_name="circular_shift_v2",
            expected_params=MappingProxyType({"min_shift": 1}),
        )


def test_registry_freezer_types_invalid_registry_structures() -> None:
    with pytest.raises(V2IntegrityError, match="exact AdapterRegistry"):
        resolution_v2._freeze_adapter_registry(object(), subject="registry")

    registry = resolution_v2._STATISTIC_REGISTRY_V2
    original_mapping = registry._factories
    original_factory = original_mapping["lagged_pearson_v1"]
    invalid_mappings = (
        {},
        MappingProxyType({"valid_v2": original_factory, 1: original_factory}),
        MappingProxyType({"valid_v2": object()}),
    )
    expected_messages = (
        "exact mapping proxy",
        "snapshot is invalid",
        "snapshot is invalid",
    )
    try:
        for invalid_mapping, expected_message in zip(
            invalid_mappings,
            expected_messages,
            strict=True,
        ):
            object.__setattr__(registry, "_factories", invalid_mapping)
            with pytest.raises(V2IntegrityError, match=expected_message):
                resolution_v2._freeze_adapter_registry(registry, subject="registry")
    finally:
        object.__setattr__(registry, "_factories", original_mapping)


def test_pure_resolution_gate_compares_frozen_registry_snapshot() -> None:
    exact = resolution()
    key = id(exact)
    original_record = _ownership_records()[key]
    _, registered = original_record
    replacement_identity = registered.statistic_registry_identity._replace(factories=())
    replacement = registered._replace(statistic_registry_identity=replacement_identity)
    _ownership_records()[key] = original_record._replace(
        snapshot=replacement,
    )
    try:
        with pytest.raises(V2IntegrityError, match="adapter registry snapshot drifted"):
            resolution_v2._require_resolver_owned_resolution_v2_pure(exact)
    finally:
        _ownership_records()[key] = original_record


@pytest.mark.parametrize("role", ["statistic", "null"])
def test_resolution_capture_rejects_a_missing_bound_registration(
    role: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact = resolution()
    validated = resolution_v2._validate_resolution_fields(exact)
    monkeypatch.setattr(
        resolution_v2,
        "_validate_resolution_fields",
        lambda value: validated,
    )
    registrations = (
        resolution_v2._STATISTIC_REGISTRATIONS_V2
        if role == "statistic"
        else resolution_v2._NULL_REGISTRATIONS_V2
    )
    name = (
        exact.plan.statistic_name if role == "statistic" else exact.plan.null_name
    )
    referents = gc.get_referents(registrations)
    assert len(referents) == 1 and type(referents[0]) is dict
    backing = referents[0]
    registration = backing.pop(name)
    try:
        with pytest.raises(V2IntegrityError, match="registration snapshot"):
            resolution_v2._capture_resolution_snapshot_fail_closed(exact)
    finally:
        backing[name] = registration


@pytest.mark.parametrize("role", ["statistic", "null"])
def test_resolution_capture_rejects_incomplete_bound_registration(role: str) -> None:
    exact = resolution()
    registrations = (
        resolution_v2._STATISTIC_REGISTRATIONS_V2
        if role == "statistic"
        else resolution_v2._NULL_REGISTRATIONS_V2
    )
    name = (
        exact.plan.statistic_name if role == "statistic" else exact.plan.null_name
    )
    referents = gc.get_referents(registrations)
    assert len(referents) == 1 and type(referents[0]) is dict
    backing = referents[0]
    registration = backing[name]
    field = "bound_statistic_snapshot" if role == "statistic" else "bound_null_snapshot"
    backing[name] = replace(registration, **{field: None})
    try:
        with pytest.raises(V2IntegrityError, match=r"registration|metadata"):
            resolution_v2._capture_resolution_snapshot_fail_closed(exact)
    finally:
        backing[name] = registration


@pytest.mark.parametrize("role", ["statistic", "null"])
def test_pure_resolution_gate_rejects_replaced_bound_registration(role: str) -> None:
    exact = resolution()
    registrations = (
        resolution_v2._STATISTIC_REGISTRATIONS_V2
        if role == "statistic"
        else resolution_v2._NULL_REGISTRATIONS_V2
    )
    name = (
        exact.plan.statistic_name if role == "statistic" else exact.plan.null_name
    )
    referents = gc.get_referents(registrations)
    assert len(referents) == 1 and type(referents[0]) is dict
    backing = referents[0]
    registration = backing[name]
    backing[name] = replace(registration)
    try:
        with pytest.raises(V2IntegrityError, match=r"registration.*drift"):
            resolution_v2._require_resolver_owned_resolution_v2_pure(exact)
    finally:
        backing[name] = registration


@pytest.mark.parametrize("role", ["statistic", "null"])
def test_pure_resolution_gate_compares_frozen_registration_entry_identity(
    role: str,
) -> None:
    exact = resolution()
    key = id(exact)
    original_record = _ownership_records()[key]
    _, registered = original_record
    field = f"{role}_registration"
    original_entry = getattr(registered, field)
    replacement = registered._replace(**{field: replace(original_entry)})
    _ownership_records()[key] = original_record._replace(
        snapshot=replacement,
    )
    try:
        with pytest.raises(V2IntegrityError, match="bound registration snapshot drifted"):
            resolution_v2._require_resolver_owned_resolution_v2_pure(exact)
    finally:
        _ownership_records()[key] = original_record


def test_execution_snapshot_comparison_is_type_sensitive() -> None:
    assert not resolution_v2._exact_snapshot_value_matches(
        (1, (2,)),
        (np.int64(1), (2,)),
    )


class _SnapshotStrSubclass(str):
    pass


class _SnapshotBytesSubclass(bytes):
    pass


class _SnapshotTupleSubclass(tuple[object, ...]):
    pass


@pytest.mark.parametrize(
    ("component", "field", "replacement_kind"),
    [
        ("statistic", "name", "str"),
        ("statistic", "parameters_bytes", "bytes"),
        ("statistic", "candidates", "tuple"),
        ("statistic", "backend_identity", "str"),
        ("statistic", "preprocessing_identity", "str"),
        ("null", "name", "str"),
        ("null", "parameters_bytes", "bytes"),
        ("null", "observed_length", "numpy_int"),
        ("null", "total_state_count", "numpy_int"),
        ("null", "null_parameter_sha256", "str"),
        ("null", "bound_null_owner_sha256", "str"),
        ("null", "semantic_input_sha256", "str"),
        ("null", "scientific_plan_sha256", "str"),
        ("null_bind", "diagnostics", "tuple"),
    ],
)
def test_preparer_ownership_rejects_equal_nonexact_snapshot_field_types(
    component: str,
    field: str,
    replacement_kind: str,
) -> None:
    prepared = _prepare_calibration(valid_observed(), resolution())
    assert type(prepared) is calibration_v2._PreparedCalibration
    target = {
        "statistic": prepared.statistic_snapshot,
        "null": prepared.null_snapshot,
        "null_bind": prepared.null_bind_snapshot,
    }[component]
    current = getattr(target, field)
    if replacement_kind == "str":
        replacement: object = _SnapshotStrSubclass(current)
    elif replacement_kind == "bytes":
        replacement = _SnapshotBytesSubclass(current)
    elif replacement_kind == "tuple":
        replacement = _SnapshotTupleSubclass(current)
    elif replacement_kind == "numpy_int":
        replacement = np.int64(current)
    else:
        raise AssertionError(f"unknown replacement kind: {replacement_kind}")
    assert replacement == current
    assert type(replacement) is not type(current)
    object.__setattr__(target, field, replacement)
    with pytest.raises(V2IntegrityError, match=r"preparer|prepared|snapshot|exact|drift"):
        require_preparer_owned(prepared)


@pytest.mark.parametrize(
    ("path", "trigger_read", "attack"),
    [
        ("disabled", 2, "plan"),
        ("analytical_failure", 4, "plan"),
        ("prepared", 6, "plan"),
        ("prepared", 6, "plan_parameters_type"),
        ("prepared", 6, "plan_candidates_type"),
        ("prepared", 6, "adapter_association"),
        ("prepared", 6, "unbound_statistic_parameters"),
        ("prepared", 6, "unbound_statistic_cycle"),
        ("prepared", 6, "unbound_null_parameter"),
        ("prepared", 6, "bound_parameters"),
    ],
)
def test_last_live_bound_hook_cannot_escape_the_final_active_state_checkpoint(
    path: str,
    trigger_read: int,
    attack: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if path == "disabled":
        exact = resolve_plan_v2(
            PlanRequestV2(
                candidates=(1, 3),
                statistic_name="lagged_pearson_v1",
                statistic_params={},
                selection_rule="max_upper",
                null_name="circular_shift_v2",
                null_params={"min_shift": 3},
                replicates=5,
                alpha=0.05,
                tie_tolerance=1e-12,
                root_seed=17,
            )
        )
        observed = pair([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
    else:
        exact = resolution()
        observed = valid_observed()

    original_backend = _BoundLaggedPearsonAdapter.backend_identity
    if path != "disabled":
        validity = (
            Validity.ANALYTIC_FAILURE
            if path == "analytical_failure"
            else Validity.VALID
        )

        def manual_vector(
            self: _BoundLaggedPearsonAdapter,
            supplied: SeriesPair,
            /,
        ) -> tuple[StatisticResult, ...]:
            del supplied
            return _manual_pearson_vector(
                self,
                backend_identity=original_backend.__get__(self, type(self)),
                validity=validity,
            )

        monkeypatch.setattr(_BoundLaggedPearsonAdapter, "evaluate_all", manual_vector)

    reads = 0

    def stateful_backend(self: _BoundLaggedPearsonAdapter) -> str:
        nonlocal reads
        reads += 1
        if reads == trigger_read:
            if attack == "plan":
                object.__setattr__(exact.plan, "replicates", 999)
            elif attack == "plan_parameters_type":
                object.__setattr__(exact.plan, "statistic_params", {})
            elif attack == "plan_candidates_type":
                object.__setattr__(
                    exact.plan,
                    "candidates",
                    _SnapshotTupleSubclass(exact.plan.candidates),
                )
            elif attack == "adapter_association":
                object.__setattr__(exact.adapters, "statistic", object())
            elif attack == "unbound_statistic_parameters":
                object.__setattr__(
                    exact.adapters.statistic,
                    "_parameters",
                    MappingProxyType({"evil": 1}),
                )
            elif attack == "unbound_statistic_cycle":
                cyclic: dict[str, object] = {}
                cyclic_proxy = MappingProxyType(cyclic)
                cyclic["cycle"] = cyclic_proxy
                object.__setattr__(
                    exact.adapters.statistic,
                    "_parameters",
                    cyclic_proxy,
                )
            elif attack == "unbound_null_parameter":
                object.__setattr__(exact.adapters.null_model, "min_shift", 2)
            elif attack == "bound_parameters":
                object.__setattr__(self, "_parameters", MappingProxyType({"evil": 1}))
            else:
                raise AssertionError(f"unknown active-state attack: {attack}")
        return original_backend.__get__(self, type(self))

    monkeypatch.setattr(
        _BoundLaggedPearsonAdapter,
        "backend_identity",
        property(stateful_backend),
    )
    with pytest.raises(V2IntegrityError, match=r"plan|resolution|adapter|statistic|drift"):
        _prepare_calibration(observed, exact)


def test_final_checkpoint_does_not_invoke_a_replaced_bound_slot_descriptor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact = resolve_plan_v2(
        PlanRequestV2(
            candidates=(1, 3),
            statistic_name="lagged_pearson_v1",
            statistic_params={},
            selection_rule="max_upper",
            null_name="circular_shift_v2",
            null_params={"min_shift": 3},
            replicates=5,
            alpha=0.05,
            tie_tolerance=1e-12,
            root_seed=17,
        )
    )
    observed = pair([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
    original_backend = _BoundLaggedPearsonAdapter.backend_identity
    original_length = inspect.getattr_static(
        _BoundLaggedPearsonAdapter,
        "observed_length",
    )
    backend_reads = 0
    length_reads = 0

    def stateful_length(self: _BoundLaggedPearsonAdapter) -> int:
        nonlocal length_reads
        length_reads += 1
        value = original_length.__get__(self, type(self))
        if length_reads == 3:
            object.__setattr__(exact.plan, "replicates", 999)
        return value

    def stateful_backend(self: _BoundLaggedPearsonAdapter) -> str:
        nonlocal backend_reads
        backend_reads += 1
        if backend_reads == 1:
            monkeypatch.setattr(
                _BoundLaggedPearsonAdapter,
                "observed_length",
                property(stateful_length),
            )
        return original_backend.__get__(self, type(self))

    monkeypatch.setattr(
        _BoundLaggedPearsonAdapter,
        "backend_identity",
        property(stateful_backend),
    )
    with pytest.raises(V2IntegrityError, match=r"descriptor|slot|statistic|snapshot"):
        _prepare_calibration(observed, exact)


@pytest.mark.parametrize("slot_name", ["source", "target"])
@pytest.mark.parametrize("path", ["disabled", "analytical_failure"])
def test_final_checkpoint_does_not_invoke_a_replaced_input_pair_slot_descriptor(
    path: str,
    slot_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if path == "disabled":
        exact = resolve_plan_v2(
            PlanRequestV2(
                candidates=(1, 3),
                statistic_name="lagged_pearson_v1",
                statistic_params={},
                selection_rule="max_upper",
                null_name="circular_shift_v2",
                null_params={"min_shift": 3},
                replicates=5,
                alpha=0.05,
                tie_tolerance=1e-12,
                root_seed=17,
            )
        )
        observed = pair([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])
    else:
        exact = resolution()
        observed = valid_observed()

    original_backend = _BoundLaggedPearsonAdapter.backend_identity
    original_slot = inspect.getattr_static(SeriesPair, slot_name)
    if path == "analytical_failure":

        def manual_vector(
            self: _BoundLaggedPearsonAdapter,
            supplied: SeriesPair,
            /,
        ) -> tuple[StatisticResult, ...]:
            del supplied
            return _manual_pearson_vector(
                self,
                backend_identity=original_backend.__get__(self, type(self)),
                validity=Validity.ANALYTIC_FAILURE,
            )

        monkeypatch.setattr(_BoundLaggedPearsonAdapter, "evaluate_all", manual_vector)

    backend_reads = 0
    slot_reads = 0
    attacked = False

    def stateful_slot(self: SeriesPair) -> np.ndarray:
        nonlocal attacked, slot_reads
        slot_reads += 1
        value = original_slot.__get__(self, type(self))
        if not attacked and any(
            frame.function == "_reverify_final_active_state"
            for frame in inspect.stack()
        ):
            object.__setattr__(exact.plan, "replicates", 999)
            attacked = True
        return value

    def stateful_backend(self: _BoundLaggedPearsonAdapter) -> str:
        nonlocal backend_reads
        backend_reads += 1
        install_read = 1 if path == "disabled" else 4
        if backend_reads == install_read:
            monkeypatch.setattr(SeriesPair, slot_name, property(stateful_slot))
        return original_backend.__get__(self, type(self))

    monkeypatch.setattr(
        _BoundLaggedPearsonAdapter,
        "backend_identity",
        property(stateful_backend),
    )
    try:
        outcome = _prepare_calibration(observed, exact)
    except V2IntegrityError:
        return
    assert attacked
    pytest.fail(
        "replaced SeriesPair slot escaped final checkpoint: "
        f"{path=}, {type(outcome).__name__=}, {backend_reads=}, "
        f"{slot_name=}, {slot_reads=}, live_replicates={exact.plan.replicates}"
    )


def test_forged_block_bound_over_the_shared_limit_fails_before_factorial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    block_count = block_shuffle_v2._MAX_BLOCK_COUNT + 1
    block_length = 2
    observed_length = block_count * block_length
    observed = SeriesPair(
        source=np.arange(observed_length, dtype=np.float64),
        target=np.arange(observed_length - 1, -1, -1, dtype=np.float64),
    )
    exact = resolution_with_null("block_shuffle_v2", {"block_length": block_length})
    semantic_digest = semantic_input_sha256(observed, exact.plan.candidates)
    plan_digest = scientific_plan_v2_sha256(exact.plan)
    parameter_sha256 = exact.adapters.null_model.null_parameter_sha256  # type: ignore[attr-defined]
    bound = object.__new__(_BoundBlockShuffleV2)
    values = {
        "_observed_pair": observed,
        "_block_length": block_length,
        "_block_count": block_count,
        "_semantic_input_sha256": semantic_digest,
        "_scientific_plan_sha256": plan_digest,
        "_null_parameter_sha256": parameter_sha256,
        "_bound_null_owner_sha256": owner_digest(
            semantic_input_sha256=semantic_digest,
            scientific_plan_sha256=plan_digest,
            null_parameter_sha256=parameter_sha256,
            observed_length=observed_length,
        ),
        "_observed_content_guard_sha256": _owned_pair_content_guard_sha256(
            observed
        ),
        "_parameters": MappingProxyType({"block_length": block_length}),
        "_observed_length": observed_length,
        "_total_state_count": 1,
    }
    for field, value in values.items():
        object.__setattr__(bound, field, value)
    forged_bind = NullBindResult(
        status=NullBindStatus.ENABLED,
        bound=bound,
        disabled_reason=None,
        diagnostics=(),
    )
    monkeypatch.setattr(
        type(exact.adapters.null_model),
        "bind",
        lambda *args, **kwargs: forged_bind,
    )
    factorial_calls: list[int] = []

    def bomb_factorial(value: int) -> int:
        factorial_calls.append(value)
        raise AssertionError("factorial must not run above the shared block-count cap")

    monkeypatch.setattr(resolution_v2, "factorial", bomb_factorial)
    with pytest.raises(V2IntegrityError, match=r"block|null|supported|range"):
        try:
            _prepare_calibration(observed, exact)
        finally:
            assert factorial_calls == []


def _call_final_active_state(
    prepared: calibration_v2._PreparedCalibration,
    *,
    semantic_digest: str | None = None,
    plan_digest: str | None = None,
    bound_statistic: object | None = None,
    bound_null: object | None = None,
    null_snapshot: object | None = None,
) -> None:
    calibration_v2._reverify_final_active_state(
        prepared.observed_pair,
        prepared.resolution,
        semantic_digest=(
            prepared.semantic_input_sha256
            if semantic_digest is None
            else semantic_digest
        ),
        plan_digest=(
            prepared.scientific_plan_sha256 if plan_digest is None else plan_digest
        ),
        bound_statistic=(
            prepared.bound_statistic
            if bound_statistic is None
            else bound_statistic
        ),  # type: ignore[arg-type]
        statistic_snapshot=prepared.statistic_snapshot,
        bound_null=prepared.bound_null if bound_null is None else bound_null,  # type: ignore[arg-type]
        null_snapshot=(
            prepared.null_snapshot if null_snapshot is None else null_snapshot
        ),  # type: ignore[arg-type]
    )


def test_final_active_state_preserves_registered_snapshot_errors() -> None:
    prepared = _prepare_calibration(valid_observed(), resolution())
    assert type(prepared) is calibration_v2._PreparedCalibration
    with pytest.raises(V2IntegrityError, match=r"statistic|registered|exact"):
        _call_final_active_state(prepared, bound_statistic=object())


def test_final_active_state_normalizes_structural_snapshot_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepare_calibration(valid_observed(), resolution())
    assert type(prepared) is calibration_v2._PreparedCalibration
    monkeypatch.setattr(
        calibration_v2,
        "scientific_plan_v2_sha256",
        lambda value: (_ for _ in ()).throw(ValueError("structural")),
    )
    with pytest.raises(V2IntegrityError, match="final active-state snapshot"):
        _call_final_active_state(prepared)


@pytest.mark.parametrize("digest_role", ["plan", "semantic"])
def test_final_active_state_rejects_run_identity_mismatch(digest_role: str) -> None:
    prepared = _prepare_calibration(valid_observed(), resolution())
    assert type(prepared) is calibration_v2._PreparedCalibration
    kwargs = (
        {"plan_digest": "a" * 64}
        if digest_role == "plan"
        else {"semantic_digest": "a" * 64}
    )
    with pytest.raises(V2IntegrityError, match="final run identity"):
        _call_final_active_state(prepared, **kwargs)


def test_final_active_state_rejects_a_statistic_snapshot_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepare_calibration(valid_observed(), resolution())
    assert type(prepared) is calibration_v2._PreparedCalibration
    monkeypatch.setattr(
        calibration_v2,
        "_bound_statistic_execution_snapshot_matches",
        lambda current, initial: False,
    )
    with pytest.raises(V2IntegrityError, match="final bound statistic"):
        _call_final_active_state(prepared)


def test_final_active_state_rejects_an_unpaired_null_snapshot() -> None:
    prepared = _prepare_calibration(valid_observed(), resolution())
    assert type(prepared) is calibration_v2._PreparedCalibration
    with pytest.raises(V2IntegrityError, match="null snapshot association"):
        calibration_v2._reverify_final_active_state(
            prepared.observed_pair,
            prepared.resolution,
            semantic_digest=prepared.semantic_input_sha256,
            plan_digest=prepared.scientific_plan_sha256,
            bound_statistic=prepared.bound_statistic,
            statistic_snapshot=prepared.statistic_snapshot,
            bound_null=None,
            null_snapshot=prepared.null_snapshot,
        )


def test_final_active_state_preserves_null_registry_errors() -> None:
    prepared = _prepare_calibration(valid_observed(), resolution())
    assert type(prepared) is calibration_v2._PreparedCalibration
    with pytest.raises(V2IntegrityError, match=r"null|registered|exact"):
        _call_final_active_state(prepared, bound_null=object())


def test_final_active_state_normalizes_null_structural_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepare_calibration(valid_observed(), resolution())
    assert type(prepared) is calibration_v2._PreparedCalibration
    monkeypatch.setattr(
        calibration_v2,
        "_snapshot_registered_bound_null_v2",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("structural")),
    )
    with pytest.raises(V2IntegrityError, match="final bound null snapshot"):
        _call_final_active_state(prepared)


def test_final_active_state_rejects_a_null_snapshot_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepare_calibration(valid_observed(), resolution())
    assert type(prepared) is calibration_v2._PreparedCalibration
    monkeypatch.setattr(
        calibration_v2,
        "_bound_null_execution_snapshot_matches",
        lambda current, initial: False,
    )
    with pytest.raises(V2IntegrityError, match="final bound null snapshot drifted"):
        _call_final_active_state(prepared)


def test_prepared_signature_rejects_a_nonexact_null_bind_status() -> None:
    prepared = _prepare_calibration(valid_observed(), resolution())
    assert type(prepared) is calibration_v2._PreparedCalibration
    object.__setattr__(prepared.null_bind_snapshot, "status", object())
    with pytest.raises(V2IntegrityError, match="null bind status"):
        require_preparer_owned(prepared)


def test_ownership_capture_rechecks_its_final_nonlive_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepare_calibration(valid_observed(), resolution())
    assert type(prepared) is calibration_v2._PreparedCalibration
    original = calibration_v2._reverify_final_active_state

    def drift_after_final_checkpoint(*args: object, **kwargs: object) -> None:
        original(*args, **kwargs)  # type: ignore[arg-type]
        diagnostics = ("drifted-after-final-checkpoint",)
        object.__setattr__(prepared, "diagnostics", diagnostics)
        object.__setattr__(prepared.null_bind_snapshot, "diagnostics", diagnostics)

    monkeypatch.setattr(
        calibration_v2,
        "_reverify_final_active_state",
        drift_after_final_checkpoint,
    )
    with pytest.raises(V2IntegrityError, match="during ownership capture"):
        calibration_v2._capture_prepared_ownership(prepared)


def test_prepare_defensively_rejects_nonexact_terminal_plan_scalars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact = resolution()
    object.__setattr__(exact.plan, "replicates", True)
    monkeypatch.setattr(
        calibration_v2,
        "_require_resolver_owned_resolution_v2",
        lambda value: value,
    )
    with pytest.raises(V2IntegrityError, match="terminal plan scalars"):
        _prepare_calibration(valid_observed(), exact)


def test_bound_private_identity_rejects_a_missing_static_descriptor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact = resolution()
    observed = valid_observed()
    bound = exact.adapters.statistic.bind(observed, exact.plan.candidates)
    monkeypatch.delattr(_BoundLaggedPearsonAdapter, "backend_identity")
    with pytest.raises(V2IntegrityError, match="private identity snapshot"):
        resolution_v2._snapshot_registered_bound_statistic_v2(
            exact,
            bound,
            expected_pair=observed,
        )


@pytest.mark.parametrize("attack", ["nonproxy", "invalid_mapping"])
def test_bound_private_identity_rejects_invalid_private_parameters(attack: str) -> None:
    exact = resolution()
    observed = valid_observed()
    bound = exact.adapters.statistic.bind(observed, exact.plan.candidates)
    replacement: object = (
        {} if attack == "nonproxy" else MappingProxyType({object(): 1})
    )
    object.__setattr__(bound, "_parameters", replacement)
    with pytest.raises(V2IntegrityError, match=r"parameters|private"):
        resolution_v2._snapshot_registered_bound_statistic_v2(
            exact,
            bound,
            expected_pair=observed,
        )


def test_pure_resolution_gate_requires_an_exact_registered_resolution() -> None:
    with pytest.raises(V2IntegrityError, match="exact resolver-owned"):
        resolution_v2._require_resolver_owned_resolution_v2_pure(object())
    naked = object.__new__(resolution_v2.PlanResolutionV2)
    with pytest.raises(V2IntegrityError, match="not resolver-owned"):
        resolution_v2._require_resolver_owned_resolution_v2_pure(naked)


def test_pure_resolution_gate_normalizes_missing_storage() -> None:
    exact = resolution()
    object.__delattr__(exact, "adapters")
    with pytest.raises(V2IntegrityError, match=r"resolution adapters slot|pure resolution"):
        resolution_v2._require_resolver_owned_resolution_v2_pure(exact)


def test_pure_resolution_gate_rejects_replaced_plan_association() -> None:
    exact = resolution()
    foreign = resolution()
    object.__setattr__(exact, "plan", foreign.plan)
    with pytest.raises(V2IntegrityError, match="association was replaced"):
        resolution_v2._require_resolver_owned_resolution_v2_pure(exact)


def test_pure_resolution_gate_rejects_plan_byte_drift() -> None:
    exact = resolution()
    object.__setattr__(exact.plan, "replicates", 6)
    with pytest.raises(V2IntegrityError, match=r"pure identity snapshot drifted"):
        resolution_v2._require_resolver_owned_resolution_v2_pure(exact)


def test_exact_slot_reader_normalizes_descriptor_lookup_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        resolution_v2.inspect,
        "getattr_static",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("lookup")),
    )
    with pytest.raises(V2IntegrityError, match="descriptor is invalid"):
        resolution_v2._read_exact_slot(object(), "field", subject="test")


def test_exact_series_pair_reader_normalizes_descriptor_lookup_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        resolution_v2.inspect,
        "getattr_static",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("lookup")),
    )
    with pytest.raises(V2IntegrityError, match="slot descriptor is invalid"):
        resolution_v2._exact_series_pair_read_view(valid_observed())


def test_exact_series_pair_reader_normalizes_missing_slot_storage() -> None:
    observed = valid_observed()
    object.__delattr__(observed, "source")
    with pytest.raises(V2IntegrityError, match="slot snapshot is invalid"):
        resolution_v2._exact_series_pair_read_view(observed)


def test_input_snapshot_normalizes_structural_reader_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        calibration_v2,
        "_exact_series_pair_read_view",
        lambda value: (_ for _ in ()).throw(ValueError("reader")),
    )
    with pytest.raises(V2IntegrityError, match="input snapshot is invalid"):
        calibration_v2._snapshot_input_pair(valid_observed())


@pytest.mark.parametrize("stage", ["resolution", "semantic"])
def test_run_identity_normalizes_structural_checkpoint_errors(
    stage: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact = resolution()
    observed = valid_observed()
    semantic_digest = semantic_input_sha256(observed, exact.plan.candidates)
    plan_digest = scientific_plan_v2_sha256(exact.plan)
    target = (
        "_require_resolver_owned_resolution_v2"
        if stage == "resolution"
        else "_semantic_digest_for_exact_pair"
    )
    monkeypatch.setattr(
        calibration_v2,
        target,
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("checkpoint")),
    )
    with pytest.raises(V2IntegrityError, match="run identity snapshot is invalid"):
        _reverify_run_identity(
            observed,
            exact,
            semantic_digest=semantic_digest,
            plan_digest=plan_digest,
        )


def test_exact_mapping_proxy_rejects_a_nonproxy_container() -> None:
    with pytest.raises(V2IntegrityError, match="exact frozen mapping"):
        resolution_v2._exact_mapping_proxy_bytes({}, subject="mapping")


def test_exact_mapping_proxy_normalizes_canonicalization_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        resolution_v2,
        "canonical_json_bytes",
        lambda value: (_ for _ in ()).throw(ValueError("canonical")),
    )
    with pytest.raises(V2IntegrityError, match="mapping is invalid"):
        resolution_v2._exact_mapping_proxy_bytes(
            MappingProxyType({}),
            subject="mapping",
        )


@pytest.mark.parametrize("error_type", [RuntimeError, OSError, AssertionError])
def test_exact_mapping_proxy_nonstructural_errors_propagate(
    error_type: type[Exception],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = error_type("nonstructural")
    monkeypatch.setattr(
        resolution_v2,
        "freeze_exact_json_mapping",
        lambda *args, **kwargs: (_ for _ in ()).throw(error),
    )
    with pytest.raises(error_type, match="nonstructural"):
        resolution_v2._exact_mapping_proxy_bytes(
            MappingProxyType({}),
            subject="mapping",
        )


def test_pure_resolution_gate_normalizes_cyclic_adapter_parameters() -> None:
    exact = resolution()
    cyclic: dict[str, object] = {}
    cyclic_proxy = MappingProxyType(cyclic)
    cyclic["cycle"] = cyclic_proxy
    object.__setattr__(exact.adapters.statistic, "_parameters", cyclic_proxy)
    with pytest.raises(V2IntegrityError, match=r"parameters|mapping|identity"):
        resolution_v2._require_resolver_owned_resolution_v2_pure(exact)


@pytest.mark.parametrize("role", ["pearson", "binned", "circular", "block"])
def test_bound_snapshotter_wraps_structural_slot_reader_errors(
    role: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if role == "pearson":
        exact = resolution()
        observed = valid_observed()
        bound = exact.adapters.statistic.bind(observed, exact.plan.candidates)
        call = resolution_v2._pearson_bound_snapshot
        parameters = exact.plan.statistic_params
    elif role == "binned":
        exact = binned_resolution()
        observed = binned_observed()
        bound = exact.adapters.statistic.bind(observed, exact.plan.candidates)
        call = resolution_v2._binned_bound_snapshot
        parameters = exact.plan.statistic_params
    else:
        null_name = "circular_shift_v2" if role == "circular" else "block_shuffle_v2"
        null_params = {"min_shift": 1} if role == "circular" else {"block_length": 2}
        exact = resolution_with_null(null_name, null_params)
        observed = valid_observed()
        bind = exact.adapters.null_model.bind(
            observed,
            semantic_input_sha256=semantic_input_sha256(
                observed,
                exact.plan.candidates,
            ),
            scientific_plan_sha256=scientific_plan_v2_sha256(exact.plan),
        )
        assert bind.bound is not None
        bound = bind.bound
        call = (
            resolution_v2._circular_bound_snapshot
            if role == "circular"
            else resolution_v2._block_bound_snapshot
        )
        parameters = exact.plan.null_params
    monkeypatch.setattr(
        resolution_v2,
        "_read_exact_slot",
        lambda *args, **kwargs: (_ for _ in ()).throw(AttributeError("slot")),
    )
    with pytest.raises(V2IntegrityError, match="execution snapshot"):
        call(bound, observed, parameters, None)


def test_bound_null_common_snapshot_wraps_structural_private_identity_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact = resolution()
    observed = valid_observed()
    bind = exact.adapters.null_model.bind(
        observed,
        semantic_input_sha256=semantic_input_sha256(observed, exact.plan.candidates),
        scientific_plan_sha256=scientific_plan_v2_sha256(exact.plan),
    )
    assert bind.bound is not None
    monkeypatch.setattr(
        resolution_v2,
        "_bound_private_identity_snapshot",
        lambda *args, **kwargs: (_ for _ in ()).throw(AttributeError("identity")),
    )
    with pytest.raises(V2IntegrityError, match="bound null execution snapshot"):
        resolution_v2._snapshot_registered_bound_null_v2(
            exact,
            bind.bound,
            expected_pair=observed,
        )


def test_static_descriptor_snapshot_normalizes_lookup_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        resolution_v2.inspect,
        "getattr_static",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("descriptor")),
    )
    with pytest.raises(V2IntegrityError, match="descriptor snapshot"):
        resolution_v2._static_descriptor_snapshot(
            LaggedPearsonAdapter(),
            ("name",),
            subject="statistic",
        )


def test_pure_binned_adapter_rejects_an_invalid_scalar() -> None:
    adapter = BinnedNetTEAdapter(bins=3)
    object.__setattr__(adapter, "bins", True)
    with pytest.raises(V2IntegrityError, match="binned adapter scalar"):
        resolution_v2._pure_statistic_adapter_snapshot(adapter)


def test_pure_null_adapter_rejects_unknown_and_invalid_state() -> None:
    with pytest.raises(V2IntegrityError, match="no pure registered snapshot"):
        resolution_v2._pure_null_adapter_snapshot(object())
    adapter = resolution().adapters.null_model
    object.__setattr__(adapter, "min_shift", True)
    with pytest.raises(V2IntegrityError, match="null adapter scalar"):
        resolution_v2._pure_null_adapter_snapshot(adapter)


def test_pure_null_adapter_rejects_digest_drift() -> None:
    adapter = resolution().adapters.null_model
    object.__setattr__(adapter, "_null_parameter_sha256", "a" * 64)
    with pytest.raises(V2IntegrityError, match="parameter digest drifted"):
        resolution_v2._pure_null_adapter_snapshot(adapter)


def test_resolution_capture_rejects_live_and_pure_adapter_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact = resolution()
    original = resolution_v2._pure_statistic_adapter_snapshot

    def mismatch(adapter: object) -> tuple[object, ...]:
        implementation, parameters, _operational, descriptors = original(adapter)
        return implementation, parameters, (99,), descriptors

    monkeypatch.setattr(
        resolution_v2,
        "_pure_statistic_adapter_snapshot",
        mismatch,
    )
    with pytest.raises(V2IntegrityError, match="pure adapter identity"):
        resolution_v2._capture_resolution_snapshot(exact)


@pytest.mark.parametrize("role", ["plan", "adapters"])
def test_pure_resolution_gate_rejects_nonexact_components(role: str) -> None:
    exact = resolution()
    object.__setattr__(exact, role, object())
    with pytest.raises(V2IntegrityError, match=f"pure resolution {role}"):
        resolution_v2._require_resolver_owned_resolution_v2_pure(exact)


def test_pure_resolution_gate_rejects_nonexact_plan_scalars() -> None:
    exact = resolution()
    object.__setattr__(exact.plan, "replicates", np.int64(5))
    with pytest.raises(V2IntegrityError, match="plan scalar types"):
        resolution_v2._require_resolver_owned_resolution_v2_pure(exact)


def test_pure_resolution_gate_normalizes_plan_serialization_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact = resolution()
    monkeypatch.setattr(
        resolution_v2,
        "scientific_plan_v2_payload",
        lambda value: (_ for _ in ()).throw(ValueError("payload")),
    )
    with pytest.raises(V2IntegrityError, match="pure resolution identity"):
        resolution_v2._require_resolver_owned_resolution_v2_pure(exact)


def test_pure_resolution_gate_rejects_a_valid_but_replaced_adapter() -> None:
    exact = resolution()
    object.__setattr__(exact.adapters, "statistic", LaggedPearsonAdapter())
    with pytest.raises(V2IntegrityError, match="adapter identity was replaced"):
        resolution_v2._require_resolver_owned_resolution_v2_pure(exact)
