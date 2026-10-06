from __future__ import annotations

import builtins
import importlib
import itertools
import math
from collections.abc import Callable
from dataclasses import FrozenInstanceError, fields
from types import ModuleType
from typing import cast

import numpy as np
import pytest

from selcal.contracts import SeriesPair, StatisticResult, Validity
from selcal.contracts_v2 import (
    PlanRequestV2,
    ResourceLimitError,
    V2IntegrityError,
)
from selcal.resolution_v2 import PlanResolutionV2, resolve_plan_v2
from selcal.statistics.lagged_pearson import (
    LaggedPearsonAdapter,
    _BoundLaggedPearsonAdapter,
)


def _oracle_module() -> ModuleType:
    try:
        return importlib.import_module("selcal.exact_oracle_v0")
    except ModuleNotFoundError as error:
        pytest.fail(f"Task 9 exact oracle module is absent: {error}")


def _oracle() -> Callable[[SeriesPair, PlanResolutionV2], object]:
    candidate = getattr(_oracle_module(), "exact_state_oracle_v0", None)
    assert callable(candidate), "Task 9 must define exact_state_oracle_v0(pair, resolution, /)"
    return cast(Callable[[SeriesPair, PlanResolutionV2], object], candidate)


def _resolution(
    *,
    null_name: str = "circular_shift_v2",
    null_params: dict[str, int] | None = None,
    candidates: tuple[int, ...] = (1, 2),
) -> PlanResolutionV2:
    if null_params is None:
        null_params = {"min_shift": 1}
    return resolve_plan_v2(
        PlanRequestV2(
            candidates=candidates,
            statistic_name="lagged_pearson_v1",
            statistic_params={},
            selection_rule="max_upper",
            null_name=null_name,
            null_params=null_params,
            replicates=1,
            alpha=0.05,
            tie_tolerance=1e-12,
            root_seed=17,
        )
    )


def _pair(source: list[float], target: list[float]) -> SeriesPair:
    return SeriesPair(
        source=np.asarray(source, dtype=np.float64),
        target=np.asarray(target, dtype=np.float64),
    )


def test_v0_surface_is_internal_minimal_frozen_and_nonhashable() -> None:
    import selcal

    module = _oracle_module()
    result_type = getattr(module, "ExactOracleResultV0", None)

    assert module.MAX_EXACT_STATES_V0 == 100_000
    assert module.EXACT_NULL_NAMES_V0 == frozenset(
        {"circular_shift_v2", "block_shuffle_v2"}
    )
    assert result_type is not None
    assert tuple(field.name for field in fields(result_type)) == (
        "p_exact",
        "total_state_count",
        "nonidentity_exceedance_count",
    )
    result = result_type(
        p_exact=0.5,
        total_state_count=2,
        nonidentity_exceedance_count=0,
    )
    with pytest.raises(TypeError, match="unhashable"):
        hash(result)
    with pytest.raises(FrozenInstanceError):
        result.p_exact = 1.0
    assert not hasattr(selcal, "ExactOracleResultV0")
    assert not hasattr(selcal, "exact_state_oracle_v0")


def test_normative_circular_pearson_table_is_one_half_with_explicit_index_loops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _oracle_module()
    range_calls: list[tuple[int, ...]] = []

    def recording_range(*args: int) -> range:
        range_calls.append(args)
        return builtins.range(*args)

    monkeypatch.setattr(module, "range", recording_range, raising=False)
    pair = _pair(
        [0, 3, 1, 2, 4, 5],
        [0, 1, 3, 2, 5, 4],
    )

    result = _oracle()(pair, _resolution())

    assert result.total_state_count == 6
    assert result.nonidentity_exceedance_count == 2
    assert result.p_exact == 3 / 6 == 0.5
    assert range_calls.count((6,)) == 5


def test_scripted_full_reselection_is_three_quarters_not_fixed_candidate_half(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scans = iter(((3.0, 1.0), (2.0, 4.0), (3.0, 0.0), (1.0, 2.0)))
    calls: list[tuple[float, float]] = []

    def scripted_evaluate(
        bound: _BoundLaggedPearsonAdapter,
        supplied: SeriesPair,
        /,
    ) -> tuple[StatisticResult, ...]:
        del supplied
        scores = next(scans)
        calls.append(scores)
        return tuple(
            StatisticResult(
                candidate_id=candidate,
                estimate=score,
                selection_score=None,
                support_n=bound.observed_length - max(bound.candidates),
                validity=Validity.VALID,
                diagnostics=("scripted_exact_oracle_v0",),
                backend_identity=bound.backend_identity,
                preprocessing_identity=bound.preprocessing_identity,
            )
            for candidate, score in zip(bound.candidates, scores, strict=True)
        )

    monkeypatch.setattr(_BoundLaggedPearsonAdapter, "evaluate_all", scripted_evaluate)
    pair = _pair([0, 1, 2, 3], [3, 2, 1, 0])

    result = _oracle()(pair, _resolution())

    assert calls == [(3.0, 1.0), (2.0, 4.0), (3.0, 0.0), (1.0, 2.0)]
    assert result.nonidentity_exceedance_count == 2
    assert result.p_exact == 3 / 4
    fixed_observed_candidate_p = (1 + sum(row[0] >= calls[0][0] for row in calls[1:])) / 4
    assert fixed_observed_candidate_p == 1 / 2
    assert result.p_exact != fixed_observed_candidate_p


def test_block_oracle_uses_labelled_permutations_and_retains_numeric_multiplicity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _oracle_module()
    real_permutations = itertools.permutations
    permutation_calls: list[tuple[int, ...]] = []
    evaluated_sources: list[bytes] = []
    real_evaluate = _BoundLaggedPearsonAdapter.evaluate_all

    def recording_permutations(labels: object) -> object:
        exact_labels = tuple(cast(object, labels))  # type: ignore[arg-type]
        permutation_calls.append(cast(tuple[int, ...], exact_labels))
        return real_permutations(exact_labels)

    def recording_evaluate(
        bound: _BoundLaggedPearsonAdapter,
        supplied: SeriesPair,
        /,
    ) -> tuple[StatisticResult, ...]:
        evaluated_sources.append(supplied.source.tobytes(order="C"))
        return real_evaluate(bound, supplied)

    monkeypatch.setattr(module.itertools, "permutations", recording_permutations)
    monkeypatch.setattr(_BoundLaggedPearsonAdapter, "evaluate_all", recording_evaluate)
    pair = _pair(
        [0, 1, 0, 1, 2, 3],
        [0, 2, 1, 4, 3, 5],
    )
    resolution = _resolution(
        null_name="block_shuffle_v2",
        null_params={"block_length": 2},
    )

    result = _oracle()(pair, resolution)

    assert permutation_calls == [(0, 1, 2)]
    assert result.total_state_count == 6
    assert len(evaluated_sources) == 6
    assert len(set(evaluated_sources)) < len(evaluated_sources)


def test_circular_cap_fails_before_shift_enumeration_or_statistic_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _oracle_module()
    length = module.MAX_EXACT_STATES_V0 + 1
    pair = SeriesPair(
        source=np.arange(length, dtype=np.float64),
        target=np.arange(length, dtype=np.float64)[::-1],
    )
    resolution = _resolution(candidates=(1,))

    def explode(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("state enumeration or statistic binding happened before cap")

    monkeypatch.setattr(module, "range", explode, raising=False)
    monkeypatch.setattr(LaggedPearsonAdapter, "bind", explode)

    with pytest.raises(ResourceLimitError, match="EXACT_STATE_CAP_EXCEEDED"):
        _oracle()(pair, resolution)


def test_block_cap_is_incremental_and_precedes_permutations_and_statistic_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _oracle_module()
    pair = SeriesPair(
        source=np.arange(4096, dtype=np.float64),
        target=np.arange(4096, dtype=np.float64)[::-1],
    )
    resolution = _resolution(
        null_name="block_shuffle_v2",
        null_params={"block_length": 1},
        candidates=(1,),
    )

    def explode(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("factorial, permutations, or statistic binding preceded cap")

    monkeypatch.setattr(module.itertools, "permutations", explode)
    monkeypatch.setattr(module, "factorial", explode, raising=False)
    monkeypatch.setattr(math, "factorial", explode)
    monkeypatch.setattr(LaggedPearsonAdapter, "bind", explode)

    with pytest.raises(ResourceLimitError, match="EXACT_STATE_CAP_EXCEEDED"):
        _oracle()(pair, resolution)


@pytest.mark.parametrize(
    ("length", "block_length", "diagnostic"),
    ((5, 2, "divisible"), (2, 2, "at least two")),
)
def test_block_applicability_fails_closed(
    length: int,
    block_length: int,
    diagnostic: str,
) -> None:
    pair = SeriesPair(
        source=np.arange(length, dtype=np.float64),
        target=np.arange(length, dtype=np.float64)[::-1],
    )
    resolution = _resolution(
        null_name="block_shuffle_v2",
        null_params={"block_length": block_length},
        candidates=(1,),
    )

    with pytest.raises(V2IntegrityError, match=diagnostic):
        _oracle()(pair, resolution)


def test_any_analytical_failure_aborts_instead_of_changing_the_denominator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def failing_state(
        bound: _BoundLaggedPearsonAdapter,
        supplied: SeriesPair,
        /,
    ) -> tuple[StatisticResult, ...]:
        nonlocal calls
        del supplied
        calls += 1
        if calls == 2:
            return tuple(
                StatisticResult(
                    candidate_id=candidate,
                    estimate=None,
                    selection_score=None,
                    support_n=bound.observed_length - max(bound.candidates),
                    validity=Validity.ANALYTIC_FAILURE,
                    diagnostics=("scripted_exact_oracle_failure",),
                    backend_identity=bound.backend_identity,
                    preprocessing_identity=bound.preprocessing_identity,
                )
                for candidate in bound.candidates
            )
        return tuple(
            StatisticResult(
                candidate_id=candidate,
                estimate=float(candidate),
                selection_score=None,
                support_n=bound.observed_length - max(bound.candidates),
                validity=Validity.VALID,
                diagnostics=(),
                backend_identity=bound.backend_identity,
                preprocessing_identity=bound.preprocessing_identity,
            )
            for candidate in bound.candidates
        )

    monkeypatch.setattr(_BoundLaggedPearsonAdapter, "evaluate_all", failing_state)
    pair = _pair([0, 1, 2, 3], [3, 2, 1, 0])

    with pytest.raises(V2IntegrityError, match=r"VALID|analytical|selection"):
        _oracle()(pair, _resolution())


def test_oracle_requires_exact_series_pair_and_resolver_owned_resolution() -> None:
    pair = _pair([0, 1, 2, 3], [3, 2, 1, 0])
    resolution = _resolution()

    with pytest.raises(V2IntegrityError, match="exact SeriesPair"):
        _oracle()(cast(SeriesPair, object()), resolution)
    with pytest.raises(V2IntegrityError, match="resolver-owned"):
        _oracle()(pair, cast(PlanResolutionV2, object()))
