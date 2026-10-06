from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import FrozenInstanceError, replace
from operator import setitem
from types import MappingProxyType

import numpy as np
import pytest

from selcal.contracts import SelectionRule, SeriesPair, Validity
from selcal.statistics.base import BoundStatisticAdapter
from selcal.statistics.binned_nette import BinnedNetTEAdapter
from selcal.statistics.lagged_pearson import LaggedPearsonAdapter
from selcal.support import CommonSupport
from selcal.support import common_support as build_common_support


def pair(source: list[float], target: list[float]) -> SeriesPair:
    return SeriesPair(
        source=np.asarray(source, dtype=np.float64),
        target=np.asarray(target, dtype=np.float64),
    )


def pearson_pair() -> SeriesPair:
    return pair(
        [0, 1, 4, 2, 5, 3, 7, 6],
        [8, 6, 9, -4, -2, -5, -3, -7],
    )


def nette_pair() -> SeriesPair:
    return pair(
        [0, 1, 0, 1, 2, 1, 2, 0, 2, 1],
        [1, 0, 1, 2, 1, 0, 2, 1, 2, 0],
    )


@pytest.mark.parametrize(
    ("adapter_factory", "observed", "candidates", "expected_parameters"),
    [
        (LaggedPearsonAdapter, pearson_pair, (1, 3), {}),
        (lambda: BinnedNetTEAdapter(3), nette_pair, (1, 2), {"bins": 3}),
    ],
)
def test_bound_adapters_expose_frozen_parameters_and_owned_candidates(
    adapter_factory: Callable[[], object],
    observed: Callable[[], SeriesPair],
    candidates: tuple[int, ...],
    expected_parameters: dict[str, int],
) -> None:
    adapter = adapter_factory()
    bound = adapter.bind(observed(), tuple(reversed(candidates)))  # type: ignore[attr-defined]

    assert isinstance(bound, BoundStatisticAdapter)
    assert bound.candidates == candidates
    assert type(bound.candidates) is tuple
    assert bound.parameters == expected_parameters
    assert isinstance(bound.parameters, MappingProxyType)
    with pytest.raises(TypeError):
        setitem(bound.parameters, "hidden", 1)
    with pytest.raises(FrozenInstanceError):
        bound.candidates = (99,)  # type: ignore[misc]


@pytest.mark.parametrize(
    ("adapter_factory", "observed", "candidates"),
    [
        (LaggedPearsonAdapter, pearson_pair, (1, 3)),
        (lambda: BinnedNetTEAdapter(3), nette_pair, (1, 2)),
    ],
)
def test_evaluate_all_is_raw_full_vector_and_v1_evaluate_only_adds_scores(
    adapter_factory: Callable[[], object],
    observed: Callable[[], SeriesPair],
    candidates: tuple[int, ...],
) -> None:
    data = observed()
    bound = adapter_factory().bind(data, candidates)  # type: ignore[attr-defined]

    raw = bound.evaluate_all(data)
    upper = bound.evaluate(data, candidates, SelectionRule.MAX_UPPER)
    absolute = bound.evaluate(data, candidates, SelectionRule.MAX_ABSOLUTE)

    assert tuple(result.candidate_id for result in raw) == candidates
    assert {result.validity for result in raw} == {Validity.VALID}
    assert {result.selection_score for result in raw} == {None}
    assert raw == tuple(replace(result, selection_score=None) for result in upper)
    assert raw == tuple(replace(result, selection_score=None) for result in absolute)
    assert all(result.selection_score is not None for result in upper)
    assert all(result.selection_score is not None for result in absolute)
    for raw_result, v1_result in zip(raw, upper, strict=True):
        assert raw_result.estimate == v1_result.estimate
        assert raw_result.support_n == v1_result.support_n
        assert raw_result.diagnostics == v1_result.diagnostics
        assert raw_result.backend_identity == v1_result.backend_identity
        assert raw_result.preprocessing_identity == v1_result.preprocessing_identity


@pytest.mark.parametrize(
    ("adapter_factory", "observed", "candidates"),
    [
        (LaggedPearsonAdapter, pearson_pair, (1, 3)),
        (lambda: BinnedNetTEAdapter(3), nette_pair, (1, 2)),
    ],
)
def test_evaluate_all_rebuilds_internal_common_support_on_full_length_surrogate_source(
    adapter_factory: Callable[[], object],
    observed: Callable[[], SeriesPair],
    candidates: tuple[int, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = observed()
    bound = adapter_factory().bind(data, candidates)  # type: ignore[attr-defined]
    surrogate = SeriesPair(source=data.source[::-1], target=data.target)
    calls: list[tuple[SeriesPair, tuple[int, ...]]] = []

    def spy_common_support(
        supplied_pair: SeriesPair, supplied_candidates: tuple[int, ...]
    ) -> CommonSupport:
        calls.append((supplied_pair, supplied_candidates))
        return build_common_support(supplied_pair, supplied_candidates)

    module = (
        "lagged_pearson"
        if isinstance(adapter_factory(), LaggedPearsonAdapter)
        else "binned_nette"
    )
    monkeypatch.setattr(f"selcal.statistics.{module}.common_support", spy_common_support)

    results = bound.evaluate_all(surrogate)

    assert calls == [(surrogate, candidates)]
    assert tuple(result.candidate_id for result in results) == candidates
    assert {result.support_n for result in results} == {data.source.size - max(candidates)}
    assert {result.selection_score for result in results} == {None}


@pytest.mark.parametrize(
    ("adapter_factory", "observed", "candidates"),
    [
        (
            LaggedPearsonAdapter,
            lambda: pair([0, 1, 5, 5, 5, 5, 5, 9], [8, 7, 6, 1, 3, 2, 5, 4]),
            (1, 3),
        ),
        (
            lambda: BinnedNetTEAdapter(3),
            lambda: pair([1, 1, 1, 1, 1, 1], [0, 1, 2, 0, 1, 2]),
            (1, 2),
        ),
    ],
)
def test_evaluate_all_analytic_failures_are_raw_and_match_v1_records(
    adapter_factory: Callable[[], object],
    observed: Callable[[], SeriesPair],
    candidates: tuple[int, ...],
) -> None:
    data = observed()
    bound = adapter_factory().bind(data, candidates)  # type: ignore[attr-defined]

    raw = bound.evaluate_all(data)
    v1 = bound.evaluate(data, candidates, SelectionRule.MAX_UPPER)

    assert any(result.validity is Validity.ANALYTIC_FAILURE for result in raw)
    assert raw == tuple(replace(result, selection_score=None) for result in v1)
    assert {result.selection_score for result in raw} == {None}


def test_bound_parameter_snapshots_do_not_alias_adapter_mappings() -> None:
    adapter = BinnedNetTEAdapter(3)
    bound = adapter.bind(nette_pair(), (1, 2))

    assert isinstance(adapter.parameters, Mapping)
    assert bound.parameters == adapter.parameters == {"bins": 3}
    assert bound.parameters is not adapter.parameters
