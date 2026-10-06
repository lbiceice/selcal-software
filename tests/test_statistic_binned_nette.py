from __future__ import annotations

import ast
import math
from collections import Counter
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from selcal.contracts import AnalyticFailure, SelectionRule, SeriesPair, Validity
from selcal.statistics.base import BoundStatisticAdapter, StatisticAdapter
from selcal.statistics.binned_nette import (
    BinnedNetTEAdapter,
    _BoundBinnedNetTEAdapter,
    conditional_mutual_information,
)
from selcal.support import CommonSupport, LaggedTriplet
from selcal.support import common_support as build_common_support

PROVENANCE_ROW = (
    "src/selcal/statistics/binned_nette.py\tclean_room_mathematical_implementation\t"
    "approved SelCal statistic contract\t749629b418b877e5ce776e6e7954d97b0f80d1c2\t"
    "independent_count_table_implementation\t"
    "BSD-3-Clause_rights_confirmed_by_author_20261005\t"
    "development_reference_statistic"
)


def _dictionary_cmi(x: NDArray[np.int64], y: NDArray[np.int64], z: NDArray[np.int64]) -> float:
    """Independent test oracle using Python dictionaries rather than array tables."""

    triples = [(int(a), int(b), int(c)) for a, b, c in zip(x, y, z, strict=True)]
    xyz = Counter(triples)
    xz = Counter((a, c) for a, _, c in triples)
    yz = Counter((b, c) for _, b, c in triples)
    z_counts = Counter(c for _, _, c in triples)
    sample_n = len(triples)

    information = 0.0
    for (a, b, c), joint_count in xyz.items():
        joint_probability = joint_count / sample_n
        ratio = (joint_count * z_counts[c]) / (xz[a, c] * yz[b, c])
        information += joint_probability * math.log(ratio)
    return information


def _dictionary_nette(
    source: NDArray[np.float64],
    target: NDArray[np.float64],
    *,
    bins: int,
    candidates: tuple[int, ...],
) -> tuple[float, ...]:
    """Independent hand oracle with explicit directional indexing loops."""

    source_edges = np.linspace(float(np.min(source)), float(np.max(source)), bins + 1)
    target_edges = np.linspace(float(np.min(target)), float(np.max(target)), bins + 1)
    source_codes = np.searchsorted(source_edges[1:-1], source, side="right").astype(np.int64)
    target_codes = np.searchsorted(target_edges[1:-1], target, side="right").astype(np.int64)

    start = max(candidates)
    estimates: list[float] = []
    for candidate in candidates:
        forward_x: list[int] = []
        forward_y: list[int] = []
        forward_z: list[int] = []
        reverse_x: list[int] = []
        reverse_y: list[int] = []
        reverse_z: list[int] = []
        for time in range(start, source.size):
            forward_x.append(int(source_codes[time - candidate]))
            forward_y.append(int(target_codes[time]))
            forward_z.append(int(target_codes[time - 1]))
            reverse_x.append(int(target_codes[time - candidate]))
            reverse_y.append(int(source_codes[time]))
            reverse_z.append(int(source_codes[time - 1]))
        forward = _dictionary_cmi(
            np.asarray(forward_x, dtype=np.int64),
            np.asarray(forward_y, dtype=np.int64),
            np.asarray(forward_z, dtype=np.int64),
        )
        reverse = _dictionary_cmi(
            np.asarray(reverse_x, dtype=np.int64),
            np.asarray(reverse_y, dtype=np.int64),
            np.asarray(reverse_z, dtype=np.int64),
        )
        estimates.append(forward - reverse)
    return tuple(estimates)


def _pair(source: list[float], target: list[float]) -> SeriesPair:
    return SeriesPair(
        source=np.asarray(source, dtype=np.float64),
        target=np.asarray(target, dtype=np.float64),
    )


def _direct_bound(**overrides: object) -> _BoundBinnedNetTEAdapter:
    values: dict[str, object] = {
        "bins": 3,
        "observed_length": 6,
        "candidates": (1, 2),
        "source_edges": np.array([0.0, 1.0, 2.0, 3.0]),
        "target_edges": np.array([0.0, 1.0, 2.0, 3.0]),
        "bind_diagnostics": (),
    }
    values.update(overrides)
    return _BoundBinnedNetTEAdapter(**values)  # type: ignore[arg-type]


def test_cmi_exact_hand_fixtures() -> None:
    conditionally_constant = conditional_mutual_information(
        np.array([0, 0, 1, 1], dtype=np.int64),
        np.array([0, 1, 0, 1], dtype=np.int64),
        np.array([0, 0, 1, 1], dtype=np.int64),
    )
    perfect_copy = conditional_mutual_information(
        np.array([0, 0, 1, 1], dtype=np.int64),
        np.array([0, 0, 1, 1], dtype=np.int64),
        np.array([0, 0, 0, 0], dtype=np.int64),
    )

    assert conditionally_constant == 0.0
    assert perfect_copy == pytest.approx(math.log(2.0), abs=1e-15)


def test_cmi_nontrivial_hand_count_table() -> None:
    x = np.array([0, 0, 0, 1, 0, 1, 1, 1], dtype=np.int16)
    y = np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.uint8)
    z = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    expected = 0.5 * math.log(4.0 / 3.0) + 0.25 * math.log(2.0 / 3.0) + 0.25 * math.log(2.0)

    assert conditional_mutual_information(x, y, z) == pytest.approx(expected, abs=1e-15)


def test_cmi_is_symmetric_and_invariant_to_joint_sample_permutation() -> None:
    x = np.array([-2, -2, 4, 4, 4, -2, 4, -2], dtype=np.int64)
    y = np.array([7, 9, 7, 9, 9, 7, 7, 9], dtype=np.int64)
    z = np.array([1, 1, 1, 1, 3, 3, 3, 3], dtype=np.uint64)
    x_before = x.copy()
    y_before = y.copy()
    z_before = z.copy()
    permutation = np.array([6, 0, 5, 2, 7, 1, 4, 3], dtype=np.int64)
    observed = conditional_mutual_information(x, y, z)

    assert conditional_mutual_information(y, x, z) == pytest.approx(observed, abs=1e-15)
    assert conditional_mutual_information(
        x[permutation], y[permutation], z[permutation]
    ) == pytest.approx(observed, abs=1e-15)
    assert np.array_equal(x, x_before)
    assert np.array_equal(y, y_before)
    assert np.array_equal(z, z_before)


@pytest.mark.parametrize(
    ("x", "y", "z"),
    [
        ([0, 1], np.array([0, 1]), np.array([0, 1])),
        (np.array([[0, 1]]), np.array([0, 1]), np.array([0, 1])),
        (np.array([], dtype=np.int64),) * 3,
        (np.array([0, 1]), np.array([0]), np.array([0, 1])),
        (np.array([0.0, 1.0]), np.array([0, 1]), np.array([0, 1])),
        (np.array([True, False]), np.array([0, 1]), np.array([0, 1])),
        (np.array([0, 1], dtype=object), np.array([0, 1]), np.array([0, 1])),
        (np.array([0, 1]), np.array([0.0, 1.0]), np.array([0, 1])),
        (np.array([0, 1]), np.array([0, 1]), np.array([False, True])),
    ],
)
def test_cmi_rejects_non_exact_integer_vectors(x: object, y: object, z: object) -> None:
    with pytest.raises(ValueError, match=r"ndarray|one-dimensional|non-empty|shape|integer"):
        conditional_mutual_information(x, y, z)  # type: ignore[arg-type]


def test_adapter_protocols_parameters_and_edge_buffers_are_immutable() -> None:
    adapter = BinnedNetTEAdapter(np.int64(3))
    observed = _pair(
        [0, 1, 0, 2, 1, 2, 0, 1],
        [2, 0, 1, 0, 2, 1, 2, 0],
    )
    bound = adapter.bind(observed, (2, 1))

    assert isinstance(adapter, StatisticAdapter)
    assert isinstance(bound, BoundStatisticAdapter)
    assert adapter.bins == 3
    assert type(adapter.bins) is int
    assert adapter.name == "equal_width_binned_nette_v1"
    assert dict(adapter.parameters) == {"bins": 3}
    with pytest.raises(TypeError):
        adapter.parameters["bins"] = 4  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        adapter.bins = 4  # type: ignore[misc]

    assert not bound.source_edges.flags.writeable
    assert not bound.target_edges.flags.writeable
    with pytest.raises(ValueError):
        bound.source_edges.setflags(write=True)
    with pytest.raises(ValueError):
        bound.target_edges.setflags(write=True)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"bins": np.int64(3)}, "bins"),
        ({"bins": True}, "bins"),
        ({"bins": 1}, "bins"),
        ({"observed_length": np.int64(6)}, "observed_length"),
        ({"observed_length": True}, "observed_length"),
        ({"observed_length": 2}, "observed_length"),
        ({"candidates": [1, 2]}, "canonical"),
        ({"candidates": (2, 1)}, "canonical"),
        ({"candidates": (np.int64(1), 2)}, "canonical"),
    ],
)
def test_direct_bound_rejects_noncanonical_scalar_and_candidate_invariants(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _direct_bound(**overrides)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"source_edges": np.array([0.0, 2.0, 1.0, 3.0])}, "source_edges"),
        ({"source_edges": np.array([np.nan, np.nan, np.nan, np.nan])}, "source_edges"),
        ({"source_edges": np.array([1.0, 1.0, 1.0, 1.0])}, "source_edges"),
        ({"source_edges": np.array([0.0, 1.0, 2.0])}, "source_edges"),
        (
            {
                "source_edges": np.array([0.0, 1.0, 2.0, 3.0]),
                "bind_diagnostics": ("SELCAL_BINNED_NETTE_CONSTANT_SOURCE",),
            },
            "CONSTANT_SOURCE",
        ),
        (
            {
                "source_edges": np.array([0.0, 1.0, 2.0, 3.0]),
                "bind_diagnostics": ("SELCAL_BINNED_NETTE_UNUSABLE_SOURCE_EDGES",),
            },
            "UNUSABLE_SOURCE_EDGES",
        ),
    ],
)
def test_direct_bound_rejects_edges_inconsistent_with_role_diagnostics(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _direct_bound(**overrides)


@pytest.mark.parametrize(
    "diagnostics",
    [
        ["SELCAL_BINNED_NETTE_CONSTANT_SOURCE"],
        ("UNKNOWN_DIAGNOSTIC",),
        (
            "SELCAL_BINNED_NETTE_CONSTANT_SOURCE",
            "SELCAL_BINNED_NETTE_CONSTANT_SOURCE",
        ),
        (
            "SELCAL_BINNED_NETTE_CONSTANT_SOURCE",
            "SELCAL_BINNED_NETTE_UNUSABLE_SOURCE_EDGES",
        ),
    ],
)
def test_direct_bound_rejects_nonexact_unknown_duplicate_or_conflicting_diagnostics(
    diagnostics: object,
) -> None:
    with pytest.raises(ValueError, match="diagnostic"):
        _direct_bound(bind_diagnostics=diagnostics)


def test_direct_bound_copies_and_deeply_freezes_valid_edges() -> None:
    source_edges = np.array([0.0, 1.0, 2.0, 3.0])
    target_edges = np.array([-3.0, -2.0, -1.0, 0.0])
    bound = _direct_bound(source_edges=source_edges, target_edges=target_edges)
    source_edges[0] = -99.0
    target_edges[-1] = 99.0

    np.testing.assert_array_equal(bound.source_edges, np.array([0.0, 1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(bound.target_edges, np.array([-3.0, -2.0, -1.0, 0.0]))
    for edges in (bound.source_edges, bound.target_edges):
        assert edges.dtype == np.float64
        assert not edges.flags.writeable
        with pytest.raises(ValueError):
            edges.setflags(write=True)


@pytest.mark.parametrize(
    ("source_edges", "diagnostic", "source"),
    [
        (
            np.array([1.0, 1.0, 1.0, 1.0]),
            "SELCAL_BINNED_NETTE_CONSTANT_SOURCE",
            [1, 1, 1, 1, 1, 1],
        ),
        (
            np.array([0.0, 0.5, 0.5, 1.0]),
            "SELCAL_BINNED_NETTE_UNUSABLE_SOURCE_EDGES",
            [0, 0.5, 1, 0.5, 0, 1],
        ),
    ],
)
def test_direct_bound_accepts_classified_invalid_edges_and_never_evaluates_valid(
    source_edges: NDArray[np.float64],
    diagnostic: str,
    source: list[float],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bound = _direct_bound(
        source_edges=source_edges,
        bind_diagnostics=(diagnostic,),
    )

    def reject_support(*args: object, **kwargs: object) -> None:
        raise AssertionError("known bind failures must not allocate common support")

    monkeypatch.setattr("selcal.statistics.binned_nette.common_support", reject_support)
    pair = _pair(source, [0, 1, 2, 1, 0, 2])

    results = bound.evaluate(pair, (1, 2), SelectionRule.MAX_UPPER)

    assert {result.validity for result in results} == {Validity.ANALYTIC_FAILURE}
    assert {result.diagnostics for result in results} == {(diagnostic,)}


@pytest.mark.parametrize("bins", [True, 1, 0, -1, 2.0, "3", None, object()])
def test_adapter_rejects_invalid_bin_counts(bins: object) -> None:
    with pytest.raises(ValueError, match="bins"):
        BinnedNetTEAdapter(bins)  # type: ignore[arg-type]


def test_bind_canonicalizes_candidates_and_rejects_short_pairs() -> None:
    adapter = BinnedNetTEAdapter(3)
    pair = _pair([0, 1], [1, 0])

    with pytest.raises(ValueError, match=r"length|maximum|lag"):
        adapter.bind(pair, (2, 1))
    with pytest.raises(ValueError, match="SeriesPair"):
        adapter.bind(object(), (1,))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("pair", "diagnostic"),
    [
        (
            _pair([1, 1, 1, 1, 1, 1], [0, 1, 2, 0, 1, 2]),
            "SELCAL_BINNED_NETTE_CONSTANT_SOURCE",
        ),
        (
            _pair([0, 1, 2, 0, 1, 2], [1, 1, 1, 1, 1, 1]),
            "SELCAL_BINNED_NETTE_CONSTANT_TARGET",
        ),
    ],
)
def test_constant_observed_series_return_full_analytic_failure_vectors(
    pair: SeriesPair, diagnostic: str
) -> None:
    bound = BinnedNetTEAdapter(3).bind(pair, (2, 1))

    first = bound.evaluate(pair, (1, 2), SelectionRule.MAX_UPPER)
    second = bound.evaluate(pair, (1, 2), SelectionRule.MAX_ABSOLUTE)

    assert first == second
    assert tuple(result.candidate_id for result in first) == (1, 2)
    assert {result.validity for result in first} == {Validity.ANALYTIC_FAILURE}
    assert {result.estimate for result in first} == {None}
    assert {result.selection_score for result in first} == {None}
    assert {result.support_n for result in first} == {4}
    assert {result.diagnostics for result in first} == {(diagnostic,)}
    assert all(result.backend_identity for result in first)
    assert all(result.preprocessing_identity for result in first)


def test_non_strict_observed_edges_are_an_analytic_failure() -> None:
    adjacent = np.nextafter(1.0, 2.0)
    pair = _pair(
        [1.0, adjacent, 1.0, adjacent, 1.0, adjacent],
        [0, 1, 2, 3, 2, 1],
    )
    bound = BinnedNetTEAdapter(4).bind(pair, (1, 2))

    results = bound.evaluate(pair, (1, 2), SelectionRule.MAX_UPPER)

    assert {result.validity for result in results} == {Validity.ANALYTIC_FAILURE}
    assert {result.diagnostics for result in results} == {
        ("SELCAL_BINNED_NETTE_UNUSABLE_SOURCE_EDGES",)
    }


def test_extreme_finite_range_uses_overflow_safe_equal_width_edges() -> None:
    pair = _pair(
        [-1e308, 1e308, 0.0, -5e307, 5e307, -1e308],
        [1e308, -1e308, 0.0, 5e307, -5e307, 1e308],
    )
    bound = BinnedNetTEAdapter(2).bind(pair, (1, 2))

    np.testing.assert_array_equal(bound.source_edges, np.array([-1e308, 0.0, 1e308]))
    np.testing.assert_array_equal(bound.target_edges, np.array([-1e308, 0.0, 1e308]))
    results = bound.evaluate(pair, (1, 2), SelectionRule.MAX_UPPER)

    assert {result.validity for result in results} == {Validity.VALID}
    assert all(result.estimate is not None and math.isfinite(result.estimate) for result in results)


def test_bound_evaluation_uses_exact_indexing_support_rules_and_identities() -> None:
    observed = _pair(
        [0, 1, 0, 1, 2, 1, 2, 0, 2, 1],
        [1, 0, 1, 2, 1, 0, 2, 1, 2, 0],
    )
    adapter = BinnedNetTEAdapter(3)
    bound = adapter.bind(observed, (2, 1))
    source_before = observed.source.copy()
    target_before = observed.target.copy()
    expected = _dictionary_nette(observed.source, observed.target, bins=3, candidates=(1, 2))

    upper = bound.evaluate(observed, (1, 2), SelectionRule.MAX_UPPER)
    absolute = bound.evaluate(observed, (1, 2), "max_absolute")
    repeated = bound.evaluate(observed, (1, 2), SelectionRule.MAX_UPPER)

    assert repeated == upper
    assert tuple(result.estimate for result in upper) == pytest.approx(expected, abs=1e-15)
    assert tuple(result.selection_score for result in upper) == pytest.approx(expected, abs=1e-15)
    assert tuple(result.selection_score for result in absolute) == pytest.approx(
        tuple(abs(value) for value in expected), abs=1e-15
    )
    assert all(result.estimate is not None and math.isfinite(result.estimate) for result in upper)
    assert {result.support_n for result in upper} == {observed.source.size - 2}
    assert {result.validity for result in upper} == {Validity.VALID}
    assert {result.diagnostics for result in upper} == {("SELCAL_BINNED_NETTE_OK",)}
    assert {result.backend_identity for result in upper} == {
        f"selcal.equal_width_binned_nette.v1|numpy={np.__version__}|bins=3|units=nats"
    }
    assert {result.preprocessing_identity for result in upper} == {
        "no_hidden_transform|observed_exact_rational_equal_width_edges_reused"
        "|exact_count_decimal_information|common_support_max_lag"
    }
    assert np.array_equal(observed.source, source_before)
    assert np.array_equal(observed.target, target_before)


def test_evaluate_builds_support_only_for_cmi_computation_after_failure_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[SeriesPair, tuple[int, ...]]] = []

    def spy_common_support(pair: SeriesPair, candidates: tuple[int, ...]) -> CommonSupport:
        calls.append((pair, candidates))
        return build_common_support(pair, candidates)

    monkeypatch.setattr(
        "selcal.statistics.binned_nette.common_support",
        spy_common_support,
        raising=False,
    )
    observed = _pair([0, 1, 2, 1, 0, 2], [2, 1, 0, 1, 2, 0])
    bound = BinnedNetTEAdapter(3).bind(observed, (1, 2))
    outside_source = observed.source.copy()
    outside_source[0] = -1.0
    outside = SeriesPair(source=outside_source, target=observed.target)
    constant = _pair([1, 1, 1, 1, 1, 1], [0, 1, 2, 0, 1, 2])
    constant_bound = BinnedNetTEAdapter(3).bind(constant, (1, 2))

    valid_results = bound.evaluate(observed, (1, 2), SelectionRule.MAX_UPPER)
    outside_results = bound.evaluate(outside, (1, 2), SelectionRule.MAX_UPPER)
    constant_results = constant_bound.evaluate(constant, (1, 2), SelectionRule.MAX_UPPER)

    def fail_cmi(x: NDArray[np.int64], y: NDArray[np.int64], z: NDArray[np.int64]) -> float:
        raise AnalyticFailure("simulated analytic-domain failure")

    cmi_target = "selcal.statistics.binned_nette.conditional_mutual_information"
    monkeypatch.setattr(cmi_target, fail_cmi)
    analytic_results = bound.evaluate(observed, (1, 2), SelectionRule.MAX_UPPER)

    assert {result.validity for result in valid_results} == {Validity.VALID}
    assert {result.validity for result in outside_results} == {Validity.ANALYTIC_FAILURE}
    assert {result.validity for result in constant_results} == {Validity.ANALYTIC_FAILURE}
    assert {result.validity for result in analytic_results} == {Validity.ANALYTIC_FAILURE}
    assert calls == [(observed, (1, 2)), (observed, (1, 2))]
    with pytest.raises(ValueError, match="canonical"):
        bound.evaluate(observed, (2, 1), SelectionRule.MAX_UPPER)
    assert len(calls) == 2


def test_evaluate_codes_shared_triplet_roles_with_role_correct_frozen_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = _pair(
        [0, 4, 1, 3, 2, 1, 4, 0],
        [4, 1, 3, 0, 2, 4, 1, 0],
    )
    raw_support = build_common_support(observed, (1, 2))
    method_calls: list[tuple[str, int]] = []

    class SupportSpy:
        support_n = raw_support.support_n

        def forward_for(self, candidate: int) -> LaggedTriplet:
            method_calls.append(("forward", candidate))
            return raw_support.forward_for(candidate)

        def reverse_for(self, candidate: int) -> LaggedTriplet:
            method_calls.append(("reverse", candidate))
            return raw_support.reverse_for(candidate)

    support_calls = 0

    def spy_common_support(pair: SeriesPair, candidates: tuple[int, ...]) -> SupportSpy:
        nonlocal support_calls
        assert pair == observed
        assert candidates == (1, 2)
        support_calls += 1
        return SupportSpy()

    cmi_calls: list[tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]] = []

    def capture_cmi(x: NDArray[np.int64], y: NDArray[np.int64], z: NDArray[np.int64]) -> float:
        cmi_calls.append((x.copy(), y.copy(), z.copy()))
        return 0.0

    monkeypatch.setattr(
        "selcal.statistics.binned_nette.common_support",
        spy_common_support,
        raising=False,
    )
    cmi_target = "selcal.statistics.binned_nette.conditional_mutual_information"
    monkeypatch.setattr(cmi_target, capture_cmi)

    results = (
        BinnedNetTEAdapter(4)
        .bind(observed, (1, 2))
        .evaluate(observed, (1, 2), SelectionRule.MAX_UPPER)
    )

    source_edges = np.linspace(float(np.min(observed.source)), float(np.max(observed.source)), 5)
    target_edges = np.linspace(float(np.min(observed.target)), float(np.max(observed.target)), 5)

    def code(values: NDArray[np.float64], edges: NDArray[np.float64]) -> NDArray[np.int64]:
        return np.searchsorted(edges[1:-1], values, side="right").astype(np.int64)

    expected_calls: list[tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]] = []
    for candidate in (1, 2):
        forward = raw_support.forward_for(candidate)
        reverse = raw_support.reverse_for(candidate)
        expected_calls.extend(
            [
                (
                    code(forward.source_past, source_edges),
                    code(forward.future, target_edges),
                    code(forward.past, target_edges),
                ),
                (
                    code(reverse.source_past, target_edges),
                    code(reverse.future, source_edges),
                    code(reverse.past, source_edges),
                ),
            ]
        )

    assert support_calls == 1
    assert method_calls == [
        ("forward", 1),
        ("reverse", 1),
        ("forward", 2),
        ("reverse", 2),
    ]
    assert len(cmi_calls) == len(expected_calls)
    for observed_call, expected_call in zip(cmi_calls, expected_calls, strict=True):
        assert all(
            np.array_equal(observed_field, expected_field)
            for observed_field, expected_field in zip(observed_call, expected_call, strict=True)
        )
    assert {result.support_n for result in results} == {raw_support.support_n}


def test_bound_evaluation_rejects_configuration_mismatches() -> None:
    observed = _pair([0, 1, 2, 1, 0, 2], [2, 1, 0, 1, 2, 0])
    bound = BinnedNetTEAdapter(3).bind(observed, (1, 2))
    shorter = _pair([0, 1, 2, 1, 0], [2, 1, 0, 1, 2])

    with pytest.raises(ValueError, match="length"):
        bound.evaluate(shorter, (1, 2), SelectionRule.MAX_UPPER)
    with pytest.raises(ValueError, match="bound candidates"):
        bound.evaluate(observed, (1,), SelectionRule.MAX_UPPER)
    with pytest.raises(ValueError, match="canonical"):
        bound.evaluate(observed, (2, 1), SelectionRule.MAX_UPPER)
    with pytest.raises(ValueError, match="selection rule"):
        bound.evaluate(observed, (1, 2), "minimum")
    with pytest.raises(ValueError, match="SeriesPair"):
        bound.evaluate(object(), (1, 2), SelectionRule.MAX_UPPER)  # type: ignore[arg-type]


@pytest.mark.parametrize("outside_role", ["source", "target"])
def test_values_outside_frozen_observed_ranges_fail_the_full_vector(
    outside_role: str,
) -> None:
    observed = _pair([0, 1, 2, 1, 0, 2], [2, 1, 0, 1, 2, 0])
    bound = BinnedNetTEAdapter(3).bind(observed, (1, 2))
    source = observed.source.copy()
    target = observed.target.copy()
    if outside_role == "source":
        source[0] = -1.0
    else:
        target[-1] = 3.0
    surrogate = SeriesPair(source=source, target=target)

    results = bound.evaluate(surrogate, (1, 2), SelectionRule.MAX_UPPER)

    assert tuple(result.candidate_id for result in results) == (1, 2)
    assert {result.validity for result in results} == {Validity.ANALYTIC_FAILURE}
    assert {result.estimate for result in results} == {None}
    assert {result.selection_score for result in results} == {None}
    assert {result.support_n for result in results} == {4}
    assert {result.diagnostics for result in results} == {
        (f"SELCAL_BINNED_NETTE_{outside_role.upper()}_OUTSIDE_OBSERVED_RANGE",)
    }


def test_analytic_failure_during_computation_fails_the_full_vector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = _pair([0, 1, 2, 1, 0, 2], [2, 1, 0, 1, 2, 0])
    bound = BinnedNetTEAdapter(3).bind(observed, (1, 2))

    def fail_cmi(*args: object, **kwargs: object) -> float:
        raise AnalyticFailure("simulated analytic-domain failure")

    monkeypatch.setattr("selcal.statistics.binned_nette.conditional_mutual_information", fail_cmi)

    results = bound.evaluate(observed, (1, 2), SelectionRule.MAX_UPPER)

    assert tuple(result.candidate_id for result in results) == (1, 2)
    assert {result.validity for result in results} == {Validity.ANALYTIC_FAILURE}
    assert {result.estimate for result in results} == {None}
    assert {result.selection_score for result in results} == {None}
    assert {result.diagnostics for result in results} == {("SELCAL_BINNED_NETTE_ANALYTIC_FAILURE",)}


def test_bound_edges_remain_observed_edges_during_source_reordering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = _pair(
        [0, 4, 1, 3, 2, 1, 4, 0],
        [4, 1, 3, 0, 2, 4, 1, 0],
    )
    bound = BinnedNetTEAdapter(4).bind(observed, (1, 2))
    frozen_source_edges = bound.source_edges.copy()
    frozen_target_edges = bound.target_edges.copy()
    reordered = SeriesPair(
        source=observed.source[np.array([7, 3, 5, 0, 6, 2, 4, 1])],
        target=observed.target,
    )
    expected = _dictionary_nette(reordered.source, reordered.target, bins=4, candidates=(1, 2))

    def reject_edge_refit(*args: object, **kwargs: object) -> None:
        raise AssertionError("evaluation must not refit observed bin edges")

    monkeypatch.setattr(np, "linspace", reject_edge_refit)

    results = bound.evaluate(reordered, (1, 2), SelectionRule.MAX_UPPER)

    assert np.array_equal(bound.source_edges, frozen_source_edges)
    assert np.array_equal(bound.target_edges, frozen_target_edges)
    assert tuple(result.estimate for result in results) == pytest.approx(expected, abs=1e-15)


@pytest.mark.parametrize(
    ("source", "target"),
    [
        (
            [0, 1, 2, 0, 1, 2, 0, 2],
            [1, 0, 1, 2, 0, 2, 1, 0],
        ),
        (
            [-3, -1, 0, 2, 4, 1, -2, 3],
            [2, 2, 1, 0, -1, 0, 1, 3],
        ),
        (
            [0.1, 0.5, 0.2, 0.8, 0.3, 0.9, 0.4, 0.7, 0.6],
            [0.9, 0.2, 0.7, 0.3, 0.8, 0.1, 0.6, 0.4, 0.5],
        ),
    ],
)
def test_fixed_hand_examples_match_independent_dictionary_nette_oracle(
    source: list[float], target: list[float]
) -> None:
    pair = _pair(source, target)
    expected = _dictionary_nette(pair.source, pair.target, bins=3, candidates=(1, 2))

    results = (
        BinnedNetTEAdapter(3).bind(pair, (1, 2)).evaluate(pair, (1, 2), SelectionRule.MAX_UPPER)
    )

    assert tuple(result.estimate for result in results) == pytest.approx(expected, abs=1e-15)


def test_statistic_source_is_clean_room_and_provenance_row_is_exact() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    source_path = repository_root / "src" / "selcal" / "statistics" / "binned_nette.py"
    source_text = source_path.read_text(encoding="utf-8")
    syntax = ast.parse(source_text)
    imported_modules = {
        node.module.split(".")[0]
        for node in ast.walk(syntax)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imported_modules.update(
        alias.name.split(".")[0]
        for node in ast.walk(syntax)
        if isinstance(node, ast.Import)
        for alias in node.names
    )

    assert "p3" not in source_text.lower()
    assert "nette-repro" not in source_text.lower()
    assert "p3" not in imported_modules
    provenance_rows = (
        (repository_root / "docs" / "provenance" / "SOURCE_ORIGIN.tsv")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert provenance_rows.count(PROVENANCE_ROW) == 1


def test_equal_width_edges_are_exact_and_independent_of_numpy_rounding() -> None:
    # np.linspace gave 1.8199999999999998 here, and its last bit differed between NumPy 1.26 and
    # 2.x; integer data on an edge then moved between bins (hosted CI, 2026-10-06).
    from selcal.statistics.binned_nette import _observed_edges

    edges, diagnostics = _observed_edges(
        np.array([0.7000000000000001, 3.5], dtype=np.float64), bins=5, role="source"
    )
    assert diagnostics == ()
    assert edges.tolist() == [0.7000000000000001, 1.26, 1.82, 2.38, 2.94, 3.5]
    integer_edges, _ = _observed_edges(np.arange(7, dtype=np.float64), bins=3, role="target")
    assert integer_edges.tolist() == [0.0, 2.0, 4.0, 6.0]
