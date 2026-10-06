from __future__ import annotations

from collections.abc import Mapping
from dataclasses import FrozenInstanceError, fields, is_dataclass

import numpy as np
import pytest

from selcal.contracts import SeriesPair
from selcal.support import CommonSupport, LaggedTriplet, common_support


def _fixture_pair() -> SeriesPair:
    return SeriesPair(
        source=np.arange(8, dtype=np.float64),
        target=np.arange(10, 18, dtype=np.float64),
    )


def _retained_ndarray_bytes(root: object) -> int:
    seen: set[int] = set()

    def visit(value: object) -> int:
        identity = id(value)
        if identity in seen:
            return 0
        seen.add(identity)
        if isinstance(value, np.ndarray):
            return int(value.nbytes)
        if isinstance(value, Mapping):
            return sum(visit(key) + visit(item) for key, item in value.items())
        if isinstance(value, tuple | list | set | frozenset):
            return sum(visit(item) for item in value)
        if is_dataclass(value) and not isinstance(value, type):
            return sum(visit(getattr(value, item.name)) for item in fields(value))
        return 0

    return visit(root)


def test_common_support_matches_forward_and_reverse_hand_oracles() -> None:
    support = common_support(_fixture_pair(), (1, 3))

    assert isinstance(support, CommonSupport)
    assert support.candidates == (1, 3)
    assert support.max_lag == 3
    assert support.support_n == 5
    np.testing.assert_array_equal(support.time_index, np.arange(3, 8))

    forward_one = support.forward_for(1)
    forward_three = support.forward_for(3)
    np.testing.assert_array_equal(forward_one.future, np.arange(13, 18))
    np.testing.assert_array_equal(forward_three.future, np.arange(13, 18))
    np.testing.assert_array_equal(forward_one.past, np.arange(12, 17))
    np.testing.assert_array_equal(forward_three.past, np.arange(12, 17))
    np.testing.assert_array_equal(forward_one.source_past, np.arange(2, 7))
    np.testing.assert_array_equal(forward_three.source_past, np.arange(0, 5))

    reverse_one = support.reverse_for(1)
    reverse_three = support.reverse_for(3)
    np.testing.assert_array_equal(reverse_one.future, np.arange(3, 8))
    np.testing.assert_array_equal(reverse_three.future, np.arange(3, 8))
    np.testing.assert_array_equal(reverse_one.past, np.arange(2, 7))
    np.testing.assert_array_equal(reverse_three.past, np.arange(2, 7))
    np.testing.assert_array_equal(reverse_one.source_past, np.arange(12, 17))
    np.testing.assert_array_equal(reverse_three.source_past, np.arange(10, 15))


def test_common_support_canonicalizes_candidate_order_on_one_frozen_time_axis() -> None:
    support = common_support(_fixture_pair(), (3, 1))

    assert support.candidates == (1, 3)
    assert support.support_n == len(support.time_index) == 5
    for candidate in support.candidates:
        forward = support.forward_for(candidate)
        reverse = support.reverse_for(candidate)
        assert forward.future.shape == (support.support_n,)
        assert reverse.future.shape == (support.support_n,)
        np.testing.assert_array_equal(forward.future, np.arange(13, 18))
        np.testing.assert_array_equal(reverse.future, np.arange(3, 8))


def test_lagged_triplet_copies_and_deeply_freezes_float64_vectors() -> None:
    future = np.array([1.0, 2.0])
    past = np.array([3.0, 4.0])
    source_past = np.array([5.0, 6.0])
    triplet = LaggedTriplet(future=future, past=past, source_past=source_past)
    future[0] = 99.0
    past[0] = 99.0
    source_past[0] = 99.0

    assert triplet.future.dtype == np.float64
    assert triplet.past.dtype == np.float64
    assert triplet.source_past.dtype == np.float64
    np.testing.assert_array_equal(triplet.future, np.array([1.0, 2.0]))
    np.testing.assert_array_equal(triplet.past, np.array([3.0, 4.0]))
    np.testing.assert_array_equal(triplet.source_past, np.array([5.0, 6.0]))

    for array in (triplet.future, triplet.past, triplet.source_past):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.setflags(write=True)
        with pytest.raises(ValueError):
            array[0] = -1.0


@pytest.mark.parametrize(
    ("future", "past", "source_past", "message"),
    [
        (np.array([]), np.array([]), np.array([]), "non-empty"),
        (np.ones((1, 2)), np.ones((1, 2)), np.ones((1, 2)), "one-dimensional"),
        (np.arange(2), np.arange(3), np.arange(2), "equal"),
        (np.array([1.0, np.nan]), np.arange(2), np.arange(2), "finite"),
    ],
)
def test_lagged_triplet_validates_vector_shape_and_values(
    future: np.ndarray,
    past: np.ndarray,
    source_past: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        LaggedTriplet(future=future, past=past, source_past=source_past)


def test_common_support_has_exact_lean_frozen_pair_backed_topology() -> None:
    pair = _fixture_pair()
    support = common_support(pair, (1, 3))

    assert tuple(item.name for item in fields(support)) == (
        "candidates",
        "max_lag",
        "time_index",
        "support_n",
        "_pair",
    )
    assert support._pair is pair
    assert not hasattr(support, "__dict__")
    assert not hasattr(support, "forward")
    assert not hasattr(support, "reverse")
    assert not hasattr(support, "_forward")
    assert not hasattr(support, "_reverse")
    with pytest.raises(FrozenInstanceError):
        support.support_n = 99  # type: ignore[misc]


def test_common_support_direct_construction_validates_pair_type_and_exact_length() -> None:
    time_index = np.arange(3, 8, dtype=np.int64)

    with pytest.raises(ValueError, match="SeriesPair"):
        CommonSupport(
            candidates=(1, 3),
            max_lag=3,
            time_index=time_index,
            support_n=5,
            _pair=object(),  # type: ignore[arg-type]
        )
    short_pair = SeriesPair(source=np.arange(7), target=np.arange(10, 17))
    with pytest.raises(ValueError, match="length"):
        CommonSupport(
            candidates=(1, 3),
            max_lag=3,
            time_index=time_index,
            support_n=5,
            _pair=short_pair,
        )


def test_common_support_is_lazy_and_retains_constant_memory_in_candidate_count() -> None:
    series_length = 20_000
    candidates = tuple(range(1, 51))
    pair = SeriesPair(
        source=np.linspace(-1.0, 1.0, series_length),
        target=np.linspace(1.0, -1.0, series_length),
    )

    support = common_support(pair, candidates)
    retained_bytes = _retained_ndarray_bytes(support)

    assert retained_bytes < 1_000_000, f"retained ndarray bytes: {retained_bytes}"
    first = support.forward_for(25)
    second = support.forward_for(25)
    assert first is not second
    assert first == second


def test_common_support_equality_does_not_generate_triplets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair = _fixture_pair()
    first = common_support(pair, (1, 3))
    equal = common_support(SeriesPair(source=pair.source, target=pair.target), (3, 1))

    def reject_generation(self: CommonSupport, candidate: int) -> LaggedTriplet:
        raise AssertionError("equality must not generate lagged triplets")

    monkeypatch.setattr(CommonSupport, "forward_for", reject_generation)
    monkeypatch.setattr(CommonSupport, "reverse_for", reject_generation)

    assert first == equal


def test_common_support_equality_ignores_samples_outside_candidate_exposed_slices() -> None:
    first_pair = SeriesPair(
        source=np.array([0.0, 10.0, 20.0, 30.0, 40.0]),
        target=np.array([100.0, 110.0, 120.0, 130.0, 140.0]),
    )
    second_pair = SeriesPair(
        source=np.array([0.0, 999.0, 888.0, 30.0, 40.0]),
        target=np.array([100.0, 777.0, 666.0, 130.0, 140.0]),
    )
    first = common_support(first_pair, (4,))
    second = common_support(second_pair, (4,))

    assert first.candidates == second.candidates == (4,)
    assert first.max_lag == second.max_lag == 4
    assert first.support_n == second.support_n == 1
    np.testing.assert_array_equal(first.time_index, second.time_index)
    assert first.forward_for(4) == second.forward_for(4)
    assert first.reverse_for(4) == second.reverse_for(4)
    assert first == second


def test_common_support_arrays_cannot_be_mutated_or_made_writeable() -> None:
    support = common_support(_fixture_pair(), (1, 3))

    arrays = [support.time_index]
    for candidate in support.candidates:
        forward = support.forward_for(candidate)
        reverse = support.reverse_for(candidate)
        arrays.extend(
            [
                forward.future,
                forward.past,
                forward.source_past,
                reverse.future,
                reverse.past,
                reverse.source_past,
            ]
        )

    for array in arrays:
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.setflags(write=True)
        with pytest.raises(ValueError):
            array[0] = -1

    np.testing.assert_array_equal(support.forward_for(1).future, np.arange(13, 18))
    np.testing.assert_array_equal(support.reverse_for(3).source_past, np.arange(10, 15))


@pytest.mark.parametrize("candidate", [2, 1.0, "1", True, None])
def test_common_support_lookup_rejects_unknown_noninteger_and_boolean_candidates(
    candidate: object,
) -> None:
    support = common_support(_fixture_pair(), (1, 3))

    with pytest.raises(ValueError, match="candidate"):
        support.forward_for(candidate)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="candidate"):
        support.reverse_for(candidate)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "bad_candidates",
    [(), (1, 1), (0, 1), (-1, 1), (True, 2), (1, 2.0)],
)
def test_common_support_uses_shared_candidate_validation(
    bad_candidates: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError, match="candidates"):
        common_support(_fixture_pair(), bad_candidates)


def test_common_support_requires_length_greater_than_maximum_lag() -> None:
    pair = SeriesPair(source=np.arange(3), target=np.arange(10, 13))

    with pytest.raises(ValueError, match=r"length|maximum|max"):
        common_support(pair, (1, 3))


def test_lagged_triplet_has_array_aware_equality_and_is_unhashable() -> None:
    first = LaggedTriplet(np.arange(2), np.arange(2, 4), np.arange(4, 6))
    equal = LaggedTriplet(np.arange(2), np.arange(2, 4), np.arange(4, 6))
    different = LaggedTriplet(np.arange(2), np.arange(2, 4), np.arange(5, 7))

    assert first == equal
    assert first != different
    assert LaggedTriplet.__hash__ is None
    with pytest.raises(TypeError, match="unhashable"):
        hash(first)
