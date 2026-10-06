"""Frozen common-support construction for directional lag analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from selcal.contracts import SeriesPair, canonical_candidates


def _frozen_float64_vector(values: NDArray[np.float64], *, name: str) -> NDArray[np.float64]:
    array = np.array(values, dtype=np.float64, copy=True)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    frozen = np.frombuffer(array.tobytes(order="C"), dtype=np.float64)
    frozen.setflags(write=False)
    return frozen


def _frozen_int64_vector(values: NDArray[np.int64], *, name: str) -> NDArray[np.int64]:
    array = np.array(values, dtype=np.int64, copy=True)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must be non-empty")
    frozen = np.frombuffer(array.tobytes(order="C"), dtype=np.int64)
    frozen.setflags(write=False)
    return frozen


@dataclass(frozen=True, slots=True, eq=False)
class LaggedTriplet:
    """Three aligned immutable vectors for one directional candidate."""

    __hash__: ClassVar[None] = None  # type: ignore[assignment]

    future: NDArray[np.float64]
    past: NDArray[np.float64]
    source_past: NDArray[np.float64]

    def __post_init__(self) -> None:
        future = _frozen_float64_vector(self.future, name="future")
        past = _frozen_float64_vector(self.past, name="past")
        source_past = _frozen_float64_vector(self.source_past, name="source_past")
        if future.shape != past.shape or future.shape != source_past.shape:
            raise ValueError("future, past, and source_past must have equal shapes")
        object.__setattr__(self, "future", future)
        object.__setattr__(self, "past", past)
        object.__setattr__(self, "source_past", source_past)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LaggedTriplet):
            return NotImplemented
        return bool(
            np.array_equal(self.future, other.future)
            and np.array_equal(self.past, other.past)
            and np.array_equal(self.source_past, other.source_past)
        )


@dataclass(frozen=True, slots=True, eq=False)
class CommonSupport:
    """Lean immutable metadata for lazily generated common-support triplets."""

    __hash__: ClassVar[None] = None  # type: ignore[assignment]

    candidates: tuple[int, ...]
    max_lag: int
    time_index: NDArray[np.int64]
    support_n: int
    _pair: SeriesPair = field(repr=False)

    def __post_init__(self) -> None:
        candidates = canonical_candidates(self.candidates)
        if isinstance(self.max_lag, bool) or not isinstance(self.max_lag, Integral):
            raise ValueError("max_lag must be an integer")
        max_lag = int(self.max_lag)
        if max_lag != max(candidates):
            raise ValueError("max_lag must equal the maximum canonical candidate")
        if isinstance(self.support_n, bool) or not isinstance(self.support_n, Integral):
            raise ValueError("support_n must be an integer")
        support_n = int(self.support_n)
        if support_n <= 0:
            raise ValueError("support_n must be positive")

        time_index = _frozen_int64_vector(self.time_index, name="time_index")
        if time_index.shape != (support_n,):
            raise ValueError("time_index length must equal support_n")
        expected_index = np.arange(max_lag, max_lag + support_n, dtype=np.int64)
        if not np.array_equal(time_index, expected_index):
            raise ValueError("time_index must be the contiguous common-support axis")
        if not isinstance(self._pair, SeriesPair):
            raise ValueError("_pair must be a SeriesPair")
        if int(self._pair.source.size) != max_lag + support_n:
            raise ValueError("pair length must equal max_lag plus support_n")

        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "max_lag", max_lag)
        object.__setattr__(self, "time_index", time_index)
        object.__setattr__(self, "support_n", support_n)

    @staticmethod
    def _lookup_candidate(candidate: object) -> int:
        if isinstance(candidate, bool) or not isinstance(candidate, Integral):
            raise ValueError("candidate must be a positive integer in the frozen set")
        normalized = int(candidate)
        if normalized <= 0:
            raise ValueError("candidate must be a positive integer in the frozen set")
        return normalized

    def forward_for(self, candidate: int) -> LaggedTriplet:
        normalized = self._lookup_candidate(candidate)
        if normalized not in self.candidates:
            raise ValueError("candidate is not in the frozen common-support set")
        start = self.max_lag
        stop = self.max_lag + self.support_n
        return LaggedTriplet(
            future=self._pair.target[start:stop],
            past=self._pair.target[start - 1 : stop - 1],
            source_past=self._pair.source[start - normalized : stop - normalized],
        )

    def reverse_for(self, candidate: int) -> LaggedTriplet:
        normalized = self._lookup_candidate(candidate)
        if normalized not in self.candidates:
            raise ValueError("candidate is not in the frozen common-support set")
        start = self.max_lag
        stop = self.max_lag + self.support_n
        return LaggedTriplet(
            future=self._pair.source[start:stop],
            past=self._pair.source[start - 1 : stop - 1],
            source_past=self._pair.target[start - normalized : stop - normalized],
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CommonSupport):
            return NotImplemented
        metadata_equal = bool(
            self.candidates == other.candidates
            and self.max_lag == other.max_lag
            and self.support_n == other.support_n
            and np.array_equal(self.time_index, other.time_index)
        )
        if not metadata_equal:
            return False

        start = self.max_lag
        stop = self.max_lag + self.support_n
        common_slices_equal = bool(
            np.array_equal(
                self._pair.target[start:stop],
                other._pair.target[start:stop],
            )
            and np.array_equal(
                self._pair.target[start - 1 : stop - 1],
                other._pair.target[start - 1 : stop - 1],
            )
            and np.array_equal(
                self._pair.source[start:stop],
                other._pair.source[start:stop],
            )
            and np.array_equal(
                self._pair.source[start - 1 : stop - 1],
                other._pair.source[start - 1 : stop - 1],
            )
        )
        if not common_slices_equal:
            return False

        return all(
            np.array_equal(
                self._pair.source[start - candidate : stop - candidate],
                other._pair.source[start - candidate : stop - candidate],
            )
            and np.array_equal(
                self._pair.target[start - candidate : stop - candidate],
                other._pair.target[start - candidate : stop - candidate],
            )
            for candidate in self.candidates
        )


def common_support(pair: SeriesPair, candidates: tuple[int, ...]) -> CommonSupport:
    """Freeze a common time axis and lazily backed directional-series reference."""

    if not isinstance(pair, SeriesPair):
        raise ValueError("pair must be a SeriesPair")
    canonical = canonical_candidates(candidates)
    max_lag = max(canonical)
    series_length = int(pair.source.size)
    if series_length <= max_lag:
        raise ValueError("series length must be greater than the maximum candidate lag")

    time_index = np.arange(max_lag, series_length, dtype=np.int64)
    support_n = series_length - max_lag

    return CommonSupport(
        candidates=canonical,
        max_lag=max_lag,
        time_index=time_index,
        support_n=support_n,
        _pair=pair,
    )
