"""Clean-room equal-width binned directional conditional information statistic."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from numbers import Integral
from types import MappingProxyType
from typing import Any, Self, cast

import numpy as np
from numpy.typing import NDArray

from selcal.contracts import (
    AnalyticFailure,
    JsonValue,
    SelectionRule,
    SeriesPair,
    StatisticResult,
    Validity,
    canonical_candidates,
)
from selcal.parameters import (
    freeze_exact_json_mapping,
    require_builtin_int,
    require_exact_keys,
)
from selcal.support import common_support

__all__ = ["BinnedNetTEAdapter", "conditional_mutual_information"]

_NAME = "equal_width_binned_nette_v1"
_PREPROCESSING_IDENTITY = (
    "no_hidden_transform|observed_equal_width_edges_reused|common_support_max_lag"
)
_OK_DIAGNOSTICS = ("SELCAL_BINNED_NETTE_OK",)
_ROUNDING_TOLERANCE = 1e-15
_BIND_DIAGNOSTIC_ORDER = (
    "SELCAL_BINNED_NETTE_CONSTANT_SOURCE",
    "SELCAL_BINNED_NETTE_UNUSABLE_SOURCE_EDGES",
    "SELCAL_BINNED_NETTE_CONSTANT_TARGET",
    "SELCAL_BINNED_NETTE_UNUSABLE_TARGET_EDGES",
)

IntegerCodeVector = NDArray[np.integer[Any]]


def _validated_code_vector(values: object, *, name: str) -> IntegerCodeVector:
    if type(values) is not np.ndarray:
        raise ValueError(f"{name} must be an exact ndarray")
    array = cast(NDArray[Any], values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if array.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{name} must have a signed or unsigned integer dtype")
    return cast(IntegerCodeVector, array)


def conditional_mutual_information(
    x: IntegerCodeVector,
    y: IntegerCodeVector,
    z: IntegerCodeVector,
) -> float:
    """Return empirical I(X;Y|Z) from exact integer-code vectors, in nats."""

    x_array = _validated_code_vector(x, name="x")
    y_array = _validated_code_vector(y, name="y")
    z_array = _validated_code_vector(z, name="z")
    if x_array.shape != y_array.shape or x_array.shape != z_array.shape:
        raise ValueError("x, y, and z must have equal shapes")

    _, x_inverse = np.unique(x_array, return_inverse=True)
    _, y_inverse = np.unique(y_array, return_inverse=True)
    _, z_inverse = np.unique(z_array, return_inverse=True)
    dense_codes = np.column_stack((x_inverse, y_inverse, z_inverse))

    _, representatives, joint_counts = np.unique(
        dense_codes,
        axis=0,
        return_index=True,
        return_counts=True,
    )
    _, xz_inverse, xz_counts = np.unique(
        dense_codes[:, (0, 2)],
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    _, yz_inverse, yz_counts = np.unique(
        dense_codes[:, (1, 2)],
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    _, z_counts = np.unique(z_inverse, return_counts=True)

    representative_xz = xz_inverse[representatives]
    representative_yz = yz_inverse[representatives]
    representative_z = z_inverse[representatives]
    joint_float = joint_counts.astype(np.float64)
    sample_n = float(x_array.size)
    probabilities = joint_float / sample_n
    ratios = (joint_float * z_counts[representative_z].astype(np.float64)) / (
        xz_counts[representative_xz].astype(np.float64)
        * yz_counts[representative_yz].astype(np.float64)
    )
    information = float(np.sum(probabilities * np.log(ratios), dtype=np.float64))

    if not math.isfinite(information):
        raise AnalyticFailure("conditional mutual information is not finite")
    if -_ROUNDING_TOLERANCE <= information < 0.0:
        return 0.0
    if information < -_ROUNDING_TOLERANCE:
        raise AnalyticFailure("conditional mutual information is materially negative")
    return information


def _frozen_float64_vector(values: NDArray[np.float64]) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    frozen = np.frombuffer(array.tobytes(order="C"), dtype=np.float64)
    frozen.setflags(write=False)
    return frozen


def _validated_bind_diagnostics(values: object) -> tuple[str, ...]:
    if type(values) is not tuple or any(type(value) is not str for value in values):
        raise ValueError("bind_diagnostics must be an exact tuple of known diagnostic codes")
    diagnostics = cast(tuple[str, ...], values)
    if any(diagnostic not in _BIND_DIAGNOSTIC_ORDER for diagnostic in diagnostics):
        raise ValueError("bind_diagnostics contains an unknown diagnostic code")
    if len(set(diagnostics)) != len(diagnostics):
        raise ValueError("bind_diagnostics must not contain duplicate diagnostic codes")
    for role in ("SOURCE", "TARGET"):
        role_diagnostics = tuple(diagnostic for diagnostic in diagnostics if role in diagnostic)
        if len(role_diagnostics) > 1:
            raise ValueError("bind_diagnostics contains conflicting role diagnostics")
    canonical = tuple(
        diagnostic for diagnostic in _BIND_DIAGNOSTIC_ORDER if diagnostic in diagnostics
    )
    if diagnostics != canonical:
        raise ValueError("bind_diagnostics must use canonical diagnostic order")
    return diagnostics


def _validated_bound_edges(
    values: object,
    *,
    bins: int,
    role: str,
    diagnostics: tuple[str, ...],
) -> NDArray[np.float64]:
    name = f"{role.lower()}_edges"
    try:
        array = np.array(values, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a float64 vector") from error
    if array.ndim != 1 or array.shape != (bins + 1,):
        raise ValueError(f"{name} must have shape ({bins + 1},)")

    finite = bool(np.isfinite(array).all())
    all_equal = finite and bool(np.all(array == array[0]))
    strictly_increasing = finite and bool(np.all(array[1:] > array[:-1]))
    constant_code = f"SELCAL_BINNED_NETTE_CONSTANT_{role}"
    unusable_code = f"SELCAL_BINNED_NETTE_UNUSABLE_{role}_EDGES"
    if constant_code in diagnostics:
        if not all_equal:
            raise ValueError(f"{constant_code} requires finite all-equal {name}")
    elif unusable_code in diagnostics:
        if strictly_increasing or all_equal:
            raise ValueError(f"{unusable_code} requires nonconstant unusable {name}")
    elif not strictly_increasing:
        raise ValueError(f"{name} must be finite and strictly increasing")
    return _frozen_float64_vector(array)


def _observed_edges(
    series: NDArray[np.float64], *, bins: int, role: str
) -> tuple[NDArray[np.float64], tuple[str, ...]]:
    minimum = float(np.min(series))
    maximum = float(np.max(series))
    with np.errstate(over="ignore", invalid="ignore"):
        edges = np.linspace(minimum, maximum, bins + 1, dtype=np.float64)
    if not np.isfinite(edges).all():
        fractions = np.linspace(0.0, 1.0, bins + 1, dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            edges = np.asarray(
                (1.0 - fractions) * minimum + fractions * maximum,
                dtype=np.float64,
            )
        edges[0] = minimum
        edges[-1] = maximum
    frozen_edges = _frozen_float64_vector(edges)
    normalized_role = role.upper()
    if minimum == maximum:
        return frozen_edges, (f"SELCAL_BINNED_NETTE_CONSTANT_{normalized_role}",)
    if (
        frozen_edges.shape != (bins + 1,)
        or not np.isfinite(frozen_edges).all()
        or not np.all(np.diff(frozen_edges) > 0.0)
    ):
        return frozen_edges, (f"SELCAL_BINNED_NETTE_UNUSABLE_{normalized_role}_EDGES",)
    return frozen_edges, ()


def _code_with_edges(series: NDArray[np.float64], edges: NDArray[np.float64]) -> NDArray[np.int64]:
    return np.searchsorted(edges[1:-1], series, side="right").astype(np.int64)


def _normalize_rule(rule: SelectionRule | str) -> SelectionRule:
    if isinstance(rule, SelectionRule):
        return rule
    if type(rule) is str:
        try:
            return SelectionRule(rule)
        except ValueError as error:
            raise ValueError("selection rule is unsupported") from error
    raise ValueError("selection rule is unsupported")


@dataclass(frozen=True, slots=True)
class BinnedNetTEAdapter:
    """Immutable equal-width binned directional CMI-difference adapter."""

    bins: int
    _parameters: Mapping[str, JsonValue] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if isinstance(self.bins, bool) or not isinstance(self.bins, Integral):
            raise ValueError("bins must be an integer >= 2")
        normalized_bins = int(self.bins)
        if normalized_bins < 2:
            raise ValueError("bins must be an integer >= 2")
        object.__setattr__(self, "bins", normalized_bins)
        object.__setattr__(
            self,
            "_parameters",
            MappingProxyType({"bins": normalized_bins}),
        )

    @classmethod
    def from_parameters(cls, parameters: object) -> Self:
        frozen = freeze_exact_json_mapping(parameters, name="binned NetTE parameters")
        require_exact_keys(
            frozen,
            expected=frozenset({"bins"}),
            name="binned NetTE parameters",
        )
        return cls(bins=require_builtin_int(frozen["bins"], minimum=2, name="bins"))

    @property
    def name(self) -> str:
        return _NAME

    @property
    def parameters(self) -> Mapping[str, JsonValue]:
        return self._parameters

    def bind(
        self, observed_pair: SeriesPair, candidates: tuple[int, ...]
    ) -> _BoundBinnedNetTEAdapter:
        if not isinstance(observed_pair, SeriesPair):
            raise ValueError("observed_pair must be a SeriesPair")
        canonical = canonical_candidates(candidates)
        observed_length = int(observed_pair.source.size)
        if observed_length <= max(canonical):
            raise ValueError("series length must be greater than the maximum candidate lag")

        source_edges, source_diagnostics = _observed_edges(
            observed_pair.source, bins=self.bins, role="source"
        )
        target_edges, target_diagnostics = _observed_edges(
            observed_pair.target, bins=self.bins, role="target"
        )
        return _BoundBinnedNetTEAdapter(
            bins=self.bins,
            observed_length=observed_length,
            candidates=canonical,
            source_edges=source_edges,
            target_edges=target_edges,
            bind_diagnostics=source_diagnostics + target_diagnostics,
        )


@dataclass(frozen=True, slots=True, eq=False)
class _BoundBinnedNetTEAdapter:
    bins: int
    observed_length: int
    candidates: tuple[int, ...]
    source_edges: NDArray[np.float64]
    target_edges: NDArray[np.float64]
    bind_diagnostics: tuple[str, ...]
    _parameters: Mapping[str, JsonValue] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if type(self.bins) is not int or self.bins < 2:
            raise ValueError("bins must be a built-in int >= 2")
        if type(self.candidates) is not tuple or any(
            type(candidate) is not int for candidate in self.candidates
        ):
            raise ValueError("candidates must be an exact canonical tuple")
        canonical = canonical_candidates(self.candidates)
        if self.candidates != canonical:
            raise ValueError("candidates must be an exact canonical tuple")
        if type(self.observed_length) is not int or self.observed_length <= max(canonical):
            raise ValueError("observed_length must be a built-in int greater than max candidate")

        diagnostics = _validated_bind_diagnostics(self.bind_diagnostics)
        source_edges = _validated_bound_edges(
            self.source_edges,
            bins=self.bins,
            role="SOURCE",
            diagnostics=diagnostics,
        )
        target_edges = _validated_bound_edges(
            self.target_edges,
            bins=self.bins,
            role="TARGET",
            diagnostics=diagnostics,
        )
        object.__setattr__(self, "source_edges", source_edges)
        object.__setattr__(self, "target_edges", target_edges)
        object.__setattr__(self, "bind_diagnostics", diagnostics)
        object.__setattr__(self, "_parameters", MappingProxyType({"bins": self.bins}))

    @property
    def name(self) -> str:
        return _NAME

    @property
    def parameters(self) -> Mapping[str, JsonValue]:
        return self._parameters

    @property
    def backend_identity(self) -> str:
        return (
            f"selcal.equal_width_binned_nette.v1|numpy={np.__version__}|bins={self.bins}|units=nats"
        )

    @property
    def preprocessing_identity(self) -> str:
        return _PREPROCESSING_IDENTITY

    def _failure_vector(
        self, diagnostics: tuple[str, ...], *, support_n: int
    ) -> tuple[StatisticResult, ...]:
        return tuple(
            StatisticResult(
                candidate_id=candidate,
                estimate=None,
                selection_score=None,
                support_n=support_n,
                validity=Validity.ANALYTIC_FAILURE,
                diagnostics=diagnostics,
                backend_identity=self.backend_identity,
                preprocessing_identity=self.preprocessing_identity,
            )
            for candidate in self.candidates
        )

    def _validate_pair(self, pair: SeriesPair) -> None:
        if not isinstance(pair, SeriesPair):
            raise ValueError("pair must be a SeriesPair")
        if int(pair.source.size) != self.observed_length:
            raise ValueError("pair length must exactly match the bound observed length")

    def _evaluate_raw(self, pair: SeriesPair) -> tuple[StatisticResult, ...]:
        support_n = self.observed_length - max(self.candidates)

        diagnostics = list(self.bind_diagnostics)
        if np.any(pair.source < self.source_edges[0]) or np.any(
            pair.source > self.source_edges[-1]
        ):
            diagnostics.append("SELCAL_BINNED_NETTE_SOURCE_OUTSIDE_OBSERVED_RANGE")
        if np.any(pair.target < self.target_edges[0]) or np.any(
            pair.target > self.target_edges[-1]
        ):
            diagnostics.append("SELCAL_BINNED_NETTE_TARGET_OUTSIDE_OBSERVED_RANGE")
        if diagnostics:
            return self._failure_vector(tuple(diagnostics), support_n=support_n)

        support = common_support(pair, self.candidates)
        results: list[StatisticResult] = []
        try:
            for candidate in self.candidates:
                forward_triplet = support.forward_for(candidate)
                reverse_triplet = support.reverse_for(candidate)
                forward = conditional_mutual_information(
                    _code_with_edges(forward_triplet.source_past, self.source_edges),
                    _code_with_edges(forward_triplet.future, self.target_edges),
                    _code_with_edges(forward_triplet.past, self.target_edges),
                )
                reverse = conditional_mutual_information(
                    _code_with_edges(reverse_triplet.source_past, self.target_edges),
                    _code_with_edges(reverse_triplet.future, self.source_edges),
                    _code_with_edges(reverse_triplet.past, self.source_edges),
                )
                estimate = forward - reverse
                if not math.isfinite(estimate):
                    raise AnalyticFailure("NetTE estimate is not finite")
                results.append(
                    StatisticResult(
                        candidate_id=candidate,
                        estimate=estimate,
                        selection_score=None,
                        support_n=support.support_n,
                        validity=Validity.VALID,
                        diagnostics=_OK_DIAGNOSTICS,
                        backend_identity=self.backend_identity,
                        preprocessing_identity=self.preprocessing_identity,
                    )
                )
        except AnalyticFailure:
            return self._failure_vector(
                ("SELCAL_BINNED_NETTE_ANALYTIC_FAILURE",),
                support_n=support.support_n,
            )
        return tuple(results)

    def evaluate_all(self, pair: SeriesPair, /) -> tuple[StatisticResult, ...]:
        self._validate_pair(pair)
        return self._evaluate_raw(pair)

    def evaluate(
        self,
        pair: SeriesPair,
        candidates: tuple[int, ...],
        rule: SelectionRule | str,
    ) -> tuple[StatisticResult, ...]:
        self._validate_pair(pair)
        if type(candidates) is not tuple:
            raise ValueError("candidates must be an exact canonical tuple")
        canonical = canonical_candidates(candidates)
        if candidates != canonical or any(type(candidate) is not int for candidate in candidates):
            raise ValueError("candidates must be an exact canonical tuple")
        if canonical != self.candidates:
            raise ValueError("candidates must exactly match the bound candidates")
        normalized_rule = _normalize_rule(rule)
        raw_results = self._evaluate_raw(pair)
        return tuple(
            replace(
                result,
                selection_score=(
                    result.estimate
                    if normalized_rule is SelectionRule.MAX_UPPER
                    else abs(result.estimate)
                ),
            )
            if result.validity is Validity.VALID and result.estimate is not None
            else result
            for result in raw_results
        )
