"""Parameter-free lagged Pearson statistic on frozen max-lag common support."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Literal, Self

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
from selcal.parameters import freeze_exact_json_mapping, require_exact_keys
from selcal.support import common_support

__all__ = ["LaggedPearsonAdapter"]

_NAME = "lagged_pearson_v1"
# The numeric method is part of the recorded identity. R10v3 Windows core audit (2026-10-03,
# ALG-01): dividing by max|x| and centering once lost exactly representable differences under
# a large common offset (r = 0.94 for an exact correlation of 1). Scaling by a power of two is
# exact, and the corrected second centering pass removes the rounding error of the mean.
# R11 Windows return (R11-02): BLAS dot/norm summed in a platform-dependent order, so values
# differed by 1-2 ULP between OpenBLAS and Accelerate. Every sum is now math.fsum, the correctly
# rounded exact sum, and all other steps are elementwise IEEE operations, so the result no longer
# depends on BLAS summation order. Bytes matched on the environments tested (Apple Accelerate,
# Linux arm64/x86_64 OpenBLAS); this is not a guarantee for every platform or build (math.fsum
# documents rare double-rounding on some builds), so cross-platform checks remain necessary.
_PREPROCESSING_IDENTITY = (
    "no_hidden_transform|formula_two_pass_centered_after_power_of_two_scaling"
    "|correctly_rounded_sums|common_support_max_lag"
)
_OK_DIAGNOSTICS = ("SELCAL_LAGGED_PEARSON_OK",)
_SOURCE_ZERO_DIAGNOSTICS = ("SELCAL_LAGGED_PEARSON_ZERO_SOURCE_VARIANCE",)
_TARGET_ZERO_DIAGNOSTICS = ("SELCAL_LAGGED_PEARSON_ZERO_TARGET_VARIANCE",)
_NUMERIC_FAILURE_DIAGNOSTICS = ("SELCAL_LAGGED_PEARSON_NUMERIC_FAILURE",)
_BOUNDARY_TOLERANCE = 32.0 * np.finfo(np.float64).eps

FloatVector = NDArray[np.float64]


class _ZeroSourceVariance(AnalyticFailure):
    pass


class _ZeroTargetVariance(AnalyticFailure):
    pass


class _NumericFailure(AnalyticFailure):
    pass


def _clamp_correlation(value: float) -> float:
    if not math.isfinite(value):
        raise _NumericFailure("Pearson correlation must be finite")
    if value > 1.0:
        if value <= 1.0 + _BOUNDARY_TOLERANCE:
            return 1.0
        raise _NumericFailure("Pearson correlation exceeds its upper bound")
    if value < -1.0:
        if value >= -1.0 - _BOUNDARY_TOLERANCE:
            return -1.0
        raise _NumericFailure("Pearson correlation exceeds its lower bound")
    return value


def _centered_unit_vector(values: FloatVector, *, role: Literal["source", "target"]) -> FloatVector:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2 or not bool(np.isfinite(array).all()):
        raise _NumericFailure(f"{role} vector is outside the numeric boundary")

    try:
        with np.errstate(over="raise", divide="raise", invalid="raise", under="ignore"):
            scale = float(np.max(np.abs(array)))
            if not math.isfinite(scale):
                raise _NumericFailure(f"{role} scale is not finite")
            if scale == 0.0:
                if role == "source":
                    raise _ZeroSourceVariance("source variance is zero")
                raise _ZeroTargetVariance("target variance is zero")
            # frexp gives scale = m * 2**exponent with 0.5 <= m < 1, so |scaled| <= 1 and
            # ldexp changes only exponents; tiny entries may round only in the subnormal range.
            scaled = np.ldexp(array, -math.frexp(scale)[1])
            count = scaled.size
            first_pass = scaled - math.fsum(scaled.tolist()) / count
            centered = first_pass - math.fsum(first_pass.tolist()) / count
            norm = math.sqrt(math.fsum((centered * centered).tolist()))
            if not math.isfinite(norm):
                raise _NumericFailure(f"{role} norm is not finite")
            if norm == 0.0:
                if role == "source":
                    raise _ZeroSourceVariance("source variance is zero")
                raise _ZeroTargetVariance("target variance is zero")
            unit = centered / norm
    except FloatingPointError as error:
        raise _NumericFailure(f"{role} vector arithmetic failed") from error

    if not bool(np.isfinite(unit).all()):
        raise _NumericFailure(f"{role} normalized vector is not finite")
    return unit


def _lagged_pearson(source_past: FloatVector, future: FloatVector) -> float:
    if source_past.shape != future.shape or source_past.ndim != 1 or source_past.size < 2:
        raise _NumericFailure("Pearson vectors must have equal one-dimensional support >= 2")
    source_unit = _centered_unit_vector(source_past, role="source")
    target_unit = _centered_unit_vector(future, role="target")
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise", under="ignore"):
            estimate = math.fsum((source_unit * target_unit).tolist())
    except FloatingPointError as error:
        raise _NumericFailure("Pearson normalized dot product failed") from error
    return _clamp_correlation(estimate)


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
class LaggedPearsonAdapter:
    """Immutable parameter-free lagged Pearson adapter."""

    _parameters: Mapping[str, JsonValue] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_parameters", MappingProxyType({}))

    @classmethod
    def from_parameters(cls, parameters: object) -> Self:
        frozen = freeze_exact_json_mapping(parameters, name="lagged Pearson parameters")
        require_exact_keys(
            frozen,
            expected=frozenset(),
            name="lagged Pearson parameters",
        )
        return cls()

    @property
    def name(self) -> str:
        return _NAME

    @property
    def parameters(self) -> Mapping[str, JsonValue]:
        return self._parameters

    def bind(
        self, observed_pair: SeriesPair, candidates: tuple[int, ...]
    ) -> _BoundLaggedPearsonAdapter:
        if not isinstance(observed_pair, SeriesPair):
            raise ValueError("observed_pair must be a SeriesPair")
        canonical = canonical_candidates(candidates)
        observed_length = int(observed_pair.source.size)
        if observed_length - max(canonical) < 2:
            raise ValueError("max-lag common support must contain at least 2 samples")
        return _BoundLaggedPearsonAdapter(
            observed_length=observed_length,
            candidates=canonical,
        )


@dataclass(frozen=True, slots=True, eq=False)
class _BoundLaggedPearsonAdapter:
    observed_length: int
    candidates: tuple[int, ...]
    _parameters: Mapping[str, JsonValue] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if type(self.candidates) is not tuple or any(
            type(candidate) is not int for candidate in self.candidates
        ):
            raise ValueError("candidates must be an exact canonical tuple")
        try:
            canonical = canonical_candidates(self.candidates)
        except ValueError as error:
            raise ValueError("candidates must be an exact canonical tuple") from error
        if self.candidates != canonical:
            raise ValueError("candidates must be an exact canonical tuple")
        if type(self.observed_length) is not int:
            raise ValueError("observed_length must be a built-in int")
        if self.observed_length - max(canonical) < 2:
            raise ValueError("observed_length must leave at least 2 common-support samples")
        object.__setattr__(self, "_parameters", MappingProxyType({}))

    @property
    def name(self) -> str:
        return _NAME

    @property
    def parameters(self) -> Mapping[str, JsonValue]:
        return self._parameters

    @property
    def backend_identity(self) -> str:
        return f"selcal.lagged_pearson.v1|numpy={np.__version__}|units=pearson_correlation"

    @property
    def preprocessing_identity(self) -> str:
        return _PREPROCESSING_IDENTITY

    def _failure(
        self,
        candidate: int,
        diagnostics: tuple[str, ...],
        *,
        support_n: int,
    ) -> StatisticResult:
        return StatisticResult(
            candidate_id=candidate,
            estimate=None,
            selection_score=None,
            support_n=support_n,
            validity=Validity.ANALYTIC_FAILURE,
            diagnostics=diagnostics,
            backend_identity=self.backend_identity,
            preprocessing_identity=self.preprocessing_identity,
        )

    def _validate_pair(self, pair: SeriesPair) -> None:
        if not isinstance(pair, SeriesPair):
            raise ValueError("pair must be a SeriesPair")
        if int(pair.source.size) != self.observed_length:
            raise ValueError("pair length must exactly match the bound observed length")

    def _evaluate_raw(self, pair: SeriesPair) -> tuple[StatisticResult, ...]:
        support = common_support(pair, self.candidates)

        results: list[StatisticResult] = []
        for candidate in self.candidates:
            try:
                triplet = support.forward_for(candidate)
                estimate = _lagged_pearson(triplet.source_past, triplet.future)
            except _ZeroSourceVariance:
                results.append(
                    self._failure(
                        candidate,
                        _SOURCE_ZERO_DIAGNOSTICS,
                        support_n=support.support_n,
                    )
                )
                continue
            except _ZeroTargetVariance:
                results.append(
                    self._failure(
                        candidate,
                        _TARGET_ZERO_DIAGNOSTICS,
                        support_n=support.support_n,
                    )
                )
                continue
            except (_NumericFailure, FloatingPointError):
                results.append(
                    self._failure(
                        candidate,
                        _NUMERIC_FAILURE_DIAGNOSTICS,
                        support_n=support.support_n,
                    )
                )
                continue

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
        if type(candidates) is not tuple or any(
            type(candidate) is not int for candidate in candidates
        ):
            raise ValueError("candidates must be an exact canonical tuple")
        canonical = canonical_candidates(candidates)
        if candidates != canonical:
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
