"""Immutable scientific contracts for the SelCal core."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import InitVar, dataclass
from enum import StrEnum
from numbers import Integral, Real
from types import MappingProxyType
from typing import ClassVar, cast

import numpy as np
from numpy.typing import NDArray

from selcal.identifiers import require_versioned_name
from selcal.parameters import JsonValue as JsonValue
from selcal.parameters import freeze_exact_json_mapping

_RESOLUTION_SEAL = object()


class SelectionRule(StrEnum):
    """Supported candidate-selection rules."""

    MAX_UPPER = "max_upper"
    MAX_ABSOLUTE = "max_absolute"


class RunStatus(StrEnum):
    """Terminal status of a calibration run."""

    COMPLETE = "complete"
    NOT_EVALUABLE = "not_evaluable"


class Validity(StrEnum):
    """Validity of a statistic result."""

    VALID = "valid"
    ANALYTIC_FAILURE = "analytic_failure"


class ReplicateStatus(StrEnum):
    """Status of one null replicate."""

    COMPLETE = "complete"
    ANALYTIC_FAILURE = "analytic_failure"


class AnalyticFailure(RuntimeError):
    """A frozen statistic or null contract is undefined for otherwise valid input."""


def canonical_candidates(values: Iterable[int]) -> tuple[int, ...]:
    """Validate and canonicalize the identity of a candidate collection."""

    try:
        candidates = tuple(values)
    except TypeError as error:
        raise ValueError("candidates must be a non-empty iterable of positive integers") from error

    if not candidates:
        raise ValueError("candidates must be non-empty")

    normalized: list[int] = []
    for candidate in candidates:
        if isinstance(candidate, bool) or not isinstance(candidate, Integral):
            raise ValueError("candidates must contain only positive integers")
        normalized.append(int(candidate))

    if any(candidate <= 0 for candidate in normalized):
        raise ValueError("candidates must contain only positive integers")
    if len(set(normalized)) != len(normalized):
        raise ValueError("candidates must not contain duplicates")
    return tuple(sorted(normalized))


def _validated_series(values: NDArray[np.float64], name: str) -> NDArray[np.float64]:
    array = np.array(values, dtype=np.float64, copy=True)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    array[array == 0.0] = 0.0
    immutable_buffer = array.tobytes(order="C")
    frozen = np.frombuffer(immutable_buffer, dtype=np.float64)
    frozen.setflags(write=False)
    return frozen


def _validated_integer(value: object, *, minimum: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    normalized = int(value)
    if normalized < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return normalized


def _validated_finite_real(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be a finite real number")
    if normalized == 0.0:
        return 0.0
    return normalized


def _validated_real(
    value: object,
    *,
    minimum: float,
    maximum: float | None,
    minimum_inclusive: bool,
    maximum_inclusive: bool,
    name: str,
) -> float:
    normalized = _validated_finite_real(value, name=name)

    below_minimum = normalized < minimum if minimum_inclusive else normalized <= minimum
    above_maximum = False
    if maximum is not None:
        above_maximum = normalized > maximum if maximum_inclusive else normalized >= maximum
    if below_minimum or above_maximum:
        raise ValueError(f"{name} is outside its valid range")
    return normalized


def _validated_optional_finite_real(value: object, *, name: str) -> float | None:
    if value is None:
        return None
    return _validated_finite_real(value, name=name)


def _validated_optional_probability(value: object, *, name: str) -> float | None:
    if value is None:
        return None
    return _validated_real(
        value,
        minimum=0.0,
        maximum=1.0,
        minimum_inclusive=True,
        maximum_inclusive=True,
        name=name,
    )


def _validated_diagnostics(values: Iterable[str], *, name: str) -> tuple[str, ...]:
    try:
        diagnostics = tuple(values)
    except TypeError as error:
        raise ValueError(f"{name} must be an iterable of strings") from error
    if not all(isinstance(entry, str) for entry in diagnostics):
        raise ValueError(f"{name} must contain only strings")
    return diagnostics


def _freeze_json(value: object, *, name: str) -> JsonValue:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return str(value)
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        normalized = float(value)
        if not math.isfinite(normalized):
            raise ValueError(f"{name} must contain only finite JSON numbers")
        return normalized
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        frozen: dict[str, JsonValue] = {}
        for key, nested_value in mapping.items():
            if not isinstance(key, str):
                raise ValueError(f"{name} must contain only string mapping keys")
            frozen[str(key)] = _freeze_json(nested_value, name=name)
        return MappingProxyType(frozen)
    if isinstance(value, list | tuple):
        return tuple(_freeze_json(item, name=name) for item in value)
    raise ValueError(f"{name} contains a value that is not JSON-compatible")


def _freeze_mapping(value: object, *, name: str) -> Mapping[str, JsonValue]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    frozen = _freeze_json(value, name=name)
    return cast(Mapping[str, JsonValue], frozen)


def _require_exact_keys(
    value: Mapping[str, JsonValue], *, expected: set[str], name: str
) -> None:
    if set(value) != expected:
        raise ValueError(f"{name} must contain exactly {sorted(expected)}")


@dataclass(frozen=True, slots=True, eq=False)
class SeriesPair:
    """An immutable pair of aligned finite series."""

    __hash__: ClassVar[None] = None  # type: ignore[assignment]

    source: NDArray[np.float64]
    target: NDArray[np.float64]

    def __post_init__(self) -> None:
        source = _validated_series(self.source, "source")
        target = _validated_series(self.target, "target")
        if source.shape != target.shape:
            raise ValueError("source and target must have equal shape")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "target", target)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SeriesPair):
            return NotImplemented
        return bool(
            np.array_equal(self.source, other.source)
            and np.array_equal(self.target, other.target)
        )


def normalize_selection_rule(value: object) -> SelectionRule:
    """Normalize one exact supported selection-rule identifier."""

    if isinstance(value, SelectionRule):
        return value
    if type(value) is str:
        try:
            return SelectionRule(value)
        except ValueError as error:
            raise ValueError("selection_rule is unsupported") from error
    raise ValueError("selection_rule is unsupported")


def _normalize_plan_fields(plan: PlanRequest | ResolvedScientificPlan) -> None:
    object.__setattr__(plan, "candidates", canonical_candidates(plan.candidates))

    object.__setattr__(
        plan, "selection_rule", normalize_selection_rule(plan.selection_rule)
    )

    object.__setattr__(
        plan,
        "statistic_name",
        require_versioned_name(plan.statistic_name, name="statistic_name"),
    )
    object.__setattr__(
        plan,
        "null_name",
        require_versioned_name(plan.null_name, name="null_name"),
    )
    if type(plan.failure_policy) is not str or plan.failure_policy != "fail_closed_v1":
        raise ValueError("failure_policy is unsupported")

    object.__setattr__(
        plan,
        "statistic_params",
        freeze_exact_json_mapping(plan.statistic_params, name="statistic_params"),
    )
    object.__setattr__(
        plan,
        "null_params",
        freeze_exact_json_mapping(plan.null_params, name="null_params"),
    )
    object.__setattr__(
        plan,
        "replicates",
        _validated_integer(plan.replicates, minimum=1, name="replicates"),
    )
    object.__setattr__(
        plan,
        "alpha",
        _validated_real(
            plan.alpha,
            minimum=0.0,
            maximum=1.0,
            minimum_inclusive=False,
            maximum_inclusive=False,
            name="alpha",
        ),
    )
    object.__setattr__(
        plan,
        "tie_tolerance",
        _validated_real(
            plan.tie_tolerance,
            minimum=0.0,
            maximum=None,
            minimum_inclusive=True,
            maximum_inclusive=True,
            name="tie_tolerance",
        ),
    )
    object.__setattr__(
        plan,
        "root_seed",
        _validated_integer(plan.root_seed, minimum=0, name="root_seed"),
    )


def _plan_fields_equal(
    left: PlanRequest | ResolvedScientificPlan,
    right: PlanRequest | ResolvedScientificPlan,
) -> bool:
    return (
        left.candidates == right.candidates
        and left.statistic_name == right.statistic_name
        and left.statistic_params == right.statistic_params
        and left.selection_rule == right.selection_rule
        and left.null_name == right.null_name
        and left.null_params == right.null_params
        and left.replicates == right.replicates
        and left.alpha == right.alpha
        and left.tie_tolerance == right.tie_tolerance
        and left.root_seed == right.root_seed
        and left.failure_policy == right.failure_policy
    )


@dataclass(frozen=True, slots=True, eq=False)
class PlanRequest:
    """An immutable unresolved request for registered scientific contracts."""

    __hash__: ClassVar[None] = None  # type: ignore[assignment]

    candidates: tuple[int, ...]
    statistic_name: str
    statistic_params: Mapping[str, JsonValue]
    selection_rule: SelectionRule | str
    null_name: str
    null_params: Mapping[str, JsonValue]
    replicates: int
    alpha: float
    tie_tolerance: float
    root_seed: int
    failure_policy: str

    def __post_init__(self) -> None:
        _normalize_plan_fields(self)

    def __eq__(self, other: object) -> bool:
        if type(self) is not PlanRequest or type(other) is not PlanRequest:
            return False
        return _plan_fields_equal(self, other)


@dataclass(frozen=True, slots=True, eq=False)
class ResolvedScientificPlan:
    """An immutable plan whose registered contracts were verified by the resolver."""

    __hash__: ClassVar[None] = None  # type: ignore[assignment]

    candidates: tuple[int, ...]
    statistic_name: str
    statistic_params: Mapping[str, JsonValue]
    selection_rule: SelectionRule | str
    null_name: str
    null_params: Mapping[str, JsonValue]
    replicates: int
    alpha: float
    tie_tolerance: float
    root_seed: int
    failure_policy: str
    seal: InitVar[object] = None

    def __post_init__(self, seal: object) -> None:
        if seal is not _RESOLUTION_SEAL:
            raise ValueError("ResolvedScientificPlan must be created by plan resolution")
        _normalize_plan_fields(self)

    def __eq__(self, other: object) -> bool:
        if type(self) is not ResolvedScientificPlan or type(other) is not ResolvedScientificPlan:
            return False
        return _plan_fields_equal(self, other)


@dataclass(frozen=True, slots=True)
class StatisticResult:
    candidate_id: int
    estimate: float | None
    selection_score: float | None
    support_n: int
    validity: Validity
    diagnostics: tuple[str, ...]
    backend_identity: str
    preprocessing_identity: str

    def __post_init__(self) -> None:
        for field_name in ("backend_identity", "preprocessing_identity"):
            identity = getattr(self, field_name)
            if not isinstance(identity, str):
                raise ValueError(f"{field_name} must be a non-empty string")
            normalized_identity = str(identity)
            if not normalized_identity:
                raise ValueError(f"{field_name} must be a non-empty string")
            object.__setattr__(self, field_name, normalized_identity)

        object.__setattr__(self, "candidate_id", canonical_candidates((self.candidate_id,))[0])
        object.__setattr__(
            self,
            "support_n",
            _validated_integer(self.support_n, minimum=1, name="support_n"),
        )
        validity = Validity(self.validity)
        object.__setattr__(self, "validity", validity)
        object.__setattr__(
            self,
            "diagnostics",
            _validated_diagnostics(self.diagnostics, name="diagnostics"),
        )

        if validity is Validity.VALID:
            if self.estimate is None:
                raise ValueError("VALID statistic results require an estimate")
            object.__setattr__(
                self,
                "estimate",
                _validated_finite_real(self.estimate, name="estimate"),
            )
            object.__setattr__(
                self,
                "selection_score",
                _validated_optional_finite_real(
                    self.selection_score, name="selection_score"
                ),
            )
        elif self.estimate is not None or self.selection_score is not None:
            raise ValueError(
                "ANALYTIC_FAILURE statistic results require estimate and selection_score None"
            )


@dataclass(frozen=True, slots=True)
class SelectionResult:
    selected_candidate: int
    selected_index: int
    decision_statistic: float
    tied_candidates: tuple[int, ...]

    def __post_init__(self) -> None:
        selected_candidate = canonical_candidates((self.selected_candidate,))[0]
        object.__setattr__(
            self,
            "selected_candidate",
            selected_candidate,
        )
        object.__setattr__(
            self,
            "selected_index",
            _validated_integer(self.selected_index, minimum=0, name="selected_index"),
        )
        object.__setattr__(
            self,
            "decision_statistic",
            _validated_finite_real(self.decision_statistic, name="decision_statistic"),
        )
        tied_candidates = canonical_candidates(self.tied_candidates)
        if selected_candidate not in tied_candidates:
            raise ValueError("selected_candidate must belong to tied_candidates")
        object.__setattr__(self, "tied_candidates", tied_candidates)


def _validated_complete_vector(
    statistic_results: tuple[StatisticResult, ...],
    selection: SelectionResult | None,
    *,
    name: str,
) -> tuple[int, ...]:
    if not statistic_results:
        raise ValueError(f"COMPLETE {name} requires nonempty statistic_results")

    candidate_ids = tuple(result.candidate_id for result in statistic_results)
    try:
        canonical_ids = canonical_candidates(candidate_ids)
    except ValueError as error:
        raise ValueError(f"COMPLETE {name} requires canonical candidate IDs") from error
    if candidate_ids != canonical_ids:
        raise ValueError(f"COMPLETE {name} requires canonical candidate IDs")
    if selection is None:
        raise ValueError(f"COMPLETE {name} requires a selection")
    if selection.selected_index >= len(statistic_results):
        raise ValueError(f"COMPLETE {name} selection index is outside the result vector")
    if candidate_ids[selection.selected_index] != selection.selected_candidate:
        raise ValueError(f"COMPLETE {name} selection does not match the result vector")
    if not set(selection.tied_candidates).issubset(candidate_ids):
        raise ValueError(f"COMPLETE {name} tied candidates must occur in the result vector")
    if any(result.validity is not Validity.VALID for result in statistic_results):
        raise ValueError(f"COMPLETE {name} requires only VALID statistic results")
    return candidate_ids


@dataclass(frozen=True, slots=True, eq=False)
class ReplicateOutcome:
    __hash__: ClassVar[None] = None  # type: ignore[assignment]

    replicate_id: int
    seed: int
    status: ReplicateStatus
    statistic_results: tuple[StatisticResult, ...]
    selection: SelectionResult | None
    transform_token: Mapping[str, JsonValue] | None
    diagnostics: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "replicate_id",
            _validated_integer(self.replicate_id, minimum=0, name="replicate_id"),
        )
        object.__setattr__(
            self,
            "seed",
            _validated_integer(self.seed, minimum=0, name="seed"),
        )
        status = ReplicateStatus(self.status)
        statistic_results = tuple(self.statistic_results)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "statistic_results", statistic_results)
        if self.transform_token is not None:
            object.__setattr__(
                self,
                "transform_token",
                _freeze_mapping(self.transform_token, name="transform_token"),
            )
        object.__setattr__(
            self,
            "diagnostics",
            _validated_diagnostics(self.diagnostics, name="diagnostics"),
        )

        if status is ReplicateStatus.COMPLETE:
            _validated_complete_vector(
                statistic_results,
                self.selection,
                name="replicate",
            )
        elif self.selection is not None:
            raise ValueError("ANALYTIC_FAILURE replicates require no selection")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ReplicateOutcome):
            return NotImplemented
        return (
            self.replicate_id == other.replicate_id
            and self.seed == other.seed
            and self.status == other.status
            and self.statistic_results == other.statistic_results
            and self.selection == other.selection
            and self.transform_token == other.transform_token
            and self.diagnostics == other.diagnostics
        )


@dataclass(frozen=True, slots=True, eq=False)
class CalibrationResult:
    __hash__: ClassVar[None] = None  # type: ignore[assignment]

    status: RunStatus
    semantic_input_sha256: str
    scientific_plan_sha256: str
    observed_results: tuple[StatisticResult, ...]
    observed_selection: SelectionResult | None
    replicates: tuple[ReplicateOutcome, ...]
    exceedance_count: int
    failure_count: int
    p_value: float | None
    exceedance_bound_low: float | None
    exceedance_bound_high: float | None
    reject_null: bool | None
    diagnostics: tuple[str, ...]

    def __post_init__(self) -> None:
        status = RunStatus(self.status)
        observed_results = tuple(self.observed_results)
        replicates = tuple(self.replicates)
        exceedance_count = _validated_integer(
            self.exceedance_count, minimum=0, name="exceedance_count"
        )
        failure_count = _validated_integer(
            self.failure_count, minimum=0, name="failure_count"
        )
        p_value = _validated_optional_probability(self.p_value, name="p_value")
        bound_low = _validated_optional_probability(
            self.exceedance_bound_low, name="exceedance_bound_low"
        )
        bound_high = _validated_optional_probability(
            self.exceedance_bound_high, name="exceedance_bound_high"
        )

        object.__setattr__(self, "status", status)
        object.__setattr__(self, "observed_results", observed_results)
        object.__setattr__(self, "replicates", replicates)
        object.__setattr__(self, "exceedance_count", exceedance_count)
        object.__setattr__(self, "failure_count", failure_count)
        object.__setattr__(self, "p_value", p_value)
        object.__setattr__(self, "exceedance_bound_low", bound_low)
        object.__setattr__(self, "exceedance_bound_high", bound_high)
        object.__setattr__(
            self,
            "diagnostics",
            _validated_diagnostics(self.diagnostics, name="diagnostics"),
        )

        if exceedance_count + failure_count > len(replicates):
            raise ValueError(
                "exceedance_count + failure_count must not exceed the replicate count"
            )
        if replicates:
            analytic_failures = sum(
                replicate.status is ReplicateStatus.ANALYTIC_FAILURE
                for replicate in replicates
            )
            if failure_count != analytic_failures:
                raise ValueError(
                    "failure_count must equal the ANALYTIC_FAILURE replicate count"
                )

        if (bound_low is None) != (bound_high is None):
            raise ValueError("exceedance bounds must both be absent or both be present")
        if bound_low is not None and bound_high is not None and bound_low > bound_high:
            raise ValueError("exceedance bounds must satisfy low <= high")

        if status is RunStatus.COMPLETE:
            complete_state = (
                bool(replicates)
                and all(
                    replicate.status is ReplicateStatus.COMPLETE
                    for replicate in replicates
                )
                and failure_count == 0
                and p_value is not None
                and bound_low is not None
                and bound_high is not None
                and type(self.reject_null) is bool
            )
            if not complete_state:
                raise ValueError("COMPLETE calibration result fields are contradictory")
            observed_candidate_ids = _validated_complete_vector(
                observed_results,
                self.observed_selection,
                name="calibration observation",
            )
            if any(
                tuple(result.candidate_id for result in replicate.statistic_results)
                != observed_candidate_ids
                for replicate in replicates
            ):
                raise ValueError(
                    "COMPLETE calibration replicate vectors must match observed candidates"
                )
            if bound_low != p_value or bound_high != p_value:
                raise ValueError("COMPLETE calibration bounds must both equal p_value")
        elif p_value is not None or self.reject_null is not None:
            raise ValueError(
                "NOT_EVALUABLE calibration results require p_value and reject_null None"
            )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CalibrationResult):
            return NotImplemented
        return (
            self.status == other.status
            and self.semantic_input_sha256 == other.semantic_input_sha256
            and self.scientific_plan_sha256 == other.scientific_plan_sha256
            and self.observed_results == other.observed_results
            and self.observed_selection == other.observed_selection
            and self.replicates == other.replicates
            and self.exceedance_count == other.exceedance_count
            and self.failure_count == other.failure_count
            and self.p_value == other.p_value
            and self.exceedance_bound_low == other.exceedance_bound_low
            and self.exceedance_bound_high == other.exceedance_bound_high
            and self.reject_null == other.reject_null
            and self.diagnostics == other.diagnostics
        )
