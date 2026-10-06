"""Immutable fail-closed scientific contracts for SelCal v2."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import InitVar, dataclass, fields
from enum import StrEnum
from types import FunctionType, MemberDescriptorType
from typing import Any, ClassVar, Literal, NamedTuple, cast

from selcal._verifier_primitives_v2 import (
    DerivedCountsV2,
    ExactSlotSchemaV2,
    ReplicateExpectationV2,
    TerminalEvidenceV2,
    VectorVerificationPolicyV2,
    VerifiedVectorEvidenceV2,
    VerifierExternalOpsV2,
    VerifierPrimitiveOpsV2,
    VerifierSchemasV2,
    VerifierTypesV2,
    ZipValuesV2,
    freeze_verifier_primitives_v2,
)
from selcal.contracts import (
    JsonValue,
    ReplicateStatus,
    RunStatus,
    SelectionResult,
    SelectionRule,
    SeriesPair,
    StatisticResult,
    Validity,
    canonical_candidates,
    normalize_selection_rule,
)
from selcal.identifiers import require_versioned_name
from selcal.parameters import freeze_exact_json_mapping

_V2_RESOLUTION_SEAL = object()


class SelCalV2Error(RuntimeError):
    """Base class for explicitly typed SelCal v2 failures."""


class V2IntegrityError(SelCalV2Error):
    """Stored v2 values contradict their fail-closed scientific contract."""


class PlanVersionError(SelCalV2Error):
    """An executable v2 boundary received a plan from another version."""


class ResourceLimitError(SelCalV2Error):
    """A frozen v2 operation exceeds an explicit resource ceiling."""


class NullBindStatus(StrEnum):
    """Applicability status of an executable null bound to observed data."""

    ENABLED = "enabled"
    DISABLED = "disabled"


class NullDisabledReason(StrEnum):
    """Exhaustive M0--M2 reasons for disabling a supported v2 null."""

    SHIFT_SPACE_EMPTY = "shift_space_empty"
    NON_DIVISIBLE_TAIL = "non_divisible_tail"
    FEWER_THAN_TWO_BLOCKS = "fewer_than_two_blocks"
    TOO_MANY_BLOCKS = "too_many_blocks"


class RunFailureStage(StrEnum):
    """Scientific stage at which a v2 run became non-evaluable."""

    NULL_BIND = "null_bind"
    OBSERVED_STATISTIC_SCAN = "observed_statistic_scan"
    REPLICATE_EXECUTION = "replicate_execution"


class ReplicateFailureStage(StrEnum):
    """Scientific stage at which one retained replicate failed."""

    STATISTIC_SCAN = "statistic_scan"


def _raise_integrity(message: str) -> None:
    raise V2IntegrityError(message)


def _require_exact_schema(value: object, *, expected: str, name: str) -> str:
    if type(value) is not str or value != expected:
        _raise_integrity(f"{name} must equal {expected!r}")
    return cast(str, value)


def _require_lowercase_sha256(value: object, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _raise_integrity(f"{name} must be a lowercase SHA-256 hex digest")
    return cast(str, value)


def _require_exact_int(
    value: object,
    *,
    minimum: int,
    maximum: int | None,
    name: str,
) -> int:
    if type(value) is not int:
        _raise_integrity(f"{name} must be a built-in integer >= {minimum}")
    exact_value = cast(int, value)
    if exact_value < minimum:
        _raise_integrity(f"{name} must be a built-in integer >= {minimum}")
    if maximum is not None and exact_value > maximum:
        _raise_integrity(f"{name} must be <= {maximum}")
    return exact_value


def _require_exact_float(
    value: object,
    *,
    minimum: float,
    maximum: float,
    minimum_inclusive: bool,
    maximum_inclusive: bool,
    name: str,
) -> float:
    if type(value) is not float:
        _raise_integrity(f"{name} must be a finite built-in float")
    exact_value = cast(float, value)
    if not math.isfinite(exact_value):
        _raise_integrity(f"{name} must be a finite built-in float")
    below = exact_value < minimum if minimum_inclusive else exact_value <= minimum
    above = exact_value > maximum if maximum_inclusive else exact_value >= maximum
    if below or above:
        _raise_integrity(f"{name} is outside its valid range")
    return 0.0 if exact_value == 0.0 else exact_value


def _require_optional_probability(value: object, *, name: str) -> float | None:
    if value is None:
        return None
    return _require_exact_float(
        value,
        minimum=0.0,
        maximum=1.0,
        minimum_inclusive=True,
        maximum_inclusive=True,
        name=name,
    )


def _require_exact_diagnostics(value: object, *, name: str) -> tuple[str, ...]:
    if type(value) is not tuple:
        _raise_integrity(f"{name} must be an exact tuple of built-in strings")
    exact_value = cast(tuple[object, ...], value)
    if any(type(entry) is not str for entry in exact_value):
        _raise_integrity(f"{name} must be an exact tuple of built-in strings")
    return cast(tuple[str, ...], exact_value)


def _require_bounded_int(value: object, *, minimum: int, maximum: int, name: str) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be a built-in integer in [{minimum}, {maximum}]")
    return value


def _require_builtin_float(
    value: object,
    *,
    minimum: float,
    maximum: float | None,
    minimum_inclusive: bool,
    maximum_inclusive: bool,
    name: str,
) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite built-in float")
    below_minimum = value < minimum if minimum_inclusive else value <= minimum
    above_maximum = False
    if maximum is not None:
        above_maximum = value > maximum if maximum_inclusive else value >= maximum
    if below_minimum or above_maximum:
        raise ValueError(f"{name} is outside its valid range")
    return 0.0 if value == 0.0 else value


@dataclass(frozen=True, slots=True, eq=False)
class PlanRequestV2:
    """An immutable unresolved request for scientific-plan v2 resolution."""

    __hash__: ClassVar[None] = None  # type: ignore[assignment]

    schema: ClassVar[str] = "selcal.scientific-plan.v2"
    decision_contract: ClassVar[str] = "family_max_with_canonical_tie_label_v2"
    failure_contract: ClassVar[str] = "exact_b_fail_closed_v2"
    rng_contract: ClassVar[str] = "sha256_framed_plan_rid_stream_to_pcg64_raw64_v1"
    common_support: ClassVar[str] = "max_candidate_lag_v1"
    calibration_contract: ClassVar[str] = "full_reselection_global_mc_plus_one_v2"

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

    def __post_init__(self) -> None:
        _normalize_plan_v2_fields(self)

    def __eq__(self, other: object) -> bool:
        if type(self) is not PlanRequestV2 or type(other) is not PlanRequestV2:
            return False
        return _plan_v2_fields_equal(self, other)


def _normalize_plan_v2_fields(
    plan: PlanRequestV2 | ResolvedScientificPlanV2,
) -> None:
    object.__setattr__(plan, "candidates", canonical_candidates(plan.candidates))
    object.__setattr__(
        plan,
        "statistic_name",
        require_versioned_name(plan.statistic_name, name="statistic_name"),
    )
    object.__setattr__(
        plan,
        "statistic_params",
        freeze_exact_json_mapping(plan.statistic_params, name="statistic_params"),
    )
    object.__setattr__(
        plan,
        "selection_rule",
        normalize_selection_rule(plan.selection_rule),
    )
    object.__setattr__(
        plan,
        "null_name",
        require_versioned_name(plan.null_name, name="null_name"),
    )
    object.__setattr__(
        plan,
        "null_params",
        freeze_exact_json_mapping(plan.null_params, name="null_params"),
    )
    object.__setattr__(
        plan,
        "replicates",
        _require_bounded_int(
            plan.replicates,
            minimum=1,
            maximum=1_000_000,
            name="replicates",
        ),
    )
    object.__setattr__(
        plan,
        "alpha",
        _require_builtin_float(
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
        _require_builtin_float(
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
        _require_bounded_int(
            plan.root_seed,
            minimum=0,
            maximum=2**64 - 1,
            name="root_seed",
        ),
    )


def _plan_v2_fields_equal(
    left: PlanRequestV2 | ResolvedScientificPlanV2,
    right: PlanRequestV2 | ResolvedScientificPlanV2,
) -> bool:
    # Imported lazily to avoid a contracts/canonical initialization cycle.
    from selcal.canonical import canonical_json_bytes

    return (
        left.candidates == right.candidates
        and left.statistic_name == right.statistic_name
        and canonical_json_bytes(left.statistic_params)
        == canonical_json_bytes(right.statistic_params)
        and left.selection_rule == right.selection_rule
        and left.null_name == right.null_name
        and canonical_json_bytes(left.null_params) == canonical_json_bytes(right.null_params)
        and left.replicates == right.replicates
        and left.alpha == right.alpha
        and left.tie_tolerance == right.tie_tolerance
        and left.root_seed == right.root_seed
    )


@dataclass(frozen=True, slots=True, eq=False)
class ResolvedScientificPlanV2:
    """A v2 plan whose exact built-in adapters were verified by its resolver."""

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
    seal: InitVar[object] = None

    def __post_init__(self, seal: object) -> None:
        if seal is not _V2_RESOLUTION_SEAL:
            raise ValueError("ResolvedScientificPlanV2 must be created by v2 plan resolution")
        if type(self) is not ResolvedScientificPlanV2:
            raise ValueError("resolved plan must be an exact ResolvedScientificPlanV2")
        _normalize_plan_v2_fields(self)

    def __eq__(self, other: object) -> bool:
        if (
            type(self) is not ResolvedScientificPlanV2
            or type(other) is not ResolvedScientificPlanV2
        ):
            return False
        return _plan_v2_fields_equal(self, other)


@dataclass(frozen=True, slots=True, eq=False)
class CircularShiftStateV2:
    """One exact labelled state in the circular-shift v2 universe."""

    __hash__: ClassVar[None] = None  # type: ignore[assignment]

    schema: Literal["selcal.circular-shift-state.v2"]
    shift: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "schema",
            _require_exact_schema(
                self.schema,
                expected="selcal.circular-shift-state.v2",
                name="schema",
            ),
        )
        object.__setattr__(
            self,
            "shift",
            _require_exact_int(
                self.shift,
                minimum=0,
                maximum=None,
                name="shift",
            ),
        )

    def __eq__(self, other: object) -> bool:
        return (
            type(self) is CircularShiftStateV2
            and type(other) is CircularShiftStateV2
            and self.schema == other.schema
            and self.shift == other.shift
        )


@dataclass(frozen=True, slots=True, eq=False)
class BlockShuffleStateV2:
    """One exact labelled block permutation in the block-shuffle v2 universe."""

    __hash__: ClassVar[None] = None  # type: ignore[assignment]

    schema: Literal["selcal.block-shuffle-state.v2"]
    block_order: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "schema",
            _require_exact_schema(
                self.schema,
                expected="selcal.block-shuffle-state.v2",
                name="schema",
            ),
        )
        if (
            type(self.block_order) is not tuple
            or len(self.block_order) < 2
            or any(type(label) is not int for label in self.block_order)
            or set(self.block_order) != set(range(len(self.block_order)))
        ):
            _raise_integrity(
                "block_order must be an exact tuple forming a permutation of 0..q-1 with q >= 2"
            )

    def __eq__(self, other: object) -> bool:
        return (
            type(self) is BlockShuffleStateV2
            and type(other) is BlockShuffleStateV2
            and self.schema == other.schema
            and self.block_order == other.block_order
        )


NullStateV2 = CircularShiftStateV2 | BlockShuffleStateV2


@dataclass(frozen=True, slots=True, eq=False)
class NullTransformToken:
    """An exact owned token for one v2 null-transformation state."""

    __hash__: ClassVar[None] = None  # type: ignore[assignment]

    schema: Literal["selcal.null-transform-token.v2"]
    null_name: str
    null_parameter_sha256: str
    semantic_input_sha256: str
    scientific_plan_sha256: str
    bound_null_owner_sha256: str
    is_identity: bool
    state: NullStateV2

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "schema",
            _require_exact_schema(
                self.schema,
                expected="selcal.null-transform-token.v2",
                name="schema",
            ),
        )
        if type(self.null_name) is not str or self.null_name not in {
            "circular_shift_v2",
            "circular_shift_exact_v1",
            "block_shuffle_v2",
        }:
            _raise_integrity("null_name must identify a supported executable v2 null")
        for field_name in (
            "null_parameter_sha256",
            "semantic_input_sha256",
            "scientific_plan_sha256",
            "bound_null_owner_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_lowercase_sha256(getattr(self, field_name), name=field_name),
            )
        if type(self.is_identity) is not bool:
            _raise_integrity("is_identity must be a built-in bool")

        if self.null_name in {"circular_shift_v2", "circular_shift_exact_v1"}:
            if type(self.state) is not CircularShiftStateV2:
                _raise_integrity("circular shift nulls require CircularShiftStateV2")
            circular_state = cast(CircularShiftStateV2, self.state)
            expected_identity = circular_state.shift == 0
        else:
            if type(self.state) is not BlockShuffleStateV2:
                _raise_integrity("block_shuffle_v2 requires BlockShuffleStateV2")
            block_state = cast(BlockShuffleStateV2, self.state)
            expected_identity = block_state.block_order == tuple(
                range(len(block_state.block_order))
            )
        if self.is_identity is not expected_identity:
            _raise_integrity("is_identity contradicts the exact null state")

    def __eq__(self, other: object) -> bool:
        return (
            type(self) is NullTransformToken
            and type(other) is NullTransformToken
            and self.schema == other.schema
            and self.null_name == other.null_name
            and self.null_parameter_sha256 == other.null_parameter_sha256
            and self.semantic_input_sha256 == other.semantic_input_sha256
            and self.scientific_plan_sha256 == other.scientific_plan_sha256
            and self.bound_null_owner_sha256 == other.bound_null_owner_sha256
            and self.is_identity is other.is_identity
            and self.state == other.state
        )


@dataclass(frozen=True, slots=True, eq=False)
class NullTransformResult:
    """The immutable output of applying one exact owned null token."""

    __hash__: ClassVar[None] = None  # type: ignore[assignment]

    pair: SeriesPair
    token: NullTransformToken
    source_changed: bool
    diagnostics: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.pair) is not SeriesPair:
            _raise_integrity("pair must be an exact SeriesPair")
        if type(self.token) is not NullTransformToken:
            _raise_integrity("token must be an exact NullTransformToken")
        if type(self.source_changed) is not bool:
            _raise_integrity("source_changed must be a built-in bool")
        if self.token.is_identity and self.source_changed:
            _raise_integrity("an identity token cannot report a changed source")
        object.__setattr__(
            self,
            "diagnostics",
            _require_exact_diagnostics(self.diagnostics, name="diagnostics"),
        )

    def __eq__(self, other: object) -> bool:
        return (
            type(self) is NullTransformResult
            and type(other) is NullTransformResult
            and self.pair == other.pair
            and self.token == other.token
            and self.source_changed is other.source_changed
            and self.diagnostics == other.diagnostics
        )


def _require_statistic_vector(
    value: object,
    *,
    name: str,
    allow_empty: bool,
) -> tuple[StatisticResult, ...]:
    if type(value) is not tuple:
        _raise_integrity(f"{name} must be an exact tuple")
    exact_value = cast(tuple[object, ...], value)
    if not allow_empty and not exact_value:
        _raise_integrity(f"{name} must be non-empty")
    if any(type(result) is not StatisticResult for result in exact_value):
        _raise_integrity(f"{name} must contain only exact StatisticResult values")
    results = cast(tuple[StatisticResult, ...], exact_value)
    candidate_ids = tuple(result.candidate_id for result in results)
    if candidate_ids:
        try:
            canonical_ids = canonical_candidates(candidate_ids)
        except ValueError as error:
            raise V2IntegrityError(f"{name} has invalid candidate IDs") from error
        if candidate_ids != canonical_ids:
            _raise_integrity(f"{name} candidate IDs must be canonical and unique")
    return results


def _require_selection(value: object, *, name: str) -> SelectionResult:
    if type(value) is not SelectionResult:
        _raise_integrity(f"{name} must be an exact SelectionResult")
    return cast(SelectionResult, value)


def _validate_complete_scored_vector(
    statistic_results: tuple[StatisticResult, ...],
    selection: SelectionResult | None,
    *,
    name: str,
) -> tuple[int, ...]:
    results = _require_statistic_vector(
        statistic_results,
        name=name,
        allow_empty=False,
    )
    if any(
        result.validity is not Validity.VALID or result.selection_score is None
        for result in results
    ):
        _raise_integrity(f"{name} must contain only centrally scored VALID results")
    exact_selection = _require_selection(selection, name=f"{name} selection")
    candidate_ids = tuple(result.candidate_id for result in results)
    if exact_selection.selected_index >= len(results):
        _raise_integrity(f"{name} selection index is outside the result vector")
    if candidate_ids[exact_selection.selected_index] != exact_selection.selected_candidate:
        _raise_integrity(f"{name} selection candidate and index disagree")
    if not set(exact_selection.tied_candidates).issubset(candidate_ids):
        _raise_integrity(f"{name} tied candidates must occur in the result vector")
    if exact_selection.selected_candidate != min(exact_selection.tied_candidates):
        _raise_integrity(f"{name} selected candidate must be the canonical tie label")
    maximum_score = max(
        result.selection_score for result in results if result.selection_score is not None
    )
    if exact_selection.decision_statistic != maximum_score:
        _raise_integrity(f"{name} decision statistic must equal the family maximum")
    return candidate_ids


def _validate_raw_failure_vector(
    statistic_results: tuple[StatisticResult, ...],
    *,
    name: str,
) -> tuple[int, ...]:
    results = _require_statistic_vector(
        statistic_results,
        name=name,
        allow_empty=False,
    )
    if not any(result.validity is Validity.ANALYTIC_FAILURE for result in results):
        _raise_integrity(f"{name} must retain at least one analytical failure")
    if any(result.selection_score is not None for result in results):
        _raise_integrity(f"{name} must remain an unscored raw vector")
    return tuple(result.candidate_id for result in results)


@dataclass(frozen=True, slots=True, eq=False)
class ReplicateOutcome:
    """One retained exact-B v2 replicate outcome."""

    __hash__: ClassVar[None] = None  # type: ignore[assignment]

    replicate_id: int
    seed_digest_sha256: str
    status: ReplicateStatus
    failure_stage: ReplicateFailureStage | None
    transform_token: NullTransformToken
    statistic_results: tuple[StatisticResult, ...]
    selection: SelectionResult | None
    diagnostics: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "replicate_id",
            _require_exact_int(
                self.replicate_id,
                minimum=0,
                maximum=None,
                name="replicate_id",
            ),
        )
        object.__setattr__(
            self,
            "seed_digest_sha256",
            _require_lowercase_sha256(
                self.seed_digest_sha256,
                name="seed_digest_sha256",
            ),
        )
        if type(self.status) is not ReplicateStatus:
            _raise_integrity("status must be an exact ReplicateStatus")
        if self.failure_stage is not None and type(self.failure_stage) is not ReplicateFailureStage:
            _raise_integrity("failure_stage must be an exact ReplicateFailureStage or None")
        if type(self.transform_token) is not NullTransformToken:
            _raise_integrity("transform_token must be an exact NullTransformToken")
        object.__setattr__(
            self,
            "diagnostics",
            _require_exact_diagnostics(self.diagnostics, name="diagnostics"),
        )

        if self.status is ReplicateStatus.COMPLETE:
            if self.failure_stage is not None:
                _raise_integrity("COMPLETE replicate requires failure_stage None")
            _validate_complete_scored_vector(
                self.statistic_results,
                self.selection,
                name="replicate statistic_results",
            )
        else:
            if self.failure_stage is not ReplicateFailureStage.STATISTIC_SCAN:
                _raise_integrity("ANALYTIC_FAILURE replicate requires STATISTIC_SCAN failure stage")
            if self.selection is not None:
                _raise_integrity("ANALYTIC_FAILURE replicate requires selection None")
            _validate_raw_failure_vector(
                self.statistic_results,
                name="replicate statistic_results",
            )

    def __eq__(self, other: object) -> bool:
        return (
            type(self) is ReplicateOutcome
            and type(other) is ReplicateOutcome
            and self.replicate_id == other.replicate_id
            and self.seed_digest_sha256 == other.seed_digest_sha256
            and self.status is other.status
            and self.failure_stage is other.failure_stage
            and self.transform_token == other.transform_token
            and self.statistic_results == other.statistic_results
            and self.selection == other.selection
            and self.diagnostics == other.diagnostics
        )


def _require_exact_replicates(value: object) -> tuple[ReplicateOutcome, ...]:
    if type(value) is not tuple:
        _raise_integrity("replicates must be an exact tuple")
    exact_value = cast(tuple[object, ...], value)
    if any(type(outcome) is not ReplicateOutcome for outcome in exact_value):
        _raise_integrity("replicates must contain only exact ReplicateOutcome values")
    return cast(tuple[ReplicateOutcome, ...], exact_value)


def _require_exact_b_ids(replicates: tuple[ReplicateOutcome, ...], planned_replicates: int) -> None:
    if len(replicates) != planned_replicates or any(
        outcome.replicate_id != expected_id for expected_id, outcome in enumerate(replicates)
    ):
        _raise_integrity("replicate IDs must be exactly ordered 0..B-1")


def _require_candidate_agreement(
    expected: tuple[int, ...],
    replicates: tuple[ReplicateOutcome, ...],
) -> None:
    if any(
        tuple(result.candidate_id for result in outcome.statistic_results) != expected
        for outcome in replicates
    ):
        _raise_integrity("every replicate must retain the observed candidate vector")


def _require_run_token_ownership(
    replicates: tuple[ReplicateOutcome, ...],
    *,
    semantic_input_sha256: str,
    scientific_plan_sha256: str,
) -> None:
    if not replicates:
        _raise_integrity("token ownership requires at least one retained replicate")
    snapshot = replicates[0].transform_token
    for outcome in replicates:
        token = outcome.transform_token
        if token.semantic_input_sha256 != semantic_input_sha256:
            _raise_integrity("replicate token belongs to a foreign semantic input")
        if token.scientific_plan_sha256 != scientific_plan_sha256:
            _raise_integrity("replicate token belongs to a foreign scientific plan")
        if (
            token.null_name != snapshot.null_name
            or token.null_parameter_sha256 != snapshot.null_parameter_sha256
            or token.bound_null_owner_sha256 != snapshot.bound_null_owner_sha256
        ):
            _raise_integrity("all replicate tokens must share one null contract and bound owner")


def _require_no_decision_fields(
    *,
    p_value: float | None,
    bound_low: float | None,
    bound_high: float | None,
    reject_null: bool | None,
) -> None:
    if (
        p_value is not None
        or bound_low is not None
        or bound_high is not None
        or reject_null is not None
    ):
        _raise_integrity("this failure stage permits no p-value, bounds, or decision")


@dataclass(frozen=True, slots=True, eq=False)
class CalibrationResult:
    """Self-contained fail-closed v2 calibration result."""

    __hash__: ClassVar[None] = None  # type: ignore[assignment]

    status: RunStatus
    failure_stage: RunFailureStage | None
    semantic_input_sha256: str
    scientific_plan_sha256: str
    planned_replicates: int
    alpha: float
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
        if type(self.status) is not RunStatus:
            _raise_integrity("status must be an exact RunStatus")
        if self.failure_stage is not None and type(self.failure_stage) is not RunFailureStage:
            _raise_integrity("failure_stage must be an exact RunFailureStage or None")
        object.__setattr__(
            self,
            "semantic_input_sha256",
            _require_lowercase_sha256(
                self.semantic_input_sha256,
                name="semantic_input_sha256",
            ),
        )
        object.__setattr__(
            self,
            "scientific_plan_sha256",
            _require_lowercase_sha256(
                self.scientific_plan_sha256,
                name="scientific_plan_sha256",
            ),
        )
        object.__setattr__(
            self,
            "planned_replicates",
            _require_exact_int(
                self.planned_replicates,
                minimum=1,
                maximum=1_000_000,
                name="planned_replicates",
            ),
        )
        object.__setattr__(
            self,
            "alpha",
            _require_exact_float(
                self.alpha,
                minimum=0.0,
                maximum=1.0,
                minimum_inclusive=False,
                maximum_inclusive=False,
                name="alpha",
            ),
        )
        observed_results = _require_statistic_vector(
            self.observed_results,
            name="observed_results",
            allow_empty=True,
        )
        replicates = _require_exact_replicates(self.replicates)
        object.__setattr__(self, "observed_results", observed_results)
        object.__setattr__(self, "replicates", replicates)
        exceedance_count = _require_exact_int(
            self.exceedance_count,
            minimum=0,
            maximum=self.planned_replicates,
            name="exceedance_count",
        )
        failure_count = _require_exact_int(
            self.failure_count,
            minimum=0,
            maximum=self.planned_replicates,
            name="failure_count",
        )
        object.__setattr__(self, "exceedance_count", exceedance_count)
        object.__setattr__(self, "failure_count", failure_count)
        p_value = _require_optional_probability(self.p_value, name="p_value")
        bound_low = _require_optional_probability(
            self.exceedance_bound_low,
            name="exceedance_bound_low",
        )
        bound_high = _require_optional_probability(
            self.exceedance_bound_high,
            name="exceedance_bound_high",
        )
        object.__setattr__(self, "p_value", p_value)
        object.__setattr__(self, "exceedance_bound_low", bound_low)
        object.__setattr__(self, "exceedance_bound_high", bound_high)
        if self.reject_null is not None and type(self.reject_null) is not bool:
            _raise_integrity("reject_null must be a built-in bool or None")
        object.__setattr__(
            self,
            "diagnostics",
            _require_exact_diagnostics(self.diagnostics, name="diagnostics"),
        )

        if self.status is RunStatus.COMPLETE:
            self._validate_complete(p_value, bound_low, bound_high)
        elif self.failure_stage is RunFailureStage.NULL_BIND:
            self._validate_null_bind(p_value, bound_low, bound_high)
        elif self.failure_stage is RunFailureStage.OBSERVED_STATISTIC_SCAN:
            self._validate_observed_failure(p_value, bound_low, bound_high)
        elif self.failure_stage is RunFailureStage.REPLICATE_EXECUTION:
            self._validate_replicate_failure(p_value, bound_low, bound_high)
        else:
            _raise_integrity("NOT_EVALUABLE requires an approved failure stage")

    def _validate_complete(
        self,
        p_value: float | None,
        bound_low: float | None,
        bound_high: float | None,
    ) -> None:
        if self.failure_stage is not None:
            _raise_integrity("COMPLETE requires failure_stage None")
        observed_candidates = _validate_complete_scored_vector(
            self.observed_results,
            self.observed_selection,
            name="observed_results",
        )
        _require_exact_b_ids(self.replicates, self.planned_replicates)
        _require_run_token_ownership(
            self.replicates,
            semantic_input_sha256=self.semantic_input_sha256,
            scientific_plan_sha256=self.scientific_plan_sha256,
        )
        if any(outcome.status is not ReplicateStatus.COMPLETE for outcome in self.replicates):
            _raise_integrity("COMPLETE requires every replicate outcome to be complete")
        _require_candidate_agreement(observed_candidates, self.replicates)
        if self.failure_count != 0:
            _raise_integrity("COMPLETE requires failure_count zero")
        assert self.observed_selection is not None
        observed_decision = self.observed_selection.decision_statistic
        derived_exceedances = sum(
            outcome.selection is not None
            and outcome.selection.decision_statistic >= observed_decision
            for outcome in self.replicates
        )
        if self.exceedance_count != derived_exceedances:
            _raise_integrity("exceedance_count contradicts stored complete outcomes")
        expected_p = (1 + derived_exceedances) / (self.planned_replicates + 1)
        if p_value != expected_p:
            _raise_integrity("p_value must equal the exact plus-one expression")
        if bound_low is not None or bound_high is not None:
            _raise_integrity("COMPLETE requires diagnostic bounds None")
        if type(self.reject_null) is not bool or self.reject_null is not (expected_p <= self.alpha):
            _raise_integrity("reject_null must equal p_value <= alpha")

    def _validate_null_bind(
        self,
        p_value: float | None,
        bound_low: float | None,
        bound_high: float | None,
    ) -> None:
        if (
            self.observed_results
            or self.observed_selection is not None
            or self.replicates
            or self.exceedance_count != 0
            or self.failure_count != 0
        ):
            _raise_integrity("NULL_BIND requires zero scans, outcomes, and counts")
        _require_no_decision_fields(
            p_value=p_value,
            bound_low=bound_low,
            bound_high=bound_high,
            reject_null=self.reject_null,
        )

    def _validate_observed_failure(
        self,
        p_value: float | None,
        bound_low: float | None,
        bound_high: float | None,
    ) -> None:
        _validate_raw_failure_vector(
            self.observed_results,
            name="observed_results",
        )
        if (
            self.observed_selection is not None
            or self.replicates
            or self.exceedance_count != 0
            or self.failure_count != 0
        ):
            _raise_integrity("OBSERVED_STATISTIC_SCAN requires no selection, outcomes, or counts")
        _require_no_decision_fields(
            p_value=p_value,
            bound_low=bound_low,
            bound_high=bound_high,
            reject_null=self.reject_null,
        )

    def _validate_replicate_failure(
        self,
        p_value: float | None,
        bound_low: float | None,
        bound_high: float | None,
    ) -> None:
        observed_candidates = _validate_complete_scored_vector(
            self.observed_results,
            self.observed_selection,
            name="observed_results",
        )
        _require_exact_b_ids(self.replicates, self.planned_replicates)
        _require_run_token_ownership(
            self.replicates,
            semantic_input_sha256=self.semantic_input_sha256,
            scientific_plan_sha256=self.scientific_plan_sha256,
        )
        _require_candidate_agreement(observed_candidates, self.replicates)
        derived_failures = sum(
            outcome.status is ReplicateStatus.ANALYTIC_FAILURE for outcome in self.replicates
        )
        if derived_failures < 1 or self.failure_count != derived_failures:
            _raise_integrity("failure_count must equal at least one failed outcome")
        assert self.observed_selection is not None
        observed_decision = self.observed_selection.decision_statistic
        derived_exceedances = sum(
            outcome.status is ReplicateStatus.COMPLETE
            and outcome.selection is not None
            and outcome.selection.decision_statistic >= observed_decision
            for outcome in self.replicates
        )
        if self.exceedance_count != derived_exceedances:
            _raise_integrity("exceedance_count contradicts retained complete outcomes")
        if self.exceedance_count + self.failure_count > self.planned_replicates:
            _raise_integrity("exceedance_count + failure_count must not exceed B")
        if p_value is not None or self.reject_null is not None:
            _raise_integrity("replicate failure permits no p-value or rejection decision")
        expected_low = (1 + derived_exceedances) / (self.planned_replicates + 1)
        expected_high = (1 + derived_exceedances + derived_failures) / (self.planned_replicates + 1)
        if bound_low != expected_low or bound_high != expected_high:
            _raise_integrity("replicate failure bounds contradict E, F, and B")

    def __eq__(self, other: object) -> bool:
        return (
            type(self) is CalibrationResult
            and type(other) is CalibrationResult
            and self.status is other.status
            and self.failure_stage is other.failure_stage
            and self.semantic_input_sha256 == other.semantic_input_sha256
            and self.scientific_plan_sha256 == other.scientific_plan_sha256
            and self.planned_replicates == other.planned_replicates
            and self.alpha == other.alpha
            and self.observed_results == other.observed_results
            and self.observed_selection == other.observed_selection
            and self.replicates == other.replicates
            and self.exceedance_count == other.exceedance_count
            and self.failure_count == other.failure_count
            and self.p_value == other.p_value
            and self.exceedance_bound_low == other.exceedance_bound_low
            and self.exceedance_bound_high == other.exceedance_bound_high
            and self.reject_null is other.reject_null
            and self.diagnostics == other.diagnostics
        )


def _freeze_verifier_slots(
    contract_type: type[object],
) -> tuple[tuple[str, MemberDescriptorType], ...]:
    frozen: list[tuple[str, MemberDescriptorType]] = []
    for field in fields(cast(Any, contract_type)):
        name = field.name
        descriptor = contract_type.__dict__.get(name)
        if type(descriptor) is not MemberDescriptorType:
            raise RuntimeError(f"result verifier slot {contract_type.__name__}.{name} is invalid")
        frozen.append((name, descriptor))
    return tuple(frozen)


_CALIBRATION_RESULT_VERIFIER_SLOTS = _freeze_verifier_slots(CalibrationResult)
_REPLICATE_OUTCOME_VERIFIER_SLOTS = _freeze_verifier_slots(ReplicateOutcome)
_STATISTIC_RESULT_VERIFIER_SLOTS = _freeze_verifier_slots(StatisticResult)
_SELECTION_RESULT_VERIFIER_SLOTS = _freeze_verifier_slots(SelectionResult)
_NULL_TRANSFORM_TOKEN_VERIFIER_SLOTS = _freeze_verifier_slots(NullTransformToken)
_CIRCULAR_STATE_VERIFIER_SLOTS = _freeze_verifier_slots(CircularShiftStateV2)
_BLOCK_STATE_VERIFIER_SLOTS = _freeze_verifier_slots(BlockShuffleStateV2)

# These names are retained solely as ordinary module-rebinding sentinels for the
# verifier-boundary regression suite.  Runtime verification is implemented only
# by the zero-global functions inside ``_freeze_result_verifier_capsule_v2``.
_NONEXECUTING_VERIFIER_COMPATIBILITY_SENTINEL_V2 = object()
_read_verifier_slots = _NONEXECUTING_VERIFIER_COMPATIBILITY_SENTINEL_V2
_verify_exact_diagnostics = _NONEXECUTING_VERIFIER_COMPATIBILITY_SENTINEL_V2
_verify_selection_against_scores = _NONEXECUTING_VERIFIER_COMPATIBILITY_SENTINEL_V2
_verify_result_vector_against_plan = _NONEXECUTING_VERIFIER_COMPATIBILITY_SENTINEL_V2


class _ResultVerifierCapsuleV2(NamedTuple):
    verify: Callable[[object, object], None]
    external_leaves: tuple[object, ...]


_PlanSha256V2 = Callable[[object], str]
_ResolutionContextV2 = Callable[[object], Any]
_SeedDigestV2 = Callable[[str, int, int], str]
_ResultVerifierFactoryV2 = Callable[
    [_PlanSha256V2, _ResolutionContextV2, _SeedDigestV2],
    _ResultVerifierCapsuleV2,
]
verify_calibration_result: Callable[[object, object], None]

_RESULT_VERIFIER_EXTERNAL_LEAVES_V2 = (
    type,
    tuple,
    int,
    float,
    str,
    bool,
    any,
    len,
    max,
    enumerate,
    zip,
    abs,
    math.isfinite,
    AttributeError,
    TypeError,
    ValueError,
    OverflowError,
)
_RESULT_VERIFIER_CONTRACT_LEAVES_V2 = (
    V2IntegrityError,
    CalibrationResult,
    ReplicateOutcome,
    StatisticResult,
    SelectionResult,
    SelectionRule,
    Validity,
    RunStatus,
    RunFailureStage,
    ReplicateStatus,
    ReplicateFailureStage,
    _CALIBRATION_RESULT_VERIFIER_SLOTS,
    _REPLICATE_OUTCOME_VERIFIER_SLOTS,
    _STATISTIC_RESULT_VERIFIER_SLOTS,
    _SELECTION_RESULT_VERIFIER_SLOTS,
)


def _freeze_verifier_zip() -> ZipValuesV2:
    zip_values = zip

    def paired_values(
        left: tuple[object, ...], right: tuple[object, ...], *, strict: bool
    ) -> Iterable[tuple[object, object]]:
        return zip_values(left, right, strict=strict)

    return paired_values


def _verifier_construction_inputs(
    seed_digest: _SeedDigestV2,
    /,
) -> tuple[VerifierTypesV2, VerifierSchemasV2, VerifierExternalOpsV2]:
    types = VerifierTypesV2(
        CalibrationResult,
        ReplicateOutcome,
        StatisticResult,
        SelectionResult,
        SelectionRule,
        SelectionRule.MAX_UPPER,
        Validity,
        Validity.VALID,
        Validity.ANALYTIC_FAILURE,
        RunStatus,
        RunStatus.COMPLETE,
        RunStatus.NOT_EVALUABLE,
        RunFailureStage,
        RunFailureStage.NULL_BIND,
        RunFailureStage.OBSERVED_STATISTIC_SCAN,
        RunFailureStage.REPLICATE_EXECUTION,
        ReplicateStatus,
        ReplicateStatus.COMPLETE,
        ReplicateStatus.ANALYTIC_FAILURE,
        ReplicateFailureStage,
        ReplicateFailureStage.STATISTIC_SCAN,
    )
    schemas = VerifierSchemasV2(
        ExactSlotSchemaV2(CalibrationResult, _CALIBRATION_RESULT_VERIFIER_SLOTS),
        ExactSlotSchemaV2(ReplicateOutcome, _REPLICATE_OUTCOME_VERIFIER_SLOTS),
        ExactSlotSchemaV2(StatisticResult, _STATISTIC_RESULT_VERIFIER_SLOTS),
        ExactSlotSchemaV2(SelectionResult, _SELECTION_RESULT_VERIFIER_SLOTS),
    )
    externals = VerifierExternalOpsV2(
        V2IntegrityError,
        type,
        tuple,
        list,
        any,
        len,
        max,
        enumerate,
        _freeze_verifier_zip(),
        abs,
        math.isfinite,
        (AttributeError, TypeError, ValueError, OverflowError),
        seed_digest,
    )
    return types, schemas, externals


def _freeze_context_check(externals: VerifierExternalOpsV2, /) -> Callable[[Any], Any]:
    tuple_type = tuple

    tuple_getitem: Callable[[tuple[object, ...], int], Any] = tuple.__getitem__

    def check_context(context: Any) -> Any:
        if (
            tuple_getitem(externals, 1)(context) is not tuple_type
            or tuple_getitem(externals, 5)(context) != 11
        ):
            raise tuple_getitem(externals, 0)("sealed resolution context is invalid")
        return context

    return check_context


def _freeze_selection_policy_check(
    types: VerifierTypesV2,
    externals: VerifierExternalOpsV2,
    /,
) -> Callable[[Any], VectorVerificationPolicyV2]:
    tuple_type, int_type, float_type = tuple, int, float
    policy_type = VectorVerificationPolicyV2
    construct_policy = policy_type.__new__

    tuple_getitem: Callable[[tuple[object, ...], int], Any] = tuple.__getitem__

    def selection_policy(context: Any) -> VectorVerificationPolicyV2:
        candidates, rule, tolerance = context[2], context[3], context[6]
        if (
            tuple_getitem(externals, 1)(candidates) is not tuple_type
            or not candidates
            or tuple_getitem(externals, 4)(
                tuple_getitem(externals, 1)(item) is not int_type for item in candidates
            )
            or tuple_getitem(externals, 1)(rule) is not tuple_getitem(types, 4)
            or tuple_getitem(externals, 1)(tolerance) is not float_type
            or not tuple_getitem(externals, 10)(tolerance)
            or tolerance < 0.0
        ):
            raise tuple_getitem(externals, 0)("sealed plan selection contract is invalid")
        return construct_policy(
            policy_type,
            candidates,
            rule,
            tuple_getitem(types, 5),
            tolerance,
            context[7],
            context[8],
        )

    return selection_policy


def _freeze_result_prefix(
    types: VerifierTypesV2,
    schemas: VerifierSchemasV2,
    externals: VerifierExternalOpsV2,
    ops: VerifierPrimitiveOpsV2,
    /,
) -> Callable[[object, Any], tuple[str, str, VectorVerificationPolicyV2, TerminalEvidenceV2]]:
    tuple_type, int_type, float_type = tuple, int, float
    terminal_type = TerminalEvidenceV2
    construct_terminal = terminal_type.__new__
    selection_policy = _freeze_selection_policy_check(types, externals)

    tuple_getitem: Callable[[tuple[object, ...], int], Any] = tuple.__getitem__

    def result_prefix(
        result: object,
        context: Any,
    ) -> tuple[str, str, VectorVerificationPolicyV2, TerminalEvidenceV2]:
        snapshot: Any = tuple_getitem(ops, 2)(
            result,
            tuple_getitem(schemas, 0),
            subject="calibration result",
        )
        (
            status,
            stage,
            semantic,
            plan_digest,
            planned,
            alpha,
            observed,
            selection,
            replicates,
            exceedances,
            failures,
            p_value,
            low,
            high,
            reject,
            diagnostic,
        ) = snapshot
        tuple_getitem(ops, 1)(diagnostic, subject="calibration result")
        if tuple_getitem(externals, 1)(replicates) is not tuple_type:
            raise tuple_getitem(externals, 0)("calibration replicates must be an exact tuple")
        semantic_digest = tuple_getitem(ops, 0)(semantic, name="semantic_input_sha256")
        result_plan_digest = tuple_getitem(ops, 0)(plan_digest, name="scientific_plan_sha256")
        if (
            tuple_getitem(externals, 1)(status) is not tuple_getitem(types, 9)
            or (
                stage is not None
                and tuple_getitem(externals, 1)(stage) is not tuple_getitem(types, 12)
            )
            or tuple_getitem(externals, 1)(planned) is not int_type
            or planned != context[4]
            or tuple_getitem(externals, 1)(alpha) is not float_type
            or alpha != context[5]
            or result_plan_digest != context[1]
        ):
            raise tuple_getitem(externals, 0)(
                "calibration result scalars contradict the sealed plan"
            )
        policy = selection_policy(context)
        terminal = construct_terminal(
            terminal_type,
            status,
            stage,
            observed,
            selection,
            replicates,
            exceedances,
            failures,
            p_value,
            low,
            high,
            reject,
            planned,
            alpha,
        )
        return semantic_digest, result_plan_digest, policy, terminal

    return result_prefix


def _freeze_observed_verification(
    types: VerifierTypesV2,
    ops: VerifierPrimitiveOpsV2,
    /,
) -> Callable[[TerminalEvidenceV2, VectorVerificationPolicyV2], VerifiedVectorEvidenceV2 | None]:
    cast_value = cast
    tuple_getitem: Callable[[tuple[object, ...], int], Any] = tuple.__getitem__

    def verify_observed(
        terminal: TerminalEvidenceV2,
        policy: VectorVerificationPolicyV2,
    ) -> VerifiedVectorEvidenceV2 | None:
        if tuple_getitem(terminal, 1) is tuple_getitem(types, 13):
            return None
        if tuple_getitem(terminal, 1) is tuple_getitem(types, 14):
            failed_evidence: VerifiedVectorEvidenceV2 = cast_value(
                "VerifiedVectorEvidenceV2",
                tuple_getitem(ops, 5)(
                    tuple_getitem(terminal, 2),
                    tuple_getitem(terminal, 3),
                    policy,
                    subject="observed result vector",
                ),
            )
            return failed_evidence
        complete_evidence: VerifiedVectorEvidenceV2 = cast_value(
            "VerifiedVectorEvidenceV2",
            tuple_getitem(ops, 4)(
                tuple_getitem(terminal, 2),
                tuple_getitem(terminal, 3),
                policy,
                subject="observed result vector",
            ),
        )
        return complete_evidence

    return verify_observed


def _freeze_final_context_check(
    externals: VerifierExternalOpsV2,
    /,
) -> Callable[[Any, Any, str], None]:
    tuple_getitem: Callable[[tuple[object, ...], int], Any] = tuple.__getitem__

    def check_final_context(context: Any, final: Any, result_plan_digest: str) -> None:
        if (
            final[0] is not context[0]
            or final[1] != result_plan_digest
            or final[2] != context[2]
            or final[3] is not context[3]
            or final[4] != context[4]
            or final[5] != context[5]
            or final[6] != context[6]
            or final[7] != context[7]
            or final[8] != context[8]
            or final[9] is not context[9]
            or final[10] is not context[10]
        ):
            raise tuple_getitem(externals, 0)("scientific plan changed during result verification")

    return check_final_context


def _freeze_result_verifier_capsule_v2(
    plan_sha256: _PlanSha256V2,
    resolution_context: _ResolutionContextV2,
    seed_digest: _SeedDigestV2,
    /,
) -> _ResultVerifierCapsuleV2:
    """Freeze the verifier, preserving its public bootstrap and sealed dependencies."""
    types, schemas, externals = _verifier_construction_inputs(seed_digest)
    primitive_ops = freeze_verifier_primitives_v2(types, schemas, externals)
    check_context = _freeze_context_check(externals)
    result_prefix = _freeze_result_prefix(types, schemas, externals, primitive_ops)
    verify_observed = _freeze_observed_verification(types, primitive_ops)
    check_final_context = _freeze_final_context_check(externals)
    expectation_type, counts_type = ReplicateExpectationV2, DerivedCountsV2
    construct_expectation = expectation_type.__new__
    construct_counts = counts_type.__new__

    tuple_getitem: Callable[[tuple[object, ...], int], Any] = tuple.__getitem__

    def verify_calibration_result(result: object, resolution: object, /) -> None:
        """Verify representation, arithmetic and sealed ownership, not input provenance.

        The pair is absent here and retained owner fields are publicly recomputable.
        The public calibrator checks input provenance against its entry-time pair.
        """
        context = check_context(resolution_context(resolution))
        semantic_digest, result_plan_digest, policy, terminal = result_prefix(result, context)
        observed = verify_observed(terminal, policy)
        observed_decision: Any = None
        if tuple_getitem(terminal, 1) is not tuple_getitem(types, 13) and tuple_getitem(
            terminal, 1
        ) is not tuple_getitem(types, 14):
            observed_decision = tuple_getitem(primitive_ops, 2)(
                tuple_getitem(terminal, 3),
                tuple_getitem(schemas, 3),
                subject="observed selection",
            )[2]
        derived_exceedances = 0
        derived_failures = 0
        if tuple_getitem(terminal, 4):
            if (
                tuple_getitem(externals, 5)(tuple_getitem(terminal, 4))
                != tuple_getitem(terminal, 11)
                or observed is None
            ):
                raise tuple_getitem(externals, 0)(
                    "calibration result does not retain exact B outcomes"
                )
            observed_length = tuple_getitem(observed, 0) + tuple_getitem(externals, 6)(
                tuple_getitem(policy, 0)
            )
            sealed_token_verifier = context[10]
            null_adapter = context[9]
            for expected_id, outcome in tuple_getitem(externals, 7)(tuple_getitem(terminal, 4)):
                subject = f"replicate {expected_id}"
                expected = construct_expectation(
                    expectation_type,
                    expected_id,
                    result_plan_digest,
                    tuple_getitem(terminal, 11),
                    tuple_getitem(observed, 0),
                )
                pending = tuple_getitem(primitive_ops, 6)(
                    outcome,
                    expected,
                    policy,
                    subject=subject,
                )
                sealed_token_verifier(
                    null_adapter,
                    tuple_getitem(pending, 4),
                    semantic_digest,
                    result_plan_digest,
                    observed_length,
                    replicate_id=expected_id,
                    planned_replicates=tuple_getitem(terminal, 11),
                )
                verified = tuple_getitem(primitive_ops, 7)(pending, subject=subject)
                if tuple_getitem(verified, 2):
                    derived_failures += 1
                else:
                    if observed_decision is None:
                        raise tuple_getitem(externals, 0)(
                            "replicate result has no observed decision"
                        )
                    decision: Any = tuple_getitem(verified, 1)
                    derived_exceedances += decision >= observed_decision
        derived = construct_counts(
            counts_type,
            derived_exceedances,
            derived_failures,
            tuple_getitem(externals, 5)(tuple_getitem(terminal, 4)),
        )
        tuple_getitem(primitive_ops, 8)(terminal, derived, subject="calibration result")
        final_context = check_context(resolution_context(resolution))
        check_final_context(context, final_context, result_plan_digest)

    return _ResultVerifierCapsuleV2(
        verify=verify_calibration_result,
        external_leaves=(plan_sha256, resolution_context, seed_digest),
    )


_RESULT_VERIFIER_BOOTSTRAPPED_V2 = False


def __getattr__(name: str) -> object:
    if name == "verify_calibration_result" and not _RESULT_VERIFIER_BOOTSTRAPPED_V2:
        raise V2IntegrityError("result verifier capsule is not initialized")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _bootstrap_result_verifier_v2(
    plan_sha256: _PlanSha256V2,
    resolution_context: _ResolutionContextV2,
    seed_digest: _SeedDigestV2,
    /,
) -> None:
    """Install the canonical verifier exactly once during module initialization."""

    global _RESULT_VERIFIER_BOOTSTRAPPED_V2, verify_calibration_result
    if _RESULT_VERIFIER_BOOTSTRAPPED_V2:
        raise V2IntegrityError("result verifier capsule was already bootstrapped")
    capsule = _freeze_result_verifier_capsule_v2(
        plan_sha256,
        resolution_context,
        seed_digest,
    )
    if (
        type(capsule) is not _ResultVerifierCapsuleV2
        or type(capsule.verify) is not FunctionType
        or type(capsule.external_leaves) is not tuple
    ):
        raise V2IntegrityError("result verifier capsule construction is invalid")
    published_verifier = capsule.verify
    try:
        published_verifier.__module__ = "selcal.contracts_v2"
        published_verifier.__name__ = "verify_calibration_result"
        published_verifier.__qualname__ = "verify_calibration_result"
    except (AttributeError, TypeError, ValueError) as error:
        raise V2IntegrityError("result verifier public identity is invalid") from error
    verify_calibration_result = published_verifier
    _RESULT_VERIFIER_BOOTSTRAPPED_V2 = True
