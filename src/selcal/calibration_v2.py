"""Internal observed-path preparation for SelCal v2 calibration."""

from __future__ import annotations

import dis
import inspect
import math
from collections.abc import Callable, Mapping
from dataclasses import InitVar, dataclass, fields
from types import MemberDescriptorType
from typing import Any, NamedTuple, Protocol, cast
from weakref import WeakKeyDictionary

import selcal.contracts_v2 as _contracts_v2_module
from selcal.canonical import canonical_json_bytes, semantic_input_sha256
from selcal.canonical_v2 import scientific_plan_v2_sha256
from selcal.contracts import (
    JsonValue,
    ReplicateStatus,
    RunStatus,
    SelectionResult,
    SelectionRule,
    SeriesPair,
    StatisticResult,
    Validity,
)
from selcal.contracts_v2 import (
    CalibrationResult,
    NullBindStatus,
    NullDisabledReason,
    NullTransformResult,
    NullTransformToken,
    PlanVersionError,
    ReplicateFailureStage,
    ReplicateOutcome,
    ResourceLimitError,
    RunFailureStage,
    V2IntegrityError,
)
from selcal.nulls.executable_base import (
    BoundNullModel,
    NullBindResult,
    _build_random_index_capability_v2,
)
from selcal.nulls.owned_transform_v2 import (
    owner_digest,
    parameter_digest,
    require_sha256,
)
from selcal.randomness import _RANDOM_CAPSULE_V2, _RandomCapsuleV2
from selcal.resolution_v2 import (
    PlanResolutionV2,
    _bound_null_execution_snapshot_matches,
    _bound_statistic_execution_snapshot_matches,
    _BoundNullExecutionSnapshotV2,
    _BoundStatisticExecutionSnapshotV2,
    _exact_series_pair_read_view,
    _exact_snapshot_value_matches,
    _FrozenOperationV2,
    _require_resolver_owned_resolution_v2,
    _require_resolver_owned_resolution_v2_pure,
    _snapshot_registered_bound_null_v2,
    _snapshot_registered_bound_statistic_v2,
)
from selcal.result_wire import _encode_replicate_outcome
from selcal.selection_v2 import select_family_v2
from selcal.statistics.base import BoundStatisticAdapter

verify_calibration_result = _contracts_v2_module.verify_calibration_result

__all__ = ["calibrate_selected_family", "verify_calibration_result"]

_PREPARATION_SEAL = object()
_MAX_IN_MEMORY_REPLICATES_V2 = 1_000
_MAX_IN_MEMORY_CANDIDATE_EVALUATIONS_V2 = 5_000
_MAX_IN_MEMORY_TOKEN_STATE_UNITS_V2 = 25_000
_MAX_IN_MEMORY_WORK_UNITS_V2 = 2_500_000


class _BudgetValuesGuard(Protocol):
    def __call__(self, *, B: int, C: int, N: int, S: int) -> None: ...


def _freeze_in_memory_execution_budget_values(
    max_replicates: int,
    max_candidate_evaluations: int,
    max_token_state_units: int,
    max_work_units: int,
    /,
) -> _BudgetValuesGuard:
    """Capture the versioned caps and typed failures outside later module lookups."""

    exact_type = type
    integrity_error_type = V2IntegrityError
    resource_error_type = ResourceLimitError
    max_safe_operand_bits = 256
    oversized_integer_message = (
        "IN_MEMORY_EXECUTION_BUDGET_EXCEEDED_V2: "
        "integer magnitude exceeds safe diagnostic envelope; "
        f"caps: B<={max_replicates}, BC<={max_candidate_evaluations}, "
        f"BS<={max_token_state_units}, work<={max_work_units}"
    )

    def _require_in_memory_execution_budget_values(
        *,
        B: int,
        C: int,
        N: int,
        S: int,
    ) -> None:
        """Enforce the versioned integer envelope before replicate allocation."""

        for name, value in (("B", B), ("C", C), ("N", N), ("S", S)):
            if exact_type(value) is not int or value < 1:
                raise integrity_error_type(f"{name} must be a positive built-in integer")
        if any(
            value.bit_length() > max_safe_operand_bits
            for value in (B, C, N, S)
        ):
            raise resource_error_type(oversized_integer_message)
        candidate_evaluations = B * C
        token_state_units = B * S
        work_units = B * (C * N + S + N)
        if (
            B > max_replicates
            or candidate_evaluations > max_candidate_evaluations
            or token_state_units > max_token_state_units
            or work_units > max_work_units
        ):
            raise resource_error_type(
                "IN_MEMORY_EXECUTION_BUDGET_EXCEEDED_V2: "
                f"B={B}, C={C}, N={N}, S={S}, "
                f"BC={candidate_evaluations}, BS={token_state_units}, work={work_units}; "
                f"caps: B<={max_replicates}, BC<={max_candidate_evaluations}, "
                f"BS<={max_token_state_units}, work<={max_work_units}"
            )

    return _require_in_memory_execution_budget_values


_require_in_memory_execution_budget_values = _freeze_in_memory_execution_budget_values(
    _MAX_IN_MEMORY_REPLICATES_V2,
    _MAX_IN_MEMORY_CANDIDATE_EVALUATIONS_V2,
    _MAX_IN_MEMORY_TOKEN_STATE_UNITS_V2,
    _MAX_IN_MEMORY_WORK_UNITS_V2,
)
del _freeze_in_memory_execution_budget_values


@dataclass(frozen=True, slots=True)
class _BoundStatisticIdentitySnapshot:
    implementation_type: type[object]
    name: str
    parameters_bytes: bytes
    candidates: tuple[int, ...]
    backend_identity: str
    preprocessing_identity: str
    execution: _BoundStatisticExecutionSnapshotV2


@dataclass(frozen=True, slots=True)
class _BoundNullIdentitySnapshot:
    implementation_type: type[object]
    name: str
    parameters_bytes: bytes
    observed_length: int
    total_state_count: int
    null_parameter_sha256: str
    bound_null_owner_sha256: str
    semantic_input_sha256: str
    scientific_plan_sha256: str
    execution: _BoundNullExecutionSnapshotV2


@dataclass(frozen=True, slots=True, eq=False)
class _NullBindIdentitySnapshot:
    status: NullBindStatus
    bound: BoundNullModel | None
    disabled_reason: NullDisabledReason | None
    diagnostics: tuple[str, ...]


@dataclass(frozen=True, slots=True, eq=False, weakref_slot=True)
class _PreparedCalibration:
    """Immutable observed state ready for the later exact-B replicate loop."""

    resolution: PlanResolutionV2
    observed_pair: SeriesPair
    semantic_input_sha256: str
    scientific_plan_sha256: str
    bound_statistic: BoundStatisticAdapter
    bound_null: BoundNullModel
    statistic_snapshot: _BoundStatisticIdentitySnapshot
    null_snapshot: _BoundNullIdentitySnapshot
    null_bind_snapshot: _NullBindIdentitySnapshot
    raw_observed_results: tuple[StatisticResult, ...]
    observed_results: tuple[StatisticResult, ...]
    observed_selection: SelectionResult
    diagnostics: tuple[str, ...]
    seal: InitVar[object] = None

    def __post_init__(self, seal: object) -> None:
        if seal is not _PREPARATION_SEAL:
            raise V2IntegrityError(
                "prepared calibration must be created by the observed-path preparer"
            )
        if type(self.diagnostics) is not tuple or any(
            type(entry) is not str for entry in self.diagnostics
        ):
            raise V2IntegrityError("prepared diagnostics must be an exact string tuple")


class _PreparedBudgetGuard(Protocol):
    def __call__(self, value: object, /) -> _PreparedCalibration: ...


_ReplicateObserver = Callable[[int, bytes], None]


class _ReplicateExecutor(Protocol):
    def __call__(
        self,
        prepared: _PreparedCalibration,
        /,
        *,
        observer: _ReplicateObserver | None,
    ) -> tuple[ReplicateOutcome, ...]: ...


class _TerminalResultVerifier(Protocol):
    def __call__(
        self,
        result: CalibrationResult,
        resolution: PlanResolutionV2,
        *,
        expected_semantic_input_sha256: str,
        expected_scientific_plan_sha256: str,
    ) -> CalibrationResult: ...


def _freeze_runtime_contract_types(
    prepared_type: type[_PreparedCalibration],
    outcome_type: type[ReplicateOutcome],
) -> tuple[
    Callable[[], type[_PreparedCalibration]],
    Callable[[], type[ReplicateOutcome]],
]:
    """Close over the module-init contract types, not their mutable aliases."""

    def require_prepared() -> type[_PreparedCalibration]:
        if _PreparedCalibration is not prepared_type:
            raise V2IntegrityError("prepared calibration canonical type alias drifted")
        return prepared_type

    def require_outcome() -> type[ReplicateOutcome]:
        if ReplicateOutcome is not outcome_type:
            raise V2IntegrityError(
                "replicate outcome canonical type identity alias drifted; "
                "expected exact ReplicateOutcome"
            )
        return outcome_type

    return require_prepared, require_outcome


(
    _require_frozen_prepared_type,
    _require_frozen_outcome_type,
) = _freeze_runtime_contract_types(
    _PreparedCalibration,
    ReplicateOutcome,
)
del _freeze_runtime_contract_types


@dataclass(frozen=True, slots=True, eq=False)
class _PreparedOwnershipSnapshot:
    signature: tuple[object, ...]


_PreparedLightFingerprint = tuple[object, ...]


@dataclass(frozen=True, slots=True, eq=False)
class _ReplicateHistorySnapshot:
    outcome: ReplicateOutcome
    raw_results: tuple[StatisticResult, ...]
    signature: tuple[object, ...]


_Task8BExpectedOutcome = tuple[object, ...]


class _Task8BSelector(Protocol):
    def __call__(
        self,
        results: tuple[StatisticResult, ...],
        expected_candidates: tuple[int, ...],
        rule: SelectionRule | str,
        tie_tolerance: float,
    ) -> tuple[SelectionResult, tuple[StatisticResult, ...]]: ...


_Task8BLightOps = tuple[
    Callable[[_PreparedCalibration], _PreparedLightFingerprint],
    Callable[[_PreparedCalibration, object], None],
    Callable[[_PreparedCalibration], _PreparedLightFingerprint],
    Callable[[object], bool],
    _Task8BSelector,
]


_Task8BSlotSpec = tuple[type[object], tuple[tuple[str, MemberDescriptorType], ...]]
_Task8BTokenGuard = tuple[tuple[object, ...], tuple[object, ...], _Task8BSlotSpec]
_Task8BSignatureLink = tuple[tuple[object, ...], object]
_Task8BEnumMembers = tuple[tuple[type[object], tuple[object, ...]], ...]
_Task8BDataclassFields = Callable[[Any], tuple[Any, ...]]


class _Task8BHistoryIntegrityOps(NamedTuple):
    freeze_slot_spec: Callable[[type[object], _Task8BDataclassFields], _Task8BSlotSpec]
    read_exact_slots: Callable[
        [object, _Task8BSlotSpec, type[V2IntegrityError]],
        tuple[object, ...],
    ]
    exact_value_snapshot: Callable[
        [object, _Task8BEnumMembers, type[V2IntegrityError]],
        tuple[object, ...],
    ]
    snapshot_replicate_token: Callable[
        [object, _PreparedCalibration, bool],
        tuple[object, ...],
    ]
    token_signature: Callable[
        [object, _Task8BTokenGuard, _Task8BHistoryIntegrityOps],
        tuple[object, ...],
    ]
    capture_token_guard: Callable[
        [object, _PreparedCalibration, _Task8BHistoryIntegrityOps],
        _Task8BTokenGuard,
    ]
    result_vector_signature: Callable[
        [object, tuple[tuple[object, ...], ...] | None, _Task8BHistoryIntegrityOps],
        tuple[object, ...],
    ]
    selection_signature: Callable[
        [object, tuple[object, ...] | None, _Task8BHistoryIntegrityOps],
        tuple[object, ...],
    ]
    outcome_signature: Callable[
        [
            object,
            object,
            tuple[object, ...],
            _Task8BTokenGuard,
            tuple[tuple[object, ...], ...] | None,
            tuple[tuple[object, ...], ...] | None,
            tuple[object, ...] | None,
            _Task8BHistoryIntegrityOps,
        ],
        tuple[object, ...],
    ]
    register_outcome: Callable[
        [
            ReplicateOutcome,
            tuple[StatisticResult, ...],
            _PreparedCalibration,
            _Task8BExpectedOutcome,
            _Task8BHistoryIntegrityOps,
        ],
        tuple[object, ...],
    ]
    capture_terminal_tokens: Callable[
        [
            _PreparedCalibration,
            tuple[_ReplicateHistorySnapshot, ...],
            _Task8BHistoryIntegrityOps,
        ],
        tuple[_Task8BTokenGuard, ...],
    ]
    terminal_history: Callable[
        [
            tuple[_ReplicateHistorySnapshot, ...],
            tuple[_Task8BTokenGuard, ...],
            tuple[tuple[object, ...], ...],
            _Task8BHistoryIntegrityOps,
        ],
        tuple[ReplicateOutcome, ...],
    ]
    exact_match: Callable[[object, object], bool]
    dataclass_fields: _Task8BDataclassFields
    history_slot_spec: _Task8BSlotSpec
    outcome_slot_spec: _Task8BSlotSpec
    result_slot_spec: _Task8BSlotSpec
    selection_slot_spec: _Task8BSlotSpec
    token_slot_spec: _Task8BSlotSpec
    enum_members: _Task8BEnumMembers
    token_type: type[NullTransformToken]
    outcome_validator: Callable[[ReplicateOutcome], None]
    integrity_error: type[V2IntegrityError]


def _freeze_task8b_slot_spec(
    contract_type: type[object],
    dataclass_fields: _Task8BDataclassFields,
) -> _Task8BSlotSpec:
    descriptors: list[tuple[str, MemberDescriptorType]] = []
    for field in dataclass_fields(cast(Any, contract_type)):
        field_name = field.name
        descriptor = contract_type.__dict__.get(field_name)
        if type(descriptor) is not MemberDescriptorType:
            raise RuntimeError(
                f"Task 8B canonical slot {contract_type.__name__}.{field_name} is invalid"
            )
        descriptors.append((field_name, descriptor))
    return contract_type, tuple(descriptors)


_TASK8B_DATACLASS_FIELDS = cast(_Task8BDataclassFields, fields)
_TASK8B_HISTORY_SLOT_SPEC = _freeze_task8b_slot_spec(
    _ReplicateHistorySnapshot, _TASK8B_DATACLASS_FIELDS
)
_TASK8B_OUTCOME_SLOT_SPEC = _freeze_task8b_slot_spec(ReplicateOutcome, _TASK8B_DATACLASS_FIELDS)
_TASK8B_RESULT_SLOT_SPEC = _freeze_task8b_slot_spec(StatisticResult, _TASK8B_DATACLASS_FIELDS)
_TASK8B_SELECTION_SLOT_SPEC = _freeze_task8b_slot_spec(SelectionResult, _TASK8B_DATACLASS_FIELDS)
_TASK8B_TOKEN_SLOT_SPEC = _freeze_task8b_slot_spec(NullTransformToken, _TASK8B_DATACLASS_FIELDS)
_TASK8B_ENUM_MEMBERS: _Task8BEnumMembers = (
    (Validity, tuple(Validity)),
    (ReplicateStatus, tuple(ReplicateStatus)),
    (ReplicateFailureStage, tuple(ReplicateFailureStage)),
)
_TASK8B_TOKEN_TYPE = NullTransformToken
_TASK8B_OUTCOME_VALIDATOR = ReplicateOutcome.__post_init__


def _task8b_read_exact_slots(
    value: object,
    spec: _Task8BSlotSpec,
    integrity_error: type[V2IntegrityError],
) -> tuple[object, ...]:
    contract_type, descriptors = spec
    if type(value) is not contract_type:
        raise integrity_error(
            f"Task 8B history requires exact {contract_type.__name__}"
        )
    try:
        fields: list[object] = []
        for field_name, descriptor in descriptors:
            if contract_type.__dict__.get(field_name) is not descriptor:
                raise integrity_error(
                    "Task 8B canonical member descriptor drifted: "
                    f"{contract_type.__name__}.{field_name}"
                )
            fields.append(descriptor.__get__(value, contract_type))
        return tuple(fields)
    except integrity_error:
        raise
    except (AttributeError, TypeError, ValueError, OverflowError) as error:
        raise integrity_error("Task 8B history exact slot snapshot is invalid") from error


def _task8b_exact_value_snapshot(
    value: object,
    enum_members: _Task8BEnumMembers,
    integrity_error: type[V2IntegrityError],
) -> tuple[object, ...]:
    """Snapshot only exact immutable values admitted by Task 8B contracts."""

    def visit(current: object) -> tuple[object, ...]:
        value_type = type(current)
        if current is None:
            return (0,)
        if value_type in (bool, int, str, bytes):
            return (id(value_type), current)
        if value_type is float:
            return (id(float), cast(float, current).hex())
        if value_type is type:
            return (id(type), id(current))
        if value_type is tuple:
            return (
                id(tuple),
                tuple(visit(item) for item in cast(tuple[object, ...], current)),
            )
        for enum_type, members in enum_members:
            if value_type is enum_type and any(current is member for member in members):
                return (id(enum_type), id(current))
        raise integrity_error(
            "Task 8B outcome contract contains an unsupported exact value or subclass"
        )

    return visit(value)


def _task8b_canonical_snapshot_matches(current: object, initial: object) -> bool:
    value_type = type(current)
    if value_type is not type(initial):
        return False
    if value_type is tuple:
        current_tuple = cast(tuple[object, ...], current)
        initial_tuple = cast(tuple[object, ...], initial)
        return len(current_tuple) == len(initial_tuple) and all(
            _task8b_canonical_snapshot_matches(left, right)
            for left, right in zip(current_tuple, initial_tuple, strict=True)
        )
    return value_type in (bool, int, str, bytes) and bool(current == initial)


def _task8b_trusted_canonical_snapshots_match(current: object, initial: object) -> bool:
    return type(current) is tuple and type(initial) is tuple and current == initial


def _task8b_token_signature(
    token: object,
    guard: _Task8BTokenGuard,
    ops: _Task8BHistoryIntegrityOps,
) -> tuple[object, ...]:
    frozen_canonical, _, state_spec = guard
    token_fields = ops.read_exact_slots(token, ops.token_slot_spec, ops.integrity_error)
    state = token_fields[-1]
    state_fields = ops.read_exact_slots(state, state_spec, ops.integrity_error)
    direct_snapshot = (*token_fields[:-1], type(state), state_fields, id(token), id(state))
    direct_canonical = ops.exact_value_snapshot(
        direct_snapshot, ops.enum_members, ops.integrity_error
    )
    if not ops.exact_match(direct_canonical, frozen_canonical):
        raise ops.integrity_error("Task 8B outcome token snapshot drifted")
    return id(token), id(state), frozen_canonical


def _capture_task8b_token_guard(
    token: object,
    prepared: _PreparedCalibration,
    ops: _Task8BHistoryIntegrityOps,
) -> _Task8BTokenGuard:
    snapshot = ops.snapshot_replicate_token(token, prepared, True)
    if type(snapshot[7]) is not type:
        raise ops.integrity_error("Task 8B token state type snapshot is invalid")
    try:
        state_spec = ops.freeze_slot_spec(
            cast(type[object], snapshot[7]),
            ops.dataclass_fields,
        )
    except (TypeError, ValueError) as error:
        raise ops.integrity_error("Task 8B token state slot snapshot is invalid") from error
    canonical_snapshot = ops.exact_value_snapshot(
        snapshot, ops.enum_members, ops.integrity_error
    )
    canonical_prefix = ops.exact_value_snapshot(
        snapshot[:9], ops.enum_members, ops.integrity_error
    )
    guard = (canonical_snapshot, canonical_prefix, state_spec)
    ops.token_signature(token, guard, ops)
    return guard


def _task8b_result_vector_signature(
    results: object,
    frozen_fields: tuple[tuple[object, ...], ...] | None,
    ops: _Task8BHistoryIntegrityOps,
) -> tuple[object, ...]:
    if type(results) is not tuple:
        raise ops.integrity_error("Task 8B history result vector must be an exact tuple")
    exact_results = cast(tuple[object, ...], results)
    if any(type(result) is not ops.result_slot_spec[0] for result in exact_results):
        raise ops.integrity_error("Task 8B history result vector contains a subclass")
    field_vectors = tuple(
        ops.read_exact_slots(result, ops.result_slot_spec, ops.integrity_error)
        for result in exact_results
    )

    def signature(fields: tuple[tuple[object, ...], ...]) -> tuple[object, ...]:
        rows = tuple(
            (id(result), ops.exact_value_snapshot(row, ops.enum_members, ops.integrity_error))
            for result, row in zip(exact_results, fields, strict=True)
        )
        return id(exact_results), rows

    current_signature = signature(field_vectors)
    if frozen_fields is not None:
        if type(frozen_fields) is not tuple or len(frozen_fields) != len(exact_results):
            raise ops.integrity_error("Task 8B frozen result vector is invalid")
        if not ops.exact_match(current_signature, signature(frozen_fields)):
            raise ops.integrity_error("Task 8B result vector drifted from its frozen snapshot")
    return current_signature


def _task8b_selection_signature(
    selection: object,
    frozen_fields: tuple[object, ...] | None,
    ops: _Task8BHistoryIntegrityOps,
) -> tuple[object, ...]:
    if selection is None:
        if frozen_fields is not None:
            raise ops.integrity_error("Task 8B absent selection has frozen fields")
        return (0,)
    fields = ops.read_exact_slots(selection, ops.selection_slot_spec, ops.integrity_error)
    snapshot = ops.exact_value_snapshot(fields, ops.enum_members, ops.integrity_error)
    current_signature = (id(selection), snapshot)
    if frozen_fields is not None:
        expected_snapshot = ops.exact_value_snapshot(
            frozen_fields, ops.enum_members, ops.integrity_error
        )
        if not ops.exact_match(current_signature, (id(selection), expected_snapshot)):
            raise ops.integrity_error("Task 8B selection drifted from its frozen snapshot")
    return current_signature


def _task8b_outcome_signature_from_fields(
    outcome: object,
    raw_results: object,
    outcome_fields: tuple[object, ...],
    token_guard: _Task8BTokenGuard,
    raw_fields: tuple[tuple[object, ...], ...] | None,
    result_fields: tuple[tuple[object, ...], ...] | None,
    selection_fields: tuple[object, ...] | None,
    ops: _Task8BHistoryIntegrityOps,
) -> tuple[object, ...]:
    if type(outcome_fields) is not tuple or len(outcome_fields) != 8:
        raise ops.integrity_error("Task 8B outcome field snapshot is invalid")
    (
        replicate_id,
        seed_digest,
        status,
        failure_stage,
        token,
        results,
        selection,
        diagnostics,
    ) = outcome_fields
    if type(token) is not ops.token_type:
        raise ops.integrity_error("Task 8B outcome token must be exact")
    if type(diagnostics) is not tuple:
        raise ops.integrity_error("Task 8B outcome diagnostics must be an exact tuple")
    return (
        id(outcome),
        ops.exact_value_snapshot(replicate_id, ops.enum_members, ops.integrity_error),
        ops.exact_value_snapshot(seed_digest, ops.enum_members, ops.integrity_error),
        ops.exact_value_snapshot(status, ops.enum_members, ops.integrity_error),
        ops.exact_value_snapshot(failure_stage, ops.enum_members, ops.integrity_error),
        ops.token_signature(token, token_guard, ops),
        ops.result_vector_signature(raw_results, raw_fields, ops),
        ops.result_vector_signature(results, result_fields, ops),
        ops.selection_signature(selection, selection_fields, ops),
        (
            id(diagnostics),
            ops.exact_value_snapshot(diagnostics, ops.enum_members, ops.integrity_error),
        ),
    )


def _register_task8b_outcome_signature(
    outcome: ReplicateOutcome,
    raw_results: tuple[StatisticResult, ...],
    prepared: _PreparedCalibration,
    expected_contract: _Task8BExpectedOutcome,
    ops: _Task8BHistoryIntegrityOps,
) -> tuple[object, ...]:
    if type(expected_contract) is not tuple or len(expected_contract) != 12:
        raise ops.integrity_error("replicate outcome expected contract is invalid")
    ops.outcome_validator(outcome)
    current_fields = ops.read_exact_slots(
        outcome,
        ops.outcome_slot_spec,
        ops.integrity_error,
    )
    token_guard = ops.capture_token_guard(current_fields[4], prepared, ops)
    expected_token = ops.exact_value_snapshot(
        expected_contract[5], ops.enum_members, ops.integrity_error
    )
    if not ops.exact_match(token_guard[1], expected_token):
        raise ops.integrity_error("replicate outcome token contradicts its execution contract")
    current_fields = ops.read_exact_slots(
        outcome,
        ops.outcome_slot_spec,
        ops.integrity_error,
    )
    (replicate_id, seed_digest, status, failure_stage, token, results, selection, diagnostics) = (
        current_fields
    )
    if (
        type(replicate_id) is not int or replicate_id != expected_contract[0]
        or type(seed_digest) is not str or seed_digest != expected_contract[1]
        or type(status) is not ReplicateStatus or status is not expected_contract[2]
        or failure_stage is not expected_contract[3]
        or (failure_stage is not None and type(failure_stage) is not ReplicateFailureStage)
        or type(token) is not ops.token_type or token is not expected_contract[4]
        or results is not expected_contract[6]
        or selection is not expected_contract[8]
        or type(diagnostics) is not tuple or any(type(item) is not str for item in diagnostics)
        or diagnostics is not expected_contract[10]
    ):
        raise ops.integrity_error("replicate outcome contradicts its execution contract")
    return ops.outcome_signature(
        outcome,
        raw_results,
        current_fields,
        token_guard,
        cast(tuple[tuple[object, ...], ...], expected_contract[11]),
        cast(tuple[tuple[object, ...], ...], expected_contract[7]),
        cast(tuple[object, ...] | None, expected_contract[9]),
        ops,
    )


def _capture_task8b_terminal_token_snapshots(
    prepared: _PreparedCalibration,
    history: tuple[_ReplicateHistorySnapshot, ...],
    ops: _Task8BHistoryIntegrityOps,
) -> tuple[_Task8BTokenGuard, ...]:
    snapshots: list[_Task8BTokenGuard] = []
    for record in history:
        outcome, _, _ = ops.read_exact_slots(
            record,
            ops.history_slot_spec,
            ops.integrity_error,
        )
        outcome_fields = ops.read_exact_slots(
            outcome,
            ops.outcome_slot_spec,
            ops.integrity_error,
        )
        snapshots.append(ops.capture_token_guard(outcome_fields[4], prepared, ops))
    return tuple(snapshots)


def _task8b_terminal_history_snapshot(
    history: tuple[_ReplicateHistorySnapshot, ...],
    token_snapshots: tuple[_Task8BTokenGuard, ...],
    registered_signatures: tuple[tuple[object, ...], ...],
    ops: _Task8BHistoryIntegrityOps,
) -> tuple[ReplicateOutcome, ...]:
    containers = history, token_snapshots, registered_signatures
    if any(type(container) is not tuple for container in containers):
        raise ops.integrity_error("Task 8B terminal history containers must be exact")
    if len(history) != len(token_snapshots) or len(history) != len(registered_signatures):
        raise ops.integrity_error("Task 8B terminal token history is incomplete")
    outcomes: list[ReplicateOutcome] = []
    rows = zip(history, token_snapshots, registered_signatures, strict=True)
    for record, token_guard, registered_signature in rows:
        outcome, raw_results, record_signature = ops.read_exact_slots(
            record,
            ops.history_slot_spec,
            ops.integrity_error,
        )
        if record_signature is not registered_signature:
            raise ops.integrity_error("Task 8B registered history signature drifted")
        outcome_fields = ops.read_exact_slots(
            outcome,
            ops.outcome_slot_spec,
            ops.integrity_error,
        )
        current_signature = ops.outcome_signature(
            outcome,
            raw_results,
            outcome_fields,
            token_guard,
            None,
            None,
            None,
            ops,
        )
        if not ops.exact_match(current_signature, registered_signature):
            raise ops.integrity_error("Task 8B terminal history snapshot drifted before return")
        outcomes.append(cast(ReplicateOutcome, outcome))
    return tuple(outcomes)


_PREPARER_OWNED: WeakKeyDictionary[_PreparedCalibration, _PreparedOwnershipSnapshot] = (
    WeakKeyDictionary()
)


def _snapshot_bound_statistic(
    bound: object,
    resolution: PlanResolutionV2,
    *,
    observed_pair: SeriesPair,
    initial_execution: _BoundStatisticExecutionSnapshotV2 | None = None,
) -> tuple[BoundStatisticAdapter, _BoundStatisticIdentitySnapshot]:
    try:
        planned_name = resolution.plan.statistic_name
        planned_parameters = resolution.plan.statistic_params
        planned_candidates = resolution.plan.candidates
        name = bound.name  # type: ignore[attr-defined]
        parameters = bound.parameters  # type: ignore[attr-defined]
        candidates = bound.candidates  # type: ignore[attr-defined]
        backend_identity = bound.backend_identity  # type: ignore[attr-defined]
        preprocessing_identity = bound.preprocessing_identity  # type: ignore[attr-defined]
        static_evaluate_all = inspect.getattr_static(bound, "evaluate_all")
    except (AttributeError, TypeError, ValueError, OverflowError) as error:
        raise V2IntegrityError("bound statistic identity snapshot is invalid") from error
    if type(name) is not str or name != planned_name:
        raise V2IntegrityError("bound statistic name does not match the resolved plan")
    if not isinstance(parameters, Mapping):
        raise V2IntegrityError("bound statistic parameters must be a mapping")
    try:
        parameters_bytes = canonical_json_bytes(cast(Mapping[str, JsonValue], parameters))
        planned_parameters_bytes = canonical_json_bytes(planned_parameters)
    except (AttributeError, TypeError, ValueError, OverflowError) as error:
        raise V2IntegrityError("bound statistic parameters are invalid") from error
    if parameters_bytes != planned_parameters_bytes:
        raise V2IntegrityError("bound statistic parameters do not match the resolved plan")
    if (
        type(candidates) is not tuple
        or any(type(candidate) is not int for candidate in candidates)
        or candidates != planned_candidates
    ):
        raise V2IntegrityError("bound statistic candidates do not match the resolved plan")
    for identity_name, identity in (
        ("backend_identity", backend_identity),
        ("preprocessing_identity", preprocessing_identity),
    ):
        if type(identity) is not str or not identity:
            raise V2IntegrityError(f"bound statistic {identity_name} is invalid")
    if not callable(static_evaluate_all) and not hasattr(type(static_evaluate_all), "__get__"):
        raise V2IntegrityError("bound statistic evaluate_all must be callable")
    execution = _snapshot_registered_bound_statistic_v2(
        resolution,
        bound,
        expected_pair=observed_pair,
        initial=initial_execution,
    )
    exact_bound = cast(BoundStatisticAdapter, bound)
    return exact_bound, _BoundStatisticIdentitySnapshot(
        implementation_type=type(bound),
        name=name,
        parameters_bytes=parameters_bytes,
        candidates=candidates,
        backend_identity=backend_identity,
        preprocessing_identity=preprocessing_identity,
        execution=execution,
    )


def _snapshot_bound_null(
    bound: object,
    resolution: PlanResolutionV2,
    *,
    observed_length: int,
    semantic_digest: str,
    plan_digest: str,
    observed_pair: SeriesPair,
    initial_execution: _BoundNullExecutionSnapshotV2 | None = None,
) -> tuple[BoundNullModel, _BoundNullIdentitySnapshot]:
    try:
        planned_name = resolution.plan.null_name
        planned_parameters = resolution.plan.null_params
        name = bound.name  # type: ignore[attr-defined]
        parameters = bound.parameters  # type: ignore[attr-defined]
        bound_observed_length = bound.observed_length  # type: ignore[attr-defined]
        total_state_count = bound.total_state_count  # type: ignore[attr-defined]
        null_parameter_sha256 = bound.null_parameter_sha256  # type: ignore[attr-defined]
        bound_null_owner_sha256 = bound.bound_null_owner_sha256  # type: ignore[attr-defined]
        bound_semantic_digest = bound._semantic_input_sha256  # type: ignore[attr-defined]
        bound_plan_digest = bound._scientific_plan_sha256  # type: ignore[attr-defined]
    except (AttributeError, TypeError, ValueError, OverflowError) as error:
        raise V2IntegrityError("bound null identity snapshot is invalid") from error
    if type(name) is not str or name != planned_name:
        raise V2IntegrityError("bound null name does not match the resolved plan")
    if not isinstance(parameters, Mapping):
        raise V2IntegrityError("bound null parameters must be a mapping")
    try:
        parameters_bytes = canonical_json_bytes(cast(Mapping[str, JsonValue], parameters))
        planned_parameters_bytes = canonical_json_bytes(planned_parameters)
        expected_parameter_digest = parameter_digest(
            null_name=name,
            params=cast(Mapping[str, JsonValue], parameters),
        )
    except (AttributeError, TypeError, ValueError, OverflowError) as error:
        raise V2IntegrityError("bound null parameters are invalid") from error
    if parameters_bytes != planned_parameters_bytes:
        raise V2IntegrityError("bound null parameters do not match the resolved plan")
    if type(bound_observed_length) is not int or bound_observed_length != observed_length:
        raise V2IntegrityError("bound null observed length does not match the input")
    if type(total_state_count) is not int or total_state_count < 1:
        raise V2IntegrityError("bound null total state count is invalid")
    exact_parameter_digest = require_sha256(
        null_parameter_sha256,
        name="bound null parameter digest",
    )
    if exact_parameter_digest != expected_parameter_digest:
        raise V2IntegrityError("bound null parameter digest does not match its parameters")
    exact_owner_digest = require_sha256(
        bound_null_owner_sha256,
        name="bound null owner digest",
    )
    exact_semantic_digest = require_sha256(
        bound_semantic_digest,
        name="bound null semantic input digest",
    )
    exact_plan_digest = require_sha256(
        bound_plan_digest,
        name="bound null scientific plan digest",
    )
    if exact_semantic_digest != semantic_digest or exact_plan_digest != plan_digest:
        raise V2IntegrityError("bound null input or plan identity does not match this run")
    expected_owner_digest = owner_digest(
        semantic_input_sha256=semantic_digest,
        scientific_plan_sha256=plan_digest,
        null_parameter_sha256=expected_parameter_digest,
        observed_length=observed_length,
    )
    if exact_owner_digest != expected_owner_digest:
        raise V2IntegrityError("bound null owner digest does not match this run")
    execution = _snapshot_registered_bound_null_v2(
        resolution,
        bound,
        expected_pair=observed_pair,
        initial=initial_execution,
    )
    exact_bound = cast(BoundNullModel, bound)
    return exact_bound, _BoundNullIdentitySnapshot(
        implementation_type=type(bound),
        name=name,
        parameters_bytes=parameters_bytes,
        observed_length=bound_observed_length,
        total_state_count=total_state_count,
        null_parameter_sha256=exact_parameter_digest,
        bound_null_owner_sha256=exact_owner_digest,
        semantic_input_sha256=exact_semantic_digest,
        scientific_plan_sha256=exact_plan_digest,
        execution=execution,
    )


def _snapshot_input_pair(pair: object) -> SeriesPair:
    if type(pair) is not SeriesPair:
        raise V2IntegrityError("pair must be an exact SeriesPair")
    try:
        view = _exact_series_pair_read_view(pair)
        return SeriesPair(source=view.source, target=view.target)
    except V2IntegrityError as error:
        raise V2IntegrityError("input snapshot is invalid") from error
    except (AttributeError, TypeError, ValueError, OverflowError) as error:
        raise V2IntegrityError("input snapshot is invalid") from error


def _semantic_digest_for_exact_pair(
    pair: SeriesPair,
    candidates: tuple[int, ...],
) -> str:
    view = _exact_series_pair_read_view(pair)
    exact_view_pair = SeriesPair(source=view.source, target=view.target)
    return semantic_input_sha256(exact_view_pair, candidates)


def _verify_bound_statistic_unchanged(
    bound: BoundStatisticAdapter,
    resolution: PlanResolutionV2,
    initial: _BoundStatisticIdentitySnapshot,
    *,
    observed_pair: SeriesPair,
) -> None:
    try:
        current_bound, current = _snapshot_bound_statistic(
            bound,
            resolution,
            observed_pair=observed_pair,
            initial_execution=initial.execution,
        )
    except V2IntegrityError as error:
        raise V2IntegrityError("bound statistic identity snapshot drifted") from error
    if (
        current_bound is not bound
        or current.implementation_type is not initial.implementation_type
        or current.name != initial.name
        or current.parameters_bytes != initial.parameters_bytes
        or current.candidates != initial.candidates
        or current.backend_identity != initial.backend_identity
        or current.preprocessing_identity != initial.preprocessing_identity
        or not _bound_statistic_execution_snapshot_matches(
            current.execution,
            initial.execution,
        )
    ):
        raise V2IntegrityError("bound statistic identity snapshot drifted")


def _verify_bound_null_unchanged(
    bound: BoundNullModel,
    resolution: PlanResolutionV2,
    initial: _BoundNullIdentitySnapshot,
    *,
    observed_pair: SeriesPair,
    observed_length: int,
    semantic_digest: str,
    plan_digest: str,
) -> None:
    try:
        current_bound, current = _snapshot_bound_null(
            bound,
            resolution,
            observed_length=observed_length,
            semantic_digest=semantic_digest,
            plan_digest=plan_digest,
            observed_pair=observed_pair,
            initial_execution=initial.execution,
        )
    except V2IntegrityError as error:
        raise V2IntegrityError("bound null identity snapshot drifted") from error
    if (
        current_bound is not bound
        or current.implementation_type is not initial.implementation_type
        or current.name != initial.name
        or current.parameters_bytes != initial.parameters_bytes
        or current.observed_length != initial.observed_length
        or current.total_state_count != initial.total_state_count
        or current.null_parameter_sha256 != initial.null_parameter_sha256
        or current.bound_null_owner_sha256 != initial.bound_null_owner_sha256
        or current.semantic_input_sha256 != initial.semantic_input_sha256
        or current.scientific_plan_sha256 != initial.scientific_plan_sha256
        or not _bound_null_execution_snapshot_matches(
            current.execution,
            initial.execution,
        )
    ):
        raise V2IntegrityError("bound null identity snapshot drifted")


def _validate_raw_observed_vector(
    results: object,
    statistic_snapshot: _BoundStatisticIdentitySnapshot,
    resolution: PlanResolutionV2,
) -> tuple[StatisticResult, ...]:
    if type(results) is not tuple or any(type(result) is not StatisticResult for result in results):
        raise V2IntegrityError("observed candidate vector must be an exact result tuple")
    exact = cast(tuple[StatisticResult, ...], results)
    snapshots = tuple(
        _snapshot_statistic_result(result, subject="observed result") for result in exact
    )
    if tuple(snapshot[0] for snapshot in snapshots) != resolution.plan.candidates:
        raise V2IntegrityError("observed candidate vector must exactly match the plan")
    if any(snapshot[2] is not None for snapshot in snapshots):
        raise V2IntegrityError("observed raw candidate vector must be unscored")
    for snapshot in snapshots:
        backend_identity = cast(str, snapshot[6])
        preprocessing_identity = cast(str, snapshot[7])
        if (
            backend_identity != statistic_snapshot.backend_identity
            or preprocessing_identity != statistic_snapshot.preprocessing_identity
        ):
            raise V2IntegrityError("observed result implementation identity drifted")
    return exact


def _validate_raw_replicate_vector(
    results: object,
    prepared: _PreparedCalibration,
    *,
    replicate_id: int,
) -> tuple[StatisticResult, ...]:
    if type(results) is not tuple or any(type(result) is not StatisticResult for result in results):
        raise V2IntegrityError(
            f"replicate {replicate_id} candidate vector must be an exact result tuple"
        )
    exact = cast(tuple[StatisticResult, ...], results)
    snapshots = tuple(
        _snapshot_statistic_result(
            result,
            subject=f"replicate {replicate_id} result",
        )
        for result in exact
    )
    if tuple(snapshot[0] for snapshot in snapshots) != prepared.resolution.plan.candidates:
        raise V2IntegrityError(
            f"replicate {replicate_id} candidate vector must exactly match the plan"
        )
    if any(snapshot[2] is not None for snapshot in snapshots):
        raise V2IntegrityError(f"replicate {replicate_id} raw candidate vector must be unscored")
    for snapshot in snapshots:
        if (
            snapshot[6] != prepared.statistic_snapshot.backend_identity
            or snapshot[7] != prepared.statistic_snapshot.preprocessing_identity
        ):
            raise V2IntegrityError(
                f"replicate {replicate_id} result implementation identity drifted"
            )
    return exact


def _snapshot_replicate_token(
    token: object,
    prepared: _PreparedCalibration,
    include_identity: bool = False,
) -> tuple[object, ...]:
    try:
        snapshot_value = prepared.null_snapshot.execution.snapshot_token.callable(token)
    except (AttributeError, TypeError, ValueError, OverflowError) as error:
        raise V2IntegrityError("sampled token snapshot is invalid") from error
    if type(snapshot_value) is not tuple or len(snapshot_value) != 11:
        raise V2IntegrityError("sampled token snapshot is not an exact immutable tuple")
    snapshot = snapshot_value

    def is_immutable_snapshot_value(value: object) -> bool:
        if type(value) in {bool, int, float, str, bytes, type}:
            return True
        if type(value) is tuple:
            return all(is_immutable_snapshot_value(item) for item in value)
        return False

    if not all(is_immutable_snapshot_value(value) for value in snapshot):
        raise V2IntegrityError("sampled token snapshot contains a mutable value")
    schema = snapshot[0]
    null_name = snapshot[1]
    null_parameter_sha256 = snapshot[2]
    semantic_input_sha256_value = snapshot[3]
    scientific_plan_sha256_value = snapshot[4]
    bound_null_owner_sha256 = snapshot[5]
    is_identity = snapshot[6]
    if type(schema) is not str or schema != "selcal.null-transform-token.v2":
        raise V2IntegrityError("sampled token schema drifted")
    if type(null_name) is not str or null_name != prepared.null_snapshot.name:
        raise V2IntegrityError("sampled token null identity drifted")
    expected_digests = (
        (null_parameter_sha256, prepared.null_snapshot.null_parameter_sha256),
        (semantic_input_sha256_value, prepared.semantic_input_sha256),
        (scientific_plan_sha256_value, prepared.scientific_plan_sha256),
        (bound_null_owner_sha256, prepared.null_snapshot.bound_null_owner_sha256),
    )
    if any(type(actual) is not str or actual != expected for actual, expected in expected_digests):
        raise V2IntegrityError("sampled token ownership identity drifted")
    if type(is_identity) is not bool:
        raise V2IntegrityError("sampled token identity flag drifted")
    return snapshot if include_identity else snapshot[:9]


_TASK8B_HISTORY_INTEGRITY_OPS = _Task8BHistoryIntegrityOps(
    freeze_slot_spec=_freeze_task8b_slot_spec,
    read_exact_slots=_task8b_read_exact_slots,
    exact_value_snapshot=_task8b_exact_value_snapshot,
    snapshot_replicate_token=_snapshot_replicate_token,
    token_signature=_task8b_token_signature,
    capture_token_guard=_capture_task8b_token_guard,
    result_vector_signature=_task8b_result_vector_signature,
    selection_signature=_task8b_selection_signature,
    outcome_signature=_task8b_outcome_signature_from_fields,
    register_outcome=_register_task8b_outcome_signature,
    capture_terminal_tokens=_capture_task8b_terminal_token_snapshots,
    terminal_history=_task8b_terminal_history_snapshot,
    exact_match=_task8b_trusted_canonical_snapshots_match,
    dataclass_fields=_TASK8B_DATACLASS_FIELDS,
    history_slot_spec=_TASK8B_HISTORY_SLOT_SPEC,
    outcome_slot_spec=_TASK8B_OUTCOME_SLOT_SPEC,
    result_slot_spec=_TASK8B_RESULT_SLOT_SPEC,
    selection_slot_spec=_TASK8B_SELECTION_SLOT_SPEC,
    token_slot_spec=_TASK8B_TOKEN_SLOT_SPEC,
    enum_members=_TASK8B_ENUM_MEMBERS,
    token_type=_TASK8B_TOKEN_TYPE,
    outcome_validator=_TASK8B_OUTCOME_VALIDATOR,
    integrity_error=V2IntegrityError,
)


def _validate_transform_result(
    result: object,
    *,
    sampled_token: NullTransformToken,
    sampled_token_snapshot: tuple[object, ...],
    prepared: _PreparedCalibration,
) -> tuple[SeriesPair, tuple[str, ...]]:
    if type(result) is not NullTransformResult:
        raise V2IntegrityError("null apply must return an exact NullTransformResult")
    try:
        transformed_pair = result.pair
        returned_token = result.token
        source_changed = result.source_changed
        diagnostics = result.diagnostics
    except (AttributeError, TypeError, ValueError, OverflowError) as error:
        raise V2IntegrityError("null transform result snapshot is invalid") from error
    if _snapshot_replicate_token(returned_token, prepared) != sampled_token_snapshot:
        raise V2IntegrityError("sampled token drifted during null application")
    if type(source_changed) is not bool:
        raise V2IntegrityError("null transform source_changed must be a built-in bool")
    if type(diagnostics) is not tuple or any(type(item) is not str for item in diagnostics):
        raise V2IntegrityError("null transform diagnostics must be an exact string tuple")
    exact_pair = _snapshot_input_pair(transformed_pair)
    observed_view = _exact_series_pair_read_view(prepared.observed_pair)
    transformed_view = _exact_series_pair_read_view(exact_pair)
    if (
        transformed_view.source.shape != observed_view.source.shape
        or transformed_view.target.shape != observed_view.target.shape
    ):
        raise V2IntegrityError("null transform output shape drifted")
    if transformed_view.target.tobytes(order="C") != observed_view.target.tobytes(order="C"):
        raise V2IntegrityError("null transform target identity drifted")
    actual_source_changed = transformed_view.source.tobytes(
        order="C"
    ) != observed_view.source.tobytes(order="C")
    if source_changed is not actual_source_changed:
        raise V2IntegrityError("null transform source_changed contradicts the transformed source")
    return exact_pair, diagnostics


def _snapshot_statistic_result(
    result: StatisticResult,
    *,
    subject: str,
) -> tuple[object, ...]:
    try:
        candidate_id = result.candidate_id
        estimate = result.estimate
        selection_score = result.selection_score
        support_n = result.support_n
        validity = result.validity
        diagnostics = result.diagnostics
        backend_identity = result.backend_identity
        preprocessing_identity = result.preprocessing_identity
    except (AttributeError, TypeError, ValueError, OverflowError) as error:
        raise V2IntegrityError(f"{subject} snapshot is invalid") from error
    if type(candidate_id) is not int or candidate_id < 1:
        raise V2IntegrityError(f"{subject} candidate vector is invalid")
    if type(support_n) is not int or support_n < 1:
        raise V2IntegrityError(f"{subject} support_n must be a positive built-in int")
    if type(diagnostics) is not tuple or any(type(entry) is not str for entry in diagnostics):
        raise V2IntegrityError(f"{subject} diagnostics must be an exact string tuple")
    if (
        type(backend_identity) is not str
        or not backend_identity
        or type(preprocessing_identity) is not str
        or not preprocessing_identity
    ):
        raise V2IntegrityError(f"{subject} implementation identity is invalid")
    if type(validity) is not Validity:
        raise V2IntegrityError(f"{subject} validity is unsupported")
    if validity is Validity.VALID:
        if type(estimate) is not float or not math.isfinite(estimate):
            raise V2IntegrityError(f"{subject} VALID result must have a finite estimate")
        if selection_score is not None and (
            type(selection_score) is not float or not math.isfinite(selection_score)
        ):
            raise V2IntegrityError(f"{subject} selection score must be finite or None")
    elif estimate is not None or selection_score is not None:
        raise V2IntegrityError(f"{subject} analytical failure must not retain an estimate or score")
    return (
        candidate_id,
        estimate,
        selection_score,
        support_n,
        validity,
        diagnostics,
        backend_identity,
        preprocessing_identity,
    )


def _snapshot_selection(selection: SelectionResult) -> tuple[object, ...]:
    if type(selection) is not SelectionResult:
        raise V2IntegrityError("selector must return an exact SelectionResult")
    try:
        selected_candidate = selection.selected_candidate
        selected_index = selection.selected_index
        decision_statistic = selection.decision_statistic
        tied_candidates = selection.tied_candidates
    except (AttributeError, TypeError, ValueError, OverflowError) as error:
        raise V2IntegrityError("selection snapshot is invalid") from error
    if type(selected_candidate) is not int or selected_candidate < 1:
        raise V2IntegrityError("selection candidate is invalid")
    if type(selected_index) is not int or selected_index < 0:
        raise V2IntegrityError("selection index is invalid")
    if type(decision_statistic) is not float or not math.isfinite(decision_statistic):
        raise V2IntegrityError("selection decision statistic is invalid")
    if type(tied_candidates) is not tuple or any(
        type(candidate) is not int or candidate < 1 for candidate in tied_candidates
    ):
        raise V2IntegrityError("selection tied candidates are invalid")
    return (
        selected_candidate,
        selected_index,
        decision_statistic,
        tied_candidates,
    )


def _result_vector_snapshot(
    results: tuple[StatisticResult, ...],
    *,
    subject: str,
) -> tuple[tuple[object, ...], ...]:
    if type(results) is not tuple or any(type(result) is not StatisticResult for result in results):
        raise V2IntegrityError(f"{subject} must be an exact StatisticResult tuple")
    return tuple(_snapshot_statistic_result(result, subject=subject) for result in results)


def _validate_selector_output(
    raw_results: tuple[StatisticResult, ...],
    raw_snapshot: tuple[tuple[object, ...], ...],
    selection: object,
    scored_results: object,
    candidates: tuple[int, ...],
    rule_is_max_upper: bool,
    tie_tolerance: float,
) -> tuple[SelectionResult, tuple[StatisticResult, ...]]:
    current_raw = _result_vector_snapshot(raw_results, subject="raw observed vector")
    if current_raw != raw_snapshot:
        raise V2IntegrityError("selector mutated the raw observed vector")
    if type(scored_results) is not tuple or any(
        type(result) is not StatisticResult for result in scored_results
    ):
        raise V2IntegrityError("selector scored output must be an exact result tuple")
    scored = cast(tuple[StatisticResult, ...], scored_results)
    scored_snapshot = _result_vector_snapshot(scored, subject="scored observed vector")
    if len(scored_snapshot) != len(raw_snapshot):
        raise V2IntegrityError("selector scored vector does not match the observed family")

    if type(rule_is_max_upper) is not bool:
        raise V2IntegrityError("selection rule discriminator identity drifted")
    if type(candidates) is not tuple or any(
        type(candidate) is not int or candidate < 1 for candidate in candidates
    ):
        raise V2IntegrityError("selection candidates identity drifted")
    if type(tie_tolerance) is not float or not math.isfinite(tie_tolerance) or tie_tolerance < 0.0:
        raise V2IntegrityError("selection tie tolerance identity drifted")
    expected_scores: list[float] = []
    for raw, scored_row in zip(raw_snapshot, scored_snapshot, strict=True):
        estimate = cast(float, raw[1])
        score = estimate if rule_is_max_upper else abs(estimate)
        expected_score = 0.0 if score == 0.0 else score
        expected_scores.append(expected_score)
        if (
            scored_row[0] != raw[0]
            or scored_row[1] != raw[1]
            or scored_row[3:] != raw[3:]
            or scored_row[2] != expected_score
        ):
            raise V2IntegrityError(
                "selector scored output is not welded to the raw observed vector"
            )

    exact_selection = cast(SelectionResult, selection)
    selection_snapshot = _snapshot_selection(exact_selection)
    if not expected_scores:
        raise V2IntegrityError("selector cannot select an empty observed family")
    if len(candidates) != len(raw_snapshot) or any(
        raw[0] != candidate for raw, candidate in zip(raw_snapshot, candidates, strict=True)
    ):
        raise V2IntegrityError("selection candidates do not match the observed family")
    decision = max(expected_scores)
    tied_indices = tuple(
        index for index, score in enumerate(expected_scores) if decision - score <= tie_tolerance
    )
    selected_index = tied_indices[0]
    expected_selection = (
        candidates[selected_index],
        selected_index,
        decision,
        tuple(candidates[index] for index in tied_indices),
    )
    if selection_snapshot != expected_selection:
        raise V2IntegrityError("selector selection is inconsistent with scored observations")
    return exact_selection, scored


def _operation_signature(operation: object, *, name: str) -> tuple[int, int]:
    if type(operation) is not _FrozenOperationV2:
        raise V2IntegrityError(f"{name} must be an exact frozen operation")
    try:
        return (id(operation.descriptor), id(operation.callable))
    except (AttributeError, TypeError, ValueError, OverflowError) as error:
        raise V2IntegrityError(f"{name} snapshot is invalid") from error


def _prepared_signature(prepared: _PreparedCalibration) -> tuple[object, ...]:
    """Freeze every prepared field that a later phase is allowed to consume."""

    try:
        statistic_snapshot = prepared.statistic_snapshot
        null_snapshot = prepared.null_snapshot
        null_bind_snapshot = prepared.null_bind_snapshot
        if (
            type(statistic_snapshot) is not _BoundStatisticIdentitySnapshot
            or type(null_snapshot) is not _BoundNullIdentitySnapshot
            or type(null_bind_snapshot) is not _NullBindIdentitySnapshot
        ):
            raise V2IntegrityError("preparer-owned component snapshot must be exact")
        statistic_execution = statistic_snapshot.execution
        null_execution = null_snapshot.execution
        diagnostics = prepared.diagnostics
        if (
            type(statistic_execution) is not _BoundStatisticExecutionSnapshotV2
            or type(null_execution) is not _BoundNullExecutionSnapshotV2
        ):
            raise V2IntegrityError("preparer-owned execution snapshot must be exact")
        if (
            type(statistic_snapshot.implementation_type) is not type
            or type(statistic_snapshot.name) is not str
            or not statistic_snapshot.name
            or type(statistic_snapshot.parameters_bytes) is not bytes
            or type(statistic_snapshot.candidates) is not tuple
            or any(type(candidate) is not int for candidate in statistic_snapshot.candidates)
            or type(statistic_snapshot.backend_identity) is not str
            or not statistic_snapshot.backend_identity
            or type(statistic_snapshot.preprocessing_identity) is not str
            or not statistic_snapshot.preprocessing_identity
            or type(statistic_execution.operational_snapshot) is not tuple
        ):
            raise V2IntegrityError("prepared statistic identity snapshot must be exact")
        if (
            type(null_snapshot.implementation_type) is not type
            or type(null_snapshot.name) is not str
            or not null_snapshot.name
            or type(null_snapshot.parameters_bytes) is not bytes
            or type(null_snapshot.observed_length) is not int
            or null_snapshot.observed_length < 1
            or type(null_snapshot.total_state_count) is not int
            or null_snapshot.total_state_count < 1
            or type(null_execution.operational_snapshot) is not tuple
            or type(null_execution.observed_pair) is not SeriesPair
            or type(null_execution.retained_token_state_units) is not int
            or null_execution.retained_token_state_units < 1
        ):
            raise V2IntegrityError("prepared null identity snapshot must be exact")
        for digest_name, digest in (
            ("prepared null parameter digest", null_snapshot.null_parameter_sha256),
            ("prepared null owner digest", null_snapshot.bound_null_owner_sha256),
            ("prepared null semantic digest", null_snapshot.semantic_input_sha256),
            ("prepared null plan digest", null_snapshot.scientific_plan_sha256),
        ):
            require_sha256(digest, name=digest_name)
        if type(null_bind_snapshot.status) is not NullBindStatus:
            raise V2IntegrityError("prepared null bind status must be exact")
        if type(null_bind_snapshot.diagnostics) is not tuple or any(
            type(entry) is not str for entry in null_bind_snapshot.diagnostics
        ):
            raise V2IntegrityError("prepared null bind diagnostics must be exact")
        if type(diagnostics) is not tuple or any(type(entry) is not str for entry in diagnostics):
            raise V2IntegrityError("prepared diagnostics must be an exact string tuple")
        semantic_digest = require_sha256(
            prepared.semantic_input_sha256,
            name="prepared semantic input digest",
        )
        plan_digest = require_sha256(
            prepared.scientific_plan_sha256,
            name="prepared scientific plan digest",
        )
        resolution = prepared.resolution
        current_plan_digest = scientific_plan_v2_sha256(resolution.plan)
        current_semantic_digest = _semantic_digest_for_exact_pair(
            prepared.observed_pair,
            resolution.plan.candidates,
        )
        raw_fields = _result_vector_snapshot(
            prepared.raw_observed_results,
            subject="raw observed vector",
        )
        scored_fields = _result_vector_snapshot(
            prepared.observed_results,
            subject="scored observed vector",
        )
        selection_fields = _snapshot_selection(prepared.observed_selection)
        return (
            id(prepared.resolution),
            id(resolution.plan),
            id(resolution.adapters),
            id(resolution.adapters.statistic),
            id(resolution.adapters.null_model),
            id(prepared.observed_pair),
            semantic_digest,
            plan_digest,
            current_semantic_digest,
            current_plan_digest,
            id(prepared.bound_statistic),
            id(prepared.bound_null),
            id(statistic_snapshot),
            statistic_snapshot.implementation_type,
            statistic_snapshot.name,
            statistic_snapshot.parameters_bytes,
            statistic_snapshot.candidates,
            statistic_snapshot.backend_identity,
            statistic_snapshot.preprocessing_identity,
            statistic_execution.implementation_type,
            statistic_execution.operational_snapshot,
            _operation_signature(
                statistic_execution.evaluate_all,
                name="prepared statistic evaluate_all",
            ),
            id(null_snapshot),
            null_snapshot.implementation_type,
            null_snapshot.name,
            null_snapshot.parameters_bytes,
            null_snapshot.observed_length,
            null_snapshot.total_state_count,
            null_snapshot.null_parameter_sha256,
            null_snapshot.bound_null_owner_sha256,
            null_snapshot.semantic_input_sha256,
            null_snapshot.scientific_plan_sha256,
            null_execution.implementation_type,
            id(null_execution.observed_pair),
            null_execution.retained_token_state_units,
            null_execution.operational_snapshot,
            _operation_signature(
                null_execution.identity_token,
                name="prepared null identity_token",
            ),
            _operation_signature(
                null_execution.sample_token,
                name="prepared null sample_token",
            ),
            _operation_signature(
                null_execution.snapshot_token,
                name="prepared null snapshot_token",
            ),
            _operation_signature(null_execution.apply, name="prepared null apply"),
            None
            if null_execution.enumerate_token is None
            else _operation_signature(
                null_execution.enumerate_token,
                name="prepared null enumerate_token",
            ),
            id(null_bind_snapshot),
            null_bind_snapshot.status,
            id(null_bind_snapshot.bound),
            null_bind_snapshot.disabled_reason,
            null_bind_snapshot.diagnostics,
            id(prepared.raw_observed_results),
            raw_fields,
            id(prepared.observed_results),
            scored_fields,
            id(prepared.observed_selection),
            selection_fields,
            diagnostics,
        )
    except V2IntegrityError:
        raise
    except (AttributeError, TypeError, ValueError, OverflowError) as error:
        raise V2IntegrityError("preparer-owned calibration snapshot is invalid") from error


def _validate_null_bind_snapshot(
    null_bind: NullBindResult,
) -> _NullBindIdentitySnapshot:
    try:
        status = null_bind.status
        bound = null_bind.bound
        disabled_reason = null_bind.disabled_reason
        diagnostics = null_bind.diagnostics
    except (AttributeError, TypeError, ValueError, OverflowError) as error:
        raise V2IntegrityError("null bind identity snapshot is invalid") from error
    if type(status) is not NullBindStatus:
        raise V2IntegrityError("null bind status must be an exact NullBindStatus")
    if type(diagnostics) is not tuple or any(type(entry) is not str for entry in diagnostics):
        raise V2IntegrityError("null bind diagnostics must be an exact string tuple")
    if status is NullBindStatus.ENABLED:
        if bound is None:
            raise V2IntegrityError("enabled null bind must retain a bound null")
        if disabled_reason is not None:
            raise V2IntegrityError("enabled null bind requires disabled_reason None")
    else:
        if bound is not None:
            raise V2IntegrityError("disabled null bind requires bound None")
        if type(disabled_reason) is not NullDisabledReason:
            raise V2IntegrityError("disabled null bind requires an exact disabled_reason")
    return _NullBindIdentitySnapshot(
        status=status,
        bound=bound,
        disabled_reason=disabled_reason,
        diagnostics=diagnostics,
    )


def _verify_null_bind_unchanged(
    null_bind: NullBindResult,
    initial: _NullBindIdentitySnapshot,
) -> None:
    current = _validate_null_bind_snapshot(null_bind)
    if (
        current.status is not initial.status
        or current.bound is not initial.bound
        or current.disabled_reason is not initial.disabled_reason
        or current.diagnostics != initial.diagnostics
    ):
        raise V2IntegrityError("null bind identity snapshot drifted")


def _reverify_run_identity(
    pair: SeriesPair,
    resolution: PlanResolutionV2,
    *,
    semantic_digest: str,
    plan_digest: str,
) -> None:
    try:
        _require_resolver_owned_resolution_v2(resolution)
        current_plan_digest = scientific_plan_v2_sha256(resolution.plan)
    except V2IntegrityError:
        raise
    except (AttributeError, TypeError, ValueError, OverflowError) as error:
        raise V2IntegrityError("run identity snapshot is invalid") from error
    try:
        current_semantic_digest = _semantic_digest_for_exact_pair(
            pair,
            resolution.plan.candidates,
        )
    except V2IntegrityError as error:
        raise V2IntegrityError("run identity snapshot is invalid") from error
    except (AttributeError, TypeError, ValueError, OverflowError) as error:
        raise V2IntegrityError("run identity snapshot is invalid") from error
    if current_plan_digest != plan_digest:
        raise V2IntegrityError("scientific plan identity snapshot drifted during preparation")
    if current_semantic_digest != semantic_digest:
        raise V2IntegrityError("observed input identity snapshot drifted during preparation")


def _reverify_active_preparation(
    pair: SeriesPair,
    resolution: PlanResolutionV2,
    *,
    semantic_digest: str,
    plan_digest: str,
    bound_statistic: BoundStatisticAdapter,
    statistic_snapshot: _BoundStatisticIdentitySnapshot,
    null_bind: NullBindResult,
    null_bind_snapshot: _NullBindIdentitySnapshot,
    bound_null: BoundNullModel | None,
    null_snapshot: _BoundNullIdentitySnapshot | None,
) -> None:
    _reverify_run_identity(
        pair,
        resolution,
        semantic_digest=semantic_digest,
        plan_digest=plan_digest,
    )
    _verify_bound_statistic_unchanged(
        bound_statistic,
        resolution,
        statistic_snapshot,
        observed_pair=pair,
    )
    _verify_null_bind_unchanged(null_bind, null_bind_snapshot)
    if (bound_null is None) is not (null_snapshot is None):
        raise V2IntegrityError("bound null and its identity snapshot must be paired")
    if bound_null is not None and null_snapshot is not None:
        _verify_bound_null_unchanged(
            bound_null,
            resolution,
            null_snapshot,
            observed_length=int(pair.source.size),
            semantic_digest=semantic_digest,
            plan_digest=plan_digest,
            observed_pair=pair,
        )


def _reverify_final_active_state(
    pair: SeriesPair,
    resolution: PlanResolutionV2,
    *,
    semantic_digest: str,
    plan_digest: str,
    bound_statistic: BoundStatisticAdapter,
    statistic_snapshot: _BoundStatisticIdentitySnapshot,
    bound_null: BoundNullModel | None,
    null_snapshot: _BoundNullIdentitySnapshot | None,
) -> None:
    """Perform the final checkpoint without invoking bound public getters."""

    exact_resolution = _require_resolver_owned_resolution_v2_pure(resolution)
    try:
        current_plan_digest = scientific_plan_v2_sha256(exact_resolution.plan)
        current_semantic_digest = _semantic_digest_for_exact_pair(
            pair,
            exact_resolution.plan.candidates,
        )
        current_statistic = _snapshot_registered_bound_statistic_v2(
            exact_resolution,
            bound_statistic,
            expected_pair=pair,
            initial=statistic_snapshot.execution,
        )
    except V2IntegrityError:
        raise
    except (AttributeError, TypeError, ValueError, OverflowError) as error:
        raise V2IntegrityError("final active-state snapshot is invalid") from error
    if current_plan_digest != plan_digest or current_semantic_digest != semantic_digest:
        raise V2IntegrityError("final run identity snapshot drifted")
    if not _bound_statistic_execution_snapshot_matches(
        current_statistic,
        statistic_snapshot.execution,
    ):
        raise V2IntegrityError("final bound statistic snapshot drifted")
    if (bound_null is None) is not (null_snapshot is None):
        raise V2IntegrityError("final bound null snapshot association is invalid")
    if bound_null is not None and null_snapshot is not None:
        try:
            current_null = _snapshot_registered_bound_null_v2(
                exact_resolution,
                bound_null,
                expected_pair=pair,
                initial=null_snapshot.execution,
            )
        except V2IntegrityError:
            raise
        except (AttributeError, TypeError, ValueError, OverflowError) as error:
            raise V2IntegrityError("final bound null snapshot is invalid") from error
        if not _bound_null_execution_snapshot_matches(
            current_null,
            null_snapshot.execution,
        ):
            raise V2IntegrityError("final bound null snapshot drifted")


def _capture_prepared_ownership(
    prepared: _PreparedCalibration,
) -> _PreparedOwnershipSnapshot:
    canonical_prepared_type = _require_frozen_prepared_type()
    if type(prepared) is not canonical_prepared_type:
        raise V2IntegrityError("prepared calibration canonical type drifted")
    initial_signature = _prepared_signature(prepared)
    resolution = prepared.resolution
    observed_pair = prepared.observed_pair
    semantic_digest = prepared.semantic_input_sha256
    plan_digest = prepared.scientific_plan_sha256
    bound_statistic = prepared.bound_statistic
    bound_null = prepared.bound_null
    statistic_snapshot = prepared.statistic_snapshot
    null_snapshot = prepared.null_snapshot
    raw_results = prepared.raw_observed_results
    scored_results = prepared.observed_results
    selection = prepared.observed_selection
    null_bind_snapshot = prepared.null_bind_snapshot
    exact_semantic_digest = require_sha256(
        semantic_digest,
        name="prepared semantic input digest",
    )
    exact_plan_digest = require_sha256(
        plan_digest,
        name="prepared scientific plan digest",
    )
    _reverify_run_identity(
        observed_pair,
        resolution,
        semantic_digest=exact_semantic_digest,
        plan_digest=exact_plan_digest,
    )
    _verify_bound_statistic_unchanged(
        bound_statistic,
        resolution,
        statistic_snapshot,
        observed_pair=observed_pair,
    )
    _verify_bound_null_unchanged(
        bound_null,
        resolution,
        null_snapshot,
        observed_length=int(observed_pair.source.size),
        semantic_digest=exact_semantic_digest,
        plan_digest=exact_plan_digest,
        observed_pair=observed_pair,
    )
    if (
        null_bind_snapshot.status is not NullBindStatus.ENABLED
        or null_bind_snapshot.bound is not bound_null
        or null_bind_snapshot.disabled_reason is not None
        or null_bind_snapshot.diagnostics != prepared.diagnostics
    ):
        raise V2IntegrityError("preparer-owned null bind association is invalid")
    validated_raw = _validate_raw_observed_vector(
        raw_results,
        statistic_snapshot,
        resolution,
    )
    raw_result_fields = _result_vector_snapshot(
        validated_raw,
        subject="raw observed vector",
    )
    _validate_selector_output(
        validated_raw,
        raw_result_fields,
        selection,
        scored_results,
        resolution.plan.candidates,
        _classify_frozen_selection_rule(resolution.plan.selection_rule),
        resolution.plan.tie_tolerance,
    )
    _reverify_final_active_state(
        observed_pair,
        resolution,
        semantic_digest=exact_semantic_digest,
        plan_digest=exact_plan_digest,
        bound_statistic=bound_statistic,
        statistic_snapshot=statistic_snapshot,
        bound_null=bound_null,
        null_snapshot=null_snapshot,
    )
    final_signature = _prepared_signature(prepared)
    if not _exact_snapshot_value_matches(final_signature, initial_signature):
        raise V2IntegrityError("prepared snapshot drifted during ownership capture")
    return _PreparedOwnershipSnapshot(signature=final_signature)


def _require_preparer_owned_prepared(value: object) -> _PreparedCalibration:
    """Require an unchanged process-local product of ``_prepare_calibration``."""

    canonical_prepared_type = _require_frozen_prepared_type()
    if type(value) is not canonical_prepared_type:
        raise V2IntegrityError("value is not an exact preparer-owned calibration")
    prepared = value
    try:
        registered = _PREPARER_OWNED[prepared]
    except KeyError as error:
        raise V2IntegrityError("prepared calibration is not preparer-owned") from error
    current = _capture_prepared_ownership(prepared)
    if not _exact_snapshot_value_matches(current.signature, registered.signature):
        raise V2IntegrityError("preparer-owned calibration snapshot drifted")
    return prepared


def _freeze_prepared_in_memory_execution_budget(
    require_owned: Callable[[object], _PreparedCalibration],
    require_values: _BudgetValuesGuard,
    /,
) -> _PreparedBudgetGuard:
    """Capture the canonical ownership and integer gates for later execution."""

    exact_length = len

    def _require_prepared_in_memory_execution_budget(
        value: object,
        /,
    ) -> _PreparedCalibration:
        """Require an owned prepared graph within the current in-memory envelope."""

        prepared = require_owned(value)
        require_values(
            B=prepared.resolution.plan.replicates,
            C=exact_length(prepared.resolution.plan.candidates),
            N=prepared.null_snapshot.observed_length,
            S=prepared.null_snapshot.execution.retained_token_state_units,
        )
        return prepared

    return _require_prepared_in_memory_execution_budget


_require_prepared_in_memory_execution_budget = (
    _freeze_prepared_in_memory_execution_budget(
        _require_preparer_owned_prepared,
        _require_in_memory_execution_budget_values,
    )
)
del _freeze_prepared_in_memory_execution_budget


def _freeze_task8b_light_ops() -> _Task8BLightOps:
    """Freeze the complete O(1) checkpoint walker outside module lookups."""

    canonical_prepared_type = _PreparedCalibration
    canonical_selection_rule_type = SelectionRule
    canonical_max_upper = SelectionRule.MAX_UPPER
    canonical_max_absolute = SelectionRule.MAX_ABSOLUTE
    canonical_selector = select_family_v2
    integrity_error_type = V2IntegrityError
    member_descriptor_type = MemberDescriptorType
    exact_type = type
    identity = id
    exact_tuple = tuple
    exact_int = int
    exact_str = str
    exact_bool = bool
    exact_all = all
    scalar_types = (bool, int, float, str, bytes)
    snapshot_errors = (AttributeError, TypeError, ValueError, OverflowError)

    def light_value_marker(value: object) -> tuple[object, ...]:
        """Return an O(1) marker without copying array or collection payloads."""

        value_type = exact_type(value)
        if value is None or value_type in scalar_types:
            return (identity(value_type), value)
        if value_type is exact_type:
            return (identity(exact_type), identity(value))
        try:
            shape = value.shape  # type: ignore[attr-defined]
            strides = value.strides  # type: ignore[attr-defined]
            dtype = value.dtype  # type: ignore[attr-defined]
            writable = value.flags.writeable  # type: ignore[attr-defined]
        except snapshot_errors:
            return (identity(value_type), identity(value))
        return (
            identity(value_type),
            identity(value),
            exact_tuple(exact_int(size) for size in shape),
            exact_tuple(exact_int(stride) for stride in strides),
            exact_str(dtype),
            exact_bool(writable),
        )

    def light_slots_signature(value: object) -> tuple[object, ...]:
        """Snapshot one slotted object through the frozen descriptor protocol."""

        canonical_type = exact_type(value)
        slot_names = canonical_type.__dict__.get("__slots__")
        names: tuple[str, ...]
        if exact_type(slot_names) is exact_str:
            names = (slot_names,)  # type: ignore[assignment]
        elif exact_type(slot_names) is exact_tuple and exact_all(
            exact_type(name) is exact_str
            for name in slot_names  # type: ignore[union-attr]
        ):
            names = slot_names  # type: ignore[assignment]
        else:
            raise integrity_error_type("light checkpoint requires an exact slotted object")
        fields: list[tuple[object, ...]] = []
        try:
            for name in names:
                if name == "__weakref__":
                    continue
                descriptor = canonical_type.__dict__.get(name)
                if exact_type(descriptor) is not member_descriptor_type:
                    raise integrity_error_type(
                        "light checkpoint slot descriptor drifted: "
                        f"{canonical_type.__name__}.{name}"
                    )
                fields.append(
                    (
                        name,
                        identity(descriptor),
                        light_value_marker(
                            descriptor.__get__(  # type: ignore[union-attr]
                                value,
                                canonical_type,
                            )
                        ),
                    )
                )
        except integrity_error_type:
            raise
        except snapshot_errors as error:
            raise integrity_error_type("light checkpoint slot snapshot is invalid") from error
        return (identity(canonical_type), identity(value), exact_tuple(fields))

    def prepared_light_signature(
        prepared: _PreparedCalibration,
        /,
    ) -> _PreparedLightFingerprint:
        """Capture the fixed-size live graph as recursively immutable values."""

        if exact_type(prepared) is not canonical_prepared_type:
            raise integrity_error_type("light checkpoint prepared type drifted")
        try:
            resolution = prepared.resolution
            plan = resolution.plan
            adapters = resolution.adapters
            statistic_snapshot = prepared.statistic_snapshot
            null_snapshot = prepared.null_snapshot
            statistic_execution = statistic_snapshot.execution
            null_execution = null_snapshot.execution
            objects = (
                prepared,
                resolution,
                plan,
                adapters,
                adapters.statistic,
                adapters.null_model,
                prepared.observed_pair,
                prepared.bound_statistic,
                prepared.bound_null,
                statistic_snapshot,
                null_snapshot,
                prepared.null_bind_snapshot,
                statistic_execution,
                null_execution,
                statistic_execution.evaluate_all,
                null_execution.identity_token,
                null_execution.sample_token,
                null_execution.snapshot_token,
                null_execution.apply,
                *(
                    ()
                    if null_execution.enumerate_token is None
                    else (null_execution.enumerate_token,)
                ),
            )
            operation_descriptors = (
                (
                    "evaluate_all",
                    statistic_execution.evaluate_all.descriptor,
                    exact_type(prepared.bound_statistic).__dict__.get("evaluate_all"),
                ),
                (
                    "identity_token",
                    null_execution.identity_token.descriptor,
                    exact_type(prepared.bound_null).__dict__.get("identity_token"),
                ),
                (
                    "sample_token",
                    null_execution.sample_token.descriptor,
                    exact_type(prepared.bound_null).__dict__.get("sample_token"),
                ),
                (
                    "snapshot_token",
                    null_execution.snapshot_token.descriptor,
                    exact_type(prepared.bound_null).__dict__.get("snapshot_token"),
                ),
                (
                    "apply",
                    null_execution.apply.descriptor,
                    exact_type(prepared.bound_null).__dict__.get("apply"),
                ),
                *(
                    ()
                    if null_execution.enumerate_token is None
                    else (
                        (
                            "enumerate_token",
                            null_execution.enumerate_token.descriptor,
                            exact_type(prepared.bound_null).__dict__.get("enumerate_token"),
                        ),
                    )
                ),
            )
        except snapshot_errors as error:
            raise integrity_error_type("light prepared fingerprint is invalid") from error
        return (
            exact_tuple(light_slots_signature(value) for value in objects),
            exact_tuple(
                (
                    name,
                    identity(frozen_descriptor),
                    identity(current_descriptor),
                )
                for name, frozen_descriptor, current_descriptor in operation_descriptors
            ),
        )

    def capture_prepared_light_fingerprint(
        prepared: _PreparedCalibration,
        /,
    ) -> _PreparedLightFingerprint:
        """Precompute one immutable shallow fingerprint per replicate run."""

        return prepared_light_signature(prepared)

    def verify_prepared_light_checkpoint(
        prepared: _PreparedCalibration,
        fingerprint: object,
        /,
    ) -> None:
        """Verify live execution identity without a full ownership recapture."""

        if exact_type(prepared) is not canonical_prepared_type:
            raise integrity_error_type("light checkpoint prepared type drifted")
        if exact_type(fingerprint) is not exact_tuple:
            raise integrity_error_type("light checkpoint fingerprint must be exact")
        try:
            current_signature = prepared_light_signature(prepared)
        except integrity_error_type as error:
            raise integrity_error_type("light prepared fingerprint snapshot drifted") from error
        if current_signature != fingerprint:
            raise integrity_error_type("light prepared fingerprint drifted")

    def classify_selection_rule(value: object, /) -> bool:
        """Interpret a rule only through module-init canonical member identity."""

        if exact_type(value) is not canonical_selection_rule_type:
            raise integrity_error_type("selection rule identity drifted")
        if value is canonical_max_upper:
            return True
        if value is canonical_max_absolute:
            return False
        raise integrity_error_type("selection rule member identity drifted")

    return (
        capture_prepared_light_fingerprint,
        verify_prepared_light_checkpoint,
        prepared_light_signature,
        classify_selection_rule,
        canonical_selector,
    )


_TASK8B_LIGHT_OPS = _freeze_task8b_light_ops()
(
    _capture_prepared_light_fingerprint,
    _verify_prepared_light_checkpoint,
    _prepared_light_signature,
    _classify_frozen_selection_rule,
    _canonical_replicate_selector,
) = _TASK8B_LIGHT_OPS
del _freeze_task8b_light_ops


def _terminal_result(
    *,
    semantic_digest: str,
    plan_digest: str,
    planned_replicates: int,
    alpha: float,
    failure_stage: RunFailureStage,
    observed_results: tuple[StatisticResult, ...],
    diagnostics: tuple[str, ...],
) -> CalibrationResult:
    return CalibrationResult(
        status=RunStatus.NOT_EVALUABLE,
        failure_stage=failure_stage,
        semantic_input_sha256=semantic_digest,
        scientific_plan_sha256=plan_digest,
        planned_replicates=planned_replicates,
        alpha=alpha,
        observed_results=observed_results,
        observed_selection=None,
        replicates=(),
        exceedance_count=0,
        failure_count=0,
        p_value=None,
        exceedance_bound_low=None,
        exceedance_bound_high=None,
        reject_null=None,
        diagnostics=diagnostics,
    )


def _prepare_calibration(
    pair: SeriesPair,
    resolution: PlanResolutionV2,
    /,
) -> CalibrationResult | _PreparedCalibration:
    """Bind once and prepare the observed scan without executing replicates."""

    _require_frozen_prepared_type()
    if type(resolution) is not PlanResolutionV2:
        raise PlanVersionError("resolution must be an exact PlanResolutionV2")
    exact_resolution = _require_resolver_owned_resolution_v2(resolution)
    observed_pair = _snapshot_input_pair(pair)
    semantic_digest = _semantic_digest_for_exact_pair(
        observed_pair,
        exact_resolution.plan.candidates,
    )
    plan_digest = scientific_plan_v2_sha256(exact_resolution.plan)
    planned_replicates = exact_resolution.plan.replicates
    alpha = exact_resolution.plan.alpha
    if type(planned_replicates) is not int or type(alpha) is not float:
        raise V2IntegrityError("terminal plan scalars must be exact built-in values")

    bound_statistic, statistic_snapshot = _snapshot_bound_statistic(
        exact_resolution.adapters.statistic.bind(
            observed_pair,
            exact_resolution.plan.candidates,
        ),
        exact_resolution,
        observed_pair=observed_pair,
    )
    null_bind = exact_resolution.adapters.null_model.bind(
        observed_pair,
        semantic_input_sha256=semantic_digest,
        scientific_plan_sha256=plan_digest,
    )
    if type(null_bind) is not NullBindResult:
        raise V2IntegrityError("null bind must return an exact NullBindResult")
    null_bind_snapshot = _validate_null_bind_snapshot(null_bind)
    if null_bind_snapshot.status is NullBindStatus.DISABLED:
        _reverify_active_preparation(
            observed_pair,
            exact_resolution,
            semantic_digest=semantic_digest,
            plan_digest=plan_digest,
            bound_statistic=bound_statistic,
            statistic_snapshot=statistic_snapshot,
            null_bind=null_bind,
            null_bind_snapshot=null_bind_snapshot,
            bound_null=None,
            null_snapshot=None,
        )
        _reverify_final_active_state(
            observed_pair,
            exact_resolution,
            semantic_digest=semantic_digest,
            plan_digest=plan_digest,
            bound_statistic=bound_statistic,
            statistic_snapshot=statistic_snapshot,
            bound_null=None,
            null_snapshot=None,
        )
        return _terminal_result(
            semantic_digest=semantic_digest,
            plan_digest=plan_digest,
            planned_replicates=planned_replicates,
            alpha=alpha,
            failure_stage=RunFailureStage.NULL_BIND,
            observed_results=(),
            diagnostics=null_bind_snapshot.diagnostics,
        )
    assert null_bind_snapshot.bound is not None
    bound_null, null_snapshot = _snapshot_bound_null(
        null_bind_snapshot.bound,
        exact_resolution,
        observed_length=int(observed_pair.source.size),
        semantic_digest=semantic_digest,
        plan_digest=plan_digest,
        observed_pair=observed_pair,
    )
    _reverify_active_preparation(
        observed_pair,
        exact_resolution,
        semantic_digest=semantic_digest,
        plan_digest=plan_digest,
        bound_statistic=bound_statistic,
        statistic_snapshot=statistic_snapshot,
        null_bind=null_bind,
        null_bind_snapshot=null_bind_snapshot,
        bound_null=bound_null,
        null_snapshot=null_snapshot,
    )

    if (
        null_snapshot.execution.enumerate_token is not None
        and planned_replicates != null_snapshot.total_state_count - 1
    ):
        # Enumeration means every non-identity state exactly once; never partially enumerate.
        _reverify_final_active_state(
            observed_pair,
            exact_resolution,
            semantic_digest=semantic_digest,
            plan_digest=plan_digest,
            bound_statistic=bound_statistic,
            statistic_snapshot=statistic_snapshot,
            bound_null=bound_null,
            null_snapshot=null_snapshot,
        )
        return _terminal_result(
            semantic_digest=semantic_digest,
            plan_digest=plan_digest,
            planned_replicates=planned_replicates,
            alpha=alpha,
            failure_stage=RunFailureStage.NULL_BIND,
            observed_results=(),
            diagnostics=(
                *null_bind_snapshot.diagnostics,
                "enumeration_replicate_count_mismatch_v1",
            ),
        )

    evaluated_observed = statistic_snapshot.execution.evaluate_all.callable(observed_pair)
    _reverify_active_preparation(
        observed_pair,
        exact_resolution,
        semantic_digest=semantic_digest,
        plan_digest=plan_digest,
        bound_statistic=bound_statistic,
        statistic_snapshot=statistic_snapshot,
        null_bind=null_bind,
        null_bind_snapshot=null_bind_snapshot,
        bound_null=bound_null,
        null_snapshot=null_snapshot,
    )
    raw_observed = _validate_raw_observed_vector(
        evaluated_observed,
        statistic_snapshot,
        exact_resolution,
    )
    raw_observed_snapshot = _result_vector_snapshot(
        raw_observed,
        subject="raw observed vector",
    )
    if any(snapshot[4] is Validity.ANALYTIC_FAILURE for snapshot in raw_observed_snapshot):
        _reverify_active_preparation(
            observed_pair,
            exact_resolution,
            semantic_digest=semantic_digest,
            plan_digest=plan_digest,
            bound_statistic=bound_statistic,
            statistic_snapshot=statistic_snapshot,
            null_bind=null_bind,
            null_bind_snapshot=null_bind_snapshot,
            bound_null=bound_null,
            null_snapshot=null_snapshot,
        )
        final_raw = _validate_raw_observed_vector(
            raw_observed,
            statistic_snapshot,
            exact_resolution,
        )
        if (
            _result_vector_snapshot(
                final_raw,
                subject="raw observed vector",
            )
            != raw_observed_snapshot
        ):
            raise V2IntegrityError("observed analytical-failure vector drifted before return")
        _reverify_final_active_state(
            observed_pair,
            exact_resolution,
            semantic_digest=semantic_digest,
            plan_digest=plan_digest,
            bound_statistic=bound_statistic,
            statistic_snapshot=statistic_snapshot,
            bound_null=bound_null,
            null_snapshot=null_snapshot,
        )
        return _terminal_result(
            semantic_digest=semantic_digest,
            plan_digest=plan_digest,
            planned_replicates=planned_replicates,
            alpha=alpha,
            failure_stage=RunFailureStage.OBSERVED_STATISTIC_SCAN,
            observed_results=final_raw,
            diagnostics=(
                *null_bind_snapshot.diagnostics,
                "observed_statistic_scan_analytical_failure_v2",
            ),
        )

    observed_selection_rule = exact_resolution.plan.selection_rule
    if type(observed_selection_rule) is not SelectionRule:
        raise V2IntegrityError("observed selection rule identity drifted")
    observed_rule_is_max_upper = _classify_frozen_selection_rule(observed_selection_rule)
    observed_selection_contract: tuple[tuple[int, ...], SelectionRule, float] = (
        exact_resolution.plan.candidates,
        observed_selection_rule,
        exact_resolution.plan.tie_tolerance,
    )
    selector_output = select_family_v2(raw_observed, *observed_selection_contract)
    if type(selector_output) is not tuple or len(selector_output) != 2:
        raise V2IntegrityError("selector output must be an exact two-item tuple")
    if (
        exact_resolution.plan.candidates is not observed_selection_contract[0]
        or exact_resolution.plan.selection_rule is not observed_selection_contract[1]
        or exact_resolution.plan.tie_tolerance is not observed_selection_contract[2]
    ):
        raise V2IntegrityError("observed selection rule or contract drifted")
    observed_selection, scored_observed = selector_output
    observed_selection, scored_observed = _validate_selector_output(
        raw_observed,
        raw_observed_snapshot,
        observed_selection,
        scored_observed,
        observed_selection_contract[0],
        observed_rule_is_max_upper,
        observed_selection_contract[2],
    )
    _reverify_active_preparation(
        observed_pair,
        exact_resolution,
        semantic_digest=semantic_digest,
        plan_digest=plan_digest,
        bound_statistic=bound_statistic,
        statistic_snapshot=statistic_snapshot,
        null_bind=null_bind,
        null_bind_snapshot=null_bind_snapshot,
        bound_null=bound_null,
        null_snapshot=null_snapshot,
    )
    _reverify_final_active_state(
        observed_pair,
        exact_resolution,
        semantic_digest=semantic_digest,
        plan_digest=plan_digest,
        bound_statistic=bound_statistic,
        statistic_snapshot=statistic_snapshot,
        bound_null=bound_null,
        null_snapshot=null_snapshot,
    )
    canonical_prepared_type = _require_frozen_prepared_type()
    prepared = canonical_prepared_type(
        resolution=exact_resolution,
        observed_pair=observed_pair,
        semantic_input_sha256=semantic_digest,
        scientific_plan_sha256=plan_digest,
        bound_statistic=bound_statistic,
        bound_null=bound_null,
        statistic_snapshot=statistic_snapshot,
        null_snapshot=null_snapshot,
        null_bind_snapshot=null_bind_snapshot,
        raw_observed_results=raw_observed,
        observed_results=scored_observed,
        observed_selection=observed_selection,
        diagnostics=null_bind_snapshot.diagnostics,
        seal=_PREPARATION_SEAL,
    )
    _PREPARER_OWNED[prepared] = _capture_prepared_ownership(prepared)
    return _require_preparer_owned_prepared(prepared)


def _freeze_behavior_global_guard(
    module_globals: Mapping[str, object],
    roots: tuple[object, ...],
    /,
) -> Callable[[], None]:
    """Freeze recursive same-module global identities for trusted call paths."""

    missing = object()
    expected: dict[str, object] = {}
    pending = list(roots)
    seen: set[int] = set()
    while pending:
        candidate = pending.pop()
        if id(candidate) in seen:
            continue
        seen.add(id(candidate))
        if inspect.isfunction(candidate):
            if candidate.__globals__ is not module_globals:
                continue
            current_code_type = type(candidate.__code__)
            closure = inspect.getclosurevars(candidate)
            pending_code = [candidate.__code__]
            seen_code: set[int] = set()
            while pending_code:
                code = pending_code.pop()
                if id(code) in seen_code:
                    continue
                seen_code.add(id(code))
                for instruction in dis.get_instructions(code):
                    if type(instruction.argval) is current_code_type:
                        pending_code.append(instruction.argval)
                    if instruction.opname != "LOAD_GLOBAL":
                        continue
                    name = instruction.argval
                    value = module_globals.get(name, missing)
                    expected[name] = value
                    if value is not missing:
                        pending.append(value)
            pending.extend(closure.nonlocals.values())
        elif isinstance(candidate, tuple):
            pending.extend(candidate)

    expected_globals = tuple(sorted(expected.items()))
    get_global = module_globals.get
    integrity_error_type = V2IntegrityError

    def verify_behavior_globals() -> None:
        for name, expected_value in expected_globals:
            if get_global(name, missing) is not expected_value:
                raise integrity_error_type(
                    f"behavior-critical module global drifted: {name}"
                )

    return verify_behavior_globals


def _freeze_budgeted_replicate_executor(
    require_budget: _PreparedBudgetGuard,
    history_integrity_ops: _Task8BHistoryIntegrityOps,
    light_ops: _Task8BLightOps,
    random_capsule: _RandomCapsuleV2,
    random_capsule_type: type[_RandomCapsuleV2],
    tuple_getitem: Callable[[tuple[Any, ...], int], Any],
    build_random_capability: Callable[
        [Callable[[object, int], int], object],
        tuple[object, ...],
    ],
    module_globals: Mapping[str, object],
    /,
) -> _ReplicateExecutor:
    """Construct the guarded full-body executor without a raw loop callable."""

    exact_type = type
    callable_value = callable
    type_error_type = TypeError
    encode_outcome = _encode_replicate_outcome
    verify_codec_globals = _freeze_behavior_global_guard(
        encode_outcome.__globals__, (encode_outcome,)
    )
    tuple_type = tuple
    int_type = int
    float_type = float
    bool_type = bool
    any_value = any
    exact_length = len
    replicate_ids = range
    finite_value = math.isfinite
    cast_value = cast
    integrity_error_type = V2IntegrityError
    selection_rule_type = SelectionRule
    analytical_failure_validity = Validity.ANALYTIC_FAILURE
    analytical_failure_status = ReplicateStatus.ANALYTIC_FAILURE
    complete_status = ReplicateStatus.COMPLETE
    statistic_scan_stage = ReplicateFailureStage.STATISTIC_SCAN
    create_random_stream = cast(
        Callable[[str, int, int], tuple[Any, str]],
        tuple_getitem(random_capsule, 0),
    )
    read_random_seed_digest = cast(
        Callable[[tuple[Any, str]], str],
        tuple_getitem(random_capsule, 1),
    )
    random_randbelow = cast(
        Callable[[object, int], int],
        tuple_getitem(random_capsule, 2),
    )
    seed_digest_for_replicate = cast(
        Callable[[str, int, int], str],
        tuple_getitem(random_capsule, 3),
    )
    null_transform_token_type = NullTransformToken
    history_snapshot_type = _ReplicateHistorySnapshot
    signature_link_type = _Task8BSignatureLink | None
    require_prepared_type = _require_frozen_prepared_type
    require_outcome_type = _require_frozen_outcome_type
    require_owned_prepared = _require_preparer_owned_prepared
    validate_transform_result = _validate_transform_result
    validate_raw_replicate_vector = _validate_raw_replicate_vector
    result_vector_snapshot = _result_vector_snapshot
    validate_selector_output = _validate_selector_output
    snapshot_selection = _snapshot_selection
    verify_behavior_globals = _freeze_behavior_global_guard(
        module_globals,
        (
            require_budget,
            history_integrity_ops,
            light_ops,
            require_prepared_type,
            require_outcome_type,
            require_owned_prepared,
            validate_transform_result,
            validate_raw_replicate_vector,
            result_vector_snapshot,
            validate_selector_output,
            snapshot_selection,
        ),
    )

    def _execute_replicates(
        prepared: _PreparedCalibration,
        /,
        *,
        observer: _ReplicateObserver | None,
    ) -> tuple[ReplicateOutcome, ...]:
        """Execute every planned v2 replicate without finalizing a p-value."""

        exact_prepared = require_budget(prepared)
        if observer is not None and not callable_value(observer):
            raise type_error_type("observer must be callable or None")
        if (
            exact_type(random_capsule) is not random_capsule_type
            or tuple_getitem(random_capsule, 0) is not create_random_stream
            or tuple_getitem(random_capsule, 1) is not read_random_seed_digest
            or tuple_getitem(random_capsule, 2) is not random_randbelow
            or tuple_getitem(random_capsule, 3) is not seed_digest_for_replicate
        ):
            raise integrity_error_type("random capsule execution leaves drifted")
        verify_behavior_globals()
        frozen_history_integrity_ops = history_integrity_ops
        register_outcome = frozen_history_integrity_ops.register_outcome
        capture_terminal_tokens = frozen_history_integrity_ops.capture_terminal_tokens
        terminal_history = frozen_history_integrity_ops.terminal_history
        snapshot_replicate_token = frozen_history_integrity_ops.snapshot_replicate_token
        integrity_error = frozen_history_integrity_ops.integrity_error
        require_prepared_type()
        (
            capture_light_fingerprint,
            verify_light_checkpoint,
            _,
            classify_selection_rule,
            frozen_selector,
        ) = light_ops
        if module_globals["select_family_v2"] is not frozen_selector:
            raise integrity_error("replicate selector canonical alias drifted")
        light_fingerprint = capture_light_fingerprint(exact_prepared)
        verify_light_checkpoint(exact_prepared, light_fingerprint)
        verify_behavior_globals()
        selection_candidates = exact_prepared.resolution.plan.candidates
        selection_rule = exact_prepared.resolution.plan.selection_rule
        selection_tie_tolerance = exact_prepared.resolution.plan.tie_tolerance
        if exact_type(selection_candidates) is not tuple_type or any_value(
            exact_type(candidate) is not int_type or candidate < 1
            for candidate in selection_candidates
        ):
            raise integrity_error("replicate selection candidates identity drifted")
        if exact_type(selection_rule) is not selection_rule_type:
            raise integrity_error("replicate selection rule identity drifted")
        selection_rule = cast_value(selection_rule_type, selection_rule)
        selection_rule_is_max_upper = classify_selection_rule(selection_rule)
        if (
            exact_type(selection_tie_tolerance) is not float_type
            or not finite_value(selection_tie_tolerance)
            or selection_tie_tolerance < 0.0
        ):
            raise integrity_error("replicate selection tie tolerance identity drifted")
        selection_contract: tuple[tuple[int, ...], SelectionRule, float] = (
            selection_candidates,
            selection_rule,
            selection_tie_tolerance,
        )
        history: list[_ReplicateHistorySnapshot] = []
        registered_signature_link: _Task8BSignatureLink | None = None
        planned_replicates = exact_prepared.resolution.plan.replicates

        for replicate_id in replicate_ids(planned_replicates):
            random_stream = create_random_stream(
                exact_prepared.scientific_plan_sha256,
                replicate_id,
                planned_replicates,
            )
            seed_digest = read_random_seed_digest(random_stream)
            random_capability = build_random_capability(
                random_randbelow,
                random_stream,
            )

            enumerate_operation = exact_prepared.null_snapshot.execution.enumerate_token
            if enumerate_operation is None:
                sampled_token_value = (
                    exact_prepared.null_snapshot.execution.sample_token.callable(
                        random_capability
                    )
                )
            else:
                # Exact enumeration: the replicate index, not the draw, selects the state.
                sampled_token_value = enumerate_operation.callable(replicate_id)
            verify_behavior_globals()
            sampled_token_snapshot = snapshot_replicate_token(
                sampled_token_value, exact_prepared, False
            )
            sampled_token = cast_value(null_transform_token_type, sampled_token_value)
            verify_light_checkpoint(exact_prepared, light_fingerprint)

            transformed_value = exact_prepared.null_snapshot.execution.apply.callable(sampled_token)
            verify_behavior_globals()
            transformed_pair, transform_diagnostics = validate_transform_result(
                transformed_value,
                sampled_token=sampled_token,
                sampled_token_snapshot=sampled_token_snapshot,
                prepared=exact_prepared,
            )
            verify_behavior_globals()
            if read_random_seed_digest(random_stream) != seed_digest:
                raise integrity_error("replicate seed digest drifted during transformation")
            verify_light_checkpoint(exact_prepared, light_fingerprint)

            evaluated_value = exact_prepared.statistic_snapshot.execution.evaluate_all.callable(
                transformed_pair
            )
            verify_behavior_globals()
            raw_results = validate_raw_replicate_vector(
                evaluated_value,
                exact_prepared,
                replicate_id=replicate_id,
            )
            verify_behavior_globals()
            raw_snapshot = result_vector_snapshot(
                raw_results,
                subject=f"replicate {replicate_id} raw vector",
            )
            if (
                snapshot_replicate_token(sampled_token, exact_prepared, False)
                != sampled_token_snapshot
            ):
                raise integrity_error("sampled token drifted during statistic evaluation")
            if read_random_seed_digest(random_stream) != seed_digest:
                raise integrity_error("replicate seed digest drifted during statistic evaluation")
            verify_light_checkpoint(exact_prepared, light_fingerprint)

            has_analytical_failure = any_value(
                snapshot[4] is analytical_failure_validity for snapshot in raw_snapshot
            )
            is_identity = cast_value(bool_type, sampled_token_snapshot[6])
            if has_analytical_failure:
                if is_identity:
                    raise integrity_error(
                        "sampled identity failed to reproduce the observed statistic family"
                    )
                outcome_status = analytical_failure_status
                outcome_failure_stage = statistic_scan_stage
                outcome_results = raw_results
                outcome_results_snapshot = raw_snapshot
                outcome_selection = None
                outcome_selection_snapshot = None
                outcome_diagnostics = (
                    *transform_diagnostics,
                    "replicate_statistic_scan_analytical_failure_v2",
                )
                canonical_outcome_type = require_outcome_type()
                outcome = canonical_outcome_type(
                    replicate_id=replicate_id,
                    seed_digest_sha256=seed_digest,
                    status=outcome_status,
                    failure_stage=outcome_failure_stage,
                    transform_token=sampled_token,
                    statistic_results=outcome_results,
                    selection=outcome_selection,
                    diagnostics=outcome_diagnostics,
                )
            else:
                if module_globals["select_family_v2"] is not frozen_selector:
                    raise integrity_error("replicate selector canonical alias drifted")
                selector_output = frozen_selector(
                    raw_results,
                    *selection_contract,
                )
                verify_behavior_globals()
                if (
                    exact_type(selector_output) is not tuple_type
                    or exact_length(selector_output) != 2
                ):
                    raise integrity_error("selector output must be an exact two-item tuple")
                selection_value, scored_value = selector_output
                selection, scored_results = validate_selector_output(
                    raw_results,
                    raw_snapshot,
                    selection_value,
                    scored_value,
                    selection_contract[0],
                    selection_rule_is_max_upper,
                    selection_contract[2],
                )
                verify_behavior_globals()
                if is_identity and (
                    selection.decision_statistic
                    != exact_prepared.observed_selection.decision_statistic
                ):
                    raise integrity_error(
                        "sampled identity decision statistic differs from observed A"
                    )
                if (
                    snapshot_replicate_token(sampled_token, exact_prepared, False)
                    != sampled_token_snapshot
                ):
                    raise integrity_error("sampled token drifted during central reselection")
                if read_random_seed_digest(random_stream) != seed_digest:
                    raise integrity_error(
                        "replicate seed digest drifted during central reselection"
                    )
                outcome_status = complete_status
                outcome_failure_stage = None
                outcome_results = scored_results
                outcome_results_snapshot = result_vector_snapshot(
                    outcome_results,
                    subject=f"replicate {replicate_id} scored vector before outcome construction",
                )
                outcome_selection = selection
                outcome_selection_snapshot = snapshot_selection(outcome_selection)
                outcome_diagnostics = transform_diagnostics
                canonical_outcome_type = require_outcome_type()
                outcome = canonical_outcome_type(
                    replicate_id=replicate_id,
                    seed_digest_sha256=seed_digest,
                    status=outcome_status,
                    failure_stage=outcome_failure_stage,
                    transform_token=sampled_token,
                    statistic_results=outcome_results,
                    selection=outcome_selection,
                    diagnostics=outcome_diagnostics,
                )

            canonical_outcome_type = require_outcome_type()
            if exact_type(outcome) is not canonical_outcome_type:
                raise integrity_error("replicate outcome must be an exact ReplicateOutcome")
            registered_signature = register_outcome(
                outcome,
                raw_results,
                exact_prepared,
                (
                    replicate_id,
                    seed_digest,
                    outcome_status,
                    outcome_failure_stage,
                    sampled_token,
                    sampled_token_snapshot,
                    outcome_results,
                    outcome_results_snapshot,
                    outcome_selection,
                    outcome_selection_snapshot,
                    outcome_diagnostics,
                    raw_snapshot,
                ),
                frozen_history_integrity_ops,
            )
            verify_behavior_globals()
            history.append(history_snapshot_type(outcome, raw_results, registered_signature))
            registered_signature_link = (registered_signature, registered_signature_link)
            if observer is not None:
                verify_codec_globals()
                payload = encode_outcome(outcome)
                verify_behavior_globals()
                verify_light_checkpoint(exact_prepared, light_fingerprint)
                observer(replicate_id, payload)
                verify_codec_globals()
                verify_behavior_globals()
                verify_light_checkpoint(exact_prepared, light_fingerprint)

        frozen_history = tuple_type(history)
        reverse_signatures: list[tuple[object, ...]] = []
        link = registered_signature_link
        while link is not None:
            signature, previous = link
            reverse_signatures.append(signature)
            link = cast_value(signature_link_type, previous)
        registered_outcome_snapshots = tuple_type(reverse_signatures[::-1])
        del reverse_signatures
        terminal_token_snapshots = capture_terminal_tokens(
            exact_prepared,
            frozen_history,
            frozen_history_integrity_ops,
        )
        verify_behavior_globals()
        exact_prepared = require_owned_prepared(exact_prepared)
        verify_light_checkpoint(exact_prepared, light_fingerprint)
        try:
            final_outcomes = terminal_history(
                frozen_history,
                terminal_token_snapshots,
                registered_outcome_snapshots,
                frozen_history_integrity_ops,
            )
        except integrity_error as error:
            raise integrity_error(
                "Task 8B terminal history snapshot drifted or is invalid"
            ) from error
        verify_behavior_globals()
        return final_outcomes

    return _execute_replicates


_execute_replicates = _freeze_budgeted_replicate_executor(
    _require_prepared_in_memory_execution_budget,
    _TASK8B_HISTORY_INTEGRITY_OPS,
    _TASK8B_LIGHT_OPS,
    _RANDOM_CAPSULE_V2,
    _RandomCapsuleV2,
    cast(Callable[[tuple[Any, ...], int], Any], tuple.__getitem__),
    _build_random_index_capability_v2,
    globals(),
)
del _freeze_budgeted_replicate_executor


def _freeze_terminal_result_verifier(
    real_verifier: Callable[[object, object], None],
    module_globals: Mapping[str, object],
    /,
) -> _TerminalResultVerifier:
    """Capture the real verifier while retaining one dynamic public hook call."""

    integrity_error_type = V2IntegrityError
    require_resolution = _require_resolver_owned_resolution_v2_pure
    plan_digest_for_plan = scientific_plan_v2_sha256
    callable_value = callable
    verify_behavior_globals = _freeze_behavior_global_guard(
        module_globals,
        (real_verifier, require_resolution, plan_digest_for_plan),
    )

    def _verify_terminal_result_before_return(
        result: CalibrationResult,
        resolution: PlanResolutionV2,
        *,
        expected_semantic_input_sha256: str,
        expected_scientific_plan_sha256: str,
    ) -> CalibrationResult:
        """Bind real input provenance, call the public hook, then attest internally."""

        verify_behavior_globals()
        if result.semantic_input_sha256 != expected_semantic_input_sha256:
            raise integrity_error_type("calibration result semantic input identity drifted")
        if result.scientific_plan_sha256 != expected_scientific_plan_sha256:
            raise integrity_error_type("calibration result scientific plan identity drifted")
        public_verifier = module_globals["verify_calibration_result"]
        if not callable_value(public_verifier):
            raise integrity_error_type("public calibration result verifier is not callable")
        public_verifier(result, resolution)
        verify_behavior_globals()
        real_verifier(result, resolution)
        verify_behavior_globals()
        if result.semantic_input_sha256 != expected_semantic_input_sha256:
            raise integrity_error_type("calibration result semantic input identity drifted")
        if result.scientific_plan_sha256 != expected_scientific_plan_sha256:
            raise integrity_error_type("calibration result scientific plan identity drifted")
        exact_resolution = require_resolution(resolution)
        verify_behavior_globals()
        if plan_digest_for_plan(exact_resolution.plan) != expected_scientific_plan_sha256:
            raise integrity_error_type("scientific plan drifted after terminal verification")
        verify_behavior_globals()
        return result

    return _verify_terminal_result_before_return


_verify_terminal_result_before_return = _freeze_terminal_result_verifier(
    verify_calibration_result,
    globals(),
)
del _freeze_terminal_result_verifier


def _freeze_public_calibrator(
    require_budget: _PreparedBudgetGuard,
    execute_replicates: _ReplicateExecutor,
    terminal_verifier: _TerminalResultVerifier,
    /,
) -> tuple[
    Callable[[SeriesPair, PlanResolutionV2], CalibrationResult],
    Callable[[SeriesPair, PlanResolutionV2, _ReplicateObserver], CalibrationResult],
]:
    """Construct both entry points over one guarded preparation and finalizer."""

    exact_type = type
    callable_value = callable
    type_error_type = TypeError
    resolution_type = PlanResolutionV2
    result_type = CalibrationResult
    prepared_type = _PreparedCalibration
    plan_version_error = PlanVersionError
    integrity_error = V2IntegrityError
    run_status_type = RunStatus
    run_failure_stage_type = RunFailureStage
    replicate_status_type = ReplicateStatus
    require_resolution = _require_resolver_owned_resolution_v2_pure
    snapshot_input_pair = _snapshot_input_pair
    semantic_digest_for_pair = _semantic_digest_for_exact_pair
    plan_digest_for_plan = scientific_plan_v2_sha256
    prepare_calibration = _prepare_calibration
    sum_values = sum
    cast_value = cast
    verify_behavior_globals = _freeze_behavior_global_guard(
        globals(),
        (
            require_budget,
            execute_replicates,
            terminal_verifier,
            require_resolution,
            snapshot_input_pair,
            semantic_digest_for_pair,
            plan_digest_for_plan,
            prepare_calibration,
        ),
    )

    def calibrate_shared(
        pair: SeriesPair,
        resolution: PlanResolutionV2,
        /,
        *,
        observer: _ReplicateObserver | None,
    ) -> CalibrationResult:
        """Execute and verify one exact-B selected-family calibration."""

        verify_behavior_globals()
        if exact_type(resolution) is not resolution_type:
            raise plan_version_error("resolution must be an exact PlanResolutionV2")
        exact_resolution = require_resolution(resolution)
        verify_behavior_globals()
        entry_pair = snapshot_input_pair(pair)
        verify_behavior_globals()
        expected_semantic_digest = semantic_digest_for_pair(
            entry_pair,
            exact_resolution.plan.candidates,
        )
        expected_plan_digest = plan_digest_for_plan(exact_resolution.plan)
        verify_behavior_globals()
        prepared_or_terminal = prepare_calibration(entry_pair, exact_resolution)
        verify_behavior_globals()
        if exact_type(prepared_or_terminal) is result_type:
            return terminal_verifier(
                cast_value(result_type, prepared_or_terminal),
                exact_resolution,
                expected_semantic_input_sha256=expected_semantic_digest,
                expected_scientific_plan_sha256=expected_plan_digest,
            )
        if exact_type(prepared_or_terminal) is not prepared_type:
            raise integrity_error("calibration preparation returned an invalid result")
        prepared = require_budget(cast_value(prepared_type, prepared_or_terminal))
        verify_behavior_globals()
        outcomes = execute_replicates(prepared, observer=observer)
        verify_behavior_globals()
        observed_decision = prepared.observed_selection.decision_statistic
        exceedance_count = sum_values(
            outcome.status is replicate_status_type.COMPLETE
            and outcome.selection is not None
            and outcome.selection.decision_statistic >= observed_decision
            for outcome in outcomes
        )
        failure_count = sum_values(
            outcome.status is replicate_status_type.ANALYTIC_FAILURE for outcome in outcomes
        )
        planned_replicates = prepared.resolution.plan.replicates
        alpha = prepared.resolution.plan.alpha
        if failure_count:
            result = result_type(
                status=run_status_type.NOT_EVALUABLE,
                failure_stage=run_failure_stage_type.REPLICATE_EXECUTION,
                semantic_input_sha256=prepared.semantic_input_sha256,
                scientific_plan_sha256=prepared.scientific_plan_sha256,
                planned_replicates=planned_replicates,
                alpha=alpha,
                observed_results=prepared.observed_results,
                observed_selection=prepared.observed_selection,
                replicates=outcomes,
                exceedance_count=exceedance_count,
                failure_count=failure_count,
                p_value=None,
                exceedance_bound_low=(1 + exceedance_count) / (planned_replicates + 1),
                exceedance_bound_high=(1 + exceedance_count + failure_count)
                / (planned_replicates + 1),
                reject_null=None,
                diagnostics=(
                    *prepared.diagnostics,
                    "replicate_execution_analytical_failure_v2",
                ),
            )
        else:
            p_value = (1 + exceedance_count) / (planned_replicates + 1)
            result = result_type(
                status=run_status_type.COMPLETE,
                failure_stage=None,
                semantic_input_sha256=prepared.semantic_input_sha256,
                scientific_plan_sha256=prepared.scientific_plan_sha256,
                planned_replicates=planned_replicates,
                alpha=alpha,
                observed_results=prepared.observed_results,
                observed_selection=prepared.observed_selection,
                replicates=outcomes,
                exceedance_count=exceedance_count,
                failure_count=0,
                p_value=p_value,
                exceedance_bound_low=None,
                exceedance_bound_high=None,
                reject_null=p_value <= alpha,
                diagnostics=prepared.diagnostics,
            )
        return terminal_verifier(
            result,
            exact_resolution,
            expected_semantic_input_sha256=expected_semantic_digest,
            expected_scientific_plan_sha256=expected_plan_digest,
        )

    def calibrate_selected_family(
        pair: SeriesPair,
        resolution: PlanResolutionV2,
        /,
    ) -> CalibrationResult:
        """Execute and verify one exact-B selected-family calibration."""
        verify_behavior_globals()
        return calibrate_shared(pair, resolution, observer=None)

    def calibrate_observed(
        pair: SeriesPair,
        resolution: PlanResolutionV2,
        observer: _ReplicateObserver,
        /,
    ) -> CalibrationResult:
        """Observe immutable registered-outcome bytes from the same calibration."""
        verify_behavior_globals()
        if not callable_value(observer):
            raise type_error_type("observer must be callable")
        return calibrate_shared(pair, resolution, observer=observer)

    return calibrate_selected_family, calibrate_observed


calibrate_selected_family, _calibrate_selected_family_observed = _freeze_public_calibrator(
    _require_prepared_in_memory_execution_budget,
    _execute_replicates,
    _verify_terminal_result_before_return,
)
del _freeze_public_calibrator
