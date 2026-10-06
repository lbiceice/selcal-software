"""Explicit, non-equivalent migration from historical v1 requests to v2."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import InitVar, dataclass
from enum import StrEnum
from typing import ClassVar, Literal
from weakref import WeakKeyDictionary

from selcal.canonical import canonical_json_bytes, scientific_plan_v1_sha256
from selcal.canonical_v2 import scientific_plan_v2_payload, scientific_plan_v2_sha256
from selcal.contracts import JsonValue, PlanRequest, ResolvedScientificPlan, SelectionRule
from selcal.contracts_v2 import PlanRequestV2, SelCalV2Error, V2IntegrityError
from selcal.parameters import freeze_exact_json_mapping
from selcal.resolution import resolve_plan
from selcal.resolution_v2 import (
    PlanResolutionV2,
    _require_resolver_owned_resolution_v2,
    resolve_plan_v2,
)

_MIGRATION_SEAL = object()
_REFUSAL_PATHS = frozenset({"null.name", "replicates", "root_seed"})


class MigrationFieldStatus(StrEnum):
    """Exact disposition of one canonical field during explicit migration."""

    PRESERVED = "preserved"
    CHANGED = "changed"
    ADDED = "added"
    REMOVED = "removed"
    REFUSED = "refused"


def _freeze_ledger_value(value: object, *, name: str) -> JsonValue:
    if isinstance(value, Mapping):
        return freeze_exact_json_mapping(value, name=name)
    if type(value) is list:
        return tuple(
            _freeze_ledger_value(item, name=name)
            for item in value
        )
    if type(value) is tuple:
        return tuple(
            _freeze_ledger_value(item, name=name)
            for item in value
        )
    if value is None or type(value) in {bool, int, str}:
        return value  # type: ignore[return-value]
    if type(value) is float and math.isfinite(value):
        return 0.0 if value == 0.0 else value
    raise ValueError(f"{name} must contain only finite exact JSON values")


def _derived_change_status(
    *,
    source_present: bool,
    source_value: JsonValue,
    target_present: bool,
    target_value: JsonValue,
) -> MigrationFieldStatus:
    if not source_present:
        return MigrationFieldStatus.ADDED
    if not target_present:
        return MigrationFieldStatus.REMOVED
    if canonical_json_bytes(source_value) == canonical_json_bytes(target_value):
        return MigrationFieldStatus.PRESERVED
    return MigrationFieldStatus.CHANGED


def _valid_refusal_value(path: str, value: JsonValue) -> bool:
    if path == "root_seed":
        return type(value) is int and value > 2**64 - 1
    if path == "replicates":
        return type(value) is int and value > 1_000_000
    if path == "null.name":
        return type(value) is str and value not in {
            "circular_shift_v1",
            "block_shuffle_v1",
        }
    return False


@dataclass(frozen=True, slots=True, eq=False)
class PlanMigrationChange:
    """One immutable field-level source/target migration record."""

    __hash__: ClassVar[None] = None  # type: ignore[assignment]

    path: str
    source_present: bool
    source_value: JsonValue
    target_present: bool
    target_value: JsonValue
    status: MigrationFieldStatus

    def __post_init__(self) -> None:
        if type(self.path) is not str or not self.path or self.path != self.path.strip():
            raise ValueError("migration path must be a non-empty built-in string")
        if type(self.source_present) is not bool or type(self.target_present) is not bool:
            raise ValueError("migration presence markers must be built-in bools")
        if not self.source_present and not self.target_present:
            raise ValueError("migration status requires at least one present field")
        source_value = _freeze_ledger_value(
            self.source_value,
            name="migration source value",
        )
        target_value = _freeze_ledger_value(
            self.target_value,
            name="migration target value",
        )
        if not self.source_present and self.source_value is not None:
            raise ValueError("an absent source field requires source_value None")
        if not self.target_present and self.target_value is not None:
            raise ValueError("an absent target field requires target_value None")
        if type(self.status) is not MigrationFieldStatus:
            raise ValueError("migration status must be an exact MigrationFieldStatus")
        object.__setattr__(
            self,
            "source_value",
            source_value,
        )
        object.__setattr__(
            self,
            "target_value",
            target_value,
        )
        if self.status is MigrationFieldStatus.REFUSED:
            if (
                self.path not in _REFUSAL_PATHS
                or not self.source_present
                or self.target_present
                or target_value is not None
                or not _valid_refusal_value(self.path, source_value)
            ):
                raise ValueError("REFUSED requires an explicit approved refusal shape")
            return
        expected = _derived_change_status(
            source_present=self.source_present,
            source_value=source_value,
            target_present=self.target_present,
            target_value=target_value,
        )
        if self.status is not expected:
            raise ValueError("migration status contradicts the frozen field values")


class PlanMigrationRefusal(SelCalV2Error):
    """A typed refusal carrying the immutable field that cannot enter v2."""

    def __init__(
        self,
        message: str,
        *,
        change_ledger: tuple[PlanMigrationChange, ...],
    ) -> None:
        if (
            type(change_ledger) is not tuple
            or not change_ledger
            or any(type(entry) is not PlanMigrationChange for entry in change_ledger)
            or any(
                entry.status is not MigrationFieldStatus.REFUSED
                for entry in change_ledger
            )
        ):
            raise ValueError("refusal ledger must contain exact REFUSED entries")
        for entry in change_ledger:
            PlanMigrationChange(
                path=entry.path,
                source_present=entry.source_present,
                source_value=entry.source_value,
                target_present=entry.target_present,
                target_value=entry.target_value,
                status=entry.status,
            )
        super().__init__(message)
        self.reason: Literal["V2_VALIDATION_REFUSAL"] = "V2_VALIDATION_REFUSAL"
        self.change_ledger = change_ledger


@dataclass(frozen=True, slots=True, eq=False, weakref_slot=True)
class PlanMigrationV1ToV2:
    """Explicit non-equivalent v1-to-v2 identity and resolution result."""

    v1_scientific_plan_sha256: str
    v2_request: PlanRequestV2
    v2_resolution: PlanResolutionV2
    v2_scientific_plan_sha256: str
    change_ledger: tuple[PlanMigrationChange, ...]
    semantic_equivalence: Literal["NON_EQUIVALENT"]
    required_action: Literal["RECALIBRATION_REQUIRED"]
    seal: InitVar[object] = None

    def __post_init__(self, seal: object) -> None:
        if seal is not _MIGRATION_SEAL:
            raise ValueError("PlanMigrationV1ToV2 must be created by migration factory")
        if type(self) is not PlanMigrationV1ToV2:
            raise ValueError("migration result must be an exact PlanMigrationV1ToV2")
        _validate_migration_result(self)


def _require_digest(value: object, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _validated_v2_request_snapshot(request: object) -> PlanRequestV2:
    if type(request) is not PlanRequestV2:
        raise ValueError("v2_request must be an exact PlanRequestV2")
    return PlanRequestV2(
        candidates=request.candidates,
        statistic_name=request.statistic_name,
        statistic_params=request.statistic_params,
        selection_rule=request.selection_rule,
        null_name=request.null_name,
        null_params=request.null_params,
        replicates=request.replicates,
        alpha=request.alpha,
        tie_tolerance=request.tie_tolerance,
        root_seed=request.root_seed,
    )


def _v2_request_payload(request: PlanRequestV2) -> JsonValue:
    return {
        "schema": request.schema,
        "statistic": {
            "name": request.statistic_name,
            "params": request.statistic_params,
        },
        "candidates": list(request.candidates),
        "selection": {
            "rule": SelectionRule(request.selection_rule).value,
            "tie_tolerance": request.tie_tolerance,
            "decision_contract": request.decision_contract,
        },
        "null": {"name": request.null_name, "params": request.null_params},
        "common_support": request.common_support,
        "replicates": request.replicates,
        "alpha": request.alpha,
        "root_seed": request.root_seed,
        "rng_contract": request.rng_contract,
        "calibration_contract": request.calibration_contract,
        "failure_contract": request.failure_contract,
    }


def _change_payload(entry: PlanMigrationChange) -> JsonValue:
    return {
        "path": entry.path,
        "source_present": entry.source_present,
        "source_value": entry.source_value,
        "target_present": entry.target_present,
        "target_value": entry.target_value,
        "status": entry.status.value,
    }


def _validate_change_entry(entry: object, *, allow_refused: bool) -> PlanMigrationChange:
    if type(entry) is not PlanMigrationChange:
        raise ValueError("migration ledger entries must be exact PlanMigrationChange values")
    rebuilt = PlanMigrationChange(
        path=entry.path,
        source_present=entry.source_present,
        source_value=entry.source_value,
        target_present=entry.target_present,
        target_value=entry.target_value,
        status=entry.status,
    )
    if rebuilt.status is MigrationFieldStatus.REFUSED and not allow_refused:
        raise ValueError("successful migration ledger cannot contain REFUSED")
    return rebuilt


def _validated_success_ledger(value: object) -> tuple[PlanMigrationChange, ...]:
    if type(value) is not tuple or not value:
        raise ValueError("change_ledger must be a non-empty exact tuple")
    entries = tuple(_validate_change_entry(entry, allow_refused=False) for entry in value)
    paths = tuple(entry.path for entry in entries)
    if paths != tuple(sorted(paths)) or len(set(paths)) != len(paths):
        raise ValueError("change_ledger paths must be unique and canonical")
    return entries


def _ledger_canonical_bytes(entries: tuple[PlanMigrationChange, ...]) -> bytes:
    return canonical_json_bytes(tuple(_change_payload(entry) for entry in entries))


def _validate_migration_result(result: PlanMigrationV1ToV2) -> None:
    _require_digest(
        result.v1_scientific_plan_sha256,
        name="v1_scientific_plan_sha256",
    )
    _require_digest(
        result.v2_scientific_plan_sha256,
        name="v2_scientific_plan_sha256",
    )
    request = _validated_v2_request_snapshot(result.v2_request)
    if type(result.v2_resolution) is not PlanResolutionV2:
        raise ValueError("v2_resolution must be an exact PlanResolutionV2")
    _require_resolver_owned_resolution_v2(result.v2_resolution)
    ledger = _validated_success_ledger(result.change_ledger)
    if (
        type(result.semantic_equivalence) is not str
        or result.semantic_equivalence != "NON_EQUIVALENT"
    ):
        raise ValueError("semantic_equivalence must be the built-in string NON_EQUIVALENT")
    if (
        type(result.required_action) is not str
        or result.required_action != "RECALIBRATION_REQUIRED"
    ):
        raise ValueError(
            "required_action must be the built-in string RECALIBRATION_REQUIRED"
        )
    if scientific_plan_v2_sha256(result.v2_resolution.plan) != result.v2_scientific_plan_sha256:
        raise ValueError("v2 digest contradicts the resolved v2 plan")
    if canonical_json_bytes(_v2_request_payload(request)) != canonical_json_bytes(
        scientific_plan_v2_payload(result.v2_resolution.plan)
    ):
        raise ValueError("v2 request contradicts the resolved v2 plan")
    expected_v1_digest, source_payload, target_payload = (
        _expected_success_migration_identity(
            request,
            result.v2_resolution,
        )
    )
    if result.v1_scientific_plan_sha256 != expected_v1_digest:
        raise ValueError("v1 digest contradicts the canonical migration source")
    _validate_complete_success_ledger(
        ledger,
        source=source_payload,
        target=target_payload,
    )


@dataclass(frozen=True, slots=True)
class _MigrationOwnershipSnapshot:
    v2_request: PlanRequestV2
    v2_resolution: PlanResolutionV2
    change_ledger: tuple[PlanMigrationChange, ...]
    v1_scientific_plan_sha256: str
    v2_scientific_plan_sha256: str
    v2_request_canonical_bytes: bytes
    ledger_sha256: str
    semantic_equivalence: str
    required_action: str


_MIGRATION_OWNED: WeakKeyDictionary[
    PlanMigrationV1ToV2, _MigrationOwnershipSnapshot
] = WeakKeyDictionary()


def _capture_migration_snapshot(
    result: PlanMigrationV1ToV2,
) -> _MigrationOwnershipSnapshot:
    _validate_migration_result(result)
    request = _validated_v2_request_snapshot(result.v2_request)
    ledger = _validated_success_ledger(result.change_ledger)
    ledger_bytes = _ledger_canonical_bytes(ledger)
    return _MigrationOwnershipSnapshot(
        v2_request=result.v2_request,
        v2_resolution=result.v2_resolution,
        change_ledger=result.change_ledger,
        v1_scientific_plan_sha256=result.v1_scientific_plan_sha256,
        v2_scientific_plan_sha256=result.v2_scientific_plan_sha256,
        v2_request_canonical_bytes=canonical_json_bytes(_v2_request_payload(request)),
        ledger_sha256=hashlib.sha256(ledger_bytes).hexdigest(),
        semantic_equivalence=result.semantic_equivalence,
        required_action=result.required_action,
    )


def _capture_migration_snapshot_fail_closed(
    result: PlanMigrationV1ToV2,
) -> _MigrationOwnershipSnapshot:
    try:
        return _capture_migration_snapshot(result)
    except V2IntegrityError:
        raise
    except (AttributeError, TypeError, ValueError) as error:
        raise V2IntegrityError("migration identity or ledger is invalid") from error


def _require_resolver_owned_migration_v1_to_v2(
    result: object,
) -> PlanMigrationV1ToV2:
    """Verify process-local factory ownership; this is not a cryptographic proof."""

    if type(result) is not PlanMigrationV1ToV2:
        raise V2IntegrityError("migration must be an exact factory-owned result")
    try:
        registered = _MIGRATION_OWNED[result]
    except KeyError as error:
        raise V2IntegrityError("migration result is not factory-owned") from error
    if result.v2_request is not registered.v2_request:
        raise V2IntegrityError("migration v2 request identity was replaced")
    if result.v2_resolution is not registered.v2_resolution:
        raise V2IntegrityError("migration v2 resolution identity was replaced")
    if result.change_ledger is not registered.change_ledger:
        raise V2IntegrityError("migration ledger identity was replaced")
    current = _capture_migration_snapshot_fail_closed(result)
    if (
        current.v1_scientific_plan_sha256
        != registered.v1_scientific_plan_sha256
        or current.v2_scientific_plan_sha256
        != registered.v2_scientific_plan_sha256
        or current.v2_request_canonical_bytes
        != registered.v2_request_canonical_bytes
        or current.ledger_sha256 != registered.ledger_sha256
        or current.semantic_equivalence != registered.semantic_equivalence
        or current.required_action != registered.required_action
    ):
        raise V2IntegrityError("migration identity or ledger fingerprint drifted")
    return result


def _v1_payload(plan: ResolvedScientificPlan) -> JsonValue:
    return {
        "schema": "selcal.scientific-plan.v1",
        "candidates": list(plan.candidates),
        "statistic": {"name": plan.statistic_name, "params": plan.statistic_params},
        "selection": {
            "rule": SelectionRule(plan.selection_rule).value,
            "tie_tolerance": plan.tie_tolerance,
        },
        "null": {"name": plan.null_name, "params": plan.null_params},
        "replicates": plan.replicates,
        "alpha": plan.alpha,
        "root_seed": plan.root_seed,
        "failure_policy": plan.failure_policy,
        "rng_contract": "sha256-to-pcg64-v1",
        "common_support": "max_candidate_lag_v1",
    }


def _flatten_payload(
    value: JsonValue,
    *,
    path: str = "",
) -> dict[str, JsonValue]:
    if isinstance(value, Mapping) and value:
        flattened: dict[str, JsonValue] = {}
        for key in sorted(value):
            child = f"{path}.{key}" if path else key
            flattened.update(_flatten_payload(value[key], path=child))
        return flattened
    if not path:
        raise ValueError("canonical migration payload must be a non-empty object")
    return {path: value}


def _status_for(
    *,
    source_present: bool,
    source_value: JsonValue,
    target_present: bool,
    target_value: JsonValue,
) -> MigrationFieldStatus:
    return _derived_change_status(
        source_present=source_present,
        source_value=source_value,
        target_present=target_present,
        target_value=target_value,
    )


def _complete_change_ledger(
    source: JsonValue,
    target: JsonValue,
) -> tuple[PlanMigrationChange, ...]:
    source_fields = _flatten_payload(source)
    target_fields = _flatten_payload(target)
    entries: list[PlanMigrationChange] = []
    for path in sorted(source_fields.keys() | target_fields.keys()):
        source_present = path in source_fields
        target_present = path in target_fields
        source_value = source_fields.get(path)
        target_value = target_fields.get(path)
        entries.append(
            PlanMigrationChange(
                path=path,
                source_present=source_present,
                source_value=source_value,
                target_present=target_present,
                target_value=target_value,
                status=_status_for(
                    source_present=source_present,
                    source_value=source_value,
                    target_present=target_present,
                    target_value=target_value,
                ),
            )
        )
    return tuple(entries)


def _change_ledger(source: JsonValue, target: JsonValue) -> tuple[PlanMigrationChange, ...]:
    return _complete_change_ledger(source, target)


def _validate_complete_success_ledger(
    entries: tuple[PlanMigrationChange, ...],
    *,
    source: JsonValue,
    target: JsonValue,
) -> None:
    source_fields = _flatten_payload(source)
    target_fields = _flatten_payload(target)
    expected_paths = tuple(sorted(source_fields.keys() | target_fields.keys()))
    if tuple(entry.path for entry in entries) != expected_paths:
        raise ValueError("successful migration ledger is not complete and exact")
    for entry in entries:
        source_present = entry.path in source_fields
        target_present = entry.path in target_fields
        source_value = source_fields.get(entry.path)
        target_value = target_fields.get(entry.path)
        expected_payload: JsonValue = {
            "path": entry.path,
            "source_present": source_present,
            "source_value": source_value,
            "target_present": target_present,
            "target_value": target_value,
            "status": _derived_change_status(
                source_present=source_present,
                source_value=source_value,
                target_present=target_present,
                target_value=target_value,
            ).value,
        }
        if canonical_json_bytes(_change_payload(entry)) != canonical_json_bytes(
            expected_payload
        ):
            raise ValueError("successful migration ledger is not complete and exact")


def _expected_success_migration_identity(
    request: PlanRequestV2,
    resolution: PlanResolutionV2,
) -> tuple[str, JsonValue, JsonValue]:
    if request.null_name == "circular_shift_v2":
        source_null_name = "circular_shift_v1"
    elif request.null_name == "block_shuffle_v2":
        source_null_name = "block_shuffle_v1"
    else:
        raise ValueError("v2 null has no canonical v1 migration source")
    source_request = PlanRequest(
        candidates=request.candidates,
        statistic_name=request.statistic_name,
        statistic_params=request.statistic_params,
        selection_rule=request.selection_rule,
        null_name=source_null_name,
        null_params=request.null_params,
        replicates=request.replicates,
        alpha=request.alpha,
        tie_tolerance=request.tie_tolerance,
        root_seed=request.root_seed,
        failure_policy="fail_closed_v1",
    )
    source_resolution = resolve_plan(source_request)
    source_payload = _v1_payload(source_resolution.plan)
    target_payload = scientific_plan_v2_payload(resolution.plan)
    return (
        scientific_plan_v1_sha256(source_resolution.plan),
        source_payload,
        target_payload,
    )


def _refuse(path: str, source_value: JsonValue, message: str) -> None:
    entry = PlanMigrationChange(
        path=path,
        source_present=True,
        source_value=source_value,
        target_present=False,
        target_value=None,
        status=MigrationFieldStatus.REFUSED,
    )
    raise PlanMigrationRefusal(message, change_ledger=(entry,))


def migrate_plan_v1_to_v2(request: PlanRequest, /) -> PlanMigrationV1ToV2:
    """Resolve an explicit non-equivalent v2 successor without executing it."""

    if type(request) is not PlanRequest:
        raise TypeError("request must be an exact PlanRequest")
    null_mapping = {
        "circular_shift_v1": "circular_shift_v2",
        "block_shuffle_v1": "block_shuffle_v2",
    }
    if request.null_name not in null_mapping:
        _refuse(
            "null.name",
            request.null_name,
            "migration requires a supported v1 null",
        )
    v2_null_name = null_mapping[request.null_name]
    if request.root_seed > 2**64 - 1:
        _refuse(
            "root_seed",
            request.root_seed,
            "root_seed is invalid under scientific-plan v2",
        )
    if request.replicates > 1_000_000:
        _refuse(
            "replicates",
            request.replicates,
            "replicates is invalid under scientific-plan v2",
        )

    v1_resolution = resolve_plan(request)
    v2_request = PlanRequestV2(
        candidates=request.candidates,
        statistic_name=request.statistic_name,
        statistic_params=request.statistic_params,
        selection_rule=request.selection_rule,
        null_name=v2_null_name,
        null_params=request.null_params,
        replicates=request.replicates,
        alpha=request.alpha,
        tie_tolerance=request.tie_tolerance,
        root_seed=request.root_seed,
    )
    v2_resolution = resolve_plan_v2(v2_request)
    v1_digest = scientific_plan_v1_sha256(v1_resolution.plan)
    v2_digest = scientific_plan_v2_sha256(v2_resolution.plan)
    if v1_digest == v2_digest:
        raise RuntimeError("v1 and v2 plan identities unexpectedly collide")
    ledger = _change_ledger(
        _v1_payload(v1_resolution.plan),
        scientific_plan_v2_payload(v2_resolution.plan),
    )
    result = PlanMigrationV1ToV2(
        v1_scientific_plan_sha256=v1_digest,
        v2_request=v2_request,
        v2_resolution=v2_resolution,
        v2_scientific_plan_sha256=v2_digest,
        change_ledger=ledger,
        semantic_equivalence="NON_EQUIVALENT",
        required_action="RECALIBRATION_REQUIRED",
        seal=_MIGRATION_SEAL,
    )
    _MIGRATION_OWNED[result] = _capture_migration_snapshot_fail_closed(result)
    return _require_resolver_owned_migration_v1_to_v2(result)
