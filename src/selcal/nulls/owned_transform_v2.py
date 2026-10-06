"""Policy-free ownership and postcondition facilities for executable v2 nulls."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from types import MemberDescriptorType
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from selcal.canonical import canonical_json_bytes
from selcal.contracts import JsonValue, SeriesPair
from selcal.contracts_v2 import (
    NullStateV2,
    NullTransformResult,
    NullTransformToken,
    V2IntegrityError,
)

TOKEN_SCHEMA: Literal["selcal.null-transform-token.v2"] = (
    "selcal.null-transform-token.v2"
)
NULL_PARAMETERS_SCHEMA = "selcal.null-parameters.v2"
BOUND_OWNER_SCHEMA = "selcal.bound-null-owner.v2"
TRANSFORMED_ROLE = "source"
_CONTENT_GUARD_DOMAIN = b"selcal.internal-owned-pair-content-guard.v1"
_TOKEN_SNAPSHOT_FIELDS = (
    "schema",
    "null_name",
    "null_parameter_sha256",
    "semantic_input_sha256",
    "scientific_plan_sha256",
    "bound_null_owner_sha256",
    "is_identity",
    "state",
)


@dataclass(frozen=True, slots=True)
class _FrozenTokenDescriptorSnapshot:
    entries: tuple[
        tuple[type[object], tuple[tuple[str, MemberDescriptorType], ...]],
        ...,
    ]


def _freeze_token_descriptor_snapshot(
    state_type: type[object],
    state_fields: tuple[str, ...],
) -> _FrozenTokenDescriptorSnapshot:
    """Freeze exact token/state member descriptors for one registered null."""

    entries: list[
        tuple[type[object], tuple[tuple[str, MemberDescriptorType], ...]]
    ] = []
    for contract_type, field_names in (
        (NullTransformToken, _TOKEN_SNAPSHOT_FIELDS),
        (state_type, state_fields),
    ):
        descriptors: list[tuple[str, MemberDescriptorType]] = []
        for field_name in field_names:
            descriptor = contract_type.__dict__.get(field_name)
            if type(descriptor) is not MemberDescriptorType:
                raise RuntimeError(
                    f"token snapshot slot {contract_type.__name__}.{field_name} "
                    "is invalid"
                )
            descriptors.append((field_name, descriptor))
        entries.append((contract_type, tuple(descriptors)))
    return _FrozenTokenDescriptorSnapshot(entries=tuple(entries))


def _attest_token_descriptor_snapshot(
    snapshot: _FrozenTokenDescriptorSnapshot,
) -> None:
    """Reject any replacement of token/state descriptors frozen by a null."""

    for contract_type, descriptors in snapshot.entries:
        for field_name, descriptor in descriptors:
            if contract_type.__dict__.get(field_name) is not descriptor:
                raise V2IntegrityError(
                    "registered token/state member descriptor drifted: "
                    f"{contract_type.__name__}.{field_name}"
                )


def require_sha256(value: object, *, name: str) -> str:
    """Require one exact lowercase SHA-256 hexadecimal identity."""

    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise V2IntegrityError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def parameter_digest(*, null_name: str, params: Mapping[str, JsonValue]) -> str:
    """Hash the shared executable-null parameter envelope."""

    payload: JsonValue = {
        "schema": NULL_PARAMETERS_SCHEMA,
        "name": null_name,
        "params": params,
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def owner_digest(
    *,
    semantic_input_sha256: str,
    scientific_plan_sha256: str,
    null_parameter_sha256: str,
    observed_length: int,
) -> str:
    """Hash the shared bound-null owner envelope."""

    payload: JsonValue = {
        "schema": BOUND_OWNER_SCHEMA,
        "semantic_input_sha256": semantic_input_sha256,
        "scientific_plan_sha256": scientific_plan_sha256,
        "null_parameter_sha256": null_parameter_sha256,
        "observed_length": observed_length,
        "transformed_role": TRANSFORMED_ROLE,
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def snapshot_required_slots(
    source: object,
    names: tuple[str, ...],
    *,
    subject: str,
) -> tuple[object, ...]:
    """Read required slots once and type missing storage as an integrity failure."""

    try:
        return tuple(getattr(source, name) for name in names)
    except (AttributeError, TypeError) as error:
        raise V2IntegrityError(f"{subject} is missing an exact required slot") from error


@dataclass(frozen=True, slots=True)
class CommonTokenSnapshot:
    """One safe snapshot of policy-independent token fields."""

    schema: Literal["selcal.null-transform-token.v2"]
    null_name: str
    null_parameter_sha256: str
    semantic_input_sha256: str
    scientific_plan_sha256: str
    bound_null_owner_sha256: str
    is_identity: bool
    state: object


def snapshot_common_token(
    token: object,
    *,
    expected_null_name: str,
    expected_state_type: type[object],
) -> CommonTokenSnapshot:
    """Safely validate and snapshot common exact token fields before data access."""

    if type(token) is not NullTransformToken:
        raise V2IntegrityError("token must be an exact NullTransformToken")
    values = snapshot_required_slots(
        token,
        (
            "schema",
            "null_name",
            "null_parameter_sha256",
            "semantic_input_sha256",
            "scientific_plan_sha256",
            "bound_null_owner_sha256",
            "is_identity",
            "state",
        ),
        subject="token",
    )
    (
        schema,
        null_name,
        null_parameter_sha256,
        semantic_input_sha256,
        scientific_plan_sha256,
        bound_null_owner_sha256,
        is_identity,
        state,
    ) = values
    if type(schema) is not str or schema != TOKEN_SCHEMA:
        raise V2IntegrityError("token schema is not the exact v2 schema")
    if type(null_name) is not str or null_name != expected_null_name:
        raise V2IntegrityError(
            f"token null identity is unsupported for {expected_null_name}"
        )
    parameter_digest_value = require_sha256(
        null_parameter_sha256,
        name="null_parameter_sha256",
    )
    semantic_digest = require_sha256(
        semantic_input_sha256,
        name="semantic_input_sha256",
    )
    plan_digest = require_sha256(
        scientific_plan_sha256,
        name="scientific_plan_sha256",
    )
    owner_digest_value = require_sha256(
        bound_null_owner_sha256,
        name="bound_null_owner_sha256",
    )
    if type(is_identity) is not bool:
        raise V2IntegrityError("token identity flag must be a built-in bool")
    if type(state) is not expected_state_type:
        raise V2IntegrityError("token contains the wrong exact v2 state type")
    return CommonTokenSnapshot(
        schema=TOKEN_SCHEMA,
        null_name=null_name,
        null_parameter_sha256=parameter_digest_value,
        semantic_input_sha256=semantic_digest,
        scientific_plan_sha256=plan_digest,
        bound_null_owner_sha256=owner_digest_value,
        is_identity=is_identity,
        state=state,
    )


def build_owned_token(
    common: CommonTokenSnapshot,
    *,
    state: NullStateV2,
) -> NullTransformToken:
    """Construct a new exact token from a common and policy-specific snapshot."""

    return NullTransformToken(
        schema=common.schema,
        null_name=common.null_name,
        null_parameter_sha256=common.null_parameter_sha256,
        semantic_input_sha256=common.semantic_input_sha256,
        scientific_plan_sha256=common.scientific_plan_sha256,
        bound_null_owner_sha256=common.bound_null_owner_sha256,
        is_identity=common.is_identity,
        state=state,
    )


def common_token_payload(token: NullTransformToken) -> dict[str, JsonValue]:
    """Return common canonical token fields; callers add exact policy state."""

    return {
        "schema": token.schema,
        "null_name": token.null_name,
        "null_parameter_sha256": token.null_parameter_sha256,
        "semantic_input_sha256": token.semantic_input_sha256,
        "scientific_plan_sha256": token.scientific_plan_sha256,
        "bound_null_owner_sha256": token.bound_null_owner_sha256,
        "is_identity": token.is_identity,
    }


@dataclass(frozen=True, slots=True)
class PreparedOwnedTransform:
    """Independent golden observation and sacrificial transform input snapshots."""

    golden: SeriesPair
    transform_input: SeriesPair


def _owned_pair_content_guard_sha256(pair: SeriesPair) -> str:
    """Return an internal corruption tripwire, not a public owner identity."""

    digest = hashlib.sha256()
    digest.update(len(_CONTENT_GUARD_DOMAIN).to_bytes(8, byteorder="big"))
    digest.update(_CONTENT_GUARD_DOMAIN)
    for role, values in ((b"source", pair.source), (b"target", pair.target)):
        payload = values.tobytes(order="C")
        digest.update(len(role).to_bytes(8, byteorder="big"))
        digest.update(role)
        digest.update(int(values.size).to_bytes(8, byteorder="big"))
        digest.update(len(payload).to_bytes(8, byteorder="big"))
        digest.update(payload)
    return digest.hexdigest()


def prepare_owned_transform(
    observed: object,
    *,
    expected_content_guard_sha256: object,
) -> PreparedOwnedTransform:
    """Verify one bound content guard, then make an independent hook-facing copy."""

    if type(observed) is not SeriesPair:
        raise V2IntegrityError("bound observed pair must be an exact SeriesPair")
    expected_guard = require_sha256(
        expected_content_guard_sha256,
        name="observed_content_guard_sha256",
    )
    try:
        source = observed.source
        target = observed.target
        golden = SeriesPair(source=source, target=target)
    except (AttributeError, TypeError, ValueError) as error:
        raise V2IntegrityError("bound observed pair snapshot is invalid") from error
    if _owned_pair_content_guard_sha256(golden) != expected_guard:
        raise V2IntegrityError("bound observed pair content guard drifted")
    transform_input = SeriesPair(source=golden.source, target=golden.target)
    return PreparedOwnedTransform(golden=golden, transform_input=transform_input)


def _value_multiset(array: NDArray[np.float64]) -> tuple[bytes, ...]:
    payload = array.astype("<f8", copy=False).tobytes(order="C")
    return tuple(
        sorted(payload[index : index + 8] for index in range(0, len(payload), 8))
    )


def finalize_transform_result(
    *,
    transformed: object,
    golden: SeriesPair,
    token: NullTransformToken,
    expected_source_bytes: bytes,
    mapping_name: str,
) -> NullTransformResult:
    """Validate common transform invariants only against the immutable golden pair."""

    if type(transformed) is not SeriesPair:
        raise V2IntegrityError("transform output must be an exact SeriesPair")
    exact_transformed = transformed
    if exact_transformed.source.shape != golden.source.shape:
        raise V2IntegrityError("transform output shape drift")
    if exact_transformed.target.shape != golden.target.shape:
        raise V2IntegrityError("transform output shape drift")
    if exact_transformed.target.tobytes(order="C") != golden.target.tobytes(order="C"):
        raise V2IntegrityError("transform target byte drift")
    if _value_multiset(exact_transformed.source) != _value_multiset(golden.source):
        raise V2IntegrityError("transform source multiset drift")
    if exact_transformed.source.tobytes(order="C") != expected_source_bytes:
        if token.is_identity:
            raise V2IntegrityError("identity transform source drift")
        raise V2IntegrityError(f"transform source violates the exact {mapping_name}")

    source_changed = not np.array_equal(exact_transformed.source, golden.source)
    diagnostics = (
        ("nonidentity_data_equivalent_v2",)
        if not token.is_identity and not source_changed
        else ()
    )
    result_pair = SeriesPair(
        source=exact_transformed.source,
        target=exact_transformed.target,
    )
    return NullTransformResult(
        pair=result_pair,
        token=token,
        source_changed=source_changed,
        diagnostics=diagnostics,
    )


__all__ = [
    "CommonTokenSnapshot",
    "PreparedOwnedTransform",
    "build_owned_token",
    "common_token_payload",
    "finalize_transform_result",
    "owner_digest",
    "parameter_digest",
    "prepare_owned_transform",
    "require_sha256",
    "snapshot_common_token",
    "snapshot_required_slots",
]
