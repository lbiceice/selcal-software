"""Owned strict equal-block shuffle null for SelCal scientific-plan v2."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import ClassVar, Literal, Self, cast

import numpy as np
from numpy.typing import NDArray

from selcal.canonical import canonical_json_bytes
from selcal.contracts import JsonValue, SeriesPair
from selcal.contracts_v2 import (
    BlockShuffleStateV2,
    NullBindStatus,
    NullDisabledReason,
    NullTransformResult,
    NullTransformToken,
    V2IntegrityError,
)
from selcal.nulls.executable_base import (
    NullBindResult,
    UniformIndexSource,
    _resolve_static_callable,
)
from selcal.nulls.owned_transform_v2 import (
    _attest_token_descriptor_snapshot,
    _freeze_token_descriptor_snapshot,
    build_owned_token,
    common_token_payload,
    finalize_transform_result,
    parameter_digest,
    prepare_owned_transform,
    snapshot_common_token,
    snapshot_required_slots,
)
from selcal.nulls.owned_transform_v2 import (
    _owned_pair_content_guard_sha256 as _owned_pair_content_guard_sha256,
)
from selcal.nulls.owned_transform_v2 import owner_digest as _owner_digest
from selcal.nulls.owned_transform_v2 import require_sha256 as _require_sha256
from selcal.parameters import (
    freeze_exact_json_mapping,
    require_builtin_int,
    require_exact_keys,
)

_NULL_NAME = "block_shuffle_v2"
_TOKEN_SCHEMA: Literal["selcal.null-transform-token.v2"] = (
    "selcal.null-transform-token.v2"
)
_STATE_SCHEMA: Literal["selcal.block-shuffle-state.v2"] = (
    "selcal.block-shuffle-state.v2"
)
_MAX_BLOCK_COUNT = 4096
_TOKEN_DESCRIPTOR_SNAPSHOT = _freeze_token_descriptor_snapshot(
    BlockShuffleStateV2,
    ("schema", "block_order"),
)

# Module-level seams make the resource guard independently testable. Neither helper
# is called until divisibility, minimum-block, and maximum-block checks have passed.
_factorial = math.factorial


def _ascending_labels(block_count: int) -> list[int]:
    return list(range(block_count))


def _parameter_digest(block_length: int) -> str:
    return parameter_digest(
        null_name=_NULL_NAME,
        params={"block_length": block_length},
    )


def _validate_block_order(value: object, *, expected_count: int | None) -> tuple[int, ...]:
    if type(value) is not tuple:
        raise V2IntegrityError("block_order must be an exact tuple")
    order = cast(tuple[object, ...], value)
    if len(order) < 2 or any(type(label) is not int for label in order):
        raise V2IntegrityError(
            "block_order must contain exact built-in labels for at least two blocks"
        )
    exact_order = cast(tuple[int, ...], order)
    if expected_count is not None and len(exact_order) != expected_count:
        raise V2IntegrityError("token block order has the wrong bound length")
    if sorted(exact_order) != list(range(len(exact_order))):
        raise V2IntegrityError("block_order must be a complete labelled permutation")
    return exact_order


def _snapshot_block_token(token: object) -> NullTransformToken:
    common = snapshot_common_token(
        token,
        expected_null_name=_NULL_NAME,
        expected_state_type=BlockShuffleStateV2,
    )
    state_value = cast(BlockShuffleStateV2, common.state)
    try:
        state_schema = state_value.schema
        block_order = state_value.block_order
    except (AttributeError, TypeError, ValueError, OverflowError) as error:
        raise V2IntegrityError("sampled block token state is invalid") from error
    if type(state_schema) is not str or state_schema != _STATE_SCHEMA:
        raise V2IntegrityError("sampled block token state drifted")
    try:
        order_snapshot = _validate_block_order(block_order, expected_count=None)
    except V2IntegrityError as error:
        raise V2IntegrityError("sampled block token state drifted") from error
    expected_identity = order_snapshot == tuple(range(len(order_snapshot)))
    if common.is_identity is not expected_identity:
        raise V2IntegrityError("token identity flag contradicts its block state")

    state_snapshot = BlockShuffleStateV2(
        schema=state_schema,
        block_order=order_snapshot,
    )
    return build_owned_token(
        common,
        state=state_snapshot,
    )


def _token_payload(token: NullTransformToken) -> JsonValue:
    snapshot = _snapshot_block_token(token)
    state = cast(BlockShuffleStateV2, snapshot.state)
    payload = common_token_payload(snapshot)
    payload["state"] = {
        "schema": state.schema,
        "block_order": state.block_order,
    }
    return payload


def block_token_sha256(token: NullTransformToken, /) -> str:
    """Return the optional evidence digest of one exact block token."""

    return hashlib.sha256(canonical_json_bytes(_token_payload(token))).hexdigest()


def _apply_state(
    pair: SeriesPair,
    block_order: tuple[int, ...],
    block_length: int,
) -> SeriesPair:
    """Apply the labelled block mapping while preserving within-block order."""

    source = np.empty_like(pair.source)
    for output_block, source_block in enumerate(block_order):
        output_start = output_block * block_length
        source_start = source_block * block_length
        source[output_start : output_start + block_length] = pair.source[
            source_start : source_start + block_length
        ]
    return SeriesPair(source=source, target=pair.target)


def _expected_block_mapping_bytes(
    source: NDArray[np.float64],
    block_order: tuple[int, ...],
    block_length: int,
) -> bytes:
    """Independent byte equation for the exact labelled block mapping."""

    payload = source.tobytes(order="C")
    block_bytes = block_length * source.dtype.itemsize
    return b"".join(
        payload[label * block_bytes : (label + 1) * block_bytes]
        for label in block_order
    )


@dataclass(frozen=True, slots=True, eq=False)
class BlockShuffleNullV2:
    """Unbound exact-parameter strict equal-block shuffle v2 null."""

    __hash__: ClassVar[None] = None  # type: ignore[assignment]

    block_length: int
    _parameters: Mapping[str, JsonValue] = field(init=False, repr=False)
    _null_parameter_sha256: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            exact_length = require_builtin_int(
                self.block_length,
                minimum=1,
                name="block_length",
            )
        except ValueError as error:
            raise V2IntegrityError(str(error)) from error
        object.__setattr__(self, "block_length", exact_length)
        object.__setattr__(
            self,
            "_parameters",
            MappingProxyType({"block_length": exact_length}),
        )
        object.__setattr__(
            self,
            "_null_parameter_sha256",
            _parameter_digest(exact_length),
        )

    @classmethod
    def from_parameters(cls, parameters: object) -> Self:
        try:
            frozen = freeze_exact_json_mapping(
                parameters,
                name="block_shuffle_v2 parameters",
            )
            require_exact_keys(
                frozen,
                expected=frozenset({"block_length"}),
                name="block_shuffle_v2 parameters",
            )
            block_length = require_builtin_int(
                frozen["block_length"],
                minimum=1,
                name="block_length",
            )
        except ValueError as error:
            raise V2IntegrityError(str(error)) from error
        return cls(block_length=block_length)

    @property
    def name(self) -> str:
        return _NULL_NAME

    @property
    def parameters(self) -> Mapping[str, JsonValue]:
        return self._parameters

    @property
    def null_parameter_sha256(self) -> str:
        return self._null_parameter_sha256

    def bind(
        self,
        observed_pair: SeriesPair,
        *,
        semantic_input_sha256: str,
        scientific_plan_sha256: str,
    ) -> NullBindResult:
        if type(observed_pair) is not SeriesPair:
            raise V2IntegrityError("observed_pair must be an exact SeriesPair")
        semantic_digest = _require_sha256(
            semantic_input_sha256,
            name="semantic_input_sha256",
        )
        plan_digest = _require_sha256(
            scientific_plan_sha256,
            name="scientific_plan_sha256",
        )
        try:
            observed_source = observed_pair.source
            observed_target = observed_pair.target
            observed_length = int(observed_source.size)
        except (AttributeError, TypeError, ValueError) as error:
            raise V2IntegrityError("observed_pair fields could not be snapshotted") from error
        block_length = self.block_length
        if observed_length % block_length != 0:
            return NullBindResult(
                status=NullBindStatus.DISABLED,
                bound=None,
                disabled_reason=NullDisabledReason.NON_DIVISIBLE_TAIL,
                diagnostics=("non_divisible_tail_v2",),
            )
        block_count = observed_length // block_length
        if block_count < 2:
            return NullBindResult(
                status=NullBindStatus.DISABLED,
                bound=None,
                disabled_reason=NullDisabledReason.FEWER_THAN_TWO_BLOCKS,
                diagnostics=("fewer_than_two_blocks_v2",),
            )
        if block_count > _MAX_BLOCK_COUNT:
            return NullBindResult(
                status=NullBindStatus.DISABLED,
                bound=None,
                disabled_reason=NullDisabledReason.TOO_MANY_BLOCKS,
                diagnostics=("too_many_blocks_v2",),
            )

        observed_snapshot = SeriesPair(
            source=observed_source,
            target=observed_target,
        )
        observed_content_guard = _owned_pair_content_guard_sha256(observed_snapshot)
        owner_digest = _owner_digest(
            semantic_input_sha256=semantic_digest,
            scientific_plan_sha256=plan_digest,
            null_parameter_sha256=self._null_parameter_sha256,
            observed_length=observed_length,
        )
        bound = _BoundBlockShuffleV2(
            _observed_pair=observed_snapshot,
            _block_length=block_length,
            _block_count=block_count,
            _semantic_input_sha256=semantic_digest,
            _scientific_plan_sha256=plan_digest,
            _null_parameter_sha256=self._null_parameter_sha256,
            _bound_null_owner_sha256=owner_digest,
            _observed_content_guard_sha256=observed_content_guard,
        )
        return NullBindResult(
            status=NullBindStatus.ENABLED,
            bound=bound,
            disabled_reason=None,
            diagnostics=(),
        )


@dataclass(frozen=True, slots=True, eq=False)
class _BoundBlockShuffleV2:
    """One strict block null owned by an immutable complete observed pair."""

    __hash__: ClassVar[None] = None  # type: ignore[assignment]

    _observed_pair: SeriesPair
    _block_length: int
    _block_count: int
    _semantic_input_sha256: str
    _scientific_plan_sha256: str
    _null_parameter_sha256: str
    _bound_null_owner_sha256: str
    _observed_content_guard_sha256: str
    _parameters: Mapping[str, JsonValue] = field(init=False, repr=False)
    _observed_length: int = field(init=False, repr=False)
    _total_state_count: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if type(self._observed_pair) is not SeriesPair:
            raise V2IntegrityError("bound observed pair must be an exact SeriesPair")
        if type(self._block_length) is not int or self._block_length < 1:
            raise V2IntegrityError("bound block_length must be a positive built-in int")
        if (
            type(self._block_count) is not int
            or not 2 <= self._block_count <= _MAX_BLOCK_COUNT
        ):
            raise V2IntegrityError("bound block_count is outside the supported range")
        for name in (
            "_semantic_input_sha256",
            "_scientific_plan_sha256",
            "_null_parameter_sha256",
            "_bound_null_owner_sha256",
            "_observed_content_guard_sha256",
        ):
            _require_sha256(getattr(self, name), name=name.removeprefix("_"))
        try:
            observed_length = int(self._observed_pair.source.size)
            expected_content_guard = _owned_pair_content_guard_sha256(
                self._observed_pair
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise V2IntegrityError("bound observed pair snapshot is invalid") from error
        if self._observed_content_guard_sha256 != expected_content_guard:
            raise V2IntegrityError("bound observed pair content guard drifted")
        if (
            observed_length % self._block_length != 0
            or observed_length // self._block_length != self._block_count
        ):
            raise V2IntegrityError("bound block partition does not cover the observed source")
        expected_parameter = _parameter_digest(self._block_length)
        if self._null_parameter_sha256 != expected_parameter:
            raise V2IntegrityError("bound null-parameter digest does not match parameters")
        expected_owner = _owner_digest(
            semantic_input_sha256=self._semantic_input_sha256,
            scientific_plan_sha256=self._scientific_plan_sha256,
            null_parameter_sha256=expected_parameter,
            observed_length=observed_length,
        )
        if self._bound_null_owner_sha256 != expected_owner:
            raise V2IntegrityError("bound owner digest does not match its ownership fields")
        object.__setattr__(
            self,
            "_parameters",
            MappingProxyType({"block_length": self._block_length}),
        )
        object.__setattr__(self, "_observed_length", observed_length)
        object.__setattr__(self, "_total_state_count", _factorial(self._block_count))

    @property
    def name(self) -> str:
        return _NULL_NAME

    @property
    def parameters(self) -> Mapping[str, JsonValue]:
        return self._parameters

    @property
    def observed_length(self) -> int:
        return self._observed_length

    @property
    def total_state_count(self) -> int:
        return self._total_state_count

    @property
    def null_parameter_sha256(self) -> str:
        return self._null_parameter_sha256

    @property
    def bound_null_owner_sha256(self) -> str:
        return self._bound_null_owner_sha256

    def _token_for_order(self, block_order: tuple[int, ...]) -> NullTransformToken:
        state = BlockShuffleStateV2(
            schema=_STATE_SCHEMA,
            block_order=block_order,
        )
        return NullTransformToken(
            schema=_TOKEN_SCHEMA,
            null_name=_NULL_NAME,
            null_parameter_sha256=self._null_parameter_sha256,
            semantic_input_sha256=self._semantic_input_sha256,
            scientific_plan_sha256=self._scientific_plan_sha256,
            bound_null_owner_sha256=self._bound_null_owner_sha256,
            is_identity=block_order == tuple(range(self._block_count)),
            state=state,
        )

    def identity_token(self) -> NullTransformToken:
        return self._token_for_order(tuple(range(self._block_count)))

    def sample_token(self, random: UniformIndexSource, /) -> NullTransformToken:
        randbelow = cast(
            Callable[[int], object],
            _resolve_static_callable(
                random,
                method_name="randbelow",
                subject="random source",
            ),
        )
        labels = _ascending_labels(self._block_count)
        for index in range(self._block_count - 1, 0, -1):
            sampled_index = randbelow(index + 1)
            if type(sampled_index) is not int or not 0 <= sampled_index <= index:
                raise V2IntegrityError("random index must be an exact in-range built-in int")
            labels[index], labels[sampled_index] = labels[sampled_index], labels[index]
        return self._token_for_order(tuple(labels))

    def snapshot_token(self, token: object, /) -> tuple[object, ...]:
        """Return the exact immutable token/state snapshot owned by this null."""

        original_state = token
        try:
            if type(token) is NullTransformToken:
                original_state = snapshot_required_slots(
                    token,
                    ("state",),
                    subject="token",
                )[0]
                if type(original_state) is not BlockShuffleStateV2:
                    raise V2IntegrityError("sampled block token state drifted")
            snapshot = _snapshot_block_token(token)
        except (AttributeError, TypeError, ValueError, OverflowError) as error:
            raise V2IntegrityError("sampled token snapshot is invalid") from error
        state = cast(BlockShuffleStateV2, snapshot.state)
        _attest_token_descriptor_snapshot(_TOKEN_DESCRIPTOR_SNAPSHOT)
        return (
            snapshot.schema,
            snapshot.null_name,
            snapshot.null_parameter_sha256,
            snapshot.semantic_input_sha256,
            snapshot.scientific_plan_sha256,
            snapshot.bound_null_owner_sha256,
            snapshot.is_identity,
            type(state),
            (state.schema, state.block_order),
            id(token),
            id(original_state),
        )

    def _validate_token_before_data_access(
        self,
        token: object,
    ) -> tuple[NullTransformToken, int, int]:
        (
            block_length_value,
            block_count_value,
            semantic_value,
            plan_value,
            parameter_value,
            owner_value,
            observed_length_value,
        ) = snapshot_required_slots(
            self,
            (
                "_block_length",
                "_block_count",
                "_semantic_input_sha256",
                "_scientific_plan_sha256",
                "_null_parameter_sha256",
                "_bound_null_owner_sha256",
                "_observed_length",
            ),
            subject="bound block shuffle",
        )
        if (
            type(block_length_value) is not int
            or block_length_value < 1
            or type(block_count_value) is not int
            or not 2 <= block_count_value <= _MAX_BLOCK_COUNT
            or type(observed_length_value) is not int
            or observed_length_value < 1
        ):
            raise V2IntegrityError("bound block parameter or state space drifted")
        block_length = block_length_value
        block_count = block_count_value
        observed_length = observed_length_value
        if (
            observed_length % block_length != 0
            or observed_length // block_length != block_count
        ):
            raise V2IntegrityError("bound block parameter or state space drifted")
        semantic_digest = _require_sha256(
            semantic_value,
            name="semantic_input_sha256",
        )
        plan_digest = _require_sha256(
            plan_value,
            name="scientific_plan_sha256",
        )
        parameter_digest_value = _require_sha256(
            parameter_value,
            name="null_parameter_sha256",
        )
        owner_digest_value = _require_sha256(
            owner_value,
            name="bound_null_owner_sha256",
        )
        expected_parameter = _parameter_digest(block_length)
        expected_owner = _owner_digest(
            semantic_input_sha256=semantic_digest,
            scientific_plan_sha256=plan_digest,
            null_parameter_sha256=expected_parameter,
            observed_length=observed_length,
        )
        if parameter_digest_value != expected_parameter:
            raise V2IntegrityError("bound null-parameter digest drifted")
        if owner_digest_value != expected_owner:
            raise V2IntegrityError("bound owner digest drifted")
        snapshot = _snapshot_block_token(token)
        expected_fields = (
            ("schema", snapshot.schema, _TOKEN_SCHEMA),
            ("null_name", snapshot.null_name, _NULL_NAME),
            (
                "null_parameter_sha256",
                snapshot.null_parameter_sha256,
                expected_parameter,
            ),
            (
                "semantic_input_sha256",
                snapshot.semantic_input_sha256,
                semantic_digest,
            ),
            (
                "scientific_plan_sha256",
                snapshot.scientific_plan_sha256,
                plan_digest,
            ),
            (
                "bound_null_owner_sha256",
                snapshot.bound_null_owner_sha256,
                expected_owner,
            ),
        )
        for field_name, actual, expected in expected_fields:
            if type(actual) is not str or actual != expected:
                raise V2IntegrityError(f"foreign token {field_name}")
        state = cast(BlockShuffleStateV2, snapshot.state)
        _validate_block_order(state.block_order, expected_count=block_count)
        return snapshot, block_length, observed_length

    def apply(self, token: NullTransformToken, /) -> NullTransformResult:
        snapshot, block_length, observed_length = (
            self._validate_token_before_data_access(token)
        )
        observed_value, guard_value = snapshot_required_slots(
            self,
            ("_observed_pair", "_observed_content_guard_sha256"),
            subject="bound block shuffle",
        )
        observed_content_guard = _require_sha256(
            guard_value,
            name="observed_content_guard_sha256",
        )
        prepared = prepare_owned_transform(
            observed_value,
            expected_content_guard_sha256=observed_content_guard,
        )
        golden = prepared.golden
        if int(golden.source.size) != observed_length:
            raise V2IntegrityError("bound block observed pair length drifted")
        state = cast(BlockShuffleStateV2, snapshot.state)
        expected_source_bytes = _expected_block_mapping_bytes(
            golden.source,
            state.block_order,
            block_length,
        )
        transformed = _apply_state(
            prepared.transform_input,
            state.block_order,
            block_length,
        )
        return finalize_transform_result(
            transformed=transformed,
            golden=golden,
            token=snapshot,
            expected_source_bytes=expected_source_bytes,
            mapping_name="block mapping",
        )


__all__ = ["BlockShuffleNullV2", "block_token_sha256"]
