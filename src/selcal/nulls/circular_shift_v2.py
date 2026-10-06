"""Owned executable circular-shift null for SelCal scientific-plan v2."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import ClassVar, Literal, Self, cast

import numpy as np
from numpy.typing import NDArray

from selcal.canonical import canonical_json_bytes
from selcal.contracts import JsonValue, SeriesPair
from selcal.contracts_v2 import (
    CircularShiftStateV2,
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

_NULL_NAME = "circular_shift_v2"
_TOKEN_SCHEMA: Literal["selcal.null-transform-token.v2"] = (
    "selcal.null-transform-token.v2"
)
_STATE_SCHEMA: Literal["selcal.circular-shift-state.v2"] = (
    "selcal.circular-shift-state.v2"
)
_TOKEN_DESCRIPTOR_SNAPSHOT = _freeze_token_descriptor_snapshot(
    CircularShiftStateV2,
    ("schema", "shift"),
)


def _parameter_digest(min_shift: int) -> str:
    return parameter_digest(
        null_name=_NULL_NAME,
        params={"min_shift": min_shift},
    )


def _snapshot_circular_token(token: object) -> NullTransformToken:
    common = snapshot_common_token(
        token,
        expected_null_name=_NULL_NAME,
        expected_state_type=CircularShiftStateV2,
    )
    state_value = cast(CircularShiftStateV2, common.state)
    try:
        state_schema = state_value.schema
        shift = state_value.shift
    except (AttributeError, TypeError, ValueError, OverflowError) as error:
        raise V2IntegrityError("sampled circular token state is invalid") from error
    if (
        type(state_schema) is not str
        or state_schema != _STATE_SCHEMA
        or type(shift) is not int
        or shift < 0
    ):
        raise V2IntegrityError("sampled circular token state drifted")
    expected_identity = shift == 0
    if common.is_identity is not expected_identity:
        raise V2IntegrityError("token identity flag contradicts its circular state")

    state_snapshot = CircularShiftStateV2(schema=state_schema, shift=shift)
    return build_owned_token(
        common,
        state=state_snapshot,
    )


def _token_payload(token: NullTransformToken) -> JsonValue:
    snapshot = _snapshot_circular_token(token)
    state = cast(CircularShiftStateV2, snapshot.state)
    payload = common_token_payload(snapshot)
    payload["state"] = {"schema": state.schema, "shift": state.shift}
    return payload


def circular_token_sha256(token: NullTransformToken, /) -> str:
    """Return the optional evidence digest of one exact circular token."""

    return hashlib.sha256(canonical_json_bytes(_token_payload(token))).hexdigest()


def _apply_state(pair: SeriesPair, shift: int) -> SeriesPair:
    """Apply the independently frozen right-shift index equation."""

    source = np.empty_like(pair.source)
    length = int(pair.source.size)
    for index in range(length):
        source[index] = pair.source[(index - shift) % length]
    return SeriesPair(source=source, target=pair.target)


def _expected_right_shift_bytes(source: NDArray[np.float64], shift: int) -> bytes:
    expected = np.empty_like(source)
    length = int(source.size)
    for index in range(length):
        expected[index] = source[(index - shift) % length]
    return expected.tobytes(order="C")


@dataclass(frozen=True, slots=True, eq=False)
class CircularShiftNullV2:
    """Unbound exact-parameter circular-shift v2 null model."""

    __hash__: ClassVar[None] = None  # type: ignore[assignment]

    min_shift: int
    _parameters: Mapping[str, JsonValue] = field(init=False, repr=False)
    _null_parameter_sha256: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            exact_min_shift = require_builtin_int(
                self.min_shift,
                minimum=1,
                name="min_shift",
            )
        except ValueError as error:
            raise V2IntegrityError(str(error)) from error
        object.__setattr__(self, "min_shift", exact_min_shift)
        object.__setattr__(
            self,
            "_parameters",
            MappingProxyType({"min_shift": exact_min_shift}),
        )
        object.__setattr__(
            self,
            "_null_parameter_sha256",
            _parameter_digest(exact_min_shift),
        )

    @classmethod
    def from_parameters(cls, parameters: object) -> Self:
        try:
            frozen = freeze_exact_json_mapping(
                parameters,
                name="circular_shift_v2 parameters",
            )
            require_exact_keys(
                frozen,
                expected=frozenset({"min_shift"}),
                name="circular_shift_v2 parameters",
            )
            min_shift = require_builtin_int(
                frozen["min_shift"],
                minimum=1,
                name="min_shift",
            )
        except ValueError as error:
            raise V2IntegrityError(str(error)) from error
        return cls(min_shift=min_shift)

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
        min_shift = self.min_shift
        if 2 * min_shift > observed_length:
            return NullBindResult(
                status=NullBindStatus.DISABLED,
                bound=None,
                disabled_reason=NullDisabledReason.SHIFT_SPACE_EMPTY,
                diagnostics=("shift_space_empty_v2",),
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
        bound = _BoundCircularShiftV2(
            _observed_pair=observed_snapshot,
            _min_shift=self.min_shift,
            _semantic_input_sha256=semantic_digest,
            _scientific_plan_sha256=plan_digest,
            _null_parameter_sha256=self._null_parameter_sha256,
            _bound_null_owner_sha256=owner_digest,
            _observed_content_guard_sha256=observed_content_guard,
        )
        diagnostics = (
            ("circular_shift_orbit_resolution_v2",)
            if 2 * min_shift == observed_length
            else ()
        )
        return NullBindResult(
            status=NullBindStatus.ENABLED,
            bound=bound,
            disabled_reason=None,
            diagnostics=diagnostics,
        )


@dataclass(frozen=True, slots=True, eq=False)
class _BoundCircularShiftV2:
    """One circular null owned by an immutable complete observed pair."""

    __hash__: ClassVar[None] = None  # type: ignore[assignment]

    _observed_pair: SeriesPair
    _min_shift: int
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
        if type(self._min_shift) is not int or self._min_shift < 1:
            raise V2IntegrityError("bound min_shift must be a positive built-in int")
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
        if 2 * self._min_shift > observed_length:
            raise V2IntegrityError("bound circular shift state space must be nonempty")
        expected_parameter = _parameter_digest(self._min_shift)
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
            MappingProxyType({"min_shift": self._min_shift}),
        )
        object.__setattr__(self, "_observed_length", observed_length)
        object.__setattr__(
            self,
            "_total_state_count",
            observed_length - 2 * self._min_shift + 2,
        )

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

    def _token_for_shift(self, shift: int) -> NullTransformToken:
        state = CircularShiftStateV2(schema=_STATE_SCHEMA, shift=shift)
        return NullTransformToken(
            schema=_TOKEN_SCHEMA,
            null_name=_NULL_NAME,
            null_parameter_sha256=self._null_parameter_sha256,
            semantic_input_sha256=self._semantic_input_sha256,
            scientific_plan_sha256=self._scientific_plan_sha256,
            bound_null_owner_sha256=self._bound_null_owner_sha256,
            is_identity=shift == 0,
            state=state,
        )

    def identity_token(self) -> NullTransformToken:
        return self._token_for_shift(0)

    def sample_token(self, random: UniformIndexSource, /) -> NullTransformToken:
        randbelow = cast(
            Callable[[int], object],
            _resolve_static_callable(
                random,
                method_name="randbelow",
                subject="random source",
            ),
        )
        sampled_index = randbelow(self._total_state_count)
        if type(sampled_index) is not int or not 0 <= sampled_index < self._total_state_count:
            raise V2IntegrityError("random index must be an exact in-range built-in int")
        shift = 0 if sampled_index == 0 else self._min_shift + sampled_index - 1
        return self._token_for_shift(shift)

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
                if type(original_state) is not CircularShiftStateV2:
                    raise V2IntegrityError("sampled circular token state drifted")
            snapshot = _snapshot_circular_token(token)
        except (AttributeError, TypeError, ValueError, OverflowError) as error:
            raise V2IntegrityError("sampled token snapshot is invalid") from error
        state = cast(CircularShiftStateV2, snapshot.state)
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
            (state.schema, state.shift),
            id(token),
            id(original_state),
        )

    def _validate_token_before_data_access(
        self,
        token: object,
    ) -> tuple[NullTransformToken, int]:
        (
            min_shift_value,
            semantic_value,
            plan_value,
            parameter_value,
            owner_value,
            observed_length_value,
        ) = snapshot_required_slots(
            self,
            (
                "_min_shift",
                "_semantic_input_sha256",
                "_scientific_plan_sha256",
                "_null_parameter_sha256",
                "_bound_null_owner_sha256",
                "_observed_length",
            ),
            subject="bound circular shift",
        )
        if type(min_shift_value) is not int or min_shift_value < 1:
            raise V2IntegrityError("bound circular parameter drifted")
        min_shift = min_shift_value
        if type(observed_length_value) is not int or observed_length_value < 1:
            raise V2IntegrityError("bound circular observed length drifted")
        observed_length = observed_length_value
        if 2 * min_shift > observed_length:
            raise V2IntegrityError("bound circular state space drifted")
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
        expected_parameter = _parameter_digest(min_shift)
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
        snapshot = _snapshot_circular_token(token)
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
        state = cast(CircularShiftStateV2, snapshot.state)
        if state.shift != 0 and not (
            min_shift <= state.shift <= observed_length - min_shift
        ):
            raise V2IntegrityError("token shift is outside the bound circular state space")
        return snapshot, observed_length

    def apply(self, token: NullTransformToken, /) -> NullTransformResult:
        snapshot, observed_length = self._validate_token_before_data_access(token)
        observed_value, guard_value = snapshot_required_slots(
            self,
            ("_observed_pair", "_observed_content_guard_sha256"),
            subject="bound circular shift",
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
            raise V2IntegrityError("bound circular observed pair length drifted")
        state = cast(CircularShiftStateV2, snapshot.state)
        expected_source_bytes = _expected_right_shift_bytes(
            golden.source,
            state.shift,
        )
        transformed = _apply_state(prepared.transform_input, state.shift)
        return finalize_transform_result(
            transformed=transformed,
            golden=golden,
            token=snapshot,
            expected_source_bytes=expected_source_bytes,
            mapping_name="right shift equation",
        )


__all__ = ["CircularShiftNullV2", "circular_token_sha256"]
