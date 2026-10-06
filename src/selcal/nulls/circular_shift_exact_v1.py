"""Exact-enumeration circular null: the full cyclic group, each state used exactly once.

The sampling null `circular_shift_v2` draws B states with replacement, so p = (1 + E)/(B + 1)
carries Monte Carlo loss. Using every non-identity state exactly once makes B = n - 1 and
p = #{s : T_s >= T_0}/n, the exact test, without changing the p-value formula. This null therefore
never samples; the replicate index selects the state. The right-shift index equation is imported
from `circular_shift_v2` so that both nulls share one frozen transformation.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import ClassVar, Literal, Self, cast

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
from selcal.nulls.circular_shift_v2 import _apply_state as _apply_state
from selcal.nulls.circular_shift_v2 import (
    _expected_right_shift_bytes as _expected_right_shift_bytes,
)
from selcal.nulls.executable_base import NullBindResult, UniformIndexSource
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

_NULL_NAME = "circular_shift_exact_v1"
_TOKEN_SCHEMA: Literal["selcal.null-transform-token.v2"] = "selcal.null-transform-token.v2"
_STATE_SCHEMA: Literal["selcal.circular-shift-state.v2"] = "selcal.circular-shift-state.v2"
_ONLY_MIN_SHIFT = 1
_TOKEN_DESCRIPTOR_SNAPSHOT = _freeze_token_descriptor_snapshot(
    CircularShiftStateV2,
    ("schema", "shift"),
)


def _parameter_digest(min_shift: int) -> str:
    return parameter_digest(null_name=_NULL_NAME, params={"min_shift": min_shift})


def _snapshot_exact_token(token: object) -> NullTransformToken:
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
        raise V2IntegrityError("enumerated circular token state is invalid") from error
    if (
        type(state_schema) is not str
        or state_schema != _STATE_SCHEMA
        or type(shift) is not int
        or shift < 0
    ):
        raise V2IntegrityError("enumerated circular token state drifted")
    if common.is_identity is not (shift == 0):
        raise V2IntegrityError("token identity flag contradicts its circular state")
    return build_owned_token(common, state=CircularShiftStateV2(schema=state_schema, shift=shift))


def _token_payload(token: NullTransformToken) -> JsonValue:
    snapshot = _snapshot_exact_token(token)
    state = cast(CircularShiftStateV2, snapshot.state)
    payload = common_token_payload(snapshot)
    payload["state"] = {"schema": state.schema, "shift": state.shift}
    return payload


def exact_circular_token_sha256(token: NullTransformToken, /) -> str:
    """Return the optional evidence digest of one exact enumerated circular token."""

    return hashlib.sha256(canonical_json_bytes(_token_payload(token))).hexdigest()


@dataclass(frozen=True, slots=True, eq=False)
class CircularShiftExactNullV1:
    """Unbound exact-enumeration circular null; only the complete group is valid."""

    __hash__: ClassVar[None] = None  # type: ignore[assignment]

    min_shift: int
    _parameters: Mapping[str, JsonValue] = field(init=False, repr=False)
    _null_parameter_sha256: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            exact_min_shift = require_builtin_int(self.min_shift, minimum=1, name="min_shift")
        except ValueError as error:
            raise V2IntegrityError(str(error)) from error
        if exact_min_shift != _ONLY_MIN_SHIFT:
            raise V2IntegrityError(
                "circular_shift_exact_v1 enumerates the complete group and requires min_shift 1"
            )
        object.__setattr__(self, "min_shift", exact_min_shift)
        object.__setattr__(self, "_parameters", MappingProxyType({"min_shift": exact_min_shift}))
        object.__setattr__(self, "_null_parameter_sha256", _parameter_digest(exact_min_shift))

    @classmethod
    def from_parameters(cls, parameters: object) -> Self:
        try:
            frozen = freeze_exact_json_mapping(
                parameters, name="circular_shift_exact_v1 parameters"
            )
            require_exact_keys(
                frozen,
                expected=frozenset({"min_shift"}),
                name="circular_shift_exact_v1 parameters",
            )
            min_shift = require_builtin_int(frozen["min_shift"], minimum=1, name="min_shift")
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
        semantic_digest = _require_sha256(semantic_input_sha256, name="semantic_input_sha256")
        plan_digest = _require_sha256(scientific_plan_sha256, name="scientific_plan_sha256")
        try:
            observed_source = observed_pair.source
            observed_target = observed_pair.target
            observed_length = int(observed_source.size)
        except (AttributeError, TypeError, ValueError) as error:
            raise V2IntegrityError("observed_pair fields could not be snapshotted") from error
        if 2 * _ONLY_MIN_SHIFT > observed_length:
            return NullBindResult(
                status=NullBindStatus.DISABLED,
                bound=None,
                disabled_reason=NullDisabledReason.SHIFT_SPACE_EMPTY,
                diagnostics=("shift_space_empty_exact_v1",),
            )
        observed_snapshot = SeriesPair(source=observed_source, target=observed_target)
        bound = _BoundCircularShiftExactV1(
            _observed_pair=observed_snapshot,
            _min_shift=_ONLY_MIN_SHIFT,
            _semantic_input_sha256=semantic_digest,
            _scientific_plan_sha256=plan_digest,
            _null_parameter_sha256=self._null_parameter_sha256,
            _bound_null_owner_sha256=_owner_digest(
                semantic_input_sha256=semantic_digest,
                scientific_plan_sha256=plan_digest,
                null_parameter_sha256=self._null_parameter_sha256,
                observed_length=observed_length,
            ),
            _observed_content_guard_sha256=_owned_pair_content_guard_sha256(observed_snapshot),
        )
        return NullBindResult(
            status=NullBindStatus.ENABLED,
            bound=bound,
            disabled_reason=None,
            diagnostics=(),
        )


@dataclass(frozen=True, slots=True, eq=False)
class _BoundCircularShiftExactV1:
    """One enumerated circular null owned by an immutable complete observed pair."""

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
        if type(self._min_shift) is not int or self._min_shift != _ONLY_MIN_SHIFT:
            raise V2IntegrityError("bound exact circular null requires min_shift 1")
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
            expected_content_guard = _owned_pair_content_guard_sha256(self._observed_pair)
        except (AttributeError, TypeError, ValueError) as error:
            raise V2IntegrityError("bound observed pair snapshot is invalid") from error
        if self._observed_content_guard_sha256 != expected_content_guard:
            raise V2IntegrityError("bound observed pair content guard drifted")
        if 2 * self._min_shift > observed_length:
            raise V2IntegrityError("bound circular shift state space must be nonempty")
        expected_parameter = _parameter_digest(self._min_shift)
        if self._null_parameter_sha256 != expected_parameter:
            raise V2IntegrityError("bound null-parameter digest does not match parameters")
        if self._bound_null_owner_sha256 != _owner_digest(
            semantic_input_sha256=self._semantic_input_sha256,
            scientific_plan_sha256=self._scientific_plan_sha256,
            null_parameter_sha256=expected_parameter,
            observed_length=observed_length,
        ):
            raise V2IntegrityError("bound owner digest does not match its ownership fields")
        object.__setattr__(self, "_parameters", MappingProxyType({"min_shift": self._min_shift}))
        object.__setattr__(self, "_observed_length", observed_length)
        object.__setattr__(self, "_total_state_count", observed_length)

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
        return NullTransformToken(
            schema=_TOKEN_SCHEMA,
            null_name=_NULL_NAME,
            null_parameter_sha256=self._null_parameter_sha256,
            semantic_input_sha256=self._semantic_input_sha256,
            scientific_plan_sha256=self._scientific_plan_sha256,
            bound_null_owner_sha256=self._bound_null_owner_sha256,
            is_identity=shift == 0,
            state=CircularShiftStateV2(schema=_STATE_SCHEMA, shift=shift),
        )

    def identity_token(self) -> NullTransformToken:
        return self._token_for_shift(0)

    def enumerate_token(self, index: object, /) -> NullTransformToken:
        """Return the state for shift index + 1; identity is never enumerated."""

        if type(index) is not int or not 0 <= index < self._total_state_count - 1:
            raise V2IntegrityError("enumeration index must be an exact in-range built-in int")
        return self._token_for_shift(index + 1)

    def sample_token(self, random: UniformIndexSource, /) -> NullTransformToken:
        """Refuse: this null enumerates its complete state space and never samples."""

        del random
        raise V2IntegrityError("circular_shift_exact_v1 enumerates and never samples a state")

    def snapshot_token(self, token: object, /) -> tuple[object, ...]:
        """Return the exact immutable token/state snapshot owned by this null."""

        original_state = token
        try:
            if type(token) is NullTransformToken:
                original_state = snapshot_required_slots(token, ("state",), subject="token")[0]
                if type(original_state) is not CircularShiftStateV2:
                    raise V2IntegrityError("enumerated circular token state drifted")
            snapshot = _snapshot_exact_token(token)
        except (AttributeError, TypeError, ValueError, OverflowError) as error:
            raise V2IntegrityError("enumerated token snapshot is invalid") from error
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

    def _validate_token_before_data_access(self, token: object) -> tuple[NullTransformToken, int]:
        (
            min_shift_value,
            semantic_value,
            plan_value,
            parameter_value,
            owner_value,
            observed_value,
        ) = snapshot_required_slots(
            self,
            (
                "_min_shift",
                "_semantic_input_sha256",
                "_scientific_plan_sha256",
                "_null_parameter_sha256",
                "_bound_null_owner_sha256",
                "_observed_pair",
            ),
            subject="bound exact circular shift",
        )
        if type(min_shift_value) is not int or min_shift_value != _ONLY_MIN_SHIFT:
            raise V2IntegrityError("bound exact circular null requires min_shift 1")
        if type(observed_value) is not SeriesPair:
            raise V2IntegrityError("bound observed pair must be an exact SeriesPair")
        observed_length = int(observed_value.source.size)
        snapshot = _snapshot_exact_token(token)
        if (
            snapshot.null_parameter_sha256 != parameter_value
            or snapshot.semantic_input_sha256 != semantic_value
            or snapshot.scientific_plan_sha256 != plan_value
            or snapshot.bound_null_owner_sha256 != owner_value
        ):
            raise V2IntegrityError("token ownership does not match this bound null")
        state = cast(CircularShiftStateV2, snapshot.state)
        if not 0 <= state.shift < observed_length:
            raise V2IntegrityError("token shift is outside the enumerated circular state space")
        return snapshot, observed_length

    def apply(self, token: NullTransformToken, /) -> NullTransformResult:
        snapshot, observed_length = self._validate_token_before_data_access(token)
        observed_value, guard_value = snapshot_required_slots(
            self,
            ("_observed_pair", "_observed_content_guard_sha256"),
            subject="bound exact circular shift",
        )
        observed_content_guard = _require_sha256(guard_value, name="observed_content_guard_sha256")
        prepared = prepare_owned_transform(
            observed_value, expected_content_guard_sha256=observed_content_guard
        )
        golden = prepared.golden
        if int(golden.source.size) != observed_length:
            raise V2IntegrityError("bound circular observed pair length drifted")
        state = cast(CircularShiftStateV2, snapshot.state)
        return finalize_transform_result(
            transformed=_apply_state(prepared.transform_input, state.shift),
            golden=golden,
            token=snapshot,
            expected_source_bytes=_expected_right_shift_bytes(golden.source, state.shift),
            mapping_name="right shift equation",
        )


__all__ = ["CircularShiftExactNullV1", "exact_circular_token_sha256"]
