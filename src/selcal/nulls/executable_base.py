"""Runtime protocols and bind result for executable SelCal v2 nulls."""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import ClassVar, Protocol, cast, runtime_checkable

from selcal.contracts import JsonValue
from selcal.contracts_v2 import (
    NullBindStatus,
    NullDisabledReason,
    NullTransformResult,
    NullTransformToken,
    V2IntegrityError,
)

_MISSING_DESCRIPTOR = object()
_RANDOM_INDEX_CAPABILITY_SEAL_V2 = object()


def _freeze_random_index_capability_builder_v2(
    seal: object,
    tuple_type: type[tuple[object, ...]],
    callable_value: Callable[[object], bool],
    integrity_error_type: type[V2IntegrityError],
    /,
) -> Callable[[Callable[[object, int], int], object], tuple[object, ...]]:
    """Build an immutable per-stream bridge exposing only ``randbelow(bound)``."""

    def _build_random_index_capability_v2(
        random_operation: Callable[[object, int], int],
        random_stream: object,
        /,
    ) -> tuple[object, ...]:
        if not callable_value(random_operation):
            raise integrity_error_type("random capability operation must be callable")
        operation = random_operation
        stream = random_stream

        def randbelow(bound: int, /) -> int:
            return operation(stream, bound)

        return tuple_type((seal, randbelow))

    return _build_random_index_capability_v2


_build_random_index_capability_v2 = _freeze_random_index_capability_builder_v2(
    _RANDOM_INDEX_CAPABILITY_SEAL_V2,
    tuple,
    callable,
    V2IntegrityError,
)
del _freeze_random_index_capability_builder_v2


def _resolve_static_callable(
    source: object,
    *,
    method_name: str,
    subject: str,
) -> Callable[..., object]:
    """Resolve a method shape without invoking property getters during validation."""

    if type(source) is tuple and len(source) == 2 and source[0] is _RANDOM_INDEX_CAPABILITY_SEAL_V2:
        if method_name != "randbelow":
            raise V2IntegrityError(f"{subject} must provide a callable {method_name} method")
        candidate = source[1]
        if not callable(candidate):
            raise V2IntegrityError(f"{subject} {method_name} must be callable")
        return cast(Callable[..., object], candidate)

    try:
        descriptor = inspect.getattr_static(source, method_name)
        class_descriptor = inspect.getattr_static(
            type(source),
            method_name,
            _MISSING_DESCRIPTOR,
        )
    except (AttributeError, TypeError):
        raise V2IntegrityError(
            f"{subject} must provide a callable {method_name} method"
        ) from None

    if descriptor is class_descriptor:
        if isinstance(descriptor, property):
            raise V2IntegrityError(f"{subject} {method_name} property is forbidden")
        try:
            descriptor_get = descriptor.__get__
        except AttributeError:
            candidate = descriptor
        else:
            try:
                candidate = descriptor_get(source, type(source))
            except (AttributeError, TypeError) as error:
                raise V2IntegrityError(
                    f"{subject} {method_name} descriptor could not be bound"
                ) from error
    else:
        candidate = descriptor
    if not callable(candidate):
        raise V2IntegrityError(f"{subject} {method_name} must be callable")
    return cast(Callable[..., object], candidate)


@runtime_checkable
class UniformIndexSource(Protocol):
    """The only random-index operation executable v2 nulls may consume."""

    def randbelow(self, bound: int, /) -> int: ...


@runtime_checkable
class BoundNullModel(Protocol):
    """An executable v2 null irrevocably bound to one observed input."""

    @property
    def name(self) -> str: ...

    @property
    def parameters(self) -> Mapping[str, JsonValue]: ...

    @property
    def observed_length(self) -> int: ...

    @property
    def total_state_count(self) -> int: ...

    def identity_token(self) -> NullTransformToken: ...

    def sample_token(self, random: UniformIndexSource, /) -> NullTransformToken: ...

    def snapshot_token(self, token: object, /) -> tuple[object, ...]: ...

    def apply(self, token: NullTransformToken, /) -> NullTransformResult: ...


def _require_diagnostics(value: object) -> tuple[str, ...]:
    if type(value) is not tuple or any(type(entry) is not str for entry in value):
        raise V2IntegrityError("diagnostics must be an exact tuple of built-in strings")
    return value


_MAX_PARAMETER_NESTING_DEPTH = 64


def _validate_exact_json_snapshot(
    value: object,
    *,
    name: str,
    active_container_ids: set[int],
    depth: int,
) -> None:
    if depth > _MAX_PARAMETER_NESTING_DEPTH:
        raise V2IntegrityError(f"{name} exceeds the maximum JSON nesting depth")
    if value is None or type(value) in {bool, int, str}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise V2IntegrityError(f"{name} must contain only finite JSON numbers")
        return
    if type(value) in {list, tuple}:
        container_id = id(value)
        if container_id in active_container_ids:
            raise V2IntegrityError(f"{name} must not contain recursive containers")
        active_container_ids.add(container_id)
        try:
            items = tuple(cast(list[object] | tuple[object, ...], value))
            for item in items:
                _validate_exact_json_snapshot(
                    item,
                    name=name,
                    active_container_ids=active_container_ids,
                    depth=depth + 1,
                )
        finally:
            active_container_ids.remove(container_id)
        return
    if isinstance(value, Mapping):
        container_id = id(value)
        if container_id in active_container_ids:
            raise V2IntegrityError(f"{name} must not contain recursive containers")
        active_container_ids.add(container_id)
        try:
            mapping = cast(Mapping[object, object], value)
            mapping_items: tuple[tuple[object, object], ...] = tuple(mapping.items())
            for key, item in mapping_items:
                if type(key) is not str:
                    raise V2IntegrityError(f"{name} keys must be built-in strings")
                _validate_exact_json_snapshot(
                    item,
                    name=name,
                    active_container_ids=active_container_ids,
                    depth=depth + 1,
                )
        finally:
            active_container_ids.remove(container_id)
        return
    raise V2IntegrityError(f"{name} must contain only exact JSON values")


def _validate_structural_bound_snapshot(bound: object) -> None:
    if not isinstance(bound, BoundNullModel):
        raise V2IntegrityError("ENABLED requires a structurally valid bound null")
    name = bound.name
    parameters = bound.parameters
    observed_length = bound.observed_length
    total_state_count = bound.total_state_count
    methods = (
        bound.identity_token,
        bound.sample_token,
        bound.snapshot_token,
        bound.apply,
    )

    if type(name) is not str or name == "":
        raise V2IntegrityError("bound name must be a non-empty exact string")
    if not isinstance(parameters, Mapping):
        raise V2IntegrityError("bound parameters must satisfy the Mapping protocol")
    _validate_exact_json_snapshot(
        parameters,
        name="bound parameters",
        active_container_ids=set(),
        depth=0,
    )
    for field_name, value in (
        ("observed_length", observed_length),
        ("total_state_count", total_state_count),
    ):
        if type(value) is not int or value < 1:
            raise V2IntegrityError(f"bound {field_name} must be a positive built-in int")
    for method_name, method in zip(
        ("identity_token", "sample_token", "snapshot_token", "apply"),
        methods,
        strict=True,
    ):
        if not callable(method):
            raise V2IntegrityError(f"bound {method_name} must be callable")


@dataclass(frozen=True, slots=True, eq=False)
class NullBindResult:
    """Fail-closed enabled/disabled result from binding an executable v2 null."""

    __hash__: ClassVar[None] = None  # type: ignore[assignment]

    status: NullBindStatus
    bound: BoundNullModel | None
    disabled_reason: NullDisabledReason | None
    diagnostics: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.status) is not NullBindStatus:
            raise V2IntegrityError("status must be an exact NullBindStatus")
        object.__setattr__(self, "diagnostics", _require_diagnostics(self.diagnostics))
        if self.status is NullBindStatus.ENABLED:
            if self.bound is None:
                raise V2IntegrityError("ENABLED requires a structurally valid bound null")
            _validate_structural_bound_snapshot(self.bound)
            if self.disabled_reason is not None:
                raise V2IntegrityError("ENABLED requires disabled_reason None")
        else:
            if self.bound is not None:
                raise V2IntegrityError("DISABLED requires bound None")
            if type(self.disabled_reason) is not NullDisabledReason:
                raise V2IntegrityError(
                    "DISABLED requires an exact enumerated disabled_reason"
                )

    def __eq__(self, other: object) -> bool:
        return (
            type(self) is NullBindResult
            and type(other) is NullBindResult
            and self.status is other.status
            and self.bound is other.bound
            and self.disabled_reason is other.disabled_reason
            and self.diagnostics == other.diagnostics
        )
