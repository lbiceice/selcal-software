"""Exact, immutable parameter handling for registered scientific adapters."""

from __future__ import annotations

import gc
import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import TypeAlias, cast

JsonValue: TypeAlias = (
    bool
    | int
    | float
    | str
    | list["JsonValue"]
    | tuple["JsonValue", ...]
    | Mapping[str, "JsonValue"]
    | None
)

# Bound recursion so malformed parameter payloads fail as ValueError, not RecursionError.
_MAX_JSON_NESTING_DEPTH = 64


def freeze_exact_json_mapping(value: object, *, name: str) -> Mapping[str, JsonValue]:
    """Copy an exact JSON mapping into recursively immutable canonical values."""

    return _freeze_exact_json_mapping(value, name=name, active=set(), depth=0)


def _exact_plain_dict(value: object, *, name: str) -> dict[object, object]:
    if type(value) is dict:
        return cast(dict[object, object], value)
    if type(value) is MappingProxyType:
        referents = gc.get_referents(value)
        if len(referents) == 1 and type(referents[0]) is dict:
            return cast(dict[object, object], referents[0])
    raise ValueError(f"{name} must be an exact built-in dict or MappingProxyType")


def _require_safe_container(
    container: object,
    *,
    name: str,
    active: set[int],
    depth: int,
) -> int:
    if depth > _MAX_JSON_NESTING_DEPTH:
        raise ValueError(f"{name} exceeds the maximum JSON nesting depth")
    identifier = id(container)
    if identifier in active:
        raise ValueError(f"{name} contains a cyclic JSON container")
    active.add(identifier)
    return identifier


def _freeze_exact_json_mapping(
    value: object,
    *,
    name: str,
    active: set[int],
    depth: int,
) -> Mapping[str, JsonValue]:
    source = _exact_plain_dict(value, name=name)
    identifier = _require_safe_container(
        source,
        name=name,
        active=active,
        depth=depth,
    )
    frozen: dict[str, JsonValue] = {}
    try:
        for key, nested in dict.items(source):
            if type(key) is not str:
                raise ValueError(f"{name} keys must be built-in strings")
            if key == "$float64":
                raise ValueError(f"{name} key $float64 is reserved")
            frozen[key] = _freeze_exact_json(
                nested,
                name=name,
                active=active,
                depth=depth + 1,
            )
    finally:
        active.remove(identifier)
    return MappingProxyType(frozen)


def _freeze_exact_json(
    value: object,
    *,
    name: str,
    active: set[int],
    depth: int,
) -> JsonValue:
    if value is None or type(value) in {bool, str, int}:
        return cast(JsonValue, value)
    if type(value) is float:
        number = value
        if not math.isfinite(number):
            raise ValueError(f"{name} numbers must be finite")
        return 0.0 if number == 0.0 else number
    if type(value) is dict or type(value) is MappingProxyType:
        return _freeze_exact_json_mapping(
            value,
            name=name,
            active=active,
            depth=depth,
        )
    if type(value) is list:
        identifier = _require_safe_container(
            value,
            name=name,
            active=active,
            depth=depth,
        )
        try:
            return tuple(
                _freeze_exact_json(
                    item,
                    name=name,
                    active=active,
                    depth=depth + 1,
                )
                for item in value
            )
        finally:
            active.remove(identifier)
    raise ValueError(f"{name} must contain exact JSON values")


def require_exact_keys(
    parameters: Mapping[str, JsonValue], *, expected: frozenset[str], name: str
) -> None:
    if frozenset(parameters) != expected:
        raise ValueError(f"{name} must contain exactly {sorted(expected)}")


def require_builtin_int(value: object, *, minimum: int, name: str) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be a built-in integer >= {minimum}")
    return value
