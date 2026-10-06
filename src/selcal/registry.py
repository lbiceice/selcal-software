"""Explicit, duplicate-safe registries for built-in scientific contracts."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Generic, TypeAlias, TypeVar

from selcal.contracts import JsonValue
from selcal.identifiers import require_versioned_name as require_versioned_name

T_co = TypeVar("T_co", covariant=True)
Factory: TypeAlias = Callable[[Mapping[str, JsonValue]], T_co]
_MAX_REGISTRY_ENTRIES = 1024


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class AdapterRegistry(Generic[T_co]):
    """Build adapters from explicit versioned names, capped at 1,024 entries."""

    _factories: Mapping[str, Factory[T_co]]

    def __init__(self, entries: Iterable[tuple[str, Factory[T_co]]]) -> None:
        validated_entries: list[tuple[str, Factory[T_co]]] = []
        seen_names: set[str] = set()
        for position, entry in enumerate(entries, start=1):
            if position > _MAX_REGISTRY_ENTRIES:
                raise ValueError(
                    f"adapter registry may contain at most {_MAX_REGISTRY_ENTRIES} entries"
                )
            try:
                entry_name, factory = entry
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "adapter registry entry must contain a name and factory"
                ) from error
            normalized_name = require_versioned_name(entry_name, name="adapter name")
            if not callable(factory):
                raise ValueError("adapter factory must be callable")
            if normalized_name in seen_names:
                raise ValueError(f"duplicate adapter name: {normalized_name}")
            seen_names.add(normalized_name)
            validated_entries.append((normalized_name, factory))

        object.__setattr__(self, "_factories", MappingProxyType(dict(validated_entries)))

    def __repr__(self) -> str:
        """Represent only validated names, never arbitrary factory objects."""

        return f"{type(self).__name__}(names={self.names!r})"

    def __eq__(self, other: object) -> bool:
        """Use object identity without invoking any registered factory methods."""

        return self is other

    def __hash__(self) -> int:
        """Hash by object identity without invoking any registered factory methods."""

        return object.__hash__(self)

    @property
    def names(self) -> tuple[str, ...]:
        """Return registered names in a deterministic order."""

        return tuple(sorted(self._factories))

    def build(self, name: str, parameters: Mapping[str, JsonValue]) -> T_co:
        """Build exactly the adapter registered under ``name``."""

        normalized_name = require_versioned_name(name, name="adapter name")
        try:
            factory = self._factories[normalized_name]
        except KeyError as error:
            raise ValueError(f"adapter name is not registered: {normalized_name}") from error
        return factory(parameters)
