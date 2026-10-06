"""Declarative null-model contracts; transformations are a separate milestone."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from selcal.contracts import JsonValue


@runtime_checkable
class NullModelContract(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def parameters(self) -> Mapping[str, JsonValue]: ...

    @property
    def transformed_role(self) -> str: ...

    @property
    def preserved_quantities(self) -> tuple[str, ...]: ...

    @property
    def broken_relations(self) -> tuple[str, ...]: ...

    @property
    def applicability_assumptions(self) -> tuple[str, ...]: ...

    @property
    def disabled_conditions(self) -> tuple[str, ...]: ...
