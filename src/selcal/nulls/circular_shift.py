"""Declarative circular-shift null-model contract."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Self

from selcal.contracts import JsonValue
from selcal.parameters import (
    freeze_exact_json_mapping,
    require_builtin_int,
    require_exact_keys,
)


@dataclass(frozen=True, slots=True)
class CircularShiftContract:
    """Immutable parameter and semantic contract for circular-shift nulls."""

    min_shift: int
    _parameters: Mapping[str, JsonValue] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        normalized = require_builtin_int(self.min_shift, minimum=1, name="min_shift")
        object.__setattr__(self, "min_shift", normalized)
        object.__setattr__(
            self,
            "_parameters",
            MappingProxyType({"min_shift": normalized}),
        )

    @classmethod
    def from_parameters(cls, parameters: object) -> Self:
        frozen = freeze_exact_json_mapping(parameters, name="circular_shift parameters")
        require_exact_keys(
            frozen,
            expected=frozenset({"min_shift"}),
            name="circular_shift parameters",
        )
        return cls(
            min_shift=require_builtin_int(frozen["min_shift"], minimum=1, name="min_shift")
        )

    @property
    def name(self) -> str:
        return "circular_shift_v1"

    @property
    def parameters(self) -> Mapping[str, JsonValue]:
        return self._parameters

    @property
    def transformed_role(self) -> str:
        return "source"

    @property
    def preserved_quantities(self) -> tuple[str, ...]:
        return (
            "source_value_multiset_v1",
            "source_circular_autocorrelation_v1",
        )

    @property
    def broken_relations(self) -> tuple[str, ...]:
        return ("source_target_alignment_v1",)

    @property
    def applicability_assumptions(self) -> tuple[str, ...]:
        return ("cross_series_null_by_shift_exchangeability_v1",)

    @property
    def disabled_conditions(self) -> tuple[str, ...]:
        return ("series_length_not_greater_than_min_shift_v1",)
