"""Declarative block-shuffle null-model contract."""

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
class BlockShuffleContract:
    """Immutable parameter and semantic contract for block-shuffle nulls."""

    block_length: int
    _parameters: Mapping[str, JsonValue] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        normalized = require_builtin_int(
            self.block_length,
            minimum=1,
            name="block_length",
        )
        object.__setattr__(self, "block_length", normalized)
        object.__setattr__(
            self,
            "_parameters",
            MappingProxyType({"block_length": normalized}),
        )

    @classmethod
    def from_parameters(cls, parameters: object) -> Self:
        frozen = freeze_exact_json_mapping(parameters, name="block_shuffle parameters")
        require_exact_keys(
            frozen,
            expected=frozenset({"block_length"}),
            name="block_shuffle parameters",
        )
        return cls(
            block_length=require_builtin_int(
                frozen["block_length"],
                minimum=1,
                name="block_length",
            )
        )

    @property
    def name(self) -> str:
        return "block_shuffle_v1"

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
            "source_within_block_order_v1",
        )

    @property
    def broken_relations(self) -> tuple[str, ...]:
        return (
            "source_target_alignment_v1",
            "source_between_block_order_v1",
        )

    @property
    def applicability_assumptions(self) -> tuple[str, ...]:
        return ("local_dependence_captured_within_blocks_v1",)

    @property
    def disabled_conditions(self) -> tuple[str, ...]:
        return ("fewer_than_two_blocks_v1",)
