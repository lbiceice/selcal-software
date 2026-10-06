"""Public structural contracts for SelCal statistic adapters."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, Self, runtime_checkable

from selcal.contracts import JsonValue, SelectionRule, SeriesPair, StatisticResult


@runtime_checkable
class StatisticAdapter(Protocol):
    @classmethod
    def from_parameters(cls, parameters: object) -> Self: ...

    @property
    def name(self) -> str: ...

    @property
    def parameters(self) -> Mapping[str, JsonValue]: ...

    def bind(
        self, observed_pair: SeriesPair, candidates: tuple[int, ...]
    ) -> BoundStatisticAdapter: ...


@runtime_checkable
class BoundStatisticAdapter(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def parameters(self) -> Mapping[str, JsonValue]: ...

    @property
    def candidates(self) -> tuple[int, ...]: ...

    @property
    def backend_identity(self) -> str: ...

    @property
    def preprocessing_identity(self) -> str: ...

    def evaluate_all(self, pair: SeriesPair, /) -> tuple[StatisticResult, ...]: ...

    def evaluate(
        self,
        pair: SeriesPair,
        candidates: tuple[int, ...],
        rule: SelectionRule,
    ) -> tuple[StatisticResult, ...]: ...
