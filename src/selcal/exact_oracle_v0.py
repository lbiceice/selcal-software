"""Independent finite-state algorithm oracle for the two frozen v2 nulls.

This internal reference path checks enumeration and full reselection. It does
not establish exchangeability, Type-I-error control, or scientific validity.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import ClassVar, NoReturn

import numpy as np

from selcal.contracts import SeriesPair
from selcal.contracts_v2 import ResourceLimitError, V2IntegrityError
from selcal.resolution_v2 import (
    PlanResolutionV2,
    _require_resolver_owned_resolution_v2,
)
from selcal.selection_v2 import select_family_v2
from selcal.statistics.base import BoundStatisticAdapter

MAX_EXACT_STATES_V0 = 100_000
EXACT_NULL_NAMES_V0 = frozenset({"circular_shift_v2", "block_shuffle_v2"})

__all__ = [
    "EXACT_NULL_NAMES_V0",
    "MAX_EXACT_STATES_V0",
    "ExactOracleResultV0",
    "exact_state_oracle_v0",
]


def _integrity(message: str) -> NoReturn:
    raise V2IntegrityError(message)


@dataclass(frozen=True, slots=True, eq=False)
class ExactOracleResultV0:
    """Minimal non-persistent result for a complete mathematical state set."""

    __hash__: ClassVar[None] = None  # type: ignore[assignment]

    p_exact: float
    total_state_count: int
    nonidentity_exceedance_count: int

    def __post_init__(self) -> None:
        if (
            type(self.total_state_count) is not int
            or not 1 <= self.total_state_count <= MAX_EXACT_STATES_V0
        ):
            _integrity("total_state_count is outside the V0 exact-oracle boundary")
        if (
            type(self.nonidentity_exceedance_count) is not int
            or not 0
            <= self.nonidentity_exceedance_count
            < self.total_state_count
        ):
            _integrity("nonidentity_exceedance_count is outside the state set")
        expected = (1 + self.nonidentity_exceedance_count) / self.total_state_count
        if type(self.p_exact) is not float or not math.isfinite(self.p_exact):
            _integrity("p_exact must be an exact finite built-in float")
        if self.p_exact != expected:
            _integrity("p_exact contradicts the exact-oracle counts")


def _cap_exceeded(total_state_count: int) -> NoReturn:
    raise ResourceLimitError(
        "EXACT_STATE_CAP_EXCEEDED: total mathematical state count "
        f"{total_state_count} exceeds {MAX_EXACT_STATES_V0}"
    )


def _circular_state_count(observed_length: int, min_shift: object) -> tuple[int, int]:
    if type(min_shift) is not int or min_shift < 1:
        _integrity("circular exact oracle requires a positive built-in min_shift")
    if 2 * min_shift > observed_length:
        _integrity("circular exact-oracle nonidentity shift space is empty")
    total_state_count = observed_length - 2 * min_shift + 2
    if total_state_count > MAX_EXACT_STATES_V0:
        _cap_exceeded(total_state_count)
    return total_state_count, min_shift


def _block_state_count(observed_length: int, block_length: object) -> tuple[int, int, int]:
    if type(block_length) is not int or block_length < 1:
        _integrity("block exact oracle requires a positive built-in block_length")
    if observed_length % block_length != 0:
        _integrity("block exact oracle requires the source length to be strictly divisible")
    block_count = observed_length // block_length
    if block_count < 2:
        _integrity("block exact oracle requires at least two labelled blocks")

    total_state_count = 1
    factor = 2
    while factor <= block_count:
        if total_state_count > MAX_EXACT_STATES_V0 // factor:
            _cap_exceeded(total_state_count * factor)
        total_state_count *= factor
        factor += 1
    return total_state_count, block_length, block_count


def _decision_statistic(
    bound_statistic: BoundStatisticAdapter,
    pair: SeriesPair,
    resolution: PlanResolutionV2,
) -> float:
    try:
        raw_results = bound_statistic.evaluate_all(pair)
    except V2IntegrityError:
        raise
    except (AttributeError, FloatingPointError, OverflowError, TypeError, ValueError) as error:
        raise V2IntegrityError("exact-oracle statistic evaluation failed") from error
    selection, _ = select_family_v2(
        raw_results,
        resolution.plan.candidates,
        resolution.plan.selection_rule,
        resolution.plan.tie_tolerance,
    )
    return selection.decision_statistic


def _circular_pair(pair: SeriesPair, shift: int) -> SeriesPair:
    source = np.empty_like(pair.source)
    observed_length = int(pair.source.size)
    for index in range(observed_length):
        source[index] = pair.source[(index - shift) % observed_length]
    return SeriesPair(source=source, target=pair.target)


def _block_pair(
    pair: SeriesPair,
    block_order: tuple[int, ...],
    block_length: int,
) -> SeriesPair:
    source = np.empty_like(pair.source)
    destination = 0
    for block_label in block_order:
        block_start = block_label * block_length
        for offset in range(block_length):
            source[destination] = pair.source[block_start + offset]
            destination += 1
    return SeriesPair(source=source, target=pair.target)


def _circular_exceedances(
    pair: SeriesPair,
    resolution: PlanResolutionV2,
    bound_statistic: BoundStatisticAdapter,
    *,
    observed_decision_statistic: float,
    min_shift: int,
) -> int:
    observed_length = int(pair.source.size)
    exceedance_count = 0
    for shift in range(min_shift, observed_length - min_shift + 1):
        transformed = _circular_pair(pair, shift)
        state_decision = _decision_statistic(bound_statistic, transformed, resolution)
        if state_decision >= observed_decision_statistic:
            exceedance_count += 1
    return exceedance_count


def _block_exceedances(
    pair: SeriesPair,
    resolution: PlanResolutionV2,
    bound_statistic: BoundStatisticAdapter,
    *,
    observed_decision_statistic: float,
    block_length: int,
    block_count: int,
) -> int:
    identity_order = tuple(range(block_count))
    exceedance_count = 0
    for block_order in itertools.permutations(range(block_count)):
        if block_order == identity_order:
            continue
        transformed = _block_pair(pair, block_order, block_length)
        state_decision = _decision_statistic(bound_statistic, transformed, resolution)
        if state_decision >= observed_decision_statistic:
            exceedance_count += 1
    return exceedance_count


def exact_state_oracle_v0(
    pair: SeriesPair,
    resolution: PlanResolutionV2,
    /,
) -> ExactOracleResultV0:
    """Enumerate V0 mathematical states with a full scan and reselection per state."""

    if type(pair) is not SeriesPair:
        _integrity("pair must be an exact SeriesPair")
    exact_resolution = _require_resolver_owned_resolution_v2(resolution)
    plan = exact_resolution.plan
    observed_length = int(pair.source.size)

    if plan.null_name == "circular_shift_v2":
        if frozenset(plan.null_params) != frozenset({"min_shift"}):
            _integrity("circular exact-oracle parameters are invalid")
        total_state_count, min_shift = _circular_state_count(
            observed_length,
            plan.null_params["min_shift"],
        )
        block_parameters: tuple[int, int] | None = None
    elif plan.null_name == "block_shuffle_v2":
        if frozenset(plan.null_params) != frozenset({"block_length"}):
            _integrity("block exact-oracle parameters are invalid")
        total_state_count, block_length, block_count = _block_state_count(
            observed_length,
            plan.null_params["block_length"],
        )
        min_shift = None
        block_parameters = (block_length, block_count)
    else:
        _integrity(
            "exact oracle V0 supports only circular_shift_v2 and block_shuffle_v2"
        )

    try:
        bound_statistic = exact_resolution.adapters.statistic.bind(
            pair,
            plan.candidates,
        )
    except V2IntegrityError:
        raise
    except (AttributeError, OverflowError, TypeError, ValueError) as error:
        raise V2IntegrityError("exact-oracle statistic binding failed") from error
    observed_decision_statistic = _decision_statistic(
        bound_statistic,
        pair,
        exact_resolution,
    )

    if plan.null_name == "circular_shift_v2":
        assert min_shift is not None
        nonidentity_exceedance_count = _circular_exceedances(
            pair,
            exact_resolution,
            bound_statistic,
            observed_decision_statistic=observed_decision_statistic,
            min_shift=min_shift,
        )
    else:
        assert block_parameters is not None
        block_length, block_count = block_parameters
        nonidentity_exceedance_count = _block_exceedances(
            pair,
            exact_resolution,
            bound_statistic,
            observed_decision_statistic=observed_decision_statistic,
            block_length=block_length,
            block_count=block_count,
        )

    return ExactOracleResultV0(
        p_exact=(1 + nonidentity_exceedance_count) / total_state_count,
        total_state_count=total_state_count,
        nonidentity_exceedance_count=nonidentity_exceedance_count,
    )
