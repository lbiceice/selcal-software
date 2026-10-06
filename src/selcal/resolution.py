"""Explicit resolution of immutable scientific plan requests."""

from __future__ import annotations

from dataclasses import InitVar, dataclass

from selcal.canonical import canonical_json_bytes
from selcal.contracts import (
    _RESOLUTION_SEAL,
    JsonValue,
    PlanRequest,
    ResolvedScientificPlan,
)
from selcal.nulls.base import NullModelContract
from selcal.nulls.block_shuffle import BlockShuffleContract
from selcal.nulls.circular_shift import CircularShiftContract
from selcal.parameters import freeze_exact_json_mapping
from selcal.registry import AdapterRegistry, require_versioned_name
from selcal.statistics.base import StatisticAdapter
from selcal.statistics.binned_nette import BinnedNetTEAdapter as EqualWidthBinnedNetTE
from selcal.statistics.lagged_pearson import LaggedPearsonAdapter

_STATISTIC_REGISTRY: AdapterRegistry[StatisticAdapter] = AdapterRegistry(
    (
        ("equal_width_binned_nette_v1", EqualWidthBinnedNetTE.from_parameters),
        ("lagged_pearson_v1", LaggedPearsonAdapter.from_parameters),
    )
)
_NULL_REGISTRY: AdapterRegistry[NullModelContract] = AdapterRegistry(
    (
        ("circular_shift_v1", CircularShiftContract.from_parameters),
        ("block_shuffle_v1", BlockShuffleContract.from_parameters),
    )
)


@dataclass(frozen=True, slots=True)
class ResolvedAdapters:
    """The runtime adapters verified against a resolved scientific plan."""

    statistic: StatisticAdapter
    null_model: NullModelContract


@dataclass(frozen=True, slots=True)
class PlanResolution:
    """A resolver-sealed association between a plan and its exact adapters."""

    plan: ResolvedScientificPlan
    adapters: ResolvedAdapters
    seal: InitVar[object] = None

    def __post_init__(self, seal: object) -> None:
        # This private-object seal is an API barrier, not a cryptographic boundary.
        if seal is not _RESOLUTION_SEAL:
            raise ValueError("PlanResolution must be created by plan resolution")
        if type(self) is not PlanResolution:
            raise ValueError("resolution result must be an exact PlanResolution")
        if type(self.plan) is not ResolvedScientificPlan:
            raise ValueError("resolution plan must be an exact ResolvedScientificPlan")
        if type(self.adapters) is not ResolvedAdapters:
            raise ValueError("resolution adapters must be an exact ResolvedAdapters")

        # These helpers are defined later in the module and are available whenever
        # construction can occur after normal module initialization.
        _verified_statistic(
            self.adapters.statistic,
            expected_name=self.plan.statistic_name,
            expected_params=self.plan.statistic_params,
        )
        _verified_null(
            self.adapters.null_model,
            expected_name=self.plan.null_name,
            expected_params=self.plan.null_params,
        )


def _verified_statistic(
    adapter: object,
    *,
    expected_name: str,
    expected_params: JsonValue,
) -> StatisticAdapter:
    if not isinstance(adapter, StatisticAdapter):
        raise ValueError("statistic registry result must implement StatisticAdapter")
    _verify_identity(
        adapter.name,
        adapter.parameters,
        expected_name=expected_name,
        expected_params=expected_params,
        role="statistic",
    )
    return adapter


def _verified_null(
    adapter: object,
    *,
    expected_name: str,
    expected_params: JsonValue,
) -> NullModelContract:
    if not isinstance(adapter, NullModelContract):
        raise ValueError("null registry result must implement NullModelContract")
    _verify_identity(
        adapter.name,
        adapter.parameters,
        expected_name=expected_name,
        expected_params=expected_params,
        role="null",
    )
    return adapter


def _verify_identity(
    reported_name: object,
    reported_params: object,
    *,
    expected_name: str,
    expected_params: JsonValue,
    role: str,
) -> None:
    normalized_name = require_versioned_name(reported_name, name=f"{role} adapter name")
    if normalized_name != expected_name:
        raise ValueError(f"{role} adapter name does not match the plan request")
    normalized_params = freeze_exact_json_mapping(
        reported_params,
        name=f"{role} adapter parameters",
    )
    if canonical_json_bytes(normalized_params) != canonical_json_bytes(expected_params):
        raise ValueError(f"{role} adapter parameters do not match the plan request")


def resolve_plan(request: PlanRequest) -> PlanResolution:
    """Resolve an exact request through only the explicit built-in registries."""

    if type(request) is not PlanRequest:
        raise TypeError("request must be an exact PlanRequest")

    statistic = _verified_statistic(
        _STATISTIC_REGISTRY.build(request.statistic_name, request.statistic_params),
        expected_name=request.statistic_name,
        expected_params=request.statistic_params,
    )
    null = _verified_null(
        _NULL_REGISTRY.build(request.null_name, request.null_params),
        expected_name=request.null_name,
        expected_params=request.null_params,
    )
    plan = ResolvedScientificPlan(
        candidates=request.candidates,
        statistic_name=request.statistic_name,
        statistic_params=request.statistic_params,
        selection_rule=request.selection_rule,
        null_name=request.null_name,
        null_params=request.null_params,
        replicates=request.replicates,
        alpha=request.alpha,
        tie_tolerance=request.tie_tolerance,
        root_seed=request.root_seed,
        failure_policy=request.failure_policy,
        seal=_RESOLUTION_SEAL,
    )

    _verify_identity(
        statistic.name,
        statistic.parameters,
        expected_name=plan.statistic_name,
        expected_params=plan.statistic_params,
        role="statistic",
    )
    _verify_identity(
        null.name,
        null.parameters,
        expected_name=plan.null_name,
        expected_params=plan.null_params,
        role="null",
    )
    return PlanResolution(
        plan=plan,
        adapters=ResolvedAdapters(statistic=statistic, null_model=null),
        seal=_RESOLUTION_SEAL,
    )
