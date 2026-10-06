from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, replace
from types import MappingProxyType
from typing import Self, cast

import numpy as np
import pytest

import selcal.resolution as resolution
from selcal.canonical import scientific_plan_sha256
from selcal.contracts import JsonValue, PlanRequest, ResolvedScientificPlan, SeriesPair
from selcal.nulls.base import NullModelContract
from selcal.nulls.block_shuffle import BlockShuffleContract
from selcal.nulls.circular_shift import CircularShiftContract
from selcal.registry import AdapterRegistry
from selcal.statistics.base import BoundStatisticAdapter, StatisticAdapter
from selcal.statistics.binned_nette import BinnedNetTEAdapter
from selcal.statistics.lagged_pearson import LaggedPearsonAdapter


def valid_request(**changes: object) -> PlanRequest:
    values = {
        "candidates": (3, 1, 2),
        "statistic_name": "equal_width_binned_nette_v1",
        "statistic_params": {"bins": 3},
        "selection_rule": "max_upper",
        "null_name": "circular_shift_v1",
        "null_params": {"min_shift": 1},
        "replicates": 9,
        "alpha": 0.05,
        "tie_tolerance": 1e-12,
        "root_seed": 17,
        "failure_policy": "fail_closed_v1",
    }
    values.update(changes)
    return PlanRequest(**values)  # type: ignore[arg-type]


def resolved_plan_kwargs() -> dict[str, object]:
    request = valid_request()
    return {field.name: getattr(request, field.name) for field in fields(PlanRequest)}


def test_builtin_resolution_returns_aligned_immutable_contracts() -> None:
    result = resolution.resolve_plan(valid_request())

    assert type(result) is resolution.PlanResolution
    assert type(result.adapters) is resolution.ResolvedAdapters
    assert type(result.plan) is ResolvedScientificPlan
    assert isinstance(result.adapters.statistic, BinnedNetTEAdapter)
    assert isinstance(result.adapters.statistic, StatisticAdapter)
    assert isinstance(result.adapters.null_model, CircularShiftContract)
    assert isinstance(result.adapters.null_model, NullModelContract)
    assert result.plan.candidates == (1, 2, 3)
    assert result.plan.statistic_name == result.adapters.statistic.name
    assert result.plan.statistic_params == result.adapters.statistic.parameters == {"bins": 3}
    assert result.plan.null_name == result.adapters.null_model.name
    assert result.plan.null_params == result.adapters.null_model.parameters == {"min_shift": 1}
    assert isinstance(result.plan.statistic_params, MappingProxyType)
    assert isinstance(result.plan.null_params, MappingProxyType)
    assert "seal" not in repr(result.plan)
    assert "seal" not in {field.name for field in fields(ResolvedScientificPlan)}
    with pytest.raises(TypeError):
        result.plan.statistic_params["bins"] = 4


def test_lagged_pearson_resolution_is_aligned_sealed_and_scientifically_distinct() -> None:
    binned = resolution.resolve_plan(valid_request())
    pearson = resolution.resolve_plan(
        valid_request(
            statistic_name="lagged_pearson_v1",
            statistic_params={},
        )
    )

    assert type(pearson) is resolution.PlanResolution
    assert type(pearson.plan) is ResolvedScientificPlan
    assert type(pearson.adapters) is resolution.ResolvedAdapters
    assert isinstance(pearson.adapters.statistic, LaggedPearsonAdapter)
    assert pearson.plan.statistic_name == pearson.adapters.statistic.name
    assert pearson.plan.statistic_params == pearson.adapters.statistic.parameters == {}
    assert isinstance(pearson.plan.statistic_params, MappingProxyType)
    assert isinstance(pearson.adapters.statistic.parameters, MappingProxyType)
    with pytest.raises(TypeError):
        pearson.plan.statistic_params["hidden"] = 1
    assert scientific_plan_sha256(pearson.plan) != scientific_plan_sha256(binned.plan)


def test_plan_resolution_rejects_forged_pearson_binned_association() -> None:
    legitimate = resolution.resolve_plan(valid_request())
    mismatched = resolution.ResolvedAdapters(
        statistic=LaggedPearsonAdapter(),
        null_model=legitimate.adapters.null_model,
    )

    with pytest.raises(ValueError, match="statistic"):
        resolution.PlanResolution(
            plan=legitimate.plan,
            adapters=mismatched,
            seal=resolution._RESOLUTION_SEAL,  # type: ignore[attr-defined]
        )


def test_resolution_public_field_names_are_exact() -> None:
    assert tuple(field.name for field in fields(resolution.PlanResolution)) == (
        "plan",
        "adapters",
    )
    assert tuple(field.name for field in fields(resolution.ResolvedAdapters)) == (
        "statistic",
        "null_model",
    )


@pytest.mark.parametrize(
    ("statistic", "null_model", "message"),
    [
        (
            BinnedNetTEAdapter(bins=4),
            BlockShuffleContract(block_length=2),
            "statistic",
        ),
        (
            BinnedNetTEAdapter(bins=4),
            CircularShiftContract(min_shift=1),
            "statistic",
        ),
        (
            BinnedNetTEAdapter(bins=3),
            CircularShiftContract(min_shift=2),
            "null",
        ),
    ],
)
def test_plan_resolution_rejects_mismatched_executable_adapters_even_with_private_seal(
    statistic: StatisticAdapter,
    null_model: NullModelContract,
    message: str,
) -> None:
    legitimate = resolution.resolve_plan(valid_request())
    mismatched = resolution.ResolvedAdapters(
        statistic=statistic,
        null_model=null_model,
    )

    with pytest.raises(ValueError, match=message):
        resolution.PlanResolution(
            plan=legitimate.plan,
            adapters=mismatched,
            seal=resolution._RESOLUTION_SEAL,  # type: ignore[attr-defined]
        )


def test_plan_resolution_rejects_omitted_and_wrong_seals() -> None:
    legitimate = resolution.resolve_plan(valid_request())

    with pytest.raises(ValueError, match=r"resolver|resolution"):
        resolution.PlanResolution(
            plan=legitimate.plan,
            adapters=legitimate.adapters,
        )
    with pytest.raises(ValueError, match=r"resolver|resolution"):
        resolution.PlanResolution(
            plan=legitimate.plan,
            adapters=legitimate.adapters,
            seal=object(),
        )


def test_plan_resolution_rejects_nonexact_plan_and_adapter_carriers() -> None:
    class PlanSubclass(ResolvedScientificPlan):
        pass

    class AdaptersSubclass(resolution.ResolvedAdapters):
        pass

    legitimate = resolution.resolve_plan(valid_request())
    plan_subclass = object.__new__(PlanSubclass)
    for field in fields(ResolvedScientificPlan):
        object.__setattr__(
            plan_subclass,
            field.name,
            getattr(legitimate.plan, field.name),
        )
    adapters_subclass = AdaptersSubclass(
        statistic=legitimate.adapters.statistic,
        null_model=legitimate.adapters.null_model,
    )

    for plan in (cast(ResolvedScientificPlan, object()), plan_subclass):
        with pytest.raises(ValueError, match="exact ResolvedScientificPlan"):
            resolution.PlanResolution(
                plan=plan,
                adapters=legitimate.adapters,
                seal=resolution._RESOLUTION_SEAL,  # type: ignore[attr-defined]
            )
    for adapters in (
        cast(resolution.ResolvedAdapters, object()),
        adapters_subclass,
    ):
        with pytest.raises(ValueError, match="exact ResolvedAdapters"):
            resolution.PlanResolution(
                plan=legitimate.plan,
                adapters=adapters,
                seal=resolution._RESOLUTION_SEAL,  # type: ignore[attr-defined]
            )


def test_plan_resolution_rejects_subclass_construction() -> None:
    class ResolutionSubclass(resolution.PlanResolution):
        pass

    legitimate = resolution.resolve_plan(valid_request())
    with pytest.raises(ValueError, match="exact PlanResolution"):
        ResolutionSubclass(
            plan=legitimate.plan,
            adapters=legitimate.adapters,
            seal=resolution._RESOLUTION_SEAL,  # type: ignore[attr-defined]
        )


def test_dataclass_replace_does_not_carry_the_resolution_seal() -> None:
    legitimate = resolution.resolve_plan(valid_request())

    with pytest.raises(ValueError, match=r"resolver|resolution"):
        replace(legitimate)


def test_only_resolved_plan_can_cross_the_scientific_hash_gate() -> None:
    with pytest.raises(TypeError, match="ResolvedScientificPlan"):
        scientific_plan_sha256(valid_request())  # type: ignore[arg-type]


def test_direct_resolved_plan_construction_rejects_a_foreign_seal() -> None:
    with pytest.raises(ValueError, match=r"resolver|resolution"):
        ResolvedScientificPlan(**resolved_plan_kwargs(), seal=object())  # type: ignore[arg-type]


def test_direct_resolved_plan_construction_rejects_an_omitted_seal() -> None:
    with pytest.raises(ValueError, match=r"resolver|resolution"):
        ResolvedScientificPlan(**resolved_plan_kwargs())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "changes",
    [
        {"statistic_name": "unknown_statistic_v1"},
        {"null_name": "unknown_null_v1"},
        {"statistic_params": {"bins": 1}},
        {"null_params": {"min_shift": 0}},
    ],
)
def test_unknown_names_and_invalid_adapter_parameters_fail_at_resolution(
    changes: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match=r"registered|bins|min_shift"):
        resolution.resolve_plan(valid_request(**changes))


def test_request_defensively_copies_nested_parameters_before_resolution() -> None:
    statistic_params = {"bins": 3, "nested": {"values": [1, 2]}}
    request = valid_request(
        statistic_name="unknown_statistic_v1",
        statistic_params=statistic_params,
    )
    statistic_params["nested"]["values"][0] = 99  # type: ignore[index]

    assert request.statistic_params["nested"] == {"values": (1, 2)}
    with pytest.raises(ValueError, match="registered"):
        resolution.resolve_plan(request)


class _WrongNameStatistic:
    bind_called = False

    @classmethod
    def from_parameters(cls, parameters: object) -> Self:
        del parameters
        return cls()

    @property
    def name(self) -> str:
        return "lying_statistic_v1"

    @property
    def parameters(self) -> Mapping[str, JsonValue]:
        return MappingProxyType({"bins": 3})

    def bind(
        self, observed_pair: SeriesPair, candidates: tuple[int, ...]
    ) -> BoundStatisticAdapter:
        del observed_pair, candidates
        type(self).bind_called = True
        raise AssertionError("resolver must not bind a mismatched adapter")


def test_wrong_adapter_name_fails_before_statistic_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _WrongNameStatistic.bind_called = False
    registry: AdapterRegistry[StatisticAdapter] = AdapterRegistry(
        (("equal_width_binned_nette_v1", lambda parameters: _WrongNameStatistic()),)
    )
    monkeypatch.setattr(resolution, "_STATISTIC_REGISTRY", registry)

    with pytest.raises(ValueError, match="name"):
        resolution.resolve_plan(valid_request())
    assert not _WrongNameStatistic.bind_called


def test_non_protocol_registry_result_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    registry: AdapterRegistry[StatisticAdapter] = AdapterRegistry(
        (("equal_width_binned_nette_v1", lambda parameters: cast(StatisticAdapter, object())),)
    )
    monkeypatch.setattr(resolution, "_STATISTIC_REGISTRY", registry)

    with pytest.raises(ValueError, match="StatisticAdapter"):
        resolution.resolve_plan(valid_request())


def test_non_protocol_null_registry_result_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry: AdapterRegistry[NullModelContract] = AdapterRegistry(
        (
            (
                "circular_shift_v1",
                lambda parameters: cast(NullModelContract, object()),
            ),
        )
    )
    monkeypatch.setattr(resolution, "_NULL_REGISTRY", registry)

    with pytest.raises(ValueError, match="NullModelContract"):
        resolution.resolve_plan(valid_request())


def test_registry_factory_exceptions_pass_through_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FactoryFailure(RuntimeError):
        pass

    failure = FactoryFailure("unchanged factory failure")

    def fail(parameters: Mapping[str, JsonValue]) -> StatisticAdapter:
        del parameters
        raise failure

    registry: AdapterRegistry[StatisticAdapter] = AdapterRegistry(
        (("equal_width_binned_nette_v1", fail),)
    )
    monkeypatch.setattr(resolution, "_STATISTIC_REGISTRY", registry)

    with pytest.raises(FactoryFailure) as captured:
        resolution.resolve_plan(valid_request())
    assert captured.value is failure


class _DriftingStatistic(_WrongNameStatistic):
    @property
    def name(self) -> str:
        return "equal_width_binned_nette_v1"

    @property
    def parameters(self) -> Mapping[str, JsonValue]:
        return MappingProxyType({"bins": 4})


def test_adapter_parameter_drift_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    registry: AdapterRegistry[StatisticAdapter] = AdapterRegistry(
        (("equal_width_binned_nette_v1", lambda parameters: _DriftingStatistic()),)
    )
    monkeypatch.setattr(resolution, "_STATISTIC_REGISTRY", registry)

    with pytest.raises(ValueError, match="parameter"):
        resolution.resolve_plan(valid_request())


def test_registry_entry_order_cannot_change_resolved_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = valid_request()
    baseline = resolution.resolve_plan(request)
    reversed_nulls: AdapterRegistry[NullModelContract] = AdapterRegistry(
        tuple(
            reversed(
                (
                    ("circular_shift_v1", CircularShiftContract.from_parameters),
                    ("block_shuffle_v1", BlockShuffleContract.from_parameters),
                )
            )
        )
    )
    monkeypatch.setattr(resolution, "_NULL_REGISTRY", reversed_nulls)
    reordered = resolution.resolve_plan(request)

    assert reordered.plan == baseline.plan
    assert scientific_plan_sha256(reordered.plan) == scientific_plan_sha256(baseline.plan)


def test_resolve_plan_requires_the_exact_request_type() -> None:
    class RequestSubclass(PlanRequest):
        pass

    subclass = RequestSubclass(**resolved_plan_kwargs())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="exact PlanRequest"):
        resolution.resolve_plan(subclass)
    with pytest.raises(TypeError, match="exact PlanRequest"):
        resolution.resolve_plan(object())  # type: ignore[arg-type]


def test_plan_equality_rejects_subclasses_in_both_operand_orders() -> None:
    class RequestSubclass(PlanRequest):
        pass

    class ResolvedSubclass(ResolvedScientificPlan):
        pass

    request = valid_request()
    request_subclass = RequestSubclass(**resolved_plan_kwargs())  # type: ignore[arg-type]
    resolved = resolution.resolve_plan(request).plan
    resolved_subclass = object.__new__(ResolvedSubclass)
    for field in fields(ResolvedScientificPlan):
        object.__setattr__(resolved_subclass, field.name, getattr(resolved, field.name))

    assert request != request_subclass
    assert request_subclass != request
    assert resolved != resolved_subclass
    assert resolved_subclass != resolved


def test_plan_equality_returns_false_for_hostile_unrelated_operands_in_both_orders() -> None:
    class HostileEquality:
        def __eq__(self, other: object) -> bool:
            del other
            return True

    hostile = HostileEquality()
    request = valid_request()
    resolved = resolution.resolve_plan(request).plan

    assert PlanRequest.__eq__(request, hostile) is False
    assert PlanRequest.__eq__(hostile, request) is False  # type: ignore[operator]
    assert ResolvedScientificPlan.__eq__(resolved, hostile) is False
    assert ResolvedScientificPlan.__eq__(hostile, resolved) is False  # type: ignore[operator]
    assert not (request == hostile)
    assert not (resolved == hostile)


class _CustomMapping(Mapping[str, object]):
    def __getitem__(self, key: str) -> object:
        return 3

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(("bins",))

    def __len__(self) -> int:
        return 1


def test_request_keeps_the_exact_hardened_parameter_boundary() -> None:
    cyclic: dict[str, object] = {}
    cyclic["self"] = cyclic
    nested: object = 0
    for _ in range(65):
        nested = [nested]

    hostile = (
        _CustomMapping(),
        {"bins": np.int64(3)},
        {"bins": (3,)},
        {"bins": float("nan")},
        {"$float64": "collision"},
        {"nested": cyclic},
        {"nested": nested},
    )
    for parameters in hostile:
        with pytest.raises(ValueError):
            valid_request(statistic_params=parameters)
