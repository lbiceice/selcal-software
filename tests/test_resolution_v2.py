from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, replace
from types import MappingProxyType
from typing import Self, cast

import pytest

import selcal.resolution_v2 as resolution_v2
from selcal import contracts_v2
from selcal.contracts import JsonValue, SeriesPair
from selcal.contracts_v2 import PlanRequestV2, ResolvedScientificPlanV2, V2IntegrityError
from selcal.nulls.block_shuffle import BlockShuffleContract
from selcal.nulls.block_shuffle_v2 import BlockShuffleNullV2
from selcal.nulls.circular_shift_v2 import CircularShiftNullV2
from selcal.nulls.executable_base import NullBindResult
from selcal.registry import AdapterRegistry
from selcal.statistics.base import BoundStatisticAdapter, StatisticAdapter
from selcal.statistics.binned_nette import BinnedNetTEAdapter
from selcal.statistics.lagged_pearson import LaggedPearsonAdapter


def valid_request(**changes: object) -> PlanRequestV2:
    values: dict[str, object] = {
        "candidates": (3, 1, 2),
        "statistic_name": "lagged_pearson_v1",
        "statistic_params": {},
        "selection_rule": "max_upper",
        "null_name": "circular_shift_v2",
        "null_params": {"min_shift": 1},
        "replicates": 9,
        "alpha": 0.05,
        "tie_tolerance": 1e-12,
        "root_seed": 17,
    }
    values.update(changes)
    return PlanRequestV2(**values)  # type: ignore[arg-type]


def test_builtin_v2_resolution_returns_exact_distinct_immutable_types() -> None:
    result = resolution_v2.resolve_plan_v2(valid_request())

    assert type(result) is resolution_v2.PlanResolutionV2
    assert type(result.adapters) is resolution_v2.ResolvedAdaptersV2
    assert type(result.plan) is ResolvedScientificPlanV2
    assert type(result.adapters.statistic) is LaggedPearsonAdapter
    assert type(result.adapters.null_model) is CircularShiftNullV2
    assert result.plan.candidates == (1, 2, 3)
    assert result.plan.statistic_name == result.adapters.statistic.name
    assert result.plan.statistic_params == result.adapters.statistic.parameters == {}
    assert result.plan.null_name == result.adapters.null_model.name
    assert result.plan.null_params == result.adapters.null_model.parameters == {"min_shift": 1}
    assert isinstance(result.plan.statistic_params, MappingProxyType)
    assert isinstance(result.plan.null_params, MappingProxyType)
    assert tuple(field.name for field in fields(ResolvedScientificPlanV2)) == (
        "candidates",
        "statistic_name",
        "statistic_params",
        "selection_rule",
        "null_name",
        "null_params",
        "replicates",
        "alpha",
        "tie_tolerance",
        "root_seed",
    )
    assert tuple(field.name for field in fields(resolution_v2.ResolvedAdaptersV2)) == (
        "statistic",
        "null_model",
    )
    assert tuple(field.name for field in fields(resolution_v2.PlanResolutionV2)) == (
        "plan",
        "adapters",
    )
    assert resolution_v2._require_resolver_owned_resolution_v2(result) is result


def test_v2_resolution_uses_separate_registries_and_only_executable_v2_nulls() -> None:
    assert (
        resolution_v2._STATISTIC_REGISTRY_V2
        is not __import__("selcal.resolution", fromlist=["_STATISTIC_REGISTRY"])._STATISTIC_REGISTRY
    )
    assert (
        resolution_v2._NULL_REGISTRY_V2
        is not __import__("selcal.resolution", fromlist=["_NULL_REGISTRY"])._NULL_REGISTRY
    )
    assert resolution_v2._STATISTIC_REGISTRY_V2.names == (
        "equal_width_binned_nette_v1",
        "lagged_pearson_v1",
    )
    assert resolution_v2._NULL_REGISTRY_V2.names == (
        "block_shuffle_v2",
        "circular_shift_exact_v1",
        "circular_shift_v2",
    )
    for legacy_name in ("circular_shift_v1", "block_shuffle_v1"):
        with pytest.raises(ValueError, match="registered"):
            resolution_v2.resolve_plan_v2(
                valid_request(null_name=legacy_name, null_params={"min_shift": 1})
            )


def test_v2_resolution_rejects_duplicate_and_unknown_registry_names() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        AdapterRegistry(
            (
                ("circular_shift_v2", CircularShiftNullV2.from_parameters),
                ("circular_shift_v2", CircularShiftNullV2.from_parameters),
            )
        )
    with pytest.raises(ValueError, match="registered"):
        resolution_v2.resolve_plan_v2(valid_request(null_name="unknown_null_v2"))
    with pytest.raises(ValueError, match="registered"):
        resolution_v2.resolve_plan_v2(valid_request(statistic_name="unknown_statistic_v2"))


class _WrongNameStatistic:
    bind_called = False

    @classmethod
    def from_parameters(cls, parameters: object) -> Self:
        del parameters
        return cls()

    @property
    def name(self) -> str:
        return "lying_statistic_v2"

    @property
    def parameters(self) -> Mapping[str, JsonValue]:
        return MappingProxyType({})

    def bind(self, observed_pair: SeriesPair, candidates: tuple[int, ...]) -> BoundStatisticAdapter:
        del observed_pair, candidates
        type(self).bind_called = True
        raise AssertionError("identity failure must occur before bind")


class _WrongNameNull:
    bind_called = False

    @classmethod
    def from_parameters(cls, parameters: object) -> Self:
        del parameters
        return cls()

    @property
    def name(self) -> str:
        return "lying_null_v2"

    @property
    def parameters(self) -> Mapping[str, JsonValue]:
        return MappingProxyType({"min_shift": 1})

    def bind(
        self,
        observed_pair: SeriesPair,
        *,
        semantic_input_sha256: str,
        scientific_plan_sha256: str,
    ) -> NullBindResult:
        del observed_pair, semantic_input_sha256, scientific_plan_sha256
        type(self).bind_called = True
        raise AssertionError("identity failure must occur before bind")


def test_v2_resolution_revalidates_statistic_and_null_identity_before_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _WrongNameStatistic.bind_called = False
    _WrongNameNull.bind_called = False
    statistic_registry: AdapterRegistry[StatisticAdapter] = AdapterRegistry(
        (("lagged_pearson_v1", lambda parameters: _WrongNameStatistic()),)
    )
    monkeypatch.setattr(resolution_v2, "_STATISTIC_REGISTRY_V2", statistic_registry)
    with pytest.raises(
        V2IntegrityError,
        match=r"statistic.*name|exact registered implementation",
    ):
        resolution_v2.resolve_plan_v2(valid_request())
    assert not _WrongNameStatistic.bind_called

    monkeypatch.setattr(
        resolution_v2,
        "_STATISTIC_REGISTRY_V2",
        AdapterRegistry((("lagged_pearson_v1", LaggedPearsonAdapter.from_parameters),)),
    )
    monkeypatch.setattr(
        CircularShiftNullV2,
        "name",
        property(lambda self: "lying_null_v2"),
    )
    with pytest.raises(V2IntegrityError, match=r"null.*name"):
        resolution_v2.resolve_plan_v2(valid_request())
    assert not _WrongNameNull.bind_called


def test_v2_resolution_revalidates_parameters_and_rejects_v1_null_substitution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrong_statistic: AdapterRegistry[StatisticAdapter] = AdapterRegistry(
        (("equal_width_binned_nette_v1", lambda parameters: BinnedNetTEAdapter(4)),)
    )
    monkeypatch.setattr(resolution_v2, "_STATISTIC_REGISTRY_V2", wrong_statistic)
    with pytest.raises(V2IntegrityError, match=r"statistic.*parameters"):
        resolution_v2.resolve_plan_v2(
            valid_request(
                statistic_name="equal_width_binned_nette_v1",
                statistic_params={"bins": 3},
            )
        )

    monkeypatch.setattr(
        resolution_v2,
        "_STATISTIC_REGISTRY_V2",
        AdapterRegistry((("lagged_pearson_v1", LaggedPearsonAdapter.from_parameters),)),
    )
    monkeypatch.setattr(
        resolution_v2,
        "_NULL_REGISTRY_V2",
        AdapterRegistry((("circular_shift_v2", lambda parameters: BlockShuffleContract(1)),)),
    )
    with pytest.raises(V2IntegrityError, match=r"executable.*v2|null"):
        resolution_v2.resolve_plan_v2(valid_request())


def test_resolution_rechecks_plan_candidates_and_adapter_identity_at_ownership_gate() -> None:
    legitimate = resolution_v2.resolve_plan_v2(valid_request())
    object.__setattr__(legitimate.plan, "candidates", (3, 2, 1))
    with pytest.raises(V2IntegrityError, match=r"candidates|canonical"):
        resolution_v2._require_resolver_owned_resolution_v2(legitimate)


def test_manual_replace_subclass_and_pickle_like_resolution_forgery_are_not_owned() -> None:
    legitimate = resolution_v2.resolve_plan_v2(valid_request())

    with pytest.raises(ValueError, match=r"resolver|resolution"):
        ResolvedScientificPlanV2(
            candidates=legitimate.plan.candidates,
            statistic_name=legitimate.plan.statistic_name,
            statistic_params=legitimate.plan.statistic_params,
            selection_rule=legitimate.plan.selection_rule,
            null_name=legitimate.plan.null_name,
            null_params=legitimate.plan.null_params,
            replicates=legitimate.plan.replicates,
            alpha=legitimate.plan.alpha,
            tie_tolerance=legitimate.plan.tie_tolerance,
            root_seed=legitimate.plan.root_seed,
        )
    with pytest.raises(ValueError, match=r"resolver|resolution"):
        resolution_v2.PlanResolutionV2(
            plan=legitimate.plan,
            adapters=legitimate.adapters,
        )
    with pytest.raises(ValueError, match=r"resolver|resolution"):
        replace(legitimate.plan)
    with pytest.raises(ValueError, match=r"resolver|resolution"):
        replace(legitimate)

    class PlanSubclass(ResolvedScientificPlanV2):
        pass

    class AdaptersSubclass(resolution_v2.ResolvedAdaptersV2):
        pass

    class ResolutionSubclass(resolution_v2.PlanResolutionV2):
        pass

    assert PlanSubclass is not ResolvedScientificPlanV2
    adapters_subclass = AdaptersSubclass(
        statistic=legitimate.adapters.statistic,
        null_model=legitimate.adapters.null_model,
    )
    with pytest.raises(ValueError, match=r"resolver|resolution|exact"):
        ResolutionSubclass(plan=legitimate.plan, adapters=legitimate.adapters)
    with pytest.raises(ValueError, match=r"resolver|resolution|exact"):
        resolution_v2.PlanResolutionV2(
            plan=legitimate.plan,
            adapters=adapters_subclass,
        )

    forged = object.__new__(resolution_v2.PlanResolutionV2)
    object.__setattr__(forged, "plan", legitimate.plan)
    object.__setattr__(forged, "adapters", legitimate.adapters)
    with pytest.raises(V2IntegrityError, match="resolver-owned"):
        resolution_v2._require_resolver_owned_resolution_v2(forged)
    with pytest.raises(V2IntegrityError, match="exact resolver-owned"):
        resolution_v2._require_resolver_owned_resolution_v2(object())


def test_even_private_seal_cannot_authorize_subclasses_or_invalid_carriers() -> None:
    legitimate = resolution_v2.resolve_plan_v2(valid_request())

    class PlanSubclass(ResolvedScientificPlanV2):
        pass

    class ResolutionSubclass(resolution_v2.PlanResolutionV2):
        pass

    with pytest.raises(ValueError, match="exact ResolvedScientificPlanV2"):
        PlanSubclass(
            candidates=legitimate.plan.candidates,
            statistic_name=legitimate.plan.statistic_name,
            statistic_params=legitimate.plan.statistic_params,
            selection_rule=legitimate.plan.selection_rule,
            null_name=legitimate.plan.null_name,
            null_params=legitimate.plan.null_params,
            replicates=legitimate.plan.replicates,
            alpha=legitimate.plan.alpha,
            tie_tolerance=legitimate.plan.tie_tolerance,
            root_seed=legitimate.plan.root_seed,
            seal=contracts_v2._V2_RESOLUTION_SEAL,
        )
    with pytest.raises(ValueError, match="exact PlanResolutionV2"):
        ResolutionSubclass(
            plan=legitimate.plan,
            adapters=legitimate.adapters,
            seal=contracts_v2._V2_RESOLUTION_SEAL,
        )
    with pytest.raises(V2IntegrityError, match="exact ResolvedScientificPlanV2"):
        resolution_v2.PlanResolutionV2(
            plan=cast(ResolvedScientificPlanV2, object()),
            adapters=legitimate.adapters,
            seal=contracts_v2._V2_RESOLUTION_SEAL,
        )
    with pytest.raises(V2IntegrityError, match="exact ResolvedAdaptersV2"):
        resolution_v2.PlanResolutionV2(
            plan=legitimate.plan,
            adapters=cast(resolution_v2.ResolvedAdaptersV2, object()),
            seal=contracts_v2._V2_RESOLUTION_SEAL,
        )


def test_ownership_gate_rejects_invalid_candidates_and_replaced_association() -> None:
    invalid = resolution_v2.resolve_plan_v2(valid_request())
    object.__setattr__(invalid.plan, "candidates", cast(tuple[int, ...], ("bad",)))
    with pytest.raises(V2IntegrityError, match=r"candidates|identity snapshot"):
        resolution_v2._require_resolver_owned_resolution_v2(invalid)

    replaced = resolution_v2.resolve_plan_v2(valid_request())
    object.__setattr__(
        replaced,
        "adapters",
        resolution_v2.ResolvedAdaptersV2(
            statistic=replaced.adapters.statistic,
            null_model=replaced.adapters.null_model,
        ),
    )
    with pytest.raises(V2IntegrityError, match="association was replaced"):
        resolution_v2._require_resolver_owned_resolution_v2(replaced)


def test_registry_rejects_nonprotocol_statistic_and_nonexact_executable_null(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        resolution_v2,
        "_STATISTIC_REGISTRY_V2",
        AdapterRegistry(
            (("lagged_pearson_v1", lambda parameters: cast(StatisticAdapter, object())),)
        ),
    )
    with pytest.raises(V2IntegrityError, match=r"StatisticAdapter|exact registered"):
        resolution_v2.resolve_plan_v2(valid_request())

    monkeypatch.setattr(
        resolution_v2,
        "_STATISTIC_REGISTRY_V2",
        AdapterRegistry((("lagged_pearson_v1", LaggedPearsonAdapter.from_parameters),)),
    )
    monkeypatch.setattr(
        resolution_v2,
        "_NULL_REGISTRY_V2",
        AdapterRegistry((("circular_shift_v2", lambda parameters: _WrongNameNull()),)),
    )
    with pytest.raises(V2IntegrityError, match=r"exact.*v2 null|exact registered"):
        resolution_v2.resolve_plan_v2(valid_request())


def test_invalid_adapter_parameter_snapshot_fails_as_v2_integrity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        LaggedPearsonAdapter,
        "parameters",
        property(lambda self: cast(Mapping[str, JsonValue], {"bad": object()})),
    )
    with pytest.raises(V2IntegrityError, match="exact JSON values"):
        resolution_v2.resolve_plan_v2(valid_request())


def test_v2_resolver_requires_exact_request_type() -> None:
    class RequestSubclass(PlanRequestV2):
        pass

    with pytest.raises(TypeError, match="exact PlanRequestV2"):
        resolution_v2.resolve_plan_v2(cast(PlanRequestV2, object()))
    with pytest.raises(TypeError, match="exact PlanRequestV2"):
        resolution_v2.resolve_plan_v2(
            RequestSubclass(
                candidates=(1,),
                statistic_name="lagged_pearson_v1",
                statistic_params={},
                selection_rule="max_upper",
                null_name="circular_shift_v2",
                null_params={"min_shift": 1},
                replicates=1,
                alpha=0.05,
                tie_tolerance=0.0,
                root_seed=0,
            )
        )


def test_block_v2_resolution_is_exact_and_aligned() -> None:
    result = resolution_v2.resolve_plan_v2(
        valid_request(null_name="block_shuffle_v2", null_params={"block_length": 2})
    )

    assert type(result.adapters.null_model) is BlockShuffleNullV2
    assert result.plan.null_name == "block_shuffle_v2"
    assert result.plan.null_params == {"block_length": 2}


@pytest.mark.parametrize(
    ("field_name", "mutated"),
    (
        ("replicates", 0),
        ("root_seed", 2**64),
        ("alpha", 2.0),
        ("tie_tolerance", -1.0),
        ("selection_rule", "unsupported"),
        ("candidates", (3, 2, 1)),
        ("statistic_name", "lying_statistic_v2"),
        ("statistic_params", MappingProxyType({"hidden": 1})),
        ("null_name", "lying_null_v2"),
        ("null_params", MappingProxyType({"min_shift": 2})),
    ),
)
def test_resolver_ownership_freezes_every_plan_identity_field(
    field_name: str, mutated: object
) -> None:
    result = resolution_v2.resolve_plan_v2(valid_request())
    object.__setattr__(result.plan, field_name, mutated)

    with pytest.raises(V2IntegrityError, match=r"plan|identity|canonical|registered"):
        resolution_v2._require_resolver_owned_resolution_v2(result)


@pytest.mark.parametrize("role", ("statistic", "null_model"))
def test_resolver_ownership_freezes_adapter_container_identities(role: str) -> None:
    result = resolution_v2.resolve_plan_v2(valid_request())
    replacement: object
    if role == "statistic":
        replacement = LaggedPearsonAdapter()
    else:
        replacement = CircularShiftNullV2(min_shift=1)
    object.__setattr__(result.adapters, role, replacement)

    with pytest.raises(V2IntegrityError, match=r"association|adapter|identity"):
        resolution_v2._require_resolver_owned_resolution_v2(result)


@pytest.mark.parametrize(
    ("plan_request", "operational_field", "mutated"),
    (
        (
            valid_request(
                statistic_name="equal_width_binned_nette_v1",
                statistic_params={"bins": 3},
            ),
            "bins",
            4,
        ),
        (valid_request(), "min_shift", 2),
        (
            valid_request(
                null_name="block_shuffle_v2",
                null_params={"block_length": 2},
            ),
            "block_length",
            3,
        ),
    ),
)
def test_resolver_ownership_freezes_factory_operational_scalars(
    plan_request: PlanRequestV2, operational_field: str, mutated: int
) -> None:
    result = resolution_v2.resolve_plan_v2(plan_request)
    subject = (
        result.adapters.statistic if operational_field == "bins" else result.adapters.null_model
    )
    object.__setattr__(subject, operational_field, mutated)

    with pytest.raises(V2IntegrityError, match=r"operational|identity|snapshot"):
        resolution_v2._require_resolver_owned_resolution_v2(result)


def test_parameter_free_pearson_has_an_exact_empty_operational_snapshot() -> None:
    result = resolution_v2.resolve_plan_v2(valid_request())

    assert type(result.adapters.statistic) is LaggedPearsonAdapter
    assert resolution_v2._require_resolver_owned_resolution_v2(result) is result


def test_identity_snapshot_validators_fail_closed_on_type_and_scalar_drift() -> None:
    with pytest.raises(V2IntegrityError, match="lagged_pearson_v1"):
        resolution_v2._pearson_identity_snapshot(object())

    invalid_binned = BinnedNetTEAdapter(3)
    object.__setattr__(invalid_binned, "bins", 1)
    with pytest.raises(V2IntegrityError, match="binned statistic"):
        resolution_v2._binned_identity_snapshot(invalid_binned)
    with pytest.raises(V2IntegrityError, match="equal_width_binned_nette_v1"):
        resolution_v2._binned_identity_snapshot(object())

    invalid_circular = CircularShiftNullV2(min_shift=1)
    object.__setattr__(invalid_circular, "min_shift", 0)
    with pytest.raises(V2IntegrityError, match="circular null"):
        resolution_v2._circular_identity_snapshot(invalid_circular)
    with pytest.raises(V2IntegrityError, match="circular_shift_v2"):
        resolution_v2._circular_identity_snapshot(object())

    invalid_block = BlockShuffleNullV2(block_length=2)
    object.__setattr__(invalid_block, "block_length", 0)
    with pytest.raises(V2IntegrityError, match="block null"):
        resolution_v2._block_identity_snapshot(invalid_block)
    with pytest.raises(V2IntegrityError, match="block_shuffle_v2"):
        resolution_v2._block_identity_snapshot(object())


def test_statistic_registry_rejects_protocol_liar_and_builtin_subclasses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Liar(_WrongNameStatistic):
        @property
        def name(self) -> str:
            return "lagged_pearson_v1"

        @property
        def parameters(self) -> Mapping[str, JsonValue]:
            return MappingProxyType({})

    class PearsonSubclass(LaggedPearsonAdapter):
        pass

    for factory in (
        lambda parameters: Liar(),
        lambda parameters: PearsonSubclass(),
    ):
        monkeypatch.setattr(
            resolution_v2,
            "_STATISTIC_REGISTRY_V2",
            AdapterRegistry((("lagged_pearson_v1", factory),)),
        )
        with pytest.raises(V2IntegrityError, match=r"exact|registered.*implementation"):
            resolution_v2.resolve_plan_v2(valid_request())


def test_statistic_registry_rejects_name_to_wrong_builtin_implementation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BinnedPretendingPearson(BinnedNetTEAdapter):
        @property
        def name(self) -> str:
            return "lagged_pearson_v1"

        @property
        def parameters(self) -> Mapping[str, JsonValue]:
            return MappingProxyType({})

    monkeypatch.setattr(
        resolution_v2,
        "_STATISTIC_REGISTRY_V2",
        AdapterRegistry((("lagged_pearson_v1", lambda parameters: BinnedPretendingPearson(3)),)),
    )
    with pytest.raises(V2IntegrityError, match=r"exact|registered.*implementation"):
        resolution_v2.resolve_plan_v2(valid_request())


def test_ownership_gate_normalizes_structural_and_runtime_snapshot_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid = resolution_v2.resolve_plan_v2(valid_request())
    object.__setattr__(invalid.plan, "root_seed", object())
    with pytest.raises(V2IntegrityError):
        resolution_v2._require_resolver_owned_resolution_v2(invalid)

    runtime = resolution_v2.resolve_plan_v2(valid_request())

    def infrastructure_failure(self: LaggedPearsonAdapter) -> Mapping[str, JsonValue]:
        del self
        raise RuntimeError("infrastructure failure")

    monkeypatch.setattr(
        LaggedPearsonAdapter,
        "parameters",
        property(infrastructure_failure),
    )
    with pytest.raises(
        V2IntegrityError,
        match=r"^resolution identity snapshot is invalid$",
    ) as caught:
        resolution_v2._require_resolver_owned_resolution_v2(runtime)
    assert type(caught.value.__cause__) is RuntimeError
    assert str(caught.value.__cause__) == "infrastructure failure"


@pytest.mark.parametrize(
    "plan_request",
    (
        valid_request(),
        valid_request(
            null_name="block_shuffle_v2",
            null_params={"block_length": 2},
        ),
    ),
)
def test_null_operational_snapshot_freezes_internal_parameter_digest(
    plan_request: PlanRequestV2,
) -> None:
    result = resolution_v2.resolve_plan_v2(plan_request)
    object.__setattr__(
        result.adapters.null_model,
        "_null_parameter_sha256",
        "f" * 64,
    )

    with pytest.raises(V2IntegrityError, match=r"null.*parameter|digest|identity"):
        resolution_v2._require_resolver_owned_resolution_v2(result)


@pytest.mark.parametrize(
    ("plan_request", "null_type"),
    (
        (valid_request(), CircularShiftNullV2),
        (
            valid_request(
                null_name="block_shuffle_v2",
                null_params={"block_length": 2},
            ),
            BlockShuffleNullV2,
        ),
    ),
)
def test_resolver_rejects_exact_null_factory_with_preforged_parameter_digest(
    monkeypatch: pytest.MonkeyPatch,
    plan_request: PlanRequestV2,
    null_type: type[CircularShiftNullV2] | type[BlockShuffleNullV2],
) -> None:
    def forged_factory(parameters: object) -> object:
        null_model = null_type.from_parameters(parameters)
        object.__setattr__(null_model, "_null_parameter_sha256", "f" * 64)
        return null_model

    monkeypatch.setattr(
        resolution_v2,
        "_NULL_REGISTRY_V2",
        AdapterRegistry(((plan_request.null_name, forged_factory),)),
    )

    with pytest.raises(V2IntegrityError, match=r"null.*parameter|digest|identity"):
        resolution_v2.resolve_plan_v2(plan_request)


@pytest.mark.parametrize(
    ("plan_request", "mode"),
    (
        (valid_request(), "missing"),
        (valid_request(), "str_subclass"),
        (
            valid_request(
                null_name="block_shuffle_v2",
                null_params={"block_length": 2},
            ),
            "missing",
        ),
        (
            valid_request(
                null_name="block_shuffle_v2",
                null_params={"block_length": 2},
            ),
            "str_subclass",
        ),
    ),
)
def test_null_internal_parameter_digest_shape_is_fail_closed(
    plan_request: PlanRequestV2,
    mode: str,
) -> None:
    class DigestSubclass(str):
        pass

    result = resolution_v2.resolve_plan_v2(plan_request)
    if mode == "missing":
        object.__delattr__(result.adapters.null_model, "_null_parameter_sha256")
    else:
        digest = result.adapters.null_model.null_parameter_sha256  # type: ignore[attr-defined]
        object.__setattr__(
            result.adapters.null_model,
            "_null_parameter_sha256",
            DigestSubclass(digest),
        )

    with pytest.raises(V2IntegrityError, match=r"null|digest|identity"):
        resolution_v2._require_resolver_owned_resolution_v2(result)


@pytest.mark.parametrize("failure_type", (ValueError, TypeError, AttributeError))
def test_null_internal_digest_property_failures_are_typed(
    monkeypatch: pytest.MonkeyPatch,
    failure_type: type[Exception],
) -> None:
    result = resolution_v2.resolve_plan_v2(valid_request())

    def structural_failure(self: CircularShiftNullV2) -> str:
        del self
        raise failure_type("structural digest failure")

    monkeypatch.setattr(
        CircularShiftNullV2,
        "_null_parameter_sha256",
        property(structural_failure),
    )
    with pytest.raises(V2IntegrityError, match=r"null|snapshot|identity"):
        resolution_v2._require_resolver_owned_resolution_v2(result)


def test_null_internal_digest_runtime_failure_is_typed_at_ownership_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = resolution_v2.resolve_plan_v2(valid_request())

    def infrastructure_failure(self: CircularShiftNullV2) -> str:
        del self
        raise RuntimeError("digest infrastructure failure")

    monkeypatch.setattr(
        CircularShiftNullV2,
        "_null_parameter_sha256",
        property(infrastructure_failure),
    )
    with pytest.raises(
        V2IntegrityError,
        match=r"^resolution identity snapshot is invalid$",
    ) as caught:
        resolution_v2._require_resolver_owned_resolution_v2(result)
    assert type(caught.value.__cause__) is RuntimeError
    assert str(caught.value.__cause__) == "digest infrastructure failure"


def test_null_operational_snapshot_preserves_typed_errors_and_rejects_name_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    null_model = CircularShiftNullV2(min_shift=1)

    def typed_failure(self: CircularShiftNullV2) -> Mapping[str, JsonValue]:
        del self
        raise V2IntegrityError("typed parameter failure")

    with monkeypatch.context() as patch:
        patch.setattr(
            CircularShiftNullV2,
            "parameters",
            property(typed_failure),
        )
        with pytest.raises(V2IntegrityError, match="typed parameter failure"):
            resolution_v2._circular_identity_snapshot(null_model)

    with monkeypatch.context() as patch:
        patch.setattr(
            CircularShiftNullV2,
            "name",
            property(lambda self: "lying_null_v2"),
        )
        with pytest.raises(V2IntegrityError, match="operational name"):
            resolution_v2._circular_identity_snapshot(null_model)


@pytest.mark.parametrize(
    ("plan_request", "null_type", "property_name"),
    (
        (valid_request(), CircularShiftNullV2, "name"),
        (valid_request(), CircularShiftNullV2, "parameters"),
        (
            valid_request(
                null_name="block_shuffle_v2",
                null_params={"block_length": 2},
            ),
            BlockShuffleNullV2,
            "name",
        ),
        (
            valid_request(
                null_name="block_shuffle_v2",
                null_params={"block_length": 2},
            ),
            BlockShuffleNullV2,
            "parameters",
        ),
    ),
)
def test_resolver_types_structural_public_null_property_failures(
    monkeypatch: pytest.MonkeyPatch,
    plan_request: PlanRequestV2,
    null_type: type[CircularShiftNullV2] | type[BlockShuffleNullV2],
    property_name: str,
) -> None:
    null_model = null_type.from_parameters(plan_request.null_params)

    def structural_failure(self: object) -> object:
        del self
        raise ValueError("public null identity failure")

    monkeypatch.setattr(
        null_type,
        property_name,
        property(structural_failure),
    )
    monkeypatch.setattr(
        resolution_v2,
        "_NULL_REGISTRY_V2",
        AdapterRegistry(((plan_request.null_name, lambda parameters: null_model),)),
    )

    with pytest.raises(V2IntegrityError, match=r"null|identity|snapshot"):
        resolution_v2.resolve_plan_v2(plan_request)


def test_resolver_does_not_swallow_public_null_property_runtime_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_request = valid_request()
    null_model = CircularShiftNullV2.from_parameters(plan_request.null_params)

    def infrastructure_failure(self: CircularShiftNullV2) -> object:
        del self
        raise RuntimeError("public null infrastructure failure")

    monkeypatch.setattr(
        CircularShiftNullV2,
        "parameters",
        property(infrastructure_failure),
    )
    monkeypatch.setattr(
        resolution_v2,
        "_NULL_REGISTRY_V2",
        AdapterRegistry(((plan_request.null_name, lambda parameters: null_model),)),
    )

    with pytest.raises(RuntimeError, match="public null infrastructure failure"):
        resolution_v2.resolve_plan_v2(plan_request)


@pytest.mark.parametrize(
    ("plan_request", "null_model", "parameter_name"),
    (
        (
            valid_request(null_params={"min_shift": 2}),
            CircularShiftNullV2(min_shift=1),
            "min_shift",
        ),
        (
            valid_request(
                null_name="block_shuffle_v2",
                null_params={"block_length": 2},
            ),
            BlockShuffleNullV2(block_length=1),
            "block_length",
        ),
    ),
)
def test_resolver_rejects_stateful_null_parameters_from_one_safe_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    plan_request: PlanRequestV2,
    null_model: CircularShiftNullV2 | BlockShuffleNullV2,
    parameter_name: str,
) -> None:
    reads = 0

    def alternating_parameters(
        self: CircularShiftNullV2 | BlockShuffleNullV2,
    ) -> Mapping[str, JsonValue]:
        del self
        nonlocal reads
        reads += 1
        reported_value = 2 if reads % 2 == 1 else 1
        return MappingProxyType({parameter_name: reported_value})

    monkeypatch.setattr(
        type(null_model),
        "parameters",
        property(alternating_parameters),
    )
    monkeypatch.setattr(
        resolution_v2,
        "_NULL_REGISTRY_V2",
        AdapterRegistry(((plan_request.null_name, lambda _: null_model),)),
    )

    with pytest.raises(V2IntegrityError, match=r"null|parameter|identity|snapshot"):
        resolution_v2.resolve_plan_v2(plan_request)


def test_resolver_rejects_exact_statistic_with_drifting_public_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        LaggedPearsonAdapter,
        "name",
        property(lambda _: "lying_statistic_v2"),
    )

    with pytest.raises(V2IntegrityError, match=r"statistic.*name"):
        resolution_v2.resolve_plan_v2(valid_request())


def test_resolver_fails_closed_for_null_registration_without_snapshot_validator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registration = resolution_v2._NULL_REGISTRATIONS_V2["circular_shift_v2"]
    monkeypatch.setattr(
        resolution_v2,
        "_NULL_REGISTRATIONS_V2",
        MappingProxyType(
            {
                **resolution_v2._NULL_REGISTRATIONS_V2,
                "circular_shift_v2": replace(
                    registration,
                    null_identity_snapshot=None,
                ),
            }
        ),
    )

    with pytest.raises(V2IntegrityError, match=r"null.*snapshot validator"):
        resolution_v2.resolve_plan_v2(valid_request())


def test_resolver_rejects_exact_binned_factory_with_preforged_operational_bins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_request = valid_request(
        statistic_name="equal_width_binned_nette_v1",
        statistic_params={"bins": 3},
    )

    def forged_factory(parameters: object) -> object:
        statistic = BinnedNetTEAdapter.from_parameters(parameters)
        object.__setattr__(statistic, "bins", 4)
        return statistic

    monkeypatch.setattr(
        resolution_v2,
        "_STATISTIC_REGISTRY_V2",
        AdapterRegistry(((plan_request.statistic_name, forged_factory),)),
    )

    with pytest.raises(V2IntegrityError, match=r"statistic|parameter|bins|identity"):
        resolution_v2.resolve_plan_v2(plan_request)


@pytest.mark.parametrize("property_name", ("name", "parameters"))
@pytest.mark.parametrize("failure_type", (AttributeError, TypeError, ValueError))
def test_resolver_types_initial_structural_statistic_property_failures(
    monkeypatch: pytest.MonkeyPatch,
    property_name: str,
    failure_type: type[Exception],
) -> None:
    def structural_failure(self: LaggedPearsonAdapter) -> object:
        del self
        raise failure_type("statistic identity failure")

    monkeypatch.setattr(
        LaggedPearsonAdapter,
        property_name,
        property(structural_failure),
    )

    with pytest.raises(V2IntegrityError, match=r"statistic|identity|snapshot"):
        resolution_v2.resolve_plan_v2(valid_request())


@pytest.mark.parametrize("property_name", ("name", "parameters"))
def test_resolver_does_not_swallow_statistic_property_runtime_failures(
    monkeypatch: pytest.MonkeyPatch,
    property_name: str,
) -> None:
    def infrastructure_failure(self: LaggedPearsonAdapter) -> object:
        del self
        raise RuntimeError("statistic infrastructure failure")

    monkeypatch.setattr(
        LaggedPearsonAdapter,
        property_name,
        property(infrastructure_failure),
    )

    with pytest.raises(RuntimeError, match="statistic infrastructure failure"):
        resolution_v2.resolve_plan_v2(valid_request())


def test_resolver_rejects_stateful_statistic_parameters_in_ownership_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reads = 0

    def stateful_parameters(
        self: LaggedPearsonAdapter,
    ) -> Mapping[str, JsonValue]:
        del self
        nonlocal reads
        reads += 1
        if reads >= 4 and reads % 2 == 0:
            return MappingProxyType({"hidden": 1})
        return MappingProxyType({})

    monkeypatch.setattr(
        LaggedPearsonAdapter,
        "parameters",
        property(stateful_parameters),
    )

    with pytest.raises(V2IntegrityError, match=r"statistic|parameter|identity|snapshot"):
        resolution_v2.resolve_plan_v2(valid_request())


def test_statistic_snapshot_preserves_typed_integrity_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def typed_failure(self: LaggedPearsonAdapter) -> object:
        del self
        raise V2IntegrityError("typed statistic failure")

    monkeypatch.setattr(
        LaggedPearsonAdapter,
        "parameters",
        property(typed_failure),
    )

    with pytest.raises(V2IntegrityError, match="typed statistic failure"):
        resolution_v2.resolve_plan_v2(valid_request())


def test_resolver_fails_closed_for_statistic_registration_without_snapshot_validator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registration = resolution_v2._STATISTIC_REGISTRATIONS_V2["lagged_pearson_v1"]
    monkeypatch.setattr(
        resolution_v2,
        "_STATISTIC_REGISTRATIONS_V2",
        MappingProxyType(
            {
                **resolution_v2._STATISTIC_REGISTRATIONS_V2,
                "lagged_pearson_v1": replace(
                    registration,
                    statistic_identity_snapshot=None,
                ),
            }
        ),
    )

    with pytest.raises(V2IntegrityError, match=r"statistic.*snapshot validator"):
        resolution_v2.resolve_plan_v2(valid_request())


def test_identity_comparison_rejects_wrong_exact_name() -> None:
    with pytest.raises(V2IntegrityError, match=r"statistic.*name"):
        resolution_v2._verify_identity(
            "lying_statistic_v2",
            MappingProxyType({}),
            expected_name="lagged_pearson_v1",
            expected_params=MappingProxyType({}),
            role="statistic",
        )


def test_ownership_rejects_valid_but_unregistered_plan_drift() -> None:
    result = resolution_v2.resolve_plan_v2(valid_request())
    object.__setattr__(result.plan, "replicates", 10)

    with pytest.raises(V2IntegrityError, match=r"identity snapshot drifted"):
        resolution_v2._require_resolver_owned_resolution_v2(result)
