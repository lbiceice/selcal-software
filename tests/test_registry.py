from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Protocol, assert_type, cast

import pytest

from selcal.contracts import JsonValue, PlanRequest
from selcal.contracts_v2 import PlanRequestV2
from selcal.nulls.block_shuffle import BlockShuffleContract
from selcal.registry import AdapterRegistry, Factory


@dataclass(frozen=True)
class Built:
    name: str


class BuiltShape(Protocol):
    """A protocol used to check widened registry result types."""

    @property
    def name(self) -> str: ...


def _factory(name: str) -> Factory[Built]:
    def build(parameters: Mapping[str, JsonValue]) -> Built:
        assert parameters == {}
        return Built(name)

    return build


def _v1_request(name: object) -> PlanRequest:
    return PlanRequest(
        candidates=(1,),
        statistic_name=name,  # type: ignore[arg-type]
        statistic_params={},
        selection_rule="max_upper",
        null_name="circular_shift_v1",
        null_params={"min_shift": 1},
        replicates=1,
        alpha=0.05,
        tie_tolerance=0.0,
        root_seed=0,
        failure_policy="fail_closed_v1",
    )


def _v2_request(name: object) -> PlanRequestV2:
    return PlanRequestV2(
        candidates=(1,),
        statistic_name=name,  # type: ignore[arg-type]
        statistic_params={},
        selection_rule="max_upper",
        null_name="circular_shift_v2",
        null_params={"min_shift": 1},
        replicates=1,
        alpha=0.05,
        tie_tolerance=0.0,
        root_seed=0,
    )


@pytest.mark.parametrize("name", ["alpha_v1", "adapter_2_v12", "a_v999"])
def test_request_versions_and_registry_accept_the_same_name_language(name: str) -> None:
    registry = AdapterRegistry(((name, _factory(name)),))

    assert _v1_request(name).statistic_name == name
    assert _v2_request(name).statistic_name == name
    assert registry.names == (name,)


@pytest.mark.parametrize(
    "name",
    [
        "alpha",
        "alpha_v0",
        "alpha_v01",
        "Alpha_v1",
        "alpha-v1",
        "_alpha_v1",
        " alpha_v1",
        "alpha_v1 ",
        "",
        True,
        1,
        None,
    ],
)
def test_request_versions_and_registry_reject_the_same_name_language(
    name: object,
) -> None:
    with pytest.raises(ValueError, match=r"name|string"):
        _v1_request(name)
    with pytest.raises(ValueError, match=r"name|string"):
        _v2_request(name)
    with pytest.raises(ValueError, match=r"name|string"):
        AdapterRegistry(((name, _factory("invalid")),))  # type: ignore[arg-type]


def test_registry_type_inference_and_protocol_widening() -> None:
    inferred = AdapterRegistry((("alpha_v1", _factory("alpha")),))
    assert_type(inferred, AdapterRegistry[Built])
    assert_type(inferred.build("alpha_v1", {}), Built)

    widened: AdapterRegistry[BuiltShape] = inferred
    assert_type(widened.build("alpha_v1", {}), BuiltShape)


def test_registry_names_are_sorted_and_instances_are_immutable() -> None:
    beta = _factory("beta")
    alpha = _factory("alpha")
    first = AdapterRegistry((("beta_v1", beta), ("alpha_v1", alpha)))
    second = AdapterRegistry((("alpha_v1", alpha), ("beta_v1", beta)))

    assert first.names == ("alpha_v1", "beta_v1")
    assert second.names == ("alpha_v1", "beta_v1")
    assert first.build("alpha_v1", {}) == second.build("alpha_v1", {}) == Built("alpha")
    with pytest.raises(AttributeError):
        first._factories = {}  # type: ignore[misc]


def test_registry_rejects_duplicate_names_before_overwrite() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        AdapterRegistry((("alpha_v1", _factory("first")), ("alpha_v1", _factory("second"))))


@pytest.mark.parametrize(
    "name",
    ("unknown_v1", "Alpha_v1", "alpha", "", " alpha_v1"),
)
def test_registry_build_fails_closed_for_unknown_or_invalid_names(name: str) -> None:
    registry = AdapterRegistry((("alpha_v1", _factory("alpha")),))

    with pytest.raises(ValueError, match=r"registered|name"):
        registry.build(name, {})


def test_registry_rejects_string_subclass_names() -> None:
    class StringSubclass(str):
        pass

    entry_name = cast(str, StringSubclass("alpha_v1"))
    with pytest.raises(ValueError, match="built-in string"):
        AdapterRegistry(((entry_name, _factory("alpha")),))


def test_registry_rejects_noncallable_factories() -> None:
    with pytest.raises(ValueError, match="callable"):
        AdapterRegistry((("alpha_v1", cast(Factory[Built], "not a factory")),))


def test_registry_owns_a_single_unpacked_inner_entry() -> None:
    factory = _factory("alpha")

    class OneShotEntry:
        def __init__(self) -> None:
            self.unpacks = 0

        def __iter__(self) -> Iterator[object]:
            self.unpacks += 1
            if self.unpacks > 1:
                raise AssertionError("entry unpacked more than once")
            yield "alpha_v1"
            yield factory

    entry = OneShotEntry()
    registry = AdapterRegistry((cast(tuple[str, Factory[Built]], entry),))

    assert entry.unpacks == 1
    assert registry._factories["alpha_v1"] is factory
    assert registry.build("alpha_v1", {}) == Built("alpha")


def test_registry_owns_factory_after_source_entry_mutation() -> None:
    source: list[object] = ["alpha_v1", _factory("original")]
    registry = AdapterRegistry((cast(tuple[str, Factory[Built]], source),))
    source[1] = _factory("replacement")

    assert registry.build("alpha_v1", {}) == Built("original")


def test_registry_rejects_malformed_entries_as_value_errors() -> None:
    malformed_entries = cast(
        Iterable[tuple[str, Factory[Built]]],
        (("alpha_v1", _factory("alpha"), _factory("extra")),),
    )

    with pytest.raises(ValueError, match="entry"):
        AdapterRegistry(malformed_entries)


def test_invalid_first_entry_stops_before_later_outer_source_error() -> None:
    def entries() -> Iterator[tuple[str, Factory[Built]]]:
        yield "Alpha_v1", _factory("alpha")
        raise RuntimeError("the registry must not consume this entry")

    with pytest.raises(ValueError, match="name"):
        AdapterRegistry(entries())


def test_registry_rejects_tables_larger_than_the_documented_ceiling() -> None:
    entries = (
        (f"adapter_{index}_v1", _factory(str(index)))
        for index in range(1025)
    )

    with pytest.raises(ValueError, match="at most 1024"):
        AdapterRegistry(entries)


def test_unknown_name_never_calls_a_registered_factory() -> None:
    calls = 0

    def factory(parameters: Mapping[str, JsonValue]) -> Built:
        nonlocal calls
        calls += 1
        assert parameters == {}
        return Built("alpha")

    registry = AdapterRegistry((("alpha_v1", factory),))

    with pytest.raises(ValueError, match="registered"):
        registry.build("unknown_v1", {})
    assert calls == 0


def test_factory_exceptions_pass_through_unchanged() -> None:
    class FactoryFailure(RuntimeError):
        pass

    def factory(parameters: Mapping[str, JsonValue]) -> Built:
        assert parameters == {}
        raise FactoryFailure("factory failure")

    registry = AdapterRegistry((("alpha_v1", factory),))

    with pytest.raises(FactoryFailure, match="factory failure"):
        registry.build("alpha_v1", {})


def test_registry_integrates_an_existing_object_parameter_factory() -> None:
    registry = AdapterRegistry((("block_shuffle_v1", BlockShuffleContract.from_parameters),))
    assert_type(registry, AdapterRegistry[BlockShuffleContract])
    built = registry.build("block_shuffle_v1", {"block_length": 2})
    assert_type(built, BlockShuffleContract)

    assert built == BlockShuffleContract(2)


def test_registry_surface_never_calls_hostile_factory_dunders() -> None:
    class HostileFactory:
        def __call__(self, parameters: Mapping[str, JsonValue]) -> Built:
            assert parameters == {}
            return Built("alpha")

        def __repr__(self) -> str:
            raise AssertionError("factory repr must not run")

        def __eq__(self, other: object) -> bool:
            del other
            raise AssertionError("factory equality must not run")

        def __hash__(self) -> int:
            raise AssertionError("factory hashing must not run")

    factory = HostileFactory()
    first = AdapterRegistry((("alpha_v1", factory),))
    second = AdapterRegistry((("alpha_v1", factory),))

    assert repr(first) == "AdapterRegistry(names=('alpha_v1',))"
    assert first == first
    assert first != second
    assert hash(first) == hash(first)
