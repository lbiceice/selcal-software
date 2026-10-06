from __future__ import annotations

from collections.abc import Iterator, Mapping
from types import MappingProxyType

import numpy as np
import pytest

from selcal.canonical import canonical_json_bytes
from selcal.nulls.block_shuffle import BlockShuffleContract
from selcal.nulls.circular_shift import CircularShiftContract
from selcal.parameters import freeze_exact_json_mapping
from selcal.statistics.binned_nette import BinnedNetTEAdapter


class _HidingExtraDict(dict[str, object]):
    def items(self) -> list[tuple[str, object]]:
        return [("bins", 3)]


class _DuplicateKeyMapping(Mapping[str, object]):
    def __getitem__(self, key: str) -> object:
        return 3

    def __iter__(self) -> Iterator[str]:
        return iter(("bins", "bins"))

    def __len__(self) -> int:
        return 2


@pytest.mark.parametrize(
    ("factory", "valid", "expected_name", "expected_params"),
    [
        (
            BinnedNetTEAdapter.from_parameters,
            {"bins": 3},
            "equal_width_binned_nette_v1",
            {"bins": 3},
        ),
        (
            CircularShiftContract.from_parameters,
            {"min_shift": 1},
            "circular_shift_v1",
            {"min_shift": 1},
        ),
        (
            BlockShuffleContract.from_parameters,
            {"block_length": 2},
            "block_shuffle_v1",
            {"block_length": 2},
        ),
    ],
)
def test_factories_return_canonical_immutable_identity(
    factory: object,
    valid: dict[str, int],
    expected_name: str,
    expected_params: dict[str, int],
) -> None:
    built = factory(valid)  # type: ignore[operator]
    valid[next(iter(valid))] = 99
    assert built.name == expected_name
    assert built.parameters == expected_params
    assert isinstance(built.parameters, MappingProxyType)
    with pytest.raises(TypeError):
        built.parameters[next(iter(expected_params))] = 8  # type: ignore[index]


@pytest.mark.parametrize(
    ("factory", "bad"),
    [
        (BinnedNetTEAdapter.from_parameters, {}),
        (BinnedNetTEAdapter.from_parameters, {"bins": 3, "extra": 1}),
        (BinnedNetTEAdapter.from_parameters, {"bins": True}),
        (BinnedNetTEAdapter.from_parameters, {"bins": np.int64(3)}),
        (CircularShiftContract.from_parameters, {"min_shift": 0}),
        (CircularShiftContract.from_parameters, {"min_shift": 1.0}),
        (BlockShuffleContract.from_parameters, {"block_length": -1}),
        (BlockShuffleContract.from_parameters, {"block_length": "2"}),
    ],
)
def test_factories_reject_missing_extra_nonexact_or_out_of_range_values(
    factory: object,
    bad: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match=r"parameter|bins|min_shift|block_length"):
        factory(bad)  # type: ignore[operator]


def test_null_contracts_freeze_scientific_semantics() -> None:
    circular = CircularShiftContract.from_parameters({"min_shift": 1})
    block = BlockShuffleContract.from_parameters({"block_length": 2})

    assert circular.transformed_role == block.transformed_role == "source"
    assert circular.preserved_quantities == (
        "source_value_multiset_v1",
        "source_circular_autocorrelation_v1",
    )
    assert circular.broken_relations == ("source_target_alignment_v1",)
    assert circular.applicability_assumptions == (
        "cross_series_null_by_shift_exchangeability_v1",
    )
    assert circular.disabled_conditions == (
        "series_length_not_greater_than_min_shift_v1",
    )
    assert block.preserved_quantities == (
        "source_value_multiset_v1",
        "source_within_block_order_v1",
    )
    assert block.broken_relations == (
        "source_target_alignment_v1",
        "source_between_block_order_v1",
    )
    assert block.applicability_assumptions == (
        "local_dependence_captured_within_blocks_v1",
    )
    assert block.disabled_conditions == ("fewer_than_two_blocks_v1",)


@pytest.mark.parametrize(
    "parameters",
    [
        _HidingExtraDict({"bins": 3, "extra": 1}),
        _DuplicateKeyMapping(),
    ],
)
def test_freezer_rejects_nonexact_mappings_that_can_hide_or_duplicate_keys(
    parameters: Mapping[str, object],
) -> None:
    with pytest.raises(ValueError, match=r"exact built-in dict"):
        freeze_exact_json_mapping(parameters, name="test parameters")


def test_freezer_accepts_and_copies_an_exact_plain_mapping_proxy() -> None:
    source = {"bins": 3}
    frozen = freeze_exact_json_mapping(MappingProxyType(source), name="test parameters")
    source["bins"] = 99

    assert frozen == {"bins": 3}
    assert isinstance(frozen, MappingProxyType)


@pytest.mark.parametrize("make_cyclic", [lambda: {"self": None}, lambda: []])
def test_freezer_rejects_cycles_as_value_errors(make_cyclic: object) -> None:
    cyclic = make_cyclic()  # type: ignore[operator]
    if type(cyclic) is dict:
        cyclic["self"] = cyclic
    else:
        cyclic.append(cyclic)

    with pytest.raises(ValueError, match="cyclic"):
        freeze_exact_json_mapping({"nested": cyclic}, name="test parameters")


def test_freezer_rejects_excessive_nesting_as_a_value_error() -> None:
    nested: object = 0
    for _ in range(65):
        nested = [nested]

    with pytest.raises(ValueError, match="nesting depth"):
        freeze_exact_json_mapping({"nested": nested}, name="test parameters")


def test_freezer_copies_nested_values_and_canonicalizes_frozen_tuple_arrays() -> None:
    source = {"nested": {"values": [-0.0, 1.5]}}
    frozen = freeze_exact_json_mapping(source, name="test parameters")
    source["nested"]["values"][0] = 99.0

    assert canonical_json_bytes(frozen) == (
        b'{"nested":{"values":[{"$float64":"0x0.0p+0"},'
        b'{"$float64":"0x1.8000000000000p+0"}]}}'
    )


def test_freezer_rejects_the_reserved_canonical_float_key() -> None:
    with pytest.raises(ValueError, match=r"\$float64 is reserved"):
        freeze_exact_json_mapping({"$float64": 1}, name="test parameters")
