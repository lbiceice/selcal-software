from __future__ import annotations

import hashlib
import re
from dataclasses import fields
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from selcal.canonical import (
    canonical_json_bytes,
    implementation_sha256,
    raw_input_sha256,
    run_id,
    scientific_plan_sha256,
    semantic_input_sha256,
)
from selcal.contracts import PlanRequest, ResolvedScientificPlan, SeriesPair
from selcal.resolution import resolve_plan


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


def valid_plan(**changes: object) -> ResolvedScientificPlan:
    return resolve_plan(valid_request(**changes)).plan


def test_canonical_json_sorts_mapping_keys_without_whitespace() -> None:
    assert canonical_json_bytes({"b": 1, "a": 2}) == b'{"a":2,"b":1}'


def test_canonical_json_encodes_float64_by_exact_hex_value() -> None:
    assert canonical_json_bytes({"x": 0.05}) == (
        b'{"x":{"$float64":"0x1.999999999999ap-5"}}'
    )


def test_canonical_json_reserves_internal_float_tag_from_user_mappings() -> None:
    float_encoding = canonical_json_bytes(0.05)
    assert float_encoding == b'{"$float64":"0x1.999999999999ap-5"}'

    with pytest.raises(ValueError, match="reserved"):
        canonical_json_bytes({"$float64": "0x1.999999999999ap-5"})


def test_canonical_json_reserves_internal_float_tag_at_any_mapping_depth() -> None:
    with pytest.raises(ValueError, match="reserved"):
        canonical_json_bytes({"outer": [{"$float64": "user value"}]})


def test_canonical_json_preserves_recursive_json_structure() -> None:
    value = MappingProxyType(
        {
            "text": "heritage",
            "flag": True,
            "nothing": None,
            "integer": -7,
            "items": [False, 0, {"nested": "value"}],
        }
    )

    assert canonical_json_bytes(value) == (
        b'{"flag":true,"integer":-7,"items":[false,0,{"nested":"value"}],'
        b'"nothing":null,"text":"heritage"}'
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_canonical_json_rejects_non_finite_floats(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        canonical_json_bytes({"value": value})


@pytest.mark.parametrize(
    "value",
    [
        {1: "not a string key"},
        Path("input.csv"),
        np.int64(1),
        np.float64(1.0),
        np.bool_(True),
        {1, 2},
        b"bytes",
        bytearray(b"bytes"),
        object(),
    ],
)
def test_canonical_json_rejects_unsupported_values_instead_of_stringifying(
    value: object,
) -> None:
    with pytest.raises(ValueError, match=r"mapping keys|unsupported JSON value type"):
        canonical_json_bytes(value)  # type: ignore[arg-type]


def test_canonical_json_mapping_order_is_invariant_and_list_order_is_significant() -> None:
    assert canonical_json_bytes({"outer": {"b": 2, "a": 1}}) == canonical_json_bytes(
        {"outer": {"a": 1, "b": 2}}
    )
    assert canonical_json_bytes([1, 2]) != canonical_json_bytes([2, 1])


def test_canonical_json_preserves_boolean_and_signed_zero_identities() -> None:
    assert canonical_json_bytes(True) != canonical_json_bytes(1)
    assert canonical_json_bytes(-0.0) != canonical_json_bytes(+0.0)
    assert canonical_json_bytes(-0.0) == b'{"$float64":"-0x0.0p+0"}'
    assert canonical_json_bytes(+0.0) == b'{"$float64":"0x0.0p+0"}'


def test_scientific_plan_hash_uses_the_exact_frozen_payload() -> None:
    plan = valid_plan()
    expected_payload = {
        "schema": "selcal.scientific-plan.v1",
        "candidates": [1, 2, 3],
        "statistic": {"name": "equal_width_binned_nette_v1", "params": {"bins": 3}},
        "selection": {"rule": "max_upper", "tie_tolerance": 1e-12},
        "null": {"name": "circular_shift_v1", "params": {"min_shift": 1}},
        "replicates": 9,
        "alpha": 0.05,
        "root_seed": 17,
        "failure_policy": "fail_closed_v1",
        "rng_contract": "sha256-to-pcg64-v1",
        "common_support": "max_candidate_lag_v1",
    }

    assert scientific_plan_sha256(plan) == hashlib.sha256(
        canonical_json_bytes(expected_payload)
    ).hexdigest()


def test_scientific_plan_hash_is_invariant_to_candidate_and_parameter_mapping_order() -> None:
    first = valid_plan(
        candidates=(3, 1, 2),
        statistic_params={"bins": 3},
        null_params=MappingProxyType({"min_shift": 1}),
    )
    second = valid_plan(
        candidates=(2, 3, 1),
        statistic_params=MappingProxyType({"bins": 3}),
        null_params={"min_shift": 1},
    )

    assert scientific_plan_sha256(first) == scientific_plan_sha256(second)


def test_scientific_hash_rejects_requests_subclasses_and_unrelated_objects() -> None:
    class ResolvedSubclass(ResolvedScientificPlan):
        pass

    request = valid_request()
    forged = object.__new__(ResolvedSubclass)
    for value in (request, forged, object()):
        with pytest.raises(TypeError, match="ResolvedScientificPlan"):
            scientific_plan_sha256(value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "changes",
    [
        {"tie_tolerance": 1e-9},
        {"null_params": {"min_shift": 2}},
        {"replicates": 10},
        {"alpha": 0.1},
        {"statistic_params": {"bins": 4}},
        {"root_seed": 18},
        {"null_name": "block_shuffle_v1", "null_params": {"block_length": 2}},
        {"selection_rule": "max_absolute"},
    ],
)
def test_scientific_plan_hash_changes_with_each_valid_scientific_choice(
    changes: dict[str, object],
) -> None:
    assert scientific_plan_sha256(valid_plan(**changes)) != scientific_plan_sha256(valid_plan())


@pytest.mark.parametrize("setting", ["worker_count", "display_path", "ui_settings"])
def test_scientific_plan_cannot_accept_operational_or_ui_settings(setting: str) -> None:
    assert setting not in {field.name for field in fields(PlanRequest)}
    assert setting not in {field.name for field in fields(ResolvedScientificPlan)}
    with pytest.raises(TypeError):
        valid_request(**{setting: object()})


def test_semantic_hash_tracks_roles_candidates_and_not_candidate_order() -> None:
    pair = SeriesPair(
        source=np.array([1.0, 2.0, 3.0]),
        target=np.array([4.0, 5.0, 6.0]),
    )
    swapped = SeriesPair(source=pair.target, target=pair.source)

    baseline = semantic_input_sha256(pair, (3, 1, 2))
    assert baseline == semantic_input_sha256(pair, (2, 3, 1))
    assert baseline != semantic_input_sha256(swapped, (1, 2, 3))
    assert baseline != semantic_input_sha256(pair, (1, 2, 4))


def test_semantic_hash_normalizes_dtype_and_endianness_representations() -> None:
    native = SeriesPair(
        source=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        target=np.array([4, 5, 6], dtype=np.int32),
    )
    big_endian = SeriesPair(
        source=np.array([1.0, 2.0, 3.0], dtype=">f8"),
        target=np.array([4.0, 5.0, 6.0], dtype=">f8"),
    )

    assert semantic_input_sha256(native, (1, 2)) == semantic_input_sha256(
        big_endian, (2, 1)
    )


def test_semantic_hash_normalizes_signed_zero_at_series_contract_entry() -> None:
    negative_zero = SeriesPair(
        source=np.array([-0.0, 1.0]),
        target=np.array([2.0, -0.0]),
    )
    positive_zero = SeriesPair(
        source=np.array([+0.0, 1.0]),
        target=np.array([2.0, +0.0]),
    )

    assert negative_zero == positive_zero
    assert not np.signbit(negative_zero.source[0])
    assert not np.signbit(negative_zero.target[1])
    assert semantic_input_sha256(negative_zero, (1, 2)) == semantic_input_sha256(
        positive_zero, (2, 1)
    )


def test_plan_hash_normalizes_signed_zero_at_scientific_contract_entry() -> None:
    negative_zero = valid_plan(tie_tolerance=-0.0)
    positive_zero = valid_plan(tie_tolerance=+0.0)

    assert negative_zero == positive_zero
    assert not np.signbit(negative_zero.tie_tolerance)
    assert scientific_plan_sha256(negative_zero) == scientific_plan_sha256(positive_zero)


def test_raw_hash_tracks_original_bytes_independently_of_semantic_values() -> None:
    compact = b"1,2,3\n4,5,6\n"
    spaced = b"1, 2, 3\n4, 5, 6\n"
    compact_pair = SeriesPair(
        source=np.array([1.0, 2.0, 3.0]),
        target=np.array([4.0, 5.0, 6.0]),
    )
    spaced_pair = SeriesPair(
        source=np.array([1, 2, 3]),
        target=np.array([4, 5, 6]),
    )

    assert raw_input_sha256(compact) != raw_input_sha256(spaced)
    assert semantic_input_sha256(compact_pair, (1, 2)) == semantic_input_sha256(
        spaced_pair, (2, 1)
    )


@pytest.mark.parametrize("payload", [bytearray(b"data"), "data"])
def test_raw_hash_requires_actual_bytes(payload: object) -> None:
    with pytest.raises(TypeError, match="bytes"):
        raw_input_sha256(payload)  # type: ignore[arg-type]


def test_implementation_hash_uses_exact_payload_and_each_component() -> None:
    components = ("0.1.0.dev0", "revision-1", "numpy-2.3.2")
    expected_payload = {
        "schema": "selcal.implementation.v1",
        "package_version": components[0],
        "source_revision": components[1],
        "backend_identity": components[2],
    }
    baseline = implementation_sha256(*components)

    assert baseline == hashlib.sha256(canonical_json_bytes(expected_payload)).hexdigest()
    assert baseline != implementation_sha256("0.1.0.dev1", components[1], components[2])
    assert baseline != implementation_sha256(components[0], "revision-2", components[2])
    assert baseline != implementation_sha256(components[0], components[1], "numpy-2.3.3")


def test_run_id_uses_exact_payload_and_changes_with_each_hash() -> None:
    hashes = ("a" * 64, "b" * 64, "c" * 64)
    expected_payload = {
        "semantic_input_sha256": hashes[0],
        "scientific_plan_sha256": hashes[1],
        "implementation_sha256": hashes[2],
    }
    baseline = run_id(*hashes)

    assert baseline == hashlib.sha256(canonical_json_bytes(expected_payload)).hexdigest()
    assert baseline != run_id("d" * 64, hashes[1], hashes[2])
    assert baseline != run_id(hashes[0], "d" * 64, hashes[2])
    assert baseline != run_id(hashes[0], hashes[1], "d" * 64)


@pytest.mark.parametrize("component_index", [0, 1, 2])
@pytest.mark.parametrize(
    "invalid_hash",
    ["", "A" * 64, "g" * 64, "a" * 63, "a" * 65],
)
def test_run_id_rejects_noncanonical_sha256_components(
    component_index: int,
    invalid_hash: str,
) -> None:
    components = ["a" * 64, "b" * 64, "c" * 64]
    components[component_index] = invalid_hash

    with pytest.raises(ValueError, match="lowercase"):
        run_id(*components)


def test_run_id_rejects_string_subclasses() -> None:
    class StringSubclass(str):
        pass

    with pytest.raises(TypeError, match="string"):
        run_id(StringSubclass("a" * 64), "b" * 64, "c" * 64)


@pytest.mark.parametrize("component_index", [0, 1, 2])
@pytest.mark.parametrize("invalid_identity", ["", "   ", " value", "value ", "\tvalue"])
def test_implementation_hash_rejects_empty_or_boundary_whitespace_components(
    component_index: int,
    invalid_identity: str,
) -> None:
    components = ["version", "source revision", "backend identity"]
    components[component_index] = invalid_identity

    with pytest.raises(ValueError, match=r"nonempty|whitespace"):
        implementation_sha256(*components)


def test_implementation_hash_preserves_interior_text_and_rejects_string_subclasses() -> None:
    class StringSubclass(str):
        pass

    identity = implementation_sha256("version β", "source revision", "backend identity")
    assert re.fullmatch(r"[0-9a-f]{64}", identity)
    assert identity != implementation_sha256("versionβ", "source revision", "backend identity")

    with pytest.raises(TypeError, match="string"):
        implementation_sha256(StringSubclass("version"), "revision", "backend")


@pytest.mark.parametrize(
    ("function", "arguments"),
    [
        (implementation_sha256, (1, "revision", "backend")),
        (implementation_sha256, ("version", b"revision", "backend")),
        (run_id, (b"a", "b", "c")),
    ],
)
def test_string_hash_helpers_require_actual_strings(
    function: object,
    arguments: tuple[object, ...],
) -> None:
    with pytest.raises(TypeError, match="string"):
        function(*arguments)  # type: ignore[operator]


def test_all_identity_hashes_are_deterministic_lowercase_sha256() -> None:
    pair = SeriesPair(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    identity_functions = [
        lambda: raw_input_sha256(b"payload"),
        lambda: semantic_input_sha256(pair, (2, 1)),
        lambda: scientific_plan_sha256(valid_plan()),
        lambda: implementation_sha256("version", "revision", "backend"),
        lambda: run_id("a" * 64, "b" * 64, "c" * 64),
    ]

    for identity_function in identity_functions:
        first = identity_function()
        assert re.fullmatch(r"[0-9a-f]{64}", first)
        assert identity_function() == first


def test_known_answer_smallest_subnormal_float_encoding() -> None:
    literal_preimage = b'{"tiny":{"$float64":"0x0.0000000000001p-1022"}}'
    expected_digest = "af66841b06e0b4f207e382744b6b5b9395583ad859a0bbabd1a81ae4915609d1"
    smallest_subnormal = float.fromhex("0x0.0000000000001p-1022")

    assert hashlib.sha256(literal_preimage).hexdigest() == expected_digest
    production_preimage = canonical_json_bytes({"tiny": smallest_subnormal})
    assert production_preimage == literal_preimage
    assert hashlib.sha256(production_preimage).hexdigest() == expected_digest


def test_known_answer_unicode_sorted_key_encoding() -> None:
    literal_preimage = b'{"a":"\xce\xbc","\xc3\xa9":"\xe9\x9b\xaa"}'
    expected_digest = "b9028d8f93dcd14564456a8f84190fd4eaae0d1aae353011ba5566606918cb76"

    assert hashlib.sha256(literal_preimage).hexdigest() == expected_digest
    production_preimage = canonical_json_bytes({"é": "雪", "a": "μ"})
    assert production_preimage == literal_preimage
    assert hashlib.sha256(production_preimage).hexdigest() == expected_digest


def test_known_answer_semantic_input_framing() -> None:
    literal_preimage = (
        b"selcal.semantic-input.v1\0"
        b'{"candidates":[1,2],"roles":["source","target"],"shape":[2]}'
        b"\x00\x00\x00\x00\x00\x00\xf0\x3f"
        b"\x00\x00\x00\x00\x00\x00\x04\xc0"
        b"\x00\x00\x00\x00\x00\x00\x0a\x40"
        b"\x00\x00\x00\x00\x00\x00\x12\x40"
    )
    expected_digest = "f59aa0a63241f45d161beb1906601a13c44c785fbb576379c6cd2c5cc9e80353"
    pair = SeriesPair(
        source=np.array([1.0, -2.5]),
        target=np.array([3.25, 4.5]),
    )

    assert hashlib.sha256(literal_preimage).hexdigest() == expected_digest
    assert semantic_input_sha256(pair, (2, 1)) == expected_digest


def test_known_answer_scientific_plan_payload() -> None:
    literal_preimage = (
        b'{"alpha":{"$float64":"0x1.999999999999ap-5"},'
        b'"candidates":[1,2,3],"common_support":"max_candidate_lag_v1",'
        b'"failure_policy":"fail_closed_v1","null":{"name":"circular_shift_v1",'
        b'"params":{"min_shift":1}},"replicates":9,'
        b'"rng_contract":"sha256-to-pcg64-v1","root_seed":17,'
        b'"schema":"selcal.scientific-plan.v1","selection":{"rule":"max_upper",'
        b'"tie_tolerance":{"$float64":"0x0.0p+0"}},'
        b'"statistic":{"name":"equal_width_binned_nette_v1","params":{"bins":3}}}'
    )
    expected_digest = "7371854f3ee4c9aff04571fc175857cdcc76fe3a879e70fc267f10cec3aada4e"

    assert hashlib.sha256(literal_preimage).hexdigest() == expected_digest
    assert scientific_plan_sha256(valid_plan(tie_tolerance=-0.0)) == expected_digest
