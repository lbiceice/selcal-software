"""Lossless content storage tests using the public v2 calibration path."""

from __future__ import annotations

import importlib
import importlib.util
import json
from dataclasses import fields, is_dataclass
from enum import Enum

import numpy as np
import pytest

from selcal import PlanRequestV2, resolve_plan_v2

# The kernel, not the public analysis entry: these fixtures build a null-bind terminal from a
# restricted circular shift, which the public entry now refuses before running (R11-01).
from selcal.calibration_v2 import calibrate_selected_family
from selcal.canonical import canonical_json_bytes
from selcal.contracts import RunStatus, SeriesPair
from selcal.contracts_v2 import CalibrationResult, RunFailureStage


def _wire():
    assert importlib.util.find_spec("selcal.result_wire") is not None, (
        "the production lossless result wire API must exist"
    )
    return importlib.import_module("selcal.result_wire")


def _null_bind_result() -> CalibrationResult:
    pair = SeriesPair(
        source=np.asarray([0, 1, 2, 3, 4], dtype=np.float64),
        target=np.asarray([4, 3, 2, 1, 0], dtype=np.float64),
    )
    resolution = resolve_plan_v2(
        PlanRequestV2(
            candidates=(1, 3),
            statistic_name="lagged_pearson_v1",
            statistic_params={},
            selection_rule="max_upper",
            null_name="circular_shift_v2",
            null_params={"min_shift": 3},
            replicates=7,
            alpha=0.05,
            tie_tolerance=1e-12,
            root_seed=17,
        )
    )
    return calibrate_selected_family(pair, resolution)


def test_public_null_bind_result_roundtrips_losslessly() -> None:
    wire = _wire()
    result = _null_bind_result()
    assert result.failure_stage is RunFailureStage.NULL_BIND
    assert result.planned_replicates == 7
    encoded = wire.encode_calibration_result(result, max_bytes=100_000)
    restored = wire.decode_calibration_result(encoded, max_bytes=len(encoded))
    assert type(restored) is CalibrationResult
    assert restored == result
    assert encoded.endswith(b"\n") and not encoded.endswith(b"\n\n")
    assert wire.encode_calibration_result(restored, max_bytes=len(encoded)) == encoded


def _run(kind: str) -> CalibrationResult:
    if kind == "null_bind":
        return _null_bind_result()
    source = [0, 1, 4, 2, 5, 3]
    target = [4, 1, 3, 0, 5, 2]
    if kind == "observed_statistic_scan":
        source = [0] * 6
    elif kind == "replicate_execution":
        source = [0, 0, 1, 2, 0, 0]
        target = [0, 1, 2, 3, 4, 5]
    block = kind == "block_complete"
    resolution = resolve_plan_v2(
        PlanRequestV2(
            candidates=(1, 2),
            statistic_name="lagged_pearson_v1",
            statistic_params={},
            selection_rule="max_upper",
            null_name="block_shuffle_v2" if block else "circular_shift_v2",
            null_params={"block_length": 2} if block else {"min_shift": 1},
            replicates=9,
            alpha=0.05,
            tie_tolerance=1e-12,
            root_seed=17,
        )
    )
    return calibrate_selected_family(
        SeriesPair(np.asarray(source, dtype=np.float64), np.asarray(target, dtype=np.float64)),
        resolution,
    )


@pytest.fixture(scope="module")
def results():
    return {
        kind: _run(kind)
        for kind in (
            "null_bind",
            "observed_statistic_scan",
            "replicate_execution",
            "circular_complete",
            "block_complete",
        )
    }


def _all_fields(value):
    """Independent dataclass traversal detects any dropped or changed field."""
    if is_dataclass(value):
        return {field.name: _all_fields(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Enum):
        return value.value
    if type(value) is tuple:
        return tuple(_all_fields(item) for item in value)
    return value


@pytest.mark.parametrize(
    "kind",
    [
        "null_bind",
        "observed_statistic_scan",
        "replicate_execution",
        "circular_complete",
        "block_complete",
    ],
)
def test_every_field_in_each_real_terminal_branch_roundtrips(results, kind) -> None:
    wire = _wire()
    result = results[kind]
    expected_stage = None if kind.endswith("complete") else RunFailureStage(kind)
    assert result.failure_stage is expected_stage
    assert result.status is (
        RunStatus.COMPLETE if expected_stage is None else RunStatus.NOT_EVALUABLE
    )
    if kind == "replicate_execution":
        assert (result.exceedance_count, result.failure_count) == (4, 4)
        assert (result.exceedance_bound_low, result.exceedance_bound_high) == (0.5, 0.9)
        assert len(result.replicates) == 9
    expected = (
        canonical_json_bytes(
            {
                "schema": "selcal.calibration-result-wire.v1",
                **_all_fields(result),
            }
        )
        + b"\n"
    )
    encoded = wire.encode_calibration_result(result, max_bytes=len(expected))
    assert encoded == expected
    restored = wire.decode_calibration_result(encoded, max_bytes=len(encoded))
    assert restored == result and restored is not result
    assert _all_fields(restored) == _all_fields(result)
    assert wire.encode_calibration_result(restored, max_bytes=len(encoded)) == encoded


def _payload(results, kind="circular_complete"):
    return json.loads(_wire().encode_calibration_result(results[kind], max_bytes=100_000))


def _json_bytes(payload):
    return (
        json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
        + b"\n"
    )


def _at(payload, path):
    for component in path:
        payload = payload[component]
    return payload


def _reject(data, code):
    wire = _wire()
    with pytest.raises(wire.ResultWireError) as error:
        wire.decode_calibration_result(data, max_bytes=max(len(data), 1))
    assert isinstance(error.value, ValueError)
    assert error.value.code == code


@pytest.mark.parametrize("value", [0, -1, True, False, 1.0, "100", None, np.int64(100)])
def test_both_apis_require_positive_builtin_integer_limit(results, value) -> None:
    wire = _wire()
    for operation, argument in (
        (wire.encode_calibration_result, results["null_bind"]),
        (wire.decode_calibration_result, b"{}\n"),
    ):
        with pytest.raises(wire.ResultWireError) as error:
            operation(argument, max_bytes=value)
        assert error.value.code == "invalid_limit"


def test_cap_boundary_is_exact_for_both_apis(results) -> None:
    wire = _wire()
    result = results["circular_complete"]
    data = wire.encode_calibration_result(result, max_bytes=100_000)
    for operation, argument in (
        (wire.encode_calibration_result, result),
        (wire.decode_calibration_result, data),
    ):
        with pytest.raises(wire.ResultWireError) as error:
            operation(argument, max_bytes=len(data) - 1)
        assert error.value.code == "size_limit"
        operation(argument, max_bytes=len(data))


@pytest.mark.parametrize("data", ["{}", bytearray(b"{}"), memoryview(b"{}"), None])
def test_decoder_requires_actual_bytes(data) -> None:
    with pytest.raises(_wire().ResultWireError) as error:
        _wire().decode_calibration_result(data, max_bytes=100)
    assert error.value.code == "type"


@pytest.mark.parametrize(
    "replacement",
    [
        b'"p_value":null,"p_value":null',
        b'"p_value":null,"p_value":{"$float64":"0x1.0000000000000p-1"}',
        b'"p_value":{"$float64":"0x1.0000000000000p-1"},"p_value":null',
    ],
)
def test_duplicate_root_keys_rejected_even_when_identical(results, replacement) -> None:
    data = _json_bytes(_payload(results, "null_bind"))
    _reject(data.replace(b'"p_value":null', replacement), "duplicate_key")


@pytest.mark.parametrize(
    "path,key",
    [
        (("replicates", 0), "replicate_id"),
        (("replicates", 0, "transform_token"), "is_identity"),
        (("replicates", 0, "transform_token", "state"), "shift"),
        (("replicates", 0, "statistic_results", 0), "candidate_id"),
        (("replicates", 0, "selection"), "selected_index"),
        (("replicates", 0, "statistic_results", 0, "estimate"), "$float64"),
    ],
)
def test_nested_duplicate_keys_rejected(results, path, key) -> None:
    payload = _payload(results)
    nested = _at(payload, path)
    original = _json_bytes(nested)[:-1]
    member = _json_bytes({key: nested[key]})[1:-2]
    duplicated = original[:-1] + b"," + member + b"}"
    _reject(_json_bytes(payload).replace(original, duplicated, 1), "duplicate_key")


@pytest.mark.parametrize(
    "path",
    [
        (),
        ("replicates", 0),
        ("replicates", 0, "transform_token"),
        ("replicates", 0, "transform_token", "state"),
        ("observed_results", 0),
        ("observed_selection",),
        ("replicates", 0, "statistic_results", 0),
        ("replicates", 0, "selection"),
        ("alpha",),
    ],
)
@pytest.mark.parametrize("change", ["missing", "extra"])
def test_exact_key_sets_at_every_object_level(results, path, change) -> None:
    payload = _payload(results)
    nested = _at(payload, path)
    if change == "missing":
        del nested[next(iter(nested))]
    else:
        nested["unexpected"] = None
    _reject(_json_bytes(payload), "shape")


@pytest.mark.parametrize(
    "path,value,code",
    [
        (("planned_replicates",), True, "type"),
        (("diagnostics",), {}, "type"),
        (("diagnostics",), [1], "type"),
        (("status",), "COMPLETE", "value"),
        (("status",), 1, "type"),
        (("schema",), "selcal.calibration-result-wire.v2", "value"),
        (("alpha",), 1, "type"),
        (("alpha", "$float64"), 1, "type"),
        (("alpha", "$float64"), "nan", "value"),
        (("alpha", "$float64"), "inf", "value"),
        (("alpha", "$float64"), "not-a-float", "value"),
        (("alpha", "$float64"), "0x1p+999999", "value"),
        (("alpha", "$float64"), "0x1.999999999999a0p-5", "noncanonical"),
        (("observed_results", 0, "candidate_id"), True, "type"),
        (("observed_results", 0, "support_n"), "4", "type"),
        (("observed_results", 0, "validity"), "VALID", "value"),
        (("observed_results", 0, "backend_identity"), {"$float64": "0x0.0p+0"}, "type"),
        (("observed_selection", "selected_index"), False, "type"),
        (("observed_selection", "tied_candidates"), [True], "type"),
        (("replicates", 0, "replicate_id"), True, "type"),
        (("replicates", 0, "status"), "unknown", "value"),
        (("replicates", 0, "transform_token", "is_identity"), 0, "type"),
        (("replicates", 0, "transform_token", "state", "shift"), True, "type"),
        (("replicates", 0, "transform_token", "state", "schema"), "unknown", "value"),
        (("replicates", 0, "transform_token", "null_name"), "unknown", "value"),
        (("replicates", 0, "transform_token", "semantic_input_sha256"), "a" * 64, "value"),
        (("semantic_input_sha256",), "A" * 64, "value"),
        (("reject_null",), 0, "type"),
        (("exceedance_count",), 100, "value"),
        (("p_value",), None, "value"),
    ],
)
def test_strict_leaf_and_scientific_invariant_rejections(results, path, value, code) -> None:
    payload = _payload(results)
    _at(payload, path[:-1])[path[-1]] = value
    _reject(_json_bytes(payload), code)


@pytest.mark.parametrize("value", [0.05, float("nan"), float("inf"), -float("inf")])
def test_raw_json_float_tokens_and_constants_rejected(results, value) -> None:
    payload = _payload(results)
    payload["alpha"] = value
    _reject(_json_bytes(payload), "value")


@pytest.mark.parametrize(
    "mutation",
    [
        lambda data: data[:-1],
        lambda data: data + b"\n",
        lambda data: b" " + data,
        lambda data: data.replace(b'"alpha":', b'"alpha": ', 1),
        lambda data: data.replace(b'"alpha"', b'"\\u0061lpha"', 1),
        lambda data: data.replace(b'"exceedance_count":0', b'"exceedance_count":-0', 1),
    ],
)
def test_noncanonical_framing_and_spelling_rejected(results, mutation) -> None:
    data = _json_bytes(_payload(results, "null_bind"))
    changed = mutation(data)
    assert changed != data
    _reject(changed, "noncanonical")


@pytest.mark.parametrize("data", [b"", b"\xff", b"{", b"{}{}\n", b'"unterminated'])
def test_invalid_syntax_rejected(data) -> None:
    _reject(data, "syntax")


@pytest.mark.parametrize("data", [b"[]\n", b"null\n", b'"text"\n'])
def test_nonobject_roots_rejected(data) -> None:
    _reject(data, "type")


def test_depth_admission_precedes_json_parser(monkeypatch) -> None:
    wire = _wire()

    def must_not_parse(*args, **kwargs):
        raise AssertionError("too deep input reached json parser")

    monkeypatch.setattr(wire.json, "loads", must_not_parse)
    _reject(b"[" * 100_000 + b"]" * 100_000, "shape")


def test_size_admission_precedes_json_parser(monkeypatch) -> None:
    wire = _wire()

    def must_not_parse(*args, **kwargs):
        raise AssertionError("oversize input reached json parser")

    monkeypatch.setattr(wire.json, "loads", must_not_parse)
    with pytest.raises(wire.ResultWireError) as error:
        wire.decode_calibration_result(b"x" * 101, max_bytes=100)
    assert error.value.code == "size_limit"


def test_strings_are_not_counted_as_nesting(results) -> None:
    payload = _payload(results, "null_bind")
    payload["diagnostics"] = ['[[[[[[[[[[{{{{{{\\"\\\\', "文物", "", "ordered", "ordered"]
    data = _json_bytes(payload)
    result = _wire().decode_calibration_result(data, max_bytes=len(data))
    assert result.diagnostics == tuple(payload["diagnostics"])
    assert _wire().encode_calibration_result(result, max_bytes=len(data)) == data


@pytest.mark.parametrize("value", [[0, 0, 2], [0, True, 2], [0, 1], None])
def test_block_state_permutation_is_strict(results, value) -> None:
    payload = _payload(results, "block_complete")
    payload["replicates"][0]["transform_token"]["state"]["block_order"] = value
    # [0,1] can describe a different valid state universe, which RESULT alone cannot bind.
    if value == [0, 1] and all(type(item) is int for item in value):
        payload["replicates"][0]["transform_token"]["is_identity"] = False
        expected = "value"
    else:
        expected = (
            "type" if value is None or any(type(item) is not int for item in value) else "value"
        )
    _reject(_json_bytes(payload), expected)


@pytest.mark.parametrize(
    "field,value",
    [
        ("failure_count", 0),
        ("exceedance_bound_low", None),
        ("exceedance_bound_high", {"$float64": "0x1.0000000000000p+0"}),
        ("reject_null", False),
        ("failure_stage", None),
    ],
)
def test_retained_failure_arithmetic_cannot_be_changed(results, field, value) -> None:
    payload = _payload(results, "replicate_execution")
    payload[field] = value
    _reject(_json_bytes(payload), "value")


@pytest.mark.parametrize(
    "path,value",
    [
        (("planned_replicates",), True),
        (("diagnostics",), ["list"]),
        (("observed_results", 0, "candidate_id"), True),
        (("observed_results", 0, "estimate"), np.float64(0.5)),
        (("observed_results", 0, "validity"), "valid"),
        (("observed_selection", "tied_candidates"), [1]),
        (("replicates", 0, "transform_token", "state", "shift"), True),
    ],
)
def test_encoder_rejects_tampered_nested_leaf_types(path, value) -> None:
    result = _run("circular_complete")
    owner = result
    for component in path[:-1]:
        owner = owner[component] if type(component) is int else getattr(owner, component)
    object.__setattr__(owner, path[-1], value)
    wire = _wire()
    with pytest.raises(wire.ResultWireError) as error:
        wire.encode_calibration_result(result, max_bytes=100_000)
    assert error.value.code == "type"


def test_encoder_reconstructs_tampered_nested_invariants() -> None:
    result = _run("circular_complete")
    token = result.replicates[0].transform_token
    object.__setattr__(token, "is_identity", not token.is_identity)
    with pytest.raises(_wire().ResultWireError) as error:
        _wire().encode_calibration_result(result, max_bytes=100_000)
    assert error.value.code == "value"


def test_encoder_requires_exact_result_type() -> None:
    with pytest.raises(_wire().ResultWireError) as error:
        _wire().encode_calibration_result({}, max_bytes=100)
    assert error.value.code == "type"


def test_process_memory_failure_is_not_relabelled(monkeypatch, results) -> None:
    wire = _wire()
    data = _json_bytes(_payload(results, "null_bind"))

    def no_memory(*args, **kwargs):
        raise MemoryError("test allocation failure")

    monkeypatch.setattr(wire.json, "loads", no_memory)
    with pytest.raises(MemoryError):
        wire.decode_calibration_result(data, max_bytes=len(data))
