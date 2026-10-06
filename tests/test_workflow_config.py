"""Behavioral contract for bounded, readable workflow configuration."""

from __future__ import annotations

import importlib
import importlib.util
import json
from dataclasses import FrozenInstanceError
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from selcal.contracts import SelectionRule
from selcal.contracts_v2 import PlanRequestV2


def _api() -> ModuleType:
    spec = importlib.util.find_spec("selcal.workflow_config")
    assert spec is not None, "missing strict readable workflow config API"
    return importlib.import_module("selcal.workflow_config")


def _record() -> dict[str, Any]:
    return {
        "schema": "selcal.workflow-config.v1",
        "input": {"format": "csv", "source_column": "source", "target_column": "target"},
        "plan": {
            "candidates": [3, 1, 2],
            "statistic_name": "lagged_pearson_v1",
            "statistic_params": {},
            "selection_rule": "max_absolute",
            "null_name": "circular_shift_v2",
            "null_params": {"min_shift": 1},
            "replicates": 19,
            "alpha": 0.05,
            "tie_tolerance": 0.0,
            "root_seed": 17,
        },
    }


def _bytes(record: object) -> bytes:
    return json.dumps(record, ensure_ascii=False).encode("utf-8")


def _request(**changes: Any) -> PlanRequestV2:
    plan = _record()["plan"]
    plan.update(changes)
    plan["candidates"] = tuple(plan["candidates"])
    return PlanRequestV2(**plan)


def _reject(data: object, code: str) -> None:
    api = _api()
    with pytest.raises(api.WorkflowConfigError) as caught:
        api.decode_workflow_config(data)
    assert isinstance(caught.value, ValueError)
    assert caught.value.code == code


def test_csv_decodes_to_frozen_plan_with_normalized_candidates() -> None:
    api = _api()
    config = api.decode_workflow_config(_bytes(_record()))
    assert type(config) is api.WorkflowConfig
    assert config.request == _request()
    assert config.request.candidates == (1, 2, 3)
    assert config.request.selection_rule is SelectionRule.MAX_ABSOLUTE
    assert config.source_format == "csv"
    assert config.source_column == "source"
    assert config.target_column == "target"
    with pytest.raises(FrozenInstanceError):
        config.source_format = "npz"


def test_npz_round_trip_uses_no_column_fields() -> None:
    api = _api()
    record = _record()
    record["input"] = {"format": "npz"}
    config = api.decode_workflow_config(_bytes(record))
    assert config.source_format == "npz"
    assert config.source_column is config.target_column is None
    encoded = api.encode_workflow_config(config)
    assert json.loads(encoded)["input"] == {"format": "npz"}
    assert api.decode_workflow_config(encoded) == config


def test_readable_encoding_is_sorted_compact_plain_json_plus_one_lf() -> None:
    api = _api()
    record = _record()
    config = api.decode_workflow_config(b" \n" + _bytes(record) + b"\t\n\n")
    encoded = api.encode_workflow_config(config)
    record["plan"]["candidates"] = [1, 2, 3]
    expected = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert encoded == expected.encode("utf-8") + b"\n"
    assert b'"alpha":0.05' in encoded
    assert b"$float64" not in encoded
    assert api.encode_workflow_config(api.decode_workflow_config(encoded)) == encoded


def test_nette_and_block_parameters_retain_integer_types() -> None:
    api = _api()
    record = _record()
    record["plan"].update(
        statistic_name="equal_width_binned_nette_v1",
        statistic_params={"bins": 3},
        selection_rule="max_upper",
        null_name="block_shuffle_v2",
        null_params={"block_length": 4},
    )
    config = api.decode_workflow_config(_bytes(record))
    assert config.request == _request(**record["plan"])
    assert type(config.request.statistic_params["bins"]) is int
    assert type(config.request.null_params["block_length"]) is int
    assert api.decode_workflow_config(api.encode_workflow_config(config)).request == config.request


def test_nested_parameters_keep_all_json_value_kinds_and_round_trip() -> None:
    api = _api()
    record = _record()
    record["plan"]["statistic_params"] = {
        "data": [1, 1.0, True, False, None, "文化", {"zero": -0.0}],
        "limits": {"positive": 1.7976931348623157e308, "tiny": 5e-324},
    }
    config = api.decode_workflow_config(_bytes(record))
    values = config.request.statistic_params["data"]
    assert [type(value) for value in values[:6]] == [int, float, bool, bool, type(None), str]
    assert values[6]["zero"] == 0.0
    restored = api.decode_workflow_config(api.encode_workflow_config(config))
    assert restored.request == config.request
    assert restored.request.statistic_params == config.request.statistic_params


def test_configuration_leaves_registered_name_and_parameter_semantics_to_resolver() -> None:
    api = _api()
    record = _record()
    record["plan"]["statistic_name"] = "unregistered_statistic_v1"
    record["plan"]["statistic_params"] = {"bins": True, "arbitrary": 1.0}
    config = api.decode_workflow_config(_bytes(record))
    assert config.request.statistic_name == "unregistered_statistic_v1"
    assert config.request.statistic_params["bins"] is True
    assert type(config.request.statistic_params["arbitrary"]) is float


@pytest.mark.parametrize("container", [(), ("input",), ("plan",)])
def test_missing_and_extra_keys_are_rejected_at_each_closed_object(
    container: tuple[str, ...],
) -> None:
    record = _record()
    mapping = record
    for part in container:
        mapping = mapping[part]
    mapping["unexpected"] = True
    _reject(_bytes(record), "shape")
    del mapping["unexpected"]
    for field in tuple(mapping):
        value = mapping.pop(field)
        _reject(_bytes(record), "shape")
        mapping[field] = value


@pytest.mark.parametrize("record", [None, [], 7, True, "config"])
def test_root_requires_exact_json_object(record: object) -> None:
    _reject(_bytes(record), "type")


@pytest.mark.parametrize("schema", ["selcal.workflow-config.v2", "selcal.scientific-plan.v2", ""])
def test_unrecognized_schema_is_distinct_from_syntax(schema: str) -> None:
    record = _record()
    record["schema"] = schema
    _reject(_bytes(record), "schema")


@pytest.mark.parametrize("schema", [None, True, 1, []])
def test_schema_requires_string(schema: object) -> None:
    record = _record()
    record["schema"] = schema
    _reject(_bytes(record), "type")


@pytest.mark.parametrize(
    ("input_value", "code"),
    [
        (None, "type"),
        ([], "type"),
        ({"format": True}, "type"),
        ({"format": "CSV"}, "value"),
        ({"format": "json"}, "value"),
        ({"format": "npz", "source_column": "x"}, "shape"),
        ({"format": "npz", "target_column": None}, "shape"),
        ({"format": "csv", "source_column": "x", "target_column": "x"}, "value"),
        ({"format": "csv", "source_column": "", "target_column": "y"}, "value"),
        ({"format": "csv", "source_column": "x", "target_column": ""}, "value"),
        ({"format": "csv", "source_column": 1, "target_column": "y"}, "type"),
        ({"format": "csv", "source_column": "x", "target_column": None}, "type"),
    ],
)
def test_input_contract_has_exact_format_and_column_rules(input_value: object, code: str) -> None:
    record = _record()
    record["input"] = input_value
    _reject(_bytes(record), code)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("candidates", "1,2"),
        ("candidates", [1.0]),
        ("candidates", [True]),
        ("candidates", ["1"]),
        ("statistic_name", None),
        ("statistic_params", []),
        ("selection_rule", False),
        ("null_name", 1),
        ("null_params", None),
        ("replicates", True),
        ("replicates", 19.0),
        ("alpha", 1),
        ("alpha", True),
        ("alpha", "0.05"),
        ("tie_tolerance", 0),
        ("root_seed", False),
        ("root_seed", 17.0),
    ],
)
def test_plan_does_not_coerce_scalar_or_container_types(field: str, value: object) -> None:
    record = _record()
    record["plan"][field] = value
    _reject(_bytes(record), "type")


@pytest.mark.parametrize("plan", [None, [], "plan"])
def test_plan_requires_object(plan: object) -> None:
    record = _record()
    record["plan"] = plan
    _reject(_bytes(record), "type")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("candidates", []),
        ("candidates", [0]),
        ("candidates", [1, 1]),
        ("statistic_name", "unversioned"),
        ("null_name", ""),
        ("selection_rule", "unsupported"),
        ("replicates", 0),
        ("replicates", 1_000_001),
        ("alpha", 0.0),
        ("alpha", 1.0),
        ("tie_tolerance", -0.1),
        ("root_seed", -1),
        ("root_seed", 2**64),
    ],
)
def test_existing_plan_constructor_value_constraints_still_apply(field: str, value: object) -> None:
    record = _record()
    record["plan"][field] = value
    _reject(_bytes(record), "value")


@pytest.mark.parametrize(
    "data",
    [
        b'{"schema":0,"schema":1}',
        _bytes(_record()).replace(b'"min_shift": 1', b'"min_shift": 1,"min_shift": 2'),
        _bytes(_record()).replace(
            b'"statistic_params": {}',
            b'"statistic_params":{"a":[{"x":1,"x":2}]}',
        ),
        _bytes(_record()).replace(b'"alpha": 0.05', b'"alpha":0.05,"alpha":0.05'),
    ],
)
def test_duplicate_keys_are_rejected_recursively(data: bytes) -> None:
    _reject(data, "duplicate_key")


@pytest.mark.parametrize("token", [b"NaN", b"Infinity", b"-Infinity", b"1e400", b"-1e400"])
def test_nonfinite_and_overflow_json_numbers_are_rejected(token: bytes) -> None:
    _reject(_bytes(_record()).replace(b'"alpha": 0.05', b'"alpha":' + token), "value")
    nested = b'"statistic_params":{"a":[' + token + b"]}"
    _reject(_bytes(_record()).replace(b'"statistic_params": {}', nested), "value")


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"\xff",
        b"\xef\xbb\xbf{}",
        b"{} {}",
        b"{",
        b"}",
        b'{"x":}',
        b'[[["unfinished]',
    ],
)
def test_malformed_utf8_and_json_raise_typed_syntax_error(data: bytes) -> None:
    _reject(data, "syntax")


@pytest.mark.parametrize("data", ["{}", bytearray(b"{}"), memoryview(b"{}"), None])
def test_decoder_accepts_only_builtin_bytes(data: object) -> None:
    _reject(data, "type")


@pytest.mark.parametrize("field", ["statistic_params", "null_params"])
def test_reserved_float_tag_is_rejected_in_nested_parameters(field: str) -> None:
    record = _record()
    record["plan"][field] = {"outer": [{"$float64": "0x1p+0"}]}
    _reject(_bytes(record), "value")


def test_reserved_float_tag_is_not_interpreted_as_a_plan_float() -> None:
    record = _record()
    record["plan"]["alpha"] = {"$float64": "0x1p-3"}
    _reject(_bytes(record), "value")


def test_byte_limit_accepts_exact_boundary_and_rejects_one_more_before_parsing() -> None:
    api = _api()
    data = _bytes(_record())
    padded = data + b" " * (65536 - len(data))
    assert api.decode_workflow_config(padded).request == _request()
    _reject(padded + b" ", "size_limit")
    _reject(b"!" * 65537, "size_limit")


def test_normalized_encoding_including_lf_accepts_exact_byte_limit() -> None:
    api = _api()
    record = _record()
    record["plan"]["candidates"] = [1, 2, 3]
    record["input"]["source_column"] = ""
    empty = json.dumps(record, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    record["input"]["source_column"] = "s" * (65536 - len(empty))
    data = json.dumps(record, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    assert len(data) == 65536
    config = api.decode_workflow_config(data)
    assert config.request == _request()
    assert config.source_column != config.target_column
    assert api.encode_workflow_config(config) == data


def test_incoming_byte_limit_rejects_normalized_encoding_one_byte_over_limit() -> None:
    record = _record()
    record["plan"]["candidates"] = [1, 2, 3]
    record["input"]["source_column"] = ""
    empty = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    record["input"]["source_column"] = "s" * (65536 - len(empty))
    data = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    assert len(data) == 65536
    assert len(data + b"\n") == 65537
    assert json.loads(data) == record
    _reject(data, "size_limit")


def test_depth_limit_counts_containers_and_ignores_quoted_escaped_delimiters() -> None:
    api = _api()
    record = _record()
    # root / plan / params / five arrays = exactly eight open containers.
    record["plan"]["statistic_params"] = {"nested": [[[[[0]]]]], "text": '[{\\"' * 20}
    config = api.decode_workflow_config(_bytes(record))
    assert api.decode_workflow_config(api.encode_workflow_config(config)).request == config.request
    record["plan"]["statistic_params"]["nested"] = [[[[[[0]]]]]]
    _reject(_bytes(record), "depth_limit")
    _reject(b"[" * 10000, "depth_limit")


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        ({"source_format": "npz", "source_column": "x", "target_column": None}, "value"),
        ({"source_format": "csv", "source_column": None, "target_column": "y"}, "type"),
        ({"source_format": "csv", "source_column": "x", "target_column": "x"}, "value"),
        ({"source_format": "csv", "source_column": "", "target_column": "y"}, "value"),
        ({"source_format": "json", "source_column": None, "target_column": None}, "value"),
        ({"source_format": True, "source_column": None, "target_column": None}, "type"),
    ],
)
def test_direct_construction_rejects_invalid_input_configuration(
    kwargs: dict[str, object],
    code: str,
) -> None:
    api = _api()
    with pytest.raises(api.WorkflowConfigError) as caught:
        api.WorkflowConfig(request=_request(), **kwargs)
    assert caught.value.code == code


def test_direct_construction_requires_exact_plan_type() -> None:
    api = _api()
    with pytest.raises(api.WorkflowConfigError) as caught:
        api.WorkflowConfig(request=_record()["plan"], source_format="npz")
    assert caught.value.code == "type"


@pytest.mark.parametrize(
    ("params", "code"),
    [({"text": "x" * 65536}, "size_limit"), ({"nested": [[[[[[0]]]]]]}, "depth_limit")],
)
def test_direct_construction_obeys_reader_byte_and_depth_limits(
    params: dict[str, Any],
    code: str,
) -> None:
    api = _api()
    with pytest.raises(api.WorkflowConfigError) as caught:
        api.WorkflowConfig(request=_request(statistic_params=params), source_format="npz")
    assert caught.value.code == code


def test_direct_npz_construction_has_optional_column_defaults() -> None:
    api = _api()
    config = api.WorkflowConfig(request=_request(), source_format="npz")
    assert config.source_column is config.target_column is None
    assert api.decode_workflow_config(api.encode_workflow_config(config)) == config


@pytest.mark.parametrize("config", [None, {}, _record(), "config"])
def test_encoder_rejects_objects_of_wrong_type(config: object) -> None:
    api = _api()
    with pytest.raises(api.WorkflowConfigError) as caught:
        api.encode_workflow_config(config)
    assert caught.value.code == "type"


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [("replicates", True, "type"), ("alpha", 1, "type"), ("alpha", float("inf"), "value")],
)
def test_encoder_revalidates_plan_slots_even_if_frozen_guard_was_bypassed(
    field: str,
    value: object,
    code: str,
) -> None:
    api = _api()
    config = api.WorkflowConfig(request=_request(), source_format="npz")
    object.__setattr__(config.request, field, value)
    with pytest.raises(api.WorkflowConfigError) as caught:
        api.encode_workflow_config(config)
    assert caught.value.code == code


def test_encoder_revalidates_config_slots_even_if_frozen_guard_was_bypassed() -> None:
    api = _api()
    config = api.WorkflowConfig(request=_request(), source_format="npz")
    object.__setattr__(config, "target_column", "target")
    with pytest.raises(api.WorkflowConfigError) as caught:
        api.encode_workflow_config(config)
    assert caught.value.code == "value"


@pytest.mark.parametrize("field", ["statistic_params", "null_params"])
@pytest.mark.parametrize("value", [np.uint64(3), {"bins": np.uint64(3)}])
def test_numpy_parameter_values_reach_typed_error_at_both_config_boundaries(
    field: str,
    value: object,
) -> None:
    api = _api()
    request = _request()
    config = api.WorkflowConfig(request=request, source_format="npz")
    # A normal PlanRequestV2 already rejects these values. Exercise the config
    # boundary itself using a request whose frozen guard was explicitly bypassed.
    object.__setattr__(request, field, value)
    with pytest.raises(api.WorkflowConfigError) as direct_error:
        api.WorkflowConfig(request=request, source_format="npz")
    assert direct_error.value.code == "type"
    with pytest.raises(api.WorkflowConfigError) as encoding_error:
        api.encode_workflow_config(config)
    assert encoding_error.value.code == "type"
