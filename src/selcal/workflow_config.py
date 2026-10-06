"""Bounded, readable user configuration for the file workflow.

This JSON document is not a canonical scientific identity or an execution
permission. The application still resolves registered adapter names and their
parameter contracts before running a plan.

Both the incoming document and its normalized encoding must fit 65,536 bytes.
The normalized size includes its final LF, so an otherwise valid input at the
incoming byte limit can be rejected if normalization would exceed that limit.
Every admitted configuration can therefore be encoded within the same limit.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, TypeVar, cast

from selcal.contracts import JsonValue, SelectionRule
from selcal.contracts_v2 import PlanRequestV2

_SCHEMA = "selcal.workflow-config.v1"
_MAX_BYTES = 65536
_MAX_DEPTH = 8
_ROOT_KEYS = frozenset({"schema", "input", "plan"})
_CSV_KEYS = frozenset({"format", "source_column", "target_column"})
_NPZ_KEYS = frozenset({"format"})
_PLAN_KEYS = frozenset(
    {
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
    }
)
_T = TypeVar("_T")


class WorkflowConfigError(ValueError):
    """Rejected config; ``code`` gives a stable category, not message parsing.

    Categories are type, shape, schema, duplicate_key, value, syntax,
    size_limit and depth_limit.
    """

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class WorkflowConfig:
    """An unresolved request and input selection with a bounded JSON encoding.

    Direct construction also checks the normalized 65,536-byte limit, including
    the final LF, and the depth-8 limit enforced by the reader.
    """

    request: PlanRequestV2
    source_format: Literal["csv", "npz"]
    source_column: str | None = None
    target_column: str | None = None

    def __post_init__(self) -> None:
        # Direct construction must obey the same value and document limits.
        _encode_config(self)


def _exact(value: object, expected: type[_T]) -> _T:
    if type(value) is not expected:
        raise WorkflowConfigError("type", f"config value must have exact type {expected.__name__}")
    return value


def _object(value: object, keys: frozenset[str]) -> dict[str, object]:
    mapping = _exact(value, dict)
    if mapping.keys() != keys:
        raise WorkflowConfigError("shape", f"config object must contain exactly {sorted(keys)}")
    return mapping


def _finite_float(value: object) -> float:
    number = _exact(value, float)
    if not math.isfinite(number):
        raise WorkflowConfigError("value", "config numbers must be finite")
    return number


def _columns(source: object, target: object) -> tuple[str, str]:
    source_name = _exact(source, str)
    target_name = _exact(target, str)
    if not source_name or not target_name or source_name == target_name:
        raise WorkflowConfigError("value", "CSV columns must be nonempty and distinct")
    return source_name, target_name


def _read_input(value: object) -> tuple[Literal["csv", "npz"], str | None, str | None]:
    mapping = _exact(value, dict)
    if "format" not in mapping:
        raise WorkflowConfigError("shape", "config input must include format")
    source_format = _exact(mapping["format"], str)
    if source_format == "csv":
        mapping = _object(value, _CSV_KEYS)
        source, target = _columns(mapping["source_column"], mapping["target_column"])
        return "csv", source, target
    if source_format == "npz":
        _object(value, _NPZ_KEYS)
        return "npz", None, None
    raise WorkflowConfigError("value", "input format must be csv or npz")


def _read_plan(value: object) -> PlanRequestV2:
    plan = _object(value, _PLAN_KEYS)
    try:
        return PlanRequestV2(
            candidates=tuple(_exact(item, int) for item in _exact(plan["candidates"], list)),
            statistic_name=_exact(plan["statistic_name"], str),
            statistic_params=_exact(plan["statistic_params"], dict),
            selection_rule=_exact(plan["selection_rule"], str),
            null_name=_exact(plan["null_name"], str),
            null_params=_exact(plan["null_params"], dict),
            replicates=_exact(plan["replicates"], int),
            alpha=_finite_float(plan["alpha"]),
            tie_tolerance=_finite_float(plan["tie_tolerance"]),
            root_seed=_exact(plan["root_seed"], int),
        )
    except WorkflowConfigError:
        raise
    except ValueError as error:
        raise WorkflowConfigError("value", "config violates the plan request contract") from error


def _check_document(data: bytes) -> None:
    if len(data) > _MAX_BYTES:
        raise WorkflowConfigError("size_limit", "config exceeds the 65536-byte limit")
    depth = 0
    quoted = False
    escaped = False
    for byte in data:
        if quoted:
            if escaped:
                escaped = False
            elif byte == 92:
                escaped = True
            elif byte == 34:
                quoted = False
        elif byte == 34:
            quoted = True
        elif byte in (91, 123):
            depth += 1
            if depth > _MAX_DEPTH:
                raise WorkflowConfigError("depth_limit", "config exceeds nesting depth 8")
        elif byte in (93, 125):
            depth -= 1
            if depth < 0:
                raise WorkflowConfigError("syntax", "invalid config container framing")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise WorkflowConfigError("duplicate_key", "config contains a duplicate object key")
        if key == "$float64":
            raise WorkflowConfigError("value", "$float64 is reserved, not a user config number")
        result[key] = value
    return result


def _parse_float(spelling: str) -> float:
    return _finite_float(float(spelling))


def _reject_constant(_spelling: str) -> object:
    raise WorkflowConfigError("value", "config numbers must be finite JSON numbers")


def _thaw_json(value: object, *, depth: int) -> JsonValue:
    if value is None or type(value) in {str, int, bool}:
        return cast(JsonValue, value)
    if type(value) is float:
        return _finite_float(value)
    if depth > _MAX_DEPTH:
        raise WorkflowConfigError("depth_limit", "config exceeds nesting depth 8")
    if type(value) in {dict, MappingProxyType}:
        mapping = cast(dict[object, object], value)
        result: dict[str, JsonValue] = {}
        for key, nested in mapping.items():
            name = _exact(key, str)
            if name == "$float64":
                raise WorkflowConfigError("value", "$float64 is reserved")
            result[name] = _thaw_json(nested, depth=depth + 1)
        return result
    if type(value) in {tuple, list}:
        return [_thaw_json(item, depth=depth + 1) for item in cast(tuple[object, ...], value)]
    raise WorkflowConfigError("type", "plan parameters must contain only exact JSON values")


def _project_plan(value: object) -> dict[str, object]:
    request = _exact(value, PlanRequestV2)
    plan: dict[str, object] = {
        "candidates": list(_exact(request.candidates, tuple)),
        "statistic_name": request.statistic_name,
        "statistic_params": _thaw_json(request.statistic_params, depth=3),
        "selection_rule": _exact(request.selection_rule, SelectionRule).value,
        "null_name": request.null_name,
        "null_params": _thaw_json(request.null_params, depth=3),
        "replicates": request.replicates,
        "alpha": request.alpha,
        "tie_tolerance": request.tie_tolerance,
        "root_seed": request.root_seed,
    }
    restored = _read_plan(plan)
    if restored != request:
        raise WorkflowConfigError("value", "request must contain normalized plan values")
    return plan


def _encode_config(value: object) -> bytes:
    config = _exact(value, WorkflowConfig)
    try:
        source_format = _exact(config.source_format, str)
        input_record: dict[str, object] = {"format": source_format}
        if source_format == "csv":
            input_record.update(
                source_column=config.source_column,
                target_column=config.target_column,
            )
        elif config.source_column is not None or config.target_column is not None:
            raise WorkflowConfigError("value", "NPZ config must not contain CSV column selections")
        _read_input(input_record)
        payload = {"schema": _SCHEMA, "input": input_record, "plan": _project_plan(config.request)}
        data = (
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
                ensure_ascii=False,
            ).encode("utf-8")
            + b"\n"
        )
    except WorkflowConfigError:
        raise
    except AttributeError as error:
        raise WorkflowConfigError(
            "type",
            "config or request has missing required fields",
        ) from error
    except (ValueError, OverflowError) as error:
        raise WorkflowConfigError("value", "config contains an unencodable value") from error
    _check_document(data)
    return data


def encode_workflow_config(config: WorkflowConfig) -> bytes:
    """Return stable, compact UTF-8 JSON and one LF, after complete validation.

    The normalized document, including its LF, must fit 65,536 bytes and depth 8.
    Limits bound document admission, not peak allocation or execution resources.
    The encoding is readable application configuration, not scientific hashing.
    """
    return _encode_config(config)


def decode_workflow_config(data: bytes) -> WorkflowConfig:
    """Read strict JSON with ordinary finite numbers and no scientific defaults.

    Whitespace and key order are unconstrained. Byte and quote-aware depth checks
    precede parsing; duplicates and reserved float tags are rejected recursively.
    Both incoming bytes and the normalized encoding including LF must fit 65,536
    bytes. An input that fits may still fail ``size_limit`` after normalization.
    """
    _exact(data, bytes)
    _check_document(data)
    try:
        payload = json.loads(
            data.decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_object,
            parse_float=_parse_float,
            parse_constant=_reject_constant,
        )
    except WorkflowConfigError:
        raise
    except ValueError as error:
        raise WorkflowConfigError("syntax", "config is not valid UTF-8 JSON") from error
    record = _object(payload, _ROOT_KEYS)
    if _exact(record["schema"], str) != _SCHEMA:
        raise WorkflowConfigError("schema", f"config schema must equal {_SCHEMA!r}")
    source_format, source, target = _read_input(record["input"])
    return WorkflowConfig(
        request=_read_plan(record["plan"]),
        source_format=source_format,
        source_column=source,
        target_column=target,
    )
