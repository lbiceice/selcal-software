"""Strict, lossless storage of v2 result content, with no execution authority.

The caller's byte cap bounds record admission, not process memory. Restored
content is neither authenticated execution nor verification against real input,
a resolved plan, or replay. Decoding grants no live ownership or resolver seal.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from enum import Enum
from typing import Literal, TypeVar, cast

from selcal.canonical import canonical_json_bytes
from selcal.contracts import (
    JsonValue,
    ReplicateStatus,
    RunStatus,
    SelectionResult,
    StatisticResult,
    Validity,
)
from selcal.contracts_v2 import (
    BlockShuffleStateV2,
    CalibrationResult,
    CircularShiftStateV2,
    NullTransformToken,
    ReplicateFailureStage,
    ReplicateOutcome,
    RunFailureStage,
    V2IntegrityError,
)

_SCHEMA = "selcal.calibration-result-wire.v1"
_RESULT_KEYS = frozenset(
    {
        "schema",
        "status",
        "failure_stage",
        "semantic_input_sha256",
        "scientific_plan_sha256",
        "planned_replicates",
        "alpha",
        "observed_results",
        "observed_selection",
        "replicates",
        "exceedance_count",
        "failure_count",
        "p_value",
        "exceedance_bound_low",
        "exceedance_bound_high",
        "reject_null",
        "diagnostics",
    }
)
_OUTCOME_KEYS = frozenset(
    {
        "replicate_id",
        "seed_digest_sha256",
        "status",
        "failure_stage",
        "transform_token",
        "statistic_results",
        "selection",
        "diagnostics",
    }
)
_TOKEN_KEYS = frozenset(
    {
        "schema",
        "null_name",
        "null_parameter_sha256",
        "semantic_input_sha256",
        "scientific_plan_sha256",
        "bound_null_owner_sha256",
        "is_identity",
        "state",
    }
)
_STATISTIC_KEYS = frozenset(
    {
        "candidate_id",
        "estimate",
        "selection_score",
        "support_n",
        "validity",
        "diagnostics",
        "backend_identity",
        "preprocessing_identity",
    }
)
_SELECTION_KEYS = frozenset(
    {
        "selected_candidate",
        "selected_index",
        "decision_statistic",
        "tied_candidates",
    }
)
_T = TypeVar("_T")
_E = TypeVar("_E", bound=Enum)


class ResultWireError(ValueError):
    """Rejected result record; ``code`` is a stable contract error category."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def _exact(value: object, expected: type[_T]) -> _T:
    if type(value) is not expected:
        raise ResultWireError("type", "record contains a value of the wrong exact type")
    return value


def _optional(value: object, convert: Callable[[object], _T]) -> _T | None:
    return None if value is None else convert(value)


def _array(value: object, convert: Callable[[object], _T]) -> tuple[_T, ...]:
    return tuple(convert(item) for item in _exact(value, list))


def _tuple(value: object, convert: Callable[[object], _T]) -> tuple[_T, ...]:
    return tuple(convert(item) for item in _exact(value, tuple))


def _integer(value: object) -> int:
    return _exact(value, int)


def _string(value: object) -> str:
    return _exact(value, str)


def _boolean(value: object) -> bool:
    return _exact(value, bool)


def _float(value: object) -> float:
    number = _exact(value, float)
    if not math.isfinite(number):
        raise ResultWireError("value", "record floats must be finite")
    return number


def _enum_value(value: object, expected: type[_E]) -> str:
    return _string(_exact(value, expected).value)


def _read_enum(value: object, expected: type[_E]) -> _E:
    return expected(_string(value))


def _object(value: object, keys: frozenset[str]) -> dict[str, object]:
    mapping = _exact(value, dict)
    if mapping.keys() != keys:
        raise ResultWireError("shape", "record object must have exactly its required keys")
    return mapping


def _read_float(value: object) -> float:
    tag = _object(value, frozenset({"$float64"}))
    number = float.fromhex(_string(tag["$float64"]))
    return _float(number)


def _project_statistic(value: object) -> dict[str, object]:
    record = _exact(value, StatisticResult)
    return {
        "candidate_id": _integer(record.candidate_id),
        "estimate": _optional(record.estimate, _float),
        "selection_score": _optional(record.selection_score, _float),
        "support_n": _integer(record.support_n),
        "validity": _enum_value(record.validity, Validity),
        "diagnostics": _tuple(record.diagnostics, _string),
        "backend_identity": _string(record.backend_identity),
        "preprocessing_identity": _string(record.preprocessing_identity),
    }


def _read_statistic(value: object) -> StatisticResult:
    record = _object(value, _STATISTIC_KEYS)
    return StatisticResult(
        candidate_id=_integer(record["candidate_id"]),
        estimate=_optional(record["estimate"], _read_float),
        selection_score=_optional(record["selection_score"], _read_float),
        support_n=_integer(record["support_n"]),
        validity=_read_enum(record["validity"], Validity),
        diagnostics=_array(record["diagnostics"], _string),
        backend_identity=_string(record["backend_identity"]),
        preprocessing_identity=_string(record["preprocessing_identity"]),
    )


def _project_selection(value: object) -> dict[str, object]:
    record = _exact(value, SelectionResult)
    return {
        "selected_candidate": _integer(record.selected_candidate),
        "selected_index": _integer(record.selected_index),
        "decision_statistic": _float(record.decision_statistic),
        "tied_candidates": _tuple(record.tied_candidates, _integer),
    }


def _read_selection(value: object) -> SelectionResult:
    record = _object(value, _SELECTION_KEYS)
    return SelectionResult(
        selected_candidate=_integer(record["selected_candidate"]),
        selected_index=_integer(record["selected_index"]),
        decision_statistic=_read_float(record["decision_statistic"]),
        tied_candidates=_array(record["tied_candidates"], _integer),
    )


def _project_state(value: object) -> dict[str, object]:
    if type(value) is CircularShiftStateV2:
        return {"schema": _string(value.schema), "shift": _integer(value.shift)}
    record = _exact(value, BlockShuffleStateV2)
    return {"schema": _string(record.schema), "block_order": _tuple(record.block_order, _integer)}


def _read_state(value: object) -> CircularShiftStateV2 | BlockShuffleStateV2:
    record = _exact(value, dict)
    if "schema" not in record:
        raise ResultWireError("shape", "null state requires its schema key")
    schema = _string(record["schema"])
    if schema == "selcal.circular-shift-state.v2":
        _object(record, frozenset({"schema", "shift"}))
        return CircularShiftStateV2(
            schema="selcal.circular-shift-state.v2", shift=_integer(record["shift"])
        )
    if schema == "selcal.block-shuffle-state.v2":
        _object(record, frozenset({"schema", "block_order"}))
        return BlockShuffleStateV2(
            schema="selcal.block-shuffle-state.v2",
            block_order=_array(record["block_order"], _integer),
        )
    raise ResultWireError("value", "unsupported null state schema")


def _project_token(value: object) -> dict[str, object]:
    record = _exact(value, NullTransformToken)
    return {
        "schema": _string(record.schema),
        "null_name": _string(record.null_name),
        "null_parameter_sha256": _string(record.null_parameter_sha256),
        "semantic_input_sha256": _string(record.semantic_input_sha256),
        "scientific_plan_sha256": _string(record.scientific_plan_sha256),
        "bound_null_owner_sha256": _string(record.bound_null_owner_sha256),
        "is_identity": _boolean(record.is_identity),
        "state": _project_state(record.state),
    }


def _read_token(value: object) -> NullTransformToken:
    record = _object(value, _TOKEN_KEYS)
    return NullTransformToken(
        schema=cast(Literal["selcal.null-transform-token.v2"], _string(record["schema"])),
        null_name=_string(record["null_name"]),
        null_parameter_sha256=_string(record["null_parameter_sha256"]),
        semantic_input_sha256=_string(record["semantic_input_sha256"]),
        scientific_plan_sha256=_string(record["scientific_plan_sha256"]),
        bound_null_owner_sha256=_string(record["bound_null_owner_sha256"]),
        is_identity=_boolean(record["is_identity"]),
        state=_read_state(record["state"]),
    )


def _project_outcome(value: object) -> dict[str, object]:
    record = _exact(value, ReplicateOutcome)
    return {
        "replicate_id": _integer(record.replicate_id),
        "seed_digest_sha256": _string(record.seed_digest_sha256),
        "status": _enum_value(record.status, ReplicateStatus),
        "failure_stage": _optional(
            record.failure_stage, lambda item: _enum_value(item, ReplicateFailureStage)
        ),
        "transform_token": _project_token(record.transform_token),
        "statistic_results": _tuple(record.statistic_results, _project_statistic),
        "selection": _optional(record.selection, _project_selection),
        "diagnostics": _tuple(record.diagnostics, _string),
    }


def _encode_replicate_outcome(outcome: ReplicateOutcome) -> bytes:
    """Project a live outcome; these bytes confer no execution authority."""
    return canonical_json_bytes(cast(JsonValue, _project_outcome(outcome))) + b"\n"


def _read_outcome(value: object) -> ReplicateOutcome:
    record = _object(value, _OUTCOME_KEYS)
    return ReplicateOutcome(
        replicate_id=_integer(record["replicate_id"]),
        seed_digest_sha256=_string(record["seed_digest_sha256"]),
        status=_read_enum(record["status"], ReplicateStatus),
        failure_stage=_optional(
            record["failure_stage"], lambda item: _read_enum(item, ReplicateFailureStage)
        ),
        transform_token=_read_token(record["transform_token"]),
        statistic_results=_array(record["statistic_results"], _read_statistic),
        selection=_optional(record["selection"], _read_selection),
        diagnostics=_array(record["diagnostics"], _string),
    )


def _project_result(value: object) -> dict[str, object]:
    record = _exact(value, CalibrationResult)
    return {
        "schema": _SCHEMA,
        "status": _enum_value(record.status, RunStatus),
        "failure_stage": _optional(
            record.failure_stage, lambda item: _enum_value(item, RunFailureStage)
        ),
        "semantic_input_sha256": _string(record.semantic_input_sha256),
        "scientific_plan_sha256": _string(record.scientific_plan_sha256),
        "planned_replicates": _integer(record.planned_replicates),
        "alpha": _float(record.alpha),
        "observed_results": _tuple(record.observed_results, _project_statistic),
        "observed_selection": _optional(record.observed_selection, _project_selection),
        "replicates": _tuple(record.replicates, _project_outcome),
        "exceedance_count": _integer(record.exceedance_count),
        "failure_count": _integer(record.failure_count),
        "p_value": _optional(record.p_value, _float),
        "exceedance_bound_low": _optional(record.exceedance_bound_low, _float),
        "exceedance_bound_high": _optional(record.exceedance_bound_high, _float),
        "reject_null": _optional(record.reject_null, _boolean),
        "diagnostics": _tuple(record.diagnostics, _string),
    }


def _read_result(value: object) -> CalibrationResult:
    record = _object(value, _RESULT_KEYS)
    if _string(record["schema"]) != _SCHEMA:
        raise ResultWireError("value", "unsupported result wire schema")
    return CalibrationResult(
        status=_read_enum(record["status"], RunStatus),
        failure_stage=_optional(
            record["failure_stage"], lambda item: _read_enum(item, RunFailureStage)
        ),
        semantic_input_sha256=_string(record["semantic_input_sha256"]),
        scientific_plan_sha256=_string(record["scientific_plan_sha256"]),
        planned_replicates=_integer(record["planned_replicates"]),
        alpha=_read_float(record["alpha"]),
        observed_results=_array(record["observed_results"], _read_statistic),
        observed_selection=_optional(record["observed_selection"], _read_selection),
        replicates=_array(record["replicates"], _read_outcome),
        exceedance_count=_integer(record["exceedance_count"]),
        failure_count=_integer(record["failure_count"]),
        p_value=_optional(record["p_value"], _read_float),
        exceedance_bound_low=_optional(record["exceedance_bound_low"], _read_float),
        exceedance_bound_high=_optional(record["exceedance_bound_high"], _read_float),
        reject_null=_optional(record["reject_null"], _boolean),
        diagnostics=_array(record["diagnostics"], _string),
    )


def _check_limit(max_bytes: int) -> None:
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ResultWireError("invalid_limit", "max_bytes must be a positive built-in integer")


def _check_size(data: bytes, max_bytes: int) -> None:
    if len(data) > max_bytes:
        raise ResultWireError("size_limit", "record exceeds the caller's byte limit")


def _check_depth(data: bytes) -> None:
    # Deepest legal path: result / replicates / outcome / statistic_results /
    # statistic / float tag or diagnostics. Quoted braces carry no nesting.
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
            if depth > 6:
                raise ResultWireError("shape", "record exceeds the closed value graph depth")
        elif byte in (93, 125):
            depth -= 1
            if depth < 0:
                raise ResultWireError("syntax", "invalid record container framing")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ResultWireError("duplicate_key", "record contains a duplicate object key")
        result[key] = value
    return result


def _reject_number(_spelling: str) -> object:
    raise ResultWireError("value", "raw JSON floats and nonfinite constants are forbidden")


def _canonical_result(result: CalibrationResult) -> bytes:
    return canonical_json_bytes(cast(JsonValue, _project_result(result))) + b"\n"


def encode_calibration_result(result: CalibrationResult, *, max_bytes: int) -> bytes:
    """Encode complete result content, retaining every field or rejecting it.

    Exact supported types are projected, then reconstructed to catch altered
    dataclass slots and constructor normalization. No record is truncated.
    The cap is checked after projection; it is not an allocation guarantee.
    """
    _check_limit(max_bytes)
    try:
        data = _canonical_result(result)
    except ResultWireError:
        raise
    except AttributeError as error:
        raise ResultWireError("type", "result has missing required value slots") from error
    except (ValueError, OverflowError) as error:
        raise ResultWireError("value", "result contains an unencodable value") from error
    _check_size(data, max_bytes)
    decode_calibration_result(data, max_bytes=max_bytes)
    return data


def decode_calibration_result(data: bytes, *, max_bytes: int) -> CalibrationResult:
    """Restore canonical, internally consistent, untrusted result content.

    This does not run calibration, bind a real input or plan, authenticate any
    digest, or grant execution ownership. Byte/depth guards precede parsing.
    """
    _check_limit(max_bytes)
    _exact(data, bytes)
    _check_size(data, max_bytes)
    _check_depth(data)
    try:
        text = data.decode("utf-8", errors="strict")
        payload = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_float=_reject_number,
            parse_constant=_reject_number,
        )
    except ResultWireError:
        raise
    except (ValueError, UnicodeError) as error:
        raise ResultWireError("syntax", "record is not valid UTF-8 JSON") from error
    try:
        restored = _read_result(payload)
        canonical = _canonical_result(restored)
    except ResultWireError:
        raise
    except (ValueError, OverflowError, V2IntegrityError) as error:
        raise ResultWireError(
            "value", "record contradicts its scientific value contract"
        ) from error
    if canonical != data:
        raise ResultWireError(
            "noncanonical", "record is not its exact canonical encoding plus one LF"
        )
    return restored
