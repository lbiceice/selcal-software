"""Public file-to-record workflow over the unchanged scientific v2 core.

A terminal record captures input, request and complete result content. It is not
an execution checkpoint, historical authentication or independent validation.
"""

from __future__ import annotations

import errno
import hashlib
import html
import json
import platform
import re
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from functools import cache
from math import ceil, exp, floor, lgamma, log, ulp
from pathlib import Path
from typing import Any

import numpy as np

from selcal import (
    __version__,
    calibrate_selected_family,
    resolve_plan_v2,
    verify_calibration_result,
)
from selcal.calibration_v2 import (
    _MAX_IN_MEMORY_CANDIDATE_EVALUATIONS_V2,
    _MAX_IN_MEMORY_REPLICATES_V2,
    _MAX_IN_MEMORY_TOKEN_STATE_UNITS_V2,
    _MAX_IN_MEMORY_WORK_UNITS_V2,
    _require_in_memory_execution_budget_values,
)
from selcal.canonical_v2 import scientific_plan_v2_sha256
from selcal.contracts_v2 import (
    CalibrationResult,
    PlanRequestV2,
    ResourceLimitError,
    SelCalV2Error,
)
from selcal.inference import null_supports_inference
from selcal.input_resources import INPUT_LIMITS_V1, read_regular_file_snapshot
from selcal.inputs import LoadedInput, load_csv, load_npz
from selcal.resolution_v2 import PlanResolutionV2
from selcal.result_wire import decode_calibration_result, encode_calibration_result
from selcal.workflow_config import (
    WorkflowConfig,
    WorkflowConfigError,
    decode_workflow_config,
    encode_workflow_config,
)
from selcal.workflow_store import (
    RecordStoreError,
    read_method,
    read_record,
    refuse_existing_path,
    write_record,
)

CONFIG_BYTES = 65_536
RECORD_SCHEMAS = ("selcal.workflow-record.v1", "selcal.workflow-record.v2")
PLATFORM_KEYS = frozenset({"system", "machine", "libc", "blas"})
# Decision replay: statistic values may differ by this many units in the last place of
# max(|a|, |b|, 1), i.e. about 1.4e-14 absolute for correlations and relative above 1; every
# other result field must match exactly. Rounding error of a correlation scales with its range,
# not its value: on Linux a correlation of 4.3e-4 differed from macOS by 8e-18, which is 149
# ULPs of the value but under 1 ULP of 1 (docs/status/evidence/linux_20260923/summary.md).
DECISION_REPLAY_MAX_ULP = 64
_TOLERANT_FLOAT_KEYS = frozenset({"estimate", "selection_score", "decision_statistic"})


class WorkflowError(ValueError):
    """A path-free application consistency error with a stable code."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class WorkflowRecord:
    """Checked terminal content, not a live execution or resume capability."""

    config: WorkflowConfig
    result: CalibrationResult
    metadata: dict[str, Any]


def _limit(max_bytes: int) -> None:
    if type(max_bytes) is not int or max_bytes < 1:
        raise WorkflowError("invalid_limit")


def _json(data: object) -> bytes:
    return (
        json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def _blas_identity() -> str:
    try:
        blas = np.show_config(mode="dicts")["Build Dependencies"]["blas"]
        return f"{blas.get('name', 'unknown')} {blas.get('version', 'unknown')}"
    except (AttributeError, KeyError, TypeError, ValueError):
        return "unknown"


@cache
def _platform_identity() -> tuple[tuple[str, str], ...]:
    """Where floating-point results were computed; last bits can differ between platforms."""
    identity = {
        "system": platform.system(),
        "machine": platform.machine(),
        "libc": " ".join(part for part in platform.libc_ver() if part),
        "blas": _blas_identity(),
    }
    return tuple(sorted((key, value[:256]) for key, value in identity.items()))


def _software_identity() -> dict[str, Any]:
    package = Path(__file__).parent
    sources = {}
    for path in sorted(package.rglob("*.py")):
        if path.is_symlink():
            raise WorkflowError("unsupported_source")
        sources[path.relative_to(package).as_posix()] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    if not sources:
        raise WorkflowError("source_unavailable")
    return {
        "selcal_version": __version__,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "source_files": sources,
        "platform": dict(_platform_identity()),
    }


def _load(path: Path, config: WorkflowConfig) -> LoadedInput:
    if config.source_format == "csv":
        assert config.source_column is not None and config.target_column is not None
        return load_csv(
            path,
            source_column=config.source_column,
            target_column=config.target_column,
            candidates=config.request.candidates,
        )
    return load_npz(path, candidates=config.request.candidates)


def _files(
    input_path: str | Path, config_path: str | Path
) -> tuple[WorkflowConfig, PlanResolutionV2, LoadedInput, bytes]:
    config_bytes = read_regular_file_snapshot(
        config_path, raw_limit=CONFIG_BYTES, reason="WORKFLOW_CONFIG_BYTES"
    )
    # Editors such as Windows PowerShell 5.1 and Notepad may save UTF-8 with a byte-order mark.
    # It is ignored in the user's config file; stored records keep their canonical form.
    config = decode_workflow_config(config_bytes.removeprefix(b"\xef\xbb\xbf"))
    resolution = resolve_plan_v2(config.request)
    loaded = _load(Path(input_path), config)
    raw_limit = (
        INPUT_LIMITS_V1.csv_raw_bytes
        if config.source_format == "csv"
        else INPUT_LIMITS_V1.npz_raw_bytes
    )
    raw = read_regular_file_snapshot(input_path, raw_limit=raw_limit, reason="WORKFLOW_INPUT_BYTES")
    if hashlib.sha256(raw).hexdigest() != loaded.raw_input_sha256:
        raise WorkflowError("input_changed")
    return config, resolution, loaded, raw


def _allowed_exceedances(replicates: int, alpha: float) -> int:
    """Count E in 0..B with (1 + E)/(B + 1) <= alpha, evaluated exactly as the calibrator does."""

    count = max(0, min(replicates + 1, floor(alpha * (replicates + 1))))
    while count > 0 and not count / (replicates + 1) <= alpha:
        count -= 1
    while count <= replicates and (count + 1) / (replicates + 1) <= alpha:
        count += 1
    return count


def _binomial_cdf_below(count: int, trials: int, probability: float) -> float:
    """P(X < count) for X ~ Binomial(trials, probability), in log space: O(count), no underflow."""

    if count <= 0:
        return 0.0
    if count > trials or probability <= 0.0:
        return 1.0
    if probability >= 1.0:
        return 0.0
    log_p, log_q = log(probability), log(1.0 - probability)
    head = lgamma(trials + 1)
    terms = [
        head - lgamma(k + 1) - lgamma(trials - k + 1) + k * log_p + (trials - k) * log_q
        for k in range(count)
    ]
    peak = max(terms)
    return min(1.0, exp(peak) * sum(exp(term - peak) for term in terms))


_SCIENTIFIC_ASSUMPTIONS: dict[str, tuple[tuple[str, str], ...]] = {
    "common": (
        (
            "lags_declared_before_seeing_data",
            "The candidate lag set was fixed before inspecting these data; otherwise the "
            "selection-aware p-value does not cover the extra choice.",
        ),
        (
            "series_aligned_and_sampled_as_supplied",
            "Both columns share one time index and sampling interval; SelCal does not align, "
            "resample, detrend or impute.",
        ),
        (
            "not_causal_inference",
            "A small p-value indicates lagged dependence beyond the null, not causation or an "
            "effect size.",
        ),
    ),
    "circular_shift_v2": (
        (
            "circular_shift_exchangeability",
            "Under the null, circular shifts of the source are exchangeable with the observed "
            "alignment: exact for circular series, approximate for stationary series whose ends "
            "join without a large jump; trends and seasonality should be removed beforehand.",
        ),
    ),
    "block_shuffle_v2": (
        (
            "block_exchangeability",
            "Under the null, source blocks are exchangeable: dependence within the source should "
            "be shorter than the block length.",
        ),
    ),
    "lagged_pearson_v1": (
        (
            "linear_dependence_statistic",
            "Pearson correlation detects linear lagged association; nonlinear dependence may be "
            "missed.",
        ),
    ),
    "equal_width_binned_nette_v1": (
        (
            "binned_estimator_adequacy",
            "The binned transfer-entropy estimate depends on the bin count and sample size.",
        ),
    ),
}


def _declared_assumptions(request: PlanRequestV2) -> list[dict[str, str]]:
    null_key = (
        "circular_shift_v2"
        if request.null_name in {"circular_shift_v2", "circular_shift_exact_v1"}
        else request.null_name
    )
    items = (
        *_SCIENTIFIC_ASSUMPTIONS["common"],
        *_SCIENTIFIC_ASSUMPTIONS.get(null_key, ()),
        *_SCIENTIFIC_ASSUMPTIONS.get(request.statistic_name, ()),
    )
    return [{"code": code, "statement": statement} for code, statement in items]


def _resource_budget(request: PlanRequestV2, sample_count: int) -> dict[str, Any]:
    """Apply the executor's own in-memory admission check without running anything."""

    replicates = request.replicates
    candidates = len(request.candidates)
    state_units: int | None = 1
    if request.null_name == "block_shuffle_v2":
        length = request.null_params.get("block_length")
        state_units = (
            sample_count // length
            if type(length) is int and length > 0 and sample_count % length == 0
            else None
        )
    budget: dict[str, Any] = {
        "B": replicates,
        "C": candidates,
        "N": sample_count,
        "S": state_units,
        "caps": {
            "B": _MAX_IN_MEMORY_REPLICATES_V2,
            "BC": _MAX_IN_MEMORY_CANDIDATE_EVALUATIONS_V2,
            "BS": _MAX_IN_MEMORY_TOKEN_STATE_UNITS_V2,
            "work": _MAX_IN_MEMORY_WORK_UNITS_V2,
        },
        "within_caps": None,
    }
    if state_units is None:
        return budget
    try:
        _require_in_memory_execution_budget_values(
            B=replicates, C=candidates, N=sample_count, S=state_units
        )
    except ResourceLimitError:
        budget["within_caps"] = False
    else:
        budget["within_caps"] = True
    return budget


NON_GROUP_NULL = "REFUSE_NON_GROUP_NULL"


def null_validity(request: PlanRequestV2) -> dict[str, Any]:
    """Whether the null's states form a group, the condition for a valid randomization p.

    With p = (1 + E) / (B + 1) the test keeps its level when the reference states are a group
    acting on the data (Hemerik and Goeman 2018). The circular shifts with min_shift = 1 are
    the whole cyclic group. For 1 < min_shift < n/2 the remaining shifts include two
    consecutive ones but not all shifts, so they are not a group, for any statistic and any
    number of candidates. SelCal supports only the complete group and refuses every
    min_shift > 1 as a policy: min_shift = n/2 leaves {0, n/2}, a group, yet stays unsupported
    (R11-03). R10v3 Windows core audit (2026-10-03, ALG-02): one lag with min_shift = 2
    rejected 2/24 phases of an exactly cyclic null and about 6.1% at B = 999. This validity
    refusal is separate from attainability and the resource budget; the attainability override
    does not lift it, and the public calibration API applies the same rule (R11-01).
    """
    if request.null_name == "circular_shift_exact_v1":
        return {"status": "PASS", "basis": "complete_cyclic_group_enumerated"}
    if request.null_name == "circular_shift_v2":
        if null_supports_inference(request.null_name, request.null_params):
            return {"status": "PASS", "basis": "complete_cyclic_group_sampled_with_identity"}
        return {
            "status": NON_GROUP_NULL,
            "basis": "restricted_shifts_unsupported_not_a_group_in_general",
        }
    return {"status": "NOT_ASSESSED", "basis": "no_group_condition_checked_for_this_null"}


def preflight(request: PlanRequestV2, sample_count: int) -> dict[str, Any]:
    """Separate three questions: is the input valid, is the plan executable, what is assumed.

    Called only after the input loaded, so the input level is VALID here; loader failures are
    reported by their own error codes before this point. The resource check runs first so that
    no plan-time arithmetic grows with an oversized replicate count.
    """

    reasons: list[str] = []
    warnings: list[str] = []
    budget = _resource_budget(request, sample_count)
    if budget["within_caps"] is False:
        reasons.append("resource_budget_exceeded")
    elif budget["within_caps"] is None:
        warnings.append("null_state_space_unavailable_expect_not_evaluable")
    validity = null_validity(request)
    if validity["status"] == NON_GROUP_NULL:
        reasons.append(NON_GROUP_NULL)
    attainable = attainability(request, sample_count)
    if attainable["status"].startswith("REFUSE") and attainable["status"] not in reasons:
        reasons.append(attainable["status"])
    elif attainable["status"] == "NOT_ASSESSED_EMPTY_NULL_STATE_SPACE":
        warnings.append("null_state_space_empty_expect_not_evaluable")
    return {
        "input": {"status": "VALID", "sample_count": sample_count},
        "plan": {
            "status": "NOT_EXECUTABLE" if reasons else "EXECUTABLE",
            "reasons": reasons,
            "warnings": warnings,
            "resource_budget": budget,
            "null_validity": validity,
            "attainability": attainable,
        },
        "scientific_assumptions": {
            "status": "DECLARED_NOT_VERIFIED",
            "assumptions": _declared_assumptions(request),
            "note": "SelCal cannot check these from the data; the analyst must justify them.",
        },
    }


def attainability(request: PlanRequestV2, sample_count: int) -> dict[str, Any]:
    """Report the smallest p-value a plan can produce, before any calibration.

    Every comparison mirrors the calibrator: p = (1 + E) / (B + 1) in float arithmetic and
    rejection iff p <= alpha. No run can reach p < 1 / (B + 1). For the circular-shift nulls
    with lagged Pearson, shift s = c* - c maps searched lag c onto the selected lag c*, so those
    states reproduce the observed maximum: for exact enumeration p is at least their share of
    the states; for sampling it is the chance per draw of an unavoidable exceedance, which caps
    power. Only min_shift = 1 keeps the null states a group, with one or several candidates.
    """
    alpha = request.alpha
    replicates = request.replicates
    report: dict[str, Any] = {
        "monte_carlo_p_floor": 1 / (replicates + 1),
        "null_state_count": None,
        "null_state_p_floor": None,
        "monte_carlo_power_cap": None,
        "scope": "plan_arithmetic_not_scientific_validity",
    }
    enumerating = request.null_name == "circular_shift_exact_v1"
    if (
        request.null_name not in {"circular_shift_v2", "circular_shift_exact_v1"}
        or request.statistic_name != "lagged_pearson_v1"
    ):
        report["status"] = "NOT_ASSESSED"
        return report
    min_shift = request.null_params["min_shift"]
    if type(min_shift) is not int:
        raise WorkflowError("invalid_null_parameters")
    if 2 * min_shift > sample_count:
        report["status"] = "NOT_ASSESSED_EMPTY_NULL_STATE_SPACE"
        return report
    states = sample_count - 2 * min_shift + 2
    candidates = request.candidates

    def in_state_space(shift: int) -> bool:
        return shift == 0 or min_shift <= shift <= sample_count - min_shift

    colliding = min(
        sum(in_state_space((selected - lag) % sample_count) for lag in candidates)
        for selected in candidates
    )
    share = Fraction(colliding, states)
    # The same float expression the calibrator evaluates for the smallest reachable E.
    floor_rejects = colliding / states <= alpha
    report.update(null_state_count=states, null_state_p_floor=float(share))
    if enumerating and replicates != states - 1:
        report["status"] = "REFUSE_ENUMERATION_REPLICATE_COUNT"
        return report
    if enumerating:
        report["monte_carlo_power_cap"] = 1.0 if floor_rejects else 0.0
    else:
        # Strongest signal: rejection needs an exceedance count E with (1 + E)/(B + 1) <= alpha,
        # while each draw lands on an unavoidable colliding state with probability `floor`.
        report["monte_carlo_power_cap"] = _binomial_cdf_below(
            _allowed_exceedances(replicates, alpha), replicates, float(share)
        )
    if not 1 / (replicates + 1) <= alpha:
        report["status"] = "REFUSE_REPLICATES_TOO_FEW"
    elif min_shift > 1:
        report["status"] = NON_GROUP_NULL
    elif not floor_rejects:
        report["status"] = "REFUSE_NULL_STATES_TOO_FEW"
    else:
        report["status"] = "PASS"
    return report


def validate_files(input_path: str | Path, config_path: str | Path) -> dict[str, Any]:
    """Validate actual input/configuration without running surrogate calibration."""
    config, resolution, loaded, _ = _files(input_path, config_path)
    sample_count = int(loaded.pair.source.size)
    return {
        "sample_count": sample_count,
        "planned_replicates": config.request.replicates,
        "raw_input_sha256": loaded.raw_input_sha256,
        "semantic_input_sha256": loaded.semantic_input_sha256,
        "scientific_plan_sha256": scientific_plan_v2_sha256(resolution.plan),
        "attainability": attainability(config.request, sample_count),
        "preflight": preflight(config.request, sample_count),
        "validation_scope": "input_and_plan_not_execution_admission_or_scientific_validity",
    }


def run_files(
    input_path: str | Path,
    config_path: str | Path,
    output_path: str | Path,
    *,
    max_bytes: int,
    allow_unattainable: bool = False,
) -> CalibrationResult:
    """Run a real calibration and exclusively save a complete terminal record.

    Computation uses the existing in-memory admission limits. max_bytes separately
    limits the record; it does not promise a process memory bound or early pause.
    Plans refused by `attainability` run only with allow_unattainable=True.
    """
    _limit(max_bytes)
    if type(allow_unattainable) is not bool:
        raise WorkflowError("invalid_attainability_option")
    output = Path(output_path)
    if output.exists() or output.is_symlink():
        raise FileExistsError(errno.EEXIST, "output already exists", str(output))
    if not output.parent.exists():
        raise FileNotFoundError(
            errno.ENOENT, "output parent directory does not exist", str(output.parent)
        )
    if not output.parent.is_dir():
        raise NotADirectoryError(
            errno.ENOTDIR, "output parent is not a directory", str(output.parent)
        )
    config, resolution, loaded, raw = _files(input_path, config_path)
    plan = preflight(config.request, int(loaded.pair.source.size))["plan"]
    if "resource_budget_exceeded" in plan["reasons"]:
        raise WorkflowConfigError(
            "resource_budget_exceeded",
            f"plan exceeds the in-memory execution budget: {plan['resource_budget']}",
        )
    if NON_GROUP_NULL in plan["reasons"]:
        raise WorkflowConfigError(
            "invalid_null_for_inference",
            "min_shift > 1 is not supported for inference; use min_shift = 1 (the full group)",
        )
    refusals = [reason for reason in plan["reasons"] if reason.startswith("REFUSE")]
    if refusals and not allow_unattainable:
        raise WorkflowConfigError(
            "unattainable_plan", f"plan cannot reach alpha on this input: {refusals[0]}"
        )
    software = _software_identity()
    result = calibrate_selected_family(loaded.pair, resolution)
    verify_calibration_result(result, resolution)
    if software != _software_identity():
        raise WorkflowError("source_changed")
    metadata = {
        "schema": "selcal.workflow-record.v2",
        "raw_input_sha256": loaded.raw_input_sha256,
        "semantic_input_sha256": result.semantic_input_sha256,
        "scientific_plan_sha256": result.scientific_plan_sha256,
        "software": software,
    }
    write_record(
        output_path,
        {
            "input": raw,
            "request": encode_workflow_config(config),
            "result": encode_calibration_result(result, max_bytes=max_bytes),
            "metadata": _json(metadata),
        },
        max_bytes=max_bytes,
    )
    return result


def _metadata(raw: bytes) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise WorkflowError("invalid_metadata")
            result[key] = value
        return result

    def forbidden(value: str) -> None:
        raise WorkflowError("invalid_metadata")

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=unique, parse_constant=forbidden)
        keys = {
            "schema",
            "raw_input_sha256",
            "semantic_input_sha256",
            "scientific_plan_sha256",
            "software",
        }
        if type(value) is not dict or set(value) != keys:
            raise WorkflowError("invalid_metadata")
        if value["schema"] not in RECORD_SCHEMAS:
            raise WorkflowError("invalid_metadata")
        for key in keys - {"schema", "software"}:
            if type(value[key]) is not str or re.fullmatch("[0-9a-f]{64}", value[key]) is None:
                raise WorkflowError("invalid_metadata")
        software = value["software"]
        software_keys = {"selcal_version", "python_version", "numpy_version", "source_files"}
        if value["schema"] == "selcal.workflow-record.v2":
            software_keys.add("platform")
        if type(software) is not dict or set(software) != software_keys:
            raise WorkflowError("invalid_metadata")
        if "platform" in software:
            identity = software["platform"]
            if type(identity) is not dict or set(identity) != PLATFORM_KEYS:
                raise WorkflowError("invalid_metadata")
            if any(type(item) is not str or len(item) > 256 for item in identity.values()):
                raise WorkflowError("invalid_metadata")
        for key in ("selcal_version", "python_version", "numpy_version"):
            if type(software[key]) is not str or not software[key] or len(software[key]) > 256:
                raise WorkflowError("invalid_metadata")
        sources = software["source_files"]
        if type(sources) is not dict or not sources:
            raise WorkflowError("invalid_metadata")
        for name, digest in sources.items():
            if (
                type(name) is not str
                or not re.fullmatch(r"[A-Za-z0-9_/]+\.py", name)
                or name.startswith("/")
                or "//" in name
                or type(digest) is not str
                or re.fullmatch("[0-9a-f]{64}", digest) is None
            ):
                raise WorkflowError("invalid_metadata")
        return value
    except (ValueError, TypeError, RecursionError) as error:
        raise WorkflowError("invalid_metadata") from error


# Earlier Pearson numeric methods: before ALG-01 (R6/R10 records) and the R11 two-pass method
# before correctly rounded sums (R11-02). Such records stay readable: structural verification
# substitutes the current identity, only when the record's own Pearson source differs from the
# current one and no row already carries the current identity. The record bytes and the
# returned result are unchanged, and replay still requires the recorded code.
_PRIOR_PEARSON_PREPROCESSING = (
    "no_hidden_transform|formula_centered_after_max_abs_scaling|common_support_max_lag",
    "no_hidden_transform|formula_two_pass_centered_after_power_of_two_scaling"
    "|common_support_max_lag",
)
_PEARSON_SOURCE = "statistics/lagged_pearson.py"


def _structurally_checkable(raw: bytes, metadata: dict[str, Any], max_bytes: int) -> bytes:
    encoded = (json.dumps(identity).encode() for identity in _PRIOR_PEARSON_PREPROCESSING)
    present = [prior for prior in encoded if prior in raw]
    if not present:
        return raw
    from selcal.statistics.lagged_pearson import _PREPROCESSING_IDENTITY as current_identity

    current = json.dumps(current_identity).encode()
    recorded = metadata["software"].get("source_files", {}).get(_PEARSON_SOURCE)
    actual = hashlib.sha256((Path(__file__).parent / _PEARSON_SOURCE).read_bytes()).hexdigest()
    if len(present) > 1 or current in raw or recorded is None or recorded == actual:
        raise WorkflowError("identity_mismatch")
    return raw.replace(present[0], current)


def _read_context(
    record_path: str | Path, max_bytes: int
) -> tuple[WorkflowRecord, LoadedInput, PlanResolutionV2, bytes]:
    _limit(max_bytes)
    members = read_record(record_path, max_bytes=max_bytes)
    return _context_from_members(members, max_bytes)


def _context_from_members(
    members: dict[str, bytes], max_bytes: int
) -> tuple[WorkflowRecord, LoadedInput, PlanResolutionV2, bytes]:
    """Validate one captured four-member snapshot without reopening its source."""
    _limit(max_bytes)
    if (
        type(members) is not dict
        or members.keys() != {"input", "request", "result", "metadata"}
        or any(type(value) is not bytes or not value for value in members.values())
    ):
        raise WorkflowError("invalid_record_members")
    if sum(map(len, members.values())) > max_bytes:
        raise WorkflowError("record_size_limit")
    try:
        config = decode_workflow_config(members["request"])
        metadata = _metadata(members["metadata"])
        result = decode_calibration_result(members["result"], max_bytes=max_bytes)
        resolution = resolve_plan_v2(config.request)
        # Reuse the same strict file loader; never consult the original input path.
        with tempfile.TemporaryDirectory(prefix="selcal-read-") as folder:
            path = Path(folder) / ("input." + config.source_format)
            path.write_bytes(members["input"])
            loaded = _load(path, config)
        if (
            metadata["raw_input_sha256"] != loaded.raw_input_sha256
            or metadata["semantic_input_sha256"] != loaded.semantic_input_sha256
            or result.semantic_input_sha256 != loaded.semantic_input_sha256
            or metadata["scientific_plan_sha256"] != result.scientific_plan_sha256
        ):
            raise WorkflowError("identity_mismatch")
        checkable = _structurally_checkable(members["result"], metadata, max_bytes)
        if checkable is not members["result"]:
            verify_calibration_result(
                decode_calibration_result(checkable, max_bytes=max_bytes), resolution
            )
        else:
            verify_calibration_result(result, resolution)
    except (ValueError, TypeError, SelCalV2Error) as error:
        raise WorkflowError("invalid_record_content") from error
    return WorkflowRecord(config, result, metadata), loaded, resolution, members["result"]


def read_workflow(record_path: str | Path, *, max_bytes: int) -> WorkflowRecord:
    """Read a captured terminal and verify input/plan/result consistency, not replay."""
    return _read_context(record_path, max_bytes)[0]


def result_summary(result: CalibrationResult) -> dict[str, Any]:
    """Return explicit scientific status and decision fields without dropping failures."""
    selection = result.observed_selection
    return {
        "status": result.status.value,
        "failure_stage": None if result.failure_stage is None else result.failure_stage.value,
        "planned_replicates": result.planned_replicates,
        "retained_replicates": len(result.replicates),
        "exceedance_count": result.exceedance_count,
        "failure_count": result.failure_count,
        "p_value": result.p_value,
        "selected_candidate": None if selection is None else selection.selected_candidate,
        "decision_statistic": None if selection is None else selection.decision_statistic,
        "tied_candidates": None if selection is None else list(selection.tied_candidates),
        "reject_null": result.reject_null,
        "exceedance_bound_low": result.exceedance_bound_low,
        "exceedance_bound_high": result.exceedance_bound_high,
        "semantic_input_sha256": result.semantic_input_sha256,
        "scientific_plan_sha256": result.scientific_plan_sha256,
    }


def _scaled_ulp_gap(left: float, right: float) -> int:
    """|left - right| in units of the last place of max(|left|, |right|, 1), rounded up."""
    return ceil(abs(left - right) / ulp(max(abs(left), abs(right), 1.0)))


def decision_replay_ulp(saved: bytes, actual: bytes) -> int:
    """Compare two encoded results for decision replay; return the largest statistic ULP gap.

    Seeds, transform states, selections, counts, p-values, decisions and diagnostics must be
    equal. Statistic values (estimate, selection score, decision statistic) may differ by at
    most DECISION_REPLAY_MAX_ULP units in the last place of max(|a|, |b|, 1). A NumPy version
    inside a statistic
    backend identity may differ. Anything else raises decision_replay_mismatch.
    """
    worst = 0

    def mismatch() -> WorkflowError:
        return WorkflowError("decision_replay_mismatch")

    def walk(left: object, right: object, key: str | None) -> None:
        nonlocal worst
        if type(left) is not type(right):
            raise mismatch()
        if isinstance(left, dict):
            assert isinstance(right, dict)
            if set(left) != set(right):
                raise mismatch()
            if set(left) == {"$float64"}:
                a, b = float.fromhex(left["$float64"]), float.fromhex(right["$float64"])
                if key in _TOLERANT_FLOAT_KEYS:
                    gap = _scaled_ulp_gap(a, b)
                    if gap > DECISION_REPLAY_MAX_ULP:
                        raise mismatch()
                    worst = max(worst, gap)
                elif left["$float64"] != right["$float64"]:
                    raise mismatch()
                return
            for name in left:
                walk(left[name], right[name], name)
        elif isinstance(left, list):
            assert isinstance(right, list)
            if len(left) != len(right):
                raise mismatch()
            for a, b in zip(left, right, strict=True):
                walk(a, b, key)
        elif key == "backend_identity":
            pattern = r"numpy=[^|]*"
            if re.sub(pattern, "numpy=*", str(left)) != re.sub(pattern, "numpy=*", str(right)):
                raise mismatch()
        elif left != right:
            raise mismatch()

    try:
        walk(json.loads(saved), json.loads(actual), None)
    except (ValueError, TypeError, KeyError, RecursionError) as error:
        raise mismatch() from error
    return worst


def _environment(software: dict[str, Any]) -> dict[str, Any]:
    return {
        "python_version": software["python_version"],
        "numpy_version": software["numpy_version"],
        "platform": software.get("platform"),
    }


def verify_record(
    record_path: str | Path,
    *,
    max_bytes: int,
    replay: bool = False,
    replay_decision: bool = False,
) -> dict[str, Any]:
    """Optionally replay exactly or at decision level; never authenticate past execution.

    Exact replay needs the recorded code, Python, NumPy and platform, and compares result bytes.
    Decision replay needs the recorded code only; see decision_replay_ulp for what it compares.
    """
    if (
        type(replay) is not bool
        or type(replay_decision) is not bool
        or (replay and replay_decision)
    ):
        raise WorkflowError("invalid_replay_option")
    record, loaded, resolution, saved = _read_context(record_path, max_bytes)
    decision: dict[str, Any] | None = None
    if replay:
        if record.metadata["software"] != _software_identity():
            raise WorkflowError("environment_mismatch")
        actual = calibrate_selected_family(loaded.pair, resolution)
        if encode_calibration_result(actual, max_bytes=max_bytes) != saved:
            raise WorkflowError("replay_mismatch")
    if replay_decision:
        recorded, current = record.metadata["software"], _software_identity()
        if any(recorded[key] != current[key] for key in ("selcal_version", "source_files")):
            raise WorkflowError("environment_mismatch")
        actual = calibrate_selected_family(loaded.pair, resolution)
        gap = decision_replay_ulp(saved, encode_calibration_result(actual, max_bytes=max_bytes))
        decision = {
            "max_statistic_ulp": gap,
            "tolerance_ulp": DECISION_REPLAY_MAX_ULP,
            "same_environment": recorded == current,
            "recorded_environment": _environment(recorded),
            "current_environment": _environment(current),
        }
    summary = {
        **result_summary(record.result),
        "attainability": attainability(record.config.request, int(loaded.pair.source.size)),
        "replay": "MATCH" if replay else "DECISION_MATCH" if replay_decision else "NOT_PERFORMED",
        "verification_scope": "input_plan_result_consistency",
        "historical_execution_authenticated": False,
    }
    if decision is not None:
        summary["decision_replay"] = decision
    return summary


def report_record(
    record_path: str | Path, output_path: str | Path, *, max_bytes: int
) -> dict[str, Any]:
    """Render an exclusive HTML report from checked content, without scientific replay."""
    record, loaded, _, _ = _read_context(record_path, max_bytes)
    result = record.result
    summary = {
        **result_summary(result),
        "attainability": attainability(record.config.request, int(loaded.pair.source.size)),
        "replay": "NOT_PERFORMED",
    }
    payload = _render_report(record, summary, max_bytes)
    refuse_existing_path(output_path)
    with Path(output_path).open("xb") as stream:
        stream.write(payload)
    return summary


def _render_report(record: WorkflowRecord, summary: dict[str, Any], max_bytes: int) -> bytes:
    """Render already checked content, stopping accumulation at each bounded section."""
    _limit(max_bytes)
    result = record.result

    def sections() -> Iterator[str]:
        yield (
            '<!doctype html><html lang="en"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
        )
        yield (
            "<title>SelCal calibration report</title><style>"
            "body{max-width:960px;margin:2rem auto;padding:0 1rem;"
            "font:16px system-ui;color:#172b4d;overflow-wrap:anywhere}"
            "pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f5f7;padding:1rem}"
            "section{border-top:1px solid #ccd3dc}h2{margin-top:2rem}"
            "dl{display:grid;grid-template-columns:repeat(auto-fit,minmax(14rem,1fr));gap:1rem}"
            "dl>div{min-width:0;background:#f3f5f7;padding:1rem}"
            "dt{font-weight:600}dd{margin:.5rem 0 0}"
            "details{border-top:1px solid #ccd3dc;padding:1rem 0}"
            "summary{cursor:pointer;font-weight:600}"
            "</style><body><h1>SelCal calibration report</h1>"
        )
        yield (
            "<p>Input/plan/result consistency checked; replay NOT_PERFORMED. "
            "Recorded content is not authenticated historical execution.</p>"
        )
        yield "<p>NOT_EVALUABLE is not evidence of no effect or non-significance.</p>"
        yield "<h2>Result summary</h2><dl>"
        for key in (
            "status",
            "selected_candidate",
            "decision_statistic",
            "p_value",
            "reject_null",
            "planned_replicates",
            "retained_replicates",
            "failure_count",
            "failure_stage",
            "exceedance_count",
            "exceedance_bound_low",
            "exceedance_bound_high",
        ):
            if key in summary:
                yield (
                    f'<div><dt>{key}</dt><dd data-summary-key="{key}">'
                    f"{html.escape(json.dumps(summary[key]))}</dd></div>"
                )
        yield "</dl><h2>Recorded details</h2><p>Expand a section to inspect its full content.</p>"
        yield "<details><summary>Summary</summary><pre>"
        yield html.escape(json.dumps(summary, indent=2))
        yield "</pre></details><details><summary>Configuration</summary><pre>"
        yield html.escape(encode_workflow_config(record.config).decode())
        yield "</pre></details><details><summary>Observed candidates</summary><pre>"
        yield html.escape(repr(result.observed_results))
        yield "</pre></details><details><summary>Observed selection</summary><pre>"
        yield html.escape(repr(result.observed_selection))
        yield "</pre></details><details><summary>Replicates</summary>"
        for outcome in result.replicates:
            yield (
                f'<section data-replicate-id="{outcome.replicate_id}"><h3>Replicate '
                f"{outcome.replicate_id}</h3><pre>"
            )
            yield html.escape(repr(outcome))
            yield "</pre></section>"
        yield "</details><details><summary>Diagnostics</summary><pre>"
        yield html.escape(repr(result.diagnostics))
        yield (
            "</pre></details><details>"
            "<summary>Recorded software identity (not authentication)</summary><pre>"
        )
        yield html.escape(json.dumps(record.metadata["software"], indent=2))
        yield "</pre></details></body></html>"

    body = bytearray()
    for section in sections():
        payload = section.encode("utf-8")
        if len(body) + bool(body) + len(payload) > max_bytes:
            raise WorkflowError("report_size_limit")
        if body:
            body.extend(b"\n")
        body.extend(payload)
    return bytes(body)


def _record_round_trip(folder: str | Path | None = None) -> str:
    """Write a small record in a fresh subfolder of ``folder`` (default: temp) and read it back."""
    members = {name: name.encode() for name in ("input", "request", "result", "metadata")}
    try:
        with tempfile.TemporaryDirectory(prefix="selcal-doctor-", dir=folder) as folder:
            path = Path(folder) / "check.sqlite"
            write_record(path, members, max_bytes=CONFIG_BYTES)
            if read_record(path, max_bytes=CONFIG_BYTES) != members:
                return "FAIL:content_mismatch"
    except (RecordStoreError, OSError) as error:
        return f"FAIL:{getattr(error, 'code', type(error).__name__)}"
    return "PASS"


def doctor(folder: str | Path | None = None) -> dict[str, Any]:
    """Report this runtime and whether records can be written and read back.

    ``record_round_trip`` is an executed check: a record is saved and read back in a new
    subfolder of ``folder`` (default: the system temporary folder), which is then removed.
    ``PASS`` means that ``run`` can save and ``verify``/``report`` can read records in that
    location. Check the folder where records will be kept (for example a synchronised or
    network folder) with ``folder``. It does not claim that the test suite passed or that
    results are scientifically valid.
    """
    import sqlite3

    return {
        "selcal_version": __version__,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "platform": dict(_platform_identity()),
        "sqlite_version": sqlite3.sqlite_version,
        "sqlite_deserialize": hasattr(sqlite3.Connection, "deserialize"),
        "record_read_method": read_method(),
        "record_round_trip": _record_round_trip(folder),
        "record_round_trip_folder": "system temporary folder" if folder is None else str(folder),
        "checkpoint_resume": "SAME_ENVIRONMENT_LOCAL_REPLAY_BEFORE_CONTINUE",
        "scientific_validation": "NOT_EXECUTED",
    }
