"""Internal progress bytes are observations, not saved execution authority."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
import pytest

import selcal
import selcal.calibration_v2 as kernel
import selcal.result_wire as wire
from selcal import inference
from selcal.contracts import ReplicateStatus, RunStatus, SeriesPair
from selcal.contracts_v2 import PlanRequestV2, RunFailureStage, V2IntegrityError
from selcal.resolution_v2 import resolve_plan_v2

# v3: correctly rounded Pearson sums; bytes matched on the BLAS libraries tested (R11-02).
# v1 (before ALG-01) and v2 (R11 two-pass with BLAS sums) stay unchanged as history;
# test_reference_change_is_bounded binds what each numeric-method change may alter.
FIXTURES = Path(__file__).resolve().parent / "fixtures"
REFERENCES = FIXTURES / "recovery_kernel_v3"
CASES = (
    "sampled_pearson",
    "exact_pearson",
    "block_pearson",
    "binned",
    "pre_observed_ne",
    "null_bind_ne",
    "replicate_ne",
)


def _runtime_reference_environment():
    blas = np.show_config(mode="dicts")["Build Dependencies"]["blas"]
    return {
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "byteorder": sys.byteorder,
        "blas": {
            key: blas[key]
            for key in ("name", "found", "version", "detection method", "openblas configuration")
        },
    }


def _numpy_backend_token_projection(payload):
    """Four-fixture diagnostic only: no numerical or recursive field normalization."""
    document = json.loads(payload)
    rows = list(document["observed_results"])
    for replicate in document["replicates"]:
        rows.extend(replicate["statistic_results"])
    for statistic in rows:
        parts = statistic["backend_identity"].split("|")
        assert sum(part.startswith("numpy=") for part in parts) == 1
        statistic["backend_identity"] = "|".join(
            "numpy=<version>" if part.startswith("numpy=") else part for part in parts
        )
    return _dump_reference(document)


def _assert_reference_match(saved, actual, verified_environment, actual_environment):
    if verified_environment == actual_environment:
        assert saved == actual, "exact reference bytes differ in verified environment"
        return "verified_environment_full_bytes"
    assert _numpy_backend_token_projection(saved) == _numpy_backend_token_projection(actual), (
        "non-NumPy reference content differs across environments"
    )
    return "cross_environment_numpy_token_only_diagnostic"


def _case(name: str, rule: str = "max_upper"):
    """Small real fixtures reused from the existing failure and wire tests."""
    source = [0, 1, 4, 2, 5, 3]
    target = [4, 1, 3, 0, 5, 2]
    candidates = (1, 2)
    statistic_name = "lagged_pearson_v1"
    statistic_params = {}
    null_name = "circular_shift_v2"
    null_params = {"min_shift": 1}
    replicates = 9
    if name == "exact_pearson":
        null_name = "circular_shift_exact_v1"
        replicates = len(source) - 1
    elif name in {"block_pearson", "binned"}:
        null_name = "block_shuffle_v2"
        null_params = {"block_length": 2}
        if name == "binned":
            source = [0, 1, 0, 1, 2, 1, 2, 0, 2, 1]
            target = [1, 0, 1, 2, 1, 0, 2, 1, 2, 0]
            statistic_name = "equal_width_binned_nette_v1"
            statistic_params = {"bins": 3}
            replicates = 5
    elif name == "pre_observed_ne":
        source = [0] * 6
    elif name == "null_bind_ne":
        source = [0, 1, 2, 3, 4]
        target = [4, 3, 2, 1, 0]
        candidates = (1, 3)
        null_params = {"min_shift": 3}
        replicates = 7
    elif name == "replicate_ne":
        source = [0, 0, 1, 2, 0, 0]
        target = [0, 1, 2, 3, 4, 5]
    else:
        assert name == "sampled_pearson"
    pair = SeriesPair(np.asarray(source, dtype=np.float64), np.asarray(target, dtype=np.float64))
    request = PlanRequestV2(
        candidates=candidates,
        statistic_name=statistic_name,
        statistic_params=statistic_params,
        selection_rule=rule,
        null_name=null_name,
        null_params=null_params,
        replicates=replicates,
        alpha=0.05,
        tie_tolerance=1e-12,
        root_seed=17,
    )
    return pair, request


def _observed():
    observed = getattr(kernel, "_calibrate_selected_family_observed", None)
    assert callable(observed), "private shared-kernel observer is missing"
    return observed


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("rule", ["max_upper", "max_absolute"])
def test_observed_stream_matches_complete_final_content(name, rule):
    observed = _observed()
    pair, request = _case(name, rule)
    resolution = resolve_plan_v2(request)
    events = []

    def collect(index, payload):
        assert type(index) is int
        assert type(payload) is bytes
        assert payload.endswith(b"\n") and not payload.endswith(b"\n\n")
        assert (
            payload
            == json.dumps(
                json.loads(payload),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
        with pytest.raises(TypeError):
            payload[0] = 0
        events.append((index, payload))

    actual = observed(pair, resolution, collect)
    ordinary = kernel.calibrate_selected_family(pair, resolution)
    encoded = wire.encode_calibration_result(actual, max_bytes=1_000_000)
    assert encoded == wire.encode_calibration_result(ordinary, max_bytes=1_000_000)
    assert [index for index, _ in events] == list(range(len(actual.replicates)))
    assert [json.loads(payload) for _, payload in events] == json.loads(encoded)["replicates"]
    if name in {"pre_observed_ne", "null_bind_ne"}:
        assert events == []
        assert actual.status is RunStatus.NOT_EVALUABLE
        assert actual.failure_stage is (
            RunFailureStage.OBSERVED_STATISTIC_SCAN
            if name == "pre_observed_ne"
            else RunFailureStage.NULL_BIND
        )
    else:
        assert len(events) == request.replicates
        if name == "replicate_ne":
            assert actual.status is RunStatus.NOT_EVALUABLE
            assert actual.failure_stage is RunFailureStage.REPLICATE_EXECUTION
            assert actual.failure_count > 0
            assert any(row.status is ReplicateStatus.ANALYTIC_FAILURE for row in actual.replicates)
        else:
            assert actual.status is RunStatus.COMPLETE
            assert actual.failure_count == 0


@pytest.mark.parametrize("name", ["sampled_pearson", "exact_pearson", "binned", "pre_observed_ne"])
def test_prechange_reference_bytes_match_both_paths(name):
    observed = _observed()
    manifest = json.loads((REFERENCES / "manifest.json").read_text(encoding="utf-8"))
    reference = manifest["cases"][name]
    payload = (REFERENCES / reference["file"]).read_bytes()
    assert hashlib.sha256(payload).hexdigest() == reference["sha256"]
    pair, request = _case(name, reference["request"]["selection_rule"])
    assert pair.source.tolist() == reference["source"]
    assert pair.target.tolist() == reference["target"]
    resolution = resolve_plan_v2(request)
    ordinary = kernel.calibrate_selected_family(pair, resolution)
    assert ordinary.semantic_input_sha256 == reference["semantic_input_sha256"]
    assert ordinary.scientific_plan_sha256 == reference["scientific_plan_sha256"]
    ordinary_bytes = wire.encode_calibration_result(ordinary, max_bytes=1_000_000)
    assert (
        wire.encode_calibration_result(
            observed(pair, resolution, lambda index, payload: None), max_bytes=1_000_000
        )
        == ordinary_bytes
    )
    verified = json.loads((REFERENCES / "verified_environment.json").read_text(encoding="utf-8"))
    assert verified["purpose"] == "reference full-byte reproduction verified environment"
    assert verified["not_original_generation_blas_attestation"] is True
    assert verified["scientific_validation"] is False
    verified_case = next(row for row in verified["cases"] if row["case"] == name)
    assert verified_case["sha256"] == reference["sha256"]
    assert verified_case["full_byte_reproduction"] is True
    _assert_reference_match(
        payload, ordinary_bytes, verified["environment"], _runtime_reference_environment()
    )


def _floats_and_fields(document, path=""):
    if isinstance(document, dict) and set(document) == {"$float64"}:
        yield path, float.fromhex(document["$float64"])
    elif isinstance(document, dict):
        for key, value in document.items():
            yield from _floats_and_fields(value, f"{path}/{key}")
    elif isinstance(document, list):
        for index, value in enumerate(document):
            yield from _floats_and_fields(value, f"{path}[{index}]")
    else:
        yield path, document


IDENTITY_V1 = "no_hidden_transform|formula_centered_after_max_abs_scaling|common_support_max_lag"
IDENTITY_V2 = (
    "no_hidden_transform|formula_two_pass_centered_after_power_of_two_scaling"
    "|common_support_max_lag"
)
IDENTITY_V3 = (
    "no_hidden_transform|formula_two_pass_centered_after_power_of_two_scaling"
    "|correctly_rounded_sums|common_support_max_lag"
)


@pytest.mark.parametrize(
    ("before_dir", "after_dir", "before_identity", "after_identity", "max_ulp"),
    [
        ("recovery_kernel_v1", "recovery_kernel_v2", IDENTITY_V1, IDENTITY_V2, 1),
        ("recovery_kernel_v2", "recovery_kernel_v3", IDENTITY_V2, IDENTITY_V3, 2),
    ],
)
@pytest.mark.parametrize("name", ["sampled_pearson", "exact_pearson", "binned", "pre_observed_ne"])
def test_reference_change_is_bounded(
    name, before_dir, after_dir, before_identity, after_identity, max_ulp
):
    """A Pearson numeric-method change may move values by a few scaled ULP and rename the
    preprocessing identity; p, E, decisions, status and every other field stay equal."""
    old_manifest = json.loads((FIXTURES / before_dir / "manifest.json").read_text(encoding="utf-8"))
    new_manifest = json.loads((FIXTURES / after_dir / "manifest.json").read_text(encoding="utf-8"))
    assert old_manifest["cases"][name]["request"] == new_manifest["cases"][name]["request"]
    old = json.loads((FIXTURES / before_dir / old_manifest["cases"][name]["file"]).read_bytes())
    new = json.loads((FIXTURES / after_dir / new_manifest["cases"][name]["file"]).read_bytes())
    for key in ("p_value", "exceedance_count", "reject_null", "status", "failure_count"):
        assert old[key] == new[key], key
    old_rows, new_rows = list(_floats_and_fields(old)), list(_floats_and_fields(new))
    assert [path for path, _ in old_rows] == [path for path, _ in new_rows]
    # Only Pearson statistic values may move (R12 review: whitelist the numeric fields); alpha,
    # p-values, bounds and every other float must stay bit-identical.
    statistic_fields = ("/estimate", "/selection_score", "/decision_statistic")
    for (path, before), (_, after) in zip(old_rows, new_rows, strict=True):
        if isinstance(before, float) and path.endswith(statistic_fields):
            gap = abs(after - before) / math.ulp(max(abs(before), abs(after), 1.0))
            assert gap <= max_ulp, (path, before, after)
        elif isinstance(before, float):
            assert before == after, (path, before, after)
        elif before != after:
            assert path.endswith("/preprocessing_identity"), path
            assert (before, after) == (before_identity, after_identity)
            assert name != "binned"


def test_public_signature_and_allowlist_are_unchanged():
    # R11-01: the public name is the admission-checked entry over the unchanged kernel.
    assert selcal.calibrate_selected_family is inference.calibrate_selected_family
    assert selcal.calibrate_selected_family is not kernel.calibrate_selected_family
    for entry in (kernel.calibrate_selected_family, inference.calibrate_selected_family):
        parameters = tuple(inspect.signature(entry).parameters.values())
        assert tuple(parameter.name for parameter in parameters) == ("pair", "resolution")
        assert all(parameter.kind is inspect.Parameter.POSITIONAL_ONLY for parameter in parameters)
        assert all(parameter.default is inspect.Parameter.empty for parameter in parameters)
    assert "_calibrate_selected_family_observed" not in kernel.__all__
    assert not hasattr(selcal, "_calibrate_selected_family_observed")


def _reference_matcher():
    matcher = globals().get("_assert_reference_match")
    assert callable(matcher), "explicit environment-selected reference comparator is missing"
    return matcher


def _dump_reference(data):
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode() + b"\n"


@pytest.mark.parametrize("change", ["numpy_token", "one_ulp", "whitespace"])
def test_exact_reference_environment_never_falls_back(change):
    match = _reference_matcher()
    saved = (REFERENCES / "sampled_pearson.json").read_bytes()
    data = json.loads(saved)
    statistic = data["observed_results"][0]
    if change == "numpy_token":
        statistic["backend_identity"] = statistic["backend_identity"].replace(
            "numpy=", "numpy=old-"
        )
    elif change == "one_ulp":
        value = float.fromhex(statistic["estimate"]["$float64"])
        statistic["estimate"]["$float64"] = math.nextafter(value, math.inf).hex()
    actual = saved + b" " if change == "whitespace" else _dump_reference(data)
    with pytest.raises(AssertionError, match="exact reference bytes"):
        match(saved, actual, {"environment": "same"}, {"environment": "same"})


@pytest.mark.parametrize(
    "change",
    [
        "one_ulp",
        "signed_zero",
        "seed",
        "lag",
        "count",
        "other_backend",
        "diagnostics",
    ],
)
def test_cross_environment_reference_rejects_every_non_numpy_change(change):
    match = _reference_matcher()
    saved = (REFERENCES / "sampled_pearson.json").read_bytes()
    data = json.loads(saved)
    statistic = data["observed_results"][0]
    if change == "one_ulp":
        value = float.fromhex(statistic["estimate"]["$float64"])
        statistic["estimate"]["$float64"] = math.nextafter(value, math.inf).hex()
    elif change == "signed_zero":
        statistic["estimate"]["$float64"] = (0.0).hex()
        saved = _dump_reference(data)
        statistic["estimate"]["$float64"] = (-0.0).hex()
    elif change == "seed":
        data["replicates"][0]["seed_digest_sha256"] = "0" * 64
    elif change == "lag":
        statistic["candidate_id"] += 1
    elif change == "count":
        data["exceedance_count"] += 1
    elif change == "other_backend":
        statistic["backend_identity"] = "other|" + statistic["backend_identity"]
    else:
        statistic["diagnostics"].append("changed")
    with pytest.raises(AssertionError, match="non-NumPy reference content"):
        match(saved, _dump_reference(data), {"environment": "saved"}, {"environment": "other"})


def test_cross_environment_reference_allows_only_complete_numpy_tokens():
    match = _reference_matcher()
    saved = (REFERENCES / "sampled_pearson.json").read_bytes()
    data = json.loads(saved)
    for statistic in data["observed_results"]:
        statistic["backend_identity"] = statistic["backend_identity"].replace(
            "numpy=", "numpy=old-"
        )
    for replicate in data["replicates"]:
        for statistic in replicate["statistic_results"]:
            statistic["backend_identity"] = statistic["backend_identity"].replace(
                "numpy=", "numpy=new-"
            )
    assert (
        match(saved, _dump_reference(data), {"environment": "saved"}, {"environment": "other"})
        == "cross_environment_numpy_token_only_diagnostic"
    )


@pytest.mark.parametrize("stop_id", [1, 8])
def test_callback_interruption_propagates_unchanged_before_terminal(stop_id, monkeypatch):
    observed = _observed()
    pair, request = _case("sampled_pearson")
    terminal_calls = []
    original_verifier = kernel.verify_calibration_result

    def verify(*args, **kwargs):
        terminal_calls.append(args)
        return original_verifier(*args, **kwargs)

    monkeypatch.setattr(kernel, "verify_calibration_result", verify)
    seen = []
    interruption = InterruptedError("test observer interruption")

    def interrupt(index, payload):
        seen.append(index)
        if index == stop_id:
            raise interruption

    with pytest.raises(InterruptedError) as caught:
        observed(pair, resolve_plan_v2(request), interrupt)
    assert caught.value is interruption
    assert seen == list(range(stop_id + 1))
    assert terminal_calls == []


@pytest.mark.parametrize("invalid", [None, False, 0, b"", object()])
def test_observed_entry_rejects_noncallable_observer(invalid):
    observed = _observed()
    pair, request = _case("pre_observed_ne")
    with pytest.raises(TypeError, match="observer must be callable"):
        observed(pair, resolve_plan_v2(request), invalid)


@pytest.mark.parametrize("target", ["kernel", "codec", "prepared"])
def test_callback_drift_is_rejected_before_next_payload(target, monkeypatch):
    observed = _observed()
    pair, request = _case("sampled_pearson")
    resolution = resolve_plan_v2(request)
    seen = []

    def corrupt(index, payload):
        seen.append(index)
        if target == "kernel":
            monkeypatch.setattr(kernel, "_validate_selector_output", lambda *args: None)
        elif target == "codec":
            monkeypatch.setattr(wire, "_project_outcome", lambda value: {})
        else:
            object.__setattr__(resolution.plan, "root_seed", 999)

    with pytest.raises(V2IntegrityError):
        observed(pair, resolution, corrupt)
    assert seen == [0]
