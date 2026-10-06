"""Platform identity in records and decision-level replay (2026-09-23).

Byte replay is exact only on the recording platform: floating-point results can differ in the
last bits between macOS and Linux (docs/status/evidence/linux_20260923/summary.md). Records
therefore name their platform, byte replay refuses a different platform as an environment
mismatch, and decision replay compares everything exactly except statistic values, which may
differ by a bounded number of units in the last place.
"""

from __future__ import annotations

import importlib
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

import pytest
from test_workflow import CAP, files

app = importlib.import_module("selcal.workflow")
store = importlib.import_module("selcal.workflow_store")


def _record(tmp_path: Path) -> Path:
    inp, cfg = files(tmp_path)
    record = tmp_path / "run.sqlite"
    app.run_files(inp, cfg, record, max_bytes=CAP, allow_unattainable=True)
    return record


def _rewrite(record: Path, target: Path, edit) -> Path:
    members = store.read_record(record, max_bytes=CAP)
    edit(members)
    store.write_record(target, members, max_bytes=CAP)
    return target


def _edit_metadata(change):
    def edit(members):
        meta = json.loads(members["metadata"])
        change(meta)
        members["metadata"] = json.dumps(meta).encode()

    return edit


def test_new_records_name_their_platform(tmp_path):
    meta = json.loads(store.read_record(_record(tmp_path), max_bytes=CAP)["metadata"])
    assert meta["schema"] == "selcal.workflow-record.v2"
    identity = meta["software"]["platform"]
    assert set(identity) == {"system", "machine", "libc", "blas"}
    assert identity["system"] == platform.system()
    assert identity["machine"] == platform.machine()
    assert all(type(value) is str for value in identity.values())


def test_byte_replay_refuses_a_different_platform(tmp_path):
    def other_machine(meta):
        meta["software"]["platform"]["machine"] = "other-arch"

    forged = _rewrite(_record(tmp_path), tmp_path / "other.sqlite", _edit_metadata(other_machine))
    with pytest.raises(app.WorkflowError, match="environment_mismatch"):
        app.verify_record(forged, max_bytes=CAP, replay=True)


def test_decision_replay_matches_on_the_recording_platform(tmp_path):
    summary = app.verify_record(_record(tmp_path), max_bytes=CAP, replay_decision=True)
    assert summary["replay"] == "DECISION_MATCH"
    detail = summary["decision_replay"]
    assert detail["max_statistic_ulp"] == 0
    assert detail["tolerance_ulp"] == app.DECISION_REPLAY_MAX_ULP
    assert detail["same_environment"] is True


def test_decision_replay_allows_other_python_numpy_and_platform(tmp_path):
    def elsewhere(meta):
        meta["software"]["python_version"] = "3.99.0"
        meta["software"]["numpy_version"] = "0.0.0"
        meta["software"]["platform"] = {
            "system": "Other",
            "machine": "other-arch",
            "libc": "",
            "blas": "other-blas 1",
        }

    forged = _rewrite(_record(tmp_path), tmp_path / "elsewhere.sqlite", _edit_metadata(elsewhere))
    with pytest.raises(app.WorkflowError, match="environment_mismatch"):
        app.verify_record(forged, max_bytes=CAP, replay=True)
    summary = app.verify_record(forged, max_bytes=CAP, replay_decision=True)
    assert summary["replay"] == "DECISION_MATCH"
    assert summary["decision_replay"]["same_environment"] is False
    assert summary["decision_replay"]["recorded_environment"]["platform"]["system"] == "Other"


@pytest.mark.parametrize("field", ["selcal_version", "source_files"])
def test_decision_replay_requires_the_same_code(tmp_path, field):
    def other_code(meta):
        if field == "selcal_version":
            meta["software"]["selcal_version"] = "9.9.9"
        else:
            name = next(iter(meta["software"]["source_files"]))
            meta["software"]["source_files"][name] = "0" * 64

    forged = _rewrite(_record(tmp_path), tmp_path / "code.sqlite", _edit_metadata(other_code))
    with pytest.raises(app.WorkflowError, match="environment_mismatch"):
        app.verify_record(forged, max_bytes=CAP, replay_decision=True)


def test_decision_replay_rejects_changed_recorded_content(tmp_path):
    def invent(members):
        data = json.loads(members["result"])
        data["diagnostics"] = ["invented diagnostic"]
        members["result"] = (
            json.dumps(data, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()

    forged = _rewrite(_record(tmp_path), tmp_path / "coherent.sqlite", invent)
    assert app.verify_record(forged, max_bytes=CAP)["replay"] == "NOT_PERFORMED"
    with pytest.raises(app.WorkflowError, match="decision_replay_mismatch"):
        app.verify_record(forged, max_bytes=CAP, replay_decision=True)


def test_version_1_records_remain_readable(tmp_path):
    def version_1(meta):
        meta["schema"] = "selcal.workflow-record.v1"
        del meta["software"]["platform"]

    old = _rewrite(_record(tmp_path), tmp_path / "v1.sqlite", _edit_metadata(version_1))
    assert app.verify_record(old, max_bytes=CAP)["replay"] == "NOT_PERFORMED"
    with pytest.raises(app.WorkflowError, match="environment_mismatch"):
        app.verify_record(old, max_bytes=CAP, replay=True)
    summary = app.verify_record(old, max_bytes=CAP, replay_decision=True)
    assert summary["replay"] == "DECISION_MATCH"
    assert summary["decision_replay"]["recorded_environment"]["platform"] is None


@pytest.mark.parametrize(
    "change",
    [
        lambda meta: meta["software"].__setitem__("platform", {"system": "x"}),
        lambda meta: meta["software"]["platform"].__setitem__("blas", 3),
        lambda meta: meta["software"]["platform"].__setitem__("libc", "x" * 257),
        lambda meta: meta.__setitem__("schema", "selcal.workflow-record.v1"),
    ],
)
def test_malformed_platform_identity_is_rejected(tmp_path, change):
    forged = _rewrite(_record(tmp_path), tmp_path / "bad.sqlite", _edit_metadata(change))
    with pytest.raises(app.WorkflowError, match="invalid_record_content") as error:
        app.verify_record(forged, max_bytes=CAP)
    assert getattr(error.value.__cause__, "code", None) == "invalid_metadata"


@pytest.mark.parametrize(("replay", "replay_decision"), [(True, True), (1, False), (False, "yes")])
def test_replay_options_are_exclusive_booleans(tmp_path, replay, replay_decision):
    with pytest.raises(app.WorkflowError, match="invalid_replay_option"):
        app.verify_record(
            _record(tmp_path), max_bytes=CAP, replay=replay, replay_decision=replay_decision
        )


def _float_tag(value: float) -> dict[str, str]:
    return {"$float64": value.hex()}


def _shift(value: float, units_of_one: int) -> float:
    """Move value by a whole number of ULPs of 1.0, the scale of a correlation."""
    return value + units_of_one * math.ulp(1.0)


def _result_bytes(tmp_path: Path) -> bytes:
    return store.read_record(_record(tmp_path), max_bytes=CAP)["result"]


def _first_score(data: dict) -> dict:
    return next(
        item
        for outcome in data["replicates"]
        for item in outcome["statistic_results"]
        if item["selection_score"] is not None
    )


def test_statistic_differences_within_tolerance_are_accepted(tmp_path):
    saved = _result_bytes(tmp_path)
    data = json.loads(saved)
    item = _first_score(data)
    score = float.fromhex(item["selection_score"]["$float64"])
    item["selection_score"] = _float_tag(_shift(score, 3))
    assert app.decision_replay_ulp(saved, json.dumps(data).encode()) == 3


def test_small_statistics_are_compared_on_the_correlation_scale(tmp_path):
    # Observed on Linux vs macOS: 149 ULPs of the value, 8e-18 absolute.
    saved = _result_bytes(tmp_path)
    data = json.loads(saved)
    item = _first_score(data)
    macos, linux = 0.0004314967286094956, 0.00043149672860948754
    item["selection_score"] = _float_tag(macos)
    reference = json.dumps(data).encode()
    item["selection_score"] = _float_tag(linux)
    assert app.decision_replay_ulp(reference, json.dumps(data).encode()) == 1


def test_large_statistics_are_compared_relatively(tmp_path):
    saved = _result_bytes(tmp_path)
    data = json.loads(saved)
    item = _first_score(data)
    item["selection_score"] = _float_tag(1000.0)
    reference = json.dumps(data).encode()
    item["selection_score"] = _float_tag(1000.0 + 10 * math.ulp(1000.0))
    assert app.decision_replay_ulp(reference, json.dumps(data).encode()) == 10


@pytest.mark.parametrize("units", [app.DECISION_REPLAY_MAX_ULP + 1, 10**6])
def test_statistic_differences_beyond_tolerance_are_rejected(tmp_path, units):
    saved = _result_bytes(tmp_path)
    data = json.loads(saved)
    item = _first_score(data)
    score = float.fromhex(item["selection_score"]["$float64"])
    item["selection_score"] = _float_tag(_shift(score, units))
    with pytest.raises(app.WorkflowError, match="decision_replay_mismatch"):
        app.decision_replay_ulp(saved, json.dumps(data).encode())


@pytest.mark.parametrize("path", ["p_value", "alpha", "exceedance_count", "seed", "reject_null"])
def test_decision_fields_must_match_exactly(tmp_path, path):
    saved = _result_bytes(tmp_path)
    data = json.loads(saved)
    if path == "p_value":
        p_value = float.fromhex(data["p_value"]["$float64"])
        data["p_value"] = _float_tag(math.nextafter(p_value, math.inf))
    elif path == "alpha":
        alpha = float.fromhex(data["alpha"]["$float64"])
        data["alpha"] = _float_tag(math.nextafter(alpha, math.inf))
    elif path == "exceedance_count":
        data["exceedance_count"] += 1
    elif path == "seed":
        data["replicates"][0]["seed_digest_sha256"] = "0" * 64
    else:
        data["reject_null"] = not data["reject_null"]
    with pytest.raises(app.WorkflowError, match="decision_replay_mismatch"):
        app.decision_replay_ulp(saved, json.dumps(data).encode())


def test_numpy_version_inside_backend_identity_may_differ(tmp_path):
    saved = _result_bytes(tmp_path)
    data = json.loads(saved)
    for item in data["observed_results"]:
        item["backend_identity"] = item["backend_identity"].replace("numpy=", "numpy=0.0.0-was-")
    assert app.decision_replay_ulp(saved, json.dumps(data).encode()) == 0
    data["observed_results"][0]["backend_identity"] = "other.backend|numpy=1"
    with pytest.raises(app.WorkflowError, match="decision_replay_mismatch"):
        app.decision_replay_ulp(saved, json.dumps(data).encode())


def test_cli_decision_replay(tmp_path):
    record = _record(tmp_path)
    command = [sys.executable, "-m", "selcal", "verify", str(record), "--max-bytes", str(CAP)]
    done = subprocess.run([*command, "--replay-decision"], capture_output=True, text=True,
        errors="replace")
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout)["data"]["replay"] == "DECISION_MATCH"
    both = subprocess.run(
        [*command, "--replay", "--replay-decision"], capture_output=True, text=True,
            errors="replace"
    )
    assert both.returncode == 2
