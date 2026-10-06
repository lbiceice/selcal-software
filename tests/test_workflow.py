from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

CAP = 1_048_576


def workflow():
    assert importlib.util.find_spec("selcal.workflow") is not None, "missing workflow service"
    return importlib.import_module("selcal.workflow")


def test_missing_output_parent_is_rejected_before_calibration(tmp_path, monkeypatch):
    app = workflow()
    inp, cfg = files(tmp_path)

    def must_not_compute(*args, **kwargs):
        pytest.fail("missing output parent must be rejected before calibration")

    monkeypatch.setattr(app, "calibrate_selected_family", must_not_compute)
    with pytest.raises(FileNotFoundError):
        app.run_files(
            inp, cfg, tmp_path / "absent" / "result.sqlite", max_bytes=CAP, allow_unattainable=True
        )
    assert not (tmp_path / "absent").exists()


# Tiny plumbing fixtures: their plans cannot reach alpha, so runs pass allow_unattainable=True.
def files(tmp_path: Path, *, form: str = "csv", mode: str = "complete"):
    source = [0, 3, 1, 2, 4, 5]
    target = [0, 1, 3, 2, 5, 4]
    lags = [1, 2, 3]
    null_name, null_params = "circular_shift_v2", {"min_shift": 1}
    rule = "max_absolute"
    if mode == "null_bind":
        # A valid plan whose null cannot bind: one block of five. (min_shift = 3 once did this,
        # but restricted shifts are now refused at preflight as a non-group null, ALG-02.)
        source, target, lags = [0, 1, 2, 3, 4], [4, 3, 2, 1, 0], [1, 3]
        null_name, null_params = "block_shuffle_v2", {"block_length": 5}
    elif mode == "observed_statistic_scan":
        source = [1] * 6
    elif mode == "replicate_execution":
        source, target, lags = [0, 0, 1, 2, 0, 0], [0, 1, 2, 3, 4, 5], [1, 2]
        rule = "max_upper"
    input_path = tmp_path / ("data." + form)
    if form == "csv":
        input_path.write_text(
            "x,y\n" + "".join(f"{x},{y}\n" for x, y in zip(source, target, strict=True)),
            newline="\n", encoding="utf-8"
        )
        input_config = {"format": form, "source_column": "x", "target_column": "y"}
    else:
        np.savez(input_path, source=np.asarray(source), target=np.asarray(target))
        input_config = {"format": form}
    config = {
        "schema": "selcal.workflow-config.v1",
        "input": input_config,
        "plan": {
            "candidates": lags,
            "statistic_name": "lagged_pearson_v1",
            "statistic_params": {},
            "selection_rule": rule,
            "null_name": null_name,
            "null_params": null_params,
            "replicates": 9,
            "alpha": 0.05,
            "tie_tolerance": 1e-12,
            "root_seed": 17,
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), newline="\n", encoding="utf-8")
    return input_path, config_path


@pytest.mark.parametrize("form", ["csv", "npz"])
def test_real_file_run_read_report_and_explicit_replay(tmp_path, form):
    app = workflow()
    input_path, config_path = files(tmp_path, form=form)
    validation = app.validate_files(input_path, config_path)
    assert validation["planned_replicates"] == 9
    assert validation["sample_count"] == 6
    record = tmp_path / "run.sqlite"
    result = app.run_files(input_path, config_path, record, max_bytes=CAP, allow_unattainable=True)
    assert result.p_value == 0.2
    before = record.read_bytes()
    loaded = app.read_workflow(record, max_bytes=CAP)
    assert loaded.result == result
    assert (
        loaded.metadata["raw_input_sha256"] == hashlib.sha256(input_path.read_bytes()).hexdigest()
    )
    assert app.verify_record(record, max_bytes=CAP)["replay"] == "NOT_PERFORMED"
    assert app.verify_record(record, max_bytes=CAP, replay=True)["replay"] == "MATCH"
    report = tmp_path / "report.html"
    app.report_record(record, report, max_bytes=CAP)
    html = report.read_text(encoding="utf-8")
    assert "0.2" in html and "NOT_PERFORMED" in html
    assert "Observed candidates" in html and "Replicates" in html
    assert "Observed selection" in html and "selected_candidate=2" in html
    assert "tied_candidates=(2,)" in html
    assert all(f'data-replicate-id="{i}"' in html for i in range(9))
    assert record.read_bytes() == before


@pytest.mark.parametrize("mode", ["null_bind", "observed_statistic_scan", "replicate_execution"])
def test_all_scientific_ne_stages_remain_distinct_from_io_failures(tmp_path, mode):
    app = workflow()
    inp, cfg = files(tmp_path, mode=mode)
    record = tmp_path / "ne.sqlite"
    result = app.run_files(inp, cfg, record, max_bytes=CAP, allow_unattainable=True)
    assert result.status.value == "not_evaluable"
    assert result.failure_stage.value == mode
    assert result.planned_replicates == 9
    assert result.p_value is result.reject_null is None
    assert len(result.replicates) == (9 if mode == "replicate_execution" else 0)
    assert app.verify_record(record, max_bytes=CAP, replay=True)["replay"] == "MATCH"
    report = tmp_path / "ne.html"
    app.report_record(record, report, max_bytes=CAP)
    assert "not evidence of no effect" in report.read_text(encoding="utf-8")
    assert mode in report.read_text(encoding="utf-8")
    if mode != "replicate_execution":
        assert "Observed selection</summary><pre>\nNone" in report.read_text(encoding="utf-8")


def test_existing_record_and_report_are_not_overwritten(tmp_path):
    app = workflow()
    inp, cfg = files(tmp_path)
    record = tmp_path / "existing.sqlite"
    record.write_bytes(b"user bytes")
    with pytest.raises(FileExistsError):
        app.run_files(inp, cfg, record, max_bytes=CAP, allow_unattainable=True)
    assert record.read_bytes() == b"user bytes"
    record = tmp_path / "actual.sqlite"
    app.run_files(inp, cfg, record, max_bytes=CAP, allow_unattainable=True)
    with pytest.raises(FileExistsError):
        app.report_record(record, record, max_bytes=CAP)
    assert app.read_workflow(record, max_bytes=CAP).result.p_value == 0.2


def test_documented_custom_csv_headers_complete_the_workflow(tmp_path):
    """Changing domain labels must not change the frozen example's numerical result."""
    example = Path(__file__).resolve().parents[1] / "examples" / "workflow"
    data = tmp_path / "my-data.csv"
    rows = (example / "series.csv").read_text(encoding="utf-8").splitlines()
    data.write_text("temperature,cases\n" + "\n".join(rows[1:]) + "\n", encoding="utf-8")
    config = json.loads((example / "pearson.json").read_text(encoding="utf-8"))
    config["input"].update(source_column="temperature", target_column="cases")
    plan = tmp_path / "my-plan.json"
    plan.write_text(json.dumps(config), encoding="utf-8")
    app = workflow()
    assert app.validate_files(data, plan)["sample_count"] == 100
    record = tmp_path / "my-run.sqlite"
    result = app.run_files(data, plan, record, max_bytes=CAP)
    assert result.p_value == 0.015
    assert app.verify_record(record, max_bytes=CAP, replay=True)["replay"] == "MATCH"
    report = tmp_path / "my-report.html"
    app.report_record(record, report, max_bytes=CAP)
    assert "0.015" in report.read_text(encoding="utf-8")


def test_relocated_record_uses_captured_inputs_not_original_paths(tmp_path):
    app = workflow()
    inp, cfg = files(tmp_path)
    record = tmp_path / "run.sqlite"
    app.run_files(inp, cfg, record, max_bytes=CAP, allow_unattainable=True)
    inp.write_text("original source no longer present", newline="\n", encoding="utf-8")
    cfg.write_text("not json anymore", newline="\n", encoding="utf-8")
    moved = tmp_path / "new" / "copy.sqlite"
    moved.parent.mkdir()
    moved.write_bytes(record.read_bytes())
    # Portable terminal reading/replay, not resumption of an execution instance.
    assert app.verify_record(moved, max_bytes=CAP, replay=True)["replay"] == "MATCH"


def test_invalid_input_config_and_size_do_not_create_records(tmp_path):
    app = workflow()
    inp, cfg = files(tmp_path)
    record = tmp_path / "none.sqlite"
    with pytest.raises((ValueError, RuntimeError)):
        app.run_files(inp, cfg, record, max_bytes=1, allow_unattainable=True)
    assert not record.exists()
    inp.write_text("x,y\nNaN,2\n", newline="\n", encoding="utf-8")
    with pytest.raises(ValueError):
        app.run_files(inp, cfg, record, max_bytes=CAP, allow_unattainable=True)
    assert not record.exists()


@pytest.mark.parametrize("member", ["input", "request", "metadata", "result"])
def test_coherently_rehashed_wrong_members_fail_semantic_validation(tmp_path, member):
    app = workflow()
    inp, cfg = files(tmp_path)
    record = tmp_path / "run.sqlite"
    app.run_files(inp, cfg, record, max_bytes=CAP, allow_unattainable=True)
    store = importlib.import_module("selcal.workflow_store")
    members = store.read_record(record, max_bytes=CAP)
    if member == "input":
        members[member] = members[member].replace(b"0,0", b"1,0")
    elif member == "request":
        payload = json.loads(members[member])
        payload["plan"]["root_seed"] = 99
        members[member] = json.dumps(payload).encode()
    elif member == "metadata":
        payload = json.loads(members[member])
        payload["semantic_input_sha256"] = "0" * 64
        members[member] = json.dumps(payload).encode()
    else:
        payload = json.loads(members[member])
        payload["semantic_input_sha256"] = "0" * 64
        members[member] = (
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()
    forged = tmp_path / "wrong.sqlite"
    store.write_record(forged, members, max_bytes=CAP)
    with pytest.raises(app.WorkflowError):
        app.read_workflow(forged, max_bytes=CAP)


def test_replay_environment_mismatch_is_not_silently_accepted(tmp_path):
    app = workflow()
    inp, cfg = files(tmp_path)
    record = tmp_path / "run.sqlite"
    app.run_files(inp, cfg, record, max_bytes=CAP, allow_unattainable=True)
    store = importlib.import_module("selcal.workflow_store")
    members = store.read_record(record, max_bytes=CAP)
    meta = json.loads(members["metadata"])
    meta["software"]["numpy_version"] = "0.0.0"
    members["metadata"] = json.dumps(meta).encode()
    forged = tmp_path / "mismatch.sqlite"
    store.write_record(forged, members, max_bytes=CAP)
    assert app.verify_record(forged, max_bytes=CAP)["replay"] == "NOT_PERFORMED"
    with pytest.raises(app.WorkflowError, match="environment_mismatch"):
        app.verify_record(forged, max_bytes=CAP, replay=True)


def test_changed_input_between_loading_and_capture_is_rejected(tmp_path, monkeypatch):
    app = workflow()
    inp, cfg = files(tmp_path)
    original = app.load_csv

    def changed(*args, **kwargs):
        value = original(*args, **kwargs)
        changed = inp.read_text(encoding="utf-8").replace("0,0", "1,0")
        inp.write_text(changed, newline="\n", encoding="utf-8")
        return value

    monkeypatch.setattr(app, "load_csv", changed)
    with pytest.raises(app.WorkflowError, match="input_changed"):
        app.run_files(inp, cfg, tmp_path / "none.sqlite", max_bytes=CAP, allow_unattainable=True)
    assert not (tmp_path / "none.sqlite").exists()


def test_coherent_record_content_is_not_treated_as_replay_evidence(tmp_path):
    app = workflow()
    inp, cfg = files(tmp_path)
    record = tmp_path / "actual.sqlite"
    app.run_files(inp, cfg, record, max_bytes=CAP, allow_unattainable=True)
    store = importlib.import_module("selcal.workflow_store")
    members = store.read_record(record, max_bytes=CAP)
    data = json.loads(members["result"])
    data["diagnostics"] = ["invented diagnostic"]
    members["result"] = (json.dumps(data, sort_keys=True, separators=(",", ":")) + "\n").encode()
    forged = tmp_path / "coherent.sqlite"
    store.write_record(forged, members, max_bytes=CAP)
    # This is exactly the limitation: coherent content is not proof of its execution.
    assert app.verify_record(forged, max_bytes=CAP)["replay"] == "NOT_PERFORMED"
    before = forged.read_bytes()
    with pytest.raises(app.WorkflowError, match="replay_mismatch"):
        app.verify_record(forged, max_bytes=CAP, replay=True)
    assert forged.read_bytes() == before


def test_report_escapes_recorded_metadata_and_does_not_truncate(tmp_path):
    app = workflow()
    inp, cfg = files(tmp_path)
    record = tmp_path / "actual.sqlite"
    app.run_files(inp, cfg, record, max_bytes=CAP, allow_unattainable=True)
    store = importlib.import_module("selcal.workflow_store")
    members = store.read_record(record, max_bytes=CAP)
    meta = json.loads(members["metadata"])
    meta["software"]["numpy_version"] = "<script>alert(1)</script>"
    members["metadata"] = json.dumps(meta).encode()
    altered = tmp_path / "altered.sqlite"
    store.write_record(altered, members, max_bytes=CAP)
    output = tmp_path / "escaped.html"
    app.report_record(altered, output, max_bytes=CAP)
    text = output.read_text(encoding="utf-8")
    assert "<script>" not in text and "&lt;script&gt;" in text
    assert "white-space:pre-wrap" in text
