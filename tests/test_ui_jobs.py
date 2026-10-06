from __future__ import annotations

import importlib
import importlib.util
import io
import json
import tempfile
from pathlib import Path

import pytest
from _platform_support import directory_link_or_skip, symlink_or_skip
from test_workflow import files


def jobs_module():
    assert importlib.util.find_spec("selcal.ui_jobs") is not None, "missing local UI jobs"
    return importlib.import_module("selcal.ui_jobs")


@pytest.fixture
def physical_tmp():
    # Runtime path/link fixtures stay outside the checkout, on a physical parent.
    with tempfile.TemporaryDirectory(
        prefix="selcal-ui-", dir=Path(tempfile.gettempdir()).resolve()
    ) as d:
        yield Path(d)


def admission(folder, *, form="csv", mode="complete"):
    inp, cfg = files(folder, form=form, mode=mode)
    return inp, {
        "config_text": cfg.read_text(encoding="utf-8"),
        "max_bytes": "1048576",
        "allow_unattainable": True,
    }


def test_ui_entry_is_available():
    jobs_module()
    assert importlib.util.find_spec("selcal.ui") is not None, "missing local HTTP UI"


def test_workspace_unknown_and_linked_are_refused(physical_tmp):
    app = jobs_module()
    unknown = physical_tmp / "unknown"
    unknown.mkdir()
    (unknown / "keep").write_bytes(b"original")
    with pytest.raises(app.UIError):
        app.JobManager(unknown)
    link = physical_tmp / "linked"
    directory_link_or_skip(link, unknown)
    with pytest.raises(app.UIError):
        app.JobManager(link)
    assert (unknown / "keep").read_bytes() == b"original"


def test_admission_exact_config_and_exclusive_complete_upload(physical_tmp):
    app = jobs_module()
    inp, request = admission(physical_tmp)
    manager = app.JobManager(physical_tmp / "work")
    try:
        job = manager.admit(request)
        path = manager.workspace / "jobs" / job["id"]
        assert (path / "request.json").read_text(encoding="utf-8") == request["config_text"]
        manager.upload(job["id"], io.BytesIO(inp.read_bytes()), inp.stat().st_size)
        assert manager.get(job["id"])["input_status"] == "complete"
        with pytest.raises(app.UIError):
            manager.upload(job["id"], io.BytesIO(b"replace"), 7)
        assert (path / "input.csv").read_bytes() == inp.read_bytes()
    finally:
        manager.close()
    restored = app.JobManager(physical_tmp / "work")
    assert restored.list_jobs()[0]["id"] == job["id"]
    restored.close()


@pytest.mark.parametrize(
    "mutation",
    [
        "outer_extra",
        "string_bool",
        "number_budget",
        "negative",
        "duplicate",
        "float_alpha",
        "oversize",
    ],
)
def test_invalid_admission_creates_no_job(physical_tmp, mutation):
    app = jobs_module()
    _, request = admission(physical_tmp)
    if mutation == "outer_extra":
        request["argv"] = []
    elif mutation == "string_bool":
        request["allow_unattainable"] = "false"
    elif mutation == "number_budget":
        request["max_bytes"] = 1048576
    elif mutation == "negative":
        request["max_bytes"] = "-1"
    elif mutation == "duplicate":
        request["config_text"] = request["config_text"].replace(
            '"schema":', '"schema":"bad","schema":'
        )
    elif mutation == "float_alpha":
        request["config_text"] = request["config_text"].replace('"alpha": 0.05', '"alpha": 1')
    else:
        request["config_text"] += " " * 65536
    manager = app.JobManager(physical_tmp / "work")
    try:
        with pytest.raises(app.UIError):
            manager.admit(request)
        assert manager.list_jobs() == []
    finally:
        manager.close()


def test_incomplete_input_is_retained_and_never_runnable(physical_tmp):
    app = jobs_module()
    _, request = admission(physical_tmp)
    manager = app.JobManager(physical_tmp / "work")
    try:
        job = manager.admit(request)
        with pytest.raises(app.UIError):
            manager.upload(job["id"], io.BytesIO(b"x,y\n"), 80)
        assert manager.get(job["id"])["input_status"] == "incomplete"
        assert (manager.workspace / "jobs" / job["id"] / "input.csv").read_bytes() == b"x,y\n"
        with pytest.raises(app.UIError):
            manager.start(job["id"], "run")
    finally:
        manager.close()


@pytest.mark.parametrize("form,cap", [("csv", 67108864), ("npz", 33554432)])
def test_upload_cap_rejected_before_stream_read(physical_tmp, form, cap):
    app = jobs_module()
    _, request = admission(physical_tmp, form=form)
    manager = app.JobManager(physical_tmp / "work")
    try:
        job = manager.admit(request)
        assert manager.input_limit(job["id"]) == cap
        stream = io.BytesIO(b"keep")
        with pytest.raises(app.UIError):
            manager.upload(job["id"], stream, cap + 1)
        assert stream.tell() == 0
        assert manager.get(job["id"])["input_status"] == "missing"
    finally:
        manager.close()


def test_linked_job_leaf_is_refused(physical_tmp):
    app = jobs_module()
    _, request = admission(physical_tmp)
    manager = app.JobManager(physical_tmp / "work")
    try:
        job = manager.admit(request)
        path = manager.workspace / "jobs" / job["id"] / "request.json"
        original = path.read_bytes()
        path.rename(path.with_suffix(".saved"))
        symlink_or_skip(path, path.with_suffix(".saved"))
        with pytest.raises(app.UIError):
            manager.get(job["id"])
        assert path.read_bytes() == original
    finally:
        manager.close()


def test_restart_live_observation_becomes_interrupted(physical_tmp):
    app = jobs_module()
    _, request = admission(physical_tmp)
    manager = app.JobManager(physical_tmp / "work")
    job = manager.admit(request)
    manager.close()
    observation = physical_tmp / "work" / "jobs" / job["id"] / "observation.json"
    value = json.loads(observation.read_text(encoding="utf-8"))
    value.update(state="running", operation="run")
    observation.write_text(json.dumps(value), encoding="utf-8")
    restored = app.JobManager(physical_tmp / "work")
    try:
        assert restored.get(job["id"])["state"] == "interrupted"
        assert "not verified" in restored.get(job["id"])["message"]
    finally:
        restored.close()


@pytest.mark.parametrize(
    "member,field", [("metadata.json", "format"), ("observation.json", "state")]
)
def test_malformed_saved_member_gives_bounded_refusal(physical_tmp, member, field):
    app = jobs_module()
    _, request = admission(physical_tmp)
    manager = app.JobManager(physical_tmp / "work")
    job = manager.admit(request)
    manager.close()
    target = manager.workspace / "jobs" / job["id"] / member
    value = json.loads(target.read_text(encoding="utf-8"))
    value[field] = []
    target.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(app.UIError, match="invalid"):
        app.JobManager(manager.workspace)


def test_fifo_input_is_refused_without_blocking(physical_tmp):
    import os

    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFO requires POSIX")
    app = jobs_module()
    inp, request = admission(physical_tmp)
    manager = app.JobManager(physical_tmp / "work")
    try:
        job = manager.admit(request)
        manager.upload(job["id"], io.BytesIO(inp.read_bytes()), inp.stat().st_size)
        target = manager.workspace / "jobs" / job["id"] / "input.csv"
        target.rename(target.with_suffix(".original"))
        os.mkfifo(target)
        with pytest.raises(app.UIError, match="regular"):
            manager.start(job["id"], "validate")
    finally:
        manager.close()


def test_budget_keeps_python_integer_envelope_without_javascript_rounding(physical_tmp):
    app = jobs_module()
    _, request = admission(physical_tmp)
    request["max_bytes"] = "1" + "0" * 1000
    manager = app.JobManager(physical_tmp / "work")
    try:
        assert manager.admit(request)["max_bytes"] == request["max_bytes"]
    finally:
        manager.close()
