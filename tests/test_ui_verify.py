"""The UI checks its explicitly selected record through the real CLI."""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import subprocess
import sys
import threading
import time
from contextlib import contextmanager

import pytest
from _platform_support import directory_link_or_skip, symlink_or_skip
from test_ui_jobs import admission, jobs_module
from test_ui_jobs import physical_tmp as _physical_tmp
from test_ui_process import wait_job

physical_tmp = _physical_tmp


@contextmanager
def saved_job(folder, *, form="csv", mode="complete"):
    inp, request = admission(folder, form=form, mode=mode)
    manager = jobs_module().JobManager(folder / "work")
    try:
        job = manager.admit(request)
        manager.upload(job["id"], io.BytesIO(inp.read_bytes()), inp.stat().st_size)
        manager.start(job["id"], "run")
        original = wait_job(manager, job["id"])
        assert original["state"] == ("complete" if mode == "complete" else "not_evaluable")
        yield manager, job["id"], original
    finally:
        manager.close()


def files_at(folder):
    return {str(p.relative_to(folder)): p.read_bytes() for p in folder.rglob("*") if p.is_file()}


@pytest.mark.parametrize("form", ["csv", "npz"])
@pytest.mark.parametrize(
    "mode", ["complete", "null_bind", "observed_statistic_scan", "replicate_execution"]
)
def test_real_saved_run_then_verify_matches_cli_without_new_record(physical_tmp, form, mode):
    with saved_job(physical_tmp, form=form, mode=mode) as (manager, job_id, original):
        path = manager.workspace / "jobs" / job_id
        old_operations = files_at(path / "operations")
        (record,) = (path / "operations").glob("*/result.sqlite")
        manager.start(job_id, "verify")  # The initial feature RED must reach this real action.
        checked = wait_job(manager, job_id)
        assert checked["state"] == original["state"]
        assert checked["operation"] == checked["result"]["command"] == "verify"
        data = checked["result"]["data"]
        assert data["verification_scope"] == "input_plan_result_consistency"
        assert data["replay"] == "NOT_PERFORMED"
        assert data["historical_execution_authenticated"] is False
        direct = subprocess.run(
            [sys.executable, "-m", "selcal", "verify", str(record), "--max-bytes", "1048576"],
            capture_output=True,
            timeout=30,
        )
        assert checked["result"] == json.loads(direct.stdout)
        reference = json.loads((path / "record.json").read_text(encoding="utf-8"))
        assert reference == {
            "schema": "selcal.ui-record-reference.v1",
            "operation_id": record.parent.name,
            "bytes": record.stat().st_size,
            "sha256": hashlib.sha256(record.read_bytes()).hexdigest(),
        }
        assert checked["record_target"] == {
            "available": True,
            "reference": reference,
            "reason": None,
        }
        for name, value in old_operations.items():
            assert (path / "operations" / name).read_bytes() == value
        assert list((path / "operations").glob("*/result.sqlite")) == [record]
        assert "content" in checked["message"].lower() and "replay" in checked["message"].lower()
        if mode != "complete":
            assert data["p_value"] is None and data["reject_null"] is None


def test_repeated_verify_restart_and_resume_keep_explicit_target(physical_tmp):
    with saved_job(physical_tmp) as (manager, job_id, _original):
        path = manager.workspace / "jobs" / job_id
        first = (path / "record.json").read_bytes()
        before = files_at(path / "operations")
        for _ in range(2):
            manager.start(job_id, "verify")
            assert wait_job(manager, job_id)["result"]["command"] == "verify"
            assert (path / "record.json").read_bytes() == first
        manager.close()
        reopened = jobs_module().JobManager(manager.workspace)
        try:
            assert reopened.get(job_id)["saved_observation_only"] is True
            assert reopened.get(job_id)["record_target"]["available"] is True
            reopened.start(job_id, "resume")
            assert wait_job(reopened, job_id)["state"] == "complete"
            resumed = (path / "record.json").read_bytes()
            assert json.loads(resumed)["operation_id"] != json.loads(first)["operation_id"]
            reopened.start(job_id, "verify")
            assert wait_job(reopened, job_id)["result"]["command"] == "verify"
            assert (path / "record.json").read_bytes() == resumed
            assert len(list((path / "operations").glob("*/result.sqlite"))) == 2
            for name, value in before.items():
                assert (path / "operations" / name).read_bytes() == value
        finally:
            reopened.close()


def test_legacy_records_readable_without_guessing_ui_target(physical_tmp):
    with saved_job(physical_tmp) as (manager, job_id, _original):
        path = manager.workspace / "jobs" / job_id
        (path / "record.json").rename(path / "reference-preimage")
        before = files_at(path)
        manager.close()
        reopened = jobs_module().JobManager(manager.workspace)
        try:
            target = reopened.get(job_id)["record_target"]
            assert target["available"] is False and target["reference"] is None
            assert "CLI" in target["reason"]
            with pytest.raises(jobs_module().UIError, match=r"target|reference"):
                reopened.start(job_id, "verify")
            assert not (path / "record.json").exists()
            assert before == files_at(path)
        finally:
            reopened.close()


@pytest.mark.parametrize(
    "damage",
    [
        "extra",
        "bool_bytes",
        "traversal",
        "hash",
        "missing",
        "linked_file",
        "linked_parent",
        "linked_reference",
        "corrupt",
        "foreign_input",
        "foreign_plan",
        "changed_input",
    ],
)
def test_bad_or_foreign_target_refused_without_new_operation(physical_tmp, damage):
    with saved_job(physical_tmp) as (manager, job_id, _original):
        path = manager.workspace / "jobs" / job_id
        reference_path = path / "record.json"
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        record = path / "operations" / reference["operation_id"] / "result.sqlite"
        if damage == "extra":
            reference["other"] = True
        elif damage == "bool_bytes":
            reference["bytes"] = True
        elif damage == "traversal":
            reference["operation_id"] = "../elsewhere"
        elif damage == "hash":
            record.write_bytes(record.read_bytes() + b"changed")
        elif damage == "missing":
            record.rename(record.with_suffix(".retained"))
        elif damage == "linked_file":
            record.rename(record.with_suffix(".retained"))
            symlink_or_skip(record, record.with_suffix(".retained"))
        elif damage == "linked_parent":
            operation = record.parent
            operation.rename(operation.with_name("retained"))
            directory_link_or_skip(operation, operation.with_name("retained"))
        elif damage == "linked_reference":
            reference_path.rename(path / "reference-preimage")
            symlink_or_skip(reference_path, path / "reference-preimage")
        elif damage == "changed_input":
            (path / "input.csv").write_bytes(b"x,y\n0,1\n")
        else:
            if damage == "corrupt":
                record.write_bytes(b"invalid record")
            else:
                folder = physical_tmp / "foreign"
                folder.mkdir()
                inp, request = admission(folder)
                config = json.loads(request["config_text"])
                if damage == "foreign_input":
                    changed = inp.read_text(encoding="utf-8").replace("0,0\n", "0.0,0.0\n")
                    inp.write_text(changed, encoding="utf-8")
                else:
                    config["plan"]["root_seed"] += 1
                cfg = folder / "config.json"
                cfg.write_text(json.dumps(config), encoding="utf-8")
                from selcal.workflow import run_files

                replacement = folder / "foreign.sqlite"
                run_files(inp, cfg, replacement, max_bytes=1048576, allow_unattainable=True)
                shutil.copyfile(replacement, record)
            reference["bytes"] = record.stat().st_size
            reference["sha256"] = hashlib.sha256(record.read_bytes()).hexdigest()
        if damage != "linked_reference":
            reference_path.write_text(json.dumps(reference), encoding="utf-8")
        before = files_at(path / "operations")
        with pytest.raises(jobs_module().UIError):
            manager.start(job_id, "verify")
        assert before == files_at(path / "operations")
        assert manager._active is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("verification_scope", "historical_execution"),
        ("verification_scope", None),
        ("replay", "MATCH"),
        ("replay", None),
        ("historical_execution_authenticated", True),
        ("historical_execution_authenticated", 0),
        ("historical_execution_authenticated", "false"),
        ("historical_execution_authenticated", None),
    ],
)
def test_verify_terminal_requires_exact_scope_and_boolean(physical_tmp, field, value):
    from selcal.workflow import verify_record

    with saved_job(physical_tmp) as (manager, job_id, _original):
        (record,) = (manager.workspace / "jobs" / job_id / "operations").glob("*/result.sqlite")
        data = verify_record(record, max_bytes=1048576)
        terminal = {
            "schema": "selcal.cli.v1",
            "command": "verify",
            "outcome": "COMPLETE",
            "exit_code": 0,
            "error": None,
            "detail": None,
            "data": data,
        }
        assert jobs_module()._terminal(json.dumps(terminal).encode(), "verify", 0) == terminal
        if value is None:
            data.pop(field)
        else:
            data[field] = value
        with pytest.raises(jobs_module().UIError):
            jobs_module()._terminal(json.dumps(terminal).encode(), "verify", 0)


def test_target_changed_after_cli_check_cannot_inherit_success(physical_tmp, monkeypatch):
    with saved_job(physical_tmp) as (manager, job_id, _original):
        path = manager.workspace / "jobs" / job_id
        before_reference = (path / "record.json").read_bytes()
        (record,) = (path / "operations").glob("*/result.sqlite")
        original = jobs_module()._terminal

        def change_after_terminal(data, action, code):
            terminal = original(data, action, code)
            if action == "verify":
                record.write_bytes(record.read_bytes() + b"changed after the actual CLI check")
            return terminal

        monkeypatch.setattr(jobs_module(), "_terminal", change_after_terminal)
        manager.start(job_id, "verify")
        checked = wait_job(manager, job_id)
        assert checked["state"] == "failed" and checked["result"] is None
        assert "changed" in checked["message"].lower()
        assert (path / "record.json").read_bytes() == before_reference
        assert len(list((path / "operations").glob("*/result.sqlite"))) == 1


@pytest.mark.parametrize("boundary", ["failure", "cancel", "shutdown"])
def test_verify_failure_and_lifecycle_keep_the_same_reference(physical_tmp, monkeypatch, boundary):
    with saved_job(physical_tmp) as (manager, job_id, _original):
        path = manager.workspace / "jobs" / job_id
        reference = (path / "record.json").read_bytes()
        target = path / "operations" / json.loads(reference)["operation_id"] / "result.sqlite"
        before = files_at(path / "operations")
        actual_popen = subprocess.Popen
        calls = []
        # -I already leaves the working directory off sys.path; otherwise -P does (R17 item 2).
        flags = (["-I"] if sys.flags.isolated
                 else ["-E", "-P"] if sys.flags.ignore_environment else ["-P"])

        def launch(argv, **kwargs):
            calls.append(argv)
            if boundary == "failure":
                altered = list(argv)
                altered[4 + len(flags)] += ".missing"
                return actual_popen(altered, **kwargs)
            return actual_popen([argv[0], "-c", "import time; time.sleep(30)"], **kwargs)

        monkeypatch.setattr(jobs_module().subprocess, "Popen", launch)
        manager.start(job_id, "verify")
        executable = jobs_module()._windows_image() if sys.platform == "win32" else sys.executable
        assert calls == [
            [executable, *flags, "-m", "selcal", "verify", str(target), "--max-bytes", "1048576"]
        ]
        if boundary != "failure":
            child = manager._active[1]
            with pytest.raises(jobs_module().UIError, match="active"):
                manager.start(job_id, "verify")
            if boundary == "cancel":
                manager.cancel(job_id)
            else:
                manager.close()
            assert child.poll() is not None
        observed = wait_job(manager, job_id)
        assert observed["state"] == ("failed" if boundary == "failure" else "interrupted")
        assert (path / "record.json").read_bytes() == reference
        assert list((path / "operations").glob("*/result.sqlite")) == [target]
        for name, raw in before.items():
            assert (path / "operations" / name).read_bytes() == raw


def settled_owner(manager, job_id):
    deadline = time.monotonic() + 30
    while manager._active is not None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert manager._active is None, "Owned child did not settle"
    return manager.get(job_id)


@pytest.mark.parametrize("cap", [sys.maxsize, sys.maxsize + 1, 10**100])
@pytest.mark.parametrize("form", ["csv", "npz"])
def test_large_legal_budget_real_run_then_verify(physical_tmp, monkeypatch, cap, form):
    errors = []
    monkeypatch.setattr(threading, "excepthook", lambda args: errors.append(str(args.exc_value)))
    inp, request = admission(physical_tmp, form=form)
    request["max_bytes"] = str(cap)
    manager = jobs_module().JobManager(physical_tmp / "work")
    try:
        job = manager.admit(request)
        manager.upload(job["id"], io.BytesIO(inp.read_bytes()), inp.stat().st_size)
        manager.start(job["id"], "run")
        result = settled_owner(manager, job["id"])
        path = manager.workspace / "jobs" / job["id"]
        (terminal_path,) = path.glob("operations/*/terminal.json")
        terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
        assert terminal["command"] == "run" and terminal["exit_code"] == 0
        assert result["state"] == "complete", (result["state"], errors)
        assert result["max_bytes"] == str(cap)
        reference = (path / "record.json").read_bytes()
        manager.start(job["id"], "verify")
        checked = settled_owner(manager, job["id"])
        assert checked["state"] == "complete" and checked["result"]["command"] == "verify"
        assert checked["result"]["data"]["replay"] == "NOT_PERFORMED"
        assert checked["max_bytes"] == str(cap) and not errors
        assert (path / "record.json").read_bytes() == reference
        (record,) = path.glob("operations/*/result.sqlite")
        assert record.stat().st_size < 1048576, "This probe uses only a small real record"
    finally:
        manager.close()


def test_reference_reader_exact_file_size_budget_and_one_byte_less(physical_tmp):
    with saved_job(physical_tmp) as (manager, job_id, _original):
        path = manager.workspace / "jobs" / job_id
        (record,) = path.glob("operations/*/result.sqlite")
        before = record.read_bytes()
        target, reference = manager._record_target(job_id, len(before))
        assert target == record and reference["bytes"] == len(before)
        assert reference["sha256"] == hashlib.sha256(before).hexdigest()
        with pytest.raises(jobs_module().UIError):
            manager._record_target(job_id, len(before) - 1)
        reference["bytes"] = len(before) - 1
        (path / "record.json").write_text(json.dumps(reference), encoding="utf-8")
        with pytest.raises(jobs_module().UIError, match="size limit"):
            manager._record_target(job_id, len(before) - 1)
        assert record.read_bytes() == before


@pytest.mark.parametrize("error_type", [OSError, OverflowError, ValueError, MemoryError])
def test_worker_record_read_error_becomes_failed_not_permanent_running(
    physical_tmp, monkeypatch, error_type
):
    errors = []
    monkeypatch.setattr(threading, "excepthook", lambda args: errors.append(str(args.exc_value)))
    inp, request = admission(physical_tmp)
    manager = jobs_module().JobManager(physical_tmp / "work")
    try:
        job = manager.admit(request)
        manager.upload(job["id"], io.BytesIO(inp.read_bytes()), inp.stat().st_size)

        def unreadable(*_args):
            raise error_type("simulated record-read failure")

        monkeypatch.setattr(manager, "_save_record_reference", unreadable)
        manager.start(job["id"], "run")
        result = settled_owner(manager, job["id"])
        assert result["state"] == "failed", (result["state"], errors)
        assert result["result"] is None and "record-read failure" in result["message"]
        assert not errors
        path = manager.workspace / "jobs" / job["id"]
        assert not (path / "record.json").exists()
        assert len(list(path.glob("operations/*/result.sqlite"))) == 1
        (terminal_path,) = path.glob("operations/*/terminal.json")
        assert json.loads(terminal_path.read_text(encoding="utf-8"))["exit_code"] == 0
    finally:
        manager.close()
