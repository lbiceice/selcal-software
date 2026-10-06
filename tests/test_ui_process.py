from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import threading
import time

import pytest
from test_ui_jobs import (
    admission,
    jobs_module,
)
from test_ui_jobs import physical_tmp as _physical_tmp

physical_tmp = _physical_tmp


def test_ui_help_is_real():
    result = subprocess.run(
        [sys.executable, "-m", "selcal", "ui", "--help"], capture_output=True, timeout=10
    )
    assert result.returncode == 0
    assert b"--workspace" in result.stdout and b"--no-browser" in result.stdout
    assert b"--host" not in result.stdout


def wait_job(manager, job_id, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = manager.get(job_id)
        if value["state"] != "running":
            return value
        time.sleep(0.02)
    raise AssertionError("UI child did not finish")


@pytest.mark.parametrize("form", ["csv", "npz"])
@pytest.mark.parametrize(
    "mode", ["complete", "null_bind", "observed_statistic_scan", "replicate_execution"]
)
def test_real_cli_validate_and_run_equivalence(physical_tmp, form, mode):
    app = jobs_module()
    inp, request = admission(physical_tmp, form=form, mode=mode)
    manager = app.JobManager(physical_tmp / "work")
    try:
        job = manager.admit(request)
        manager.upload(job["id"], io.BytesIO(inp.read_bytes()), inp.stat().st_size)
        manager.start(job["id"], "validate")
        validated = wait_job(manager, job["id"])
        direct = subprocess.run(
            [
                sys.executable,
                "-m",
                "selcal",
                "validate",
                str(inp),
                str(physical_tmp / "config.json"),
            ],
            capture_output=True,
            timeout=30,
        )
        assert validated["result"] == json.loads(direct.stdout)
        manager.start(job["id"], "run")
        completed = wait_job(manager, job["id"])
        direct = subprocess.run(
            [
                sys.executable,
                "-m",
                "selcal",
                "run",
                str(inp),
                str(physical_tmp / "config.json"),
                str(physical_tmp / "direct.sqlite"),
                "--max-bytes",
                request["max_bytes"],
                "--allow-unattainable-plan",
            ],
            capture_output=True,
            timeout=30,
        )
        expected = json.loads(direct.stdout)
        actual = completed["result"]["data"].copy()
        actual.pop("checkpoint")
        assert actual == expected["data"]
        assert completed["state"] == ("complete" if mode == "complete" else "not_evaluable")
        assert completed["result"]["exit_code"] == direct.returncode
        if mode != "complete":
            assert actual["failure_stage"] == mode
            assert actual["p_value"] is None and actual["reject_null"] is None
        before = completed.copy()
        assert manager.cancel(job["id"]) == before
        with pytest.raises(app.UIError, match="checkpoint"):
            manager.start(job["id"], "run")
    finally:
        manager.close()


@pytest.mark.parametrize("budget,allow", [("1048576", False), ("1", True)])
def test_guard_and_resource_refusal_are_not_scientific_ne(physical_tmp, budget, allow):
    app = jobs_module()
    inp, request = admission(physical_tmp)
    request.update(max_bytes=budget, allow_unattainable=allow)
    manager = app.JobManager(physical_tmp / "work")
    try:
        job = manager.admit(request)
        manager.upload(job["id"], io.BytesIO(inp.read_bytes()), inp.stat().st_size)
        manager.start(job["id"], "run")
        result = wait_job(manager, job["id"])
        assert result["state"] == "failed"
        direct = subprocess.run(
            [
                sys.executable,
                "-m",
                "selcal",
                "run",
                str(inp),
                str(physical_tmp / "config.json"),
                str(physical_tmp / "direct.sqlite"),
                "--max-bytes",
                budget,
                *(["--allow-unattainable-plan"] if allow else []),
            ],
            capture_output=True,
            timeout=30,
        )
        assert result["result"]["outcome"] == json.loads(direct.stdout)["outcome"]
        assert result["result"]["exit_code"] == direct.returncode
        assert direct.returncode != 7
    finally:
        manager.close()


@pytest.mark.parametrize("stop", ["cancel", "shutdown"])
def test_real_checkpoint_progress_then_stop_reaps_only_owned_child(physical_tmp, stop):
    app = jobs_module()
    inp, request = admission(physical_tmp)
    config = json.loads(request["config_text"])
    config["plan"]["replicates"] = 999
    request["config_text"] = json.dumps(config)
    request["max_bytes"] = "10000000"
    manager = app.JobManager(physical_tmp / "work")
    job = manager.admit(request)
    manager.upload(job["id"], io.BytesIO(inp.read_bytes()), inp.stat().st_size)
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        manager.start(job["id"], "run")
        child = manager._active[1]
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            value = manager.get(job["id"])
            if "retained=1/999" in value["progress"]:
                break
            assert value["state"] == "running", value
            time.sleep(0.01)
        else:
            pytest.fail("Did not observe a committed checkpoint progress event")
        with pytest.raises(app.UIError, match="active"):
            manager.start(job["id"], "validate")
        if stop == "cancel":
            manager.cancel(job["id"])
        else:
            manager.close()
        assert child.poll() is not None
        assert unrelated.poll() is None
        assert manager.get(job["id"])["state"] == "interrupted"
        assert (manager.workspace / "jobs" / job["id"] / "checkpoint").is_dir()
    finally:
        manager.close()
        unrelated.terminate()
        unrelated.wait(timeout=5)
    reopened = app.JobManager(physical_tmp / "work")
    try:
        assert reopened.get(job["id"])["state"] == "interrupted"
        with pytest.raises(app.UIError, match="checkpoint"):
            reopened.start(job["id"], "run")
    finally:
        reopened.close()


@pytest.mark.parametrize(
    "producer",
    [
        "invalid",
        "wrong_command",
        "wrong_exit",
        "overflow_stdout",
        "overflow_stderr",
        "incompatible",
        "nonfinite",
        "validate_preflight_list",
    ],
)
def test_synthetic_pipe_failures_are_operational_not_scientific(
    physical_tmp, monkeypatch, producer
):
    app = jobs_module()
    inp, request = admission(physical_tmp)
    terminal = {
        "schema": "selcal.cli.v1",
        "command": "run",
        "exit_code": 0,
        "outcome": "COMPLETE",
        "error": None,
        "detail": None,
        "data": {"status": "complete", "p_value": 0.2, "reject_null": False},
    }
    if producer == "wrong_command":
        terminal["command"] = "validate"
    elif producer == "wrong_exit":
        terminal["exit_code"] = 7
    elif producer == "incompatible":
        terminal["data"]["status"] = "not_evaluable"
    elif producer == "validate_preflight_list":
        terminal.update(command="validate", outcome="PASS", data={"preflight": []})
    script = "import sys; print(" + repr(json.dumps(terminal)) + ")"
    if producer == "invalid":
        script = "print('not a terminal JSON')"
    elif producer == "nonfinite":
        script = (
            "print("
            + repr(json.dumps(terminal).replace('"p_value": 0.2', '"p_value": 1e999'))
            + ")"
        )
    elif producer.startswith("overflow"):
        stream = "stdout" if producer.endswith("stdout") else "stderr"
        script = f"import sys; sys.{stream}.write('x' * 3000000); sys.{stream}.flush()"
    if not producer.startswith("overflow"):
        script += "; import sys; print('retained terminal diagnostic', file=sys.stderr)"
    real_popen = subprocess.Popen

    def launch(_argv, **kwargs):
        return real_popen([sys.executable, "-c", script], **kwargs)

    monkeypatch.setattr(app.subprocess, "Popen", launch)
    manager = app.JobManager(physical_tmp / "work")
    try:
        job = manager.admit(request)
        manager.upload(job["id"], io.BytesIO(inp.read_bytes()), inp.stat().st_size)
        action = "validate" if producer == "validate_preflight_list" else "run"
        with manager._lock:
            manager.start(job["id"], action)
            _, child, watcher = manager._active
        value = wait_job(manager, job["id"])
        watcher.join(timeout=5)
        assert not watcher.is_alive()
        assert manager._active is None
        assert child.poll() is not None
        assert value["state"] == "failed"
        assert value["result"] is None
        assert len(value["progress"]) <= app.PROGRESS_LIMIT
        assert not list((manager._path(job["id"]) / "operations").glob("*/terminal.json"))
        if producer.startswith("overflow"):
            assert value["message"] == (
                "Child output exceeded its bound or could not be retained; process stopped."
            )
        else:
            assert "retained terminal diagnostic" in value["progress"]
            assert value["message"] == (
                f"The SelCal process exited with code {child.returncode} and returned an "
                "invalid terminal result. See Progress and operation details for retained "
                "diagnostics."
            )
    finally:
        manager.close()


@pytest.mark.parametrize(
    "stdout,exit_code,classification",
    [
        pytest.param(b"", 0, "missing", id="empty-success-exit"),
        pytest.param(b"", 1, "missing", id="empty-failure-exit"),
        pytest.param(b" \t\r\n", 0, "missing", id="whitespace-success-exit"),
        pytest.param(b" \t\r\n", 1, "missing", id="whitespace-failure-exit"),
        pytest.param(b'{"schema":', 3, "invalid", id="truncated-json"),
        pytest.param(b"\xff\xfe", 8, "invalid", id="non-utf8"),
    ],
)
def test_real_child_missing_or_invalid_terminal_retains_diagnostics(
    physical_tmp, monkeypatch, stdout, exit_code, classification
):
    app = jobs_module()
    inp, request = admission(physical_tmp)
    real_popen = subprocess.Popen
    marker = "retained diagnostic from real child"
    diagnostic = "x" * app.PROGRESS_LIMIT + marker
    script = (
        f"import sys; sys.stderr.write({diagnostic!r}); sys.stderr.flush(); "
        f"sys.stdout.buffer.write({stdout!r}); sys.stdout.buffer.flush(); "
        f"sys.exit({exit_code})"
    )

    def launch(_argv, **kwargs):
        return real_popen([sys.executable, "-c", script], **kwargs)

    monkeypatch.setattr(app.subprocess, "Popen", launch)
    manager = app.JobManager(physical_tmp / "work")
    try:
        job = manager.admit(request)
        manager.upload(job["id"], io.BytesIO(inp.read_bytes()), inp.stat().st_size)
        with manager._lock:
            manager.start(job["id"], "validate")
            _, child, watcher = manager._active
        value = wait_job(manager, job["id"])
        watcher.join(timeout=5)
        assert not watcher.is_alive()
        assert manager._active is None
        assert child.returncode == exit_code
        assert value["state"] == "failed"
        assert value["result"] is None
        assert marker in value["progress"]
        assert len(value["progress"]) <= app.PROGRESS_LIMIT
        assert not list((manager._path(job["id"]) / "operations").glob("*/terminal.json"))
        expected = (
            "without returning a terminal result."
            if classification == "missing"
            else "and returned an invalid terminal result."
        )
        assert "Invalid UTF-8 JSON" not in value["message"]
        assert value["message"] == (
            f"The SelCal process exited with code {exit_code} {expected} "
            "See Progress and operation details for retained diagnostics."
        )
    finally:
        manager.close()


def test_real_child_valid_invalid_request_preserves_terminal_and_detail(physical_tmp, monkeypatch):
    app = jobs_module()
    inp, request = admission(physical_tmp)
    terminal = {
        "schema": "selcal.cli.v1",
        "command": "validate",
        "exit_code": 2,
        "outcome": "INVALID_REQUEST",
        "error": "WorkflowConfigError",
        "detail": "Original CLI refusal detail remains intact.",
        "data": None,
    }
    stdout = (json.dumps(terminal) + "\n").encode("utf-8")
    real_popen = subprocess.Popen
    script = (
        "import sys; print('retained CLI refusal diagnostic', file=sys.stderr); "
        f"sys.stdout.buffer.write({stdout!r}); sys.stdout.buffer.flush(); sys.exit(2)"
    )

    def launch(_argv, **kwargs):
        return real_popen([sys.executable, "-c", script], **kwargs)

    monkeypatch.setattr(app.subprocess, "Popen", launch)
    manager = app.JobManager(physical_tmp / "work")
    try:
        job = manager.admit(request)
        manager.upload(job["id"], io.BytesIO(inp.read_bytes()), inp.stat().st_size)
        with manager._lock:
            manager.start(job["id"], "validate")
            _, child, watcher = manager._active
        value = wait_job(manager, job["id"])
        watcher.join(timeout=5)
        assert not watcher.is_alive()
        assert manager._active is None
        assert child.returncode == 2
        assert value["state"] == "failed"
        assert value["message"] == terminal["detail"]
        assert value["result"] == terminal
        assert "retained CLI refusal diagnostic" in value["progress"]
        assert len(value["progress"]) <= app.PROGRESS_LIMIT
        saved = list((manager._path(job["id"]) / "operations").glob("*/terminal.json"))
        assert len(saved) == 1
        assert saved[0].read_bytes() == stdout
    finally:
        manager.close()


def test_real_pythonhome_bootstrap_failure_reports_missing_terminal(physical_tmp, monkeypatch):
    app = jobs_module()
    inp, request = admission(physical_tmp)
    empty_home = physical_tmp / "empty-pythonhome"
    empty_home.mkdir()
    parent_environment = dict(os.environ)
    real_popen = subprocess.Popen

    def launch(_argv, **kwargs):
        environment = dict(kwargs["env"] if kwargs["env"] is not None else os.environ)
        environment["PYTHONHOME"] = str(empty_home)
        kwargs["env"] = environment
        return real_popen(
            [sys.executable, "-c", "raise AssertionError('bootstrap should fail first')"],
            **kwargs,
        )

    monkeypatch.setattr(app.subprocess, "Popen", launch)
    manager = app.JobManager(physical_tmp / "work")
    try:
        job = manager.admit(request)
        manager.upload(job["id"], io.BytesIO(inp.read_bytes()), inp.stat().st_size)
        with manager._lock:
            manager.start(job["id"], "validate")
            _, child, watcher = manager._active
        value = wait_job(manager, job["id"])
        watcher.join(timeout=5)
        assert not watcher.is_alive()
        assert manager._active is None
        assert child.returncode not in {None, 0}
        assert value["state"] == "failed"
        assert value["result"] is None
        assert "encodings" in value["progress"]
        assert "bootstrap should fail first" not in value["progress"]
        assert len(value["progress"]) <= app.PROGRESS_LIMIT
        assert not list((manager._path(job["id"]) / "operations").glob("*/terminal.json"))
        assert dict(os.environ) == parent_environment
        assert "Invalid UTF-8 JSON" not in value["message"]
        assert value["message"] == (
            f"The SelCal process exited with code {child.returncode} without returning "
            "a terminal result. See Progress and operation details for retained diagnostics."
        )
    finally:
        manager.close()


def test_finalized_progress_cannot_claim_completion_before_child_exit(physical_tmp, monkeypatch):
    app = jobs_module()
    inp, request = admission(physical_tmp)
    real_popen = subprocess.Popen

    def launch(_argv, **kwargs):
        return real_popen(
            [
                sys.executable,
                "-c",
                "import sys,time; print('finalized', file=sys.stderr, flush=True); time.sleep(30)",
            ],
            **kwargs,
        )

    monkeypatch.setattr(app.subprocess, "Popen", launch)
    manager = app.JobManager(physical_tmp / "work")
    try:
        job = manager.admit(request)
        manager.upload(job["id"], io.BytesIO(inp.read_bytes()), inp.stat().st_size)
        manager.start(job["id"], "run")
        deadline = time.monotonic() + 5
        while "finalized" not in manager.get(job["id"])["progress"]:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert manager.get(job["id"])["state"] == "running"
        assert manager.get(job["id"])["result"] is None
        manager.cancel(job["id"])
        assert manager.get(job["id"])["state"] == "interrupted"
    finally:
        manager.close()


def test_concurrent_start_admits_only_one_owned_child(physical_tmp, monkeypatch):
    app = jobs_module()
    inp, request = admission(physical_tmp)
    real_popen = subprocess.Popen
    children = []

    def launch(_argv, **kwargs):
        child = real_popen([sys.executable, "-c", "import time; time.sleep(30)"], **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(app.subprocess, "Popen", launch)
    manager = app.JobManager(physical_tmp / "work")
    outcomes = []
    barrier = threading.Barrier(3)
    job = manager.admit(request)
    manager.upload(job["id"], io.BytesIO(inp.read_bytes()), inp.stat().st_size)

    def start():
        barrier.wait()
        try:
            manager.start(job["id"], "validate")
            outcomes.append("started")
        except app.UIError:
            outcomes.append("busy")

    threads = [threading.Thread(target=start) for _ in range(2)]
    try:
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join(timeout=5)
        assert sorted(outcomes) == ["busy", "started"]
        assert len(children) == 1
    finally:
        manager.close()
    assert children[0].poll() is not None


def test_shutdown_reaps_owned_child_even_if_saved_metadata_becomes_invalid(
    physical_tmp, monkeypatch
):
    app = jobs_module()
    inp, request = admission(physical_tmp)
    real_popen = subprocess.Popen

    def launch(_argv, **kwargs):
        return real_popen([sys.executable, "-c", "import time; time.sleep(30)"], **kwargs)

    monkeypatch.setattr(app.subprocess, "Popen", launch)
    manager = app.JobManager(physical_tmp / "work")
    job = manager.admit(request)
    manager.upload(job["id"], io.BytesIO(inp.read_bytes()), inp.stat().st_size)
    manager.start(job["id"], "validate")
    child = manager._active[1]
    metadata = manager.workspace / "jobs" / job["id"] / "metadata.json"
    metadata.write_text("invalid", encoding="utf-8")
    try:
        manager.close()
        assert child.poll() is not None
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)


@pytest.mark.parametrize("statistic", ["lagged_pearson_v1", "equal_width_binned_nette_v1"])
@pytest.mark.parametrize(
    "null", ["circular_shift_v2", "circular_shift_exact_v1", "block_shuffle_v2"]
)
def test_all_existing_adapter_choices_dispatch_to_same_cli(physical_tmp, statistic, null):
    app = jobs_module()
    inp, request = admission(physical_tmp)
    config = json.loads(request["config_text"])
    plan = config["plan"]
    plan.update(
        statistic_name=statistic,
        statistic_params={} if statistic == "lagged_pearson_v1" else {"bins": 3},
        null_name=null,
        null_params={"block_length": 2} if null == "block_shuffle_v2" else {"min_shift": 1},
        replicates=5 if null == "circular_shift_exact_v1" else 9,
    )
    request["config_text"] = json.dumps(config)
    (physical_tmp / "config.json").write_text(request["config_text"], encoding="utf-8")
    manager = app.JobManager(physical_tmp / "work")
    try:
        job = manager.admit(request)
        manager.upload(job["id"], io.BytesIO(inp.read_bytes()), inp.stat().st_size)
        manager.start(job["id"], "run")
        actual = wait_job(manager, job["id"])
        direct = subprocess.run(
            [
                sys.executable,
                "-m",
                "selcal",
                "run",
                str(inp),
                str(physical_tmp / "config.json"),
                str(physical_tmp / "direct.sqlite"),
                "--max-bytes",
                "1048576",
                "--allow-unattainable-plan",
            ],
            capture_output=True,
            timeout=30,
        )
        expected = json.loads(direct.stdout)
        assert actual["result"]["exit_code"] == expected["exit_code"]
        data = actual["result"]["data"].copy()
        data.pop("checkpoint")
        assert data == expected["data"]
        assert expected["exit_code"] == 0
    finally:
        manager.close()
