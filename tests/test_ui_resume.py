"""The UI delegates recovery to the real CLI and retains every attempt."""

from __future__ import annotations

import hashlib
import io
import json
import queue
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from contextlib import closing, contextmanager

import pytest
from test_ui_http import request as http_request
from test_ui_http import wait_while_progressing
from test_ui_jobs import admission, jobs_module
from test_ui_jobs import physical_tmp as _physical_tmp
from test_ui_process import wait_job

from selcal.workflow_store import read_record

physical_tmp = _physical_tmp


def saved_job(folder, *, form="csv", mode="complete"):
    inp, request = admission(folder, form=form, mode=mode)
    manager = jobs_module().JobManager(folder / "work")
    job = manager.admit(request)
    manager.upload(job["id"], io.BytesIO(inp.read_bytes()), inp.stat().st_size)
    return manager, job["id"], inp, request


@pytest.mark.parametrize(
    "mode", ["complete", "null_bind", "observed_statistic_scan", "replicate_execution"]
)
def test_resume_retains_execution_and_each_terminal_record(physical_tmp, mode):
    manager, job_id, _, request = saved_job(physical_tmp, mode=mode)
    try:
        manager.start(job_id, "run")
        original = wait_job(manager, job_id)
        path = manager.workspace / "jobs" / job_id
        record = next((path / "operations").glob("*/result.sqlite"))
        original_bytes = record.read_bytes()
        expected = read_record(record, max_bytes=int(request["max_bytes"]))
        for _ in range(2):
            manager.start(job_id, "resume")
            resumed = wait_job(manager, job_id)
            assert resumed["state"] == original["state"]
            assert resumed["result"]["command"] == "resume"
            assert resumed["result"]["exit_code"] == (0 if mode == "complete" else 7)
            checkpoint = resumed["result"]["data"]["checkpoint"]
            assert (
                checkpoint["execution_id"]
                == original["result"]["data"]["checkpoint"]["execution_id"]
            )
            assert checkpoint["appended_replicates"] == 0
        records = list((path / "operations").glob("*/result.sqlite"))
        assert len(records) == 3
        assert record.read_bytes() == original_bytes
        assert all(
            read_record(item, max_bytes=int(request["max_bytes"])) == expected for item in records
        )
    finally:
        manager.close()


@contextmanager
def helper(workspace):
    """Fresh product helper process, with its actual authenticated HTTP surface."""
    from types import SimpleNamespace

    process = subprocess.Popen(
        [sys.executable, "-m", "selcal", "ui", "--workspace", str(workspace), "--no-browser"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
    )
    try:
        lines = queue.Queue()
        threading.Thread(target=lambda: lines.put(process.stdout.readline()), daemon=True).start()
        # Hosted CI machines can take well over 10 s to start Python and import the helper
        # (macOS runners, 2026-10-06); this bounds waiting, not the behaviour checked.
        line = lines.get(timeout=60)
        match = re.fullmatch(r"SelCal local UI: http://127\.0\.0\.1:(\d+)/#token=([^\s]+)\n", line)
        assert match is not None, line
        surface = SimpleNamespace(server_port=int(match[1]), token=match[2], pid=process.pid)
        try:
            yield surface
        finally:
            _cancel_running(surface)
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGINT) if sys.platform != "win32" else process.terminate()
        try:
            process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate(timeout=5)


def _cancel_running(server):
    """Stop this helper's running jobs through its own cancel before the helper is ended.

    R17 item 5b: on Windows terminate() ends only the helper, so a child it was still running
    kept state.sqlite open and the temporary directory could not be removed.
    """
    try:
        status, _, body = http_request(server, "GET", "/api/jobs")
        running = [job["id"] for job in json.loads(body)["jobs"] if job["state"] == "running"]
    except (OSError, ValueError, KeyError):
        return
    for job_id in running:
        http_request(server, "POST", f"/api/jobs/{job_id}/cancel", b"")
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            status, _, body = http_request(server, "GET", f"/api/jobs/{job_id}")
            if status != 200 or json.loads(body)["state"] != "running":
                break
            time.sleep(0.05)


def poll(server, job_id, predicate):
    def fetch():
        status, _, body = http_request(server, "GET", f"/api/jobs/{job_id}")
        assert status == 200, body
        return json.loads(body)

    return wait_while_progressing(fetch, predicate, what="The real helper")


# Enough replicates that a cancel issued at the first observed commit usually lands mid-run;
# 199 finished before the cancel under load (NumPy 1.26 CI lane, 2026-10-04), and a late cancel
# retries in a fresh workspace below. 999 made every run take minutes where each committed
# replicate is slow (R16 Windows return, R17 item 5).
REPLICATES = 499


@pytest.mark.parametrize("form", ["csv", "npz"])
def test_real_committed_cancel_restart_and_repeated_resume_match_all_record_members(
    physical_tmp, form
):
    inp, request = admission(physical_tmp, form=form)
    config = json.loads(request["config_text"])
    config["plan"].update(replicates=REPLICATES, root_seed=9007199254740993)
    request.update(config_text=json.dumps(config), max_bytes="8388608")
    config_path = physical_tmp / "config.json"
    config_path.write_text(request["config_text"], encoding="utf-8")
    # Under heavy load the cancel can land after the last commit; that boundary is covered
    # deterministically in test_workflow_recovery. Here a mid-run cancel must be observed,
    # so a late cancel restarts in a fresh workspace rather than relaxing the assertion.
    attempts = []
    for attempt in range(3):
        workspace = physical_tmp / f"work{attempt}"
        with helper(workspace) as first:
            status, _, body = http_request(first, "POST", "/api/jobs", json.dumps(request))
            assert status == 201
            job_id = json.loads(body)["id"]
            upload = http_request(first, "PUT", f"/api/jobs/{job_id}/input", inp.read_bytes())
            assert upload[0] == 200
            assert http_request(first, "POST", f"/api/jobs/{job_id}/run", b"")[0] == 200
            progress = poll(first, job_id, lambda job: "commit; retained=" in job["progress"])
            execution = re.search(r"checkpoint ([^:]+): commit", progress["progress"])[1]
            assert http_request(first, "POST", f"/api/jobs/{job_id}/cancel", b"")[0] == 200
            stopped = poll(first, job_id, lambda job: job["state"] != "running")
            assert stopped["state"] == "interrupted"
        path = workspace / "jobs" / job_id
        checkpoint = path / "checkpoint" / "state.sqlite"
        with closing(sqlite3.connect(checkpoint)) as connection, connection:
            retained = connection.execute("SELECT count(*) FROM replicates").fetchone()[0]
            header = json.loads(
                connection.execute("SELECT payload FROM members WHERE name='header'").fetchone()[0]
            )
        # The connection context manages transactions, not handle lifetime.
        with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
            connection.execute("SELECT 1")
        attempts.append(retained)
        assert 0 < retained <= REPLICATES, attempts
        if retained < REPLICATES:
            break
    else:
        pytest.fail(f"cancel landed after the last commit in 3 fresh attempts: {attempts}")
    assert 0 < retained < REPLICATES
    assert header["execution_id"] == execution and header["max_bytes"] == 8388608
    before = {p: p.read_bytes() for p in (path / "operations").glob("*/*")}
    with helper(workspace) as second:
        assert second.pid != first.pid
        status, _, body = http_request(second, "GET", f"/api/jobs/{job_id}")
        assert status == 200
        restored = json.loads(body)
        assert restored["config_text"] == request["config_text"]
        assert restored["max_bytes"] == request["max_bytes"]
        assert restored["input_sha256"] == hashlib.sha256(inp.read_bytes()).hexdigest()
        assert http_request(second, "POST", f"/api/jobs/{job_id}/resume", b"")[0] == 200
        poll(second, job_id, lambda job: "replay; retained=" in job["progress"])
        # Cancel a real resumed child at an observed replay/commit boundary, then repeat.
        assert http_request(second, "POST", f"/api/jobs/{job_id}/cancel", b"")[0] == 200
        assert poll(second, job_id, lambda job: job["state"] != "running")["state"] == "interrupted"
        assert http_request(second, "POST", f"/api/jobs/{job_id}/resume", b"")[0] == 200
        completed = poll(second, job_id, lambda job: job["state"] != "running")
        assert completed["state"] == "complete", completed
        summary = completed["result"]["data"]["checkpoint"]
        assert summary["execution_id"] == execution
        assert summary["replayed_replicates"] >= retained
        assert summary["replayed_replicates"] + summary["appended_replicates"] == REPLICATES
        assert "replay;" in completed["progress"] and "commit;" in completed["progress"]
    assert all(p.read_bytes() == raw for p, raw in before.items())
    assert len(list((path / "operations").iterdir())) == 3
    direct = subprocess.run(
        [
            sys.executable,
            "-m",
            "selcal",
            "run",
            str(inp),
            str(config_path),
            str(physical_tmp / "direct.sqlite"),
            "--max-bytes",
            request["max_bytes"],
            "--allow-unattainable-plan",
        ],
        capture_output=True,
        timeout=45,
    )
    assert direct.returncode == 0, direct.stdout
    expected_summary = json.loads(direct.stdout)["data"]
    actual_summary = completed["result"]["data"].copy()
    actual_summary.pop("checkpoint")
    assert actual_summary == expected_summary
    expected = read_record(physical_tmp / "direct.sqlite", max_bytes=8388608)
    assert len(expected) == 4
    records = list((path / "operations").glob("*/result.sqlite"))
    assert len(records) == 1
    assert read_record(records[0], max_bytes=8388608) == expected


@pytest.mark.parametrize("damage", ["missing", "corrupt", "python_version", "source"])
def test_cli_resume_refuses_checkpoint_damage_or_identity_and_retains_files(physical_tmp, damage):
    manager, job_id, _, _ = saved_job(physical_tmp)
    try:
        path = manager.workspace / "jobs" / job_id
        if damage != "missing":
            manager.start(job_id, "run")
            assert wait_job(manager, job_id)["state"] == "complete"
            database = path / "checkpoint" / "state.sqlite"
            if damage == "corrupt":
                database.write_bytes(b"damaged checkpoint")
            else:
                with closing(sqlite3.connect(database)) as connection, connection:
                    header = json.loads(
                        connection.execute(
                            "SELECT payload FROM members WHERE name='header'"
                        ).fetchone()[0]
                    )
                    if damage == "python_version":
                        header["software"]["python_version"] = "different"
                    else:
                        source_key = next(iter(header["software"]["source_files"]))
                        header["software"]["source_files"][source_key] = "1" * 64
                    raw = (
                        json.dumps(header, sort_keys=True, separators=(",", ":"), allow_nan=False)
                        + "\n"
                    ).encode()
                    connection.execute(
                        "UPDATE members SET payload=?, sha256=? WHERE name='header'",
                        (raw, hashlib.sha256(raw).hexdigest()),
                    )
                with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
                    connection.execute("SELECT 1")
        before = {
            p: p.read_bytes()
            for p in path.rglob("*")
            if p.is_file() and p.name != "observation.json"
        }
        original = manager.get(job_id)
        operations = list((path / "operations").iterdir())
        with pytest.raises(jobs_module().UIError, match=r"Checkpoint.*refused") as failure:
            manager.start(job_id, "resume")
        if damage in {"python_version", "source"}:
            assert "software identity differs" in str(failure.value)
        assert manager._active is None
        assert manager.get(job_id) == original
        assert list((path / "operations").iterdir()) == operations
        assert all(p.read_bytes() == raw for p, raw in before.items())
    finally:
        manager.close()


@pytest.mark.parametrize(
    "producer", ["finalized", "wrong_command", "wrong_exit", "two_json", "overflow"]
)
def test_resume_child_terminal_and_lifecycle_boundaries(physical_tmp, monkeypatch, producer):
    manager, job_id, _, _ = saved_job(physical_tmp)
    try:
        manager.start(job_id, "run")
        original = wait_job(manager, job_id)
        assert original["state"] == "complete"
        terminal = original["result"].copy()
        terminal["command"] = "resume"
        if producer == "wrong_command":
            terminal["command"] = "run"
        elif producer == "wrong_exit":
            terminal["exit_code"] = 7
        script = "print(" + repr(json.dumps(terminal)) + ")"
        if producer == "two_json":
            script += "; " + script
        elif producer == "overflow":
            script = "print('x' * 3000000)"
        elif producer == "finalized":
            script = (
                "import sys,time; print('finalized',file=sys.stderr,flush=True); time.sleep(30)"
            )
        real_popen = subprocess.Popen
        calls = []

        def launch(argv, **kwargs):
            calls.append(argv)
            assert kwargs["shell"] is False
            return real_popen([argv[0], "-c", script], **kwargs)

        monkeypatch.setattr(jobs_module().subprocess, "Popen", launch)
        manager.start(job_id, "resume")
        child = manager._active[1]
        path = manager.workspace / "jobs" / job_id
        executable = jobs_module()._windows_image() if sys.platform == "win32" else sys.executable
        # -I already leaves the working directory off sys.path; otherwise -P does (R17 item 2).
        flags = (["-I"] if sys.flags.isolated
                 else ["-E", "-P"] if sys.flags.ignore_environment else ["-P"])
        assert calls[0][:5 + len(flags)] == [
            executable, *flags, "-m", "selcal", "resume", str(path / "checkpoint")
        ]
        assert calls[0][6 + len(flags):] == ["--max-bytes", "1048576"]
        assert len(list((path / "operations").iterdir())) == 2
        if producer == "finalized":
            deadline = time.monotonic() + 10
            while "finalized" not in manager.get(job_id)["progress"]:
                assert time.monotonic() < deadline
                time.sleep(0.005)
            assert manager.get(job_id)["state"] == "running"
            assert manager.get(job_id)["result"] is None
            with pytest.raises(jobs_module().UIError, match="active"):
                manager.start(job_id, "resume")
            manager.close()
            assert manager.get(job_id)["state"] == "interrupted"
        else:
            refused = wait_job(manager, job_id)
            assert refused["state"] == "failed" and refused["result"] is None
        assert child.poll() is not None
        assert len(calls) == 1
    finally:
        manager.close()


@pytest.mark.parametrize("difference", ["input", "config", "budget", "override"])
def test_resume_refuses_another_valid_jobs_checkpoint(physical_tmp, difference):
    inp, request = admission(physical_tmp)
    if difference == "override":
        config = json.loads(request["config_text"])
        config["plan"].update(alpha=0.5)
        request.update(config_text=json.dumps(config))
    manager = jobs_module().JobManager(physical_tmp / "work")
    job_id = manager.admit(request)["id"]
    manager.upload(job_id, io.BytesIO(inp.read_bytes()), inp.stat().st_size)
    try:
        if difference == "config":
            config = json.loads(request["config_text"])
            config["plan"]["root_seed"] += 1
            request["config_text"] = json.dumps(config)
        elif difference == "budget":
            request["max_bytes"] = "8388608"
        elif difference == "override":
            request["allow_unattainable"] = False
        other = manager.admit(request)
        raw = inp.read_bytes()
        if difference == "input":
            raw = raw.replace(b"0,0", b"2,0")
        manager.upload(other["id"], io.BytesIO(raw), len(raw))
        manager.start(other["id"], "run")
        finished = wait_job(manager, other["id"])
        assert finished["state"] == "complete", finished
        jobs = manager.workspace / "jobs"
        shutil.copytree(jobs / other["id"] / "checkpoint", jobs / job_id / "checkpoint")
        copied = jobs / job_id / "checkpoint" / "state.sqlite"
        before = copied.read_bytes()
        with pytest.raises(jobs_module().UIError, match=r"Checkpoint.*refused"):
            manager.start(job_id, "resume")
        assert manager._active is None and copied.read_bytes() == before
    finally:
        manager.close()
