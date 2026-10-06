"""Crash-fixture regressions; native handle claims require a Windows run."""

from __future__ import annotations

import importlib
import json
import os
import queue
import subprocess
import sys
import textwrap
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import test_checkpoint_store_crash as store_crash
import test_workflow_recovery_process as workflow_crash
from test_checkpoint_store import _case
from test_workflow import files


def helper():
    assert Path(__file__).with_name("_crash_process.py").is_file(), (
        "the crash fixture must retain and validate the actual writer identity"
    )
    return importlib.import_module("_crash_process")


def close_child(child):
    """Bounded teardown also works against the original plain-Popen fixture."""
    if hasattr(child, "cleanup"):
        child.cleanup(sys.exc_info()[1])
        return
    if child.poll() is None:
        child.kill()
    child.communicate(timeout=10)


def test_workflow_ready_reports_writer_identity_before_forced_kill(tmp_path):
    inp, cfg = files(tmp_path)
    child = workflow_crash.launch(inp, cfg, tmp_path / "output", tmp_path / "cp", "commit", 3)
    try:
        identity = workflow_crash.ready(child)
        assert isinstance(identity, dict), "READY must identify the blocked writer"
        assert identity["event"] == "READY"
        assert identity["boundary"] == "commit"
        assert identity["nonce"] == child.nonce
        assert type(identity["pid"]) is int and identity["pid"] > 0
        assert type(identity["ppid"]) is int and identity["ppid"] > 0
        if os.name == "nt":
            assert identity["created_100ns"] == child.writer_handle.snapshot()["created_100ns"]
            assert not child.writer_handle.wait(0)
        else:
            assert identity["pid"] == child.pid
            assert identity["created_100ns"] is None
        receipt = workflow_crash.kill(child)
        assert receipt["forced"] is True
        assert receipt["writer"]["pid"] == identity["pid"]
        assert receipt["writer"]["exit_code"] != 0
        assert receipt["launcher"]["pid"] == child.pid
        assert child.poll() is not None
    finally:
        close_child(child)


@pytest.mark.parametrize("barrier_input", ["", "X"], ids=["eof", "wrong-input"])
def test_workflow_barrier_failure_cannot_continue_the_workload(tmp_path, barrier_input):
    inp, cfg = files(tmp_path)
    output = tmp_path / "output"
    child = workflow_crash.launch(inp, cfg, output, tmp_path / "cp", "commit", 3)
    try:
        workflow_crash.ready(child)
        _, stderr = child.communicate(input=barrier_input, timeout=30)
        assert child.returncode != 0, "EOF/wrong input must not resume the workload"
        assert "CrashProtocolError" in stderr
        assert not output.exists()
        assert child.forced_receipt is None
        with pytest.raises(AssertionError, match=r"already exited|barrier input"):
            workflow_crash.kill(child)
    finally:
        close_child(child)


@pytest.mark.parametrize("barrier_input", ["", "X"], ids=["eof", "wrong-input"])
def test_store_barrier_failure_is_not_a_crash_or_successful_unwind(tmp_path, barrier_input):
    case = _case()
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    for name in ("header", "raw", "request"):
        (fixture / name).write_bytes(case[name])
    for i, payload in case["outcomes"]:
        (fixture / f"outcome{i}").write_bytes(payload)
    marker = tmp_path / "finally"
    code = (
        "from pathlib import Path\ntry:\n"
        + textwrap.indent(store_crash.CHILD, "    ")
        + f"\nfinally:\n    Path({str(marker)!r}).write_text('normal-unwind')\n"
    )
    child = subprocess.Popen(
        [sys.executable, "-u", "-c", code, str(fixture), str(tmp_path / "cp"), "1", "0"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            errors="replace",
        env={
            **os.environ,
            "SELCAL_CRASH_NONCE": "store-regression",
            "PYTHONPATH": os.pathsep.join(
                filter(None, [str(Path(__file__).parent), os.environ.get("PYTHONPATH")])
            ),
        },
    )
    messages = queue.Queue()
    reader = threading.Thread(
        target=store_crash._reader, args=(child.stdout, messages), daemon=True
    )
    reader.start()
    try:
        store_crash._ready(messages, "COMMITTED")
        child.stdin.write("G")
        child.stdin.flush()
        line = messages.get(timeout=30)
        assert line == "READY" or json.loads(line)["event"] == "READY"
        if barrier_input:
            child.stdin.write(barrier_input)
            child.stdin.flush()
        child.stdin.close()
        child.wait(timeout=30)
        assert child.returncode != 0, "the READY barrier must reject EOF/wrong input"
        assert marker.read_text(encoding="utf-8") == "normal-unwind"
        assert "CrashProtocolError" in child.stderr.read()
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=10)
        reader.join(timeout=5)
        child.stdout.close()
        child.stderr.close()


def test_external_forced_kill_does_not_run_finally_or_continue(tmp_path):
    module = helper()
    marker = tmp_path / "unwind"
    continued = tmp_path / "continued"
    code = f"""
from pathlib import Path
from _crash_process import crash_barrier
try:
    crash_barrier('negative-marker')
    Path({str(continued)!r}).write_text('continued')
finally:
    Path({str(marker)!r}).write_text('unwound')
"""
    with module.CrashProcess(code, [], boundary="negative-marker") as child:
        child.ready()
        receipt = child.force_kill()
        assert receipt["forced"] is True
    assert not marker.exists()
    assert not continued.exists()


@pytest.mark.parametrize("barrier_input", ["eof", "wrong-input"])
def test_protocol_failure_already_unwinding_cannot_be_counted_as_a_forced_crash(barrier_input):
    module = helper()
    code = """
import threading
from _crash_process import crash_barrier
try:
    crash_barrier('unwind')
finally:
    print('UNWIND', flush=True)
    threading.Event().wait()
"""
    with module.CrashProcess(code, [], boundary="unwind") as child:
        child.ready()
        if barrier_input == "eof":
            child.stdin.close()
        else:
            child.stdin.write("X")
            child.stdin.flush()
            assert not child.stdin.closed
        assert child.read_message() == "UNWIND"
        with pytest.raises(AssertionError, match="barrier input"):
            child.force_kill()
        assert child.forced_receipt is None


@pytest.mark.parametrize("platform", ["nt", "posix"])
def test_changed_barrier_input_refuses_both_os_paths_before_termination(monkeypatch, platform):
    """Shared protocol-state check only; this is not native Windows evidence."""
    module = helper()
    child = module.CrashProcess.__new__(module.CrashProcess)
    child._child_created = False
    child.identity = {"created_100ns": 123}
    child.forced_receipt = None
    child._barrier_input_changed = True
    child.stdin = SimpleNamespace(closed=False)
    child.poll = lambda: None

    def unexpected_native_access():
        pytest.fail("invalidated barrier must be refused before native access")

    child.send_signal = lambda sig: unexpected_native_access()
    child.writer_handle = SimpleNamespace(snapshot=unexpected_native_access)
    monkeypatch.setattr(module, "os", SimpleNamespace(name=platform))
    with pytest.raises(AssertionError, match="barrier input"):
        child.force_kill()
    assert child.forced_receipt is None


def test_repeated_recovery_persists_each_confirmed_writer_death(tmp_path):
    workflow_crash.test_interrupt_again_during_replay_and_append(tmp_path)
    path = tmp_path / "repeated_process_receipts.json"
    assert path.is_file(), "each repeated-recovery crash must retain its writer receipt"
    document = json.loads(path.read_text(encoding="utf-8"))
    events = document["interruptions"]
    assert [(e["mode"], e["boundary"], e["count"], e["retained"]) for e in events] == [
        ("run", "commit", 3, 3),
        ("resume", "replay", 2, 3),
        ("resume", "commit", 5, 5),
    ]
    assert len({event["crash"]["nonce"] for event in events}) == 3
    for event in events:
        receipt = event["crash"]
        assert receipt["forced"] is True
        assert receipt["writer"]["exit_code"] != 0
        assert receipt["launcher"]["exit_code"] is not None
    assert document["summary"]["replayed"] == 5
    assert document["summary"]["appended"] == 4


def test_recovery_failure_retains_preceding_confirmed_writer_death(tmp_path, monkeypatch):
    def failed_recovery(checkpoint):
        raise RuntimeError("injected recovery failure")

    monkeypatch.setattr(workflow_crash, "snapshot", failed_recovery)
    with pytest.raises(RuntimeError, match="injected recovery failure"):
        workflow_crash.test_interrupt_again_during_replay_and_append(tmp_path)
    path = tmp_path / "repeated_process_receipts.json"
    assert path.is_file(), "later recovery failure must preserve the preceding death evidence"
    document = json.loads(path.read_text(encoding="utf-8"))
    assert "summary" not in document
    assert len(document["interruptions"]) == 1
    receipt = document["interruptions"][0]["crash"]
    assert receipt["forced"] is True and receipt["writer"]["exit_code"] != 0


def test_creation_mismatch_and_close_error_preserve_identity_failure(monkeypatch):
    """Synthetic OS boundary only: it does not exercise native Windows handles."""
    module = helper()
    observations = []

    class MismatchedHandle:
        def __init__(self, pid, allow_terminate):
            assert allow_terminate
            observations.append(("open", pid))

        def snapshot(self):
            return {"created_100ns": 456, "signaled": False}

        def close(self):
            observations.append("close")
            raise OSError("synthetic CloseHandle failure")

    code = """
import json, sys
from _crash_process import ready_identity
identity = ready_identity('mismatch')
identity['created_100ns'] = 123
print(json.dumps(identity), flush=True)
sys.stdin.read(1)
"""
    with module.CrashProcess(
        code, [], boundary="mismatch", executable=getattr(sys, "_base_executable", sys.executable)
    ) as child:
        with monkeypatch.context() as scoped:
            scoped.setattr(module, "os", SimpleNamespace(name="nt", getpid=os.getpid))
            scoped.setattr(module, "ProcessHandle", MismatchedHandle)
            with pytest.raises(AssertionError, match="creation time mismatch") as failure:
                child.ready()
        assert "CloseHandle failure" in " ".join(failure.value.__notes__)
        assert observations == [("open", child.pid), "close"]
        assert child.writer_handle is None and child.forced_receipt is None


@pytest.mark.parametrize("failure", ["terminate-error", "wait-timeout", "zero-exit"])
def test_failed_native_termination_never_reaps_launcher_or_creates_receipt(monkeypatch, failure):
    """Synthetic failure seam; no result here constitutes native Windows validation."""
    module = helper()
    observations = []

    class ControlledHandle:
        def snapshot(self):
            exited = "wait" in observations
            return {"created_100ns": 123, "signaled": exited, "exit_code": 0 if exited else None}

        def terminate(self, code):
            observations.append("terminate")
            assert code == module.FORCED_EXIT_CODE
            if failure == "terminate-error":
                raise OSError("synthetic TerminateProcess failure")

        def wait(self, timeout):
            assert timeout == 30000
            observations.append("wait")
            return failure != "wait-timeout"

    child = module.CrashProcess.__new__(module.CrashProcess)
    child._child_created = False
    child.identity = {"created_100ns": 123}
    child.writer_handle = ControlledHandle()
    child.forced_receipt = None
    child._barrier_input_changed = False
    child.stdin = SimpleNamespace(closed=False)
    child.wait = lambda timeout: observations.append("reap-launcher")
    monkeypatch.setattr(module, "os", SimpleNamespace(name="nt"))
    error = OSError if failure == "terminate-error" else AssertionError
    with pytest.raises(error):
        child.force_kill()
    assert "reap-launcher" not in observations
    assert child.forced_receipt is None


@pytest.mark.parametrize("mutation", ["nonce", "boundary", "pid", "ppid", "created_100ns"])
def test_invalid_identity_is_refused_before_writer_termination(tmp_path, mutation):
    module = helper()
    code = f"""
import json, sys
from _crash_process import ready_identity
identity = ready_identity('identity')
field = {mutation!r}
identity[field] = 'invalid'
print(json.dumps(identity), flush=True)
sys.stdin.read(1)
"""
    with module.CrashProcess(code, [], boundary="identity") as child:
        with pytest.raises(AssertionError, match="identity"):
            child.ready()
        assert child.writer_handle is None
        assert child.forced_receipt is None


@pytest.mark.skipif(os.name != "nt", reason="Requires native Windows launcher and writer handles")
def test_windows_nested_launcher_terminates_real_writer_before_reaping_launcher(tmp_path):
    module = helper()
    marker = tmp_path / "normal-unwind"
    writer_code = f"""
from pathlib import Path
from _crash_process import crash_barrier
try:
    crash_barrier('nested')
finally:
    Path({str(marker)!r}).write_text('unwound')
"""
    interpreter = getattr(sys, "_base_executable", sys.executable)
    launcher_code = f"""
import subprocess, sys
child = subprocess.Popen([{interpreter!r}, '-u', '-c', {writer_code!r}])
sys.exit(child.wait())
"""
    with module.CrashProcess(
        launcher_code, [], boundary="nested", executable=interpreter
    ) as child:
        identity = child.ready()
        assert identity["pid"] != child.pid
        assert identity["ppid"] == child.pid
        handle = child.writer_handle
        assert handle.snapshot()["created_100ns"] == identity["created_100ns"]
        receipt = child.force_kill()
        assert handle.wait(0)
        assert receipt["writer"]["exit_code"] == module.FORCED_EXIT_CODE
        assert receipt["launcher"]["exit_code"] is not None
        assert not marker.exists()
