"""Direct Windows worker ownership without losing the installed environment."""

from __future__ import annotations

import ctypes
import hashlib
import io
import json
import os
import subprocess
import sys
import threading
from ctypes import wintypes
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_ui_jobs import admission, jobs_module
from test_ui_jobs import physical_tmp as _physical_tmp
from test_ui_process import wait_job
from test_ui_verify import files_at, saved_job

physical_tmp = _physical_tmp


def assert_stopped_downloads(manager, job_id, stop, before, references, downloads):
    """Cancellation keeps this helper usable; shutdown requires a new helper."""
    if stop == "cancel":
        assert {kind: manager.download(job_id, kind) for kind in downloads} == downloads
        return

    app = jobs_module()
    for kind in downloads:
        with pytest.raises(app.UIError) as refused:
            manager.download(job_id, kind)
        assert refused.value.status == 409
    reopened = app.JobManager(manager.workspace)
    try:
        observed = reopened.get(job_id)
        assert observed["state"] == "interrupted"
        assert observed["saved_observation_only"] is True
        path = reopened.workspace / "jobs" / job_id
        assert {
            name: (path / f"{name}.json").read_bytes() for name in references
        } == references
        assert {kind: reopened.download(job_id, kind) for kind in downloads} == downloads
    finally:
        reopened.close()
    assert all(
        (path / "operations" / name).read_bytes() == raw for name, raw in before.items()
    )
    assert {name: (path / f"{name}.json").read_bytes() for name in references} == references


def windows_runtime(monkeypatch, folder):
    app = jobs_module()
    image, launcher = folder / "base-python.exe", folder / "venv-python.exe"
    image.write_bytes(b"base image fixture")
    launcher.write_bytes(b"venv launcher fixture")
    runtime = SimpleNamespace(
        platform="win32",
        implementation=SimpleNamespace(name="cpython"),
        executable=str(launcher),
        _base_executable=str(image),
        flags=SimpleNamespace(isolated=0, ignore_environment=0),
    )
    monkeypatch.setattr(app, "sys", runtime)
    # The RED reaches start's existing launch boundary before this helper exists.
    monkeypatch.setattr(app, "_windows_image", lambda: str(image), raising=False)
    return app, runtime, image, launcher


@pytest.mark.parametrize(
    "isolated,ignore_environment,flags", [(0, 0, ["-P"]), (0, 1, ["-E", "-P"]), (1, 1, ["-I"])])
def test_start_owns_native_image_and_passes_copied_venv_environment(
    physical_tmp, monkeypatch, isolated, ignore_environment, flags
):
    app, runtime, image, launcher = windows_runtime(monkeypatch, physical_tmp)
    runtime.flags = SimpleNamespace(isolated=isolated, ignore_environment=ignore_environment)
    inp, request = admission(physical_tmp)
    manager = app.JobManager(physical_tmp / "work")
    calls = []
    before = dict(os.environ)

    def launch(argv, **kwargs):
        calls.append((argv, kwargs))
        raise OSError("capture only: no process created")

    monkeypatch.setattr(app.subprocess, "Popen", launch)
    try:
        job_id = manager.admit(request)["id"]
        manager.upload(job_id, io.BytesIO(inp.read_bytes()), inp.stat().st_size)
        with pytest.raises(app.UIError, match="could not start"):
            manager.start(job_id, "validate")
        assert len(calls) == 1
        argv, options = calls[0]
        assert argv[0] == str(image)
        assert argv[:len(flags) + 4] == [str(image), *flags, "-m", "selcal", "validate"]
        assert options["executable"] == str(image)
        assert options["env"] == {**before, "__PYVENV_LAUNCHER__": str(launcher)}
        assert options["env"] is not os.environ
        assert options["shell"] is False
        assert dict(os.environ) == before
        assert manager.get(job_id)["state"] == "failed"
        assert manager._active is None
    finally:
        manager.close()


@pytest.mark.parametrize("platform", ["darwin", "linux"])
def test_non_windows_preserves_defaults_without_native_query(monkeypatch, platform):
    app = jobs_module()
    monkeypatch.setattr(app, "sys", SimpleNamespace(platform=platform))

    def forbidden():
        pytest.fail("Non-Windows must not query a Windows native image")

    monkeypatch.setattr(app, "_windows_image", forbidden, raising=False)
    assert app._child_runtime() == (None, None)


@pytest.mark.parametrize("platform", ["darwin", "linux"])
@pytest.mark.parametrize(
    "isolated,ignore_environment,flags", [(0, 0, ["-P"]), (0, 1, ["-E", "-P"]), (1, 1, ["-I"])])
def test_non_windows_start_preserves_parent_policy(
    physical_tmp, monkeypatch, platform, isolated, ignore_environment, flags
):
    app = jobs_module()
    monkeypatch.setattr(app, "sys", SimpleNamespace(
        platform=platform, executable=sys.executable,
        flags=SimpleNamespace(isolated=isolated, ignore_environment=ignore_environment),
    ))

    def forbidden():
        pytest.fail("Non-Windows must not query a Windows native image")

    monkeypatch.setattr(app, "_windows_image", forbidden)
    calls = []
    before = dict(os.environ)
    before_path, before_cwd = list(sys.path), os.getcwd()

    def launch(argv, **kwargs):
        calls.append((argv, kwargs))
        raise OSError("capture only: no process created")

    monkeypatch.setattr(app.subprocess, "Popen", launch)
    inp, request = admission(physical_tmp)
    manager = app.JobManager(physical_tmp / "work")
    try:
        job_id = manager.admit(request)["id"]
        manager.upload(job_id, io.BytesIO(inp.read_bytes()), inp.stat().st_size)
        with pytest.raises(app.UIError, match="could not start"):
            manager.start(job_id, "validate")
        assert len(calls) == 1
        argv, options = calls[0]
        assert argv[:len(flags) + 4] == [sys.executable, *flags, "-m", "selcal", "validate"]
        assert options["executable"] is None
        assert options["env"] is None
        assert options["shell"] is False
        assert dict(os.environ) == before
        assert sys.path == before_path and os.getcwd() == before_cwd
        assert manager.get(job_id)["state"] == "failed"
        assert manager._active is None
    finally:
        manager.close()


@pytest.mark.parametrize("parent_flag", ["-I", "-E"])
@pytest.mark.parametrize("pollution", ["PYTHONHOME", "PYTHONPATH"])
def test_actual_child_inherits_parent_isolation(physical_tmp, parent_flag, pollution):
    inp, request = admission(physical_tmp)
    # This process-plumbing fixture must pass validate's attainable-plan gate.
    plan = json.loads(request["config_text"])
    plan["plan"]["alpha"] = 0.5
    request["config_text"] = json.dumps(plan)
    config = physical_tmp / "parent-request.json"
    config.write_text(request["config_text"], encoding="utf-8")
    poison = physical_tmp / "poison"
    poison.mkdir()
    if pollution == "PYTHONPATH":
        (poison / "selcal.py").write_text(
            "raise RuntimeError('F02 import trap')\n", encoding="utf-8")
    before = dict(os.environ)
    environment = {**before, pollution: str(poison)}
    script = '''
import io, json, os, sys, time
from pathlib import Path
import selcal
from selcal import ui_jobs
before = dict(os.environ)
before_path, before_cwd = list(sys.path), os.getcwd()
config, source, workspace = map(Path, sys.argv[1:])
manager = ui_jobs.JobManager(workspace)
original = ui_jobs.subprocess.Popen
children, calls = [], []
def launch(argv, **kwargs):
    child = original(argv, **kwargs)
    children.append(child)
    calls.append((list(argv), kwargs["executable"]))
    return child
ui_jobs.subprocess.Popen = launch
try:
    job_id = manager.admit({
        "config_text": config.read_text(), "max_bytes": "1048576",
        "allow_unattainable": True,
    })["id"]
    manager.upload(job_id, io.BytesIO(source.read_bytes()), source.stat().st_size)
    with manager._lock:
        manager.start(job_id, "validate")
        active = manager._active
        assert active is not None
        watcher = active[2]
    deadline = time.monotonic() + 30
    while True:
        with manager._lock:
            job = manager.get(job_id)
            finished = job["state"] != "running" and manager._active is None
        if finished:
            break
        assert time.monotonic() < deadline, "worker did not finish"
        time.sleep(0.01)
    watcher.join(timeout=5)
    assert not watcher.is_alive()
    assert job["state"] == "validated", json.dumps(job)
    assert job["result"]["command"] == "validate", job
    assert job["result"]["exit_code"] == 0, job
    assert len(children) == len(calls) == 1
    assert children[0].returncode == 0
    assert dict(os.environ) == before
    assert sys.path == before_path and os.getcwd() == before_cwd
    print(json.dumps({
        "parentflags": {"isolated": sys.flags.isolated,
                        "ignore_environment": sys.flags.ignore_environment},
        "actualargv": calls[0][0], "executable": calls[0][1],
        "module_origin": ui_jobs.__file__, "state": job["state"],
    }))
finally:
    manager.close()
    ui_jobs.subprocess.Popen = original
'''
    result = subprocess.run(
        [sys.executable, parent_flag, "-c", script,
         str(config), str(inp), str(physical_tmp / "work")],
        env=environment, shell=False, capture_output=True, text=True, errors="replace", timeout=45,
    )
    if result.returncode:
        print(result.stdout, result.stderr)
    assert result.returncode == 0, (result.stdout, result.stderr)
    observed = json.loads(result.stdout)
    print(result.stdout)
    assert observed["parentflags"] == {
        "isolated": int(parent_flag == "-I"), "ignore_environment": 1}
    # -I already leaves the working directory off sys.path; with -E the child adds -P (R17).
    flags = [parent_flag] if parent_flag == "-I" else [parent_flag, "-P"]
    assert observed["actualargv"][1:len(flags) + 4] == [*flags, "-m", "selcal", "validate"]
    assert observed["state"] == "validated"
    assert dict(os.environ) == before


def test_child_environment_is_copied_without_mutating_parent(physical_tmp, monkeypatch):
    app, _, image, launcher = windows_runtime(monkeypatch, physical_tmp)
    monkeypatch.setenv("__PYVENV_LAUNCHER__", "retained-parent-value")
    before = dict(os.environ)
    before_path = list(sys.path)
    executable, environment = app._child_runtime()
    assert executable == str(image)
    assert environment == {**before, "__PYVENV_LAUNCHER__": str(launcher)}
    assert environment is not os.environ
    environment["__PYVENV_LAUNCHER__"] = "child-only change"
    assert dict(os.environ) == before
    assert sys.path == before_path


@pytest.mark.parametrize(
    "damage",
    [
        "implementation",
        "frozen",
        "missing_executable",
        "empty_executable",
        "relative_executable",
        "missing_base",
        "empty_base",
        "relative_base",
        "missing_base_file",
        "different_image",
        "native_query",
    ],
)
def test_unsupported_runtime_refuses_start_and_records_failed_launch(
    physical_tmp, monkeypatch, damage
):
    app, runtime, image, launcher = windows_runtime(monkeypatch, physical_tmp)
    if damage == "implementation":
        runtime.implementation.name = "pypy"
    elif damage == "frozen":
        runtime.frozen = True
    elif damage == "missing_executable":
        del runtime.executable
    elif damage == "empty_executable":
        runtime.executable = ""
    elif damage == "relative_executable":
        runtime.executable = "python.exe"
    elif damage == "missing_base":
        del runtime._base_executable
    elif damage == "empty_base":
        runtime._base_executable = ""
    elif damage == "relative_base":
        runtime._base_executable = "python.exe"
    elif damage == "missing_base_file":
        image.unlink()
    elif damage == "different_image":
        monkeypatch.setattr(app, "_windows_image", lambda: str(launcher))
    else:

        def failed_query():
            raise OSError("native image query failed")

        monkeypatch.setattr(app, "_windows_image", failed_query)

    def forbidden(*args, **kwargs):
        pytest.fail("Unsupported Windows runtime must not launch any process")

    monkeypatch.setattr(app.subprocess, "Popen", forbidden)
    with pytest.raises(OSError):
        app._child_runtime()
    inp, request = admission(physical_tmp)
    manager = app.JobManager(physical_tmp / "work")
    try:
        job_id = manager.admit(request)["id"]
        manager.upload(job_id, io.BytesIO(inp.read_bytes()), inp.stat().st_size)
        with pytest.raises(app.UIError, match="could not start"):
            manager.start(job_id, "validate")
        assert manager._active is None
        assert manager.get(job_id)["state"] == "failed"
        observed = manager.workspace / "jobs" / job_id / "observation.json"
        assert json.loads(observed.read_bytes())["state"] == "failed"
    finally:
        manager.close()


@pytest.mark.parametrize("result", ["success", "failure", "truncated"])
def test_native_image_query_uses_typed_current_module_call(monkeypatch, result):
    app = jobs_module()
    monkeypatch.setattr(app, "sys", SimpleNamespace(platform="win32"))
    expected = r"C:\Program Files\Python312\python.exe"

    class Query:
        argtypes = None
        restype = None

        def __call__(self, module, buffer, capacity):
            assert module is None
            assert capacity == len(buffer) == 32768
            buffer.value = expected
            return {"success": len(expected), "failure": 0, "truncated": capacity}[result]

    query = Query()

    def library(name, *, use_last_error):
        assert name == "kernel32" and use_last_error is True
        return SimpleNamespace(GetModuleFileNameW=query)

    monkeypatch.setattr(ctypes, "WinDLL", library, raising=False)
    if result == "success":
        assert app._windows_image() == expected
    else:
        with pytest.raises(OSError, match="image"):
            app._windows_image()
    assert query.argtypes == [wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]
    assert query.restype is wintypes.DWORD


@pytest.mark.skipif(sys.platform != "win32", reason="Requires native Windows process identity")
def test_native_child_is_owned_pid_with_same_installed_environment():
    import selcal

    app = jobs_module()
    image, environment = app._child_runtime()
    script = (
        "import hashlib,json,os,sys; from pathlib import Path; import selcal,selcal.ui_jobs; "
        "print(json.dumps(dict(pid=os.getpid(),prefix=sys.prefix,executable=sys.executable,"
        "base=sys._base_executable,package=selcal.__file__,module=selcal.ui_jobs.__file__,"
        "source_sha256=hashlib.sha256(Path(selcal.ui_jobs.__file__).read_bytes()).hexdigest())))"
    )
    child = subprocess.Popen(
        [image, "-c", script],
        executable=image,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        stdout, stderr = child.communicate(timeout=10)
        assert child.returncode == 0, stderr
        identity = json.loads(stdout)
        assert identity["pid"] == child.pid
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)
    assert os.path.samefile(identity["prefix"], sys.prefix)
    assert os.path.samefile(identity["executable"], sys.executable)
    assert os.path.samefile(identity["base"], app._windows_image())
    assert os.path.samefile(identity["package"], selcal.__file__)
    assert os.path.samefile(identity["module"], app.__file__)
    assert identity["source_sha256"] == hashlib.sha256(Path(app.__file__).read_bytes()).hexdigest()


@pytest.mark.skipif(sys.platform != "win32", reason="Requires native Windows CREATE_SUSPENDED")
@pytest.mark.parametrize("action", ["report", "export"])
@pytest.mark.parametrize("stop", ["cancel", "close"])
def test_native_stop_before_python_initialization_reaps_only_owned_worker(
    physical_tmp, monkeypatch, action, stop
):
    app = jobs_module()
    image, environment = app._child_runtime()
    real_popen = subprocess.Popen
    sentinel = real_popen(
        [image, "-c", "import time; time.sleep(60)"],
        executable=image,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        with saved_job(physical_tmp) as (manager, job_id, _):
            for previous_action in ("report", "export"):
                manager.start(job_id, previous_action)
                assert wait_job(manager, job_id)["state"] == "complete"
            path = manager.workspace / "jobs" / job_id
            before = files_at(path / "operations")
            references = {
                name: (path / f"{name}.json").read_bytes()
                for name in ("record", "report", "bundle")
            }
            downloads = {
                kind: manager.download(job_id, kind) for kind in ("record", "report", "bundle")
            }

            def suspended_worker(argv, **kwargs):
                assert os.path.samefile(argv[0], image)
                assert os.path.samefile(kwargs["executable"], image)
                # Suspend only the directly owned worker before Python initialization.
                kwargs["creationflags"] = kwargs.get("creationflags", 0) | 0x00000004
                return real_popen(argv, **kwargs)

            with monkeypatch.context() as patch:
                patch.setattr(app.subprocess, "Popen", suspended_worker)
                manager.start(job_id, action)
            _, child, watcher = manager._active
            assert child.poll() is None
            assert sentinel.poll() is None
            manager.cancel(job_id) if stop == "cancel" else manager.close()
            assert child.poll() is not None and child.returncode != 0
            assert not watcher.is_alive()
            assert manager._active is None
            assert manager.get(job_id)["state"] == "interrupted"
            assert sentinel.poll() is None
            assert all(
                (path / "operations" / name).read_bytes() == raw for name, raw in before.items()
            )
            assert all(
                (path / f"{name}.json").read_bytes() == raw for name, raw in references.items()
            )
            assert_stopped_downloads(manager, job_id, stop, before, references, downloads)
    finally:
        if sentinel.poll() is None:
            sentinel.kill()
        sentinel.wait(timeout=5)


@pytest.mark.parametrize("action", ["report", "export"])
@pytest.mark.parametrize("stop", ["cancel", "close"])
def test_real_stopped_artifact_download_contract(physical_tmp, monkeypatch, action, stop):
    """Portable publication lifecycle; not Windows pre-initialization termination proof."""
    with saved_job(physical_tmp) as (manager, job_id, _):
        for previous_action in ("report", "export"):
            manager.start(job_id, previous_action)
            assert wait_job(manager, job_id)["state"] == "complete"
        path = manager.workspace / "jobs" / job_id
        before = files_at(path / "operations")
        old_operations = set((path / "operations").iterdir())
        references = {
            name: (path / f"{name}.json").read_bytes()
            for name in ("record", "report", "bundle")
        }
        downloads = {
            kind: manager.download(job_id, kind) for kind in ("record", "report", "bundle")
        }
        drained, release, stop_processed = (threading.Event() for _ in range(3))
        real_join = threading.Thread.join
        stop_errors = []

        def joined(thread, timeout=None):
            active = manager._active
            if active and threading.current_thread().name == "download-contract-stop":
                if thread is active[2]:
                    stop_processed.set()
            result = real_join(thread, timeout)
            active = manager._active
            if active and threading.current_thread() is active[2] and not drained.is_set():
                # Pause scheduling only after the actual CLI exits, before publication.
                assert active[1].poll() == 0
                drained.set()
                assert release.wait(10), "scheduler release missing"
            return result

        def stopping():
            try:
                manager.cancel(job_id) if stop == "cancel" else manager.close()
            except BaseException as error:
                stop_errors.append(error)

        with monkeypatch.context() as patch:
            patch.setattr(threading.Thread, "join", joined)
            stopper = None
            try:
                manager.start(job_id, action)
                assert drained.wait(20), "actual CLI did not exit"
                _, child, watcher = manager._active
                assert child.poll() == 0
                assert manager.get(job_id)["state"] == "running"
                stopper = threading.Thread(name="download-contract-stop", target=stopping)
                stopper.start()
                assert stop_processed.wait(5), "stop did not reach the publication window"
            finally:
                release.set()
                if stopper is None:
                    manager.close()
                else:
                    real_join(stopper, 10)
            assert not stopper.is_alive() and not stop_errors

        assert child.poll() == 0
        assert not watcher.is_alive()
        assert manager._active is None
        assert manager.get(job_id)["state"] == "interrupted"
        assert all(
            (path / "operations" / name).read_bytes() == raw for name, raw in before.items()
        )
        assert all(
            (path / f"{name}.json").read_bytes() == raw for name, raw in references.items()
        )
        (new_operation,) = set((path / "operations").iterdir()) - old_operations
        assert json.loads((new_operation / "terminal.json").read_bytes())["command"] == action
        assert_stopped_downloads(manager, job_id, stop, before, references, downloads)


def test_reopened_saved_observation_does_not_authorize_changed_input(physical_tmp):
    with saved_job(physical_tmp) as (manager, job_id, _):
        for action in ("report", "export"):
            manager.start(job_id, action)
            assert wait_job(manager, job_id)["state"] == "complete"
        path = manager.workspace / "jobs" / job_id
        downloads = {
            kind: manager.download(job_id, kind) for kind in ("record", "report", "bundle")
        }
        observation = (path / "observation.json").read_bytes()
        manager.close()
        reopened = jobs_module().JobManager(manager.workspace)
        try:
            observed = reopened.get(job_id)
            assert observed["state"] == "complete"
            assert observed["saved_observation_only"] is True
            assert {kind: reopened.download(job_id, kind) for kind in downloads} == downloads
            (path / "input.csv").write_bytes(b"changed input after reopening")
            for kind in downloads:
                with pytest.raises(jobs_module().UIError, match="input changed"):
                    reopened.download(job_id, kind)
            assert reopened.get(job_id)["state"] == "complete"
            assert (path / "observation.json").read_bytes() == observation
        finally:
            reopened.close()
