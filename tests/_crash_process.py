"""Test-only external crash barrier, bound to the actual blocked writer.

The Windows venv launcher can be a different process from the SQLite writer.
Only an identity-checked retained handle may terminate that writer. POSIX keeps
the direct-child SIGKILL path. Protocol failure unwinds and is never a crash.
"""

from __future__ import annotations

import json
import os
import queue
import secrets
import signal
import subprocess
import sys
import threading
from pathlib import Path

from _windows_process import ProcessHandle

FORCED_EXIT_CODE = 0x5343


class CrashProtocolError(RuntimeError):
    """Input arrived at a barrier that only external process death may release."""


def ready_identity(boundary):
    created = None
    if os.name == "nt":
        with ProcessHandle(os.getpid()) as process:
            created = process.snapshot()["created_100ns"]
    return {
        "event": "READY",
        "nonce": os.environ["SELCAL_CRASH_NONCE"],
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "created_100ns": created,
        "boundary": boundary,
    }


def crash_barrier(boundary):
    print(json.dumps(ready_identity(boundary)), flush=True)
    value = sys.stdin.read(1)
    raise CrashProtocolError(f"crash barrier received {'EOF' if not value else 'unexpected input'}")


class _BarrierInput:
    """Invalidate accepted READY before parent input can release its blocked writer."""

    def __init__(self, stream, owner):
        self._stream, self._owner = stream, owner

    def _changed(self):
        if self._owner.identity is not None:
            self._owner._barrier_input_changed = True

    def write(self, value):
        self._changed()
        return self._stream.write(value)

    def writelines(self, values):
        self._changed()
        return self._stream.writelines(values)

    def close(self):
        self._changed()
        return self._stream.close()

    def fileno(self):
        # Raw descriptor access could bypass write(), so it also invalidates READY.
        self._changed()
        return self._stream.fileno()

    def __getattr__(self, name):
        if name in {"buffer", "raw", "detach"}:
            self._changed()
        return getattr(self._stream, name)


class CrashProcess(subprocess.Popen):
    """Own the launcher and, once READY is checked, the real writer handle."""

    def __init__(self, code, arguments, *, boundary, executable=None):
        self.nonce = secrets.token_hex(32)
        self.boundary = boundary
        self.writer_handle = None
        self.identity = None
        self.forced_receipt = None
        self._barrier_input_changed = False
        self._readers = []
        env = {**os.environ, "SELCAL_CRASH_NONCE": self.nonce}
        helper_path = str(Path(__file__).resolve().parent)
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [helper_path, env.get("PYTHONPATH")]))
        super().__init__(
            [executable or sys.executable, "-u", "-c", code, *map(str, arguments)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, env=env,
        )
        self.stdin = _BarrierInput(self.stdin, self)

    def communicate(self, input=None, timeout=None):
        # Popen may write via os.write() or close stdin itself. Either releases READY.
        if self.identity is not None:
            self._barrier_input_changed = True
        return super().communicate(input=input, timeout=timeout)

    def read_message(self):
        messages = queue.Queue()
        reader = threading.Thread(target=lambda: messages.put(self.stdout.readline()), daemon=True)
        self._readers.append(reader)
        reader.start()
        try:
            line = messages.get(timeout=30)
        except queue.Empty as error:
            raise AssertionError("child did not reach its barrier within 30 seconds") from error
        reader.join(timeout=1)
        return line.rstrip("\n")

    def ready(self):
        assert self.identity is None, "writer identity already accepted"
        line = self.read_message()
        try:
            identity = json.loads(line)
        except ValueError as error:
            raise AssertionError(f"invalid READY identity: {line!r}") from error
        assert isinstance(identity, dict), "invalid READY identity object"
        assert identity.get("event") == "READY", "invalid READY identity event"
        assert identity.get("nonce") == self.nonce, "invalid READY identity nonce"
        assert identity.get("boundary") == self.boundary, "invalid READY identity boundary"
        pid, ppid = identity.get("pid"), identity.get("ppid")
        assert type(pid) is int and 0 < pid <= 0xFFFFFFFF, "invalid READY identity pid"
        assert type(ppid) is int and 0 < ppid <= 0xFFFFFFFF, "invalid READY identity ppid"
        assert pid != os.getpid(), "invalid READY identity: writer is the parent"
        if os.name == "nt":
            assert (pid == self.pid and ppid == os.getpid()) or (
                pid != self.pid and ppid == self.pid
            ), "invalid READY identity: writer is unrelated to launcher"
            created = identity.get("created_100ns")
            assert type(created) is int and created > 0, "invalid READY identity creation time"
            candidate = ProcessHandle(pid, allow_terminate=True)
            try:
                actual = candidate.snapshot()
                assert actual["created_100ns"] == created, "READY identity creation time mismatch"
                assert actual["signaled"] is False, "READY identity writer already exited"
            except BaseException as error:
                try:
                    candidate.close()
                except OSError as cleanup_error:
                    error.add_note(f"CloseHandle also failed: {cleanup_error}")
                raise
            self.writer_handle = candidate
        else:
            assert pid == self.pid and ppid == os.getpid(), "invalid READY identity direct child"
            assert identity.get("created_100ns") is None, "invalid READY identity creation time"
            assert self.poll() is None, "READY identity writer already exited"
        self.identity = identity
        return dict(identity)

    def force_kill(self):
        assert self.identity is not None, "cannot crash an unvalidated writer identity"
        assert self.forced_receipt is None, "writer was already forced to exit"
        assert not self._barrier_input_changed and not self.stdin.closed, (
            "changed or closed barrier input cannot establish a forced crash"
        )
        if os.name == "nt":
            before = self.writer_handle.snapshot()
            assert not before["signaled"], "writer already exited; this is not a forced crash"
            assert before["created_100ns"] == self.identity["created_100ns"]
            self.writer_handle.terminate(FORCED_EXIT_CODE)
            assert self.writer_handle.wait(30000), "actual writer did not terminate"
            writer = self.writer_handle.snapshot()
            assert writer["signaled"] and writer["exit_code"] == FORCED_EXIT_CODE
            # Reap the launcher only after actual writer death has been established.
            self.wait(timeout=30)
        else:
            assert self.poll() is None, "writer already exited; this is not a forced crash"
            super().kill()
            self.wait(timeout=30)
            assert self.returncode == -signal.SIGKILL
            writer = {**self.identity, "signaled": True, "exit_code": self.returncode}
        self.forced_receipt = {
            "forced": True,
            "boundary": self.boundary,
            "nonce": self.nonce,
            "writer": writer,
            "launcher": {"pid": self.pid, "exit_code": self.returncode},
        }
        return self.forced_receipt

    def cleanup(self, error=None):
        """Bounded teardown; never invent a crash receipt or suppress the test failure."""
        failures = []

        def attempt(action):
            try:
                action()
            except BaseException as cleanup_error:
                failures.append(cleanup_error)

        if self.writer_handle is not None:
            def stop_writer():
                if not self.writer_handle.wait(0):
                    self.writer_handle.terminate(FORCED_EXIT_CODE)
                    assert self.writer_handle.wait(30000), "cleanup writer did not terminate"
            attempt(stop_writer)
        # Before READY validation, EOF makes the owned child fail its protocol.
        # Never terminate an unvalidated PID supplied by a malformed message.
        if self.stdin is not None and not self.stdin.closed:
            attempt(self.stdin.close)
        if self.poll() is None:
            if self.identity is None:
                attempt(lambda: self.wait(timeout=5))
            if self.poll() is None:
                attempt(super().kill)
        attempt(lambda: self.wait(timeout=30))
        for reader in self._readers:
            attempt(lambda reader=reader: reader.join(timeout=5))
        if not any(reader.is_alive() for reader in self._readers):
            for stream in (self.stdout, self.stderr):
                if stream is not None and not stream.closed:
                    attempt(stream.close)
        else:
            failures.append(AssertionError("crash fixture reader did not exit"))
        if self.writer_handle is not None:
            attempt(self.writer_handle.close)
        if failures:
            if error is not None:
                for failure in failures:
                    error.add_note(f"crash fixture cleanup also failed: {failure}")
            else:
                raise failures[0]

    def __exit__(self, exc_type, exc_value, traceback):
        self.cleanup(exc_value)
