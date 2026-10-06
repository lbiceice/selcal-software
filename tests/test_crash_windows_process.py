"""Adapter boundaries; only the explicitly marked native case tests Win32."""

import ctypes
import os
import subprocess
import sys

import pytest
from _windows_process import (
    _FILETIME,
    ProcessHandle,
    _decode_wait_result,
    _filetime_ticks,
)


@pytest.mark.parametrize("pid", [True, False, 0, -1, 2**32, 1.5, "123", None])
def test_invalid_pid_refused_before_native_access(pid):
    with pytest.raises(ValueError, match="pid"):
        ProcessHandle(pid)


@pytest.mark.parametrize("permission", [1, "yes", None, []])
def test_termination_requires_explicit_boolean_permission(permission):
    with pytest.raises(ValueError, match="allow_terminate"):
        ProcessHandle(123, allow_terminate=permission)


@pytest.mark.parametrize("timeout", [True, -1, 30001, 2**32 - 1, 1.5, "0", None])
def test_native_wait_is_bounded_before_handle_access(timeout):
    unopened = ProcessHandle.__new__(ProcessHandle)
    with pytest.raises(ValueError, match="timeout_ms"):
        unopened.wait(timeout)


def test_filetime_identity_uses_exact_integer_ticks():
    timestamp = _FILETIME(0x89ABCDEF, 0x12345678)
    assert ctypes.sizeof(timestamp) == 8
    assert _filetime_ticks(timestamp) == 0x1234567889ABCDEF


@pytest.mark.parametrize("native_result,signaled", [(0, True), (258, False)])
def test_wait_decoding_distinguishes_signal_and_timeout(native_result, signaled):
    assert _decode_wait_result(native_result) is signaled


def test_failed_wait_preserves_error_and_never_reports_death():
    with pytest.raises(OSError, match="WaitForSingleObject") as failure:
        _decode_wait_result(0xFFFFFFFF, 5)
    assert failure.value.errno == 5
    with pytest.raises(OSError, match="Unexpected"):
        _decode_wait_result(128)


@pytest.mark.skipif(os.name == "nt", reason="Unsupported-host refusal is a non-Windows check")
def test_native_access_explicitly_refuses_mac_and_other_hosts():
    with pytest.raises(OSError, match="native Windows"):
        ProcessHandle(os.getpid())


@pytest.mark.skipif(os.name != "nt", reason="Requires real Windows process handles")
def test_native_retained_identity_wait_and_readonly_refusal():
    interpreter = getattr(sys, "_base_executable", sys.executable)
    child = subprocess.Popen(
        [interpreter, "-c", "import sys; sys.stdin.read(1)"], stdin=subprocess.PIPE
    )
    try:
        with ProcessHandle(child.pid) as reader:
            initial = reader.snapshot()
            assert initial["signaled"] is False and initial["exit_code"] is None
            assert type(initial["created_100ns"]) is int and initial["created_100ns"] > 0
            with pytest.raises(PermissionError):
                reader.terminate()
            with ProcessHandle(child.pid, allow_terminate=True) as writer:
                assert writer.snapshot()["created_100ns"] == initial["created_100ns"]
                writer.terminate(259)
                assert writer.wait(5000)
                assert writer.snapshot()["exit_code"] == 259
                assert reader.wait(0)
        assert child.wait(timeout=5) == 259
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)
        child.stdin.close()
