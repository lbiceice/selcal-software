"""Bounded Windows diagnostic handles; no PID-based operation after acquisition.

Use as a context manager. Calls are single-threaded: never close during a wait.
created_100ns and exited_100ns are exact FILETIME counts since 1601-01-01 UTC.
exited_100ns and exit_code are None unless the handle was signaled before the
time query; 259 can therefore be a real terminal code.
Snapshots are sequential observations, not an atomic operating-system snapshot.
The executable path is acquired once from the retained handle at construction.
Snapshots report that earlier identity observation; they do not query the image
again after termination. Signal, times and exit code are sampled on each call.

Microsoft API contracts:
https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-openprocess
https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-getprocesstimes
https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-queryfullprocessimagenamew
https://learn.microsoft.com/en-us/windows/win32/api/synchapi/nf-synchapi-waitforsingleobject
https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-getexitcodeprocess
https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-terminateprocess
"""

import ctypes
import os

_DWORD = ctypes.c_uint32
_BOOL = ctypes.c_int32
_HANDLE = ctypes.c_void_p
_QUERY_LIMITED_INFORMATION = 0x1000
_SYNCHRONIZE = 0x100000
_TERMINATE = 0x0001


class _FILETIME(ctypes.Structure):
    _fields_ = [("dwLowDateTime", _DWORD), ("dwHighDateTime", _DWORD)]


def _filetime_ticks(value):
    return (value.dwHighDateTime << 32) | value.dwLowDateTime


def _decode_wait_result(result, error_code=0):
    if result == 0:
        return True
    if result == 258:
        return False
    if result == 0xFFFFFFFF:
        raise OSError(error_code, "WaitForSingleObject failed")
    raise OSError(f"Unexpected process wait result: {result:#x}")


def _native_api():
    if os.name != "nt":
        raise OSError("ProcessHandle requires native Windows")
    api = ctypes.WinDLL("kernel32", use_last_error=True)
    time_ptr = ctypes.POINTER(_FILETIME)
    dword_ptr = ctypes.POINTER(_DWORD)
    for name, result, arguments in (
        ("OpenProcess", _HANDLE, [_DWORD, _BOOL, _DWORD]),
        ("GetProcessTimes", _BOOL, [_HANDLE, time_ptr, time_ptr, time_ptr, time_ptr]),
        ("QueryFullProcessImageNameW", _BOOL, [_HANDLE, _DWORD, ctypes.c_wchar_p, dword_ptr]),
        ("WaitForSingleObject", _DWORD, [_HANDLE, _DWORD]),
        ("GetExitCodeProcess", _BOOL, [_HANDLE, dword_ptr]),
        ("TerminateProcess", _BOOL, [_HANDLE, _DWORD]),
        ("CloseHandle", _BOOL, [_HANDLE]),
    ):
        function = getattr(api, name)
        function.restype, function.argtypes = result, arguments
    return api


def _check(success):
    if not success:
        raise ctypes.WinError(ctypes.get_last_error())


class ProcessHandle:
    """Own one noninheritable handle, with optional explicitly granted termination."""

    def __init__(self, pid, allow_terminate=False):
        if type(pid) is not int or not 0 < pid <= 0xFFFFFFFF:
            raise ValueError("pid must be a positive DWORD integer, not bool")
        if type(allow_terminate) is not bool:
            raise ValueError("allow_terminate must be an explicit bool")
        self._pid, self._allow_terminate = pid, allow_terminate
        self._api = _native_api()
        access = _QUERY_LIMITED_INFORMATION | _SYNCHRONIZE
        if allow_terminate:
            access |= _TERMINATE
        self._handle = self._api.OpenProcess(access, False, pid)
        _check(self._handle)
        try:
            image = ctypes.create_unicode_buffer(32768)
            size = _DWORD(len(image))
            _check(self._api.QueryFullProcessImageNameW(self._handle, 0, image, ctypes.byref(size)))
            self._image = image[: size.value]
        except BaseException as error:
            try:
                self.close()
            except OSError as cleanup_error:
                error.add_note(f"CloseHandle also failed: {cleanup_error}")
            raise

    def _require_handle(self):
        if not self._handle:
            raise ValueError("ProcessHandle is closed")
        return self._handle

    def snapshot(self):
        handle = self._require_handle()
        try:
            signaled = self.wait(0)
            times = [_FILETIME() for _ in range(4)]
            _check(self._api.GetProcessTimes(handle, *(ctypes.byref(t) for t in times)))
            exit_code = _DWORD()
            if signaled:
                _check(self._api.GetExitCodeProcess(handle, ctypes.byref(exit_code)))
            return {
                "pid": self._pid,
                "created_100ns": _filetime_ticks(times[0]),
                "exited_100ns": _filetime_ticks(times[1]) if signaled else None,
                "image": self._image,
                "image_observation": "handle_acquisition",
                "signaled": signaled,
                "exit_code": exit_code.value if signaled else None,
            }
        except BaseException as error:
            try:
                self.close()
            except OSError as cleanup_error:
                error.add_note(f"CloseHandle also failed: {cleanup_error}")
            raise

    def wait(self, timeout_ms):
        if type(timeout_ms) is not int or not 0 <= timeout_ms <= 30000:
            raise ValueError("timeout_ms must be an integer from 0 to 30000, not bool")
        result = self._api.WaitForSingleObject(self._require_handle(), timeout_ms)
        return _decode_wait_result(result, ctypes.get_last_error())

    def terminate(self, exit_code=1):
        if not self._allow_terminate:
            raise PermissionError("Termination requires allow_terminate=True at construction")
        _check(self._api.TerminateProcess(self._require_handle(), exit_code))

    def close(self):
        if self._handle:
            _check(self._api.CloseHandle(self._handle))
            self._handle = None

    def __enter__(self):
        self._require_handle()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.close()
        except OSError as cleanup_error:
            if exc_value is None:
                raise
            exc_value.add_note(f"CloseHandle also failed: {cleanup_error}")

