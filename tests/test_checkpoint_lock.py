"""Real local single-writer contention; no database or recovery claim."""

from __future__ import annotations

import errno
import importlib
import importlib.util
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from _platform_support import directory_link_or_skip, mkfifo_or_skip, symlink_or_skip

ROOT = Path(__file__).resolve().parents[1]
CHILD = """
import sys
from selcal.checkpoint_lock import checkpoint_writer_lock, CheckpointLockError
if sys.argv[2] == 'barrier':
    print('READY', flush=True)
    sys.stdin.read(1)
try:
    with checkpoint_writer_lock(sys.argv[1]):
        print('LOCKED', flush=True)
        if sys.argv[2] != 'probe':
            sys.stdin.read(1)
except CheckpointLockError as error:
    print(error.code, flush=True)
    sys.exit(2)
"""


def _module():
    assert importlib.util.find_spec("selcal.checkpoint_lock") is not None, (
        "the production checkpoint single-writer lock must exist"
    )
    module = importlib.import_module("selcal.checkpoint_lock")
    assert callable(getattr(module, "checkpoint_writer_lock", None))
    assert issubclass(module.CheckpointLockError, ValueError)
    return module


def _environment():
    return {**os.environ, "PYTHONPATH": str(ROOT / "src")}


def _spawn(directory, mode="hold"):
    return subprocess.Popen(
        [sys.executable, "-B", "-c", CHILD, os.fspath(directory), mode],
        cwd=ROOT,
        env=_environment(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
    )


def _line(process):
    delivered = queue.Queue()
    threading.Thread(target=lambda: delivered.put(process.stdout.readline()), daemon=True).start()
    try:
        value = delivered.get(timeout=10)
    except queue.Empty:
        pytest.fail("child did not report within the bounded readiness interval")
    assert value, f"child exited before readiness: {process.poll()}"
    return value.strip()


def _stop(process):
    if process.poll() is None:
        process.kill()
    process.wait(timeout=10)
    for stream in (process.stdin, process.stdout, process.stderr):
        stream.close()


def _probe(directory):
    return subprocess.run(
        [sys.executable, "-B", "-c", CHILD, os.fspath(directory), "probe"],
        cwd=ROOT,
        env=_environment(),
        capture_output=True,
        text=True,
        errors="replace",
        timeout=10,
        check=False,
    )


def _identity(path):
    info = path.stat()
    return info.st_dev, info.st_ino


def test_normal_and_exception_exit_retain_empty_lock_and_release(tmp_path):
    module = _module()
    marker = tmp_path / "writer.lock"
    with pytest.raises(RuntimeError, match="interrupted"):
        with module.checkpoint_writer_lock(tmp_path) as directory:
            assert type(directory) is type(tmp_path)
            assert directory == tmp_path.resolve(strict=True)
            # Windows denies reading the locked byte through a second handle.
            assert marker.stat().st_size == 0
            identity = _identity(marker)
            raise RuntimeError("interrupted")
    assert (marker.read_bytes(), _identity(marker)) == (b"", identity)
    with module.checkpoint_writer_lock(tmp_path):
        assert _identity(marker) == identity
        assert marker.stat().st_size == 0
    assert marker.read_bytes() == b""
    assert _identity(marker) == identity
    assert {p.name for p in tmp_path.iterdir()} == {"writer.lock"}


def test_existing_nonempty_file_is_never_truncated_or_written(tmp_path):
    module = _module()
    marker = tmp_path / "writer.lock"
    marker.write_bytes(b"retained bytes\x00not a PID lease")
    before = marker.read_bytes(), _identity(marker)
    with module.checkpoint_writer_lock(str(tmp_path)):
        # Windows denies reading the locked byte through a second handle.
        assert (marker.stat().st_size, _identity(marker)) == (len(before[0]), before[1])
    assert (marker.read_bytes(), _identity(marker)) == before


@pytest.mark.parametrize("alias", ["same", "dot", "symlink", "case"])
def test_real_child_contention_alias_and_kill_reacquisition(tmp_path, alias):
    _module()
    directory = tmp_path / "Checkpoint"
    directory.mkdir()
    contender = directory
    if alias == "dot":
        contender = str(directory) + os.sep + "."
    elif alias == "symlink":
        contender = tmp_path / "alias"
        directory_link_or_skip(contender, directory)
    elif alias == "case":
        contender = tmp_path / "cHECKPOINT"
        if not contender.exists() or not contender.samefile(directory):
            pytest.skip("case alias does not identify the same directory on this filesystem")
    first = _spawn(directory)
    try:
        assert _line(first) == "LOCKED"
        marker = directory / "writer.lock"
        identity = _identity(marker)
        blocked = _probe(contender)
        assert blocked.returncode == 2, blocked.stdout + blocked.stderr
        assert blocked.stdout == "CHECKPOINT_BUSY\n"
        assert blocked.stderr == ""
        first.kill()
        assert first.wait(timeout=10) != 0
        acquired = _probe(contender)
        assert acquired.returncode == 0, acquired.stdout + acquired.stderr
        assert acquired.stdout == "LOCKED\n"
        assert acquired.stderr == ""
        assert marker.read_bytes() == b""
        assert _identity(marker) == identity
    finally:
        _stop(first)


def test_two_simultaneous_first_writers_have_only_one_winner(tmp_path):
    _module()
    assert not (tmp_path / "writer.lock").exists()
    children = []
    try:
        for _ in range(2):
            children.append(_spawn(tmp_path, "barrier"))
        assert [_line(child) for child in children] == ["READY", "READY"]
        for child in children:
            child.stdin.write("g")
            child.stdin.flush()
        states = [_line(child) for child in children]
        assert sorted(states) == ["CHECKPOINT_BUSY", "LOCKED"]
        loser = children[states.index("CHECKPOINT_BUSY")]
        winner = children[states.index("LOCKED")]
        assert loser.wait(timeout=10) == 2
        assert winner.poll() is None
        marker = tmp_path / "writer.lock"
        identity = _identity(marker)
        winner.kill()
        assert winner.wait(timeout=10) != 0
        assert _probe(tmp_path).returncode == 0
        assert _identity(marker) == identity and marker.read_bytes() == b""
    finally:
        for child in children:
            _stop(child)


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "directory", "fifo"])
def test_unsafe_lock_refused_without_changing_target(tmp_path, kind):
    module = _module()
    marker = tmp_path / "writer.lock"
    target = tmp_path / "target"
    target.write_bytes(b"do not alter")
    before = target.read_bytes(), _identity(target)
    if kind == "symlink":
        symlink_or_skip(marker, target)
    elif kind == "hardlink":
        os.link(target, marker)
    elif kind == "directory":
        marker.mkdir()
    else:
        mkfifo_or_skip(marker)
    with pytest.raises(module.CheckpointLockError) as caught:
        with module.checkpoint_writer_lock(tmp_path):
            pytest.fail("unsafe lock was yielded")
    assert caught.value.code == "UNSAFE_LOCK_FILE"
    assert (target.read_bytes(), _identity(target)) == before
    assert marker.exists() or marker.is_symlink()


@pytest.mark.parametrize("kind", ["missing", "file"])
def test_invalid_directory_preserves_filesystem_error(tmp_path, kind):
    module = _module()
    path = tmp_path / "not-directory"
    if kind == "file":
        path.write_bytes(b"file")
    expected = FileNotFoundError if kind == "missing" else NotADirectoryError
    with pytest.raises(expected):
        with module.checkpoint_writer_lock(path):
            pytest.fail("invalid directory was yielded")


def test_descriptor_is_noninheritable_and_closed_after_body_error(tmp_path, monkeypatch):
    module = _module()
    real_open = module.os.open
    opened = []

    def capture_open(path, flags, mode=0o777):
        descriptor = real_open(path, flags, mode)
        opened.append(descriptor)
        assert not flags & (os.O_TRUNC | os.O_EXCL)
        return descriptor

    monkeypatch.setattr(module.os, "open", capture_open)
    with pytest.raises(RuntimeError):
        with module.checkpoint_writer_lock(tmp_path):
            assert len(opened) == 1 and not os.get_inheritable(opened[0])
            raise RuntimeError("body")
    with pytest.raises(OSError) as caught:
        os.fstat(opened[0])
    assert caught.value.errno == errno.EBADF


@pytest.mark.parametrize("operation", ["acquire", "release"])
def test_unrelated_primitive_error_propagates_and_descriptor_closes(
    tmp_path,
    monkeypatch,
    operation,
):
    module = _module()
    if os.name != "posix":
        pytest.skip("POSIX fault injection; does not claim native Windows branch coverage")
    import fcntl

    actual_flock = fcntl.flock
    descriptors = []
    failure = OSError(errno.EIO, "injected I/O error")

    def fail_operation(descriptor, flags):
        descriptors.append(descriptor)
        if (flags == fcntl.LOCK_UN) == (operation == "release"):
            raise failure
        return actual_flock(descriptor, flags)

    monkeypatch.setattr(fcntl, "flock", fail_operation)
    with pytest.raises(OSError) as caught:
        with module.checkpoint_writer_lock(tmp_path):
            assert operation == "release"
    assert caught.value is failure
    with pytest.raises(OSError) as closed:
        os.fstat(descriptors[0])
    assert closed.value.errno == errno.EBADF


def test_missing_native_primitive_is_explicitly_unsupported(tmp_path, monkeypatch):
    module = _module()
    native = "fcntl" if os.name == "posix" else "msvcrt"
    monkeypatch.setitem(sys.modules, native, None)
    with pytest.raises(module.CheckpointLockError) as caught:
        with module.checkpoint_writer_lock(tmp_path):
            pytest.fail("missing native lock was accepted")
    assert caught.value.code == "UNSUPPORTED_LOCK"


def test_replacement_after_open_is_refused_and_descriptor_closed(tmp_path, monkeypatch):
    module = _module()
    if os.name != "posix":
        pytest.skip("POSIX opened-file rename fault injection; not native Windows coverage")
    marker = tmp_path / "writer.lock"
    marker.write_bytes(b"original")
    original = tmp_path / "original-lock"
    actual_open = module.os.open
    descriptors = []

    def replacing_open(path, flags, mode=0o777):
        descriptor = actual_open(path, flags, mode)
        descriptors.append(descriptor)
        marker.rename(original)
        marker.write_bytes(b"replacement")
        return descriptor

    monkeypatch.setattr(module.os, "open", replacing_open)
    with pytest.raises(module.CheckpointLockError) as caught:
        with module.checkpoint_writer_lock(tmp_path):
            pytest.fail("replaced lock was yielded")
    assert caught.value.code == "LOCK_IDENTITY_CHANGED"
    assert original.read_bytes() == b"original" and marker.read_bytes() == b"replacement"
    with pytest.raises(OSError) as closed:
        os.fstat(descriptors[0])
    assert closed.value.errno == errno.EBADF
