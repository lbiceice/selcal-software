"""One nonblocking writer per existing local checkpoint directory.

The fixed ``writer.lock`` is retained unchanged after normal exit or interruption;
its existence does not indicate a live writer. The OS releases the lock when the
process terminates. This cooperative local primitive is not checkpoint recovery,
hostile same-user isolation, or a network-volume/power-loss guarantee.
"""

from __future__ import annotations

import errno
import os
import stat
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from .workflow_store import _regular_nonlink


class CheckpointLockError(ValueError):
    """A refused lock; ``code`` identifies the stable failure category."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def _safe_identity(info: os.stat_result) -> tuple[int, int]:
    if not _regular_nonlink(info) or info.st_nlink != 1 or not info.st_ino:
        raise CheckpointLockError(
            "UNSAFE_LOCK_FILE", "writer.lock must be a single-link regular file with an inode"
        )
    return info.st_dev, info.st_ino


def _verify_identity(
    descriptor: int, marker: Path, expected: tuple[int, int] | None
) -> tuple[int, int]:
    opened = _safe_identity(os.fstat(descriptor))
    try:
        named = _safe_identity(marker.lstat())
    except FileNotFoundError as error:
        raise CheckpointLockError("LOCK_IDENTITY_CHANGED", "writer.lock disappeared") from error
    if opened != named or (expected is not None and opened != expected):
        raise CheckpointLockError("LOCK_IDENTITY_CHANGED", "writer.lock identity changed")
    return opened


def _operations() -> tuple[Callable[[int], None], Callable[[int], None]]:
    """Select only the actual host's native nonblocking primitive."""
    if sys.platform != "win32" and os.name == "posix":
        try:
            import fcntl
        except ImportError as error:
            raise CheckpointLockError("UNSUPPORTED_LOCK", "fcntl.flock is required") from error
        if not callable(getattr(fcntl, "flock", None)):
            raise CheckpointLockError("UNSUPPORTED_LOCK", "fcntl.flock is required")

        def acquire(descriptor: int) -> None:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                if error.errno in (errno.EACCES, errno.EAGAIN):
                    raise CheckpointLockError(
                        "CHECKPOINT_BUSY", "checkpoint already has a writer"
                    ) from error
                raise

        def release(descriptor: int) -> None:
            fcntl.flock(descriptor, fcntl.LOCK_UN)

        return acquire, release
    if os.name == "nt":
        try:
            import msvcrt
        except ImportError as error:
            raise CheckpointLockError("UNSUPPORTED_LOCK", "msvcrt.locking is required") from error
        locking = getattr(msvcrt, "locking", None)
        if not callable(locking):
            raise CheckpointLockError("UNSUPPORTED_LOCK", "msvcrt.locking is required")
        # Host-independent type checking cannot expose Windows-only stub members.
        nonblocking = getattr(msvcrt, "LK_NBLCK", None)
        unlocking = getattr(msvcrt, "LK_UNLCK", None)
        if type(nonblocking) is not int or type(unlocking) is not int:
            raise CheckpointLockError("UNSUPPORTED_LOCK", "native nonblocking lock modes required")

        def acquire_windows(descriptor: int) -> None:
            os.lseek(descriptor, 0, os.SEEK_SET)
            try:
                locking(descriptor, nonblocking, 1)
            except OSError as error:
                if error.errno == errno.EACCES:
                    raise CheckpointLockError(
                        "CHECKPOINT_BUSY", "checkpoint already has a writer"
                    ) from error
                raise

        def release_windows(descriptor: int) -> None:
            os.lseek(descriptor, 0, os.SEEK_SET)
            locking(descriptor, unlocking, 1)

        return acquire_windows, release_windows
    raise CheckpointLockError("UNSUPPORTED_LOCK", "no supported native writer lock on this OS")


@contextmanager
def checkpoint_writer_lock(directory: str | os.PathLike[str]) -> Iterator[Path]:
    """Yield the strict-resolved existing directory while holding its writer lock.

    Unsafe files and changed identities are refused before yielding. Only native
    acquisition contention becomes ``CHECKPOINT_BUSY``; other filesystem errors,
    including unlock errors, propagate. No lock-file bytes are written or removed.
    """
    resolved = Path(directory).resolve(strict=True)
    if not stat.S_ISDIR(resolved.stat().st_mode):
        raise NotADirectoryError(errno.ENOTDIR, "checkpoint must be a directory", str(resolved))
    acquire, release = _operations()
    marker = resolved / "writer.lock"
    try:
        before = marker.lstat()
    except FileNotFoundError:
        expected = None
    else:
        expected = _safe_identity(before)
    flags = os.O_RDWR | os.O_CREAT
    for optional in ("O_CLOEXEC", "O_BINARY", "O_NOFOLLOW"):
        flags |= getattr(os, optional, 0)
    descriptor = os.open(marker, flags, 0o600)
    try:
        os.set_inheritable(descriptor, False)
        expected = _verify_identity(descriptor, marker, expected)
        acquire(descriptor)
        try:
            _verify_identity(descriptor, marker, expected)
            yield resolved
        finally:
            release(descriptor)
    finally:
        os.close(descriptor)
