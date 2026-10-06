"""Bounded terminal-record storage; member hashes establish byte consistency only.

The four nonempty byte members are committed together, never as replicate
checkpoints. ``max_bytes`` caps their aggregate payload size and the physical
database file size; it is not a CPU, allocation, or peak-memory guarantee.

Writes build a committed SQLite image in memory, then exclusively create the
output. Existing paths (including symlinks) are never replaced. An I/O failure
may leave a partial output; it is retained, and incomplete files fail reading.
Reads inspect a bounded snapshot in an isolated, query-only in-memory SQLite
connection, leaving the source database, schema, content, and sidecars alone.
Files are opened in binary mode. The record path must be a regular file, not a
symlink or a Windows reparse point that names another file (junctions and
similar links); other reparse points such as cloud-synchronised files are
ordinary files. POSIX opens with ``O_NOFOLLOW``; Windows opens with CreateFileW
and FILE_FLAG_OPEN_REPARSE_POINT, which never follows a link, and shares the
file for reading only while it is read. In every case the opened file must
have the same identity (device/volume, inode/file index, size, modification
time) as the path had before opening and still has after reading. A file
system that reports no file identity is refused. SQLite
serialization/deserialization is required; there is no unsafe fallback. This
format is not a resume protocol, signature, proof of execution, or
authentication mechanism.

``RecordStoreError.code`` categories:
* INVALID_LIMIT: max_bytes is not a positive built-in int.
* INVALID_MEMBERS: write input lacks the exact four nonempty built-in bytes.
* LIMIT_EXCEEDED: aggregate payload or physical database exceeds max_bytes.
* NONREGULAR_INPUT: the source is a symlink or other nonregular filesystem item.
* INPUT_CHANGED: source identity, size, or modification metadata changed while read.
* UNSUPPORTED_CAPABILITY: safe SQLite snapshot or file-identity support is absent.
* INPUT_BUSY: another program holds the record open and does not share it for reading.
* UNSUPPORTED_VERSION: a readable database has an unknown user_version.
* INVALID_RECORD: incomplete/corrupt database, noncanonical schema, bad member
  names/types/digests, or another SQLite format failure.

Filesystem errors, including missing input and an occupied output, remain
``OSError`` subclasses, except a Windows sharing or lock violation, which is
reported as INPUT_BUSY.

On Windows the identity is the volume serial, file index, size and modification
time; the creation time is not used. Two limits follow: modification time has
2-second resolution on FAT and exFAT, and Python 3.11 reports a 64-bit file
index, which is not guaranteed unique on ReFS (Python 3.12 and later report the
full 128-bit index). Within those limits a replaced file is still refused,
because size and modification time must match as well.

Schema SQL is exact for version 1, including the sole required primary-key
autoindex; alternate schemas need a future format version.
"""

from __future__ import annotations

import errno
import hashlib
import os
import sqlite3
import stat
import sys
from collections.abc import Mapping
from contextlib import closing

_MEMBER_NAMES = frozenset({"input", "request", "result", "metadata"})
_TABLE_SQL = (
    "CREATE TABLE members(name TEXT PRIMARY KEY, payload BLOB NOT NULL, sha256 TEXT NOT NULL)"
)
_CHUNK_SIZE = 64 * 1024


class RecordStoreError(ValueError):
    """A rejected terminal record; ``code`` identifies the contract category."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def _limit(max_bytes: int) -> None:
    if type(max_bytes) is not int or max_bytes <= 0:
        raise RecordStoreError("INVALID_LIMIT", "max_bytes must be a positive built-in int")


def _members(members: Mapping[str, bytes], max_bytes: int) -> dict[str, bytes]:
    if not isinstance(members, Mapping):
        raise RecordStoreError("INVALID_MEMBERS", "members must be a mapping")
    snapshot = dict(members)
    if snapshot.keys() != _MEMBER_NAMES or any(type(name) is not str for name in snapshot):
        raise RecordStoreError(
            "INVALID_MEMBERS", "exactly four named terminal members are required"
        )
    if any(type(payload) is not bytes or not payload for payload in snapshot.values()):
        raise RecordStoreError("INVALID_MEMBERS", "every payload must be nonempty built-in bytes")
    if sum(map(len, snapshot.values())) > max_bytes:
        raise RecordStoreError("LIMIT_EXCEEDED", "aggregate terminal payload exceeds max_bytes")
    return snapshot


def _capability(method: str) -> None:
    if not callable(getattr(sqlite3.Connection, method, None)):
        raise RecordStoreError("UNSUPPORTED_CAPABILITY", f"SQLite {method} is required")


def refuse_existing_path(path: str | os.PathLike[str]) -> None:
    """Raise FileExistsError if anything, including a dangling link, already occupies ``path``.

    Exclusive creation alone refuses a dangling symbolic link on POSIX but follows it on Windows
    (hosted Windows CI, 2026-10-06), so outputs check the name itself first.
    """
    if os.path.lexists(path):
        raise FileExistsError(errno.EEXIST, "Output already exists", os.fspath(path))


def write_record(
    path: str | os.PathLike[str], members: Mapping[str, bytes], *, max_bytes: int
) -> None:
    """Save the exact four terminal members to a new, exclusively created file."""
    _limit(max_bytes)
    payloads = _members(members, max_bytes)
    _capability("serialize")
    try:
        with closing(sqlite3.connect(":memory:", isolation_level=None)) as connection:
            connection.execute("BEGIN")
            connection.execute("PRAGMA user_version=1")
            connection.execute(_TABLE_SQL)
            connection.executemany(
                "INSERT INTO members(name, payload, sha256) VALUES (?, ?, ?)",
                [
                    (name, payloads[name], hashlib.sha256(payloads[name]).hexdigest())
                    for name in sorted(payloads)
                ],
            )
            connection.execute("COMMIT")
            image = connection.serialize()
    except sqlite3.Error as exc:
        raise RecordStoreError(
            "INVALID_RECORD", "SQLite could not construct terminal record"
        ) from exc
    if len(image) > max_bytes:
        raise RecordStoreError("LIMIT_EXCEEDED", "physical SQLite image exceeds max_bytes")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_BINARY", 0)
    )
    refuse_existing_path(path)
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "wb") as output:
        output.write(image)
        output.flush()
        os.fsync(output.fileno())


_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
# Reparse tags with this bit name another file (symlinks, junctions and similar links).
_NAME_SURROGATE = 0x20000000
_LINK_TAGS = frozenset(
    {
        getattr(stat, "IO_REPARSE_TAG_SYMLINK", 0xA000000C),
        getattr(stat, "IO_REPARSE_TAG_MOUNT_POINT", 0xA0000003),
        getattr(stat, "IO_REPARSE_TAG_APPEXECLINK", 0x8000001B),
        0x80000018,  # IO_REPARSE_TAG_WCI: redirects to another file inside a container
    }
)
# Windows reports a file held open by another program without read sharing as one of these.
_BUSY_WINERRORS = frozenset({32, 33})  # ERROR_SHARING_VIOLATION, ERROR_LOCK_VIOLATION


def _is_reparse_point(info: os.stat_result) -> bool:
    return bool(getattr(info, "st_file_attributes", 0) & _REPARSE_POINT)


def _is_link(info: os.stat_result) -> bool:
    """A symlink or a Windows reparse point that names another file.

    Other reparse points, such as cloud-synchronised files (OneDrive), are ordinary files. A
    reparse point whose tag cannot be read is treated as a link, so the check fails closed.
    """
    if stat.S_ISLNK(info.st_mode):
        return True
    if not _is_reparse_point(info):
        return False
    tag = getattr(info, "st_reparse_tag", 0)
    return not tag or bool(tag & _NAME_SURROGATE) or tag in _LINK_TAGS


def _regular_nonlink(info: os.stat_result) -> bool:
    return stat.S_ISREG(info.st_mode) and not _is_link(info)


def _identity(info: os.stat_result) -> tuple[int, ...]:
    if os.name == "nt":
        # st_ctime is the creation time on Windows and may change meaning; it is not used.
        return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def read_method() -> str:
    """How record files are opened on this platform (reported by ``doctor``)."""
    if os.name == "nt":
        return "windows_no_follow_open"
    return "no_follow_open" if hasattr(os, "O_NOFOLLOW") else "identity_verified_open"


def _open_windows(path: str | os.PathLike[str], *, follow: bool) -> int:
    """Open for binary reading with CreateFileW; other processes may read but not write.

    With ``follow=False`` the open uses FILE_FLAG_OPEN_REPARSE_POINT, so a link is opened as
    the link itself and never followed to its target (the equivalent of O_NOFOLLOW).
    ``follow=True`` is used only for non-link reparse points such as cloud files, which need
    their file-system filter to supply the data; identity checks still apply.
    """
    if sys.platform != "win32":
        raise RecordStoreError("UNSUPPORTED_CAPABILITY", "Windows file API is unavailable")
    import ctypes
    import msvcrt
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel32.CreateFileW
    create.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    create.restype = wintypes.HANDLE
    close = kernel32.CloseHandle
    close.argtypes = (wintypes.HANDLE,)
    close.restype = wintypes.BOOL
    name = os.path.abspath(os.fspath(path))
    if len(name) >= 260 and not name.startswith("\\\\?\\"):
        name = ("\\\\?\\UNC\\" + name[2:]) if name.startswith("\\\\") else "\\\\?\\" + name
    generic_read, share_read, open_existing = 0x80000000, 0x00000001, 3
    sequential_scan, open_reparse_point = 0x08000000, 0x00200000
    flags = sequential_scan | (0 if follow else open_reparse_point)
    handle = create(name, generic_read, share_read, None, open_existing, flags, None)
    if not handle or handle == wintypes.HANDLE(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        descriptor: int = msvcrt.open_osfhandle(handle, os.O_RDONLY | getattr(os, "O_BINARY", 0))
    except BaseException:
        close(handle)
        raise
    return descriptor


def _snapshot(path: str | os.PathLike[str], max_bytes: int) -> bytes:
    before = os.lstat(path)
    if not _regular_nonlink(before):
        raise RecordStoreError(
            "NONREGULAR_INPUT", "record source must be a regular file, not a link"
        )
    if before.st_size > max_bytes:
        raise RecordStoreError("LIMIT_EXCEEDED", "physical record size exceeds max_bytes")
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    if not no_follow and not (before.st_ino and before.st_dev):
        raise RecordStoreError(
            "UNSUPPORTED_CAPABILITY",
            "the file system must report a file identity (volume and file index)",
        )
    try:
        if os.name == "nt":
            descriptor = _open_windows(path, follow=_is_reparse_point(before))
        else:
            flags = (
                os.O_RDONLY
                | no_follow
                | getattr(os, "O_NONBLOCK", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_BINARY", 0)
            )
            descriptor = os.open(path, flags)
    except OSError as error:
        if getattr(error, "winerror", None) in _BUSY_WINERRORS:
            raise RecordStoreError(
                "INPUT_BUSY",
                "the record is open in another program and not shared for reading; "
                "close it (for example a synchronisation client, an editor or a virus "
                "scanner) and try again",
            ) from error
        raise
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise RecordStoreError("NONREGULAR_INPUT", "opened record source is not regular")
        if _identity(before) != _identity(opened):
            raise RecordStoreError("INPUT_CHANGED", "record source changed before snapshot")
        chunks = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(_CHUNK_SIZE, remaining))
            if not chunk:
                raise RecordStoreError("INPUT_CHANGED", "record source shrank during snapshot")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise RecordStoreError("INPUT_CHANGED", "record source grew during snapshot")
        if _identity(opened) != _identity(os.fstat(descriptor)):
            raise RecordStoreError("INPUT_CHANGED", "record source changed during snapshot")
        after = os.lstat(path)
        if not _regular_nonlink(after) or _identity(opened) != _identity(after):
            raise RecordStoreError("INPUT_CHANGED", "record path changed during snapshot")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _read_members(
    connection: sqlite3.Connection, physical_size: int, max_bytes: int
) -> dict[str, bytes]:
    if connection.execute("PRAGMA user_version").fetchone() != (1,):
        raise RecordStoreError("UNSUPPORTED_VERSION", "record user_version must be 1")
    schema = connection.execute(
        "SELECT type, name, tbl_name, sql FROM sqlite_schema ORDER BY type, name"
    ).fetchmany(3)
    expected = [
        ("index", "sqlite_autoindex_members_1", "members", None),
        ("table", "members", "members", _TABLE_SQL),
    ]
    if schema != expected:
        raise RecordStoreError(
            "INVALID_RECORD", "record schema differs from the exact version 1 schema"
        )
    page_count = connection.execute("PRAGMA page_count").fetchone()[0]
    page_size = connection.execute("PRAGMA page_size").fetchone()[0]
    if page_count * page_size != physical_size:
        raise RecordStoreError("INVALID_RECORD", "record file is truncated or has trailing bytes")
    if connection.execute("PRAGMA integrity_check").fetchmany(2) != [("ok",)]:
        raise RecordStoreError("INVALID_RECORD", "SQLite integrity check failed")
    rows = connection.execute("SELECT name, payload, sha256 FROM members").fetchmany(5)
    restored: dict[str, bytes] = {}
    payload_size = 0
    for name, payload, digest in rows:
        if type(name) is not str or name not in _MEMBER_NAMES or name in restored:
            raise RecordStoreError(
                "INVALID_RECORD", "record has an invalid or repeated member name"
            )
        if type(payload) is not bytes or not payload or type(digest) is not str:
            raise RecordStoreError("INVALID_RECORD", "member payload or digest has an invalid type")
        payload_size += len(payload)
        if payload_size > max_bytes:
            raise RecordStoreError("LIMIT_EXCEEDED", "aggregate terminal payload exceeds max_bytes")
        if hashlib.sha256(payload).hexdigest() != digest:
            raise RecordStoreError("INVALID_RECORD", "member digest does not match payload bytes")
        restored[name] = payload
    if restored.keys() != _MEMBER_NAMES:
        raise RecordStoreError(
            "INVALID_RECORD", "record must contain exactly four terminal members"
        )
    return restored


def read_record(path: str | os.PathLike[str], *, max_bytes: int) -> dict[str, bytes]:
    """Reopen and validate a bounded immutable snapshot, without modifying the source."""
    _limit(max_bytes)
    _capability("deserialize")
    image = _snapshot(path, max_bytes)
    if len(image) < 100 or not image.startswith(b"SQLite format 3\x00"):
        raise RecordStoreError("INVALID_RECORD", "record is not a complete SQLite image")
    try:
        with closing(sqlite3.connect(":memory:")) as connection:
            connection.deserialize(image)
            connection.execute("PRAGMA trusted_schema=OFF")
            connection.execute("PRAGMA query_only=ON")
            return _read_members(connection, len(image), max_bytes)
    except sqlite3.Error as exc:
        raise RecordStoreError("INVALID_RECORD", "SQLite rejected the record image") from exc
