"""Bounded transactional outcome bytes under one cooperative local writer lock.

Stored values establish format and byte consistency, never scientific execution
or ownership. A later recovery service must resolve the retained request against
the real input and replay the entire prefix before continuing. The byte limit
bounds payload totals and the main database, not journal size or peak memory.
Failed creations are retained for inspection; no checkpoint file is deleted.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, cast
from uuid import UUID

from . import result_wire as wire
from .checkpoint_lock import checkpoint_writer_lock
from .contracts import RunStatus
from .contracts_v2 import RunFailureStage, V2IntegrityError
from .workflow_config import decode_workflow_config, encode_workflow_config
from .workflow_store import _regular_nonlink

_TABLES = {
    "members": (
        "CREATE TABLE members(name TEXT PRIMARY KEY, payload BLOB NOT NULL, sha256 TEXT NOT NULL)"
    ),
    "replicates": (
        "CREATE TABLE replicates(rep_id INTEGER PRIMARY KEY CHECK(rep_id >= 0), "
        "payload BLOB NOT NULL, sha256 TEXT NOT NULL)"
    ),
    "state": (
        "CREATE TABLE state(id INTEGER PRIMARY KEY CHECK(id = 1), phase TEXT NOT NULL, "
        "result BLOB, result_sha256 TEXT, metadata BLOB, metadata_sha256 TEXT)"
    ),
}
_HASH_KEYS = {
    "raw_input_sha256",
    "semantic_input_sha256",
    "scientific_plan_sha256",
    "request_sha256",
}
_HEADER_KEYS = _HASH_KEYS | {
    "schema",
    "execution_id",
    "planned_replicates",
    "input_format",
    "max_bytes",
    "allow_unattainable",
    "software",
}
_METADATA_KEYS = {
    "schema",
    "raw_input_sha256",
    "semantic_input_sha256",
    "scientific_plan_sha256",
    "software",
}


class CheckpointStoreError(ValueError):
    """Rejected checkpoint; ``code`` identifies the stable contract category.

    Codes: INVALID_LIMIT, INVALID_CHECKPOINT, UNSAFE_CHECKPOINT_FILE,
    LIMIT_EXCEEDED, IDENTITY_MISMATCH, INVALID_SEQUENCE, FINALIZED, CLOSED,
    UNSUPPORTED_CAPABILITY, CHECKPOINT_BUSY. Ordinary filesystem errors and
    CheckpointLockError retain their original types.
    """

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def _require(condition: bool, message: str, code: str = "INVALID_CHECKPOINT") -> None:
    if not condition:
        raise CheckpointStoreError(code, message)


def _limit(value: int) -> None:
    _require(
        type(value) is int and value > 0,
        "max_bytes must be a positive built-in int",
        "INVALID_LIMIT",
    )


def _size(size: int, limit: int) -> None:
    _require(size <= limit, "checkpoint exceeds max_bytes", "LIMIT_EXCEEDED")


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _bytes(value: object) -> bytes:
    _require(type(value) is bytes and bool(value), "payload must be nonempty built-in bytes")
    return cast(bytes, value)


def _digest(value: object) -> bool:
    return type(value) is str and re.fullmatch("[0-9a-f]{64}", value) is not None


def _dump(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _json(payload: bytes) -> object:
    _bytes(payload)
    try:
        wire._check_depth(payload)
        return json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=wire._unique_object,
            parse_float=wire._reject_number,
            parse_constant=wire._reject_number,
        )
    except (ValueError, RecursionError) as error:
        raise CheckpointStoreError("INVALID_CHECKPOINT", "invalid bounded UTF-8 JSON") from error


def _object(payload: bytes, keys: set[str]) -> dict[str, object]:
    value = _json(payload)
    _require(type(value) is dict and value.keys() == keys, "incorrect document fields")
    _require(_dump(value) == payload, "document is not canonical JSON plus LF")
    return cast(dict[str, object], value)


def _software(value: object) -> None:
    keys = {"selcal_version", "python_version", "numpy_version", "source_files", "platform"}
    _require(type(value) is dict and value.keys() == keys, "incorrect software identity fields")
    record = cast(dict[str, object], value)
    for key in ("selcal_version", "python_version", "numpy_version"):
        item = record[key]
        _require(type(item) is str and 0 < len(item) <= 256, "invalid software version")
    platform = record["platform"]
    _require(
        type(platform) is dict and platform.keys() == {"system", "machine", "libc", "blas"},
        "incorrect platform identity fields",
    )
    _require(
        all(
            type(item) is str and len(item) <= 256
            for item in cast(dict[str, object], platform).values()
        ),
        "invalid platform identity",
    )
    sources = record["source_files"]
    _require(type(sources) is dict and bool(sources), "source identity must be nonempty")
    for name, digest in cast(dict[str, object], sources).items():
        _require(
            type(name) is str and bool(name) and "\\" not in name and "\x00" not in name,
            "invalid source path",
        )
        path = PurePosixPath(name)
        _require(
            not path.is_absolute()
            and path.suffix == ".py"
            and path.as_posix() == name
            and ".." not in path.parts
            and ":" not in name
            and _digest(digest),
            "source identity requires relative Python paths and SHA-256",
        )


def _initial(header: bytes, raw: bytes, request: bytes, limit: int) -> dict[str, object]:
    for payload in (header, raw, request):
        _bytes(payload)
    _size(len(header) + len(raw) + len(request), limit)
    _require(len(header) <= 65536, "header exceeds 65536 bytes")
    record = _object(header, _HEADER_KEYS)
    _require(record["schema"] == "selcal.checkpoint.v1", "unsupported checkpoint schema")
    identifier = record["execution_id"]
    try:
        valid_uuid = type(identifier) is str and str(UUID(identifier)) == identifier
    except ValueError:
        valid_uuid = False
    _require(valid_uuid, "execution_id must be a canonical UUID")
    _require(all(_digest(record[key]) for key in _HASH_KEYS), "invalid SHA-256 claim")
    _require(
        type(record["planned_replicates"]) is int and record["planned_replicates"] > 0,
        "planned_replicates must be a positive built-in int",
    )
    _require(type(record["max_bytes"]) is int and record["max_bytes"] > 0, "invalid frozen limit")
    _require(
        record["max_bytes"] == limit, "max_bytes differs from frozen header", "IDENTITY_MISMATCH"
    )
    _require(type(record["allow_unattainable"]) is bool, "override must be a boolean")
    _require(record["input_format"] in ("csv", "npz"), "unsupported input format")
    _software(record["software"])
    _require(
        record["raw_input_sha256"] == _sha(raw) and record["request_sha256"] == _sha(request),
        "retained input/request digest differs from header",
    )
    try:
        config = decode_workflow_config(request)
        _require(encode_workflow_config(config) == request, "request must be canonical")
        _require(
            config.request.replicates == record["planned_replicates"]
            and config.source_format == record["input_format"],
            "request differs from header",
        )
    except ValueError as error:
        if isinstance(error, CheckpointStoreError):
            raise
        raise CheckpointStoreError("INVALID_CHECKPOINT", "invalid workflow request") from error
    return record


def _outcome(payload: bytes, index: int, header: dict[str, object]) -> None:
    try:
        restored = wire._read_outcome(_json(payload))
        _require(wire._encode_replicate_outcome(restored) == payload, "outcome is not canonical")
        _require(restored.replicate_id == index, "outcome ID differs from row")
        _require(
            restored.transform_token.semantic_input_sha256 == header["semantic_input_sha256"]
            and restored.transform_token.scientific_plan_sha256 == header["scientific_plan_sha256"],
            "outcome identity differs from header",
            "IDENTITY_MISMATCH",
        )
    except (ValueError, V2IntegrityError, OverflowError) as error:
        if isinstance(error, CheckpointStoreError):
            raise
        raise CheckpointStoreError("INVALID_CHECKPOINT", "invalid outcome value") from error


@contextmanager
def _sql_errors() -> Iterator[None]:
    try:
        yield
    except sqlite3.Error as error:
        primary = getattr(error, "sqlite_errorcode", 0) & 255
        code = (
            "LIMIT_EXCEEDED"
            if primary == sqlite3.SQLITE_FULL
            else "CHECKPOINT_BUSY"
            if primary in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED)
            else "INVALID_CHECKPOINT"
        )
        raise CheckpointStoreError(code, "SQLite rejected checkpoint operation") from error


@contextmanager
def _transaction(connection: sqlite3.Connection) -> Iterator[None]:
    with _sql_errors():
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield
            connection.execute("COMMIT")
        except BaseException:
            if connection.in_transaction:
                connection.rollback()
            raise


def _file(path: Path) -> os.stat_result:
    info = path.lstat()
    _require(
        _regular_nonlink(info) and info.st_nlink == 1 and bool(info.st_ino),
        "checkpoint file must be regular, single-link, and have an inode",
        "UNSAFE_CHECKPOINT_FILE",
    )
    return info


def _files(path: Path, limit: int) -> os.stat_result:
    info = _file(path)
    _size(info.st_size, limit)
    for suffix in ("-journal", "-wal", "-shm"):
        try:
            _file(Path(str(path) + suffix))
        except FileNotFoundError:
            continue
        _require(suffix == "-journal", "WAL/SHM sidecars are unsupported", "UNSAFE_CHECKPOINT_FILE")
    return info


def _configure(connection: sqlite3.Connection, limit: int, *, new: bool) -> None:
    mode = "PRAGMA journal_mode=DELETE" if new else "PRAGMA journal_mode"
    _require(
        connection.execute(mode).fetchone() == ("delete",),
        "checkpoint requires DELETE journal mode",
    )
    connection.execute("PRAGMA synchronous=FULL")
    _require(
        connection.execute("PRAGMA synchronous").fetchone() == (2,),
        "SQLite FULL synchronization unavailable",
        "UNSUPPORTED_CAPABILITY",
    )
    page_size = connection.execute("PRAGMA page_size").fetchone()[0]
    _require(type(page_size) is int and page_size > 0, "invalid SQLite page size")
    pages = limit // page_size
    _require(pages > 0, "max_bytes cannot hold one database page", "LIMIT_EXCEEDED")
    actual = connection.execute(f"PRAGMA max_page_count={min(pages, 4294967294)}").fetchone()[0]
    _require(
        type(actual) is int and 0 < actual <= pages,
        "database page count exceeds limit",
        "LIMIT_EXCEEDED",
    )
    connection.execute("PRAGMA trusted_schema=OFF")


def _blob(payload: object, digest: object) -> bytes:
    data = _bytes(payload)
    _require(type(digest) is str and _sha(data) == digest, "payload digest mismatch")
    return data


@dataclass(frozen=True, slots=True)
class CheckpointSnapshot:
    """Immutable bytes, with no execution-authority claim."""

    header_bytes: bytes
    input_bytes: bytes
    request_bytes: bytes
    replicates: tuple[tuple[int, bytes], ...]
    phase: Literal["OPEN", "FINALIZED"]
    result_bytes: bytes | None
    metadata_bytes: bytes | None


class CheckpointSession:
    """The same exclusively held session is used from inspection through append."""

    __slots__ = (
        "__closed",
        "__connection",
        "__count",
        "__header",
        "__limit",
        "__members",
        "__phase",
        "__size",
    )

    def __init__(
        self,
        connection: sqlite3.Connection,
        header: dict[str, object],
        members: dict[str, bytes],
        limit: int,
        count: int,
        size: int,
        phase: str,
    ) -> None:
        self.__connection, self.__header, self.__members = connection, header, members
        self.__limit, self.__count, self.__size = limit, count, size
        self.__phase, self.__closed = phase, False

    def _close(self) -> None:
        self.__closed = True
        self.__connection.close()

    def __active(self, *, writing: bool = False) -> None:
        _require(not self.__closed, "checkpoint session is closed", "CLOSED")
        if writing:
            _require(self.__phase == "OPEN", "checkpoint is finalized", "FINALIZED")

    def snapshot(self) -> CheckpointSnapshot:
        """Materialize the full prefix only on explicit request."""
        self.__active()
        with _sql_errors():
            rows = self.__connection.execute(
                "SELECT rep_id, payload FROM replicates ORDER BY rep_id"
            ).fetchall()
            phase, result, metadata = self.__connection.execute(
                "SELECT phase, result, metadata FROM state WHERE id=1"
            ).fetchone()
        return CheckpointSnapshot(
            self.__members["header"],
            self.__members["input"],
            self.__members["request"],
            tuple(rows),
            phase,
            result,
            metadata,
        )

    def append(self, replicate_id: int, payload: bytes) -> None:
        """Commit exactly the next complete outcome, without rescanning old rows."""
        self.__active(writing=True)
        _require(
            type(replicate_id) is int
            and replicate_id == self.__count
            and replicate_id < cast(int, self.__header["planned_replicates"]),
            "append requires the next planned replicate ID",
            "INVALID_SEQUENCE",
        )
        _bytes(payload)
        new_size = self.__size + len(payload)
        _size(new_size, self.__limit)
        _outcome(payload, replicate_id, self.__header)
        with _transaction(self.__connection):
            self.__connection.execute(
                "INSERT INTO replicates VALUES (?, ?, ?)", (replicate_id, payload, _sha(payload))
            )
        self.__count += 1
        self.__size = new_size

    def __terminal(self, result_bytes: bytes, metadata_bytes: bytes) -> None:
        _bytes(result_bytes)
        _bytes(metadata_bytes)
        try:
            result = wire.decode_calibration_result(result_bytes, max_bytes=self.__limit)
        except ValueError as error:
            raise CheckpointStoreError("INVALID_CHECKPOINT", "invalid terminal result") from error
        b = self.__header["planned_replicates"]
        _require(
            self.__count == b
            or (
                self.__count == 0
                and result.status is RunStatus.NOT_EVALUABLE
                and result.failure_stage
                in (RunFailureStage.NULL_BIND, RunFailureStage.OBSERVED_STATISTIC_SCAN)
                and not result.replicates
                and result.p_value is None
                and result.reject_null is None
            ),
            "terminal requires all B rows or zero-row pre-replicate NE",
            "INVALID_SEQUENCE",
        )
        _require(
            result.planned_replicates == b
            and result.semantic_input_sha256 == self.__header["semantic_input_sha256"]
            and result.scientific_plan_sha256 == self.__header["scientific_plan_sha256"],
            "terminal result identity differs from checkpoint",
            "IDENTITY_MISMATCH",
        )
        _require(
            len(result.replicates) == self.__count,
            "terminal prefix length differs",
            "INVALID_SEQUENCE",
        )
        with closing(
            self.__connection.execute("SELECT rep_id, payload FROM replicates ORDER BY rep_id")
        ) as rows:
            for outcome, row in zip(result.replicates, rows, strict=True):
                _require(
                    (outcome.replicate_id, wire._encode_replicate_outcome(outcome)) == row,
                    "terminal outcome bytes differ from stored prefix",
                    "IDENTITY_MISMATCH",
                )
        metadata = _object(metadata_bytes, _METADATA_KEYS)
        expected = {key: self.__header[key] for key in _METADATA_KEYS - {"schema"}}
        expected["schema"] = "selcal.workflow-record.v2"
        _require(metadata == expected, "terminal metadata identity differs", "IDENTITY_MISMATCH")

    def finalize(self, result_bytes: bytes, metadata_bytes: bytes) -> None:
        """Atomically commit both terminal members and FINALIZED after matching."""
        self.__active(writing=True)
        _bytes(result_bytes)
        _bytes(metadata_bytes)
        new_size = self.__size + len(result_bytes) + len(metadata_bytes)
        _size(new_size, self.__limit)
        with _sql_errors():
            self.__terminal(result_bytes, metadata_bytes)
        with _transaction(self.__connection):
            self.__connection.execute(
                "UPDATE state SET phase='FINALIZED', result=?, result_sha256=?, "
                "metadata=?, metadata_sha256=? WHERE id=1",
                (result_bytes, _sha(result_bytes), metadata_bytes, _sha(metadata_bytes)),
            )
        self.__phase, self.__size = "FINALIZED", new_size

    def _validate_terminal(self, result: bytes, metadata: bytes) -> None:
        self.__terminal(result, metadata)


def _read(
    connection: sqlite3.Connection,
    path: Path,
    before: os.stat_result,
    expected_software: dict[str, object],
    limit: int,
) -> CheckpointSession:
    # This must be the first SQLite read: it performs original-file hot rollback.
    schema = connection.execute(
        "SELECT type, name, tbl_name, sql FROM sqlite_schema ORDER BY type, name"
    ).fetchmany(5)
    after = _files(path, limit)
    _require(
        (before.st_dev, before.st_ino) == (after.st_dev, after.st_ino),
        "database identity changed across recovery",
        "UNSAFE_CHECKPOINT_FILE",
    )
    expected = [("index", "sqlite_autoindex_members_1", "members", None)] + [
        ("table", name, name, sql) for name, sql in sorted(_TABLES.items())
    ]
    _require(schema == expected, "checkpoint schema differs from exact v1 schema")
    _require(
        connection.execute("PRAGMA application_id").fetchone() == (0x53434350,)
        and connection.execute("PRAGMA user_version").fetchone() == (1,),
        "incorrect database identity/version",
    )
    _configure(connection, limit, new=False)
    pages = connection.execute("PRAGMA page_count").fetchone()[0]
    page_size = connection.execute("PRAGMA page_size").fetchone()[0]
    _require(pages * page_size == after.st_size, "database truncated or has trailing bytes")
    _require(
        connection.execute("PRAGMA quick_check").fetchmany(2) == [("ok",)],
        "SQLite quick_check failed",
    )
    members = {}
    for name, payload, digest in connection.execute(
        "SELECT name, payload, sha256 FROM members"
    ).fetchmany(4):
        _require(
            type(name) is str and name in {"header", "input", "request"} and name not in members,
            "incorrect checkpoint members",
        )
        members[name] = _blob(payload, digest)
    _require(members.keys() == {"header", "input", "request"}, "missing checkpoint members")
    header = _initial(members["header"], members["input"], members["request"], limit)
    _software(expected_software)
    _require(
        header["software"] == expected_software, "software identity differs", "IDENTITY_MISMATCH"
    )
    count, size = 0, sum(map(len, members.values()))
    with closing(
        connection.execute("SELECT rep_id, payload, sha256 FROM replicates ORDER BY rep_id")
    ) as rows:
        for index, payload, digest in rows:
            _require(
                type(index) is int
                and index == count
                and count < cast(int, header["planned_replicates"]),
                "replicate prefix is not contiguous within B",
            )
            data = _blob(payload, digest)
            size += len(data)
            _size(size, limit)
            _outcome(data, index, header)
            count += 1
    states = connection.execute(
        "SELECT id, phase, result, result_sha256, metadata, metadata_sha256 FROM state"
    ).fetchmany(2)
    _require(len(states) == 1, "checkpoint requires exactly one state row")
    identifier, phase, result, result_sha, metadata, metadata_sha = states[0]
    _require(
        type(identifier) is int
        and identifier == 1
        and type(phase) is str
        and phase in ("OPEN", "FINALIZED"),
        "invalid state row",
    )
    session = CheckpointSession(connection, header, members, limit, count, size, phase)
    if phase == "OPEN":
        _require(
            all(value is None for value in (result, result_sha, metadata, metadata_sha)),
            "OPEN checkpoint has terminal content",
        )
    else:
        result, metadata = _blob(result, result_sha), _blob(metadata, metadata_sha)
        _size(size + len(result) + len(metadata), limit)
        session._validate_terminal(result, metadata)
    return session


@contextmanager
def create_checkpoint(
    directory: str | os.PathLike[str],
    *,
    header_bytes: bytes,
    input_bytes: bytes,
    request_bytes: bytes,
    max_bytes: int,
) -> Iterator[CheckpointSession]:
    """Exclusively create a new checkpoint; partial failures are never overwritten."""
    _limit(max_bytes)
    header = _initial(header_bytes, input_bytes, request_bytes, max_bytes)
    members = {"header": header_bytes, "input": input_bytes, "request": request_bytes}
    Path(directory).mkdir()
    with checkpoint_writer_lock(directory) as resolved:
        path = resolved / "state.sqlite"
        descriptor = os.open(
            path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o600
        )
        os.close(descriptor)
        before = _files(path, max_bytes)
        connection = None
        session = None
        try:
            with _sql_errors():
                connection = sqlite3.connect(
                    path.as_uri() + "?mode=rw", uri=True, timeout=0, isolation_level=None
                )
                _configure(connection, max_bytes, new=True)
                with _transaction(connection):
                    connection.execute("PRAGMA application_id=1396917072")
                    connection.execute("PRAGMA user_version=1")
                    for sql in _TABLES.values():
                        connection.execute(sql)
                    connection.executemany(
                        "INSERT INTO members VALUES (?, ?, ?)",
                        [(name, payload, _sha(payload)) for name, payload in members.items()],
                    )
                    connection.execute(
                        "INSERT INTO state VALUES (1, 'OPEN', NULL, NULL, NULL, NULL)"
                    )
                after = _files(path, max_bytes)
                _require(
                    (before.st_dev, before.st_ino) == (after.st_dev, after.st_ino),
                    "database identity changed",
                    "UNSAFE_CHECKPOINT_FILE",
                )
                session = CheckpointSession(
                    connection,
                    header,
                    members,
                    max_bytes,
                    0,
                    sum(map(len, members.values())),
                    "OPEN",
                )
            yield session
        finally:
            if session is not None:
                session._close()
            elif connection is not None:
                connection.close()


@contextmanager
def open_checkpoint(
    directory: str | os.PathLike[str], *, expected_software: dict[str, object], max_bytes: int
) -> Iterator[CheckpointSession]:
    """Recover and validate the original database while retaining its writer lock."""
    _limit(max_bytes)
    with checkpoint_writer_lock(directory) as resolved:
        path = resolved / "state.sqlite"
        before = _files(path, max_bytes)
        connection = None
        session = None
        try:
            with _sql_errors():
                connection = sqlite3.connect(
                    path.as_uri() + "?mode=rw", uri=True, timeout=0, isolation_level=None
                )
                session = _read(connection, path, before, expected_software, max_bytes)
            yield session
        finally:
            if session is not None:
                session._close()
            elif connection is not None:
                connection.close()
