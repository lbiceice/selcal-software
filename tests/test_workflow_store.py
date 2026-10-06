"""Terminal-record byte consistency using real, independently reopened SQLite files."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import sqlite3
from contextlib import closing
from pathlib import Path
from types import MappingProxyType

import pytest
from _platform_support import mkfifo_or_skip, symlink_or_skip

MEMBERS = {"input": b"input\x00bytes", "request": b"{}", "result": b"result", "metadata": b"meta"}
TABLE_SQL = (
    "CREATE TABLE members(name TEXT PRIMARY KEY, payload BLOB NOT NULL, sha256 TEXT NOT NULL)"
)


@pytest.fixture(autouse=True)
def sqlite_connections_are_closed(monkeypatch):
    """Keep real connections alive so garbage collection cannot hide leaked handles."""
    original_connect = sqlite3.connect
    connections = []

    def track_connection(*args, **kwargs):
        connection = original_connect(*args, **kwargs)
        connections.append(connection)
        return connection

    monkeypatch.setattr(sqlite3, "connect", track_connection)
    yield
    leaked = 0
    for connection in connections:
        try:
            connection.execute("SELECT 1")
        except sqlite3.ProgrammingError as error:
            assert "closed" in str(error)
        else:
            leaked += 1
        finally:
            connection.close()
    assert leaked == 0, f"{leaked} SQLite connections remain open after the test"


def _store():
    assert importlib.util.find_spec("selcal.workflow_store") is not None, (
        "the production terminal-record store API must exist"
    )
    module = importlib.import_module("selcal.workflow_store")
    assert callable(getattr(module, "write_record", None))
    assert callable(getattr(module, "read_record", None))
    assert issubclass(module.RecordStoreError, ValueError)
    return module


def _database(path: Path, *, schema: str = TABLE_SQL, version: int = 1) -> None:
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(f"PRAGMA user_version={version}")
        connection.execute(schema)
        connection.executemany(
            "INSERT INTO members VALUES (?, ?, ?)",
            [
                (name, payload, hashlib.sha256(payload).hexdigest())
                for name, payload in MEMBERS.items()
            ],
        )


def _reject(path: Path, code: str = "INVALID_RECORD", *, max_bytes: int = 100_000) -> None:
    store = _store()
    with pytest.raises(store.RecordStoreError) as raised:
        store.read_record(path, max_bytes=max_bytes)
    assert raised.value.code == code


def test_write_reopens_as_a_real_sqlite_database_with_exact_members(tmp_path) -> None:
    store = _store()
    path = tmp_path / "record.sqlite"
    assert store.write_record(path, MappingProxyType(MEMBERS), max_bytes=100_000) is None
    with closing(sqlite3.connect(path)) as connection, connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (1,)
        rows = connection.execute("SELECT name, payload, sha256 FROM members").fetchall()
    assert {name: payload for name, payload, _ in rows} == MEMBERS
    assert all(digest == hashlib.sha256(payload).hexdigest() for _, payload, digest in rows)
    assert store.read_record(path, max_bytes=path.stat().st_size) == MEMBERS


def test_read_independently_created_database_preserves_source_bytes_and_directory(tmp_path) -> None:
    store = _store()
    path = tmp_path / "independent.sqlite"
    _database(path)
    before = path.read_bytes()
    before_stat = path.stat()
    path.chmod(0o444)
    assert store.read_record(path, max_bytes=len(before)) == MEMBERS
    assert path.read_bytes() == before
    assert path.stat().st_mtime_ns == before_stat.st_mtime_ns
    assert sorted(item.name for item in tmp_path.iterdir()) == [path.name]


def test_same_exact_physical_cap_can_write_and_read(tmp_path) -> None:
    store = _store()
    first = tmp_path / "first.sqlite"
    second = tmp_path / "second.sqlite"
    store.write_record(first, MEMBERS, max_bytes=100_000)
    limit = first.stat().st_size
    store.write_record(second, MEMBERS, max_bytes=limit)
    assert second.stat().st_size == limit
    assert store.read_record(second, max_bytes=limit) == MEMBERS
    _reject(second, "LIMIT_EXCEEDED", max_bytes=limit - 1)


def test_write_rejects_final_physical_size_above_cap(tmp_path) -> None:
    store = _store()
    path = tmp_path / "too-small.sqlite"
    with pytest.raises(store.RecordStoreError) as raised:
        store.write_record(path, MEMBERS, max_bytes=sum(map(len, MEMBERS.values())))
    assert raised.value.code == "LIMIT_EXCEEDED"
    if path.exists():
        _reject(path)


def test_write_rejects_aggregate_payload_above_cap_before_creating_output(tmp_path) -> None:
    store = _store()
    path = tmp_path / "payload-overflow.sqlite"
    with pytest.raises(store.RecordStoreError) as raised:
        store.write_record(path, MEMBERS, max_bytes=sum(map(len, MEMBERS.values())) - 1)
    assert raised.value.code == "LIMIT_EXCEEDED"
    assert not path.exists()


@pytest.mark.parametrize("limit", [True, False, 0, -1, 1.0, "10000", None])
@pytest.mark.parametrize("operation", ["write_record", "read_record"])
def test_invalid_limits_are_contract_errors_before_io(tmp_path, limit, operation) -> None:
    store = _store()
    args = [tmp_path / "missing.sqlite"]
    if operation == "write_record":
        args.append(MEMBERS)
    with pytest.raises(store.RecordStoreError) as raised:
        getattr(store, operation)(*args, max_bytes=limit)
    assert raised.value.code == "INVALID_LIMIT"
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "members",
    [
        {},
        {**MEMBERS, "extra": b"extra"},
        {name: value for name, value in MEMBERS.items() if name != "result"},
        {**MEMBERS, "input": b""},
        {**MEMBERS, "request": "{}"},
        {**MEMBERS, "result": bytearray(b"result")},
        {**MEMBERS, "metadata": memoryview(b"meta")},
        list(MEMBERS.items()),
    ],
)
def test_invalid_members_fail_before_creating_output(tmp_path, members) -> None:
    store = _store()
    path = tmp_path / "invalid-members.sqlite"
    with pytest.raises(store.RecordStoreError) as raised:
        store.write_record(path, members, max_bytes=100_000)
    assert raised.value.code == "INVALID_MEMBERS"
    assert not path.exists()


@pytest.mark.parametrize("target_kind", ["file", "symlink", "dangling_symlink"])
def test_exclusive_create_never_overwrites_an_existing_path(tmp_path, target_kind) -> None:
    store = _store()
    path = tmp_path / "occupied.sqlite"
    target = tmp_path / "original.sqlite"
    if target_kind == "file":
        path.write_bytes(b"keep original bytes")
    else:
        if target_kind == "symlink":
            target.write_bytes(b"keep original bytes")
        symlink_or_skip(path, target)
    with pytest.raises(FileExistsError):
        store.write_record(path, MEMBERS, max_bytes=100_000)
    if target_kind == "file":
        assert path.read_bytes() == b"keep original bytes"
    else:
        assert path.is_symlink()
        if target_kind == "symlink":
            assert target.read_bytes() == b"keep original bytes"
        else:
            assert not target.exists()


def test_missing_input_is_io_error_and_does_not_create_database(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        _store().read_record(tmp_path / "missing.sqlite", max_bytes=100_000)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("kind", ["directory", "symlink", "fifo"])
def test_read_rejects_nonregular_inputs_without_following_or_blocking(tmp_path, kind) -> None:
    path = tmp_path / "input.sqlite"
    if kind == "directory":
        path.mkdir()
    elif kind == "symlink":
        original = tmp_path / "original.sqlite"
        _database(original)
        symlink_or_skip(path, original)
    else:
        mkfifo_or_skip(path)
    _reject(path, "NONREGULAR_INPUT")


@pytest.mark.parametrize("content", [b"", b"not a database", b"SQLite format 3\x00" + bytes(84)])
def test_empty_partial_and_corrupt_files_are_rejected(tmp_path, content) -> None:
    path = tmp_path / "bad.sqlite"
    path.write_bytes(content)
    _reject(path)


@pytest.mark.parametrize("version", [0, 2, 999])
def test_unknown_versions_are_rejected(tmp_path, version) -> None:
    path = tmp_path / "version.sqlite"
    _database(path, version=version)
    _reject(path, "UNSUPPORTED_VERSION")


@pytest.mark.parametrize(
    "mutation",
    [
        "UPDATE members SET payload=x'616c7465726564' WHERE name='result'",
        "UPDATE members SET sha256='incorrect' WHERE name='result'",
        "UPDATE members SET payload='text payload' WHERE name='result'",
        "UPDATE members SET payload=x'' WHERE name='result'",
        "UPDATE members SET sha256=cast(sha256 AS BLOB) WHERE name='result'",
        "UPDATE members SET name=NULL WHERE name='result'",
        "DELETE FROM members WHERE name='metadata'",
        "INSERT INTO members VALUES ('extra', x'61', 'extra')",
        "CREATE TABLE extra(value TEXT)",
        "CREATE VIEW extra AS SELECT * FROM members",
        "CREATE INDEX extra ON members(payload)",
        "CREATE TRIGGER extra AFTER INSERT ON members BEGIN SELECT 1; END",
    ],
)
def test_mutated_content_and_extra_schema_are_rejected(tmp_path, mutation) -> None:
    path = tmp_path / "mutated.sqlite"
    _database(path)
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(mutation)
    before = path.read_bytes()
    _reject(path)
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "schema",
    [
        TABLE_SQL.replace("TEXT PRIMARY KEY", "TEXT"),
        TABLE_SQL.replace("payload BLOB", "payload TEXT"),
        TABLE_SQL.replace("payload BLOB NOT NULL", "payload BLOB"),
        TABLE_SQL.replace("sha256 TEXT NOT NULL", "sha256 TEXT NOT NULL DEFAULT ''"),
    ],
)
def test_altered_required_schema_is_rejected(tmp_path, schema) -> None:
    path = tmp_path / "schema.sqlite"
    _database(path, schema=schema)
    _reject(path)


@pytest.mark.parametrize("mutation", ["truncate", "append"])
def test_physical_truncation_and_extra_bytes_are_rejected(tmp_path, mutation) -> None:
    path = tmp_path / "physical.sqlite"
    _database(path)
    before = path.read_bytes()
    path.write_bytes(before[:-1] if mutation == "truncate" else before + b"extra bytes")
    _reject(path)


def test_huge_integer_cap_is_not_passed_to_a_read_system_call(tmp_path) -> None:
    path = tmp_path / "huge-cap.sqlite"
    _database(path)
    assert _store().read_record(path, max_bytes=10**100) == MEMBERS


def test_consistently_rehashed_bytes_are_accepted_without_claiming_authenticity(tmp_path) -> None:
    path = tmp_path / "rehash.sqlite"
    _database(path)
    replacement = b"changed result with matching new hash"
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(
            "UPDATE members SET payload=?, sha256=? WHERE name='result'",
            (replacement, hashlib.sha256(replacement).hexdigest()),
        )
    assert _store().read_record(path, max_bytes=100_000) == {**MEMBERS, "result": replacement}


def test_refuse_existing_path_refuses_a_dangling_link_before_any_open(tmp_path) -> None:
    # Windows exclusive creation follows a dangling link (hosted CI, 2026-10-06); the name check
    # must refuse it on every platform before os.open is reached.
    store = _store()
    path = tmp_path / "occupied.html"
    symlink_or_skip(path, tmp_path / "missing-target.html")
    with pytest.raises(FileExistsError):
        store.refuse_existing_path(path)
    assert not (tmp_path / "missing-target.html").exists()
    store.refuse_existing_path(tmp_path / "free.html")  # a free name passes
