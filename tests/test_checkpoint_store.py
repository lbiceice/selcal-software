"""Transactional byte-prefix tests using actual existing calibration outputs."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import sqlite3
import uuid
from contextlib import closing
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest
from _platform_support import mkfifo_or_skip, symlink_or_skip

from selcal.calibration_v2 import _calibrate_selected_family_observed
from selcal.contracts import SeriesPair
from selcal.contracts_v2 import PlanRequestV2
from selcal.resolution_v2 import resolve_plan_v2
from selcal.result_wire import encode_calibration_result
from selcal.workflow_config import WorkflowConfig, encode_workflow_config

LIMIT = 2_000_000


def _dump(value):
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
    )


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _api():
    assert importlib.util.find_spec("selcal.checkpoint_store") is not None, (
        "transactional checkpoint store must preserve a complete committed prefix"
    )
    return importlib.import_module("selcal.checkpoint_store")


def _case(*, limit=LIMIT, ne=None, padding=0, replicates=3):
    source = [0, 1, 4, 2, 5, 3]
    target = [4, 1, 3, 0, 5, 2]
    candidates, params = (1, 2), {"min_shift": 1}
    if ne == "observed":
        source = [0] * 6
    if ne == "null":
        source, target, candidates, params = (
            [0, 1, 2, 3, 4],
            [4, 3, 2, 1, 0],
            (1, 3),
            {"min_shift": 3},
        )
    pair = SeriesPair(np.asarray(source, dtype=np.float64), np.asarray(target, dtype=np.float64))
    plan = PlanRequestV2(
        candidates=candidates,
        statistic_name="lagged_pearson_v1",
        statistic_params={},
        selection_rule="max_upper",
        null_name="circular_shift_v2",
        null_params=params,
        replicates=replicates,
        alpha=0.05,
        tie_tolerance=1e-12,
        root_seed=17,
    )
    request = encode_workflow_config(WorkflowConfig(plan, "csv", "source", "target"))
    raw = (
        "source,target\n" + "".join(f"{s},{t}\n" for s, t in zip(source, target, strict=True))
    ).encode()
    raw += b"\n" * padding
    outcomes = []
    result = _calibrate_selected_family_observed(
        pair, resolve_plan_v2(plan), lambda i, payload: outcomes.append((i, payload))
    )
    software = {
        "selcal_version": "test-identity",
        "python_version": "3.11",
        "numpy_version": np.__version__,
        "source_files": {
            "calibration_v2.py": _sha(
                Path(importlib.import_module("selcal.calibration_v2").__file__).read_bytes()
            )
        },
        "platform": {"system": "test", "machine": "test", "libc": "", "blas": "test"},
    }
    header = {
        "schema": "selcal.checkpoint.v1",
        "execution_id": str(uuid.uuid4()),
        "raw_input_sha256": _sha(raw),
        "request_sha256": _sha(request),
        "semantic_input_sha256": result.semantic_input_sha256,
        "scientific_plan_sha256": result.scientific_plan_sha256,
        "planned_replicates": replicates,
        "input_format": "csv",
        "max_bytes": limit,
        "allow_unattainable": True,
        "software": software,
    }
    metadata = {
        key: header[key]
        for key in (
            "raw_input_sha256",
            "semantic_input_sha256",
            "scientific_plan_sha256",
            "software",
        )
    }
    metadata["schema"] = "selcal.workflow-record.v2"
    return {
        "header": _dump(header),
        "raw": raw,
        "request": request,
        "outcomes": outcomes,
        "result": encode_calibration_result(result, max_bytes=LIMIT),
        "metadata": _dump(metadata),
        "software": software,
        "limit": limit,
    }


def _create(path, case):
    return _api().create_checkpoint(
        path,
        header_bytes=case["header"],
        input_bytes=case["raw"],
        request_bytes=case["request"],
        max_bytes=case["limit"],
    )


def _open(path, case, **changes):
    return _api().open_checkpoint(
        path, **{"expected_software": case["software"], "max_bytes": case["limit"], **changes}
    )


def _error(code):
    return pytest.raises(_api().CheckpointStoreError) if code is None else _Code(code)


class _Code:
    def __init__(self, code):
        self.code = code

    def __enter__(self):
        self.context = pytest.raises(_api().CheckpointStoreError)
        return self.context.__enter__()

    def __exit__(self, *args):
        result = self.context.__exit__(*args)
        if result:
            assert self.context.excinfo.value.code == self.code
        return result


def _mutate(path, sql, values=()):
    with closing(sqlite3.connect(path / "state.sqlite")) as conn, conn:
        conn.execute(sql, values)


def test_error_helper_preserves_real_lock_failure(tmp_path):
    from selcal.checkpoint_lock import CheckpointLockError

    case = _case()
    path = tmp_path / "held-checkpoint"
    with _create(path, case), pytest.raises(CheckpointLockError) as caught:
        with _error("IDENTITY_MISMATCH"), _open(path, case):
            pass
    assert caught.value.code == "CHECKPOINT_BUSY"


@pytest.mark.parametrize("exception", [ValueError("unexpected"), KeyboardInterrupt()])
def test_error_helper_preserves_unexpected_exception(exception):
    with pytest.raises(type(exception)) as caught:
        with _error("IDENTITY_MISMATCH"):
            raise exception
    assert caught.value is exception


def test_error_helper_accepts_matching_code():
    exception = _api().CheckpointStoreError("IDENTITY_MISMATCH", "expected")
    with _error("IDENTITY_MISMATCH") as caught:
        raise exception
    assert caught.value is exception


def test_error_helper_rejects_wrong_code():
    with pytest.raises(AssertionError, match="INVALID_CHECKPOINT"):
        with _error("IDENTITY_MISMATCH"):
            raise _api().CheckpointStoreError("INVALID_CHECKPOINT", "wrong category")


def test_error_helper_rejects_missing_exception():
    with pytest.raises(pytest.fail.Exception, match="DID NOT RAISE"):
        with _error("IDENTITY_MISMATCH"):
            pass


@pytest.mark.parametrize(
    "directory_name",
    [
        pytest.param("中文 space # &", id="portable-special-characters"),
        pytest.param(
            "中文 space # ?",
            id="posix-question-mark",
            marks=pytest.mark.skipif(os.name == "nt", reason="? is not a legal Windows filename"),
        ),
    ],
)
def test_prefix_round_trip_immutable_snapshot_and_closed_session(tmp_path, directory_name):
    case = _case()
    path = tmp_path / directory_name
    with _create(path, case) as session:
        empty = session.snapshot()
        assert empty.phase == "OPEN" and empty.replicates == ()
        assert empty.result_bytes is empty.metadata_bytes is None
        assert not hasattr(session, "connection")
        session.append(*case["outcomes"][0])
        saved = session.snapshot()
        assert saved.replicates == tuple(case["outcomes"][:1])
        with pytest.raises((FrozenInstanceError, AttributeError)):
            saved.phase = "FINALIZED"
    for operation in (
        session.snapshot,
        lambda: session.append(*case["outcomes"][1]),
        lambda: session.finalize(case["result"], case["metadata"]),
    ):
        with _error("CLOSED"):
            operation()
    with _open(path, case) as reopened:
        assert reopened.snapshot() == saved
        assert saved.header_bytes == case["header"]
        reopened.append(*case["outcomes"][1])
        assert empty.replicates == ()


@pytest.mark.skipif(os.name != "nt", reason="native Windows invalid-filename refusal")
def test_windows_invalid_checkpoint_name_is_refused_without_side_effects(tmp_path):
    sentinel = tmp_path / "keep.txt"
    sentinel.write_bytes(b"original")
    with pytest.raises(OSError) as refused, _create(tmp_path / "中文 space # ?", _case()):
        pytest.fail("Windows must refuse an invalid checkpoint name")
    assert refused.value.winerror == 123
    assert list(tmp_path.iterdir()) == [sentinel]
    assert sentinel.read_bytes() == b"original"


@pytest.mark.parametrize("index", [-1, 1, 3, True, 0.0, "0"])
def test_invalid_sequence_does_not_change_prefix(tmp_path, index):
    case = _case()
    with _create(tmp_path / "cp", case) as session:
        before = session.snapshot()
        with _error("INVALID_SEQUENCE"):
            session.append(index, case["outcomes"][0][1])
        assert session.snapshot() == before
        session.append(*case["outcomes"][0])
        with _error("INVALID_SEQUENCE"):
            session.append(*case["outcomes"][0])


@pytest.mark.parametrize(
    "change", ["id", "semantic", "plan", "float", "duplicate", "space", "deep", "type"]
)
def test_bad_outcome_is_refused_before_transaction(tmp_path, change):
    case = _case()
    data = json.loads(case["outcomes"][0][1])
    if change == "id":
        data["replicate_id"] = 1
    if change in {"semantic", "plan"}:
        data["transform_token"][
            "semantic_input_sha256" if change == "semantic" else "scientific_plan_sha256"
        ] = "0" * 64
    payload = _dump(data)
    if change == "float":
        payload = payload.replace(b'"replicate_id":0', b'"replicate_id":0.0')
    if change == "duplicate":
        payload = payload.replace(b'"replicate_id":0', b'"replicate_id":0,"replicate_id":0')
    if change == "space":
        payload += b" "
    if change == "deep":
        payload = b"[" * 2000 + b"0" + b"]" * 2000
    if change == "type":
        payload = bytearray(payload)
    with _create(tmp_path / "cp", case) as session:
        with _error(None):
            session.append(0, payload)
        assert session.snapshot().replicates == ()


def test_full_prefix_stays_open_until_atomic_matching_finalization(tmp_path):
    case = _case()
    path = tmp_path / "cp"
    with _create(path, case) as session:
        for row in case["outcomes"]:
            session.append(*row)
        before = session.snapshot()
        assert before.phase == "OPEN" and before.result_bytes is None
        session.finalize(case["result"], case["metadata"])
        saved = session.snapshot()
        assert saved.phase == "FINALIZED" and saved.result_bytes == case["result"]
        assert saved.metadata_bytes == case["metadata"]
        with _error("FINALIZED"):
            session.finalize(case["result"], case["metadata"])
        with _error("FINALIZED"):
            session.append(*case["outcomes"][0])
    with _open(path, case) as session:
        assert session.snapshot() == saved


@pytest.mark.parametrize("ne", ["observed", "null"])
def test_zero_row_pre_replicate_ne_is_structurally_admitted(tmp_path, ne):
    case = _case(ne=ne)
    assert case["outcomes"] == []
    with _create(tmp_path / "cp", case) as session:
        session.finalize(case["result"], case["metadata"])
        assert session.snapshot().phase == "FINALIZED"


@pytest.mark.parametrize("count", [0, 1, 2])
def test_incomplete_prefix_cannot_claim_complete_result(tmp_path, count):
    case = _case()
    with _create(tmp_path / "cp", case) as session:
        for row in case["outcomes"][:count]:
            session.append(*row)
        before = session.snapshot()
        with _error("INVALID_SEQUENCE"):
            session.finalize(case["result"], case["metadata"])
        assert session.snapshot() == before


@pytest.mark.parametrize("part", ["metadata", "result", "prefix", "uuid_in_metadata"])
def test_terminal_identity_and_exact_prefix_match(tmp_path, part):
    case = _case()
    result, metadata = case["result"], case["metadata"]
    if part == "metadata":
        data = json.loads(metadata)
        data["software"]["python_version"] = "different"
        metadata = _dump(data)
    elif part == "uuid_in_metadata":
        data = json.loads(metadata)
        data["execution_id"] = json.loads(case["header"])["execution_id"]
        metadata = _dump(data)
    else:
        data = json.loads(result)
        if part == "result":
            data["scientific_plan_sha256"] = "f" * 64
        else:
            data["replicates"][0]["diagnostics"].append("untrusted extra diagnostic")
        result = _dump(data)
    with _create(tmp_path / "cp", case) as session:
        for row in case["outcomes"]:
            session.append(*row)
        before = session.snapshot()
        with _error(None):
            session.finalize(result, metadata)
        assert session.snapshot() == before


@pytest.mark.parametrize("limit", [True, False, 0, -1, 1.0, "10000", None])
def test_invalid_limits_before_io(tmp_path, limit):
    case = _case()
    case["limit"] = limit
    with _error("INVALID_LIMIT"), _create(tmp_path / "new", case):
        pass
    with _error("INVALID_LIMIT"), _open(tmp_path / "missing", case):
        pass
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "unknown"),
        ("execution_id", "not-a-uuid"),
        ("planned_replicates", True),
        ("planned_replicates", 4),
        ("raw_input_sha256", "f" * 64),
        ("request_sha256", "0" * 64),
        ("input_format", "npz"),
        ("allow_unattainable", 1),
        ("max_bytes", LIMIT + 1),
        ("semantic_input_sha256", "A" * 64),
        ("extra", "x"),
        ("software", {}),
    ],
)
def test_invalid_header_before_create(tmp_path, field, value):
    case = _case()
    data = json.loads(case["header"])
    data[field] = value
    case["header"] = _dump(data)
    with _error(None), _create(tmp_path / "cp", case):
        pass
    assert not (tmp_path / "cp").exists()


@pytest.mark.parametrize(
    "payload", [b"", b"{", b"[" * 1000, b"\xff", b'{"x":NaN}', b'{"x":1,"x":2}']
)
def test_malformed_header_is_bounded_refusal(tmp_path, payload):
    case = _case()
    case["header"] = payload
    with _error("INVALID_CHECKPOINT"), _create(tmp_path / "cp", case):
        pass


def test_noncanonical_request_and_header_refused(tmp_path):
    case = _case()
    for member in ("request", "header"):
        broken = dict(case)
        broken[member] += b" "
        with _error(None), _create(tmp_path / member, broken):
            pass


def test_reopen_exact_identity_and_frozen_limit(tmp_path):
    case = _case()
    path = tmp_path / "cp"
    with _create(path, case):
        pass
    before = (path / "state.sqlite").read_bytes()
    wrong = {**case["software"], "numpy_version": "wrong"}
    with _error("IDENTITY_MISMATCH"), _open(path, case, expected_software=wrong):
        pass
    for limit in (LIMIT - 1, LIMIT + 1):
        with _error("IDENTITY_MISMATCH"), _open(path, case, max_bytes=limit):
            pass
    assert (path / "state.sqlite").read_bytes() == before


@pytest.mark.parametrize(
    "sql,values",
    [
        ("PRAGMA application_id=0", ()),
        ("PRAGMA user_version=2", ()),
        ("CREATE TABLE extra(x)", ()),
        ("CREATE INDEX extra ON replicates(sha256)", ()),
        ("CREATE VIEW extra AS SELECT * FROM members", ()),
        ("CREATE TRIGGER extra AFTER INSERT ON replicates BEGIN DELETE FROM state; END", ()),
        ("UPDATE state SET phase='other'", ()),
        ("UPDATE state SET result=x'31'", ()),
        ("UPDATE members SET sha256='bad' WHERE name='input'", ()),
        ("UPDATE members SET payload='text' WHERE name='input'", ()),
        ("DELETE FROM members WHERE name='request'", ()),
        ("DELETE FROM state", ()),
        ("UPDATE replicates SET rep_id=2 WHERE rep_id=0", ()),
        ("UPDATE replicates SET sha256='bad'", ()),
        ("UPDATE replicates SET payload=x'00'", ()),
    ],
)
def test_corrupted_database_refused(tmp_path, sql, values):
    case = _case()
    path = tmp_path / "cp"
    with _create(path, case) as session:
        session.append(*case["outcomes"][0])
    _mutate(path, sql, values)
    with _error("INVALID_CHECKPOINT"), _open(path, case):
        pass


def test_finalized_hash_and_phase_corruption_refused(tmp_path):
    case = _case()
    path = tmp_path / "cp"
    with _create(path, case) as session:
        for row in case["outcomes"]:
            session.append(*row)
        session.finalize(case["result"], case["metadata"])
    _mutate(path, "UPDATE state SET result_sha256='bad'")
    with _error("INVALID_CHECKPOINT"), _open(path, case):
        pass


@pytest.mark.parametrize("change", ["truncate", "trailing", "wal"])
def test_physical_database_invalidity(tmp_path, change):
    case = _case()
    path = tmp_path / "cp"
    with _create(path, case):
        pass
    database = path / "state.sqlite"
    if change == "wal":
        _mutate(path, "PRAGMA journal_mode=WAL")
    else:
        original = database.read_bytes()
        database.write_bytes(original[:-20] if change == "truncate" else original + b"junk")
    with (
        _error("UNSAFE_CHECKPOINT_FILE" if change == "wal" else "INVALID_CHECKPOINT"),
        _open(path, case),
    ):
        pass


@pytest.mark.parametrize(
    "name", ["state.sqlite", "state.sqlite-journal", "state.sqlite-wal", "state.sqlite-shm"]
)
@pytest.mark.parametrize("kind", ["symlink", "hardlink", "directory", "fifo"])
def test_unsafe_database_and_sidecars(tmp_path, name, kind):
    case = _case()
    path = tmp_path / "cp"
    with _create(path, case):
        pass
    target = path / name
    if target.exists():
        target.rename(tmp_path / "original")
    outside = tmp_path / "outside"
    outside.write_bytes(b"outside")
    if kind == "symlink":
        symlink_or_skip(target, outside)
    elif kind == "hardlink":
        os.link(outside, target)
    elif kind == "directory":
        target.mkdir()
    else:
        mkfifo_or_skip(target)
    with _error("UNSAFE_CHECKPOINT_FILE"), _open(path, case):
        pass
    assert outside.read_bytes() == b"outside"


def test_missing_database_never_created_and_occupied_creation_never_overwritten(tmp_path):
    case = _case()
    missing = tmp_path / "missing"
    missing.mkdir()
    with pytest.raises(FileNotFoundError), _open(missing, case):
        pass
    assert not (missing / "state.sqlite").exists()
    with pytest.raises(FileExistsError), _create(missing, case):
        pass
    path = tmp_path / "existing"
    with _create(path, case):
        pass
    before = (path / "state.sqlite").read_bytes()
    with pytest.raises(FileExistsError), _create(path, case):
        pass
    assert (path / "state.sqlite").read_bytes() == before


def test_schema_and_aggregate_limits(tmp_path):
    for name, limit, padding in (("schema", 8000, 0), ("aggregate", 20000, 20000)):
        case = _case(limit=limit, padding=padding)
        with _error("LIMIT_EXCEEDED"), _create(tmp_path / name, case):
            pass


def test_real_sqlite_page_full_keeps_last_committed_prefix(tmp_path):
    case = _case(limit=24576, replicates=9)
    path = tmp_path / "cp"
    with _create(path, case) as session:
        previous = session.snapshot()
        seen_full = False
        for row in case["outcomes"]:
            try:
                session.append(*row)
            except _api().CheckpointStoreError as error:
                assert error.code == "LIMIT_EXCEEDED"
                assert isinstance(error.__cause__, sqlite3.Error)
                seen_full = True
                break
            previous = session.snapshot()
        assert seen_full, "test must reach actual SQLite page-count overflow"
        assert session.snapshot() == previous
    with _open(path, case) as session:
        assert session.snapshot() == previous


def test_connections_close_on_user_exception_and_lock_reacquires(tmp_path, monkeypatch):
    case = _case()
    path = tmp_path / "cp"
    original, connections = sqlite3.connect, []

    def track(*args, **kwargs):
        connection = original(*args, **kwargs)
        connections.append(connection)
        return connection

    monkeypatch.setattr(sqlite3, "connect", track)
    with pytest.raises(RuntimeError, match="user interruption"), _create(path, case) as session:
        session.append(*case["outcomes"][0])
        raise RuntimeError("user interruption")
    with _open(path, case) as session:
        assert session.snapshot().replicates == tuple(case["outcomes"][:1])
    for connection in connections:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connection.execute("SELECT 1")


@pytest.mark.parametrize("operation", ["append", "finalize"])
def test_real_sqlite_busy_failure_preserves_prefix_and_same_session_can_retry(tmp_path, operation):
    case = _case()
    path = tmp_path / "cp"
    with _create(path, case) as session:
        if operation == "finalize":
            for row in case["outcomes"]:
                session.append(*row)
        before = session.snapshot()
        with closing(sqlite3.connect(path / "state.sqlite", isolation_level=None)) as blocker:
            blocker.execute("BEGIN IMMEDIATE")
            with _error("CHECKPOINT_BUSY") as caught:
                if operation == "append":
                    session.append(*case["outcomes"][0])
                else:
                    session.finalize(case["result"], case["metadata"])
            assert isinstance(caught.value.__cause__, sqlite3.OperationalError)
            blocker.rollback()
        assert session.snapshot() == before
        if operation == "append":
            session.append(*case["outcomes"][0])
        else:
            session.finalize(case["result"], case["metadata"])


def test_real_sqlite_full_finalization_is_atomic(tmp_path):
    case = _case(limit=28672)
    path = tmp_path / "cp"
    with _create(path, case) as session:
        for row in case["outcomes"]:
            session.append(*row)
        previous = session.snapshot()
        with _error("LIMIT_EXCEEDED") as caught:
            session.finalize(case["result"], case["metadata"])
        assert isinstance(caught.value.__cause__, sqlite3.Error)
        assert session.snapshot() == previous
    with _open(path, case) as reopened:
        assert reopened.snapshot() == previous


def test_aggregate_append_limit_precedes_sqlite_write(tmp_path):
    case = _case(limit=32768)
    data = json.loads(case["outcomes"][0][1])
    data["diagnostics"].append("x" * 32768)
    with _create(tmp_path / "cp", case) as session:
        before = session.snapshot()
        with _error("LIMIT_EXCEEDED") as caught:
            session.append(0, _dump(data))
        assert caught.value.__cause__ is None
        assert session.snapshot() == before


@pytest.mark.parametrize(
    "software",
    [
        {"python_version": ""},
        {"numpy_version": "x" * 257},
        {"selcal_version": 1},
        {"source_files": {"../x.py": "0" * 64}},
        {"source_files": {"/x.py": "0" * 64}},
        {"source_files": {"x//y.py": "0" * 64}},
        {"source_files": {"x.txt": "0" * 64}},
        {"source_files": {}},
        {"platform": {"system": "test"}},
    ],
)
def test_strict_software_identity_before_creation(tmp_path, software):
    case = _case()
    header = json.loads(case["header"])
    header["software"].update(software)
    case["header"] = _dump(header)
    with _error("INVALID_CHECKPOINT"), _create(tmp_path / "cp", case):
        pass
    assert not (tmp_path / "cp").exists()


def test_append_does_not_read_entire_old_prefix(tmp_path, monkeypatch):
    case = _case(replicates=9)
    path = tmp_path / "cp"
    original, statements = sqlite3.connect, []

    def track(*args, **kwargs):
        connection = original(*args, **kwargs)
        connection.set_trace_callback(statements.append)
        return connection

    monkeypatch.setattr(sqlite3, "connect", track)
    with _create(path, case) as session:
        statements.clear()
        for row in case["outcomes"]:
            session.append(*row)
        assert not any(sql.upper().startswith("SELECT") for sql in statements)
    statements.clear()
    with _open(path, case) as session:
        assert sum("FROM replicates" in sql for sql in statements) == 1
        snapshot = session.snapshot()
        assert snapshot.replicates == tuple(case["outcomes"])


def test_failed_validation_closes_connections_and_preserves_filesystem_errors(
    tmp_path, monkeypatch
):
    case = _case()
    path = tmp_path / "cp"
    with _create(path, case):
        pass
    _mutate(path, "DELETE FROM state")
    original, connections = sqlite3.connect, []

    def track(*args, **kwargs):
        connection = original(*args, **kwargs)
        connections.append(connection)
        return connection

    monkeypatch.setattr(sqlite3, "connect", track)
    with _error("INVALID_CHECKPOINT"), _open(path, case):
        pass
    for connection in connections:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connection.execute("SELECT 1")
    with pytest.raises(FileNotFoundError), _create(tmp_path / "absent-parent" / "cp", case):
        pass
