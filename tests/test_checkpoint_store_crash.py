"""Actual child death, lock reacquisition, and original-file hot rollback."""

from __future__ import annotations

import hashlib
import json
import os
import queue
import sys

import pytest
from _crash_process import CrashProcess
from test_checkpoint_store import _api, _case, _error, _open

from selcal.checkpoint_lock import CheckpointLockError

CHILD = r"""
import json, sqlite3, sys
from pathlib import Path
from _crash_process import crash_barrier
from selcal.checkpoint_store import create_checkpoint
fixture, directory = Path(sys.argv[1]), Path(sys.argv[2])
count, hot = int(sys.argv[3]), int(sys.argv[4])
header, raw, request = [(fixture / name).read_bytes() for name in ('header', 'raw', 'request')]
with create_checkpoint(directory, header_bytes=header, input_bytes=raw, request_bytes=request,
                       max_bytes=json.loads(header)['max_bytes']) as session:
    for i in range(count):
        session.append(i, (fixture / f'outcome{i}').read_bytes())
    print('COMMITTED', flush=True)
    assert sys.stdin.read(1) == 'G'
    if hot:
        inject = sqlite3.connect(directory / 'state.sqlite', isolation_level=None, timeout=0)
        assert inject.execute('PRAGMA journal_mode').fetchone() == ('delete',)
        inject.execute('PRAGMA synchronous=FULL')
        inject.execute('PRAGMA cache_size=2')
        inject.execute('PRAGMA cache_spill=ON')
        inject.execute('BEGIN IMMEDIATE')
        inject.execute("UPDATE members SET payload=? WHERE name='input'", (b'Z' * len(raw),))
    crash_barrier('store-ready')
"""


def _reader(stream, messages):
    for line in iter(stream.readline, ""):
        messages.put(line.rstrip("\n"))
    messages.put("EOF")


def _ready(messages, expected):
    try:
        actual = messages.get(timeout=30)
    except queue.Empty:
        pytest.fail(f"child did not reach {expected} within 30 seconds")
    assert actual == expected


@pytest.mark.parametrize(
    "directory_name",
    [
        pytest.param("checkpoint 中文 # &", id="portable-special-characters"),
        pytest.param(
            "checkpoint 中文 # ?",
            id="posix-question-mark",
            marks=pytest.mark.skipif(os.name == "nt", reason="? is not a legal Windows filename"),
        ),
    ],
)
@pytest.mark.parametrize(
    "count,hot,wrong_identity",
    [(1, False, False), (3, False, False), (1, True, False), (3, True, True)],
)
def test_real_kill_preserves_committed_prefix_and_product_recovers_hot_journal(
    tmp_path, count, hot, wrong_identity, directory_name
):
    _api()  # A missing implementation is an assertion failure, never a collection error.
    case = _case(padding=512 * 1024 if hot else 0)
    path, fixture = tmp_path / directory_name, tmp_path / "fixture"
    fixture.mkdir()
    for name in ("header", "raw", "request"):
        (fixture / name).write_bytes(case[name])
    for i, payload in case["outcomes"]:
        (fixture / f"outcome{i}").write_bytes(payload)
    child = CrashProcess(
        CHILD, [fixture, path, count, int(hot)], boundary="store-ready",
    )
    try:
        assert child.read_message() == "COMMITTED"
        database = path / "state.sqlite"
        original = database.read_bytes()
        with pytest.raises(CheckpointLockError) as busy, _open(path, case):
            pass
        assert busy.value.code == "CHECKPOINT_BUSY"
        child.stdin.write("G")
        child.stdin.flush()
        child.ready()
        journal = path / "state.sqlite-journal"
        receipt = {
            "pid": child.pid,
            "committed_count": count,
            "hot_journal": hot,
            "original_sha256": hashlib.sha256(original).hexdigest(),
            "execution_id": json.loads(case["header"])["execution_id"],
        }
        if hot:
            spilled, journal_bytes = database.read_bytes(), journal.read_bytes()
            assert spilled != original, "uncommitted changes must have spilled to the original DB"
            assert len(journal_bytes) > 512
            assert journal_bytes[:8] == bytes.fromhex("d9d505f920a163d7")
            assert int.from_bytes(journal_bytes[8:12], "big") > 0
            encoded_page_size = int.from_bytes(original[16:18], "big")
            page_size = 65536 if encoded_page_size == 1 else encoded_page_size
            assert page_size > 0 and len(original) % page_size == 0
            assert int.from_bytes(journal_bytes[16:20], "big") == len(original) // page_size
            assert int.from_bytes(journal_bytes[24:28], "big") == page_size
            # Test-owned artifacts retain the actual spill proof, not a simulated crash.
            (tmp_path / "committed.sqlite.bytes").write_bytes(original)
            (tmp_path / "spilled.sqlite.bytes").write_bytes(spilled)
            (tmp_path / "hot.journal.bytes").write_bytes(journal_bytes)
            receipt.update(
                spilled_sha256=hashlib.sha256(spilled).hexdigest(),
                journal_sha256=hashlib.sha256(journal_bytes).hexdigest(),
                journal_bytes=len(journal_bytes),
                journal_header_hex=journal_bytes[:28].hex(),
            )
        receipt["crash"] = child.force_kill()
        # From this point the first DB connection is the production opener. In the wrong
        # identity case it must recover first, then refuse, without application writes.
        if wrong_identity:
            wrong = {**case["software"], "python_version": "different"}
            with _error("IDENTITY_MISMATCH"), _open(path, case, expected_software=wrong):
                pass
            assert database.read_bytes() == original
        with _open(path, case) as recovered:
            snapshot = recovered.snapshot()
            assert snapshot.header_bytes == case["header"]
            assert (
                json.loads(snapshot.header_bytes)["execution_id"]
                == json.loads(case["header"])["execution_id"]
            )
            assert snapshot.input_bytes == case["raw"]
            assert snapshot.request_bytes == case["request"]
            assert snapshot.replicates == tuple(case["outcomes"][:count])
            assert snapshot.phase == "OPEN"
            assert snapshot.result_bytes is snapshot.metadata_bytes is None
        assert database.read_bytes() == original
        if hot:
            assert not journal.exists(), "SQLite DELETE recovery must handle the hot journal"
        receipt.update(
            returncode=child.returncode,
            recovered_sha256=hashlib.sha256(database.read_bytes()).hexdigest(),
        )
        (tmp_path / "CRASH_RECEIPT.json").write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
        )
        stderr = child.stderr.read()
        assert not stderr, stderr
    finally:
        child.cleanup(sys.exc_info()[1])
