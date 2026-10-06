"""Real process death and a fresh interpreter using the product recovery path."""

from __future__ import annotations

import json
import sys

import pytest
from _crash_process import CrashProcess
from test_workflow import CAP, files
from test_workflow_recovery import recovery, snapshot

from selcal import workflow
from selcal.checkpoint_lock import CheckpointLockError
from selcal.workflow_store import read_record

CHILD = r"""
import dataclasses, json, sqlite3, sys
from _crash_process import crash_barrier
from selcal.workflow_recovery import run_checkpointed_files, resume_checkpoint
mode, inp, config, output, cp, boundary, count = sys.argv[1:]
count = int(count)
def handshake():
    crash_barrier(boundary)
if boundary in ('BEFORE_COMMIT', 'AFTER_COMMIT'):
    connect = sqlite3.connect
    class Connection(sqlite3.Connection):
        pending = False
        def execute(self, sql, parameters=()):
            if sql.startswith('INSERT INTO replicates') and parameters[0] == count - 1:
                self.pending = True
            if sql == 'COMMIT' and self.pending and boundary == 'BEFORE_COMMIT':
                handshake()
            answer = super().execute(sql, parameters)
            if sql == 'COMMIT' and self.pending and boundary == 'AFTER_COMMIT':
                handshake()
            return answer
    def intercepted(*args, **kwargs):
        return connect(*args, **kwargs, factory=Connection)
    sqlite3.connect = intercepted
def progress(event):
    current = event.replayed_replicates if boundary == 'replay' else event.retained_replicates
    if boundary == event.phase and current == count:
        handshake()
if mode == 'run':
    result = run_checkpointed_files(inp, config, output, cp, max_bytes=1048576,
                                   allow_unattainable=True, progress=progress)
else:
    result = resume_checkpoint(cp, output, max_bytes=1048576, progress=progress)
print(json.dumps({'execution_id': result.execution_id, 'replayed': result.replayed_replicates,
                  'appended': result.appended_replicates}), flush=True)
"""


def launch(inp, cfg, out, cp, boundary, count, *, mode="run"):
    return CrashProcess(
        CHILD,
        [
            mode,
            str(inp),
            str(cfg),
            str(out),
            str(cp),
            boundary,
            str(count),
        ],
        boundary=boundary,
    )


def ready(child):
    return child.ready()


def kill(child):
    return child.force_kill()


@pytest.mark.parametrize(
    "boundary,count,retained",
    [
        ("commit", 3, 3),
        ("BEFORE_COMMIT", 3, 2),
        ("AFTER_COMMIT", 3, 3),
        ("commit", 9, 9),
    ],
)
def test_kill_then_fresh_process_replay_and_continue(tmp_path, boundary, count, retained):
    app = recovery()
    inp, cfg = files(tmp_path)
    cp, output = tmp_path / "checkpoint", tmp_path / "output"
    ordinary = tmp_path / "ordinary"
    workflow.run_files(inp, cfg, ordinary, max_bytes=CAP, allow_unattainable=True)
    child = launch(inp, cfg, output, cp, boundary, count)
    try:
        ready(child)
        with pytest.raises(CheckpointLockError) as busy:
            app.resume_checkpoint(cp, tmp_path / "concurrent", max_bytes=CAP)
        assert busy.value.code == "CHECKPOINT_BUSY"
        crash_receipt = kill(child)
    finally:
        child.cleanup(sys.exc_info()[1])
    before = snapshot(cp)
    assert before.phase == "OPEN" and len(before.replicates) == retained
    assert not output.exists() and not (tmp_path / "concurrent").exists()
    inp.unlink()
    cfg.unlink()
    resumed = launch(inp, cfg, output, cp, "none", 0, mode="resume")
    stdout, stderr = resumed.communicate(timeout=30)
    assert resumed.returncode == 0, stderr
    summary = json.loads(stdout)
    assert summary == {
        "execution_id": json.loads(before.header_bytes)["execution_id"],
        "replayed": retained,
        "appended": 9 - retained,
    }
    after = snapshot(cp)
    assert after.replicates[:retained] == before.replicates
    assert [i for i, _ in after.replicates] == list(range(9))
    assert read_record(output, max_bytes=CAP) == read_record(ordinary, max_bytes=CAP)
    (tmp_path / "process_receipt.json").write_text(
        json.dumps(
            {
                "boundary": boundary,
                "killed_pid": child.pid,
                "killed_exit": child.returncode,
                "crash": crash_receipt,
                "resume_pid": resumed.pid,
                "resume_exit": resumed.returncode,
                "summary": summary,
            },
            indent=2,
        ), encoding="utf-8"
    )


def test_interrupt_again_during_replay_and_append(tmp_path):
    recovery()
    inp, cfg = files(tmp_path)
    cp, output = tmp_path / "checkpoint", tmp_path / "output"
    interruptions = []
    receipt_path = tmp_path / "repeated_process_receipts.json"
    for mode, phase, stop, retained in [
        ("run", "commit", 3, 3),
        ("resume", "replay", 2, 3),
        ("resume", "commit", 5, 5),
    ]:
        child = launch(inp, cfg, output, cp, phase, stop, mode=mode)
        try:
            ready(child)
            crash_receipt = kill(child)
        finally:
            child.cleanup(sys.exc_info()[1])
        interruptions.append(
            {
                "mode": mode,
                "boundary": phase,
                "count": stop,
                "retained": retained,
                "crash": crash_receipt,
            }
        )
        receipt = json.dumps({"interruptions": interruptions}, indent=2) + "\n"
        receipt_path.write_text(receipt, encoding="utf-8")
        assert len(snapshot(cp).replicates) == retained and not output.exists()
    final = launch(inp, cfg, output, cp, "none", 0, mode="resume")
    stdout, stderr = final.communicate(timeout=30)
    assert final.returncode == 0, stderr
    assert json.loads(stdout)["replayed"] == 5
    assert json.loads(stdout)["appended"] == 4
    receipt_path.write_text(
        json.dumps({"interruptions": interruptions, "summary": json.loads(stdout)}, indent=2)
        + "\n",
        encoding="utf-8",
    )
