"""Real scientific replay is required before extending any stored prefix."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import importlib.util
import json
import sqlite3

import numpy as np
import pytest
from _platform_support import directory_link_or_skip, symlink_or_skip
from test_calibration_v2_progress import _case
from test_workflow import CAP, files

from selcal import workflow
from selcal.checkpoint_store import CheckpointStoreError, open_checkpoint
from selcal.workflow_config import WorkflowConfig, encode_workflow_config
from selcal.workflow_store import read_record


def recovery():
    assert importlib.util.find_spec("selcal.workflow_recovery") is not None, (
        "missing replay-before-continue workflow service"
    )
    return importlib.import_module("selcal.workflow_recovery")


@pytest.mark.parametrize("form", ["csv", "npz"])
@pytest.mark.parametrize(
    "mode", ["complete", "null_bind", "observed_statistic_scan", "replicate_execution"]
)
def test_actual_run_and_finalized_resume_equal_ordinary_bytes(tmp_path, monkeypatch, form, mode):
    app = recovery()
    inp, cfg = files(tmp_path, form=form, mode=mode)
    ordinary, output, second = (tmp_path / name for name in ("ordinary", "output", "second"))
    checkpoint = tmp_path / "checkpoint"
    workflow.run_files(inp, cfg, ordinary, max_bytes=CAP, allow_unattainable=True)
    kernel = app._calibrate_selected_family_observed
    calls = []

    def observe(*args):
        calls.append(1)
        return kernel(*args)

    monkeypatch.setattr(app, "_calibrate_selected_family_observed", observe)
    events = []
    first = app.run_checkpointed_files(
        inp, cfg, output, checkpoint, max_bytes=CAP, allow_unattainable=True, progress=events.append
    )
    expected = read_record(ordinary, max_bytes=CAP)
    assert read_record(output, max_bytes=CAP) == expected
    count = len(first.result.replicates)
    assert first.replayed_replicates == 0 and first.appended_replicates == count
    assert len(calls) == 1
    assert [e.phase for e in events] == ["commit"] * count + ["finalized"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        events[-1].retained_replicates = 0
    inp.unlink()
    cfg.unlink()
    replay_events = []
    resumed = app.resume_checkpoint(
        checkpoint, second, max_bytes=CAP, progress=replay_events.append
    )
    assert read_record(second, max_bytes=CAP) == expected
    assert resumed.execution_id == first.execution_id
    assert resumed.result is not first.result
    assert resumed.replayed_replicates == count and resumed.appended_replicates == 0
    assert len(calls) == 2
    assert [e.phase for e in replay_events] == ["replay"] * count + ["finalized"]


def test_interrupt_committed_prefix_then_replay_before_continue(tmp_path):
    app = recovery()
    inp, cfg = files(tmp_path)
    cp, out = tmp_path / "checkpoint", tmp_path / "out"

    def stop(event):
        if event.retained_replicates == 3:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        app.run_checkpointed_files(
            inp, cfg, out, cp, max_bytes=CAP, allow_unattainable=True, progress=stop
        )
    with open_checkpoint(
        cp, expected_software=workflow._software_identity(), max_bytes=CAP
    ) as store:
        before = store.snapshot()
    assert before.phase == "OPEN" and len(before.replicates) == 3
    assert not out.exists()
    events = []
    run = app.resume_checkpoint(cp, out, max_bytes=CAP, progress=events.append)
    assert (run.replayed_replicates, run.appended_replicates) == (3, 6)
    assert run.execution_id == json.loads(before.header_bytes)["execution_id"]
    assert [e.phase for e in events] == ["replay"] * 3 + ["commit"] * 6 + ["finalized"]
    with open_checkpoint(
        cp, expected_software=workflow._software_identity(), max_bytes=CAP
    ) as store:
        after = store.snapshot()
    assert after.replicates[:3] == before.replicates
    assert [i for i, _ in after.replicates] == list(range(9))


def test_interrupt_after_last_commit_before_finalization_resumes_without_new_work(tmp_path):
    """Cancellation can arrive after every replicate is committed but before finalization.

    The checkpoint then holds the complete prefix and stays OPEN. Resume must replay every
    stored replicate, append none, finalize, and export the same record as an ordinary run.
    (Observed in the UI test under load, 2026-10-02: cancel landed after the last commit.)
    """
    app = recovery()
    inp, cfg = files(tmp_path)
    ordinary, cp, out = tmp_path / "ordinary", tmp_path / "checkpoint", tmp_path / "out"
    workflow.run_files(inp, cfg, ordinary, max_bytes=CAP, allow_unattainable=True)
    probe = []
    app.run_checkpointed_files(
        inp,
        cfg,
        tmp_path / "probe",
        tmp_path / "probe-cp",
        max_bytes=CAP,
        allow_unattainable=True,
        progress=probe.append,
    )
    total = sum(1 for e in probe if e.phase == "commit")
    assert total > 1
    seen = []

    def stop_after_last(event):
        seen.append(event.phase)
        if event.phase == "commit" and event.retained_replicates == total:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        app.run_checkpointed_files(
            inp, cfg, out, cp, max_bytes=CAP, allow_unattainable=True, progress=stop_after_last
        )
    assert "finalized" not in seen
    stored = snapshot(cp)
    assert stored.phase == "OPEN" and len(stored.replicates) == total
    assert not out.exists()
    events = []
    run = app.resume_checkpoint(cp, out, max_bytes=CAP, progress=events.append)
    assert (run.replayed_replicates, run.appended_replicates) == (total, 0)
    assert [e.phase for e in events] == ["replay"] * total + ["finalized"]
    assert read_record(out, max_bytes=CAP) == read_record(ordinary, max_bytes=CAP)
    assert snapshot(cp).replicates == stored.replicates


@pytest.mark.parametrize(
    "case", ["exists", "symlink", "parent", "parent_file", "limit", "override", "admission"]
)
def test_initial_rejections_do_not_create_checkpoint(tmp_path, case):
    app = recovery()
    inp, cfg = files(tmp_path)
    cp, out, cap, override = tmp_path / "checkpoint", tmp_path / "out", CAP, True
    error = ValueError
    if case == "exists":
        out.write_bytes(b"keep")
        error = FileExistsError
    elif case == "symlink":
        symlink_or_skip(out, tmp_path / "absent")
        error = FileExistsError
    elif case == "parent":
        out = tmp_path / "absent" / "out"
        error = FileNotFoundError
    elif case == "parent_file":
        out.write_bytes(b"keep")
        out = out / "child"
        error = NotADirectoryError
    elif case == "limit":
        cap = True
    elif case == "override":
        override = 1
    else:
        override = False
    with pytest.raises(error):
        app.run_checkpointed_files(inp, cfg, out, cp, max_bytes=cap, allow_unattainable=override)
    assert not cp.exists()


@pytest.mark.parametrize("case", ["sampled_pearson", "exact_pearson", "block_pearson", "binned"])
def test_real_registered_adapter_families_preserve_scientific_bytes(tmp_path, case):
    app = recovery()
    pair, request = _case(case)
    inp, cfg = tmp_path / "input.npz", tmp_path / "config.json"
    np.savez(inp, source=pair.source, target=pair.target)
    cfg.write_bytes(encode_workflow_config(WorkflowConfig(request, "npz")))
    ordinary, output, cp = tmp_path / "ordinary", tmp_path / "output", tmp_path / "checkpoint"
    workflow.run_files(inp, cfg, ordinary, max_bytes=CAP, allow_unattainable=True)
    run = app.run_checkpointed_files(inp, cfg, output, cp, max_bytes=CAP, allow_unattainable=True)
    assert read_record(ordinary, max_bytes=CAP) == read_record(output, max_bytes=CAP)
    assert run.result.planned_replicates == request.replicates
    assert run.appended_replicates == request.replicates


def partial(tmp_path, *, count=3):
    app = recovery()
    inp, cfg = files(tmp_path)
    cp = tmp_path / "checkpoint"

    def stop(event):
        if event.phase == "commit" and event.retained_replicates == count:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        app.run_checkpointed_files(
            inp,
            cfg,
            tmp_path / "not-exported",
            cp,
            max_bytes=CAP,
            allow_unattainable=True,
            progress=stop,
        )
    return cp


def snapshot(cp):
    with open_checkpoint(
        cp, expected_software=workflow._software_identity(), max_bytes=CAP
    ) as store:
        return store.snapshot()


def _replace_member(connection, name, payload):
    connection.execute(
        "UPDATE members SET payload=?, sha256=? WHERE name=?",
        (payload, hashlib.sha256(payload).hexdigest(), name),
    )


@pytest.mark.parametrize("kind", ["unselected", "seed", "token"])
def test_rehashed_structurally_valid_prefix_tampering_fails_exact_replay(tmp_path, kind):
    app = recovery()
    cp = partial(tmp_path)
    with sqlite3.connect(cp / "state.sqlite") as connection:
        payload = json.loads(
            connection.execute("SELECT payload FROM replicates WHERE rep_id=0").fetchone()[0]
        )
        if kind == "unselected":
            selected = payload["selection"]["selected_candidate"]
            row = next(
                row for row in payload["statistic_results"] if row["candidate_id"] != selected
            )
            row["estimate"] = {"$float64": "0x1.0000000000000p-10"}
        elif kind == "seed":
            payload["seed_digest_sha256"] = "1" * 64
        else:
            # Circular-shift token remains valid but belongs to a different transform.
            payload["transform_token"]["state"]["shift"] = 1 + (
                payload["transform_token"]["state"]["shift"] % 5
            )
        altered = workflow._json(payload)
        connection.execute(
            "UPDATE replicates SET payload=?, sha256=? WHERE rep_id=0",
            (altered, hashlib.sha256(altered).hexdigest()),
        )
    # Store can check structure and hashes but cannot grant scientific authority.
    before = snapshot(cp)
    events = []
    with pytest.raises(CheckpointStoreError) as failure:
        app.resume_checkpoint(cp, tmp_path / "out", max_bytes=CAP, progress=events.append)
    assert failure.value.code == "REPLAY_MISMATCH"
    assert not events and not (tmp_path / "out").exists()
    assert snapshot(cp) == before


@pytest.mark.parametrize(
    "kind", ["raw", "request", "plan", "python_version", "numpy_version", "platform", "limit"]
)
def test_retained_identity_tampering_cannot_write_rows(tmp_path, kind):
    app = recovery()
    cp = partial(tmp_path)
    if kind != "limit":
        with sqlite3.connect(cp / "state.sqlite") as connection:
            header = json.loads(
                connection.execute("SELECT payload FROM members WHERE name='header'").fetchone()[0]
            )
            if kind == "raw":
                raw = (
                    connection.execute("SELECT payload FROM members WHERE name='input'")
                    .fetchone()[0]
                    .replace(b"0,0", b"2,0")
                )
                _replace_member(connection, "input", raw)
                header["raw_input_sha256"] = hashlib.sha256(raw).hexdigest()
            elif kind == "request":
                request = json.loads(
                    connection.execute(
                        "SELECT payload FROM members WHERE name='request'"
                    ).fetchone()[0]
                )
                request["plan"]["root_seed"] += 1
                raw = workflow._json(request)
                _replace_member(connection, "request", raw)
                header["request_sha256"] = hashlib.sha256(raw).hexdigest()
            elif kind == "plan":
                header["scientific_plan_sha256"] = "1" * 64
            elif kind == "platform":
                header["software"]["platform"]["system"] = "Different OS"
            else:
                header["software"][kind] = "different"
            _replace_member(connection, "header", workflow._json(header))
    before = (cp / "state.sqlite").read_bytes()
    events = []
    with pytest.raises(CheckpointStoreError):
        app.resume_checkpoint(
            cp, tmp_path / "out", max_bytes=CAP + (kind == "limit"), progress=events.append
        )
    assert not events and not (tmp_path / "out").exists()
    assert (cp / "state.sqlite").read_bytes() == before


def test_nonempty_prefix_with_zero_callback_ne_cannot_finalize(tmp_path, monkeypatch):
    app = recovery()
    cp = partial(tmp_path)
    before = snapshot(cp)
    ne_folder = tmp_path / "ne"
    ne_folder.mkdir()
    inp, cfg = files(ne_folder, mode="null_bind")
    ne = workflow.run_files(inp, cfg, ne_folder / "out", max_bytes=CAP, allow_unattainable=True)
    # Controlled corrupt-kernel boundary: zero callback must fail before finalization.
    monkeypatch.setattr(app, "_calibrate_selected_family_observed", lambda *args: ne)
    events = []
    with pytest.raises(CheckpointStoreError, match="before replaying"):
        app.resume_checkpoint(cp, tmp_path / "out", max_bytes=CAP, progress=events.append)
    assert snapshot(cp) == before and not events and not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "failure", [RuntimeError("callback"), ValueError("callback"), KeyboardInterrupt()]
)
def test_callback_failure_propagates_original_object_and_preserves_commit(tmp_path, failure):
    app = recovery()
    inp, cfg = files(tmp_path)
    cp = tmp_path / "checkpoint"

    def stop(event):
        raise failure

    with pytest.raises(type(failure)) as raised:
        app.run_checkpointed_files(
            inp, cfg, tmp_path / "out", cp, max_bytes=CAP, allow_unattainable=True, progress=stop
        )
    assert raised.value is failure
    assert len(snapshot(cp).replicates) == 1 and not (tmp_path / "out").exists()


def test_export_failure_preserves_finalized_checkpoint_for_new_filename(tmp_path, monkeypatch):
    app = recovery()
    inp, cfg = files(tmp_path)
    cp, output = tmp_path / "checkpoint", tmp_path / "out"
    real_write = app.write_record

    def collision(*args, **kwargs):
        output.write_bytes(b"concurrent owner")
        return real_write(*args, **kwargs)

    monkeypatch.setattr(app, "write_record", collision)
    with pytest.raises(FileExistsError):
        app.run_checkpointed_files(inp, cfg, output, cp, max_bytes=CAP, allow_unattainable=True)
    before = snapshot(cp)
    assert before.phase == "FINALIZED" and output.read_bytes() == b"concurrent owner"
    monkeypatch.setattr(app, "write_record", real_write)
    run = app.resume_checkpoint(cp, tmp_path / "new", max_bytes=CAP)
    assert run.replayed_replicates == 9 and run.appended_replicates == 0
    assert snapshot(cp) == before


def test_rehashed_nonempty_prefix_that_now_computes_real_zero_row_ne_is_rejected(tmp_path):
    app = recovery()
    cp = partial(tmp_path)
    original = snapshot(cp)
    config = workflow.decode_workflow_config(original.request_bytes)
    raw = b"x,y\n1,0\n1,1\n1,3\n1,2\n1,5\n1,4\n"
    altered_input = tmp_path / "changed.csv"
    altered_input.write_bytes(raw)
    loaded = workflow._load(altered_input, config)
    with sqlite3.connect(cp / "state.sqlite") as connection:
        header = json.loads(original.header_bytes)
        header["raw_input_sha256"] = loaded.raw_input_sha256
        header["semantic_input_sha256"] = loaded.semantic_input_sha256
        _replace_member(connection, "input", raw)
        _replace_member(connection, "header", workflow._json(header))
        for index, payload in original.replicates:
            outcome = json.loads(payload)
            outcome["transform_token"]["semantic_input_sha256"] = loaded.semantic_input_sha256
            changed = workflow._json(outcome)
            connection.execute(
                "UPDATE replicates SET payload=?,sha256=? WHERE rep_id=?",
                (changed, hashlib.sha256(changed).hexdigest(), index),
            )
    before = snapshot(cp)
    events = []
    with pytest.raises(CheckpointStoreError, match="before replaying") as error:
        app.resume_checkpoint(cp, tmp_path / "out", max_bytes=CAP, progress=events.append)
    assert error.value.code == "REPLAY_MISMATCH"
    assert snapshot(cp) == before and not events and not (tmp_path / "out").exists()


@pytest.mark.parametrize("kind", ["result", "metadata"])
def test_coherently_rehashed_terminal_content_is_never_exported(tmp_path, kind):
    app = recovery()
    inp, cfg = files(tmp_path)
    cp = tmp_path / "checkpoint"
    app.run_checkpointed_files(
        inp, cfg, tmp_path / "first", cp, max_bytes=CAP, allow_unattainable=True
    )
    with sqlite3.connect(cp / "state.sqlite") as connection:
        result, metadata = connection.execute("SELECT result,metadata FROM state").fetchone()
        value = json.loads(result if kind == "result" else metadata)
        if kind == "result":
            value["diagnostics"].append("unexecuted terminal")
        else:
            value["raw_input_sha256"] = "1" * 64
        payload = workflow._json(value)
        connection.execute(
            f"UPDATE state SET {kind}=?,{kind}_sha256=? WHERE id=1",
            (payload, hashlib.sha256(payload).hexdigest()),
        )
    before = (cp / "state.sqlite").read_bytes()
    events = []
    with pytest.raises(CheckpointStoreError) as error:
        app.resume_checkpoint(cp, tmp_path / "out", max_bytes=CAP, progress=events.append)
    assert error.value.code == ("REPLAY_MISMATCH" if kind == "result" else "IDENTITY_MISMATCH")
    assert all(event.phase == "replay" for event in events)
    assert not (tmp_path / "out").exists() and (cp / "state.sqlite").read_bytes() == before


def test_stored_override_does_not_skip_repeated_admission(tmp_path):
    app = recovery()
    cp = partial(tmp_path)
    with sqlite3.connect(cp / "state.sqlite") as connection:
        header = json.loads(
            connection.execute("SELECT payload FROM members WHERE name='header'").fetchone()[0]
        )
        header["allow_unattainable"] = False
        _replace_member(connection, "header", workflow._json(header))
    before = snapshot(cp)
    with pytest.raises(workflow.WorkflowConfigError) as error:
        app.resume_checkpoint(cp, tmp_path / "out", max_bytes=CAP)
    assert error.value.code == "unattainable_plan"
    assert snapshot(cp) == before and not (tmp_path / "out").exists()


def test_resume_opens_once_and_keeps_writer_lock_through_export(tmp_path, monkeypatch):
    from selcal.checkpoint_lock import CheckpointLockError

    app = recovery()
    cp = partial(tmp_path)
    real_open, real_write, openings = app.open_checkpoint, app.write_record, []

    def counted(*args, **kwargs):
        openings.append(1)
        return real_open(*args, **kwargs)

    def checked_export(*args, **kwargs):
        with pytest.raises(CheckpointLockError):
            with real_open(cp, expected_software=workflow._software_identity(), max_bytes=CAP):
                pytest.fail("writer lock was released before export")
        return real_write(*args, **kwargs)

    monkeypatch.setattr(app, "open_checkpoint", counted)
    monkeypatch.setattr(app, "write_record", checked_export)
    app.resume_checkpoint(cp, tmp_path / "out", max_bytes=CAP)
    assert openings == [1]


def test_source_change_after_kernel_retains_unfinalized_prefix(tmp_path, monkeypatch):
    app = recovery()
    inp, cfg = files(tmp_path)
    cp = tmp_path / "checkpoint"
    real = workflow._software_identity
    calls = []

    def changing():
        calls.append(1)
        identity = real()
        if len(calls) == 2:
            identity["python_version"] = "changed"
        return identity

    monkeypatch.setattr(workflow, "_software_identity", changing)
    with pytest.raises(CheckpointStoreError) as error:
        app.run_checkpointed_files(
            inp, cfg, tmp_path / "out", cp, max_bytes=CAP, allow_unattainable=True
        )
    assert error.value.code == "IDENTITY_MISMATCH"
    monkeypatch.setattr(workflow, "_software_identity", real)
    assert snapshot(cp).phase == "OPEN" and len(snapshot(cp).replicates) == 9
    assert not (tmp_path / "out").exists()


_CHECKPOINT_RESERVED_OUTPUTS = (
    "state.sqlite",
    "writer.lock",
    "state.sqlite-journal",
    "state.sqlite-wal",
    "state.sqlite-shm",
)


@pytest.mark.parametrize("stage", ["initial", "partial", "finalized"])
@pytest.mark.parametrize("name", _CHECKPOINT_RESERVED_OUTPUTS)
@pytest.mark.parametrize(
    "alias", ["direct", "dotdot", "symlink", "case_basename", "case_directory"]
)
def test_reserved_checkpoint_output_rejects_before_computing_or_mutating(
    tmp_path,
    monkeypatch,
    stage,
    name,
    alias,
):
    app = recovery()
    cp = tmp_path / "checkpoint"
    if stage == "partial":
        cp = partial(tmp_path)
    else:
        inp, cfg = files(tmp_path)
        if stage == "finalized":
            app.run_checkpointed_files(
                inp, cfg, tmp_path / "first", cp, max_bytes=CAP, allow_unattainable=True
            )
    if stage != "initial":
        before = snapshot(cp)
        original_files = {p.name: p.read_bytes() for p in cp.iterdir() if p.is_file()}
    parent = cp
    if alias == "dotdot":
        (tmp_path / "other").mkdir()
        parent = tmp_path / "other" / ".." / "checkpoint"
    elif alias == "symlink":
        directory_link_or_skip(tmp_path / "alias", tmp_path)
        parent = tmp_path / "alias" / "checkpoint"
    elif alias == "case_basename":
        name = name.upper()
    elif alias == "case_directory":
        parent = tmp_path / "CHECKPOINT"
        if stage != "initial" and (not parent.exists() or not parent.samefile(cp)):
            pytest.skip("filesystem does not resolve case-variant directory aliases")

    def must_not_compute(*args):
        pytest.fail("reserved output must be rejected before scientific replay or append")

    monkeypatch.setattr(app, "_calibrate_selected_family_observed", must_not_compute)
    events = []
    with pytest.raises((OSError, CheckpointStoreError)) as error:
        if stage == "initial":
            app.run_checkpointed_files(
                inp,
                cfg,
                parent / name,
                cp,
                max_bytes=CAP,
                allow_unattainable=True,
                progress=events.append,
            )
        else:
            app.resume_checkpoint(cp, parent / name, max_bytes=CAP, progress=events.append)
    assert not events
    if stage == "initial":
        assert isinstance(error.value, FileNotFoundError)
        assert not cp.exists()
    else:
        if isinstance(error.value, CheckpointStoreError):
            assert error.value.code == "RESERVED_OUTPUT"
        else:
            assert isinstance(error.value, FileExistsError)
        assert snapshot(cp) == before
        assert {p.name: p.read_bytes() for p in cp.iterdir() if p.is_file()} == original_files


@pytest.mark.parametrize("stage", ["initial", "partial", "finalized"])
@pytest.mark.parametrize("alias", ["direct", "symlink"])
def test_ordinary_output_filename_remains_usable(tmp_path, stage, alias):
    app = recovery()
    cp = tmp_path / "checkpoint"
    if stage == "partial":
        cp = partial(tmp_path)
    else:
        inp, cfg = files(tmp_path)
        if stage == "finalized":
            app.run_checkpointed_files(
                inp, cfg, tmp_path / "first", cp, max_bytes=CAP, allow_unattainable=True
            )
    # Initial output still needs an existing parent, as required by ordinary run.
    parent = tmp_path if stage == "initial" else cp
    if alias == "symlink":
        directory_link_or_skip(tmp_path / "alias", parent)
        parent = tmp_path / "alias"
    output = parent / "result.sqlite"
    if stage == "initial":
        run = app.run_checkpointed_files(
            inp, cfg, output, cp, max_bytes=CAP, allow_unattainable=True
        )
    else:
        run = app.resume_checkpoint(cp, output, max_bytes=CAP)
    assert run.result.p_value == 0.2
    assert read_record(output, max_bytes=CAP)["result"] == snapshot(cp).result_bytes
