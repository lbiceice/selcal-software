"""P19-04: offline source, truthful input identities, and observable snapshot cleanup.

These lifecycle/failure tests use real empty UI workspaces, not scientific-result mocks.
Real artifact binding, replay, ZIP and HTML checks stay in test_acceptance_tools.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys
from pathlib import Path

import pytest
from test_ui_jobs import physical_tmp as _physical_tmp

from selcal.ui_jobs import JobManager

physical_tmp = _physical_tmp
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def checker():
    spec = importlib.util.spec_from_file_location(
        "saved_downloads_under_test", ROOT / "scripts" / "check_saved_downloads.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args(base: Path) -> argparse.Namespace:
    workspace, saved, out = base / "workspace", base / "saved", base / "out"
    manager = JobManager(workspace)
    manager.close()
    saved.mkdir()
    (saved / "unknown.bin").write_bytes(b"not a SelCal artifact")
    out.mkdir()
    return argparse.Namespace(
        workspace=workspace, saved=saved, out=out, expect=None, count=None,
        python=sys.executable, max_bytes="8388608", allow_history=False,
    )


def _bytes(folder: Path) -> dict[str, bytes]:
    return {
        path.relative_to(folder).as_posix(): path.read_bytes()
        for path in folder.rglob("*") if path.is_file()
    }


def test_closed_source_retained_lock_is_valid_and_failure_snapshot_is_removed(
    checker, physical_tmp
):
    args = _args(physical_tmp)
    before, saved_before = _bytes(args.workspace), _bytes(args.saved)
    receipt = checker.check(args)
    assert receipt["status"] == "FAIL"  # An unknown filename is still refused.
    assert receipt["failure_phase"] == "saved_file_check"
    assert receipt["failure_path"] == str(args.saved / "unknown.bin")
    assert receipt["source_access"] == "OFFLINE_AVAILABLE"
    assert receipt["inputs_unchanged"] is True
    assert _bytes(args.workspace) == before and _bytes(args.saved) == saved_before
    assert receipt["snapshot_cleanup"]["status"] == "PASS"
    assert receipt["snapshot_cleanup"]["manager_close"] == "PASS"
    assert not Path(receipt["snapshot_cleanup"]["path"]).exists()
    assert receipt["input_checks"] == {
        "workspace": {"baseline": "COMPLETE", "status": "PASS"},
        "saved": {"baseline": "COMPLETE", "status": "PASS"},
    }


def test_missing_control_file_is_not_created_or_claimed_unchanged(checker, physical_tmp):
    workspace, saved, out = (physical_tmp / name for name in ("workspace", "saved", "out"))
    for path in (workspace, saved, out):
        path.mkdir()
    args = argparse.Namespace(
        workspace=workspace, saved=saved, out=out, expect=None, count=None,
        python=sys.executable, max_bytes="8388608", allow_history=False,
    )
    receipt = checker.check(args)
    assert receipt["failure_phase"] == "source_offline_check"
    assert receipt["failure_path"] == str(workspace / "writer.lock")
    assert receipt["inputs_unchanged"] is None and receipt["status"] == "FAIL"
    assert receipt["content_checks"] == "NOT_CHECKED"
    assert not (workspace / "writer.lock").exists()
    assert not list(out.iterdir())


@pytest.mark.skipif(sys.platform != "win32", reason="native Windows read-denying byte lock")
@pytest.mark.parametrize("member", ["workspace", "saved"])
def test_unreadable_noncontrol_member_is_not_skipped(checker, physical_tmp, member):
    import msvcrt

    args = _args(physical_tmp)
    target = getattr(args, member) / "locked-content.bin"
    target.write_bytes(b"must not be ignored")
    descriptor = os.open(target, os.O_RDWR | os.O_BINARY)
    try:
        msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
        receipt = checker.check(args)
        assert receipt["status"] == "FAIL"
        assert receipt["failure_phase"] == f"{member}_baseline"
        assert receipt["failure_path"] == str(target)
        assert "PermissionError" in receipt["first_error"]
        assert receipt["input_checks"][member] == {
            "baseline": "NOT_ESTABLISHED", "status": "NOT_CHECKED"
        }
        assert receipt["inputs_unchanged"] is None
        assert receipt["content_checks"] == "NOT_CHECKED"
        assert receipt["snapshot_cleanup"]["status"] == "NOT_CREATED"
    finally:
        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
        os.close(descriptor)
    assert target.read_bytes() == b"must not be ignored"


def test_cleanup_failure_is_reported_after_manager_close(checker, physical_tmp, monkeypatch):
    args = _args(physical_tmp)
    calls = []

    def refuse_cleanup(path):
        calls.append(path)
        # Native acquisition proves the snapshot manager released its Windows lock first.
        checker._require_offline_workspace(path / "workspace")
        raise PermissionError(13, "injected cleanup refusal", str(path))

    monkeypatch.setattr(checker.shutil, "rmtree", refuse_cleanup)
    receipt = checker.check(args)
    assert receipt["status"] == "FAIL" and receipt["inputs_unchanged"] is True
    cleanup = receipt["snapshot_cleanup"]
    assert cleanup["status"] == "FAIL" and cleanup["manager_close"] == "PASS"
    assert calls == [Path(cleanup["path"])] and calls[0].is_dir()
    assert any(
        row["phase"] == "snapshot_cleanup" and "injected cleanup refusal" in row["error"]
        for row in receipt["errors"]
    )
    assert receipt["failure_phase"] == "saved_file_check"  # Do not overwrite the first cause.
    monkeypatch.undo()
    shutil.rmtree(calls[0])


def test_manager_close_failure_preserves_snapshot_and_is_observable(
    checker, physical_tmp, monkeypatch
):
    args = _args(physical_tmp)
    opened = []
    original_init, original_close = JobManager.__init__, JobManager.close

    def remember(manager, *arguments, **keywords):
        original_init(manager, *arguments, **keywords)
        opened.append(manager)

    def refuse_close(manager):
        raise OSError("injected manager-close refusal")

    monkeypatch.setattr(JobManager, "__init__", remember)
    monkeypatch.setattr(JobManager, "close", refuse_close)
    try:
        receipt = checker.check(args)
        cleanup = receipt["snapshot_cleanup"]
        assert receipt["status"] == "FAIL"
        assert cleanup["status"] == "FAIL" and cleanup["manager_close"] == "FAIL"
        assert Path(cleanup["path"]).is_dir()
        assert any(row["phase"] == "snapshot_manager_close" for row in receipt["errors"])
        assert receipt["inputs_unchanged"] is True
    finally:
        for manager in opened:
            original_close(manager)
        monkeypatch.undo()


def test_snapshot_collision_never_removes_a_directory_it_did_not_create(checker, physical_tmp):
    args = _args(physical_tmp)
    collision = args.out / "workspace-snapshot"
    collision.mkdir()
    keep = collision / "keep.bin"
    keep.write_bytes(b"preexisting output")
    receipt = checker.check(args)
    assert receipt["status"] == "FAIL" and receipt["failure_phase"] == "snapshot_create"
    assert receipt["snapshot_cleanup"]["status"] == "NOT_CREATED"
    assert receipt["inputs_unchanged"] is True
    assert keep.read_bytes() == b"preexisting output"


def test_partial_copy_is_cleaned_without_opening_a_snapshot_manager(
    checker, physical_tmp, monkeypatch
):
    args = _args(physical_tmp)

    def partial_copy(source, target, *, symlinks):
        assert source == args.workspace and symlinks is True
        target.mkdir()
        (target / "partial.bin").write_bytes(b"partial copy")
        raise PermissionError(13, "injected partial-copy refusal", str(source / "member.bin"))

    monkeypatch.setattr(checker.shutil, "copytree", partial_copy)
    receipt = checker.check(args)
    assert receipt["status"] == "FAIL" and receipt["failure_phase"] == "snapshot_copy"
    assert receipt["content_checks"] == "NOT_CHECKED"
    assert receipt["snapshot_cleanup"]["status"] == "PASS"
    assert receipt["snapshot_cleanup"]["manager_close"] == "NOT_CREATED"
    assert not Path(receipt["snapshot_cleanup"]["path"]).exists()
    assert receipt["inputs_unchanged"] is True


@pytest.mark.parametrize("member", ["workspace", "saved"])
def test_a_changed_input_is_not_hidden_by_a_content_failure(
    checker, physical_tmp, monkeypatch, member
):
    args = _args(physical_tmp)
    original_regular = checker._regular

    def change_input(path):
        original_regular(path)
        (getattr(args, member) / "changed.bin").write_bytes(b"changed during checking")

    monkeypatch.setattr(checker, "_regular", change_input)
    receipt = checker.check(args)
    assert receipt["status"] == "FAIL" and receipt["inputs_unchanged"] is False
    assert receipt["input_checks"][member]["status"] == "FAIL"
    assert any(row["phase"] == f"{member}_reidentification" for row in receipt["errors"])
    assert receipt["snapshot_cleanup"]["status"] == "PASS"


def test_source_reopened_during_check_is_not_claimed_unchanged(
    checker, physical_tmp, monkeypatch
):
    args = _args(physical_tmp)
    original_regular, reopened = checker._regular, []

    def reopen_source(path):
        original_regular(path)
        reopened.append(JobManager(args.workspace))

    monkeypatch.setattr(checker, "_regular", reopen_source)
    try:
        receipt = checker.check(args)
        assert receipt["status"] == "FAIL" and receipt["inputs_unchanged"] is None
        assert receipt["input_checks"]["workspace"] == {
            "baseline": "COMPLETE", "status": "NOT_CHECKED"
        }
        assert receipt["input_checks"]["saved"]["status"] == "PASS"
        assert any(
            row["phase"] == "workspace_reidentification"
            and "source workspace has a live writer" in row["error"]
            for row in receipt["errors"]
        )
        assert receipt["snapshot_cleanup"]["status"] == "PASS"
        assert reopened[0].list_jobs() == []  # No automatic source-helper shutdown.
    finally:
        for manager in reopened:
            manager.close()


def _junction_or_skip(link: Path, target: Path) -> None:
    if os.name != "nt":
        link.symlink_to(target, target_is_directory=True)
        return
    import _winapi

    try:
        _winapi.CreateJunction(str(target), str(link))
    except (AttributeError, OSError) as error:
        pytest.skip(f"native Windows junction unavailable: {error}")


def test_workspace_member_junction_is_refused_before_reading_target(
    checker, physical_tmp, monkeypatch
):
    args = _args(physical_tmp)
    target = physical_tmp / "owned-empty-target"
    target.mkdir()
    link = args.workspace / "junction"
    _junction_or_skip(link, target)
    original_identity = checker._identity

    def no_link_reads(path):
        assert not path.is_relative_to(link), "junction target read before refusal"
        return original_identity(path)

    monkeypatch.setattr(checker, "_identity", no_link_reads)
    receipt = checker.check(args)
    assert receipt["status"] == "FAIL" and receipt["failure_phase"] == "workspace_baseline"
    assert receipt["failure_path"] == str(link)
    assert "linked workspace member refused" in receipt["first_error"]
    assert receipt["inputs_unchanged"] is None
    assert receipt["content_checks"] == "NOT_CHECKED"
    assert receipt["snapshot_cleanup"]["status"] == "NOT_CREATED"
    assert not list(target.iterdir())


@pytest.mark.parametrize(
    "kind", ["workspace_root", "workspace_ancestor", "saved_root", "saved_ancestor"]
)
def test_root_and_ancestor_junctions_are_refused_before_offline_probe(
    checker, physical_tmp, monkeypatch, kind
):
    args = _args(physical_tmp)
    member, location = kind.split("_")
    target = physical_tmp if location == "ancestor" else getattr(args, member)
    link = physical_tmp / "root-junction"
    _junction_or_skip(link, target)
    setattr(args, member, link / member if location == "ancestor" else link)

    def no_probe(_):
        raise AssertionError("linked root/ancestor accessed before refusal")

    monkeypatch.setattr(checker, "_require_offline_workspace", no_probe)
    receipt = checker.check(args)
    assert receipt["status"] == "FAIL" and receipt["failure_phase"] == "input_folders"
    assert receipt["failure_path"] == str(link)
    assert "linked folder path refused" in receipt["first_error"]
    assert receipt["inputs_unchanged"] is None and receipt["content_checks"] == "NOT_CHECKED"
    assert receipt["snapshot_cleanup"]["status"] == "NOT_CREATED"


def test_offline_probe_itself_refuses_a_junction_before_opening_control_file(
    checker, physical_tmp, monkeypatch
):
    args = _args(physical_tmp)
    link = physical_tmp / "workspace-junction"
    _junction_or_skip(link, args.workspace)
    original_open = os.open

    def no_link_open(path, *arguments, **keywords):
        assert not Path(path).is_relative_to(link), "junction control file was opened"
        return original_open(path, *arguments, **keywords)

    monkeypatch.setattr(os, "open", no_link_open)
    with pytest.raises(checker.CheckFailed, match="linked folder path refused"):
        checker._require_offline_workspace(link)


@pytest.mark.parametrize("kind", ["directory", "junction"])
def test_new_nonregular_saved_member_is_not_claimed_unchanged(
    checker, physical_tmp, monkeypatch, kind
):
    args = _args(physical_tmp)
    target = physical_tmp / "owned-target"
    target.mkdir()
    (target / "keep.bin").write_bytes(b"owned target must not be read")
    added = args.saved / "new-member"
    original_regular, original_identity = checker._regular, checker._identity

    def add_nonregular(path):
        original_regular(path)
        if kind == "junction":
            _junction_or_skip(added, target)
        else:
            added.mkdir()

    def no_added_reads(path):
        assert not path.is_relative_to(added), "nonregular saved member was read"
        return original_identity(path)

    monkeypatch.setattr(checker, "_regular", add_nonregular)
    monkeypatch.setattr(checker, "_identity", no_added_reads)
    receipt = checker.check(args)
    assert receipt["status"] == "FAIL" and receipt["inputs_unchanged"] is None
    assert receipt["input_checks"]["saved"] == {
        "baseline": "COMPLETE", "status": "NOT_CHECKED"
    }
    assert any(
        row["phase"] == "saved_reidentification" and row["path"] == str(added)
        for row in receipt["errors"]
    )
    assert (target / "keep.bin").read_bytes() == b"owned target must not be read"


def test_saved_root_replaced_by_junction_is_refused_during_reidentification(
    checker, physical_tmp, monkeypatch
):
    args = _args(physical_tmp)
    original_regular, original_identity = checker._regular, checker._identity
    saved = args.saved
    moved = physical_tmp / "owned-moved-saved"

    def replace_saved(path):
        original_regular(path)
        saved.rename(moved)
        _junction_or_skip(saved, moved)
        raise checker.CheckFailed("injected content failure after saved replacement", path=path)

    def no_replaced_reads(path):
        assert not moved.exists() or not path.is_relative_to(saved), "saved junction was read"
        return original_identity(path)

    monkeypatch.setattr(checker, "_regular", replace_saved)
    monkeypatch.setattr(checker, "_identity", no_replaced_reads)
    receipt = checker.check(args)
    assert receipt["status"] == "FAIL" and receipt["inputs_unchanged"] is None
    assert receipt["input_checks"]["saved"]["status"] == "NOT_CHECKED"
    assert any(
        row["phase"] == "saved_reidentification" and row["path"] == str(saved)
        for row in receipt["errors"]
    )


def test_cli_refuses_output_ancestor_junction_before_creating_any_output(
    checker, physical_tmp, capsys
):
    args = _args(physical_tmp)
    alias = physical_tmp / "output-alias"
    _junction_or_skip(alias, args.workspace)
    before = _bytes(args.workspace)
    code = checker.main([
        "--python", sys.executable, "--workspace", str(args.workspace),
        "--saved", str(args.saved), "--out", str(alias / "nested" / "output"),
    ])
    assert code == 2
    assert "linked folder path refused" in capsys.readouterr().err
    assert _bytes(args.workspace) == before
    assert not (args.workspace / "nested").exists()
