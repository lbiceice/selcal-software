"""Directory-link guards must refuse Windows junctions, emulated on every platform.

R10v3 Windows run (2026-10-03): with real junctions, six tests showed that UI download/verify
and export path guards accepted a junction, because ``lstat`` reports a junction as a directory
(S_ISDIR true, S_ISLNK false). Native junction tests need Windows; here ``lstat`` of one chosen
path returns what Windows reports for a junction (directory mode, FILE_ATTRIBUTE_REPARSE_POINT
and IO_REPARSE_TAG_MOUNT_POINT), so each guard's decision is exercised on every platform.
"""

from __future__ import annotations

import json
import os
import pathlib
import stat
from pathlib import Path

import pytest

from selcal import ui_jobs, workflow_export
from selcal.workflow import run_files
from selcal.workflow_store import _is_link

ROOT = Path(__file__).resolve().parents[1]
CAP = 8 * 1_048_576
DIRECTORY_ATTRIBUTE, REPARSE_ATTRIBUTE, MOUNT_POINT_TAG = 0x10, 0x400, 0xA0000003


class _JunctionStat:
    """A real directory's lstat result as Windows reports it for a directory junction."""

    def __init__(self, real: os.stat_result) -> None:
        self._real = real
        self.st_file_attributes = DIRECTORY_ATTRIBUTE | REPARSE_ATTRIBUTE
        self.st_reparse_tag = MOUNT_POINT_TAG

    def __getattr__(self, name: str) -> object:
        return getattr(self._real, name)


@pytest.fixture
def as_junction(monkeypatch: pytest.MonkeyPatch):
    """Make ``Path.lstat`` report the given directories as junctions."""
    junctions: set[Path] = set()
    original = pathlib.Path.lstat

    def lstat(self: Path) -> object:
        result = original(self)
        return _JunctionStat(result) if Path(os.path.abspath(self)) in junctions else result

    monkeypatch.setattr(pathlib.Path, "lstat", lstat)
    return lambda path: junctions.add(Path(os.path.abspath(path)))


def _record(tmp_path: Path) -> Path:
    config = tmp_path / "pearson.json"
    config.write_bytes((ROOT / "examples/workflow/pearson.json").read_bytes())
    record = tmp_path / "run.sqlite"
    run_files(ROOT / "examples/workflow/series.csv", config, record, max_bytes=CAP)
    return record


def test_emulated_junction_is_a_directory_by_mode_but_a_link_by_tag(tmp_path, as_junction):
    folder = tmp_path / "junction"
    folder.mkdir()
    as_junction(folder)
    info = folder.lstat()
    assert stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode)
    assert _is_link(info)


def test_export_refuses_a_junction_parent(tmp_path, as_junction):
    record = _record(tmp_path)
    parent = tmp_path / "linked-parent"
    parent.mkdir()
    as_junction(parent)
    with pytest.raises(workflow_export.ExportError) as error:
        workflow_export.export_record(record, parent / "bundle", max_bytes=CAP)
    assert error.value.code == "unsafe_parent"
    assert not (parent / "bundle").exists()


def test_verify_export_refuses_a_junction_root(tmp_path, as_junction):
    record = _record(tmp_path)
    bundle = tmp_path / "bundle"
    workflow_export.export_record(record, bundle, max_bytes=CAP)
    assert workflow_export.verify_export(bundle, max_bytes=CAP)
    as_junction(bundle)
    with pytest.raises(workflow_export.ExportError):
        workflow_export.verify_export(bundle, max_bytes=CAP)


def test_ui_directory_and_member_guards_refuse_junctions(tmp_path, as_junction):
    workspace = tmp_path / "workspace"
    (workspace / "jobs").mkdir(parents=True)
    member = workspace / "jobs" / "request.json"
    member.write_text(json.dumps({}), encoding="utf-8")
    ui_jobs._real_directory(workspace / "jobs")
    ui_jobs._regular(member)
    as_junction(workspace / "jobs")
    with pytest.raises(ui_jobs.UIError):
        ui_jobs._real_directory(workspace / "jobs")
    with pytest.raises(ui_jobs.UIError):
        ui_jobs._regular(member)
