"""Acceptance tools from the R14 Windows return (W14-02 saved-file binding, W14-03 evidence).

The saved-download checker must bind each file to a real UI job through the product's own
download check, separate its claims (bytes, job binding, replay, export, HTML), and fail on
missing or contradictory job metadata. The research-case flow must keep the step, partial output
and input identities when a step times out or fails. Workspaces here come from real UI jobs.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from test_saved_downloads_offline import checker as _saved_checker
from test_ui_artifacts import generated
from test_ui_jobs import physical_tmp as _physical_tmp
from test_ui_process import wait_job

physical_tmp = _physical_tmp
checker = _saved_checker
ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "workflow"
CHECKER = ROOT / "scripts" / "check_saved_downloads.py"
FLOW = ROOT / "scripts" / "research_case_flow.py"
FILES = {"record": "result.sqlite", "report": "report.html", "bundle": "evidence.zip"}
CASE = {
    "name": "example",
    "selected_candidate": 2,
    "exceedance_count": 2,
    "replicates": 199,
    "reject_null": True,
}


def _job(manager, *, reports=1):
    config = {
        "config_text": (EXAMPLE / "pearson.json").read_text(encoding="utf-8"),
        "max_bytes": "8388608",
        "allow_unattainable": False,
    }
    job_id = manager.admit(config)["id"]
    raw = (EXAMPLE / "series.csv").read_bytes()
    manager.upload(job_id, io.BytesIO(raw), len(raw))
    manager.start(job_id, "run")
    wait_job(manager, job_id)
    for _ in range(reports):
        generated(manager, job_id, "report")
    generated(manager, job_id, "export")
    return job_id


def _save_current(workspace: Path, job_id: str, saved: Path) -> dict[str, Path]:
    saved.mkdir(exist_ok=True)
    result = {}
    for kind, base in FILES.items():
        op = json.loads((workspace / "jobs" / job_id / f"{kind}.json").read_bytes())["operation_id"]
        target = saved / f"selcal-{job_id[:8]}-{op[:8]}-{base}"
        shutil.copyfile(workspace / "jobs" / job_id / "operations" / op / base, target)
        result[kind] = target
    return result


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    from selcal.ui_jobs import JobManager

    base = tmp_path_factory.mktemp("acceptance").resolve()
    manager = JobManager(base / "workspace")
    try:
        job_id = _job(manager, reports=2)
    finally:
        # R17 item 5: release the workspace and any child even when building the job fails.
        manager.close()
    expect = base / "expect.json"
    case = dict(
        CASE, raw_input_sha256=hashlib.sha256((EXAMPLE / "series.csv").read_bytes()).hexdigest()
    )
    expect.write_text(
        json.dumps({"schema": "selcal.research-cases.v1", "cases": [case]}), encoding="utf-8"
    )
    return base, job_id, expect


def _copy(built, tmp: Path):
    base, job_id, expect = built
    workspace = tmp / "workspace"
    shutil.copytree(base / "workspace", workspace)
    return workspace, job_id, expect


def _check(tmp: Path, workspace: Path, saved: Path, *extra: str) -> tuple[int, dict]:
    out = tmp / f"out-{len(list(tmp.glob('out-*')))}"
    done = subprocess.run(
        [
            sys.executable,
            "-I",
            str(CHECKER),
            "--python",
            sys.executable,
            "--workspace",
            str(workspace),
            "--saved",
            str(saved),
            "--out",
            str(out),
            *extra,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=300,
    )
    receipt = json.loads((out / "receipt.json").read_text(encoding="utf-8"))
    # This also runs for the live-source refusal under a Chinese workspace path.
    assert done.stdout.isascii()
    summary = json.loads(done.stdout)
    assert (summary["status"], summary["first_error"]) == (
        receipt["status"], receipt["first_error"]
    )
    return done.returncode, receipt


def _tree(folder: Path) -> dict:
    return {
        p.relative_to(folder).as_posix(): p.read_bytes() for p in folder.rglob("*") if p.is_file()
    }


def test_current_downloads_pass_every_separate_check_and_inputs_stay_unchanged(built, physical_tmp):
    workspace, job_id, expect = _copy(built, physical_tmp)
    _save_current(workspace, job_id, physical_tmp / "saved")
    before = _tree(workspace)
    code, receipt = _check(
        physical_tmp, workspace, physical_tmp / "saved", "--expect", str(expect), "--count", "3"
    )
    assert code == 0 and receipt["status"] == "PASS", receipt
    assert _tree(workspace) == before and receipt["inputs_unchanged"] is True
    assert receipt["source_access"] == "OFFLINE_AVAILABLE"
    assert receipt["content_checks"] == "PASS"
    assert receipt["snapshot_cleanup"]["status"] == "PASS"
    assert receipt["snapshot_cleanup"]["manager_close"] == "PASS"
    assert not Path(receipt["snapshot_cleanup"]["path"]).exists()
    assert receipt["input_checks"] == {
        "workspace": {"baseline": "COMPLETE", "status": "PASS"},
        "saved": {"baseline": "COMPLETE", "status": "PASS"},
    }
    rows = {row["kind"]: row for row in receipt["files"]}
    assert all(
        row["file_equality"] == "PASS"
        and row["job_binding"] == "PASS"
        and row["reference_scope"] == "current"
        for row in rows.values()
    )
    assert rows["record"]["record_replay"] == "PASS"
    assert rows["record"]["expected_result"] == "PASS" and rows["record"]["case"] == "example"
    assert rows["bundle"]["export_content"] == "PASS"
    assert rows["report"]["html_content"] == "PASS"
    assert receipt["browser_save_observed"] == "NOT_OBSERVED_BY_THIS_TOOL"


@pytest.mark.parametrize("workspace_name", ["workspace", "运行中 工作区"])
def test_live_source_is_refused_by_cli_without_claiming_inputs_unchanged(
    physical_tmp, workspace_name
):
    from selcal.ui_jobs import JobManager

    workspace, saved = physical_tmp / workspace_name, physical_tmp / "saved"
    saved.mkdir()
    (saved / "unknown.bin").write_bytes(b"not a SelCal artifact")
    manager = JobManager(workspace)
    marker = workspace / "writer.lock"
    identity = marker.stat()
    try:
        code, receipt = _check(physical_tmp, workspace, saved)
        assert code == 1 and receipt["status"] == "FAIL", receipt
        assert receipt["failure_phase"] == "source_offline_check"
        assert receipt["failure_path"] == str(marker)
        assert "Stop its UI helper normally" in receipt["first_error"]
        assert receipt["content_checks"] == "NOT_CHECKED" and not receipt["files"]
        assert receipt["inputs_unchanged"] is None
        assert all(
            row == {"baseline": "NOT_ESTABLISHED", "status": "NOT_CHECKED"}
            for row in receipt["input_checks"].values()
        )
        assert receipt["snapshot_cleanup"]["status"] == "NOT_CREATED"
        assert manager.list_jobs() == []  # The checker did not stop the source manager.
    finally:
        manager.close()
    after = marker.stat()
    assert (identity.st_ino, identity.st_size, identity.st_mtime_ns) == (
        after.st_ino, after.st_size, after.st_mtime_ns
    )


@pytest.mark.skipif(sys.platform != "win32", reason="native Windows open-handle delete refusal")
def test_native_cleanup_error_fails_otherwise_valid_downloads(
    checker, built, physical_tmp, monkeypatch
):
    from selcal.ui_jobs import JobManager

    workspace, job_id, _ = _copy(built, physical_tmp)
    saved = physical_tmp / "saved"
    _save_current(workspace, job_id, saved)
    before, saved_before = _tree(workspace), _tree(saved)
    out = physical_tmp / "cleanup-out"
    out.mkdir()
    args = argparse.Namespace(
        workspace=workspace, saved=saved, out=out, expect=None, count=3,
        python=sys.executable, max_bytes="8388608", allow_history=False,
    )
    original_close, held = JobManager.close, []

    def close_with_external_reader(manager):
        original_close(manager)
        # A separate real Windows handle denies deletion after the writer was released.
        held.append(os.open(manager.workspace / "writer.lock", os.O_RDONLY | os.O_BINARY))

    monkeypatch.setattr(JobManager, "close", close_with_external_reader)
    try:
        receipt = checker.check(args)
        assert receipt["content_checks"] == "PASS"
        assert all(row["status"] == "PASS" for row in receipt["files"])
        assert receipt["status"] == "FAIL" and receipt["failure_phase"] == "snapshot_cleanup"
        assert "PermissionError" in receipt["first_error"]
        assert receipt["snapshot_cleanup"]["status"] == "FAIL"
        assert receipt["snapshot_cleanup"]["manager_close"] == "PASS"
        assert Path(receipt["snapshot_cleanup"]["path"]).is_dir()
        assert receipt["inputs_unchanged"] is True
        assert _tree(workspace) == before and _tree(saved) == saved_before
    finally:
        for descriptor in held:
            os.close(descriptor)
        monkeypatch.undo()
        snapshot = out / "workspace-snapshot"
        if snapshot.exists():
            shutil.rmtree(snapshot)


def test_browser_duplicate_suffix_is_accepted_and_reported(built, physical_tmp):
    workspace, job_id, _ = _copy(built, physical_tmp)
    saved = _save_current(workspace, job_id, physical_tmp / "saved")
    renamed = saved["bundle"].with_name(saved["bundle"].stem + " (1).zip")
    saved["bundle"].rename(renamed)
    code, receipt = _check(physical_tmp, workspace, physical_tmp / "saved")
    assert code == 0, receipt
    row = next(row for row in receipt["files"] if row["saved"] == renamed.name)
    assert row["renamed_by_browser"] is True and row["export_content"] == "PASS"


def _tamper_request(workspace, job_id, update_digest):
    """Change the saved plan (alpha 0.05 -> 0.99) in the real request.json."""
    job = workspace / "jobs" / job_id
    config = json.loads((job / "request.json").read_bytes())
    config["plan"]["alpha"] = 0.99
    raw = json.dumps(config).encode("utf-8")
    (job / "request.json").write_bytes(raw)
    if update_digest:
        metadata = json.loads((job / "metadata.json").read_bytes())
        metadata["config_sha256"] = hashlib.sha256(raw).hexdigest()
        (job / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


@pytest.mark.parametrize(
    "damage",
    [
        "missing_metadata",
        "missing_reference",
        "request_changed",
        "request_and_digest_changed",
        "record_ref_to_other_operation",
        "input_changed",
    ],
)
def test_contradictory_or_missing_job_metadata_fails(built, physical_tmp, damage):
    workspace, job_id, _ = _copy(built, physical_tmp)
    _save_current(workspace, job_id, physical_tmp / "saved")
    job = workspace / "jobs" / job_id
    if damage == "missing_metadata":
        (job / "metadata.json").unlink()
    elif damage == "missing_reference":
        (job / "bundle.json").unlink()
    elif damage in {"request_changed", "request_and_digest_changed"}:
        _tamper_request(workspace, job_id, damage == "request_and_digest_changed")
    elif damage == "record_ref_to_other_operation":
        ref = json.loads((job / "record.json").read_bytes())
        ref["operation_id"] = json.loads((job / "report.json").read_bytes())["operation_id"]
        (job / "record.json").write_text(json.dumps(ref), encoding="utf-8")
    else:
        (job / "input.csv").write_bytes((job / "input.csv").read_bytes() + b"\n")
    code, receipt = _check(physical_tmp, workspace, physical_tmp / "saved")
    assert code == 1 and receipt["status"] == "FAIL", receipt
    assert receipt["first_error"]


def test_non_html_report_with_matching_original_fails(built, physical_tmp):
    """The R14 probe: a text 'report' equal to a doctored original no longer passes."""
    workspace, job_id, _ = _copy(built, physical_tmp)
    saved = _save_current(workspace, job_id, physical_tmp / "saved")
    job = workspace / "jobs" / job_id
    op = json.loads((job / "report.json").read_bytes())["operation_id"]
    for path in (saved["report"], job / "operations" / op / "report.html"):
        path.write_bytes(b"plain text, not a report\n")
    code, receipt = _check(physical_tmp, workspace, physical_tmp / "saved")
    assert code == 1
    row = next(row for row in receipt["files"] if row.get("kind") == "report")
    assert row["file_equality"] == "PASS" and row["job_binding"] == "FAIL"


def test_valid_html_of_another_record_fails_semantic_check(built, physical_tmp):
    """References updated consistently still fail when the HTML is not this record's report."""
    workspace, job_id, _ = _copy(built, physical_tmp)
    saved = _save_current(workspace, job_id, physical_tmp / "saved")
    job = workspace / "jobs" / job_id
    ref = json.loads((job / "report.json").read_bytes())
    fake = saved["report"].read_bytes().replace(b"</body>", b"<p>edited</p></body>", 1)
    assert fake != saved["report"].read_bytes()
    for path in (saved["report"], job / "operations" / ref["operation_id"] / "report.html"):
        path.write_bytes(fake)
    ref.update(bytes=len(fake), sha256=hashlib.sha256(fake).hexdigest())
    (job / "report.json").write_text(json.dumps(ref), encoding="utf-8")
    code, receipt = _check(physical_tmp, workspace, physical_tmp / "saved")
    assert code == 1
    row = next(row for row in receipt["files"] if row.get("kind") == "report")
    assert row["job_binding"] == "PASS" and row["html_content"] == "FAIL", row


@pytest.mark.parametrize("damage", ["empty", "truncated", "foreign_name", "corrupt_zip"])
def test_empty_truncated_or_unknown_files_fail(built, physical_tmp, damage):
    workspace, job_id, _ = _copy(built, physical_tmp)
    saved = _save_current(workspace, job_id, physical_tmp / "saved")
    if damage == "empty":
        saved["bundle"].write_bytes(b"")
    elif damage == "truncated":
        saved["record"].write_bytes(saved["record"].read_bytes()[:-1])
    elif damage == "corrupt_zip":
        # R19 v3 review: a partial download raised BadZipFile and crashed the checker.
        saved["bundle"].write_bytes(b"PK\x03\x04" + bytes(64))
    else:
        (physical_tmp / "saved" / "evidence (4).zip").write_bytes(b"PK")
    code, receipt = _check(physical_tmp, workspace, physical_tmp / "saved")
    assert code == 1 and receipt["status"] == "FAIL"
    if damage == "corrupt_zip":
        # Caught by the byte-equality check first; the archive guard below covers the case
        # where the bytes pass but the archive still cannot be opened.
        assert receipt["first_error"]


def test_unreadable_archive_is_a_check_failure_not_a_crash(physical_tmp):
    """R19 v3 review: zipfile.BadZipFile escaped the checker's except clause."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("check_saved_downloads", CHECKER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    archive = physical_tmp / "partial-evidence.zip"
    archive.write_bytes(b"PK\x03\x04" + bytes(64))
    with pytest.raises(module.CheckFailed, match="not a readable ZIP archive"):
        module._unpack(archive, physical_tmp / "unpacked")
    assert not (physical_tmp / "unpacked").exists()


def test_history_file_needs_explicit_mode_and_is_labelled(built, physical_tmp):
    workspace, job_id, _ = _copy(built, physical_tmp)
    job = workspace / "jobs" / job_id
    current = json.loads((job / "report.json").read_bytes())["operation_id"]
    older = [
        p.name
        for p in (job / "operations").iterdir()
        if (p / "report.html").is_file() and p.name != current
    ]
    assert len(older) == 1
    saved = physical_tmp / "saved"
    saved.mkdir()
    shutil.copyfile(
        job / "operations" / older[0] / "report.html",
        saved / f"selcal-{job_id[:8]}-{older[0][:8]}-report.html",
    )
    code, receipt = _check(physical_tmp, workspace, saved)
    assert code == 1 and "current" in receipt["first_error"]
    code, receipt = _check(physical_tmp, workspace, saved, "--allow-history")
    assert code == 0, receipt
    (row,) = receipt["files"]
    assert row["reference_scope"] == "history"
    assert row["job_binding"] == "NOT_APPLICABLE_HISTORY" and row["html_content"] == "PASS"


# -- research case flow (W14-03) -------------------------------------------------------------


def _flow(tmp: Path, python: str, *extra: str, data: Path = EXAMPLE) -> tuple[int, dict, Path]:
    expect = tmp / "cases.json"
    expect.write_text(
        json.dumps(
            {
                "schema": "selcal.research-cases.v1",
                "cases": [dict(CASE, input="series.csv", config="pearson.json")],
            }
        ),
        encoding="utf-8",
    )
    out = tmp / "flow"
    done = subprocess.run(
        [
            sys.executable,
            "-I",
            str(FLOW),
            "--python",
            python,
            "--data",
            str(data),
            "--expect",
            str(expect),
            "--out",
            str(out),
            *extra,
        ],
        capture_output=True,
        timeout=600,
    )
    receipt = json.loads((out / "receipt.json").read_text(encoding="utf-8"))
    # The console summary is ASCII-escaped JSON whatever the console code page, and says what
    # the receipt says (R17 item 8).
    summary = json.loads(done.stdout.decode("ascii"))
    assert (summary["status"], summary["first_error"]) == (receipt["status"],
                                                            receipt["first_error"])
    return done.returncode, receipt, out


def test_research_flow_passes_every_step_on_the_example(physical_tmp):
    code, receipt, _ = _flow(physical_tmp, sys.executable)
    assert code == 0 and receipt["status"] == "PASS", receipt["first_error"]
    assert len(receipt["steps"]) == 8 and all(s["status"] == "PASS" for s in receipt["steps"])
    assert receipt["cases"]["example"]["observed"] == receipt["cases"]["example"]["expected"]
    assert {v["state"] for v in receipt["inputs_after"].values()} == {"unchanged"}


def _fake_python(tmp: Path, body: str) -> str:
    script = tmp / "fake_python.py"
    script.write_text(body, encoding="utf-8")
    launcher = tmp / "fake-python"
    launcher.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', encoding="utf-8")
    launcher.chmod(0o755)
    return str(launcher)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX launcher fixture")
def test_timeout_keeps_step_partial_output_and_input_identities(physical_tmp):
    python = _fake_python(
        physical_tmp,
        (
            "import sys, time\nsys.stdout.write('partial-out'); sys.stdout.flush()\n"
            "sys.stderr.write('partial-err'); sys.stderr.flush()\ntime.sleep(30)\n"
        ),
    )
    code, receipt, out = _flow(physical_tmp, python, "--timeout", "8")
    assert code == 1 and receipt["status"] == "FAIL"
    (step,) = receipt["steps"]
    assert step["status"] == "TIMEOUT" and step["child"] == "killed_and_reaped"
    assert (out / step["stdout"]).read_bytes() == b"partial-out"
    assert (out / step["stderr"]).read_bytes() == b"partial-err"
    assert "timed out" in receipt["first_error"]
    assert {v["state"] for v in receipt["inputs_after"].values()} == {"unchanged"}


def test_unstartable_python_is_recorded(physical_tmp):
    code, receipt, _ = _flow(physical_tmp, str(physical_tmp / "missing-python"))
    assert code == 1 and receipt["steps"][0]["status"] == "CANNOT_START"


def test_unstartable_python_under_a_chinese_path_is_reported_in_ascii(physical_tmp):
    code, receipt, _ = _flow(physical_tmp, str(physical_tmp / "缺失 目录" / "python"))
    assert code == 1 and receipt["steps"][0]["status"] == "CANNOT_START"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX launcher fixture")
def test_malformed_output_and_changed_input_are_separate_failures(physical_tmp):
    data = physical_tmp / "data"
    shutil.copytree(EXAMPLE, data)
    python = _fake_python(
        physical_tmp,
        (
            "import pathlib, sys\n"
            f"pathlib.Path({str(data / 'series.csv')!r}).write_bytes(b'changed')\n"
            "print('not json')\n"
        ),
    )
    code, receipt, _ = _flow(physical_tmp, python, data=data)
    assert code == 1
    assert receipt["steps"][0]["status"] == "MALFORMED_OUTPUT"
    assert "not CLI JSON" in receipt["first_error"]
    assert receipt["inputs_after"]["series.csv"]["state"] == "changed"
    assert receipt["inputs_after"]["pearson.json"]["state"] == "unchanged"


def test_research_flow_with_relative_paths_under_a_deep_folder(physical_tmp):
    """R17 Windows item 1 class: children start in a short folder; relative paths keep meaning."""
    caller = physical_tmp / "深 目录"
    while len(str(caller)) < 200:
        caller = caller / "layer-0123456789"
    shutil.copytree(EXAMPLE, caller / "data")
    (caller / "selcal.py").write_text("raise SystemExit('stray selcal.py')\n", encoding="utf-8")
    (caller / "cases.json").write_text(
        json.dumps(
            {
                "schema": "selcal.research-cases.v1",
                "cases": [dict(CASE, input="series.csv", config="pearson.json")],
            }
        ),
        encoding="utf-8",
    )
    done = subprocess.run(
        [sys.executable, "-I", str(FLOW), "--python", sys.executable, "--data", "data",
         "--expect", "cases.json", "--out", "flow 结果"],
        cwd=caller,
        capture_output=True,
        timeout=600,
    )
    receipt = json.loads((caller / "flow 结果" / "receipt.json").read_text(encoding="utf-8"))
    assert done.returncode == 0 and receipt["status"] == "PASS", receipt["first_error"]
    assert len(receipt["launch_cwd"]) < len(str(caller))
    assert not Path(receipt["launch_cwd"]).exists()
