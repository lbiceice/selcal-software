from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest
from test_cli import cli
from test_workflow import CAP
from test_workflow_export import bundle, make_record, rehash


@pytest.mark.parametrize("form", ["csv", "npz"])
@pytest.mark.parametrize(
    "mode", ["complete", "null_bind", "observed_statistic_scan", "replicate_execution"]
)
def test_cli_export_and_verify_export_status_and_one_json(tmp_path, form, mode):
    record, result, _ = make_record(tmp_path, form=form, mode=mode)
    output = tmp_path / "导出 directory"
    exported = cli(tmp_path, "export", record, output, "--max-bytes", CAP)
    expected = 0 if mode == "complete" else 7
    assert exported.returncode == expected, exported.stderr
    assert len(exported.stdout.splitlines()) == 1
    terminal = json.loads(exported.stdout)
    assert terminal["data"]["status"] == result.status.value
    if expected == 7:
        assert "written" in exported.stderr and "NOT_EVALUABLE" in exported.stderr
    checked = cli(tmp_path, "verify-export", output, "--max-bytes", CAP)
    assert checked.returncode == expected, checked.stderr
    assert len(checked.stdout.splitlines()) == 1
    assert json.loads(checked.stdout)["data"] == terminal["data"]
    assert "Traceback" not in exported.stderr + checked.stderr
    assert str(tmp_path) not in exported.stderr + checked.stderr


def test_actual_concurrent_exports_have_one_winner(tmp_path):
    record, _, _ = make_record(tmp_path)
    output = tmp_path / "one target"
    command = [
        sys.executable,
        "-B",
        "-m",
        "selcal",
        "export",
        str(record),
        str(output),
        "--max-bytes",
        str(CAP),
    ]
    children = [
        subprocess.Popen(
            command, cwd=tmp_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                errors="replace"
        )
        for _ in range(2)
    ]
    responses = [child.communicate(timeout=30) for child in children]
    assert sorted(child.returncode for child in children) == [0, 4], responses
    for out, err in responses:
        assert len(out.splitlines()) == 1 and "Traceback" not in err
    assert cli(tmp_path, "verify-export", output, "--max-bytes", CAP).returncode == 0


@pytest.mark.parametrize(
    "mode", ["malformed_utf8", "wrong_type", "huge_number", "lone_surrogate", "invalid_request"]
)
def test_corrupt_content_exits_four_not_argument_error(tmp_path, mode):
    _, output, _, _, _ = bundle(tmp_path)
    manifest = output / "manifest.json"
    if mode == "malformed_utf8":
        manifest.write_bytes(b"\xff")
    elif mode == "wrong_type":
        manifest.write_bytes(b"false")
    elif mode == "huge_number":
        manifest.write_bytes(b'{"x":' + b"9" * 10000 + b"}")
    elif mode == "lone_surrogate":
        metadata = json.loads((output / "metadata.json").read_bytes())
        metadata["software"]["numpy_version"] = "\ud800"
        rehash(output, "metadata.json", json.dumps(metadata).encode())
    else:
        rehash(output, "request.json", b"{}")
    checked = cli(tmp_path, "verify-export", output, "--max-bytes", CAP)
    assert checked.returncode == 4, checked.stderr
    assert json.loads(checked.stdout)["outcome"] == "FAIL"
    assert len(checked.stderr) < 1000
    assert str(tmp_path) not in checked.stderr and "Traceback" not in checked.stderr


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX FIFO test")
def test_fifo_is_rejected_without_blocking(tmp_path):
    _, output, _, _, _ = bundle(tmp_path)
    member = output / "summary.csv"
    member.unlink()
    os.mkfifo(member)
    checked = cli(tmp_path, "verify-export", output, "--max-bytes", CAP)
    assert checked.returncode == 4


@pytest.mark.parametrize("command", ["export", "verify-export"])
def test_invalid_cli_limit_stays_argument_error(tmp_path, command):
    args = ["record", "directory"] if command == "export" else ["directory"]
    checked = cli(tmp_path, command, *args, "--max-bytes", 0)
    assert checked.returncode == 2
    assert "max-bytes must be a positive integer" in checked.stderr


def test_export_io_failure_explains_retained_directory(tmp_path, monkeypatch, capsys):
    from selcal import cli as command

    record, _, _ = make_record(tmp_path)
    output = tmp_path / "bundle"
    from pathlib import Path

    original = Path.open

    def fail(path, mode="r", *args, **kwargs):
        if path == output / "summary.csv" and mode == "xb":
            raise OSError("private /path/should/not/appear")
        return original(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail)
    assert command.main(["export", str(record), str(output), "--max-bytes", str(CAP)]) == 4
    captured = capsys.readouterr()
    assert "verify-export" in captured.err and "fresh" in captured.err
    assert "/path/should/not/appear" not in captured.err
    assert output.is_dir()
