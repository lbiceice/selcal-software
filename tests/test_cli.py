from __future__ import annotations

import errno
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "examples" / "workflow" / "series.csv"
CONFIG = ROOT / "examples" / "workflow" / "pearson.json"


def cli(tmp_path, *args):
    return subprocess.run(
        [sys.executable, "-B", "-m", "selcal", *map(str, args)],
        cwd=tmp_path,
        text=True,
        errors="replace",
        capture_output=True,
        timeout=30,
        check=False,
    )


def test_actual_cli_run_verify_report_are_fresh_processes(tmp_path):
    checked = cli(tmp_path, "validate", INPUT, CONFIG)
    assert checked.returncode == 0, checked.stderr
    assert len(checked.stdout.splitlines()) == 1
    assert json.loads(checked.stdout)["data"]["sample_count"] == 100
    record = tmp_path / "run.sqlite"
    run = cli(tmp_path, "run", INPUT, CONFIG, record, "--max-bytes", 1048576)
    assert run.returncode == 0, run.stderr
    payload = json.loads(run.stdout)
    assert payload["data"]["p_value"] == 0.015
    assert payload["data"]["planned_replicates"] == 199
    assert payload["data"]["selected_candidate"] == 2
    before = record.read_bytes()
    for flags, expected in (([], "NOT_PERFORMED"), (["--replay"], "MATCH")):
        checked = cli(tmp_path, "verify", record, "--max-bytes", 1048576, *flags)
        assert checked.returncode == 0, checked.stderr
        assert json.loads(checked.stdout)["data"]["replay"] == expected
    report = tmp_path / "report.html"
    rendered = cli(tmp_path, "report", record, report, "--max-bytes", 1048576)
    assert rendered.returncode == 0, rendered.stderr
    assert report.is_file()
    assert record.read_bytes() == before


def test_cli_ne_exit_is_separate_and_record_is_preserved(tmp_path):
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    # One block of all 100 samples cannot bind (min_shift = 51 once did this; restricted
    # shifts are now refused at preflight with exit 2 as a non-group null, ALG-02).
    cfg["plan"]["null_name"] = "block_shuffle_v2"
    cfg["plan"]["null_params"] = {"block_length": 100}
    config = tmp_path / "ne.json"
    config.write_text(json.dumps(cfg), newline="\n", encoding="utf-8")
    record = tmp_path / "ne.sqlite"
    result = cli(tmp_path, "run", INPUT, config, record, "--max-bytes", 1048576)
    assert result.returncode == 7, result.stderr
    payload = json.loads(result.stdout)
    assert payload["outcome"] == "NOT_EVALUABLE"
    assert payload["data"]["p_value"] is None
    assert record.is_file()
    checked = cli(tmp_path, "verify", record, "--max-bytes", 1048576, "--replay")
    assert checked.returncode == 7
    assert json.loads(checked.stdout)["data"]["replay"] == "MATCH"


@pytest.mark.parametrize("argument", ["--help", "--version"])
def test_cli_informational_commands(tmp_path, argument):
    result = cli(tmp_path, argument)
    assert result.returncode == 0, result.stderr
    assert result.stdout and "Traceback" not in result.stderr


def test_doctor_does_not_claim_test_or_resume_success(tmp_path):
    result = cli(tmp_path, "doctor")
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)["data"]
    assert data["checkpoint_resume"] == "SAME_ENVIRONMENT_LOCAL_REPLAY_BEFORE_CONTINUE"
    assert data["scientific_validation"] == "NOT_EXECUTED"


def test_checkpoint_cli_and_finalized_resume_have_one_terminal_json(tmp_path):
    from test_workflow import CAP, files

    inp, cfg = files(tmp_path)
    cp, output = tmp_path / "checkpoint", tmp_path / "output"
    run = cli(
        tmp_path,
        "run",
        inp,
        cfg,
        output,
        "--max-bytes",
        CAP,
        "--checkpoint",
        cp,
        "--allow-unattainable-plan",
    )
    assert run.returncode == 0, run.stderr
    assert len(run.stdout.splitlines()) == 1
    data = json.loads(run.stdout)["data"]
    assert data["checkpoint"] == {
        "execution_id": data["checkpoint"]["execution_id"],
        "replayed_replicates": 0,
        "appended_replicates": 9,
        "mode": "replay_before_continue",
    }
    assert "commit" in run.stderr and "finalized" in run.stderr
    resumed = cli(tmp_path, "resume", cp, tmp_path / "second", "--max-bytes", CAP)
    assert resumed.returncode == 0, resumed.stderr
    assert len(resumed.stdout.splitlines()) == 1
    resumed_data = json.loads(resumed.stdout)["data"]
    assert resumed_data["checkpoint"]["execution_id"] == data["checkpoint"]["execution_id"]
    assert resumed_data["checkpoint"]["replayed_replicates"] == 9
    assert resumed_data["checkpoint"]["appended_replicates"] == 0
    assert "replay" in resumed.stderr
    assert {k: v for k, v in data.items() if k != "checkpoint"} == {
        k: v for k, v in resumed_data.items() if k != "checkpoint"
    }


def test_checkpoint_cli_ne_is_exit_seven_and_refusal_is_two(tmp_path):
    from test_workflow import CAP, files

    inp, cfg = files(tmp_path, mode="observed_statistic_scan")
    refused = cli(
        tmp_path,
        "run",
        inp,
        cfg,
        tmp_path / "no",
        "--max-bytes",
        CAP,
        "--checkpoint",
        tmp_path / "refused",
    )
    assert refused.returncode == 2 and json.loads(refused.stdout)["error"] == "unattainable_plan"
    assert not (tmp_path / "refused").exists()
    ne = cli(
        tmp_path,
        "run",
        inp,
        cfg,
        tmp_path / "ne",
        "--max-bytes",
        CAP,
        "--checkpoint",
        tmp_path / "cp",
        "--allow-unattainable-plan",
    )
    assert ne.returncode == 7 and json.loads(ne.stdout)["outcome"] == "NOT_EVALUABLE"
    resumed = cli(tmp_path, "resume", tmp_path / "cp", tmp_path / "again", "--max-bytes", CAP)
    assert resumed.returncode == 7
    assert json.loads(resumed.stdout)["data"]["checkpoint"]["replayed_replicates"] == 0


@pytest.mark.parametrize("kind", ["busy", "store"])
def test_checkpoint_errors_precede_generic_value_error(tmp_path, monkeypatch, capsys, kind):
    from selcal import cli as cli_module
    from selcal.checkpoint_lock import CheckpointLockError
    from selcal.checkpoint_store import CheckpointStoreError

    error = (
        CheckpointLockError("CHECKPOINT_BUSY", "busy")
        if kind == "busy"
        else CheckpointStoreError("REPLAY_MISMATCH", "different")
    )

    def fail(args):
        raise error

    monkeypatch.setattr(cli_module, "_execute", fail)
    code = cli_module.main(["doctor"])
    stdout = capsys.readouterr().out
    assert code == 4 and json.loads(stdout)["error"] == error.code


@pytest.mark.parametrize("checkpointed", [False, True])
def test_cli_interrupt_guidance_does_not_guarantee_recoverability(
    monkeypatch, capsys, checkpointed
):
    from selcal import cli as cli_module

    def fail(args):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli_module, "_execute", fail)
    args = ["run", "input", "config", "output", "--max-bytes", "1048576"]
    if checkpointed:
        args += ["--checkpoint", "checkpoint"]
    code = cli_module.main(args)
    stdout = capsys.readouterr().out
    assert code == 130
    result = json.loads(stdout)
    if checkpointed:
        assert result["error"] == "interrupted_checkpoint_requires_validation"
        assert "resume" in result["detail"] and "not guaranteed" in result["detail"]
    else:
        assert result["error"] == "interrupted_no_resume_claim"


def test_cli_reserved_checkpoint_output_is_exit_four_and_preserves_recoverability(tmp_path):
    from test_workflow import CAP
    from test_workflow_recovery import partial, snapshot

    cp = partial(tmp_path)
    before = snapshot(cp)
    rejected = cli(tmp_path, "resume", cp, cp / "state.sqlite-wal", "--max-bytes", CAP)
    assert rejected.returncode == 4, rejected.stderr
    payload = json.loads(rejected.stdout)
    assert payload["error"] == "RESERVED_OUTPUT" and payload["outcome"] == "FAIL"
    assert payload["data"] is None and "commit" not in rejected.stderr
    assert snapshot(cp) == before and not (cp / "state.sqlite-wal").exists()
    completed = cli(tmp_path, "resume", cp, cp / "result.sqlite", "--max-bytes", CAP)
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("limit", ["0", "-1", "abc"])
def test_cli_bad_limits_reject_without_traceback_or_output_file(tmp_path, limit):
    record = tmp_path / "absent.sqlite"
    result = cli(tmp_path, "run", INPUT, CONFIG, record, "--max-bytes", limit)
    assert result.returncode == 2
    assert "Traceback" not in result.stderr
    assert not record.exists()


def test_cli_bad_config_and_missing_input_do_not_become_ne(tmp_path):
    config = tmp_path / "bad.json"
    config.write_text("{}", newline="\n", encoding="utf-8")
    result = cli(tmp_path, "validate", INPUT, config)
    assert result.returncode == 2, result.stderr
    assert json.loads(result.stdout)["outcome"] == "INVALID_REQUEST"
    missing = cli(tmp_path, "validate", tmp_path / "missing.csv", CONFIG)
    assert missing.returncode == 4, missing.stderr
    assert json.loads(missing.stdout)["outcome"] == "FAIL"
    assert "Traceback" not in missing.stderr


def test_cli_refuses_to_replace_existing_record(tmp_path):
    record = tmp_path / "user.txt"
    record.write_text("user content", newline="\n", encoding="utf-8")
    result = cli(tmp_path, "run", INPUT, CONFIG, record, "--max-bytes", 1048576)
    assert result.returncode == 4
    assert record.read_text(encoding="utf-8") == "user content"
    assert json.loads(result.stdout)["outcome"] == "FAIL"


def test_cli_entrypoint_declared_in_package():
    import tomllib

    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"].get("scripts", {}).get("selcal") == "selcal.cli:main"


@pytest.mark.parametrize(
    "case", ["input", "config", "record", "record_exists", "parent", "report_exists"]
)
def test_cli_file_errors_explain_recovery_without_overwriting(tmp_path, case):
    record = tmp_path / "result.sqlite"
    report = tmp_path / "report.html"
    if case in {"record_exists", "report_exists"}:
        created = cli(tmp_path, "run", INPUT, CONFIG, record, "--max-bytes", 1048576)
        assert created.returncode == 0
    if case == "report_exists":
        report.write_bytes(b"keep existing report")
    before = {p: p.read_bytes() for p in (record, report) if p.exists()}
    if case == "input":
        args, required = ("validate", tmp_path / "absent.csv", CONFIG), ("Input file", "Check")
    elif case == "config":
        args, required = (
            ("validate", INPUT, tmp_path / "absent.json"),
            ("Configuration file", "Check"),
        )
    elif case == "record":
        args, required = ("verify", record, "--max-bytes", 1048576), ("Record file", "Check")
    elif case == "record_exists":
        args = ("run", INPUT, CONFIG, record, "--max-bytes", 1048576)
        required = ("Output already exists", "Choose a new")
    elif case == "parent":
        args = ("run", INPUT, CONFIG, tmp_path / "absent" / "result.sqlite", "--max-bytes", 1048576)
        required = ("Output parent directory", "Create")
    else:
        args = ("report", record, report, "--max-bytes", 1048576)
        required = ("Output already exists", "Choose a new")
    done = cli(tmp_path, *args)
    assert done.returncode == 4
    assert len(done.stdout.splitlines()) == 1
    payload = json.loads(done.stdout)
    assert payload["error"] == "io_or_integrity_failure"
    assert isinstance(payload["detail"], str)
    assert all(text in payload["detail"] for text in required)
    assert payload["detail"] in done.stderr and "Traceback" not in done.stderr
    assert str(tmp_path) not in payload["detail"]
    assert {p: p.read_bytes() for p in before} == before
    assert not (tmp_path / "absent").exists()


def test_cli_missing_input_is_not_mislabelled_when_output_has_the_same_path(tmp_path):
    missing = tmp_path / "absent.csv"
    done = cli(tmp_path, "run", missing, CONFIG, missing, "--max-bytes", 1048576)
    assert done.returncode == 4
    detail = json.loads(done.stdout)["detail"]
    assert "Input file" in detail and "not found" in detail
    assert "Output parent" not in detail and not missing.exists()


def test_cli_input_parent_is_a_file_has_no_output_directory_advice(tmp_path):
    parent = tmp_path / "not-a-directory"
    parent.write_bytes(b"keep original")
    done = cli(tmp_path, "validate", parent / "series.csv", CONFIG)
    assert done.returncode == 4
    detail = json.loads(done.stdout)["detail"]
    assert "path" in detail and "directory" in detail
    assert "output" not in detail.lower()
    assert parent.read_bytes() == b"keep original"


def _path_error_command(tmp_path, kind, path):
    if kind == "input":
        return ["validate", path, CONFIG]
    if kind == "config":
        return ["validate", INPUT, path]
    if kind == "record":
        return ["verify", path, "--max-bytes", 1048576]
    if kind == "run-output":
        return ["run", INPUT, CONFIG, path, "--max-bytes", 1048576]
    record = tmp_path / "result.sqlite"
    created = cli(tmp_path, "run", INPUT, CONFIG, record, "--max-bytes", 1048576)
    assert created.returncode == 0, created.stderr
    return ["report", record, path, "--max-bytes", 1048576]


def _path_spelling(path, tmp_path, spelling):
    if spelling == "absolute":
        return str(path)
    relative = path.relative_to(tmp_path)
    if spelling == "relative-native":
        return str(relative)
    return "./" + relative.as_posix()


@pytest.mark.parametrize("kind", ["input", "config", "record", "run-output", "report-output"])
@pytest.mark.parametrize("spelling", ["absolute", "relative-native", "relative-forward-slash"])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("windows_missing_error", [False, True])
def test_cli_file_ancestor_diagnosis_preserves_files(
    tmp_path, monkeypatch, capsys, kind, spelling, nested, windows_missing_error
):
    from selcal import cli as cli_module

    folder = tmp_path / "中文 data"
    folder.mkdir()
    blocker = folder / "not-a-directory"
    blocker.write_bytes(b"keep original")
    target = blocker / "nested" / "child" if nested else blocker / "child"
    args = _path_error_command(tmp_path, kind, _path_spelling(target, tmp_path, spelling))
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    monkeypatch.chdir(tmp_path)
    if windows_missing_error:
        execute = cli_module._execute

        def translate_real_parent_error(namespace):
            try:
                return execute(namespace)
            except NotADirectoryError as error:
                # Windows reports this real file-ancestor failure as ENOENT.
                # Only the OS exception shape is emulated; the workflow runs.
                raise FileNotFoundError(errno.ENOENT, "Path not found", error.filename) from error

        monkeypatch.setattr(cli_module, "_execute", translate_real_parent_error)
    assert cli_module.main(list(map(str, args))) == 4
    captured = capsys.readouterr()
    assert len(captured.out.splitlines()) == 1
    payload = json.loads(captured.out)
    assert payload["error"] == "io_or_integrity_failure"
    assert "parent path is not a directory" in payload["detail"]
    assert "Create" not in payload["detail"]
    assert payload["detail"] in captured.err and "Traceback" not in captured.err
    assert str(tmp_path) not in captured.out + captured.err
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize(
    "kind,label",
    [
        ("input", "Input file"),
        ("config", "Configuration file"),
        ("record", "Record file"),
        ("run-output", "Output parent directory"),
        ("report-output", "Output parent directory"),
    ],
)
@pytest.mark.parametrize("spelling", ["absolute", "relative-native", "relative-forward-slash"])
def test_cli_missing_paths_keep_the_correct_operation_label(tmp_path, kind, label, spelling):
    target = tmp_path / "中文 missing" / "nested" / "child"
    args = _path_error_command(tmp_path, kind, _path_spelling(target, tmp_path, spelling))
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    done = cli(tmp_path, *args)
    assert done.returncode == 4, done.stderr
    detail = json.loads(done.stdout)["detail"]
    assert label in detail
    assert "not a directory" not in detail
    assert "Traceback" not in done.stderr and str(tmp_path) not in done.stdout + done.stderr
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before
    assert not (tmp_path / "中文 missing").exists()


def test_cli_ancestor_probe_does_not_replace_error_on_inaccessible_parent(
    tmp_path, monkeypatch, capsys
):
    from selcal import cli as cli_module

    folder = tmp_path / "inaccessible"
    folder.mkdir()
    target = folder / "missing.csv"
    stat = Path.stat

    def denied_parent(path, *args, **kwargs):
        if path == folder:
            raise PermissionError(errno.EACCES, "Permission denied", str(path))
        return stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", denied_parent)
    assert cli_module.main(["validate", str(target), str(CONFIG)]) == 4
    captured = capsys.readouterr()
    detail = json.loads(captured.out)["detail"]
    assert "Input file" in detail and "not found" in detail
    assert "not a directory" not in detail and "Traceback" not in captured.err


def test_cli_permission_error_keeps_its_diagnosis_without_ancestor_probes(tmp_path, monkeypatch):
    import argparse

    from selcal.cli import _io_detail

    blocker = tmp_path / "not-a-directory"
    blocker.write_bytes(b"keep original")
    target = blocker / "child"

    def unexpected_probe(*args, **kwargs):
        pytest.fail("Permission errors must not be reclassified by ancestor probing")

    monkeypatch.setattr(Path, "stat", unexpected_probe)
    error = PermissionError(errno.EACCES, "Permission denied", str(target))
    detail = _io_detail(error, argparse.Namespace(input=str(target)))
    assert detail.startswith("Permission denied.")
    assert "not a directory" not in detail and str(tmp_path) not in detail
    assert blocker.read_bytes() == b"keep original"
