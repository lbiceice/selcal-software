"""Explicit record checks must never pass without checking any records."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "acceptance_check", ROOT / "scripts" / "acceptance_check.py"
)
assert spec is not None and spec.loader is not None
acceptance = importlib.util.module_from_spec(spec)
spec.loader.exec_module(acceptance)
WINDOWS_CWD_LIMIT = 258  # R17 Windows return: a 259-character working directory cannot start


def _launch(tmp_path: Path) -> Path:
    launch = tmp_path / "launch"
    launch.mkdir(exist_ok=True)
    return launch


def _deep(root: Path, length: int) -> Path:
    """A real folder whose path is at least `length` characters, with a Chinese part."""
    folder = root / "深 目录"
    while len(str(folder)) < length:
        folder = folder / "layer-0123456789"
    folder.mkdir(parents=True)
    return folder


@pytest.mark.parametrize("mode", ["_foreign_records", "_legacy_records"])
@pytest.mark.parametrize("kind", ["missing", "empty", "file", "sqlite-directory"])
def test_requested_record_check_fails_without_record_inputs(
    tmp_path: Path, mode: str, kind: str
) -> None:
    records = tmp_path / "records"
    if kind == "empty":
        records.mkdir()
    elif kind == "file":
        records.write_text("not a directory", encoding="utf-8")
    elif kind == "sqlite-directory":
        (records / "not-a-file.sqlite").mkdir(parents=True)
    out, work = tmp_path / "out", tmp_path / "work"
    out.mkdir()
    work.mkdir()
    checker = acceptance.Checker(out, _launch(tmp_path))

    getattr(acceptance, mode)(checker, records, work)

    assert checker.failures, "an explicitly requested record check examined no inputs"
    assert checker.count >= 1
    assert "FAIL" in (out / "summary.txt").read_text(encoding="utf-8")


@pytest.mark.parametrize("mode", ["_foreign_records", "_legacy_records"])
def test_unreadable_record_is_reported_as_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    records = tmp_path / "records"
    records.mkdir()
    source = records / "unreadable.sqlite"
    source.write_bytes(b"unreadable input")
    real_read = Path.read_bytes

    def denied_read(path: Path) -> bytes:
        if path == source:
            raise PermissionError("simulated unreadable record")
        return real_read(path)

    monkeypatch.setattr(Path, "read_bytes", denied_read)
    out, work = tmp_path / "out", tmp_path / "work"
    out.mkdir()
    work.mkdir()
    checker = acceptance.Checker(out, _launch(tmp_path))

    getattr(acceptance, mode)(checker, records, work)

    assert checker.failures
    assert "simulated unreadable record" in (out / "summary.txt").read_text(encoding="utf-8")


@pytest.mark.parametrize("option", ["--records", "--legacy-records"])
def test_cli_rejects_missing_explicit_record_directory(tmp_path: Path, option: str) -> None:
    done = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "acceptance_check.py"),
         option, str(tmp_path / "missing"), "--out", str(tmp_path / "out")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
    )
    assert done.returncode == 1, done.stdout + done.stderr
    summaries = list((tmp_path / "out").glob("*/summary.txt"))
    assert len(summaries) == 1
    summary = summaries[0].read_text(encoding="utf-8")
    assert "FAIL  " in summary and "FAILED (" in summary
    assert "ALL PASSED" not in summary
    assert "Traceback" not in done.stderr


def test_cli_precision_accepts_relative_output_directory(tmp_path: Path) -> None:
    done = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "acceptance_check.py"),
         "--precision", "--out", "results"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    summaries = list((tmp_path / "results").glob("*/summary.txt"))
    assert len(summaries) == 1
    summary = summaries[0].read_text(encoding="utf-8")
    assert "ALL PASSED" in summary and "FAIL  " not in summary
    assert "moved to another folder still replays (MATCH)" in summary


def test_foreign_record_checks_preserve_local_command_receipts(tmp_path: Path) -> None:
    out, work = tmp_path / "out", tmp_path / "work"
    out.mkdir()
    work.mkdir()
    checker = acceptance.Checker(out, _launch(tmp_path))
    for name, plan in acceptance._plans(work).items():
        acceptance._case(checker, name, plan, work)
    assert not checker.failures
    before = {
        path.name: path.read_bytes()
        for path in out.iterdir()
        if path.name != "summary.txt"
    }
    assert len(before) == 36  # Two local cases, six commands each, three files each.

    acceptance._foreign_records(checker, work, work)

    assert not checker.failures
    assert {name: (out / name).read_bytes() for name in before} == before
    assert len(list(out.glob("*.receipt.json"))) == 18
    for name in ("pearson", "exact"):
        for mode in ("verify", "replay", "decision"):
            local = json.loads((out / f"{name}-{mode}.receipt.json").read_bytes())
            foreign = json.loads((out / f"foreign-{name}-{mode}.receipt.json").read_bytes())
            assert Path(local["argv"][4]) == work / f"{name}.sqlite"
            assert Path(foreign["argv"][4]) == work / f"foreign-{name}.sqlite"


@pytest.mark.parametrize("suffix", ["stdout", "stderr", "receipt.json"])
def test_duplicate_receipt_name_is_refused_without_overwriting(
    tmp_path: Path, suffix: str
) -> None:
    out, work = tmp_path / "out", tmp_path / "work"
    out.mkdir()
    work.mkdir()
    original = out / f"reused.{suffix}"
    original.write_bytes(b"earlier command evidence\n")
    checker = acceptance.Checker(out, _launch(tmp_path))

    with pytest.raises(FileExistsError, match="reused"):
        checker.cli("reused", "doctor")

    assert list(out.iterdir()) == [original]
    assert original.read_bytes() == b"earlier command evidence\n"


def test_deep_work_folder_is_never_a_child_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R17 Windows item 1: work folders of 273-277 characters failed with WinError 267."""
    real_run = subprocess.run
    seen: list[str] = []

    def windows_like_run(argv, *args, cwd=None, **kwargs):  # type: ignore[no-untyped-def]
        seen.append(str(cwd))
        if cwd is None or len(str(cwd)) > WINDOWS_CWD_LIMIT:
            raise NotADirectoryError(267, "The directory name is invalid", str(cwd))
        return real_run(argv, *args, cwd=cwd, **kwargs)

    monkeypatch.setattr(acceptance.subprocess, "run", windows_like_run)
    work = _deep(tmp_path, 300)
    out = _deep(tmp_path / "out", 290)
    checker = acceptance.Checker(out, _launch(tmp_path))
    for name, plan in acceptance._plans(work).items():
        acceptance._case(checker, name, plan, work)
    records = tmp_path / "records"
    records.mkdir()
    for name in ("pearson", "exact"):
        (records / f"{name}.sqlite").write_bytes((work / f"{name}.sqlite").read_bytes())
    acceptance._foreign_records(checker, records, work)
    acceptance._precision(checker, work)

    assert not checker.failures, checker.failures
    assert seen and set(seen) == {str(_launch(tmp_path))}
    for receipt in out.glob("*.receipt.json"):
        data = json.loads(receipt.read_bytes())
        assert data["started"] is True
        files = [a for a in data["argv"][4:] if a.endswith((".csv", ".json", ".sqlite", ".html"))]
        assert all(Path(a).is_absolute() for a in files), data["argv"]


def test_command_that_cannot_start_is_a_recorded_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def cannot_start(*args, **kwargs):  # type: ignore[no-untyped-def]
        error = NotADirectoryError(20, "The directory name is invalid")
        error.winerror = 267  # type: ignore[attr-defined]
        raise error

    monkeypatch.setattr(acceptance.subprocess, "run", cannot_start)
    out = tmp_path / "out"
    out.mkdir()
    checker = acceptance.Checker(out, _launch(tmp_path))

    code, payload = checker.cli("doctor", "doctor")
    checker.check("doctor exit 0", code == 0, f"exit {code}")

    receipt = json.loads((out / "doctor.receipt.json").read_bytes())
    assert code is None and payload is not None and payload["error"] == "not_started"
    assert receipt["started"] is False and receipt["exit_code"] is None
    assert receipt["error_type"] == "NotADirectoryError" and receipt["winerror"] == 267
    assert receipt["cwd_length"] == len(str(_launch(tmp_path)))
    assert checker.failures == ["doctor exit 0"]


def test_cli_with_deep_relative_output_and_shadowing_module(tmp_path: Path) -> None:
    """Relative --out stays relative to the caller; a stray selcal.py is never imported."""
    caller = _deep(tmp_path, 200)
    poison = "raise SystemExit('stray selcal.py was imported')\n"
    (caller / "selcal.py").write_text(poison, encoding="utf-8")
    done = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "acceptance_check.py"),
         "--installed", "--out", "results 结果"],
        cwd=caller,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="backslashreplace",
        check=False,
    )
    summaries = list((caller / "results 结果").glob("*/summary.txt"))
    assert len(summaries) == 1, done.stdout + done.stderr
    summary = summaries[0].read_text(encoding="utf-8")
    assert "stray selcal.py" not in done.stdout + done.stderr
    assert "pearson: lag 2" in summary and "exact: lag 2" in summary
    launch = json.loads((summaries[0].parent / "launch.json").read_bytes())
    assert launch["cwd_length"] < len(str(caller))
    assert not Path(launch["cwd"]).exists(), "the empty launch folder is removed afterwards"


def _precision_with(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake) -> tuple:
    """Run the real precision check with `fake(argv, record)` deciding what each `run` does."""
    real_run = subprocess.run

    def run(argv, *args, **kwargs):  # type: ignore[no-untyped-def]
        if argv[3] == "run" and "repeat" in Path(argv[6]).name:
            return fake(argv, Path(argv[6]), real_run, args, kwargs)
        return real_run(argv, *args, **kwargs)

    monkeypatch.setattr(acceptance.subprocess, "run", run)
    out, work = tmp_path / "out", tmp_path / "work"
    out.mkdir()
    work.mkdir()
    checker = acceptance.Checker(out, _launch(tmp_path))
    acceptance._precision(checker, work)  # must not raise
    summary = (out / "summary.txt").read_text(encoding="utf-8")
    return checker, summary


def _identical_line(summary: str, name: str) -> str:
    (line,) = [
        line for line in summary.splitlines()
        if f"{name}: five separate runs give one identical record" in line
    ]
    return line


@pytest.mark.parametrize("damage", ["all-cannot-start", "one-cannot-start", "exit0-no-record",
                                    "one-corrupt-record"])
def test_precision_never_calls_missing_or_damaged_runs_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    """R18 Windows item 3: five failed starts were reported as five identical records."""

    def fake(argv, record, real_run, args, kwargs):  # type: ignore[no-untyped-def]
        attempt = int(record.name[len("repeat")])
        if damage == "all-cannot-start" or (damage == "one-cannot-start" and attempt == 2):
            error = NotADirectoryError(20, "The directory name is invalid")
            error.winerror = 267  # type: ignore[attr-defined]
            raise error
        if damage == "exit0-no-record":
            return subprocess.CompletedProcess(argv, 0, b'{"exit_code": 0, "data": {}}\n', b"")
        done = real_run(argv, *args, **kwargs)
        if damage == "one-corrupt-record" and attempt == 3:
            record.write_bytes(record.read_bytes()[:-7] + b"damaged")
        return done

    checker, summary = _precision_with(tmp_path, monkeypatch, fake)
    for name in ("pearson", "exact"):
        assert _identical_line(summary, name).startswith("FAIL"), summary
        if damage != "one-corrupt-record":
            assert f"FAIL  {name}: all five separate runs exit 0" in summary
    assert checker.failures
    if damage in ("all-cannot-start", "exit0-no-record"):
        assert "not checked: the first run left no readable record" in summary
    receipts = [
        json.loads(path.read_bytes()) for path in (tmp_path / "out").glob("*repeat*.receipt.json")
    ]
    if damage == "all-cannot-start":
        assert len(receipts) == 10 and all(r["started"] is False for r in receipts)


def test_precision_normal_runs_still_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checker, summary = _precision_with(
        tmp_path, monkeypatch, lambda argv, record, real_run, a, k: real_run(argv, *a, **k)
    )
    assert not checker.failures, summary
    assert _identical_line(summary, "pearson").startswith("PASS")
    assert "moved to another folder still replays (MATCH)" in summary


@pytest.mark.parametrize("damage", ["all-empty", "all-invalid", "first-empty"])
def test_precision_rejects_readable_but_invalid_records_without_secondary_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    """Readable bytes alone are not a valid record, even when all five hashes match."""

    def fake(argv, record, real_run, args, kwargs):  # type: ignore[no-untyped-def]
        done = real_run(argv, *args, **kwargs)
        if damage == "all-invalid":
            record.write_bytes(b"not a SQLite record")
        elif damage == "all-empty" or record.name.startswith("repeat0-"):
            record.write_bytes(b"")
        return done

    checker, summary = _precision_with(tmp_path, monkeypatch, fake)
    for name in ("pearson", "exact"):
        assert _identical_line(summary, name).startswith("FAIL"), summary
        assert f"FAIL  {name}: all five separate runs exit 0" in summary
        assert f"FAIL  {name}: a record moved to another folder" in summary
        assert f"FAIL  {name}: a record with one altered byte is refused" in summary
    assert checker.failures


def test_unexpected_error_still_ends_with_failed_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("simulated defect inside the checker")

    monkeypatch.setattr(acceptance, "_main", broken)
    monkeypatch.setattr(sys, "argv", ["acceptance_check.py", "--out", str(tmp_path)])
    assert acceptance.main() == 1
    (summary,) = tmp_path.glob("*/summary.txt")
    text = summary.read_text(encoding="utf-8")
    assert "FAIL  acceptance check finished without an unexpected error" in text
    assert "simulated defect inside the checker" in text
    assert "FAILED (" in text and "stopped early" in text
