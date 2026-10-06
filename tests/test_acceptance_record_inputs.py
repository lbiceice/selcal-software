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
    checker = acceptance.Checker(out)

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
    checker = acceptance.Checker(out)

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
    checker = acceptance.Checker(out)
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
            assert local["argv"][4] == f"{name}.sqlite"
            assert foreign["argv"][4] == f"foreign-{name}.sqlite"


@pytest.mark.parametrize("suffix", ["stdout", "stderr", "receipt.json"])
def test_duplicate_receipt_name_is_refused_without_overwriting(
    tmp_path: Path, suffix: str
) -> None:
    out, work = tmp_path / "out", tmp_path / "work"
    out.mkdir()
    work.mkdir()
    original = out / f"reused.{suffix}"
    original.write_bytes(b"earlier command evidence\n")
    checker = acceptance.Checker(out)

    with pytest.raises(FileExistsError, match="reused"):
        checker.cli("reused", "doctor", cwd=work)

    assert list(out.iterdir()) == [original]
    assert original.read_bytes() == b"earlier command evidence\n"
