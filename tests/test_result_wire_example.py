from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from _documentation import public_documentation_text

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "save_and_read_result.py"


def run_example(action: str, path: Path, limit: int = 65_536) -> subprocess.CompletedProcess[str]:
    assert EXAMPLE.is_file(), "missing save/read example implementation"
    return subprocess.run(
        [sys.executable, "-B", str(EXAMPLE), action, str(path), "--max-bytes", str(limit)],
        cwd=path.parent,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=15,
        check=False,
    )


def test_real_calibration_saved_and_read_in_a_different_process(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    saved = run_example("save", path)
    assert saved.returncode == 0, saved.stderr
    before = path.read_bytes()
    loaded = run_example("read", path)
    assert loaded.returncode == 0, loaded.stderr
    assert saved.stderr == loaded.stderr == ""
    assert path.read_bytes() == before
    for output, action in ((saved, "save"), (loaded, "read")):
        report = json.loads(output.stdout)
        assert report["action"] == action
        assert report["status"] == "complete"
        assert report["planned_replicates"] == report["retained_replicates"] == 9
        assert report["failure_count"] == 0
        assert report["p_value"] == 0.2
        assert report["verification_scope"] == "content_only_not_replay"
        assert report["byte_count"] == len(before)
        assert report["sha256"] == hashlib.sha256(before).hexdigest()
    payload = json.loads(before)
    assert payload["schema"] == "selcal.calibration-result-wire.v1"
    assert len(payload) == 17
    assert len(payload["observed_results"]) == 3
    assert len(payload["replicates"]) == 9
    assert all(len(row["statistic_results"]) == 3 for row in payload["replicates"])


def test_existing_output_is_not_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "existing.json"
    path.write_bytes(b"existing user content\n")
    before = path.read_bytes()
    result = run_example("save", path)
    assert result.returncode != 0
    assert not result.stdout
    assert "Traceback" not in result.stderr
    assert path.read_bytes() == before


def test_save_cap_rejection_does_not_create_output(tmp_path: Path) -> None:
    path = tmp_path / "too-small.json"
    result = run_example("save", path, limit=1)
    assert result.returncode != 0
    assert "size_limit" in result.stderr
    assert not result.stdout
    assert not path.exists()


def test_read_exact_limit_and_one_byte_under(tmp_path: Path) -> None:
    path = tmp_path / "bounded.json"
    saved = run_example("save", path)
    assert saved.returncode == 0, saved.stderr
    size = path.stat().st_size
    assert run_example("read", path, limit=size).returncode == 0
    rejected = run_example("read", path, limit=size - 1)
    assert rejected.returncode != 0
    assert "size_limit" in rejected.stderr
    assert not rejected.stdout
    assert path.stat().st_size == size


@pytest.mark.parametrize("limit", [sys.maxsize, sys.maxsize + 1, 10**100])
def test_large_positive_cap_reads_actual_content_without_overflow(
    tmp_path: Path, limit: int
) -> None:
    path = tmp_path / "small-record-large-cap.json"
    saved = run_example("save", path)
    assert saved.returncode == 0, saved.stderr
    before = path.read_bytes()
    loaded = run_example("read", path, limit=limit)
    assert loaded.returncode == 0, loaded.stderr
    assert loaded.stderr == ""
    assert json.loads(loaded.stdout)["sha256"] == hashlib.sha256(before).hexdigest()
    assert path.read_bytes() == before


def test_invalid_content_and_missing_input_report_errors_not_success(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_bytes(b'{"schema":"invalid"}\n')
    before = path.read_bytes()
    for supplied in (path, tmp_path / "missing.json"):
        result = run_example("read", supplied)
        assert result.returncode != 0
        assert not result.stdout
        assert "Traceback" not in result.stderr
    assert path.read_bytes() == before
    assert not (tmp_path / "missing.json").exists()


@pytest.mark.parametrize("limit", [0, -1])
def test_invalid_cap_is_rejected_before_save(tmp_path: Path, limit: int) -> None:
    path = tmp_path / "invalid-limit.json"
    result = run_example("save", path, limit=limit)
    assert result.returncode != 0
    assert not result.stdout
    assert not path.exists()


def test_readme_routes_to_lossless_record_example() -> None:
    readme = public_documentation_text(ROOT)
    assert "python examples/save_and_read_result.py save result.json --max-bytes 65536" in readme
    assert "python examples/save_and_read_result.py read result.json --max-bytes 65536" in readme
    assert "content_only_not_replay" in readme
