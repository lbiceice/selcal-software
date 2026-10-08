"""Archive opening and member decoding must fail with an auditable checker receipt."""

from __future__ import annotations

import importlib.util
import struct
import zipfile
from pathlib import Path

import pytest
from test_acceptance_tools import _check, _copy, _save_current, _tree
from test_acceptance_tools import built as _built
from test_ui_artifacts import generated

built = _built
CHECKER = Path(__file__).resolve().parents[1] / "scripts" / "check_saved_downloads.py"
MODES = {
    "stored": zipfile.ZIP_STORED,
    "deflate": zipfile.ZIP_DEFLATED,
    "bzip2": zipfile.ZIP_BZIP2,
    "lzma": zipfile.ZIP_LZMA,
}


def checker():
    spec = importlib.util.spec_from_file_location("archive_checker", CHECKER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def archive_fixture(tmp_path: Path, mode: str = "stored") -> Path:
    archive = tmp_path / "evidence.zip"
    with zipfile.ZipFile(archive, "w", compression=MODES[mode]) as zipped:
        for index in range(11):
            zipped.writestr(f"member{index}.txt", b"test-data" * 256)
    return archive


def damage_payload(archive: Path) -> None:
    with zipfile.ZipFile(archive) as zipped:
        member = zipped.infolist()[0]
    raw = bytearray(archive.read_bytes())
    name_length, extra_length = struct.unpack_from("<HH", raw, member.header_offset + 26)
    offset = member.header_offset + 30 + name_length + extra_length
    raw[offset : offset + member.compress_size] = b"\xff" * member.compress_size
    archive.write_bytes(raw)


@pytest.mark.parametrize("mode", MODES)
def test_valid_archive_encoding_retains_all_eleven_exact_members(tmp_path, mode):
    module = checker()
    archive = archive_fixture(tmp_path, mode)
    target = tmp_path / "unpacked"
    module._unpack(archive, target)
    assert len(list(target.iterdir())) == 11
    assert all(path.read_bytes() == b"test-data" * 256 for path in target.iterdir())


@pytest.mark.parametrize("mode", MODES)
def test_corrupt_archive_member_is_a_bounded_failure_before_output_creation(tmp_path, mode):
    module = checker()
    archive = archive_fixture(tmp_path, mode)
    damage_payload(archive)
    target = tmp_path / "unpacked"
    with pytest.raises(module.CheckFailed, match="not a readable ZIP archive"):
        module._unpack(archive, target)
    assert not target.exists()


@pytest.mark.parametrize("kind", ["encrypted", "patched", "strong", "method", "header"])
def test_unsupported_or_corrupt_zip_headers_are_bounded_failures(tmp_path, kind):
    module = checker()
    archive = archive_fixture(tmp_path)
    raw = bytearray(archive.read_bytes())
    central = raw.index(b"PK\x01\x02")
    if kind in {"encrypted", "patched", "strong"}:
        flag = {"encrypted": 1, "patched": 32, "strong": 64}[kind]
        struct.pack_into("<H", raw, 6, flag)
        struct.pack_into("<H", raw, central + 8, flag)
    elif kind == "method":
        struct.pack_into("<H", raw, 8, 99)
        struct.pack_into("<H", raw, central + 10, 99)
    else:
        raw[0] ^= 1
    archive.write_bytes(raw)
    target = tmp_path / "unpacked"
    with pytest.raises(module.CheckFailed):
        module._unpack(archive, target)
    assert not target.exists()


def test_existing_output_is_not_overwritten(tmp_path):
    module = checker()
    archive = archive_fixture(tmp_path)
    target = tmp_path / "unpacked"
    target.mkdir()
    (target / "keep").write_bytes(b"original")
    with pytest.raises(FileExistsError):
        module._unpack(archive, target)
    assert _tree(target) == {"keep": b"original"}


def test_unrelated_programmer_exception_is_not_disguised_as_a_bad_archive(tmp_path, monkeypatch):
    module = checker()
    archive = archive_fixture(tmp_path)

    def defect(*args, **kwargs):
        raise RuntimeError("unrelated programming defect")

    monkeypatch.setattr(module.zipfile.ZipFile, "read", defect)
    target = tmp_path / "unpacked"
    with pytest.raises(RuntimeError, match="unrelated programming defect"):
        module._unpack(archive, target)
    assert not target.exists()


def test_historical_member_crc_error_returns_cli_receipt_and_cleans_snapshot(built, tmp_path):
    from selcal.ui_jobs import JobManager

    workspace, job_id, _ = _copy(built, tmp_path)
    saved = _save_current(workspace, job_id, tmp_path / "saved")
    manager = JobManager(workspace)
    try:
        # Retain the first bundle as history, without weakening any current-artifact checks.
        generated(manager, job_id, "export")
    finally:
        manager.close()
    historical_op = saved["bundle"].name.split("-")[2]
    operations = workspace / "jobs" / job_id / "operations"
    matching = [path for path in operations.iterdir() if path.name.startswith(historical_op)]
    assert len(matching) == 1
    original = matching[0] / "evidence.zip"
    damage_payload(original)
    saved["bundle"].write_bytes(original.read_bytes())
    before = _tree(workspace), _tree(tmp_path / "saved")
    code, receipt = _check(
        tmp_path, workspace, tmp_path / "saved", "--allow-history", "--count", "3"
    )
    assert code == 1 and receipt["status"] == "FAIL"
    assert receipt["failure_phase"] == "export_content"
    assert "not a readable ZIP archive" in receipt["first_error"]
    assert receipt["inputs_unchanged"] is True
    assert receipt["snapshot_cleanup"]["status"] == "PASS"
    assert receipt["snapshot_cleanup"]["manager_close"] == "PASS"
    assert (_tree(workspace), _tree(tmp_path / "saved")) == before
    assert not list(tmp_path.glob("out-*/*-unpacked"))
