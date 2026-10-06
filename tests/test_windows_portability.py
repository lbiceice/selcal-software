"""Portability of the file workflow to Windows (Windows test report, 2026-09-24).

Windows has no ``os.O_NOFOLLOW`` and opens descriptors in text mode unless ``O_BINARY`` is given.
Its record reader uses CreateFileW and a binary handle conversion. Removing ``O_NOFOLLOW`` on
POSIX exercises the separate identity-verified fallback, not the Windows API. These tests
check both platform contracts; POSIX results do not replace a real Windows run.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest
from _platform_support import directory_link_or_skip, symlink_or_skip

from selcal import workflow, workflow_store
from selcal.inputs import load_csv

MEMBERS = {
    "input": b"in\r\n\x1a\x00put",
    "request": b"{}",
    "result": b"result",
    "metadata": b"meta",
}
CAP = 1_048_576


@pytest.fixture
def no_nofollow(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove O_NOFOLLOW without changing the platform's record-open backend."""
    if hasattr(os, "O_NOFOLLOW"):
        monkeypatch.delattr(os, "O_NOFOLLOW")


def _record(tmp_path: Path, name: str = "run.sqlite") -> Path:
    path = tmp_path / name
    workflow_store.write_record(path, MEMBERS, max_bytes=CAP)
    return path


# R2-01: records must be readable without O_NOFOLLOW, keeping the protections.


def test_record_round_trip_without_o_nofollow(tmp_path: Path, no_nofollow: None) -> None:
    assert workflow_store.read_record(_record(tmp_path), max_bytes=CAP) == MEMBERS


def test_symlinked_record_is_refused_without_o_nofollow(tmp_path: Path, no_nofollow: None) -> None:
    link = tmp_path / "link.sqlite"
    symlink_or_skip(link, _record(tmp_path))
    with pytest.raises(workflow_store.RecordStoreError) as raised:
        workflow_store.read_record(link, max_bytes=CAP)
    assert raised.value.code == "NONREGULAR_INPUT"


def test_record_swapped_for_a_link_before_open_is_refused(
    tmp_path: Path, no_nofollow: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _record(tmp_path)
    other = _record(tmp_path, "other.sqlite")
    real_lstat = os.lstat
    swapped = False

    def swap_after_lstat(path: object) -> os.stat_result:
        nonlocal swapped
        info = real_lstat(path)  # type: ignore[arg-type]
        if Path(os.fspath(path)) == record and not swapped:
            swapped = True
            record.unlink()
            symlink_or_skip(record, other)
        return info

    # Both POSIX open and Windows CreateFileW must reject a path changed after inspection.
    monkeypatch.setattr(os, "lstat", swap_after_lstat)
    with pytest.raises(workflow_store.RecordStoreError) as raised:
        workflow_store.read_record(record, max_bytes=CAP)
    assert raised.value.code in {"INPUT_CHANGED", "NONREGULAR_INPUT"}


class _StatWithAttributes:
    def __init__(self, base: os.stat_result, attributes: int, inode: int | None = None) -> None:
        self._base = base
        self.st_file_attributes = attributes
        self._inode = inode

    def __getattr__(self, name: str) -> object:
        if name == "st_ino" and self._inode is not None:
            return self._inode
        return getattr(self._base, name)


def test_reparse_point_is_refused(
    tmp_path: Path, no_nofollow: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _record(tmp_path)
    real_lstat = os.lstat
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    monkeypatch.setattr(os, "lstat", lambda path: _StatWithAttributes(real_lstat(path), reparse))
    with pytest.raises(workflow_store.RecordStoreError) as raised:
        workflow_store.read_record(record, max_bytes=CAP)
    assert raised.value.code == "NONREGULAR_INPUT"


def test_missing_file_identity_fails_closed(
    tmp_path: Path, no_nofollow: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _record(tmp_path)
    real_lstat = os.lstat
    monkeypatch.setattr(os, "lstat", lambda path: _StatWithAttributes(real_lstat(path), 0, 0))
    with pytest.raises(workflow_store.RecordStoreError) as raised:
        workflow_store.read_record(record, max_bytes=CAP)
    assert raised.value.code == "UNSUPPORTED_CAPABILITY"


@pytest.mark.parametrize("operation", ["write", "read"])
def test_record_files_are_opened_in_binary_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    record = _record(tmp_path) if operation == "read" else tmp_path / "new.sqlite"
    binary_flag = getattr(os, "O_BINARY", 1 << 28)
    seen: list[int] = []

    if os.name == "nt" and operation == "read":
        import msvcrt

        real_open_osfhandle = msvcrt.open_osfhandle

        def recording_open_osfhandle(handle: int, flags: int) -> int:
            seen.append(flags)
            return real_open_osfhandle(handle, flags)

        monkeypatch.setattr(msvcrt, "open_osfhandle", recording_open_osfhandle)
    else:
        real_open = os.open
        native_binary = getattr(os, "O_BINARY", 0)
        monkeypatch.setattr(os, "O_BINARY", binary_flag, raising=False)

        def recording_open(path: object, flags: int, *args: object) -> int:
            seen.append(flags)
            # Strip only the synthetic POSIX flag; retain the actual Windows O_BINARY.
            return real_open(path, (flags & ~binary_flag) | native_binary, *args)

        monkeypatch.setattr(os, "open", recording_open)
    if operation == "write":
        workflow_store.write_record(record, MEMBERS, max_bytes=CAP)
    else:
        assert workflow_store.read_record(record, max_bytes=CAP) == MEMBERS
    assert seen and all(flags & binary_flag for flags in seen)


# Directory given as input: a request error, not an I/O failure.


def test_directory_as_csv_input_is_a_request_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="regular file"):
        load_csv(tmp_path, source_column="x", target_column="y", candidates=(1,))


# R2-08: a UTF-8 byte-order mark (as written by spreadsheet programs) is accepted.


def _csv(tmp_path: Path, name: str, prefix: bytes) -> Path:
    rows = "".join(f"{i},{(i * 7) % 5}\n" for i in range(12))
    path = tmp_path / name
    path.write_bytes(prefix + ("x,y\n" + rows).encode())
    return path


def test_utf8_bom_is_accepted_and_kept_in_the_raw_identity(tmp_path: Path) -> None:
    plain = load_csv(
        _csv(tmp_path, "a.csv", b""), source_column="x", target_column="y", candidates=(1,)
    )
    marked = load_csv(
        _csv(tmp_path, "b.csv", b"\xef\xbb\xbf"),
        source_column="x",
        target_column="y",
        candidates=(1,),
    )
    assert marked.semantic_input_sha256 == plain.semantic_input_sha256
    assert marked.raw_input_sha256 != plain.raw_input_sha256
    assert marked.raw_input_sha256 == hashlib.sha256((tmp_path / "b.csv").read_bytes()).hexdigest()


def test_header_mismatch_names_the_found_and_configured_columns(tmp_path: Path) -> None:
    with pytest.raises(ValueError) as raised:
        load_csv(
            _csv(tmp_path, "a.csv", b""), source_column="a", target_column="y", candidates=(1,)
        )
    message = str(raised.value)
    assert "'x'" in message and "'a'" in message


def _cli(*args: object, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "selcal", *map(str, args)],
        cwd=cwd,
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
    )


def test_cli_request_errors_carry_an_actionable_detail(tmp_path: Path) -> None:
    config = {
        "schema": "selcal.workflow-config.v1",
        "input": {"format": "csv", "source_column": "a", "target_column": "y"},
        "plan": {
            "candidates": [1],
            "statistic_name": "lagged_pearson_v1",
            "statistic_params": {},
            "selection_rule": "max_absolute",
            "null_name": "circular_shift_exact_v1",
            "null_params": {"min_shift": 1},
            "replicates": 11,
            "alpha": 0.05,
            "tie_tolerance": 0.0,
            "root_seed": 1,
        },
    }
    (tmp_path / "plan.json").write_bytes(json.dumps(config).encode())
    _csv(tmp_path, "data.csv", b"")
    done = _cli("validate", "data.csv", "plan.json", cwd=tmp_path)
    payload = json.loads(done.stdout)
    assert done.returncode == 2 and payload["error"] == "invalid_request"
    assert "'a'" in payload["detail"] and "'x'" in payload["detail"]
    assert payload["detail"] in done.stderr


# R2-09: doctor checks that records can actually be written and read back here.


def test_doctor_reports_the_record_round_trip() -> None:
    report = workflow.doctor()
    assert report["record_round_trip"] == "PASS"
    if os.name == "nt":
        expected = "windows_no_follow_open"
    else:
        expected = "no_follow_open" if hasattr(os, "O_NOFOLLOW") else "identity_verified_open"
    assert report["record_read_method"] == expected


def test_doctor_names_the_platform_method_without_o_nofollow(no_nofollow: None) -> None:
    report = workflow.doctor()
    assert report["record_round_trip"] == "PASS"
    expected = "windows_no_follow_open" if os.name == "nt" else "identity_verified_open"
    assert report["record_read_method"] == expected


def test_cli_doctor_fails_when_records_cannot_be_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from selcal import cli

    def unreadable(path: object, *, max_bytes: int) -> dict[str, bytes]:
        raise workflow_store.RecordStoreError("UNSUPPORTED_CAPABILITY", "simulated")

    monkeypatch.setattr(workflow, "read_record", unreadable)
    assert cli.main(["doctor"]) == 4


def test_directory_refused_like_windows_open_is_a_request_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from selcal import input_resources

    def windows_like_open(path: object, flags: int, *args: object) -> int:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(input_resources, "_os_open", windows_like_open)
    with pytest.raises(ValueError, match="regular file"):
        input_resources.read_regular_file_snapshot(tmp_path, raw_limit=8, reason="CSV_RAW_BYTES")
    missing_permission = tmp_path / "locked.csv"
    missing_permission.write_bytes(b"x,y\n")
    with pytest.raises(PermissionError):
        input_resources.read_regular_file_snapshot(
            missing_permission, raw_limit=8, reason="CSV_RAW_BYTES"
        )


# Checker review 3: only link-type reparse points are refused; cloud files are ordinary files.


class _StatWithTag(_StatWithAttributes):
    def __init__(self, base: os.stat_result, tag: int) -> None:
        super().__init__(base, getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
        self.st_reparse_tag = tag


@pytest.mark.parametrize(
    ("tag", "refused"),
    [
        (0xA000000C, True),  # IO_REPARSE_TAG_SYMLINK
        (0xA0000003, True),  # IO_REPARSE_TAG_MOUNT_POINT (junction)
        (0x8000001B, True),  # IO_REPARSE_TAG_APPEXECLINK
        (0x20000099, True),  # any other name-surrogate tag
        (0x9000001A, False),  # OneDrive / cloud-files placeholder
        (0x80000017, False),  # IO_REPARSE_TAG_WOF (compressed file)
    ],
)
def test_only_link_type_reparse_points_are_refused(tag: int, refused: bool) -> None:
    info = _StatWithTag(os.lstat(__file__), tag)
    assert workflow_store._is_link(info) is refused  # type: ignore[arg-type]


def test_missing_volume_identity_fails_closed(
    tmp_path: Path, no_nofollow: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _record(tmp_path)
    real_lstat = os.lstat

    class NoVolume(_StatWithAttributes):
        st_dev = 0

    monkeypatch.setattr(os, "lstat", lambda path: NoVolume(real_lstat(path), 0))
    with pytest.raises(workflow_store.RecordStoreError) as raised:
        workflow_store.read_record(record, max_bytes=CAP)
    assert raised.value.code == "UNSUPPORTED_CAPABILITY"


def test_windows_identity_ignores_creation_time(monkeypatch: pytest.MonkeyPatch) -> None:
    info = os.stat(__file__)
    monkeypatch.setattr(workflow_store.os, "name", "nt")
    identity = workflow_store._identity(info)
    monkeypatch.undo()
    assert identity == (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)


windows_only = pytest.mark.skipif(os.name != "nt", reason="exercises the Windows CreateFileW path")


@windows_only
def test_windows_record_round_trip_uses_createfile(tmp_path: Path) -> None:
    assert workflow_store.read_method() == "windows_no_follow_open"
    assert workflow_store.read_record(_record(tmp_path), max_bytes=CAP) == MEMBERS


@windows_only
def test_windows_link_is_opened_as_itself_not_followed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _record(tmp_path)
    other = _record(tmp_path, "other.sqlite")
    real_lstat = os.lstat
    calls = {"n": 0}

    def swap_after_first_lstat(path: object) -> os.stat_result:
        info = real_lstat(path)  # type: ignore[arg-type]
        calls["n"] += 1
        if calls["n"] == 1:
            record.unlink()
            symlink_or_skip(record, other)
        return info

    monkeypatch.setattr(os, "lstat", swap_after_first_lstat)
    with pytest.raises(workflow_store.RecordStoreError) as raised:
        workflow_store.read_record(record, max_bytes=CAP)
    assert raised.value.code in {"INPUT_CHANGED", "NONREGULAR_INPUT"}


def test_config_file_with_utf8_bom_is_accepted(tmp_path: Path) -> None:
    example = Path(__file__).resolve().parents[1] / "examples" / "workflow"
    (tmp_path / "series.csv").write_bytes((example / "series.csv").read_bytes())
    (tmp_path / "plan.json").write_bytes(b"\xef\xbb\xbf" + (example / "pearson.json").read_bytes())
    done = _cli("validate", "series.csv", "plan.json", cwd=tmp_path)
    assert done.returncode == 0, done.stdout + done.stderr


def test_doctor_checks_a_chosen_folder(tmp_path: Path) -> None:
    done = _cli("doctor", "--folder", tmp_path, cwd=tmp_path)
    data = json.loads(done.stdout)["data"]
    assert done.returncode == 0
    assert data["record_round_trip"] == "PASS"
    assert data["record_round_trip_folder"] == str(tmp_path)
    assert list(tmp_path.iterdir()) == []  # the check leaves nothing behind


def test_doctor_fails_for_a_missing_folder(tmp_path: Path) -> None:
    done = _cli("doctor", "--folder", tmp_path / "absent", cwd=tmp_path)
    assert done.returncode == 4
    assert json.loads(done.stdout)["data"]["record_round_trip"].startswith("FAIL")


# Checker review 4: a file held open by another program must say so (N3); the Windows Container
# Isolation tag redirects like a link (N5).


def test_windows_container_isolation_tag_is_refused() -> None:
    wci = 0x80000018  # IO_REPARSE_TAG_WCI: redirects to another file inside a container
    assert workflow_store._is_link(_StatWithTag(os.lstat(__file__), wci))  # type: ignore[arg-type]


@pytest.mark.parametrize("winerror", [32, 33])  # sharing violation, lock violation
def test_file_held_open_by_another_program_is_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, winerror: int
) -> None:
    record = _record(tmp_path)

    def busy(path: object, flags: int, *args: object) -> int:
        error = PermissionError(13, "The process cannot access the file")
        error.winerror = winerror  # type: ignore[attr-defined]
        raise error

    def busy_windows(path: object, *, follow: bool) -> int:
        return busy(path, 0)

    monkeypatch.setattr(os, "open", busy)
    monkeypatch.setattr(workflow_store, "_open_windows", busy_windows)
    with pytest.raises(workflow_store.RecordStoreError) as raised:
        workflow_store.read_record(record, max_bytes=CAP)
    assert raised.value.code == "INPUT_BUSY"
    assert "another program" in str(raised.value)


def test_cli_reports_a_busy_record_with_a_detail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from selcal import cli, workflow

    def busy(path: object, *, max_bytes: int) -> dict[str, bytes]:
        raise workflow_store.RecordStoreError(
            "INPUT_BUSY", "the record is open in another program; close it and try again"
        )

    monkeypatch.setattr(workflow, "read_record", busy)
    record = _record(tmp_path)
    assert cli.main(["verify", str(record), "--max-bytes", str(CAP)]) == 4


def test_other_permission_errors_are_not_relabelled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _record(tmp_path)

    def denied(path: object, flags: int, *args: object) -> int:
        raise PermissionError(13, "Permission denied")

    def denied_windows(path: object, *, follow: bool) -> int:
        return denied(path, 0)

    monkeypatch.setattr(os, "open", denied)
    monkeypatch.setattr(workflow_store, "_open_windows", denied_windows)
    with pytest.raises(PermissionError):
        workflow_store.read_record(record, max_bytes=CAP)


def test_directory_link_helper_creates_a_link_the_product_refuses(tmp_path: Path) -> None:
    """On POSIX a symlink; on Windows without privilege a junction. Either must be a link.

    Native Windows runs then show which kind was made: a junction carries the mount-point
    reparse tag and must still be classified as a link, never as an ordinary directory.
    """
    target = tmp_path / "target"
    target.mkdir()
    (target / "keep").write_bytes(b"original")
    link = tmp_path / "link"
    directory_link_or_skip(link, target)
    info = os.lstat(link)
    assert workflow_store._is_link(info)
    if os.name == "nt" and not stat.S_ISLNK(info.st_mode):
        assert info.st_reparse_tag == stat.IO_REPARSE_TAG_MOUNT_POINT
    assert (link / "keep").read_bytes() == b"original"
    os.rmdir(link) if os.name == "nt" else link.unlink()
    assert (target / "keep").read_bytes() == b"original"
