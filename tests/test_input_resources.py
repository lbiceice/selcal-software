from __future__ import annotations

import os
import stat
import subprocess
import sys
from dataclasses import FrozenInstanceError, fields, is_dataclass
from pathlib import Path

import pytest

import selcal.input_resources as input_resources
from selcal.contracts_v2 import ResourceLimitError
from selcal.input_resources import (
    INPUT_LIMITS_V1,
    InputLimitsV1,
    raise_input_resource_limit,
    read_regular_file_snapshot,
)


def test_limits_are_exact_slotted_frozen_and_formula_consistent() -> None:
    assert is_dataclass(InputLimitsV1)
    assert not hasattr(INPUT_LIMITS_V1, "__dict__")
    assert {field.name for field in fields(InputLimitsV1)} == {
        "csv_raw_bytes",
        "csv_record_characters",
        "csv_field_characters",
        "csv_data_rows",
        "npz_raw_bytes",
        "npz_central_directory_bytes",
        "npy_header_bytes",
        "npy_elements",
        "npz_member_uncompressed_bytes",
        "npz_total_uncompressed_bytes",
    }
    assert INPUT_LIMITS_V1.csv_raw_bytes == 67_108_864
    assert INPUT_LIMITS_V1.csv_record_characters == 1_024
    assert INPUT_LIMITS_V1.csv_field_characters == 256
    assert INPUT_LIMITS_V1.csv_data_rows == 1_000_000
    assert INPUT_LIMITS_V1.npz_raw_bytes == 33_554_432
    assert INPUT_LIMITS_V1.npz_central_directory_bytes == 16_384
    assert INPUT_LIMITS_V1.npy_header_bytes == 4_096
    assert INPUT_LIMITS_V1.npy_elements == 1_000_000
    assert INPUT_LIMITS_V1.npz_member_uncompressed_bytes == 8_004_108
    assert INPUT_LIMITS_V1.npz_total_uncompressed_bytes == 16_008_216
    assert INPUT_LIMITS_V1.npz_member_uncompressed_bytes == 8 + 4 + 4_096 + 1_000_000 * 8
    assert INPUT_LIMITS_V1.npz_total_uncompressed_bytes == 2 * 8_004_108
    with pytest.raises(FrozenInstanceError):
        INPUT_LIMITS_V1.csv_raw_bytes = 1  # type: ignore[misc]


def test_resource_error_has_exact_path_free_observed_message() -> None:
    with pytest.raises(ResourceLimitError) as captured:
        raise_input_resource_limit("CSV_RAW_BYTES", limit=8, observed=9)
    assert str(captured.value) == (
        "INPUT_RESOURCE_LIMIT_EXCEEDED_V1 reason=CSV_RAW_BYTES "
        "limit=8 observed=9"
    )
    assert "/" not in str(captured.value)


def test_resource_error_has_exact_path_free_observed_at_least_message() -> None:
    with pytest.raises(ResourceLimitError) as captured:
        raise_input_resource_limit("CSV_RAW_BYTES", limit=8, observed_at_least=9)
    assert str(captured.value) == (
        "INPUT_RESOURCE_LIMIT_EXCEEDED_V1 reason=CSV_RAW_BYTES "
        "limit=8 observed_at_least=9"
    )
    assert "/" not in str(captured.value)


class _StringSubclass(str):
    pass


class _IntSubclass(int):
    pass


@pytest.mark.parametrize(
    ("reason", "limit", "observed", "observed_at_least"),
    [
        ("", 8, 9, None),
        (_StringSubclass("CSV_RAW_BYTES"), 8, 9, None),
        (1, 8, 9, None),
        ("CSV_RAW_BYTES", -1, 9, None),
        ("CSV_RAW_BYTES", True, 9, None),
        ("CSV_RAW_BYTES", _IntSubclass(8), 9, None),
        ("CSV_RAW_BYTES", 8, -1, None),
        ("CSV_RAW_BYTES", 8, True, None),
        ("CSV_RAW_BYTES", 8, _IntSubclass(9), None),
        ("CSV_RAW_BYTES", 8, None, None),
        ("CSV_RAW_BYTES", 8, 9, 10),
    ],
)
def test_resource_error_arguments_are_strictly_validated(
    reason: object,
    limit: object,
    observed: object,
    observed_at_least: object,
) -> None:
    with pytest.raises(TypeError):
        raise_input_resource_limit(
            reason,  # type: ignore[arg-type]
            limit=limit,  # type: ignore[arg-type]
            observed=observed,  # type: ignore[arg-type]
            observed_at_least=observed_at_least,  # type: ignore[arg-type]
        )


def test_regular_file_snapshot_reads_limit_minus_one_and_limit_bytes(tmp_path: Path) -> None:
    for size in (7, 8):
        path = tmp_path / f"exact-{size}.bin"
        payload = bytes(range(size))
        path.write_bytes(payload)
        assert read_regular_file_snapshot(path, raw_limit=8, reason="CSV_RAW_BYTES") == payload


def test_regular_file_snapshot_rejects_limit_plus_one_without_path_in_error(tmp_path: Path) -> None:
    path = tmp_path / "too-large.bin"
    path.write_bytes(bytes(range(9)))
    with pytest.raises(ResourceLimitError) as captured:
        read_regular_file_snapshot(path, raw_limit=8, reason="CSV_RAW_BYTES")
    assert str(captured.value) == (
        "INPUT_RESOURCE_LIMIT_EXCEEDED_V1 reason=CSV_RAW_BYTES "
        "limit=8 observed_at_least=9"
    )
    assert str(path) not in str(captured.value)


def test_regular_file_snapshot_preserves_missing_file_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_regular_file_snapshot(tmp_path / "missing.bin", raw_limit=8, reason="CSV_RAW_BYTES")


def test_regular_file_snapshot_rejects_directory_and_closes_descriptor(tmp_path: Path) -> None:
    path = tmp_path / "directory"
    path.mkdir()
    opened: list[int] = []
    real_open = input_resources._os_open

    def tracking_open(file: object, flags: int, mode: int = 0o777) -> int:
        descriptor = real_open(file, flags, mode)
        opened.append(descriptor)
        return descriptor

    original_fstat = input_resources._os_fstat
    input_resources._os_open = tracking_open
    try:
        with pytest.raises(ValueError, match="regular file"):
            read_regular_file_snapshot(path, raw_limit=8, reason="CSV_RAW_BYTES")
    finally:
        input_resources._os_open = real_open
    if os.name == "nt":
        # Windows refuses to open a directory as a file, so no descriptor exists to close.
        assert opened == []
        return
    assert len(opened) == 1
    with pytest.raises(OSError):
        original_fstat(opened[0])


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO is unavailable on this platform")
def test_fifo_is_rejected_without_blocking(tmp_path: Path) -> None:
    fifo = tmp_path / "input.fifo"
    os.mkfifo(fifo)
    source_root = Path(__file__).resolve().parents[1]
    script = (
        "from selcal.input_resources import read_regular_file_snapshot\n"
        f"read_regular_file_snapshot({str(fifo)!r}, raw_limit=8, reason='CSV_RAW_BYTES')\n"
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(source_root / "src"), environment.get("PYTHONPATH", "")]
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=source_root,
        env=environment,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=5,
    )
    assert result.returncode != 0
    assert "ValueError" in result.stderr


@pytest.mark.skipif(os.name == "nt", reason="POSIX replacement ctime semantics are required")
def test_snapshot_rejects_path_replacement_without_reading_replacement_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "input.bin"
    replacement = tmp_path / "replacement.bin"
    original_payload = b"original-bytes"
    replacement_payload = b"replacement-bytes"
    path.write_bytes(original_payload)
    replacement.write_bytes(replacement_payload)
    real_read = input_resources._os_read
    read_chunks: list[bytes] = []

    def replace_before_first_real_read(descriptor: int, size: int) -> bytes:
        if not read_chunks:
            os.replace(replacement, path)
        chunk = real_read(descriptor, size)
        read_chunks.append(chunk)
        return chunk

    monkeypatch.setattr(input_resources, "_os_read", replace_before_first_real_read)
    with pytest.raises(ValueError, match="changed while reading"):
        read_regular_file_snapshot(path, raw_limit=128, reason="CSV_RAW_BYTES")
    assert read_chunks[0] == original_payload
    assert replacement_payload not in read_chunks


@pytest.mark.skipif(os.name == "nt", reason="POSIX replacement ctime semantics are required")
def test_snapshot_rejects_content_change_even_after_mtime_restore_and_path_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "input.bin"
    replacement = tmp_path / "replacement.bin"
    original_payload = b"original-bytes"
    path.write_bytes(original_payload)
    replacement.write_bytes(b"replacement-bytes")
    original_stat = path.stat()
    real_read = input_resources._os_read
    mutated = False

    def mutate_restore_mtime_replace_path(descriptor: int, size: int) -> bytes:
        nonlocal mutated
        chunk = real_read(descriptor, size)
        if not mutated:
            mutated = True
            before_mutation = os.fstat(descriptor)
            second_descriptor = os.open(path, os.O_RDWR)
            try:
                os.lseek(second_descriptor, 0, os.SEEK_SET)
                os.write(second_descriptor, b"M")
                os.utime(
                    second_descriptor,
                    ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
                )
            finally:
                os.close(second_descriptor)
            os.replace(replacement, path)
            after_mutation = os.fstat(descriptor)
            assert after_mutation.st_size == before_mutation.st_size
            assert after_mutation.st_mtime_ns == before_mutation.st_mtime_ns
            assert after_mutation.st_ctime_ns != before_mutation.st_ctime_ns
        return chunk

    monkeypatch.setattr(input_resources, "_os_read", mutate_restore_mtime_replace_path)
    with pytest.raises(ValueError, match="changed while reading"):
        read_regular_file_snapshot(path, raw_limit=128, reason="CSV_RAW_BYTES")


def test_snapshot_rejects_detectable_in_place_mutation_after_first_real_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "mutable.bin"
    path.write_bytes(b"stable-bytes")
    real_read = input_resources._os_read
    mutated = False

    def mutate_after_first_real_read(descriptor: int, size: int) -> bytes:
        nonlocal mutated
        chunk = real_read(descriptor, size)
        if not mutated:
            mutated = True
            second_descriptor = os.open(path, os.O_RDWR)
            try:
                os.lseek(second_descriptor, 0, os.SEEK_END)
                os.write(second_descriptor, b"X")
            finally:
                os.close(second_descriptor)
        return chunk

    monkeypatch.setattr(input_resources, "_os_read", mutate_after_first_real_read)
    with pytest.raises(ValueError, match="changed while reading"):
        read_regular_file_snapshot(path, raw_limit=128, reason="CSV_RAW_BYTES")


@pytest.mark.skipif(not hasattr(os, "fchmod"), reason="fchmod is unavailable on this platform")
@pytest.mark.skipif(os.name == "nt", reason="POSIX ctime semantics are required")
def test_snapshot_rejects_ctime_only_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "ctime-only.bin"
    path.write_bytes(b"stable-bytes")
    real_read = input_resources._os_read
    changed = False

    def chmod_after_first_real_read(descriptor: int, size: int) -> bytes:
        nonlocal changed
        chunk = real_read(descriptor, size)
        if not changed:
            changed = True
            second_descriptor = os.open(path, os.O_RDWR)
            try:
                before = os.fstat(second_descriptor)
                mode = before.st_mode
                os.fchmod(second_descriptor, mode ^ stat.S_IXUSR)
                after = os.fstat(second_descriptor)
                assert after.st_size == before.st_size
                assert after.st_mtime_ns == before.st_mtime_ns
                assert after.st_ctime_ns != before.st_ctime_ns
            finally:
                os.close(second_descriptor)
        return chunk

    monkeypatch.setattr(input_resources, "_os_read", chmod_after_first_real_read)
    with pytest.raises(ValueError, match="changed while reading"):
        read_regular_file_snapshot(path, raw_limit=128, reason="CSV_RAW_BYTES")


def test_snapshot_stops_after_one_read_at_raw_limit_plus_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "far-over-limit.bin"
    path.write_bytes(b"x" * 100)
    real_read = input_resources._os_read
    requests: list[int] = []

    def recording_read(descriptor: int, size: int) -> bytes:
        requests.append(size)
        return real_read(descriptor, size)

    monkeypatch.setattr(input_resources, "_os_read", recording_read)
    with pytest.raises(ResourceLimitError):
        read_regular_file_snapshot(path, raw_limit=8, reason="CSV_RAW_BYTES")
    assert requests == [9]


@pytest.mark.parametrize(
    ("raw_limit", "reason"),
    [
        (-1, "CSV_RAW_BYTES"),
        (True, "CSV_RAW_BYTES"),
        (1.0, "CSV_RAW_BYTES"),
        (8, ""),
        (8, _StringSubclass("CSV_RAW_BYTES")),
        (8, 1),
    ],
)
def test_regular_file_snapshot_arguments_are_strictly_validated(
    tmp_path: Path, raw_limit: object, reason: object
) -> None:
    path = tmp_path / "input.bin"
    path.write_bytes(b"payload")
    with pytest.raises(TypeError):
        read_regular_file_snapshot(
            path,
            raw_limit=raw_limit,  # type: ignore[arg-type]
            reason=reason,  # type: ignore[arg-type]
        )


def test_regular_file_snapshot_opens_path_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "input.bin"
    path.write_bytes(b"payload")
    real_open = input_resources._os_open
    calls = 0

    def counting_open(file: object, flags: int, mode: int = 0o777) -> int:
        nonlocal calls
        calls += 1
        return real_open(file, flags, mode)

    monkeypatch.setattr(input_resources, "_os_open", counting_open)
    assert read_regular_file_snapshot(path, raw_limit=128, reason="CSV_RAW_BYTES") == b"payload"
    assert calls == 1


@pytest.mark.parametrize("error_type", [PermissionError, MemoryError])
def test_snapshot_preserves_read_exception_identity_and_closes_fd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[Exception],
) -> None:
    path = tmp_path / "read-error.bin"
    path.write_bytes(b"payload")
    expected = error_type("sentinel")
    real_open = input_resources._os_open
    real_read = input_resources._os_read
    opened: list[int] = []

    def tracking_open(file: object, flags: int) -> int:
        descriptor = real_open(file, flags)
        opened.append(descriptor)
        return descriptor

    def raise_after_real_read(descriptor: int, size: int) -> bytes:
        real_read(descriptor, size)
        raise expected

    monkeypatch.setattr(input_resources, "_os_open", tracking_open)
    monkeypatch.setattr(input_resources, "_os_read", raise_after_real_read)
    with pytest.raises(error_type) as captured:
        read_regular_file_snapshot(path, raw_limit=128, reason="CSV_RAW_BYTES")
    assert captured.value is expected
    assert len(opened) == 1
    with pytest.raises(OSError):
        os.fstat(opened[0])


def test_snapshot_closes_fd_on_resource_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "resource-error.bin"
    path.write_bytes(b"0123456789")
    real_open = input_resources._os_open
    opened: list[int] = []

    def tracking_open(file: object, flags: int) -> int:
        descriptor = real_open(file, flags)
        opened.append(descriptor)
        return descriptor

    monkeypatch.setattr(input_resources, "_os_open", tracking_open)
    with pytest.raises(ResourceLimitError):
        read_regular_file_snapshot(path, raw_limit=8, reason="CSV_RAW_BYTES")
    assert len(opened) == 1
    with pytest.raises(OSError):
        os.fstat(opened[0])


@pytest.mark.parametrize("observed_at_least", [-1, True, _IntSubclass(9)])
def test_resource_error_rejects_strictly_invalid_observed_at_least(
    observed_at_least: object,
) -> None:
    with pytest.raises(TypeError):
        raise_input_resource_limit(
            "CSV_RAW_BYTES",
            limit=8,
            observed_at_least=observed_at_least,  # type: ignore[arg-type]
        )


def test_snapshot_identity_uses_platform_change_time_contract() -> None:
    from types import SimpleNamespace

    first = SimpleNamespace(st_size=8, st_mtime_ns=11, st_ctime_ns=21)
    same_size_mtime = SimpleNamespace(st_size=8, st_mtime_ns=11, st_ctime_ns=22)
    if os.name == "nt":
        assert input_resources._snapshot_identity(first) == (8, 11)
        assert input_resources._snapshot_identity(first) == input_resources._snapshot_identity(
            same_size_mtime
        )
    else:
        assert input_resources._snapshot_identity(first) == (8, 11, 21)
        assert input_resources._snapshot_identity(first) != input_resources._snapshot_identity(
            same_size_mtime
        )


def test_snapshot_flags_include_read_only_and_available_optional_flags() -> None:
    expected = os.O_RDONLY
    for name in ("O_CLOEXEC", "O_NONBLOCK", "O_BINARY"):
        value = getattr(os, name, None)
        if type(value) is int:
            expected |= value
    assert input_resources._snapshot_flags() == expected


def test_snapshot_succeeds_across_multiple_one_mib_reads_with_exact_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chunk_size = 1 << 20
    payload = b"x" * (2 * chunk_size)
    path = tmp_path / "multi-chunk.bin"
    path.write_bytes(payload)
    real_read = input_resources._os_read
    requests: list[int] = []

    def recording_read(descriptor: int, size: int) -> bytes:
        requests.append(size)
        return real_read(descriptor, size)

    monkeypatch.setattr(input_resources, "_os_read", recording_read)
    assert (
        read_regular_file_snapshot(path, raw_limit=len(payload), reason="CSV_RAW_BYTES") == payload
    )
    assert requests == [chunk_size, chunk_size, 1]


def test_snapshot_second_chunk_reaches_limit_plus_one_and_stops_immediately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chunk_size = 1 << 20
    path = tmp_path / "multi-chunk-limit.bin"
    path.write_bytes(b"x" * (chunk_size + 9))
    real_read = input_resources._os_read
    requests: list[int] = []

    def recording_read(descriptor: int, size: int) -> bytes:
        requests.append(size)
        return real_read(descriptor, size)

    monkeypatch.setattr(input_resources, "_os_read", recording_read)
    with pytest.raises(ResourceLimitError) as captured:
        read_regular_file_snapshot(
            path,
            raw_limit=chunk_size + 8,
            reason="CSV_RAW_BYTES",
        )
    assert str(captured.value).endswith(
        f"limit={chunk_size + 8} observed_at_least={chunk_size + 9}"
    )
    assert requests == [chunk_size, 9]
