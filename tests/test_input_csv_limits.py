from __future__ import annotations

import csv
import inspect
import io
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

import selcal.inputs as inputs
from selcal.contracts_v2 import ResourceLimitError
from selcal.inputs import _read_csv_physical_record, _validate_csv_record, load_csv


@pytest.mark.parametrize("length", [1023, 1024])
def test_physical_record_validator_accepts_at_or_below_character_limit(length: int) -> None:
    stream = io.StringIO("x" * length + "\n")

    record = _read_csv_physical_record(stream)
    assert record is not None
    assert _validate_csv_record(record) == "x" * length


def test_physical_record_validator_rejects_1025_unicode_code_points() -> None:
    stream = io.StringIO("x" * 1025 + "\n")

    with pytest.raises(ResourceLimitError, match=(
        r"INPUT_RESOURCE_LIMIT_EXCEEDED_V1 reason=CSV_RECORD_CHARACTERS "
        r"limit=1024 observed_at_least=1025"
    )):
        record = _read_csv_physical_record(stream)
        assert record is not None
        _validate_csv_record(record)


def test_record_limit_precedes_csv_structure_for_comma_bomb() -> None:
    stream = io.StringIO("," * 1025 + "\n")

    with pytest.raises(ResourceLimitError, match=(
        r"INPUT_RESOURCE_LIMIT_EXCEEDED_V1 reason=CSV_RECORD_CHARACTERS "
        r"limit=1024 observed_at_least=1025"
    )):
        record = _read_csv_physical_record(stream)
        assert record is not None
        _validate_csv_record(record)


def test_load_csv_public_signature_has_no_limits_override() -> None:
    signature = inspect.signature(load_csv)

    assert tuple(signature.parameters) == (
        "path",
        "source_column",
        "target_column",
        "candidates",
    )
    assert all(parameter.kind is not inspect.Parameter.VAR_KEYWORD
               for parameter in signature.parameters.values())


def _load(path: Path, *, source: str = "source", target: str = "target") -> inputs.LoadedInput:
    return load_csv(path, source_column=source, target_column=target, candidates=(1,))


def test_header_and_data_fields_accept_exactly_256_code_points(tmp_path: Path) -> None:
    source = "s" * 256
    path = tmp_path / "fields.csv"
    path.write_text(f"{source},target\n{'0' * 256},1\n0,1\n", encoding="utf-8", newline="\n")

    loaded = _load(path, source=source)

    np.testing.assert_array_equal(loaded.pair.source, np.array([0.0, 0.0]))


@pytest.mark.parametrize(
    "payload",
    [
        b"s" * 257 + b",target\n0,1\n0,1\n",
        b"source,target\n" + b"0" * 257 + b",1\n0,1\n",
    ],
)
def test_header_or_data_field_257_is_rejected_before_numeric_conversion(
    tmp_path: Path, payload: bytes
) -> None:
    path = tmp_path / "field-too-long.csv"
    path.write_bytes(payload)

    with pytest.raises(ResourceLimitError, match=(
        r"INPUT_RESOURCE_LIMIT_EXCEEDED_V1 reason=CSV_FIELD_CHARACTERS "
        r"limit=256 observed_at_least=257"
    )):
        _load(path, source="s" * 257 if payload.startswith(b"s" * 257) else "source")


@pytest.mark.parametrize("terminator", [b"\n", b"\r\n", b"\r", b""])
def test_lf_crlf_cr_and_no_final_terminator_have_equivalent_values(
    tmp_path: Path, terminator: bytes
) -> None:
    path = tmp_path / "newline.csv"
    separator = terminator or b"\n"
    final = terminator
    path.write_bytes(b"source,target" + separator + b"1,10" + separator + b"2,20" + final)

    loaded = _load(path)

    np.testing.assert_array_equal(loaded.pair.source, np.array([1.0, 2.0]))
    np.testing.assert_array_equal(loaded.pair.target, np.array([10.0, 20.0]))


@pytest.mark.parametrize("terminator", [b"\n", b"\r\n", b"\r", b""])
@pytest.mark.parametrize("length", [1024, 1025])
def test_record_character_boundary_is_independent_of_terminator(
    tmp_path: Path, terminator: bytes, length: int
) -> None:
    # Test the physical-record envelope independently of the narrower field envelope.
    source = b"x" * length
    path = tmp_path / f"record-{length}-{len(terminator)}.csv"
    path.write_bytes(source + terminator)
    stream = io.TextIOWrapper(io.BytesIO(path.read_bytes()), encoding="utf-8", newline=None)
    record = _read_csv_physical_record(stream)
    assert record is not None

    if length == 1024:
        assert _validate_csv_record(record) == "x" * 1024
    else:
        with pytest.raises(ResourceLimitError, match="CSV_RECORD_CHARACTERS"):
            _validate_csv_record(record)


@pytest.mark.parametrize("length", [1024, 1025])
def test_record_character_boundary_counts_unicode_code_points_not_bytes(length: int) -> None:
    stream = io.TextIOWrapper(
        io.BytesIO(("é" * length + "\n").encode("utf-8")),
        encoding="utf-8",
        errors="strict",
        newline=None,
    )
    record = _read_csv_physical_record(stream)
    assert record is not None

    if length == 1024:
        assert _validate_csv_record(record) == "é" * length
    else:
        with pytest.raises(ResourceLimitError, match="CSV_RECORD_CHARACTERS"):
            _validate_csv_record(record)


class _RecordingStringIO(io.StringIO):
    def __init__(self, value: str) -> None:
        super().__init__(value)
        self.requested_sizes: list[int] = []
        self.returned_lengths: list[int] = []

    def readline(self, size: int = -1) -> str:
        self.requested_sizes.append(size)
        value = super().readline(size)
        self.returned_lengths.append(len(value))
        return value


def test_large_record_reads_once_and_validator_rejects_without_consuming_tail() -> None:
    stream = _RecordingStringIO("," * 5_000_000 + "\n")

    record = _read_csv_physical_record(stream)
    assert record is not None
    assert stream.requested_sizes == [1025]
    assert stream.returned_lengths == [1025]
    assert stream.tell() == 1025
    with pytest.raises(ResourceLimitError, match="CSV_RECORD_CHARACTERS"):
        _validate_csv_record(record)


def test_quoted_embedded_newline_is_rejected_as_physical_records(tmp_path: Path) -> None:
    path = tmp_path / "embedded-newline.csv"
    path.write_bytes(b"source,target\n1,\"2\n3\"\n4,5\n")

    with pytest.raises(ValueError, match=r"row 2"):
        _load(path)


def test_public_csv_malformed_header_uses_header_exception_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "malformed-header.csv"
    path.write_bytes(b'source,"target\n1,2\n3,4\n')
    real_parser = inputs._parse_csv_record
    parser_calls: list[tuple[str, int]] = []

    def observe_parser(record: str, *, row_number: int) -> list[str]:
        parser_calls.append((record, row_number))
        return real_parser(record, row_number=row_number)

    monkeypatch.setattr(inputs, "_parse_csv_record", observe_parser)

    with pytest.raises(ValueError, match="CSV header cannot be parsed") as captured:
        _load(path)

    assert parser_calls == [('source,"target', 1)]
    parser_error = captured.value.__cause__
    assert isinstance(parser_error, ValueError)
    assert str(parser_error) == "CSV row cannot be parsed at physical row 1"
    assert isinstance(parser_error.__cause__, csv.Error)
    assert str(parser_error.__cause__) == "unexpected end of data"


@pytest.mark.parametrize("last_record", [b"\n", b"x" * 1025 + b",2\n"])
def test_data_row_limit_precedes_blank_or_record_validation(
    tmp_path: Path, last_record: bytes
) -> None:
    path = tmp_path / "rows.csv"
    path.write_bytes(b"source,target\n" + b"1,2\n" * 1_000_000 + last_record)

    with pytest.raises(ResourceLimitError, match=(
        r"INPUT_RESOURCE_LIMIT_EXCEEDED_V1 reason=CSV_DATA_ROWS "
        r"limit=1000000 observed_at_least=1000001"
    )):
        _load(path)


def test_data_row_limit_does_not_read_a_multi_megabyte_tail_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "rows-comma-bomb.csv"
    path.write_bytes(b"source,target\n" + b"1,2\n" * 1_000_000 + b"," * 5_000_000 + b"\n")
    real_reader = inputs._read_csv_physical_record
    returned_lengths: list[int] = []
    calls = 0

    def record_reads(stream: io.TextIOBase) -> str | None:
        nonlocal calls
        calls += 1
        record = real_reader(stream)
        if record is not None:
            returned_lengths.append(len(record))
        return record

    monkeypatch.setattr(inputs, "_read_csv_physical_record", record_reads)
    with pytest.raises(ResourceLimitError, match="CSV_DATA_ROWS"):
        _load(path)

    assert calls == 1_000_001 + 1
    assert returned_lengths[-1] <= 1025


def test_extreme_decimal_exponent_maps_to_stable_row_value_error(tmp_path: Path) -> None:
    path = tmp_path / "decimal-range.csv"
    path.write_bytes(
        b"source,target\n1,2\n"
        b"1e-99999999999999999999999999999999999999999999999999,3\n"
        b"4,5\n"
    )

    with pytest.raises(ValueError, match=(
        r"CSV row contains a numeric cell outside the supported decimal range "
        r"at physical row 3"
    )) as captured:
        _load(path)
    assert "InvalidOperation" not in str(captured.value)


class _TrackingTextIOWrapper(io.TextIOWrapper):
    instances: ClassVar[list[_TrackingTextIOWrapper]] = []

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.close_calls = 0
        self.instances.append(self)

    def close(self) -> None:
        self.close_calls += 1
        super().close()


@pytest.mark.parametrize("payload", [b"source,target\n1,2\n2,3\n", b"source,target\n1,\"2\n"])
def test_csv_text_wrapper_is_closed_on_success_and_parse_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, payload: bytes
) -> None:
    path = tmp_path / "close.csv"
    path.write_bytes(payload)
    _TrackingTextIOWrapper.instances.clear()
    monkeypatch.setattr(inputs, "_text_io_wrapper", _TrackingTextIOWrapper, raising=False)

    if b'"' in payload:
        with pytest.raises(ValueError):
            _load(path)
    else:
        _load(path)

    assert len(_TrackingTextIOWrapper.instances) == 1
    assert _TrackingTextIOWrapper.instances[0].close_calls == 1
    assert _TrackingTextIOWrapper.instances[0].closed


def test_csv_snapshot_is_called_with_fixed_raw_cap_and_no_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "snapshot.csv"
    path.write_bytes(b"source,target\n1,2\n2,3\n")
    calls: list[tuple[object, int, str]] = []
    real = inputs.read_regular_file_snapshot

    def capture(path_arg: object, *, raw_limit: int, reason: str) -> bytes:
        calls.append((path_arg, raw_limit, reason))
        return real(path_arg, raw_limit=raw_limit, reason=reason)  # type: ignore[arg-type]

    monkeypatch.setattr(inputs, "read_regular_file_snapshot", capture)
    _load(path)

    assert calls == [(path, 67_108_864, "CSV_RAW_BYTES")]


def test_csv_propagates_memory_error_and_strict_utf8_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "errors.csv"
    path.write_bytes(b"source,target\n1,2\n2,3\n")
    real_snapshot = inputs.read_regular_file_snapshot

    def exhausted(*args: object, **kwargs: object) -> bytes:
        del args, kwargs
        raise MemoryError("simulated exhaustion")

    monkeypatch.setattr(inputs, "read_regular_file_snapshot", exhausted)
    with pytest.raises(MemoryError, match="simulated exhaustion"):
        _load(path)

    path.write_bytes(b"source,target\n1,2\n\xff,3\n")
    monkeypatch.setattr(inputs, "read_regular_file_snapshot", real_snapshot)
    with pytest.raises(UnicodeDecodeError):
        _load(path)
