"""Strict, provenance-preserving input loaders for SelCal."""

from __future__ import annotations

import csv
import io
import re
from array import array
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from os import PathLike
from typing import ClassVar, Literal

import numpy as np

from selcal.canonical import raw_input_sha256, semantic_input_sha256
from selcal.contracts import SeriesPair, canonical_candidates
from selcal.input_resources import (
    INPUT_LIMITS_V1,
    raise_input_resource_limit,
    read_regular_file_snapshot,
)
from selcal.npz_safe import load_bounded_npz_arrays

_text_io_wrapper = io.TextIOWrapper


@dataclass(frozen=True, slots=True, eq=False)
class LoadedInput:
    """A path-free input value with byte-level and scientific identities."""

    __hash__: ClassVar[None] = None  # type: ignore[assignment]

    pair: SeriesPair
    raw_input_sha256: str
    semantic_input_sha256: str
    source_format: Literal["csv", "npz"]

    def __post_init__(self) -> None:
        if not isinstance(self.pair, SeriesPair):
            raise TypeError("pair must be a SeriesPair")
        for field_name, value in (
            ("raw_input_sha256", self.raw_input_sha256),
            ("semantic_input_sha256", self.semantic_input_sha256),
        ):
            if type(value) is not str:
                raise TypeError(f"{field_name} must be an actual built-in string")
            if re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError(f"{field_name} must be lowercase 64-character SHA-256 hex")
        if type(self.source_format) is not str:
            raise TypeError("source_format must be an actual built-in string")
        if self.source_format not in {"csv", "npz"}:
            raise ValueError("source_format must be exactly 'csv' or 'npz'")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LoadedInput):
            return NotImplemented
        return bool(
            self.pair == other.pair
            and self.raw_input_sha256 == other.raw_input_sha256
            and self.semantic_input_sha256 == other.semantic_input_sha256
            and self.source_format == other.source_format
        )


def _actual_nonempty_string(value: object, *, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be an actual built-in string")
    if not value:
        raise ValueError(f"{name} must be nonempty")
    return value


def _require_supported_length(pair: SeriesPair, candidates: tuple[int, ...]) -> None:
    maximum_lag = max(candidates)
    if pair.source.size <= maximum_lag:
        raise ValueError("series length must be greater than the maximum candidate lag")


def _loaded_input(
    pair: SeriesPair,
    *,
    raw_payload: bytes,
    candidates: tuple[int, ...],
    source_format: Literal["csv", "npz"],
) -> LoadedInput:
    _require_supported_length(pair, candidates)
    return LoadedInput(
        pair=pair,
        raw_input_sha256=raw_input_sha256(raw_payload),
        semantic_input_sha256=semantic_input_sha256(pair, candidates),
        source_format=source_format,
    )


def _csv_error(row_number: int, message: str) -> ValueError:
    return ValueError(f"CSV {message} at physical row {row_number}")


def _read_csv_physical_record(stream: io.TextIOBase) -> str | None:
    """Read at most one bounded physical-record prefix."""

    record = stream.readline(INPUT_LIMITS_V1.csv_record_characters + 1)
    if record == "":
        return None
    if record.endswith("\n"):
        record = record[:-1]
    return record


def _validate_csv_record(record: str) -> str:
    """Apply the fixed Unicode-code-point bound to one physical record."""

    if len(record) > INPUT_LIMITS_V1.csv_record_characters:
        raise_input_resource_limit(
            "CSV_RECORD_CHARACTERS",
            limit=INPUT_LIMITS_V1.csv_record_characters,
            observed_at_least=INPUT_LIMITS_V1.csv_record_characters + 1,
        )
    return record


def _parse_csv_record(record: str, *, row_number: int) -> tuple[str, ...]:
    """Parse exactly one already-bounded physical CSV record."""

    try:
        reader = csv.reader((record,), strict=True)
        fields = next(reader)
        if next(reader, None) is not None:
            raise csv.Error("record yielded more than one row")
    except csv.Error as error:
        raise _csv_error(row_number, "row cannot be parsed") from error
    return tuple(fields)


def _parse_csv_numeric(value: str, *, row_number: int) -> float:
    try:
        numeric_value = float(value)
    except ValueError as error:
        raise _csv_error(row_number, "row contains a nonnumeric cell") from error
    if not np.isfinite(numeric_value):
        raise _csv_error(row_number, "row contains a nonfinite cell")
    if numeric_value == 0.0:
        try:
            decimal_is_zero = Decimal(value).is_zero()
        except InvalidOperation as error:
            raise _csv_error(
                row_number,
                "row contains a numeric cell outside the supported decimal range",
            ) from error
        if not decimal_is_zero:
            raise _csv_error(row_number, "row contains a nonzero value that underflows float64")
    return numeric_value


def load_csv(
    path: str | PathLike[str],
    *,
    source_column: str,
    target_column: str,
    candidates: tuple[int, ...],
) -> LoadedInput:
    """Load an exact two-column UTF-8 CSV without preprocessing observations.

    A leading UTF-8 byte-order mark (as written by spreadsheet programs) is ignored when
    parsing; the raw identity still hashes the file bytes as read, mark included.
    """

    source_name = _actual_nonempty_string(source_column, name="source_column")
    target_name = _actual_nonempty_string(target_column, name="target_column")
    if source_name == target_name:
        raise ValueError("source_column and target_column must be distinct")
    canonical = canonical_candidates(candidates)

    raw_payload = read_regular_file_snapshot(
        path,
        raw_limit=INPUT_LIMITS_V1.csv_raw_bytes,
        reason="CSV_RAW_BYTES",
    )
    text_payload = raw_payload[3:] if raw_payload.startswith(b"\xef\xbb\xbf") else raw_payload
    with _text_io_wrapper(
        io.BytesIO(text_payload), encoding="utf-8", errors="strict", newline=None
    ) as stream:
        header_record = _read_csv_physical_record(stream)
        if header_record is None:
            raise _csv_error(1, "header must contain exactly two nonempty unique names")
        header_record = _validate_csv_record(header_record)
        try:
            headers = _parse_csv_record(header_record, row_number=1)
        except ValueError as error:
            raise _csv_error(1, "header cannot be parsed") from error
        if (
            len(headers) != 2
            or any(not header for header in headers)
            or len(set(headers)) != 2
            or any(len(header) > INPUT_LIMITS_V1.csv_field_characters for header in headers)
        ):
            if any(len(header) > INPUT_LIMITS_V1.csv_field_characters for header in headers):
                raise_input_resource_limit(
                    "CSV_FIELD_CHARACTERS",
                    limit=INPUT_LIMITS_V1.csv_field_characters,
                    observed_at_least=INPUT_LIMITS_V1.csv_field_characters + 1,
                )
            raise _csv_error(1, "header must contain exactly two nonempty unique names")

        if {source_name, target_name} != set(headers):
            raise ValueError(
                "source_column and target_column must exactly match the CSV header set: "
                f"the file has columns {headers[0]!r} and {headers[1]!r}, the configuration "
                f"names {source_name!r} and {target_name!r}"
            )

        source_values = array("d")
        target_values = array("d")
        data_rows = 0
        while True:
            physical_record = _read_csv_physical_record(stream)
            if physical_record is None:
                break
            data_rows += 1
            if data_rows > INPUT_LIMITS_V1.csv_data_rows:
                raise_input_resource_limit(
                    "CSV_DATA_ROWS",
                    limit=INPUT_LIMITS_V1.csv_data_rows,
                    observed_at_least=INPUT_LIMITS_V1.csv_data_rows + 1,
                )
            row_number = data_rows + 1
            physical_record = _validate_csv_record(physical_record)
            if physical_record == "":
                raise _csv_error(row_number, "contains a blank row")
            fields = _parse_csv_record(physical_record, row_number=row_number)
            if len(fields) < 2:
                raise _csv_error(row_number, "row has missing columns")
            if len(fields) > 2:
                raise _csv_error(row_number, "row has extra columns")
            if any(len(value) > INPUT_LIMITS_V1.csv_field_characters for value in fields):
                raise_input_resource_limit(
                    "CSV_FIELD_CHARACTERS",
                    limit=INPUT_LIMITS_V1.csv_field_characters,
                    observed_at_least=INPUT_LIMITS_V1.csv_field_characters + 1,
                )
            if any(value == "" for value in fields):
                raise _csv_error(row_number, "row contains a blank cell")

            if fields == headers:
                raise _csv_error(row_number, "row repeats the header")

            first_value = _parse_csv_numeric(fields[0], row_number=row_number)
            second_value = _parse_csv_numeric(fields[1], row_number=row_number)

            if headers[0] == source_name:
                source_values.append(first_value)
                target_values.append(second_value)
            else:
                source_values.append(second_value)
                target_values.append(first_value)

    if not source_values:
        raise ValueError("CSV must contain at least one data row")

    pair = SeriesPair(
        source=np.asarray(source_values, dtype=np.float64),
        target=np.asarray(target_values, dtype=np.float64),
    )
    return _loaded_input(
        pair,
        raw_payload=raw_payload,
        candidates=canonical,
        source_format="csv",
    )


def load_npz(
    path: str | PathLike[str],
    *,
    candidates: tuple[int, ...],
) -> LoadedInput:
    """Load an exact source/target NPZ archive without preprocessing observations."""

    canonical = canonical_candidates(candidates)
    raw_payload = read_regular_file_snapshot(
        path,
        raw_limit=INPUT_LIMITS_V1.npz_raw_bytes,
        reason="NPZ_RAW_BYTES",
    )
    source, target = load_bounded_npz_arrays(raw_payload)

    for array_name, member_array in (("source", source), ("target", target)):
        if member_array.dtype.kind in {"i", "u"}:
            for value in member_array.flat:
                integer_value = int(value)
                float_value = float(integer_value)
                if not np.isfinite(float_value) or int(float_value) != integer_value:
                    raise ValueError(
                        f"NPZ {array_name} integers must convert losslessly to float64"
                    )

    try:
        pair = SeriesPair(source=source, target=target)
    except TypeError as error:
        raise ValueError("NPZ source and target must be numeric arrays") from error
    return _loaded_input(
        pair,
        raw_payload=raw_payload,
        candidates=canonical,
        source_format="npz",
    )
