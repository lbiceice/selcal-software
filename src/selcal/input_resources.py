"""Versioned resource limits and bounded regular-file snapshots."""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from typing import NoReturn

from selcal.contracts_v2 import ResourceLimitError

_os_open = os.open
_os_read = os.read
_os_fstat = os.fstat
_os_close = os.close
_O_RDONLY = os.O_RDONLY
_O_CLOEXEC = getattr(os, "O_CLOEXEC", None)
_O_NONBLOCK = getattr(os, "O_NONBLOCK", None)
_O_BINARY = getattr(os, "O_BINARY", None)


@dataclass(frozen=True, slots=True)
class InputLimitsV1:
    """The exact, non-overridable input resource limits for envelope v1."""

    csv_raw_bytes: int
    csv_record_characters: int
    csv_field_characters: int
    csv_data_rows: int
    npz_raw_bytes: int
    npz_central_directory_bytes: int
    npy_header_bytes: int
    npy_elements: int
    npz_member_uncompressed_bytes: int
    npz_total_uncompressed_bytes: int


INPUT_LIMITS_V1 = InputLimitsV1(
    csv_raw_bytes=67_108_864,
    csv_record_characters=1_024,
    csv_field_characters=256,
    csv_data_rows=1_000_000,
    npz_raw_bytes=33_554_432,
    npz_central_directory_bytes=16_384,
    npy_header_bytes=4_096,
    npy_elements=1_000_000,
    npz_member_uncompressed_bytes=8_004_108,
    npz_total_uncompressed_bytes=16_008_216,
)


def _validate_reason(reason: object) -> str:
    if type(reason) is not str or not reason:
        raise TypeError("reason must be a nonempty built-in string")
    return reason


def _validate_nonnegative_int(value: object, *, name: str) -> int:
    if type(value) is not int or value < 0:
        raise TypeError(f"{name} must be a nonnegative built-in integer")
    return value


def raise_input_resource_limit(
    reason: str,
    *,
    limit: int,
    observed: int | None = None,
    observed_at_least: int | None = None,
) -> NoReturn:
    """Raise the stable path-free v1 resource-limit error."""

    exact_reason = _validate_reason(reason)
    if (observed is None) == (observed_at_least is None):
        raise TypeError("exactly one resource observation must be supplied")
    value = observed if observed is not None else observed_at_least
    if type(limit) is not int or limit < 0 or type(value) is not int or value < 0:
        raise TypeError("resource limits and observations must be nonnegative built-in integers")
    exact_limit = limit
    exact_value = value
    key = "observed" if observed is not None else "observed_at_least"
    raise ResourceLimitError(
        "INPUT_RESOURCE_LIMIT_EXCEEDED_V1 "
        f"reason={exact_reason} limit={exact_limit} {key}={exact_value}"
    )


def _snapshot_flags() -> int:
    flags = _O_RDONLY
    for optional_flag in (_O_CLOEXEC, _O_NONBLOCK, _O_BINARY):
        if type(optional_flag) is int:
            flags |= optional_flag
    return flags


def _snapshot_identity(metadata: os.stat_result) -> tuple[int, ...]:
    """Return the platform-supported metadata identity for a file snapshot."""

    if os.name == "nt":
        return (metadata.st_size, metadata.st_mtime_ns)
    return (metadata.st_size, metadata.st_mtime_ns, metadata.st_ctime_ns)


def read_regular_file_snapshot(
    path: str | os.PathLike[str],
    *,
    raw_limit: int,
    reason: str,
) -> bytes:
    """Read one immutable bounded snapshot from one regular-file descriptor."""

    exact_reason = _validate_reason(reason)
    exact_limit = _validate_nonnegative_int(raw_limit, name="raw_limit")
    try:
        descriptor = _os_open(path, _snapshot_flags())
    except (IsADirectoryError, PermissionError):
        # Windows refuses to open a directory as a file with PermissionError; report it as the
        # same request error as on POSIX, where the descriptor check below catches it.
        if os.path.isdir(path):
            raise ValueError("input must be a regular file") from None
        raise
    try:
        before = _os_fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("input must be a regular file")

        chunks: list[bytes] = []
        total = 0
        maximum = exact_limit + 1
        while total < maximum:
            chunk = _os_read(descriptor, min(1 << 20, maximum - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total >= maximum:
                raise_input_resource_limit(
                    exact_reason,
                    limit=exact_limit,
                    observed_at_least=maximum,
                )

        after = _os_fstat(descriptor)
        if _snapshot_identity(before) != _snapshot_identity(after):
            raise ValueError("input changed while reading")
        return b"".join(chunks)
    finally:
        _os_close(descriptor)
