"""Fail-closed, bounded NPZ and NPY parsing for SelCal inputs."""

from __future__ import annotations

import io
import stat
import struct
import zlib
from dataclasses import dataclass
from zipfile import BadZipFile as _BadZipFile

import numpy as np

from selcal.contracts_v2 import ResourceLimitError
from selcal.input_resources import INPUT_LIMITS_V1, raise_input_resource_limit

_EOCD_SIGNATURE = b"PK\x05\x06"
_ZIP64_LOCATOR_SIGNATURE = b"PK\x06\x07"
_CENTRAL_SIGNATURE = b"PK\x01\x02"
_LOCAL_SIGNATURE = b"PK\x03\x04"
_ZIP64_EXTRA_ID = 0x0001
_ZIP_METHOD_STORED = 0
_ZIP_METHOD_DEFLATED = 8
_UINT16_SENTINEL = 0xFFFF
_UINT32_SENTINEL = 0xFFFFFFFF
_EOCD_SEARCH_BYTES = 65_557

_EOCD = struct.Struct("<4s4H2LH")
_CENTRAL = struct.Struct("<4s6H3L5H2L")
_LOCAL = struct.Struct("<4s5H3L2H")
_EXTRA_HEADER = struct.Struct("<HH")
_UINT64 = struct.Struct("<Q")


@dataclass(frozen=True, slots=True)
class _MemberMetadata:
    logical_name: str
    archive_name: str
    archive_name_bytes: bytes
    flags: int
    method: int
    crc32: int
    compressed_size: int
    uncompressed_size: int
    local_offset: int
    data_offset: int
    data_end: int
    local_zip64: bool


@dataclass(frozen=True, slots=True)
class _ArchiveMetadata:
    central_offset: int
    central_size: int
    eocd_offset: int
    members: tuple[_MemberMetadata, _MemberMetadata]


def _malformed() -> ValueError:
    return ValueError("NPZ archive is malformed")


def _keys_malformed() -> ValueError:
    return ValueError("NPZ keys must be unique and exactly {'source', 'target'}")


def _find_eocd_offset(raw_payload: bytes) -> int:
    """Find the unique end-anchored EOCD candidate in the bounded ZIP search window."""

    search_start = max(0, len(raw_payload) - _EOCD_SEARCH_BYTES)
    offsets: list[int] = []
    cursor = search_start
    while True:
        candidate = raw_payload.find(_EOCD_SIGNATURE, cursor)
        if candidate < 0:
            break
        cursor = candidate + 1
        if candidate + _EOCD.size > len(raw_payload):
            continue
        try:
            fields = _EOCD.unpack_from(raw_payload, candidate)
        except struct.error:
            continue
        if candidate + _EOCD.size + fields[7] == len(raw_payload):
            offsets.append(candidate)
    if len(offsets) != 1:
        raise _malformed()
    return offsets[0]


def _enforce_central_directory_limit(size: int) -> None:
    if size > INPUT_LIMITS_V1.npz_central_directory_bytes:
        raise_input_resource_limit(
            "NPZ_CENTRAL_DIRECTORY_BYTES",
            limit=INPUT_LIMITS_V1.npz_central_directory_bytes,
            observed=size,
        )


def _enforce_declared_size_limits(sizes: tuple[int, int]) -> None:
    total = sum(sizes)
    if total > INPUT_LIMITS_V1.npz_total_uncompressed_bytes:
        raise_input_resource_limit(
            "NPZ_TOTAL_UNCOMPRESSED_BYTES",
            limit=INPUT_LIMITS_V1.npz_total_uncompressed_bytes,
            observed=total,
        )
    for size in sizes:
        if size > INPUT_LIMITS_V1.npz_member_uncompressed_bytes:
            raise_input_resource_limit(
                "NPZ_MEMBER_UNCOMPRESSED_BYTES",
                limit=INPUT_LIMITS_V1.npz_member_uncompressed_bytes,
                observed=size,
            )


def _parse_extra_fields(extra: bytes) -> dict[int, bytes]:
    fields: dict[int, bytes] = {}
    cursor = 0
    while cursor < len(extra):
        if len(extra) - cursor < _EXTRA_HEADER.size:
            raise _malformed()
        field_id, payload_size = _EXTRA_HEADER.unpack_from(extra, cursor)
        cursor += _EXTRA_HEADER.size
        end = cursor + payload_size
        if end > len(extra) or field_id in fields:
            raise _malformed()
        fields[field_id] = extra[cursor:end]
        cursor = end
    if cursor != len(extra):
        raise _malformed()
    return fields


def _logical_member_name(name: bytes) -> tuple[str, str]:
    allowed = {
        b"source": ("source", "source"),
        b"source.npy": ("source", "source.npy"),
        b"target": ("target", "target"),
        b"target.npy": ("target", "target.npy"),
    }
    try:
        return allowed[name]
    except KeyError as error:
        raise _malformed() from error


def _local_zip64_sizes(
    *,
    local_compressed: int,
    local_uncompressed: int,
    central_compressed: int,
    central_uncompressed: int,
    extra_fields: dict[int, bytes],
) -> bool:
    compressed_sentinel = local_compressed == _UINT32_SENTINEL
    uncompressed_sentinel = local_uncompressed == _UINT32_SENTINEL
    zip64_payload = extra_fields.get(_ZIP64_EXTRA_ID)
    if not compressed_sentinel and not uncompressed_sentinel:
        if zip64_payload is not None:
            raise _malformed()
        if (
            local_compressed != central_compressed
            or local_uncompressed != central_uncompressed
        ):
            raise _malformed()
        return False

    if zip64_payload is None:
        raise _malformed()
    expected: list[int] = []
    if uncompressed_sentinel:
        expected.append(central_uncompressed)
    elif local_uncompressed != central_uncompressed:
        raise _malformed()
    if compressed_sentinel:
        expected.append(central_compressed)
    elif local_compressed != central_compressed:
        raise _malformed()
    if len(zip64_payload) != len(expected) * _UINT64.size:
        raise _malformed()
    observed = [
        _UINT64.unpack_from(zip64_payload, offset)[0]
        for offset in range(0, len(zip64_payload), _UINT64.size)
    ]
    if observed != expected:
        raise _malformed()
    return True


def _parse_local_member(
    raw_payload: bytes,
    *,
    central_offset: int,
    archive_name_bytes: bytes,
    logical_name: str,
    archive_name: str,
    flags: int,
    method: int,
    crc32: int,
    compressed_size: int,
    uncompressed_size: int,
    local_offset: int,
) -> _MemberMetadata:
    if local_offset < 0 or local_offset + _LOCAL.size > central_offset:
        raise _malformed()
    try:
        local = _LOCAL.unpack_from(raw_payload, local_offset)
    except struct.error as error:
        raise _malformed() from error
    if local[0] != _LOCAL_SIGNATURE:
        raise _malformed()
    local_flags = local[2]
    local_method = local[3]
    local_crc32 = local[6]
    local_compressed = local[7]
    local_uncompressed = local[8]
    name_length = local[9]
    extra_length = local[10]
    variable_start = local_offset + _LOCAL.size
    data_offset = variable_start + name_length + extra_length
    if data_offset > central_offset:
        raise _malformed()
    local_name = raw_payload[variable_start : variable_start + name_length]
    extra_start = variable_start + name_length
    local_extra = raw_payload[extra_start:data_offset]
    if len(local_name) != name_length or len(local_extra) != extra_length:
        raise _malformed()
    extra_fields = _parse_extra_fields(local_extra)
    if (
        local_name != archive_name_bytes
        or local_flags != flags
        or local_method != method
        or local_crc32 != crc32
    ):
        raise _malformed()
    local_zip64 = _local_zip64_sizes(
        local_compressed=local_compressed,
        local_uncompressed=local_uncompressed,
        central_compressed=compressed_size,
        central_uncompressed=uncompressed_size,
        extra_fields=extra_fields,
    )
    data_end = data_offset + compressed_size
    if data_end > central_offset:
        raise _malformed()
    return _MemberMetadata(
        logical_name=logical_name,
        archive_name=archive_name,
        archive_name_bytes=archive_name_bytes,
        flags=flags,
        method=method,
        crc32=crc32,
        compressed_size=compressed_size,
        uncompressed_size=uncompressed_size,
        local_offset=local_offset,
        data_offset=data_offset,
        data_end=data_end,
        local_zip64=local_zip64,
    )


def _preflight_npz_archive(raw_payload: bytes) -> _ArchiveMetadata:
    """Parse all security-relevant ZIP metadata before member extraction."""

    try:
        eocd_offset = _find_eocd_offset(raw_payload)
        eocd = _EOCD.unpack_from(raw_payload, eocd_offset)
        disk, central_disk, disk_entries, entries = eocd[1:5]
        central_size, central_offset = eocd[5:7]
        if (
            disk_entries == _UINT16_SENTINEL
            or entries == _UINT16_SENTINEL
            or central_size == _UINT32_SENTINEL
            or central_offset == _UINT32_SENTINEL
        ):
            raise _malformed()
        if disk != 0 or central_disk != 0:
            raise _malformed()
        if disk_entries != 2 or entries != 2:
            raise _keys_malformed()
        if (
            eocd_offset >= 20
            and raw_payload[eocd_offset - 20 : eocd_offset - 16]
            == _ZIP64_LOCATOR_SIGNATURE
        ):
            raise _malformed()
        if central_offset + central_size != eocd_offset:
            raise _malformed()
        _enforce_central_directory_limit(central_size)

        members: list[_MemberMetadata] = []
        logical_names: set[str] = set()
        cursor = central_offset
        for _ in range(2):
            if cursor < 0 or cursor + _CENTRAL.size > eocd_offset:
                raise _malformed()
            central = _CENTRAL.unpack_from(raw_payload, cursor)
            if central[0] != _CENTRAL_SIGNATURE:
                raise _malformed()
            flags = central[3]
            method = central[4]
            crc32 = central[7]
            compressed_size = central[8]
            uncompressed_size = central[9]
            name_length = central[10]
            extra_length = central[11]
            comment_length = central[12]
            disk_start = central[13]
            external_attributes = central[15]
            local_offset = central[16]
            if flags & (0x0001 | 0x0008 | 0x0040 | 0x2000) or method not in {
                _ZIP_METHOD_STORED,
                _ZIP_METHOD_DEFLATED,
            }:
                raise _malformed()
            if external_attributes & 0x10 or stat.S_ISDIR(external_attributes >> 16):
                raise _malformed()
            if (
                compressed_size == _UINT32_SENTINEL
                or uncompressed_size == _UINT32_SENTINEL
                or local_offset == _UINT32_SENTINEL
                or disk_start == _UINT16_SENTINEL
                or disk_start != 0
            ):
                raise _malformed()
            variable_start = cursor + _CENTRAL.size
            record_end = variable_start + name_length + extra_length + comment_length
            if record_end > eocd_offset:
                raise _malformed()
            archive_name_bytes = raw_payload[variable_start : variable_start + name_length]
            extra_start = variable_start + name_length
            central_extra = raw_payload[extra_start : extra_start + extra_length]
            if len(archive_name_bytes) != name_length or len(central_extra) != extra_length:
                raise _malformed()
            extra_fields = _parse_extra_fields(central_extra)
            if _ZIP64_EXTRA_ID in extra_fields:
                raise _malformed()
            logical_name, archive_name = _logical_member_name(archive_name_bytes)
            if logical_name in logical_names:
                raise _keys_malformed()
            logical_names.add(logical_name)
            members.append(
                _parse_local_member(
                    raw_payload,
                    central_offset=central_offset,
                    archive_name_bytes=archive_name_bytes,
                    logical_name=logical_name,
                    archive_name=archive_name,
                    flags=flags,
                    method=method,
                    crc32=crc32,
                    compressed_size=compressed_size,
                    uncompressed_size=uncompressed_size,
                    local_offset=local_offset,
                )
            )
            cursor = record_end
        if cursor != eocd_offset or logical_names != {"source", "target"}:
            raise _keys_malformed()
        ordered_ranges = sorted((member.local_offset, member.data_end) for member in members)
        if ordered_ranges[0][1] > ordered_ranges[1][0]:
            raise _malformed()
        if any(start < 0 or end > central_offset for start, end in ordered_ranges):
            raise _malformed()
        ordered_members = tuple(sorted(members, key=lambda member: member.logical_name))
        return _ArchiveMetadata(
            central_offset=central_offset,
            central_size=central_size,
            eocd_offset=eocd_offset,
            members=(ordered_members[0], ordered_members[1]),
        )
    except (MemoryError, ResourceLimitError):
        raise
    except ValueError:
        raise
    except (EOFError, OSError, struct.error, RuntimeError) as error:
        raise _malformed() from error


class _NpyContractError(ValueError):
    pass


def _member_malformed() -> _NpyContractError:
    return _NpyContractError("NPZ array member is malformed")


def _shape_malformed() -> _NpyContractError:
    return _NpyContractError(
        "NPZ array member must be one-dimensional with a nonnegative built-in integer length"
    )


def _dtype_malformed(dtype: np.dtype[np.generic]) -> _NpyContractError:
    if dtype.hasobject:
        return _NpyContractError("NPZ object or pickled arrays are forbidden")
    if dtype.kind == "f" and dtype.itemsize > 8:
        return _NpyContractError(
            "NPZ array member floating dtype itemsize must be at most 8 bytes for float64"
        )
    return _NpyContractError(
        "NPZ array member must have a real numeric dtype "
        "(signed integer, unsigned integer, or real float)"
    )


def _preflight_npy_member(
    member_payload: bytes,
) -> tuple[tuple[int, ...], np.dtype[np.generic], int]:
    """Independently validate one bounded NPY buffer before array construction."""

    try:
        stream = io.BytesIO(member_payload)
        version = np.lib.format.read_magic(stream)
        if stream.tell() != 8 or version not in {(1, 0), (2, 0)}:
            raise _member_malformed()
        length_size = 2 if version == (1, 0) else 4
        length_end = 8 + length_size
        if len(member_payload) < length_end:
            raise _member_malformed()
        header_length = int.from_bytes(member_payload[8:length_end], "little")
        if header_length > INPUT_LIMITS_V1.npy_header_bytes:
            raise_input_resource_limit(
                "NPY_HEADER_BYTES",
                limit=INPUT_LIMITS_V1.npy_header_bytes,
                observed=header_length,
            )
        header_end = length_end + header_length
        stream.seek(8)
        if version == (1, 0):
            shape, _fortran_order, dtype = np.lib.format.read_array_header_1_0(
                stream,
                max_header_size=INPUT_LIMITS_V1.npy_header_bytes,
            )
        else:
            shape, _fortran_order, dtype = np.lib.format.read_array_header_2_0(
                stream,
                max_header_size=INPUT_LIMITS_V1.npy_header_bytes,
            )
        if stream.tell() != header_end:
            raise _member_malformed()
        if type(shape) is not tuple or len(shape) != 1:
            raise _shape_malformed()
        element_count = shape[0]
        if type(element_count) is not int or element_count < 0:
            raise _shape_malformed()
        if element_count > INPUT_LIMITS_V1.npy_elements:
            raise_input_resource_limit(
                "NPY_ELEMENTS",
                limit=INPUT_LIMITS_V1.npy_elements,
                observed=element_count,
            )
        if (
            not isinstance(dtype, np.dtype)
            or dtype.fields is not None
            or dtype.hasobject
            or dtype.kind not in {"i", "u", "f"}
            or dtype.itemsize > 8
        ):
            raise _dtype_malformed(dtype)
        if header_end + element_count * dtype.itemsize != len(member_payload):
            raise _member_malformed()
        return shape, dtype, header_end
    except (MemoryError, ResourceLimitError):
        raise
    except _NpyContractError:
        raise
    except (
        ValueError,
        EOFError,
        OSError,
        struct.error,
        SyntaxError,
        UnicodeError,
        RuntimeError,
    ) as error:
        raise _member_malformed() from error


def _load_npy_member(member_payload: bytes) -> np.ndarray:
    shape, dtype, _header_end = _preflight_npy_member(member_payload)
    try:
        loaded = np.load(
            io.BytesIO(member_payload),
            allow_pickle=False,
            max_header_size=INPUT_LIMITS_V1.npy_header_bytes,
        )
    except (MemoryError, ResourceLimitError):
        raise
    except (
        ValueError,
        EOFError,
        OSError,
        _BadZipFile,
        struct.error,
        RuntimeError,
    ) as error:
        raise _member_malformed() from error
    if (
        not isinstance(loaded, np.ndarray)
        or loaded.shape != shape
        or loaded.dtype != dtype
    ):
        raise _member_malformed()
    return loaded


def _raise_actual_member_limit() -> None:
    maximum = INPUT_LIMITS_V1.npz_member_uncompressed_bytes + 1
    raise_input_resource_limit(
        "NPZ_MEMBER_UNCOMPRESSED_BYTES",
        limit=INPUT_LIMITS_V1.npz_member_uncompressed_bytes,
        observed_at_least=maximum,
    )


def _verify_extracted_member(payload: bytes, member: _MemberMetadata) -> bytes:
    if len(payload) != member.uncompressed_size:
        raise _malformed()
    if zlib.crc32(payload) & 0xFFFFFFFF != member.crc32:
        raise _malformed()
    return payload


def _extract_stored_member(raw_payload: bytes, member: _MemberMetadata) -> bytes:
    actual_size = member.data_end - member.data_offset
    maximum = INPUT_LIMITS_V1.npz_member_uncompressed_bytes + 1
    if actual_size >= maximum:
        _raise_actual_member_limit()
    payload = raw_payload[member.data_offset : member.data_end]
    if len(payload) >= maximum:
        _raise_actual_member_limit()
    if (
        len(payload) != actual_size
        or member.compressed_size != actual_size
        or member.compressed_size != member.uncompressed_size
    ):
        raise _malformed()
    return _verify_extracted_member(payload, member)


def _extract_deflated_member(raw_payload: bytes, member: _MemberMetadata) -> bytes:
    decompressor = zlib.decompressobj(wbits=-15)
    maximum = INPUT_LIMITS_V1.npz_member_uncompressed_bytes + 1
    observed = 0
    output_chunks: list[bytes] = []
    cursor = member.data_offset
    pending = b""

    while pending or cursor < member.data_end:
        if not pending:
            chunk_end = min(cursor + 65_536, member.data_end)
            pending = raw_payload[cursor:chunk_end]
            cursor = chunk_end
            if not pending:
                raise _malformed()

        offered = pending
        remaining = maximum - observed
        if remaining <= 0:
            _raise_actual_member_limit()
        output = decompressor.decompress(offered, remaining)
        tail = decompressor.unconsumed_tail
        if output:
            output_chunks.append(output)
            observed += len(output)
            if observed >= maximum:
                _raise_actual_member_limit()
        if decompressor.eof:
            if decompressor.unused_data or tail or cursor != member.data_end:
                raise _malformed()
            pending = b""
            break
        consumed = len(offered) - len(tail)
        if consumed <= 0 and not output:
            raise _malformed()
        pending = tail

    if (
        cursor != member.data_end
        or pending
        or not decompressor.eof
        or decompressor.unused_data
        or decompressor.unconsumed_tail
    ):
        raise _malformed()
    return _verify_extracted_member(b"".join(output_chunks), member)


def _extract_member_payload(raw_payload: bytes, member: _MemberMetadata) -> bytes:
    if member.method == _ZIP_METHOD_STORED:
        return _extract_stored_member(raw_payload, member)
    if member.method == _ZIP_METHOD_DEFLATED:
        return _extract_deflated_member(raw_payload, member)
    raise _malformed()


def load_bounded_npz_arrays(raw_payload: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Return source and target arrays from one preflighted immutable NPZ snapshot."""

    try:
        metadata = _preflight_npz_archive(raw_payload)
        _enforce_declared_size_limits(
            (metadata.members[0].uncompressed_size, metadata.members[1].uncompressed_size)
        )
        arrays: dict[str, np.ndarray] = {}
        for member in metadata.members:
            member_payload = _extract_member_payload(raw_payload, member)
            arrays[member.logical_name] = _load_npy_member(member_payload)
        if set(arrays) != {"source", "target"}:
            raise _malformed()
        return arrays["source"], arrays["target"]
    except (MemoryError, ResourceLimitError):
        raise
    except ValueError:
        raise
    except (
        EOFError,
        OSError,
        struct.error,
        zlib.error,
        RuntimeError,
    ) as error:
        raise _malformed() from error
