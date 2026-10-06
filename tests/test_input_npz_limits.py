from __future__ import annotations

import ast
import hashlib
import inspect
import io
import struct
import zipfile
import zlib
from pathlib import Path

import numpy as np
import pytest

import selcal.inputs as inputs
from selcal.canonical import semantic_input_sha256
from selcal.contracts import SeriesPair
from selcal.contracts_v2 import ResourceLimitError
from selcal.input_resources import INPUT_LIMITS_V1, raise_input_resource_limit
from selcal.inputs import load_npz
from selcal.npz_safe import (
    _enforce_central_directory_limit,
    _enforce_declared_size_limits,
    _find_eocd_offset,
    _load_npy_member,
    _local_zip64_sizes,
    _preflight_npy_member,
    _preflight_npz_archive,
    load_bounded_npz_arrays,
)

_EOCD = struct.Struct("<4s4H2LH")
_CENTRAL = struct.Struct("<4s6H3L5H2L")
_LOCAL = struct.Struct("<4s5H3L2H")


def _npy_bytes(values: np.ndarray, *, version: tuple[int, int] | None = None) -> bytes:
    buffer = io.BytesIO()
    if version is None:
        np.save(buffer, values, allow_pickle=False)
    else:
        np.lib.format.write_array(buffer, values, version=version, allow_pickle=False)
    return buffer.getvalue()


def _npz_bytes(*, compressed: bool = False) -> bytes:
    buffer = io.BytesIO()
    save = np.savez_compressed if compressed else np.savez
    save(
        buffer,
        source=np.array([1.0, 2.0, 3.0]),
        target=np.array([10.0, 20.0, 30.0]),
    )
    return buffer.getvalue()


def _zip_bytes(
    members: list[tuple[str, bytes]],
    *,
    compression: int = zipfile.ZIP_STORED,
    first_extra: bytes = b"",
) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=compression) as archive:
        for index, (name, payload) in enumerate(members):
            info = zipfile.ZipInfo(name)
            info.compress_type = compression
            if index == 0:
                info.extra = first_extra
            archive.writestr(info, payload)
    return buffer.getvalue()


def _eocd_offset(payload: bytes | bytearray) -> int:
    offset = payload.rfind(b"PK\x05\x06")
    assert offset >= 0
    return offset


def _central_offsets(payload: bytes | bytearray) -> list[int]:
    eocd = _eocd_offset(payload)
    fields = _EOCD.unpack_from(payload, eocd)
    cursor = fields[6]
    offsets: list[int] = []
    for _ in range(fields[4]):
        offsets.append(cursor)
        central = _CENTRAL.unpack_from(payload, cursor)
        cursor += _CENTRAL.size + central[10] + central[11] + central[12]
    return offsets


def _local_offset(payload: bytes | bytearray, central_offset: int) -> int:
    return int(_CENTRAL.unpack_from(payload, central_offset)[16])


def _archive_with_central_size(size: int) -> bytes:
    base_size = 2 * _CENTRAL.size + len("source.npy") + len("target.npy")
    extra_size = size - base_size
    assert extra_size >= 4
    extra = struct.pack("<HH", 0xCAFE, extra_size - 4) + b"x" * (extra_size - 4)
    return _zip_bytes(
        [
            ("source.npy", _npy_bytes(np.arange(3, dtype=np.float64))),
            ("target.npy", _npy_bytes(np.arange(10, 13, dtype=np.float64))),
        ],
        first_extra=extra,
    )


def _manual_npy(
    *,
    version: tuple[int, int] = (1, 0),
    shape: object = (3,),
    descr: object = "<f8",
    header_length: int | None = None,
    data: bytes | None = None,
) -> bytes:
    dictionary = repr({"descr": descr, "fortran_order": False, "shape": shape}).encode("latin1")
    length_size = 2 if version == (1, 0) else 4
    if header_length is None:
        header_length = len(dictionary) + 1
    assert header_length >= len(dictionary) + 1
    header = dictionary + b" " * (header_length - len(dictionary) - 1) + b"\n"
    length = header_length.to_bytes(length_size, "little")
    if data is None:
        try:
            count = int(shape[0]) if isinstance(shape, tuple) and len(shape) == 1 else 0
            itemsize = np.dtype(descr).itemsize
        except (TypeError, ValueError):
            count = 0
            itemsize = 0
        data = b"\x00" * max(0, count * itemsize)
    return b"\x93NUMPY" + bytes(version) + length + header + data


def _mutate_regular_declared_sizes(
    payload: bytes,
    *,
    source_size: int,
    target_size: int,
) -> bytes:
    mutated = bytearray(payload)
    for central, uncompressed in zip(
        _central_offsets(mutated), (source_size, target_size), strict=True
    ):
        local = _local_offset(mutated, central)
        central_values = list(_CENTRAL.unpack_from(mutated, central))
        local_values = list(_LOCAL.unpack_from(mutated, local))
        central_values[9] = uncompressed
        local_values[8] = uncompressed
        _CENTRAL.pack_into(mutated, central, *central_values)
        _LOCAL.pack_into(mutated, local, *local_values)
    return bytes(mutated)


def _mask_source_tail_with_declared_prefix(
    payload: bytes,
    *,
    declared_source: bytes,
) -> bytes:
    mutated = bytearray(payload)
    central = _central_offsets(mutated)[0]
    local = _local_offset(mutated, central)
    declared_crc32 = zlib.crc32(declared_source) & 0xFFFFFFFF
    central_values = list(_CENTRAL.unpack_from(mutated, central))
    local_values = list(_LOCAL.unpack_from(mutated, local))
    central_values[7] = declared_crc32
    central_values[9] = len(declared_source)
    local_values[6] = declared_crc32
    local_values[8] = len(declared_source)
    _CENTRAL.pack_into(mutated, central, *central_values)
    _LOCAL.pack_into(mutated, local, *local_values)
    return bytes(mutated)


def _raw_deflate(payload: bytes) -> bytes:
    compressor = zlib.compressobj(level=9, wbits=-15)
    return compressor.compress(payload) + compressor.flush()


def _manual_deflate_npz(
    *,
    source_payload: bytes,
    source_compressed: bytes,
) -> bytes:
    target_payload = _npy_bytes(np.arange(10, 13, dtype=np.float64))
    target_compressed = _raw_deflate(target_payload)
    records = [
        (b"source.npy", source_payload, source_compressed),
        (b"target.npy", target_payload, target_compressed),
    ]
    local_parts: list[bytes] = []
    central_parts: list[bytes] = []
    local_offset = 0
    for name, uncompressed, compressed in records:
        crc32 = zlib.crc32(uncompressed) & 0xFFFFFFFF
        local = _LOCAL.pack(
            b"PK\x03\x04",
            20,
            0,
            zipfile.ZIP_DEFLATED,
            0,
            0,
            crc32,
            len(compressed),
            len(uncompressed),
            len(name),
            0,
        )
        local_record = local + name + compressed
        local_parts.append(local_record)
        central_parts.append(
            _CENTRAL.pack(
                b"PK\x01\x02",
                20,
                20,
                0,
                zipfile.ZIP_DEFLATED,
                0,
                0,
                crc32,
                len(compressed),
                len(uncompressed),
                len(name),
                0,
                0,
                0,
                0,
                0,
                local_offset,
            )
            + name
        )
        local_offset += len(local_record)
    local_bytes = b"".join(local_parts)
    central_bytes = b"".join(central_parts)
    eocd = _EOCD.pack(
        b"PK\x05\x06",
        0,
        0,
        2,
        2,
        len(central_bytes),
        len(local_bytes),
        0,
    )
    return local_bytes + central_bytes + eocd


def _regular_small_npz_bytes() -> bytes:
    return _zip_bytes(
        [
            ("source.npy", _npy_bytes(np.arange(3, dtype=np.float64))),
            ("target.npy", _npy_bytes(np.arange(10, 13, dtype=np.float64))),
        ]
    )


def _targeted_rejection_payload(case: str) -> bytes:
    """Return bytes whose first NPZ rejection is the named production branch."""

    payload = bytearray(_regular_small_npz_bytes())
    central = _central_offsets(payload)[0]
    local = _local_offset(payload, central)

    if case == "trailing_partial_extra_header":
        valid_extra = struct.pack("<HH", 0xCAFE, 1) + b"x"
        payload = bytearray(
            _zip_bytes(
                [
                    ("source.npy", _npy_bytes(np.arange(3, dtype=np.float64))),
                    ("target.npy", _npy_bytes(np.arange(10, 13, dtype=np.float64))),
                ],
                first_extra=valid_extra,
            )
        )
        central = _central_offsets(payload)[0]
        central_values = _CENTRAL.unpack_from(payload, central)
        assert central_values[11] == len(valid_extra) == 5
        extra_start = central + _CENTRAL.size + central_values[10]
        payload[extra_start : extra_start + 5] = struct.pack("<HH", 0xCAFE, 0) + b"x"
        return bytes(payload)

    if case in {
        "zip64_payload_missing",
        "zip64_ordinary_uncompressed_mismatch",
        "zip64_ordinary_compressed_mismatch",
    }:
        payload = bytearray(_npz_bytes())
        central = _central_offsets(payload)[0]
        local = _local_offset(payload, central)
        central_values = _CENTRAL.unpack_from(payload, central)
        local_values = list(_LOCAL.unpack_from(payload, local))
        extra_start = local + _LOCAL.size + local_values[9]
        assert local_values[7:9] == [0xFFFFFFFF, 0xFFFFFFFF]
        assert struct.unpack_from("<H", payload, extra_start)[0] == 1
        if case == "zip64_payload_missing":
            struct.pack_into("<H", payload, extra_start, 0xCAFE)
        elif case == "zip64_ordinary_uncompressed_mismatch":
            local_values[8] = central_values[9] + 1
            _LOCAL.pack_into(payload, local, *local_values)
        else:
            local_values[7] = central_values[8] + 1
            _LOCAL.pack_into(payload, local, *local_values)
        return bytes(payload)

    if case == "local_offset_out_of_range":
        central_values = list(_CENTRAL.unpack_from(payload, central))
        central_values[16] = central
        _CENTRAL.pack_into(payload, central, *central_values)
    elif case == "bad_local_signature":
        payload[local : local + 4] = b"NOPE"
    elif case == "local_variable_fields_enter_central":
        local_values = list(_LOCAL.unpack_from(payload, local))
        required_name_length = central - local - _LOCAL.size + 1
        assert 0 < required_name_length <= 0xFFFF
        local_values[9] = required_name_length
        _LOCAL.pack_into(payload, local, *local_values)
    elif case == "member_data_enters_central":
        central_values = list(_CENTRAL.unpack_from(payload, central))
        local_values = list(_LOCAL.unpack_from(payload, local))
        data_offset = local + _LOCAL.size + local_values[9] + local_values[10]
        oversized = central - data_offset + 1
        assert 0 < oversized < 0xFFFFFFFF
        central_values[8] = oversized
        local_values[7] = oversized
        _CENTRAL.pack_into(payload, central, *central_values)
        _LOCAL.pack_into(payload, local, *local_values)
    elif case == "central_record_too_short":
        eocd = _eocd_offset(payload)
        eocd_values = list(_EOCD.unpack_from(payload, eocd))
        eocd_values[5] = _CENTRAL.size - 1
        eocd_values[6] = eocd - eocd_values[5]
        _EOCD.pack_into(payload, eocd, *eocd_values)
    elif case == "bad_central_signature":
        payload[central : central + 4] = b"NOPE"
    elif case == "central_size_sentinel":
        central_values = list(_CENTRAL.unpack_from(payload, central))
        central_values[8] = 0xFFFFFFFF
        _CENTRAL.pack_into(payload, central, *central_values)
    elif case == "central_record_overflow":
        central_values = list(_CENTRAL.unpack_from(payload, central))
        central_values[12] = 0xFFFF
        _CENTRAL.pack_into(payload, central, *central_values)
    elif case == "npy_missing_header_length":
        return _zip_bytes(
            [
                ("source.npy", b"\x93NUMPY\x01\x00"),
                ("target.npy", _npy_bytes(np.arange(10, 13, dtype=np.float64))),
            ]
        )
    else:
        raise AssertionError(f"unknown targeted rejection case: {case}")
    return bytes(payload)


def _capture_structured_rejection(
    monkeypatch: pytest.MonkeyPatch,
    *,
    member_error: bool,
) -> list[tuple[str, dict[str, object]]]:
    import selcal.npz_safe as npz_safe

    observed: list[tuple[str, dict[str, object]]] = []

    def reject() -> ValueError:
        frame = inspect.currentframe()
        assert frame is not None and frame.f_back is not None
        caller = frame.f_back
        observed.append((caller.f_code.co_name, dict(caller.f_locals)))
        if member_error:
            return npz_safe._NpyContractError("structured member branch probe")
        return ValueError("structured archive branch probe")

    target = "_member_malformed" if member_error else "_malformed"
    monkeypatch.setattr(npz_safe, target, reject)
    return observed


def _assert_target_rejection_predicate(
    case: str,
    caller: str,
    local: dict[str, object],
) -> None:
    if case == "trailing_partial_extra_header":
        assert caller == "_parse_extra_fields"
        assert len(local["extra"]) - local["cursor"] < struct.calcsize("<HH")
    elif case == "zip64_payload_missing":
        assert caller == "_local_zip64_sizes"
        assert local["zip64_payload"] is None
        assert local["compressed_sentinel"] or local["uncompressed_sentinel"]
    elif case == "zip64_ordinary_uncompressed_mismatch":
        assert caller == "_local_zip64_sizes"
        assert local["compressed_sentinel"] is True
        assert local["uncompressed_sentinel"] is False
        assert local["local_uncompressed"] != local["central_uncompressed"]
    elif case == "zip64_ordinary_compressed_mismatch":
        assert caller == "_local_zip64_sizes"
        assert local["compressed_sentinel"] is False
        assert local["uncompressed_sentinel"] is True
        assert local["local_compressed"] != local["central_compressed"]
    elif case == "local_offset_out_of_range":
        assert caller == "_parse_local_member"
        assert local["local_offset"] + _LOCAL.size > local["central_offset"]
    elif case == "bad_local_signature":
        assert caller == "_parse_local_member"
        assert local["local"][0] != b"PK\x03\x04"
    elif case == "local_variable_fields_enter_central":
        assert caller == "_parse_local_member"
        assert local["data_offset"] > local["central_offset"]
    elif case == "member_data_enters_central":
        assert caller == "_parse_local_member"
        assert local["data_end"] > local["central_offset"]
    elif case == "central_record_too_short":
        assert caller == "_preflight_npz_archive"
        assert local["cursor"] + _CENTRAL.size > local["eocd_offset"]
    elif case == "bad_central_signature":
        assert caller == "_preflight_npz_archive"
        assert local["central"][0] != b"PK\x01\x02"
    elif case == "central_size_sentinel":
        assert caller == "_preflight_npz_archive"
        assert local["compressed_size"] == 0xFFFFFFFF
    elif case == "central_record_overflow":
        assert caller == "_preflight_npz_archive"
        assert local["record_end"] > local["eocd_offset"]
    elif case == "npy_missing_header_length":
        assert caller == "_preflight_npy_member"
        assert len(local["member_payload"]) < local["length_end"]
    else:
        raise AssertionError(f"unknown structured rejection case: {case}")


@pytest.mark.parametrize(
    "case",
    [
        "trailing_partial_extra_header",
        "zip64_payload_missing",
        "zip64_ordinary_uncompressed_mismatch",
        "zip64_ordinary_compressed_mismatch",
        "local_offset_out_of_range",
        "bad_local_signature",
        "local_variable_fields_enter_central",
        "member_data_enters_central",
        "central_record_too_short",
        "bad_central_signature",
        "central_size_sentinel",
        "central_record_overflow",
        "npy_missing_header_length",
    ],
)
def test_public_npz_rejection_mutations_reach_target_branch(
    tmp_path: Path,
    case: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / f"target-{case}.npz"
    path.write_bytes(_targeted_rejection_payload(case))
    observed = _capture_structured_rejection(
        monkeypatch,
        member_error=case == "npy_missing_header_length",
    )

    with pytest.raises(ValueError, match=r"structured .* branch probe"):
        load_npz(path, candidates=(1,))

    assert len(observed) == 1
    _assert_target_rejection_predicate(case, *observed[0])


def test_public_npz_rejects_numpy_header_cursor_contract_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "header-cursor-contract-drift.npz"
    path.write_bytes(_regular_small_npz_bytes())
    real_reader = np.lib.format.read_array_header_1_0

    def drift_cursor(
        stream: io.BytesIO,
        *args: object,
        **kwargs: object,
    ) -> tuple[tuple[int, ...], bool, np.dtype[np.generic]]:
        result = real_reader(stream, *args, **kwargs)
        stream.seek(stream.tell() - 1)
        return result

    monkeypatch.setattr(np.lib.format, "read_array_header_1_0", drift_cursor)
    observed = _capture_structured_rejection(monkeypatch, member_error=True)
    with pytest.raises(ValueError, match="structured member branch probe"):
        load_npz(path, candidates=(1,))

    assert len(observed) == 1
    caller, local = observed[0]
    assert caller == "_preflight_npy_member"
    assert local["stream"].tell() != local["header_end"]


def test_fixed_input_limit_contract_and_derived_npz_formulas() -> None:
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
    assert INPUT_LIMITS_V1.npz_total_uncompressed_bytes == 2 * (
        8 + 4 + 4_096 + 1_000_000 * 8
    )


def test_eocd_search_is_limited_to_last_65557_bytes() -> None:
    payload = _EOCD.pack(b"PK\x05\x06", 0, 0, 0, 0, 0, 0, 0) + b"x" * 65_558

    with pytest.raises(ValueError, match="NPZ archive is malformed"):
        _find_eocd_offset(payload)


def test_eocd_requires_one_candidate_whose_comment_ends_at_payload_end() -> None:
    first = _EOCD.pack(b"PK\x05\x06", 0, 0, 0, 0, 0, 0, 22)
    second = _EOCD.pack(b"PK\x05\x06", 0, 0, 0, 0, 0, 0, 0)

    with pytest.raises(ValueError, match="NPZ archive is malformed"):
        _find_eocd_offset(first + second)


def test_false_eocd_signature_inside_comment_is_ignored() -> None:
    comment = b"prefix-PK\x05\x06-not-an-ending-candidate"
    payload = _EOCD.pack(b"PK\x05\x06", 0, 0, 0, 0, 0, 0, len(comment)) + comment

    assert _find_eocd_offset(payload) == 0


@pytest.mark.parametrize("size", [16_383, 16_384])
def test_central_directory_limit_accepts_limit_and_limit_minus_one(size: int) -> None:
    _enforce_central_directory_limit(size)
    metadata = _preflight_npz_archive(_archive_with_central_size(size))
    assert len(metadata.members) == 2


def test_public_load_rejects_central_directory_limit_plus_one(tmp_path: Path) -> None:
    path = tmp_path / "central-too-large.npz"
    path.write_bytes(_archive_with_central_size(16_385))

    with pytest.raises(ResourceLimitError, match=(
        r"INPUT_RESOURCE_LIMIT_EXCEEDED_V1 reason=NPZ_CENTRAL_DIRECTORY_BYTES "
        r"limit=16384 observed=16385"
    )):
        load_npz(path, candidates=(1,))


def test_preflight_propagates_central_directory_resource_error() -> None:
    with pytest.raises(ResourceLimitError, match="NPZ_CENTRAL_DIRECTORY_BYTES"):
        _preflight_npz_archive(_archive_with_central_size(16_385))


@pytest.mark.parametrize(
    "changes",
    [
        {"disk": 1},
        {"central_disk": 1},
        {"disk_entries": 1},
        {"entries": 1},
        {"disk_entries": 0xFFFF},
        {"entries": 0xFFFF},
        {"central_size": 0xFFFFFFFF},
        {"central_offset": 0xFFFFFFFF},
    ],
)
def test_eocd_rejects_multidisk_wrong_count_and_zip64_sentinels(
    changes: dict[str, int],
) -> None:
    payload = bytearray(_npz_bytes())
    eocd = _eocd_offset(payload)
    values = list(_EOCD.unpack_from(payload, eocd))
    indexes = {
        "disk": 1,
        "central_disk": 2,
        "disk_entries": 3,
        "entries": 4,
        "central_size": 5,
        "central_offset": 6,
    }
    for name, value in changes.items():
        values[indexes[name]] = value
    _EOCD.pack_into(payload, eocd, *values)

    with pytest.raises(ValueError, match=r"NPZ (?:archive is malformed|keys)"):
        _preflight_npz_archive(bytes(payload))


def test_eocd_rejects_zip64_locator_in_exact_preceding_window() -> None:
    payload = bytearray(_npz_bytes())
    eocd = _eocd_offset(payload)
    payload[eocd - 20 : eocd - 16] = b"PK\x06\x07"

    with pytest.raises(ValueError, match="NPZ archive is malformed"):
        _preflight_npz_archive(bytes(payload))


def test_eocd_requires_central_directory_to_end_exactly_at_eocd() -> None:
    payload = bytearray(_npz_bytes())
    eocd = _eocd_offset(payload)
    values = list(_EOCD.unpack_from(payload, eocd))
    values[5] -= 1
    _EOCD.pack_into(payload, eocd, *values)

    with pytest.raises(ValueError, match="NPZ archive is malformed"):
        _preflight_npz_archive(bytes(payload))


@pytest.mark.parametrize("where", ["central", "local"])
@pytest.mark.parametrize(
    "extra",
    [
        struct.pack("<HH", 0xCAFE, 3) + b"xx",
        struct.pack("<HH", 0xCAFE, 2) + b"x",
        struct.pack("<HH", 0xCAFE, 0) * 2,
    ],
)
def test_all_extra_fields_reject_truncation_and_duplicate_ids(
    where: str,
    extra: bytes,
) -> None:
    valid_extra = struct.pack("<HH", 0xBEEF, len(extra) - 4) + b"x" * (len(extra) - 4)
    payload = bytearray(_zip_bytes([
        ("source.npy", _npy_bytes(np.arange(3, dtype=np.float64))),
        ("target.npy", _npy_bytes(np.arange(10, 13, dtype=np.float64))),
    ], first_extra=valid_extra))
    central = _central_offsets(payload)[0]
    local = _local_offset(payload, central)
    if where == "central":
        values = list(_CENTRAL.unpack_from(payload, central))
        name_length = values[10]
        old_extra_length = values[11]
        assert len(extra) == old_extra_length
        start = central + _CENTRAL.size + name_length
        payload[start : start + old_extra_length] = extra
    else:
        values = list(_LOCAL.unpack_from(payload, local))
        name_length = values[9]
        old_extra_length = values[10]
        assert len(extra) == old_extra_length
        start = local + _LOCAL.size + name_length
        payload[start : start + old_extra_length] = extra

    with pytest.raises(ValueError, match="NPZ archive is malformed"):
        _preflight_npz_archive(bytes(payload))


def test_central_zip64_extra_is_always_rejected() -> None:
    payload = bytearray(_npz_bytes())
    central = _central_offsets(payload)[0]
    values = list(_CENTRAL.unpack_from(payload, central))
    name_length = values[10]
    values[11] = 20
    start = central + _CENTRAL.size + name_length
    payload[start:start] = struct.pack("<HHQQ", 1, 16, values[9], values[8])
    _CENTRAL.pack_into(payload, central, *values)
    eocd = _eocd_offset(payload)
    eocd_values = list(_EOCD.unpack_from(payload, eocd))
    eocd_values[5] += 20
    _EOCD.pack_into(payload, eocd, *eocd_values)

    with pytest.raises(ValueError, match="NPZ archive is malformed"):
        _preflight_npz_archive(bytes(payload))


@pytest.mark.parametrize("compressed", [False, True])
def test_ambient_numpy_real_local_zip64_layout_is_accepted(compressed: bool) -> None:
    metadata = _preflight_npz_archive(_npz_bytes(compressed=compressed))

    assert {member.logical_name for member in metadata.members} == {"source", "target"}
    assert all(member.local_zip64 for member in metadata.members)


@pytest.mark.parametrize(
    ("local_compressed", "local_uncompressed", "payload_words"),
    [
        (0xFFFFFFFF, 152, (152,)),
        (152, 0xFFFFFFFF, (152,)),
        (0xFFFFFFFF, 0xFFFFFFFF, (152, 152)),
    ],
)
def test_local_zip64_accepts_exact_sentinel_backed_values_in_appnote_order(
    local_compressed: int,
    local_uncompressed: int,
    payload_words: tuple[int, ...],
) -> None:
    payload = bytearray(_npz_bytes())
    central = _central_offsets(payload)[0]
    local = _local_offset(payload, central)
    values = list(_LOCAL.unpack_from(payload, local))
    values[7] = local_compressed
    values[8] = local_uncompressed
    name_length = values[9]
    extra_length = values[10]
    zip64 = struct.pack("<HH", 1, 8 * len(payload_words)) + struct.pack(
        "<" + "Q" * len(payload_words), *payload_words
    )
    assert len(zip64) <= extra_length
    replacement = zip64
    if len(zip64) < extra_length:
        remaining = extra_length - len(zip64)
        replacement += struct.pack("<HH", 0xCAFE, remaining - 4) + b"x" * (remaining - 4)
    start = local + _LOCAL.size + name_length
    payload[start : start + extra_length] = replacement
    _LOCAL.pack_into(payload, local, *values)

    metadata = _preflight_npz_archive(bytes(payload))
    assert metadata.members[0].local_zip64


@pytest.mark.parametrize(
    "zip64_payload",
    [
        struct.pack("<Q", 151),
        struct.pack("<QQ", 152, 151),
        struct.pack("<QQQ", 152, 152, 0),
        struct.pack("<QQQQ", 152, 152, 0, 0),
    ],
)
def test_local_zip64_rejects_wrong_value_length_or_offset_disk_payload(
    zip64_payload: bytes,
) -> None:
    payload = bytearray(_npz_bytes())
    central = _central_offsets(payload)[0]
    local = _local_offset(payload, central)
    values = list(_LOCAL.unpack_from(payload, local))
    name_length = values[9]
    extra_length = values[10]
    encoded = struct.pack("<HH", 1, len(zip64_payload)) + zip64_payload
    if len(encoded) < extra_length:
        encoded += b"\x00" * (extra_length - len(encoded))
    start = local + _LOCAL.size + name_length
    payload[start : start + extra_length] = encoded[:extra_length]

    with pytest.raises(ValueError, match="NPZ archive is malformed"):
        _preflight_npz_archive(bytes(payload))


@pytest.mark.parametrize(
    ("case", "zip64_payload"),
    [
        ("reverse_appnote_order", struct.pack("<QQ", 73, 152)),
        ("wrong_length", struct.pack("<Q", 152)),
        ("offset_content", struct.pack("<QQQ", 152, 73, 19)),
        ("disk_start_content", struct.pack("<QQQI", 152, 73, 19, 2)),
    ],
)
def test_local_zip64_semantic_validator_rejects_forbidden_payload_forms(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    zip64_payload: bytes,
) -> None:
    import selcal.npz_safe as npz_safe

    malformed_calls: list[str] = []

    def semantic_rejection() -> ValueError:
        malformed_calls.append(case)
        return ValueError("local ZIP64 semantic rejection")

    monkeypatch.setattr(npz_safe, "_malformed", semantic_rejection)

    with pytest.raises(ValueError, match="local ZIP64 semantic rejection"):
        _local_zip64_sizes(
            local_compressed=0xFFFFFFFF,
            local_uncompressed=0xFFFFFFFF,
            central_compressed=73,
            central_uncompressed=152,
            extra_fields={1: zip64_payload},
        )

    assert malformed_calls == [case]


def test_local_zip64_is_rejected_without_any_size_sentinel() -> None:
    payload = bytearray(_npz_bytes())
    central = _central_offsets(payload)[0]
    local = _local_offset(payload, central)
    central_values = _CENTRAL.unpack_from(payload, central)
    values = list(_LOCAL.unpack_from(payload, local))
    values[7] = central_values[8]
    values[8] = central_values[9]
    _LOCAL.pack_into(payload, local, *values)

    with pytest.raises(ValueError, match="NPZ archive is malformed"):
        _preflight_npz_archive(bytes(payload))


@pytest.mark.parametrize(
    "names",
    [
        ["source.npy"],
        ["source.npy", "target.npy", "extra.npy"],
        ["source.npy", "source.npy"],
        ["source.npy", "folder/target.npy"],
        ["source.npy", "../target.npy"],
        ["source.npy", "target\\x.npy"],
        ["source.npy", "target.npy/"],
        ["source.npy", "target.npy.npy"],
        ["source.npy", "Target.npy"],
    ],
)
def test_logical_members_must_be_exactly_canonical_source_and_target(
    names: list[str],
) -> None:
    members = [(name, _npy_bytes(np.arange(3, dtype=np.float64))) for name in names]
    if len(names) != len(set(names)):
        with pytest.warns(UserWarning, match="Duplicate name"):
            payload = _zip_bytes(members)
    else:
        payload = _zip_bytes(members)

    with pytest.raises(ValueError, match=r"NPZ (?:archive is malformed|keys)"):
        _preflight_npz_archive(payload)


def test_nul_in_raw_member_name_is_rejected() -> None:
    payload = bytearray(_zip_bytes([
        ("source.npy", _npy_bytes(np.arange(3, dtype=np.float64))),
        ("target.npy", _npy_bytes(np.arange(10, 13, dtype=np.float64))),
    ]))
    central = _central_offsets(payload)[1]
    local = _local_offset(payload, central)
    payload[central + _CENTRAL.size + 6] = 0
    payload[local + _LOCAL.size + 6] = 0

    with pytest.raises(ValueError, match="NPZ archive is malformed"):
        _preflight_npz_archive(bytes(payload))


@pytest.mark.parametrize("suffix", ["", ".npy"])
def test_source_and_target_canonical_name_forms_are_accepted(suffix: str) -> None:
    payload = _zip_bytes(
        [
            ("source" + suffix, _npy_bytes(np.arange(3, dtype=np.float64))),
            ("target" + suffix, _npy_bytes(np.arange(10, 13, dtype=np.float64))),
        ]
    )

    metadata = _preflight_npz_archive(payload)
    assert {member.logical_name for member in metadata.members} == {"source", "target"}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("flags", 0x0001),
        ("flags", 0x0008),
        ("flags", 0x0040),
        ("flags", 0x2000),
        ("method", zipfile.ZIP_BZIP2),
        ("method", zipfile.ZIP_LZMA),
    ],
)
def test_forbidden_flags_and_compression_methods_are_rejected(
    field: str,
    value: int,
) -> None:
    payload = bytearray(_zip_bytes([
        ("source.npy", _npy_bytes(np.arange(3, dtype=np.float64))),
        ("target.npy", _npy_bytes(np.arange(10, 13, dtype=np.float64))),
    ]))
    central = _central_offsets(payload)[0]
    local = _local_offset(payload, central)
    central_values = list(_CENTRAL.unpack_from(payload, central))
    local_values = list(_LOCAL.unpack_from(payload, local))
    if field == "flags":
        central_values[3] = value
        local_values[2] = value
    else:
        central_values[4] = value
        local_values[3] = value
    _CENTRAL.pack_into(payload, central, *central_values)
    _LOCAL.pack_into(payload, local, *local_values)

    with pytest.raises(ValueError, match="NPZ archive is malformed"):
        _preflight_npz_archive(bytes(payload))


@pytest.mark.parametrize("external_attributes", [0x10, 0o040755 << 16])
def test_directory_member_attributes_are_rejected(external_attributes: int) -> None:
    payload = bytearray(_zip_bytes([
        ("source.npy", _npy_bytes(np.arange(3, dtype=np.float64))),
        ("target.npy", _npy_bytes(np.arange(10, 13, dtype=np.float64))),
    ]))
    central = _central_offsets(payload)[0]
    central_values = list(_CENTRAL.unpack_from(payload, central))
    central_values[15] = external_attributes
    _CENTRAL.pack_into(payload, central, *central_values)

    with pytest.raises(ValueError, match="NPZ archive is malformed"):
        _preflight_npz_archive(bytes(payload))


@pytest.mark.parametrize("field", ["name", "flags", "method", "crc", "compressed", "uncompressed"])
def test_local_and_central_metadata_must_match(field: str) -> None:
    payload = bytearray(_zip_bytes([
        ("source.npy", _npy_bytes(np.arange(3, dtype=np.float64))),
        ("target.npy", _npy_bytes(np.arange(10, 13, dtype=np.float64))),
    ]))
    central = _central_offsets(payload)[0]
    local = _local_offset(payload, central)
    values = list(_LOCAL.unpack_from(payload, local))
    indexes = {"flags": 2, "method": 3, "crc": 6, "compressed": 7, "uncompressed": 8}
    if field == "name":
        payload[local + _LOCAL.size] ^= 1
    else:
        values[indexes[field]] += 1
        _LOCAL.pack_into(payload, local, *values)

    with pytest.raises(ValueError, match="NPZ archive is malformed"):
        _preflight_npz_archive(bytes(payload))


def test_member_ranges_must_not_overlap_or_enter_central_directory() -> None:
    payload = bytearray(_zip_bytes([
        ("source.npy", _npy_bytes(np.arange(3, dtype=np.float64))),
        ("target.npy", _npy_bytes(np.arange(10, 13, dtype=np.float64))),
    ]))
    central = _central_offsets(payload)[0]
    local = _local_offset(payload, central)
    values = list(_LOCAL.unpack_from(payload, local))
    central_values = list(_CENTRAL.unpack_from(payload, central))
    extension = 64
    values[7] += extension
    values[8] += extension
    central_values[8] += extension
    central_values[9] += extension
    _LOCAL.pack_into(payload, local, *values)
    _CENTRAL.pack_into(payload, central, *central_values)

    with pytest.raises(ValueError, match="NPZ archive is malformed"):
        _preflight_npz_archive(bytes(payload))


@pytest.mark.parametrize(
    "sizes",
    [
        (INPUT_LIMITS_V1.npz_member_uncompressed_bytes - 1, 0),
        (INPUT_LIMITS_V1.npz_member_uncompressed_bytes, 0),
        (
            INPUT_LIMITS_V1.npz_member_uncompressed_bytes,
            INPUT_LIMITS_V1.npz_member_uncompressed_bytes - 1,
        ),
        (
            INPUT_LIMITS_V1.npz_member_uncompressed_bytes,
            INPUT_LIMITS_V1.npz_member_uncompressed_bytes,
        ),
    ],
)
def test_declared_member_and_total_boundaries_accept_limit_and_limit_minus_one(
    sizes: tuple[int, int],
) -> None:
    _enforce_declared_size_limits(sizes)


def test_declared_member_limit_plus_one_is_rejected_before_extraction() -> None:
    with pytest.raises(ResourceLimitError, match=(
        r"reason=NPZ_MEMBER_UNCOMPRESSED_BYTES limit=8004108 observed=8004109"
    )):
        _enforce_declared_size_limits((8_004_109, 0))


def test_declared_total_limit_plus_one_is_rejected_before_member_work() -> None:
    with pytest.raises(ResourceLimitError, match=(
        r"reason=NPZ_TOTAL_UNCOMPRESSED_BYTES limit=16008216 observed=16008217"
    )):
        _enforce_declared_size_limits((8_004_108, 8_004_109))


@pytest.mark.parametrize("header_length", [4_095, 4_096])
def test_npy_header_limit_and_limit_minus_one_are_accepted(header_length: int) -> None:
    payload = _manual_npy(header_length=header_length)

    shape, dtype, header_end = _preflight_npy_member(payload)
    assert shape == (3,)
    assert dtype == np.dtype("<f8")
    assert header_end == 8 + 2 + header_length


def test_npy_header_limit_plus_one_is_rejected_before_array_load() -> None:
    payload = _manual_npy(header_length=4_097, data=b"")

    with pytest.raises(ResourceLimitError, match=(
        r"reason=NPY_HEADER_BYTES limit=4096 observed=4097"
    )):
        _preflight_npy_member(payload)


@pytest.mark.parametrize("version", [(1, 0), (2, 0)])
def test_npy_versions_one_and_two_are_accepted(version: tuple[int, int]) -> None:
    values = np.array([1.0, 2.0, 3.0])
    payload = _npy_bytes(values, version=version)

    loaded = _load_npy_member(payload)
    np.testing.assert_array_equal(loaded, values)


def test_npy_version_three_is_rejected() -> None:
    payload = bytearray(_npy_bytes(np.arange(3, dtype=np.float64), version=(2, 0)))
    payload[6:8] = b"\x03\x00"

    with pytest.raises(ValueError, match="NPZ array member is malformed"):
        _preflight_npy_member(bytes(payload))


@pytest.mark.parametrize("count", [999_999, 1_000_000])
def test_npy_element_limit_and_limit_minus_one_are_accepted(count: int) -> None:
    payload = _manual_npy(shape=(count,), descr="|u1")

    shape, dtype, _ = _preflight_npy_member(payload)
    assert shape == (count,)
    assert dtype == np.dtype("u1")


def test_npy_element_limit_plus_one_uses_header_only_and_precedes_length_check() -> None:
    payload = _manual_npy(shape=(1_000_001,), descr="|u1", data=b"")

    with pytest.raises(ResourceLimitError, match=(
        r"reason=NPY_ELEMENTS limit=1000000 observed=1000001"
    )):
        _preflight_npy_member(payload)


@pytest.mark.parametrize(
    "shape",
    [
        (2, 2),
        (-1,),
        (),
        (True,),
        "not-a-tuple",
    ],
)
def test_npy_shape_must_be_one_dimensional_nonnegative_builtin_int(shape: object) -> None:
    payload = _manual_npy(shape=shape, descr="|u1", data=b"")

    with pytest.raises(ValueError, match=r"NPZ.*(?:malformed|one-dimensional)"):
        _preflight_npy_member(payload)


@pytest.mark.parametrize(
    "descr",
    [
        "|O",
        "|b1",
        "<c16",
        "|S1",
        "<U1",
        [("field", "<i4")],
        "<f16",
    ],
)
def test_npy_dtype_is_limited_to_real_numeric_at_most_eight_bytes(descr: object) -> None:
    payload = _manual_npy(shape=(1,), descr=descr)

    with pytest.raises(ValueError, match=r"NPZ.*(?:malformed|dtype|object|itemsize)"):
        _preflight_npy_member(payload)


@pytest.mark.parametrize("delta", [-1, 1])
def test_npy_member_size_must_equal_header_plus_exact_data_bytes(delta: int) -> None:
    payload = _manual_npy(shape=(3,), descr="<f8")
    malformed = payload[:delta] if delta < 0 else payload + b"x"

    with pytest.raises(ValueError, match="NPZ array member is malformed"):
        _preflight_npy_member(malformed)


@pytest.mark.parametrize(
    ("version", "reader_name"),
    [
        ((1, 0), "read_array_header_1_0"),
        ((2, 0), "read_array_header_2_0"),
    ],
)
def test_npy_public_header_parser_starts_at_offset_eight_and_finishes_at_header_end(
    monkeypatch: pytest.MonkeyPatch,
    version: tuple[int, int],
    reader_name: str,
) -> None:
    payload = _npy_bytes(np.arange(3, dtype=np.float64), version=version)
    real = getattr(np.lib.format, reader_name)
    observed: list[tuple[int, int]] = []
    length_size = 2 if version == (1, 0) else 4
    independent_header_end = 8 + length_size + int.from_bytes(
        payload[8 : 8 + length_size],
        "little",
    )

    def capture(
        stream: io.BytesIO,
        *args: object,
        **kwargs: object,
    ) -> tuple[tuple[int, ...], bool, np.dtype[np.generic]]:
        start = stream.tell()
        result = real(stream, *args, **kwargs)
        observed.append((start, stream.tell()))
        return result

    monkeypatch.setattr(np.lib.format, reader_name, capture)
    _, _, header_end = _preflight_npy_member(payload)

    assert header_end == independent_header_end
    assert observed == [(8, independent_header_end)]


@pytest.mark.parametrize(
    "error",
    [
        struct.error("bad struct"),
        ValueError("bad value"),
        EOFError("bad eof"),
        SyntaxError("bad syntax"),
        UnicodeError("bad unicode"),
        RuntimeError("bad runtime"),
    ],
)
def test_npy_header_parser_structural_errors_map_to_stable_value_error(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    payload = _npy_bytes(np.arange(3, dtype=np.float64), version=(1, 0))

    def fail(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise error

    monkeypatch.setattr(np.lib.format, "read_array_header_1_0", fail)
    with pytest.raises(ValueError, match="NPZ array member is malformed"):
        _preflight_npy_member(payload)


@pytest.mark.parametrize(
    "error",
    [
        ValueError("bad value"),
        EOFError("bad eof"),
        OSError("bad io"),
        zipfile.BadZipFile("bad zip"),
        RuntimeError("bad runtime"),
    ],
)
def test_np_load_malformed_errors_map_to_stable_value_error(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    payload = _npy_bytes(np.arange(3, dtype=np.float64))

    def fail(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise error

    monkeypatch.setattr(np, "load", fail)
    with pytest.raises(ValueError, match="NPZ array member is malformed"):
        _load_npy_member(payload)


@pytest.mark.parametrize(
    "error",
    [
        MemoryError("simulated exhaustion"),
        ResourceLimitError("simulated resource limit"),
    ],
)
def test_npy_memory_and_resource_errors_propagate_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    payload = _npy_bytes(np.arange(3, dtype=np.float64))

    def fail(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise error

    monkeypatch.setattr(np, "load", fail)
    with pytest.raises(type(error), match=str(error)):
        _load_npy_member(payload)


@pytest.mark.parametrize("compressed", [False, True])
def test_bounded_loader_accepts_real_ambient_numpy_archives(compressed: bool) -> None:
    source, target = load_bounded_npz_arrays(_npz_bytes(compressed=compressed))

    np.testing.assert_array_equal(source, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(target, np.array([10.0, 20.0, 30.0]))


@pytest.mark.parametrize("compressed", [False, True])
def test_public_load_never_constructs_the_stdlib_zip_parser(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    compressed: bool,
) -> None:
    path = tmp_path / ("deflated.npz" if compressed else "stored.npz")
    payload = _npz_bytes(compressed=compressed)
    path.write_bytes(payload)

    def forbidden_zip_parser(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("the manual NPZ parser must be the only ZIP parser")

    monkeypatch.setattr(zipfile, "ZipFile", forbidden_zip_parser)

    loaded = load_npz(path, candidates=(1,))

    assert loaded.raw_input_sha256 == hashlib.sha256(payload).hexdigest()
    np.testing.assert_array_equal(loaded.pair.source, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(loaded.pair.target, np.array([10.0, 20.0, 30.0]))


def test_production_npz_ast_allows_only_the_zip_exception_import() -> None:
    import selcal.npz_safe as npz_safe

    source = inspect.getsource(npz_safe)
    tree = ast.parse(source)
    zip_imports: list[tuple[str, str | None]] = []
    forbidden_references: list[str] = []
    forbidden_names = {"ZipFile", "ZipInfo", "ZipExtFile"}
    forbidden_private_attributes = {
        "_compress_left",
        "_compress_type",
        "_decompressor",
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "zipfile":
                    forbidden_references.append("import zipfile")
        elif isinstance(node, ast.ImportFrom) and node.module == "zipfile":
            zip_imports.extend((alias.name, alias.asname) for alias in node.names)
        elif isinstance(node, ast.Name) and node.id in forbidden_names:
            forbidden_references.append(node.id)
        elif isinstance(node, ast.Attribute) and (
            node.attr in forbidden_names or node.attr in forbidden_private_attributes
        ):
            forbidden_references.append(node.attr)
        elif isinstance(node, ast.Call):
            if (
                isinstance(node.func, ast.Name)
                and node.func.id in {"getattr", "hasattr", "setattr", "delattr"}
                and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value
                in forbidden_names | forbidden_private_attributes
            ):
                forbidden_references.append(
                    f"dynamic {node.func.id}({node.args[1].value!r})"
                )
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "__import__"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "zipfile"
            ):
                forbidden_references.append("dynamic __import__('zipfile')")
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "import_module"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "zipfile"
            ):
                forbidden_references.append("dynamic import_module('zipfile')")
            if isinstance(node.func, ast.Name) and node.func.id in {"eval", "exec"}:
                forbidden_references.append(node.func.id)

    assert zip_imports == [("BadZipFile", "_BadZipFile")]
    assert forbidden_references == []


def test_no_tail_deflate_uses_one_bounded_no_flush_decompressor_and_offers_exact_range(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import selcal.npz_safe as npz_safe

    generator = np.random.default_rng(20260901)
    source = generator.integers(0, 256, size=200_000, dtype=np.uint8)
    target = generator.integers(0, 256, size=200_000, dtype=np.uint8)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, source=source, target=target)
    raw_payload = buffer.getvalue()
    metadata = _preflight_npz_archive(raw_payload)
    path = tmp_path / "bounded-deflate.npz"
    path.write_bytes(raw_payload)
    real_decompressobj = zlib.decompressobj
    instances: list[TrackedDecompressor] = []

    class TrackedDecompressor:
        def __init__(self, wbits: int) -> None:
            self.inner = real_decompressobj(wbits=wbits)
            self.wbits = wbits
            self.observed = 0
            self.calls: list[tuple[int, int, int, int]] = []
            self.flush_calls = 0

        @property
        def eof(self) -> bool:
            return self.inner.eof

        @property
        def unused_data(self) -> bytes:
            return self.inner.unused_data

        @property
        def unconsumed_tail(self) -> bytes:
            return self.inner.unconsumed_tail

        def decompress(self, data: bytes, max_length: int) -> bytes:
            remaining = INPUT_LIMITS_V1.npz_member_uncompressed_bytes + 1 - self.observed
            output = self.inner.decompress(data, max_length)
            self.calls.append(
                (len(data), max_length, remaining, len(self.inner.unconsumed_tail))
            )
            self.observed += len(output)
            return output

        def flush(self) -> bytes:
            self.flush_calls += 1
            raise AssertionError("deflate extraction must not call flush")

    def tracked_factory(wbits: int = zlib.MAX_WBITS) -> TrackedDecompressor:
        instance = TrackedDecompressor(wbits)
        instances.append(instance)
        return instance

    monkeypatch.setattr(npz_safe.zlib, "decompressobj", tracked_factory)

    loaded = load_npz(path, candidates=(1,))

    np.testing.assert_array_equal(loaded.pair.source, source.astype(np.float64))
    np.testing.assert_array_equal(loaded.pair.target, target.astype(np.float64))
    assert len(instances) == 2
    assert all(instance.wbits == -15 for instance in instances)
    assert all(instance.flush_calls == 0 for instance in instances)
    assert all(instance.calls for instance in instances)
    assert all(
        0 < input_size <= 65_536 and 0 < max_length <= remaining
        for instance in instances
        for input_size, max_length, remaining, _tail_size in instance.calls
    )
    assert all(
        tail_size == 0
        for instance in instances
        for _input_size, _max_length, _remaining, tail_size in instance.calls
    )
    assert [sum(call[0] for call in instance.calls) for instance in instances] == [
        member.compressed_size for member in metadata.members
    ]


def test_deflate_drains_unconsumed_tail_before_offering_new_raw_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import selcal.npz_safe as npz_safe

    source_payload = _npy_bytes(np.arange(3, dtype=np.float64))
    raw_payload = _manual_deflate_npz(
        source_payload=source_payload,
        source_compressed=b"A" * 70_000,
    )
    source_member = _preflight_npz_archive(raw_payload).members[0]
    raw_ranges: list[tuple[int, int]] = []

    class TrackingBytes(bytes):
        def __getitem__(self, key: object) -> object:
            if isinstance(key, slice):
                start, stop, step = key.indices(len(self))
                if (
                    step == 1
                    and source_member.data_offset <= start < source_member.data_end
                    and start < stop <= source_member.data_end
                ):
                    raw_ranges.append((start, stop))
            return super().__getitem__(key)  # type: ignore[index]

    tracked_payload = TrackingBytes(raw_payload)
    instances: list[TailFirstDecompressor] = []

    class TailFirstDecompressor:
        def __init__(self) -> None:
            self.phase = 0
            self.eof = False
            self.unused_data = b""
            self.unconsumed_tail = b""
            self.calls: list[bytes] = []

        def decompress(self, data: bytes, max_length: int) -> bytes:
            assert 0 < max_length <= INPUT_LIMITS_V1.npz_member_uncompressed_bytes + 1
            self.calls.append(data)
            if self.phase == 0:
                assert data == b"A" * 65_536
                self.unconsumed_tail = b"TAIL"
                self.phase = 1
                return b""
            if self.phase == 1:
                assert data == b"TAIL", "new raw input was offered before tail drain"
                self.unconsumed_tail = b""
                self.phase = 2
                return source_payload
            assert self.phase == 2
            assert data == b"A" * (70_000 - 65_536)
            self.eof = True
            self.phase = 3
            return b""

    def fake_factory(wbits: int = zlib.MAX_WBITS) -> TailFirstDecompressor:
        assert wbits == -15
        instance = TailFirstDecompressor()
        instances.append(instance)
        return instance

    monkeypatch.setattr(npz_safe.zlib, "decompressobj", fake_factory)

    assert npz_safe._extract_member_payload(tracked_payload, source_member) == source_payload
    assert len(instances) == 1
    assert [len(data) for data in instances[0].calls] == [65_536, 4, 4_464]
    assert raw_ranges == [
        (source_member.data_offset, source_member.data_offset + 65_536),
        (source_member.data_offset + 65_536, source_member.data_end),
    ]


def test_deflate_rejects_no_output_and_no_input_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import selcal.npz_safe as npz_safe

    source_payload = _npy_bytes(np.arange(3, dtype=np.float64))
    raw_payload = _manual_deflate_npz(
        source_payload=source_payload,
        source_compressed=b"stalled-deflate-input",
    )
    source_member = _preflight_npz_archive(raw_payload).members[0]

    class StalledDecompressor:
        eof = False
        unused_data = b""
        unconsumed_tail = b""

        def decompress(self, data: bytes, max_length: int) -> bytes:
            assert max_length > 0
            self.unconsumed_tail = data
            return b""

    def stalled_factory(wbits: int = zlib.MAX_WBITS) -> StalledDecompressor:
        assert wbits == -15
        return StalledDecompressor()

    monkeypatch.setattr(npz_safe.zlib, "decompressobj", stalled_factory)

    with pytest.raises(ValueError, match="NPZ archive is malformed"):
        npz_safe._extract_member_payload(raw_payload, source_member)


def test_stored_actual_range_limit_plus_one_precedes_equality_crc_and_npy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import selcal.npz_safe as npz_safe

    path = tmp_path / "stored-actual-limit-plus-one.npz"
    target_payload = _npy_bytes(np.arange(10, 13, dtype=np.float64))
    payload = _zip_bytes(
        [
            (
                "source.npy",
                b"x" * (INPUT_LIMITS_V1.npz_member_uncompressed_bytes + 1),
            ),
            ("target.npy", target_payload),
        ]
    )
    path.write_bytes(
        _mutate_regular_declared_sizes(
            payload,
            source_size=INPUT_LIMITS_V1.npz_member_uncompressed_bytes,
            target_size=len(target_payload),
        )
    )
    forbidden_calls: list[str] = []

    def forbidden(*args: object, **kwargs: object) -> None:
        del args, kwargs
        forbidden_calls.append("called")
        raise AssertionError("stored overflow must precede equality, CRC, and NPY")

    monkeypatch.setattr(npz_safe.zlib, "crc32", forbidden)
    monkeypatch.setattr(npz_safe, "_preflight_npy_member", forbidden)
    monkeypatch.setattr(np, "load", forbidden)

    with pytest.raises(ResourceLimitError, match=(
        r"reason=NPZ_MEMBER_UNCOMPRESSED_BYTES limit=8004108 "
        r"observed_at_least=8004109"
    )):
        load_npz(path, candidates=(1,))

    assert forbidden_calls == []


def test_public_stored_length_mismatch_precedes_crc_and_npy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import selcal.npz_safe as npz_safe

    path = tmp_path / "stored-length-mismatch.npz"
    declared_source = _npy_bytes(np.arange(3, dtype=np.float64))
    payload = _zip_bytes(
        [
            ("source.npy", declared_source + b"x"),
            ("target.npy", _npy_bytes(np.arange(10, 13, dtype=np.float64))),
        ]
    )
    path.write_bytes(
        _mask_source_tail_with_declared_prefix(
            payload,
            declared_source=declared_source,
        )
    )
    forbidden_calls: list[str] = []

    def forbidden(*args: object, **kwargs: object) -> None:
        del args, kwargs
        forbidden_calls.append("called")
        raise AssertionError("stored size mismatch must precede CRC and NPY")

    monkeypatch.setattr(npz_safe.zlib, "crc32", forbidden)
    monkeypatch.setattr(npz_safe, "_preflight_npy_member", forbidden)
    monkeypatch.setattr(np, "load", forbidden)

    with pytest.raises(ValueError, match="NPZ archive is malformed"):
        load_npz(path, candidates=(1,))

    assert forbidden_calls == []


def test_public_load_propagates_deflate_factory_memory_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import selcal.npz_safe as npz_safe

    path = tmp_path / "deflate-memory-error.npz"
    path.write_bytes(_npz_bytes(compressed=True))

    def exhaust(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise MemoryError("simulated deflate allocation failure")

    monkeypatch.setattr(npz_safe.zlib, "decompressobj", exhaust)

    with pytest.raises(MemoryError, match="simulated deflate allocation failure"):
        load_npz(path, candidates=(1,))


@pytest.mark.parametrize("compressed", [False, True])
def test_public_load_preserves_values_and_independently_computed_hashes_for_real_numpy_archives(
    tmp_path: Path,
    compressed: bool,
) -> None:
    source = np.array([7, -2, 11, 5], dtype=np.int64)
    target = np.array([0.5, 1.25, -3.0, 9.5], dtype=np.float32)
    buffer = io.BytesIO()
    save = np.savez_compressed if compressed else np.savez
    save(buffer, source=source, target=target)
    raw_payload = buffer.getvalue()
    path = tmp_path / ("compressed.npz" if compressed else "stored.npz")
    path.write_bytes(raw_payload)

    loaded = load_npz(path, candidates=(2, 1))
    expected_pair = SeriesPair(source=source, target=target)

    np.testing.assert_array_equal(loaded.pair.source, expected_pair.source)
    np.testing.assert_array_equal(loaded.pair.target, expected_pair.target)
    assert loaded.raw_input_sha256 == hashlib.sha256(raw_payload).hexdigest()
    assert loaded.semantic_input_sha256 == semantic_input_sha256(
        expected_pair,
        (2, 1),
    )


def test_public_stored_crc_failure_stops_before_npy_preflight_and_np_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = bytearray(_zip_bytes([
        ("source.npy", _npy_bytes(np.arange(3, dtype=np.float64))),
        ("target.npy", _npy_bytes(np.arange(10, 13, dtype=np.float64))),
    ]))
    central = _central_offsets(payload)[0]
    local = _local_offset(payload, central)
    central_values = list(_CENTRAL.unpack_from(payload, central))
    local_values = list(_LOCAL.unpack_from(payload, local))
    central_values[7] ^= 1
    local_values[6] ^= 1
    _CENTRAL.pack_into(payload, central, *central_values)
    _LOCAL.pack_into(payload, local, *local_values)
    path = tmp_path / "stored-crc-failure.npz"
    path.write_bytes(bytes(payload))
    calls: list[str] = []

    def forbidden(*args: object, **kwargs: object) -> None:
        del args, kwargs
        calls.append("called")
        raise AssertionError("NPY parsing must not run after CRC failure")

    import selcal.npz_safe as npz_safe

    monkeypatch.setattr(npz_safe, "_preflight_npy_member", forbidden)
    monkeypatch.setattr(np, "load", forbidden)
    with pytest.raises(ValueError, match="NPZ archive is malformed"):
        load_npz(path, candidates=(1,))
    assert calls == []


def test_public_load_rejects_deflate_output_hidden_beyond_declared_npy_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import selcal.npz_safe as npz_safe

    path = tmp_path / "hidden-trailing-output.npz"
    declared_source = _npy_bytes(np.arange(3, dtype=np.float64))
    actual_source = declared_source + b"x"
    payload = _zip_bytes(
        [
            ("source.npy", actual_source),
            ("target.npy", _npy_bytes(np.arange(10, 13, dtype=np.float64))),
        ],
        compression=zipfile.ZIP_DEFLATED,
    )
    path.write_bytes(
        _mask_source_tail_with_declared_prefix(
            payload,
            declared_source=declared_source,
        )
    )
    forbidden_calls: list[str] = []

    def forbidden_zip_parser(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("hidden output must use only the manual deflate route")

    def forbidden(*args: object, **kwargs: object) -> None:
        del args, kwargs
        forbidden_calls.append("called")
        raise AssertionError("CRC and NPY parsing must not run after actual-size mismatch")

    monkeypatch.setattr(zipfile, "ZipFile", forbidden_zip_parser)
    monkeypatch.setattr(npz_safe.zlib, "crc32", forbidden)
    monkeypatch.setattr(npz_safe, "_preflight_npy_member", forbidden)
    monkeypatch.setattr(np, "load", forbidden)

    with pytest.raises(ValueError, match="NPZ archive is malformed"):
        load_npz(path, candidates=(1,))

    assert forbidden_calls == []


def test_public_load_observes_actual_deflate_member_limit_plus_one_before_crc(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import selcal.npz_safe as npz_safe

    path = tmp_path / "actual-member-limit-plus-one.npz"
    actual_source = b"x" * (INPUT_LIMITS_V1.npz_member_uncompressed_bytes + 1)
    declared_source = actual_source[:-1]
    payload = _zip_bytes(
        [
            ("source.npy", actual_source),
            ("target.npy", _npy_bytes(np.arange(10, 13, dtype=np.float64))),
        ],
        compression=zipfile.ZIP_DEFLATED,
    )
    path.write_bytes(
        _mask_source_tail_with_declared_prefix(
            payload,
            declared_source=declared_source,
        )
    )
    forbidden_calls: list[str] = []

    def forbidden(*args: object, **kwargs: object) -> None:
        del args, kwargs
        forbidden_calls.append("called")
        raise AssertionError("CRC and NPY parsing must not run after actual resource overflow")

    monkeypatch.setattr(npz_safe.zlib, "crc32", forbidden)
    monkeypatch.setattr(npz_safe, "_preflight_npy_member", forbidden)
    monkeypatch.setattr(np, "load", forbidden)

    with pytest.raises(ResourceLimitError, match=(
        r"reason=NPZ_MEMBER_UNCOMPRESSED_BYTES limit=8004108 "
        r"observed_at_least=8004109"
    )) as captured:
        load_npz(path, candidates=(1,))

    assert str(path) not in str(captured.value)
    assert forbidden_calls == []


def test_public_load_stops_at_first_real_deflate_eof_without_reading_trailing_blocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import selcal.npz_safe as npz_safe

    path = tmp_path / "early-eof-with-large-tail.npz"
    source_payload = _npy_bytes(np.arange(3, dtype=np.float64))
    source_compressed = _raw_deflate(source_payload) + b"t" * 70_000
    payload = _manual_deflate_npz(
        source_payload=source_payload,
        source_compressed=source_compressed,
    )
    source_member = _preflight_npz_archive(payload).members[0]
    source_slices: list[tuple[int, int]] = []

    class TrackingBytes(bytes):
        def __getitem__(self, key: object) -> object:
            if isinstance(key, slice):
                start, stop, step = key.indices(len(self))
                if (
                    step == 1
                    and source_member.data_offset <= start < source_member.data_end
                    and start < stop
                    and stop <= source_member.data_end
                ):
                    source_slices.append((start, stop))
            return super().__getitem__(key)  # type: ignore[index]

    tracked_payload = TrackingBytes(payload)
    real_decompressobj = zlib.decompressobj
    instances: list[EofTrackingDecompressor] = []

    class EofTrackingDecompressor:
        def __init__(self, wbits: int) -> None:
            self.inner = real_decompressobj(wbits=wbits)
            self.calls_after_eof = 0
            self.reached_eof = False

        @property
        def eof(self) -> bool:
            return self.inner.eof

        @property
        def unused_data(self) -> bytes:
            return self.inner.unused_data

        @property
        def unconsumed_tail(self) -> bytes:
            return self.inner.unconsumed_tail

        def decompress(self, data: bytes, max_length: int) -> bytes:
            if self.inner.eof:
                self.calls_after_eof += 1
                raise AssertionError("decompress called after the first real deflate EOF")
            output = self.inner.decompress(data, max_length)
            self.reached_eof = self.reached_eof or self.inner.eof
            return output

    def tracked_factory(wbits: int = zlib.MAX_WBITS) -> EofTrackingDecompressor:
        instance = EofTrackingDecompressor(wbits)
        instances.append(instance)
        return instance

    def frozen_snapshot(*args: object, **kwargs: object) -> bytes:
        del args, kwargs
        return tracked_payload

    monkeypatch.setattr(npz_safe.zlib, "decompressobj", tracked_factory)
    monkeypatch.setattr(inputs, "read_regular_file_snapshot", frozen_snapshot)

    with pytest.raises(ValueError, match="NPZ archive is malformed"):
        load_npz(path, candidates=(1,))

    assert len(instances) == 1
    assert instances[0].reached_eof
    assert instances[0].calls_after_eof == 0
    assert source_slices == [
        (
            source_member.data_offset,
            source_member.data_offset + 65_536,
        )
    ]


@pytest.mark.parametrize("compressed_boundary", ["truncated", "trailing"])
def test_public_load_requires_true_deflate_eof_and_exact_compressed_range(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    compressed_boundary: str,
) -> None:
    import selcal.npz_safe as npz_safe

    path = tmp_path / f"deflate-{compressed_boundary}.npz"
    source_payload = _npy_bytes(np.arange(3, dtype=np.float64))
    complete = _raw_deflate(source_payload)
    if compressed_boundary == "truncated":
        source_compressed = complete[:-1]
        decompressor = zlib.decompressobj(wbits=-15)
        observed = decompressor.decompress(source_compressed) + decompressor.flush()
        assert observed == source_payload
        assert not decompressor.eof
    else:
        source_compressed = complete + b"hidden-compressed-tail"
        decompressor = zlib.decompressobj(wbits=-15)
        observed = decompressor.decompress(source_compressed) + decompressor.flush()
        assert observed == source_payload
        assert decompressor.eof
        assert decompressor.unused_data == b"hidden-compressed-tail"
    path.write_bytes(
        _manual_deflate_npz(
            source_payload=source_payload,
            source_compressed=source_compressed,
        )
    )
    forbidden_calls: list[str] = []

    def forbidden(*args: object, **kwargs: object) -> None:
        del args, kwargs
        forbidden_calls.append("called")
        raise AssertionError("CRC and NPY parsing require exact true deflate EOF")

    monkeypatch.setattr(npz_safe.zlib, "crc32", forbidden)
    monkeypatch.setattr(npz_safe, "_preflight_npy_member", forbidden)
    monkeypatch.setattr(np, "load", forbidden)

    with pytest.raises(ValueError, match="NPZ archive is malformed"):
        load_npz(path, candidates=(1,))

    assert forbidden_calls == []


def test_public_load_maps_raw_deflate_decoder_error_to_stable_value_error(
    tmp_path: Path,
) -> None:
    path = tmp_path / "invalid-raw-deflate.npz"
    source_payload = _npy_bytes(np.arange(3, dtype=np.float64))
    path.write_bytes(
        _manual_deflate_npz(
            source_payload=source_payload,
            source_compressed=b"not-a-raw-deflate-stream",
        )
    )

    with pytest.raises(ValueError, match="NPZ archive is malformed"):
        load_npz(path, candidates=(1,))


def test_public_load_npz_signature_has_no_limits_override() -> None:
    signature = inspect.signature(load_npz)

    assert tuple(signature.parameters) == ("path", "candidates")
    assert signature.parameters["candidates"].kind is inspect.Parameter.KEYWORD_ONLY
    assert all(
        parameter.kind is not inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def test_public_load_uses_fixed_npz_raw_snapshot_limit_and_reason(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "raw-limit.npz"
    calls: list[tuple[object, int, str]] = []

    def reject(path_arg: object, *, raw_limit: int, reason: str) -> bytes:
        calls.append((path_arg, raw_limit, reason))
        raise_input_resource_limit(reason, limit=raw_limit, observed_at_least=raw_limit + 1)

    monkeypatch.setattr(inputs, "read_regular_file_snapshot", reject)
    with pytest.raises(ResourceLimitError, match=(
        r"reason=NPZ_RAW_BYTES limit=33554432 observed_at_least=33554433"
    )) as captured:
        load_npz(path, candidates=(1,))

    assert calls == [(path, 33_554_432, "NPZ_RAW_BYTES")]
    assert str(path) not in str(captured.value)


def test_public_load_rejects_declared_member_limit_plus_one(tmp_path: Path) -> None:
    path = tmp_path / "member-limit.npz"
    payload = _zip_bytes([
        ("source.npy", _npy_bytes(np.arange(3, dtype=np.float64))),
        ("target.npy", _npy_bytes(np.arange(10, 13, dtype=np.float64))),
    ])
    path.write_bytes(
        _mutate_regular_declared_sizes(payload, source_size=8_004_109, target_size=0)
    )

    with pytest.raises(ResourceLimitError, match=(
        r"reason=NPZ_MEMBER_UNCOMPRESSED_BYTES limit=8004108 observed=8004109"
    )) as captured:
        load_npz(path, candidates=(1,))
    assert str(path) not in str(captured.value)


def test_public_load_rejects_declared_total_limit_plus_one(tmp_path: Path) -> None:
    path = tmp_path / "total-limit.npz"
    payload = _zip_bytes([
        ("source.npy", _npy_bytes(np.arange(3, dtype=np.float64))),
        ("target.npy", _npy_bytes(np.arange(10, 13, dtype=np.float64))),
    ])
    path.write_bytes(
        _mutate_regular_declared_sizes(
            payload,
            source_size=8_004_108,
            target_size=8_004_109,
        )
    )

    with pytest.raises(ResourceLimitError, match=(
        r"reason=NPZ_TOTAL_UNCOMPRESSED_BYTES limit=16008216 observed=16008217"
    )) as captured:
        load_npz(path, candidates=(1,))
    assert str(path) not in str(captured.value)


def test_public_load_rejects_npy_header_limit_plus_one(tmp_path: Path) -> None:
    path = tmp_path / "header-limit.npz"
    path.write_bytes(_zip_bytes([
        ("source.npy", _manual_npy(header_length=4_097, data=b"")),
        ("target.npy", _npy_bytes(np.arange(10, 13, dtype=np.float64))),
    ]))

    with pytest.raises(ResourceLimitError, match=(
        r"reason=NPY_HEADER_BYTES limit=4096 observed=4097"
    )) as captured:
        load_npz(path, candidates=(1,))
    assert str(path) not in str(captured.value)


def test_public_load_rejects_npy_element_limit_plus_one_from_header_only(
    tmp_path: Path,
) -> None:
    path = tmp_path / "element-limit.npz"
    path.write_bytes(_zip_bytes([
        ("source.npy", _manual_npy(shape=(1_000_001,), descr="|u1", data=b"")),
        ("target.npy", _npy_bytes(np.arange(10, 13, dtype=np.float64))),
    ]))

    with pytest.raises(ResourceLimitError, match=(
        r"reason=NPY_ELEMENTS limit=1000000 observed=1000001"
    )) as captured:
        load_npz(path, candidates=(1,))
    assert str(path) not in str(captured.value)
