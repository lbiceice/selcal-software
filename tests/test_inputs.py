from __future__ import annotations

import hashlib
import io
import os
import stat
import warnings
import zipfile
from dataclasses import FrozenInstanceError, fields, is_dataclass

import numpy as np
import pytest
from _platform_support import symlink_or_skip

from selcal.canonical import semantic_input_sha256
from selcal.contracts import SeriesPair
from selcal.inputs import LoadedInput, load_csv, load_npz


def _valid_csv_bytes() -> bytes:
    return b"source,target\n1,10\n2,20\n3,30\n4,40\n"


def _npz_bytes(source: np.ndarray, target: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.savez(buffer, source=source, target=target)
    return buffer.getvalue()


def _npy_bytes(values: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, values, allow_pickle=False)
    return buffer.getvalue()


def test_csv_loads_exact_order_and_hashes_original_bytes(tmp_path: pytest.TempPathFactory) -> None:
    path = tmp_path / "observations.csv"
    payload = b"driver,response\n2,20\n1,10\n3,30\n"
    path.write_bytes(payload)

    loaded = load_csv(
        path,
        source_column="driver",
        target_column="response",
        candidates=(1,),
    )

    assert loaded.source_format == "csv"
    assert loaded.raw_input_sha256 == hashlib.sha256(payload).hexdigest()
    np.testing.assert_array_equal(loaded.pair.source, np.array([2.0, 1.0, 3.0]))
    np.testing.assert_array_equal(loaded.pair.target, np.array([20.0, 10.0, 30.0]))


def test_csv_missing_source_names_one_based_physical_row(tmp_path: pytest.TempPathFactory) -> None:
    path = tmp_path / "missing.csv"
    path.write_bytes(b"driver,response\n2,20\n,10\n3,30\n")

    with pytest.raises(ValueError, match=r"row 3\b"):
        load_csv(
            path,
            source_column="driver",
            target_column="response",
            candidates=(1,),
        )


def test_csv_role_swap_changes_pair_and_semantic_hash_but_candidate_order_does_not(
    tmp_path: pytest.TempPathFactory,
) -> None:
    path = tmp_path / "roles.csv"
    path.write_bytes(b"driver,response\n2,20\n1,10\n3,30\n")

    forward = load_csv(
        path,
        source_column="driver",
        target_column="response",
        candidates=(2, 1),
    )
    reordered = load_csv(
        path,
        source_column="driver",
        target_column="response",
        candidates=(1, 2),
    )
    swapped = load_csv(
        path,
        source_column="response",
        target_column="driver",
        candidates=(1, 2),
    )

    assert forward.semantic_input_sha256 == reordered.semantic_input_sha256
    assert forward.semantic_input_sha256 != swapped.semantic_input_sha256
    np.testing.assert_array_equal(swapped.pair.source, forward.pair.target)
    np.testing.assert_array_equal(swapped.pair.target, forward.pair.source)


def test_equivalent_csv_and_npz_have_distinct_raw_and_equal_semantic_hashes(
    tmp_path: pytest.TempPathFactory,
) -> None:
    csv_path = tmp_path / "observations.csv"
    npz_path = tmp_path / "observations.npz"
    csv_path.write_bytes(b"source,target\n1,10\n2,20\n3,30\n")
    np.savez(
        npz_path,
        source=np.array([1.0, 2.0, 3.0]),
        target=np.array([10.0, 20.0, 30.0]),
    )

    from_csv = load_csv(
        csv_path,
        source_column="source",
        target_column="target",
        candidates=(2, 1),
    )
    from_npz = load_npz(npz_path, candidates=(1, 2))

    assert from_csv.raw_input_sha256 != from_npz.raw_input_sha256
    assert from_csv.semantic_input_sha256 == from_npz.semantic_input_sha256


def test_loaded_input_is_frozen_slotted_path_free_and_explicitly_unhashable(
    tmp_path: pytest.TempPathFactory,
) -> None:
    path = tmp_path / "input.csv"
    path.write_bytes(_valid_csv_bytes())
    loaded = load_csv(
        path,
        source_column="source",
        target_column="target",
        candidates=(1,),
    )
    loaded_again = load_csv(
        path,
        source_column="source",
        target_column="target",
        candidates=(1,),
    )

    assert is_dataclass(LoadedInput)
    assert {field.name for field in fields(LoadedInput)} == {
        "pair",
        "raw_input_sha256",
        "semantic_input_sha256",
        "source_format",
    }
    assert not hasattr(loaded, "__dict__")
    assert loaded == loaded_again
    assert LoadedInput.__hash__ is None
    with pytest.raises(TypeError, match="unhashable"):
        hash(loaded)
    with pytest.raises(FrozenInstanceError):
        loaded.source_format = "npz"  # type: ignore[misc]


class _StringSubclass(str):
    pass


def _direct_loaded_input(**changes: object) -> LoadedInput:
    values: dict[str, object] = {
        "pair": SeriesPair(np.array([1.0, 2.0]), np.array([3.0, 4.0])),
        "raw_input_sha256": "a" * 64,
        "semantic_input_sha256": "b" * 64,
        "source_format": "csv",
    }
    values.update(changes)
    return LoadedInput(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"pair": object()}, "pair"),
        ({"raw_input_sha256": "a" * 63}, "raw_input_sha256"),
        ({"raw_input_sha256": "A" * 64}, "raw_input_sha256"),
        ({"raw_input_sha256": "g" * 64}, "raw_input_sha256"),
        ({"raw_input_sha256": _StringSubclass("a" * 64)}, "raw_input_sha256"),
        ({"semantic_input_sha256": "b" * 63}, "semantic_input_sha256"),
        ({"semantic_input_sha256": "B" * 64}, "semantic_input_sha256"),
        ({"semantic_input_sha256": _StringSubclass("b" * 64)}, "semantic_input_sha256"),
        ({"source_format": "CSV"}, "source_format"),
        ({"source_format": _StringSubclass("csv")}, "source_format"),
        ({"source_format": 1}, "source_format"),
    ],
)
def test_direct_loaded_input_rejects_malformed_contract_fields(
    changes: dict[str, object],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        _direct_loaded_input(**changes)


def test_direct_loaded_input_equality_is_total_for_valid_objects() -> None:
    first = _direct_loaded_input()
    equal = _direct_loaded_input()
    different = _direct_loaded_input(source_format="npz")

    assert first == equal
    assert first != different
    assert (first == object()) is False


@pytest.mark.parametrize(
    ("source_column", "target_column", "error_type"),
    [
        ("", "target", ValueError),
        ("source", "", ValueError),
        ("source", "source", ValueError),
        (_StringSubclass("source"), "target", TypeError),
        ("source", _StringSubclass("target"), TypeError),
    ],
)
def test_csv_column_arguments_must_be_distinct_actual_nonempty_strings(
    tmp_path: pytest.TempPathFactory,
    source_column: str,
    target_column: str,
    error_type: type[Exception],
) -> None:
    path = tmp_path / "columns.csv"
    path.write_bytes(_valid_csv_bytes())

    with pytest.raises(error_type, match=r"source_column|target_column|distinct"):
        load_csv(
            path,
            source_column=source_column,
            target_column=target_column,
            candidates=(1,),
        )


def test_csv_column_arguments_must_exactly_match_header_set(
    tmp_path: pytest.TempPathFactory,
) -> None:
    path = tmp_path / "columns.csv"
    path.write_bytes(_valid_csv_bytes())

    with pytest.raises(ValueError, match="header"):
        load_csv(
            path,
            source_column="source",
            target_column="response",
            candidates=(1,),
        )


@pytest.mark.parametrize(
    "payload",
    [
        b"source,source\n1,2\n",
        b"source,target,extra\n1,2,3\n",
        b"source,\n1,2\n",
        b"source\n1\n",
        b"",
    ],
)
def test_csv_requires_exactly_two_nonempty_unique_headers(
    tmp_path: pytest.TempPathFactory,
    payload: bytes,
) -> None:
    path = tmp_path / "bad-header.csv"
    path.write_bytes(payload)

    with pytest.raises(ValueError, match=r"header|row 1"):
        load_csv(
            path,
            source_column="source",
            target_column="target",
            candidates=(1,),
        )


@pytest.mark.parametrize(
    ("payload", "row_number"),
    [
        (b"source,target\n1,2\n\n3,4\n", 3),
        (b"source,target\n1,2\n,4\n", 3),
        (b"source,target\n1,2\n3\n", 3),
        (b"source,target\n1,2\n3,4,5\n", 3),
    ],
)
def test_csv_rejects_blank_and_structurally_incomplete_rows(
    tmp_path: pytest.TempPathFactory,
    payload: bytes,
    row_number: int,
) -> None:
    path = tmp_path / "bad-row.csv"
    path.write_bytes(payload)

    with pytest.raises(ValueError, match=rf"row {row_number}\b"):
        load_csv(
            path,
            source_column="source",
            target_column="target",
            candidates=(1,),
        )


@pytest.mark.parametrize(
    "bad_row",
    [
        b"source,target",
        b"not-a-number,4",
        b"3,nan",
        b"inf,4",
        b"3,-inf",
    ],
)
def test_csv_rejects_repeated_header_nonnumeric_and_nonfinite_cells_with_row(
    tmp_path: pytest.TempPathFactory,
    bad_row: bytes,
) -> None:
    path = tmp_path / "bad-value.csv"
    path.write_bytes(b"source,target\n1,2\n" + bad_row + b"\n4,5\n")

    with pytest.raises(ValueError, match=r"row 3\b"):
        load_csv(
            path,
            source_column="source",
            target_column="target",
            candidates=(1,),
        )


def test_csv_requires_at_least_one_data_row(tmp_path: pytest.TempPathFactory) -> None:
    path = tmp_path / "header-only.csv"
    path.write_bytes(b"source,target\n")

    with pytest.raises(ValueError, match="data row"):
        load_csv(
            path,
            source_column="source",
            target_column="target",
            candidates=(1,),
        )


def test_csv_requires_series_longer_than_maximum_candidate(
    tmp_path: pytest.TempPathFactory,
) -> None:
    path = tmp_path / "short.csv"
    path.write_bytes(b"source,target\n1,2\n3,4\n")

    with pytest.raises(ValueError, match=r"length|maximum|max"):
        load_csv(
            path,
            source_column="source",
            target_column="target",
            candidates=(2, 1),
        )


def test_csv_decodes_utf8_strictly(tmp_path: pytest.TempPathFactory) -> None:
    path = tmp_path / "invalid-utf8.csv"
    path.write_bytes(b"source,target\n1,2\n\xff,4\n")

    with pytest.raises(UnicodeDecodeError):
        load_csv(
            path,
            source_column="source",
            target_column="target",
            candidates=(1,),
        )


@pytest.mark.parametrize("bad_role", ["source", "target"])
@pytest.mark.parametrize("underflow_literal", ["1e-4000", "2e-4000"])
def test_csv_rejects_distinct_nonzero_values_that_underflow_to_float64_zero(
    tmp_path: pytest.TempPathFactory,
    bad_role: str,
    underflow_literal: str,
) -> None:
    path = tmp_path / f"underflow-{bad_role}-{underflow_literal}.csv"
    source_value = underflow_literal if bad_role == "source" else "2"
    target_value = underflow_literal if bad_role == "target" else "20"
    path.write_text(
        f"source,target\n1,10\n{source_value},{target_value}\n3,30\n",
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(ValueError, match=r"underflow|float64") as error:
        load_csv(
            path,
            source_column="source",
            target_column="target",
            candidates=(1,),
        )

    assert "row 3" in str(error.value)


def test_csv_accepts_explicit_zeros_and_representable_float64_subnormal(
    tmp_path: pytest.TempPathFactory,
) -> None:
    path = tmp_path / "zeros-and-subnormal.csv"
    path.write_bytes(b"source,target\n0,10\n-0,20\n5e-324,30\n")

    loaded = load_csv(
        path,
        source_column="source",
        target_column="target",
        candidates=(1,),
    )

    np.testing.assert_array_equal(
        loaded.pair.source,
        np.array([0.0, 0.0, np.nextafter(0.0, 1.0)]),
    )


def test_csv_does_not_reclassify_missing_file_as_analytic_value_error(
    tmp_path: pytest.TempPathFactory,
) -> None:
    with pytest.raises(FileNotFoundError):
        load_csv(
            tmp_path / "missing.csv",
            source_column="source",
            target_column="target",
            candidates=(1,),
        )


def test_npz_preserves_exact_element_order_and_hashes_entire_file(
    tmp_path: pytest.TempPathFactory,
) -> None:
    path = tmp_path / "input.npz"
    source = np.array([2.0, 1.0, 3.0])
    target = np.array([20.0, 10.0, 30.0])
    np.savez(path, source=source, target=target)

    loaded = load_npz(path, candidates=(1,))

    assert loaded.source_format == "npz"
    assert loaded.raw_input_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    np.testing.assert_array_equal(loaded.pair.source, source)
    np.testing.assert_array_equal(loaded.pair.target, target)


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX symlink replacement semantics")
def test_npz_hash_and_arrays_come_from_same_single_byte_snapshot(
    tmp_path: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import selcal.input_resources as input_resources

    path = tmp_path / "changing.npz"
    original_path = tmp_path / "original.npz"
    replacement_path = tmp_path / "replacement.npz"
    replacement_link = tmp_path / "replacement-link.npz"
    old_source = np.array([1.0, 2.0, 3.0])
    old_target = np.array([10.0, 20.0, 30.0])
    old_payload = _npz_bytes(old_source, old_target)
    new_payload = _npz_bytes(
        np.array([101.0, 102.0, 103.0]),
        np.array([110.0, 120.0, 130.0]),
    )
    original_path.write_bytes(old_payload)
    replacement_path.write_bytes(new_payload)
    symlink_or_skip(path, original_path)
    symlink_or_skip(replacement_link, replacement_path)
    real_open = input_resources._os_open
    real_read = input_resources._os_read
    observed_opens: list[object] = []
    observed_read_requests: list[int] = []
    replaced = False

    def observe_open(path_arg: object, flags: int) -> int:
        observed_opens.append(path_arg)
        return real_open(path_arg, flags)  # type: ignore[arg-type]

    def read_then_replace(descriptor: int, size: int) -> bytes:
        nonlocal replaced
        observed_read_requests.append(size)
        chunk = real_read(descriptor, size)
        if chunk and not replaced:
            os.replace(replacement_link, path)
            replaced = True
        return chunk

    monkeypatch.setattr(input_resources, "_os_open", observe_open)
    monkeypatch.setattr(input_resources, "_os_read", read_then_replace)

    loaded = load_npz(path, candidates=(1,))

    assert observed_opens == [path]
    assert len(observed_read_requests) == 2
    assert replaced
    assert loaded.raw_input_sha256 == hashlib.sha256(old_payload).hexdigest()
    np.testing.assert_array_equal(loaded.pair.source, old_source)
    np.testing.assert_array_equal(loaded.pair.target, old_target)
    assert path.read_bytes() == new_payload


def test_npz_single_platform_neutral_open_freezes_values_and_hashes(
    tmp_path: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import selcal.input_resources as input_resources

    path = tmp_path / "platform-neutral.npz"
    old_source = np.array([1.0, 2.0, 3.0])
    old_target = np.array([10.0, 20.0, 30.0])
    old_payload = _npz_bytes(old_source, old_target)
    new_payload = _npz_bytes(
        np.array([101.0, 102.0, 103.0]),
        np.array([110.0, 120.0, 130.0]),
    )
    current_payload = [old_payload]
    descriptor_payloads: dict[int, bytes] = {}
    descriptor_offsets: dict[int, int] = {}
    open_calls: list[tuple[object, int]] = []
    read_calls: list[tuple[int, int]] = []
    close_calls: list[int] = []

    def simulated_open(path_arg: object, flags: int) -> int:
        open_calls.append((path_arg, flags))
        descriptor = 9_000 + len(open_calls)
        descriptor_payloads[descriptor] = current_payload[0]
        descriptor_offsets[descriptor] = 0
        return descriptor

    def simulated_fstat(descriptor: int) -> os.stat_result:
        payload = descriptor_payloads[descriptor]
        return os.stat_result(
            (stat.S_IFREG | 0o600, 0, 0, descriptor, 0, 0, len(payload), 0, 0, 0)
        )

    def simulated_read(descriptor: int, size: int) -> bytes:
        read_calls.append((descriptor, size))
        payload = descriptor_payloads[descriptor]
        start = descriptor_offsets[descriptor]
        chunk = payload[start : start + size]
        descriptor_offsets[descriptor] = start + len(chunk)
        if chunk:
            current_payload[0] = new_payload
        return chunk

    def simulated_close(descriptor: int) -> None:
        close_calls.append(descriptor)

    monkeypatch.setattr(input_resources, "_os_open", simulated_open)
    monkeypatch.setattr(input_resources, "_os_fstat", simulated_fstat)
    monkeypatch.setattr(input_resources, "_os_read", simulated_read)
    monkeypatch.setattr(input_resources, "_os_close", simulated_close)

    loaded = load_npz(path, candidates=(1,))
    expected_pair = SeriesPair(source=old_source, target=old_target)

    assert len(open_calls) == 1
    assert open_calls[0][0] == path
    assert {descriptor for descriptor, _size in read_calls} == {9_001}
    assert close_calls == [9_001]
    assert current_payload == [new_payload]
    np.testing.assert_array_equal(loaded.pair.source, expected_pair.source)
    np.testing.assert_array_equal(loaded.pair.target, expected_pair.target)
    assert loaded.raw_input_sha256 == hashlib.sha256(old_payload).hexdigest()
    assert loaded.semantic_input_sha256 == semantic_input_sha256(expected_pair, (1,))


@pytest.mark.parametrize("keys", [("target",), ("source", "target", "extra")])
def test_npz_requires_exact_source_and_target_keys(
    tmp_path: pytest.TempPathFactory,
    keys: tuple[str, ...],
) -> None:
    path = tmp_path / "keys.npz"
    arrays = {key: np.arange(4, dtype=np.float64) for key in keys}
    np.savez(path, **arrays)

    with pytest.raises(ValueError, match="keys"):
        load_npz(path, candidates=(1,))


def test_npz_rejects_duplicate_archive_keys(tmp_path: pytest.TempPathFactory) -> None:
    path = tmp_path / "duplicate.npz"
    buffer = io.BytesIO()
    np.save(buffer, np.arange(4, dtype=np.float64), allow_pickle=False)
    payload = buffer.getvalue()

    with zipfile.ZipFile(path, mode="w") as archive:
        archive.writestr("source.npy", payload)
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("source.npy", payload)
        archive.writestr("target.npy", payload)

    with pytest.raises(ValueError, match=r"unique|keys"):
        load_npz(path, candidates=(1,))


@pytest.mark.parametrize(
    "malformed_source",
    [
        pytest.param(b"plain bytes, not an array", id="non-array-member"),
        pytest.param(b"\x93NUMPY", id="truncated-npy-member"),
    ],
)
def test_npz_rejects_malformed_exact_key_members_with_stable_value_error(
    tmp_path: pytest.TempPathFactory,
    malformed_source: bytes,
) -> None:
    path = tmp_path / "malformed-member.npz"
    with zipfile.ZipFile(path, mode="w") as archive:
        archive.writestr("source", malformed_source)
        archive.writestr("target", _npy_bytes(np.array([10.0, 20.0, 30.0])))

    with pytest.raises(ValueError, match=r"NPZ.*(?:malformed|ndarray|array member)"):
        load_npz(path, candidates=(1,))


def test_npz_rejects_malformed_archive_with_stable_value_error(
    tmp_path: pytest.TempPathFactory,
) -> None:
    path = tmp_path / "malformed-archive.npz"
    path.write_bytes(b"not a ZIP or NPY payload")

    with pytest.raises(ValueError, match=r"NPZ archive.*malformed"):
        load_npz(path, candidates=(1,))


def test_npz_does_not_reclassify_memory_exhaustion_as_malformed_data(
    tmp_path: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "valid.npz"
    path.write_bytes(_npz_bytes(np.arange(3), np.arange(10, 13)))

    def raise_memory_error(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise MemoryError("simulated exhaustion")

    monkeypatch.setattr(np, "load", raise_memory_error)

    with pytest.raises(MemoryError, match="simulated exhaustion"):
        load_npz(path, candidates=(1,))


def test_npz_rejects_object_or_pickled_arrays(tmp_path: pytest.TempPathFactory) -> None:
    path = tmp_path / "objects.npz"
    np.savez(
        path,
        source=np.array([object(), object()], dtype=object),
        target=np.array([1.0, 2.0]),
    )

    with pytest.raises(ValueError, match=r"object|pickle"):
        load_npz(path, candidates=(1,))


@pytest.mark.parametrize("dtype", ["int16", "uint16", "float16", "float32", "float64"])
def test_npz_preserves_allowed_real_numeric_kinds_as_immutable_float64(
    tmp_path: pytest.TempPathFactory,
    dtype: str,
) -> None:
    path = tmp_path / f"allowed-{dtype}.npz"
    source = np.array([1, 2, 3], dtype=dtype)
    target = np.array([10, 20, 30], dtype=dtype)
    np.savez(path, source=source, target=target)

    loaded = load_npz(path, candidates=(1,))

    assert loaded.pair.source.dtype == np.float64
    assert loaded.pair.target.dtype == np.float64
    assert not loaded.pair.source.flags.writeable
    assert not loaded.pair.target.flags.writeable
    np.testing.assert_array_equal(loaded.pair.source, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(loaded.pair.target, np.array([10.0, 20.0, 30.0]))


@pytest.mark.parametrize("bad_role", ["source", "target"])
@pytest.mark.parametrize(
    ("dtype", "lossy_value"),
    [
        pytest.param("int64", 2**53 + 1, id="signed-int64"),
        pytest.param("uint64", 9223372036854775809, id="reviewer-uint64-a"),
        pytest.param("uint64", 9223372036854775810, id="reviewer-uint64-b"),
    ],
)
def test_npz_rejects_each_lossy_integer_before_semantic_hashing(
    tmp_path: pytest.TempPathFactory,
    bad_role: str,
    dtype: str,
    lossy_value: int,
) -> None:
    path = tmp_path / f"lossy-{dtype}-{lossy_value}-{bad_role}.npz"
    invalid_values = np.array([1, lossy_value, 3], dtype=dtype)
    valid_values = np.array([10, 20, 30], dtype=np.int64)
    source = invalid_values if bad_role == "source" else valid_values
    target = invalid_values if bad_role == "target" else valid_values
    np.savez(path, source=source, target=target)

    with pytest.raises(ValueError, match=r"lossless|float64"):
        load_npz(path, candidates=(1,))


def test_reviewer_uint64_collision_values_are_distinct_before_rejection() -> None:
    first = 9223372036854775809
    second = 9223372036854775810

    assert first != second
    assert float(first) == float(second)


@pytest.mark.parametrize(
    ("dtype", "exact_value"),
    [("int64", 2**53), ("uint64", 2**63)],
)
def test_npz_accepts_exactly_representable_large_integers(
    tmp_path: pytest.TempPathFactory,
    dtype: str,
    exact_value: int,
) -> None:
    path = tmp_path / f"exact-{dtype}.npz"
    values = np.array([1, exact_value, 3], dtype=dtype)
    np.savez(path, source=values, target=values)

    loaded = load_npz(path, candidates=(1,))

    assert loaded.pair.source[1] == float(exact_value)
    assert loaded.pair.target[1] == float(exact_value)


def test_npz_rejects_extended_float_before_float64_narrowing(
    tmp_path: pytest.TempPathFactory,
) -> None:
    if np.dtype(np.longdouble).itemsize <= 8:
        pytest.skip("platform has no extended floating dtype")
    path = tmp_path / "extended-float.npz"
    values = np.array([1.0, 2.0, 3.0], dtype=np.longdouble)
    np.savez(path, source=values, target=values)

    with pytest.raises(ValueError, match=r"float64|64-bit|itemsize"):
        load_npz(path, candidates=(1,))


@pytest.mark.parametrize("bad_role", ["source", "target"])
@pytest.mark.parametrize(
    "invalid_values",
    [
        pytest.param(np.array(["1", "2", "3"], dtype="U1"), id="unicode"),
        pytest.param(np.array([b"1", b"2", b"3"], dtype="S1"), id="byte-string"),
        pytest.param(np.array([True, False, True], dtype=np.bool_), id="boolean"),
        pytest.param(
            np.array([1.0 + 4.0j, 2.0 + 5.0j, 3.0 + 6.0j]),
            id="complex",
        ),
    ],
)
def test_npz_rejects_non_real_numeric_original_dtypes_without_warning(
    tmp_path: pytest.TempPathFactory,
    invalid_values: np.ndarray,
    bad_role: str,
) -> None:
    path = tmp_path / f"invalid-{invalid_values.dtype.kind}-{bad_role}.npz"
    valid_values = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    source = invalid_values if bad_role == "source" else valid_values
    target = invalid_values if bad_role == "target" else valid_values
    np.savez(path, source=source, target=target)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match=r"real numeric|dtype"):
            load_npz(path, candidates=(1,))


@pytest.mark.parametrize(
    ("source", "target", "message"),
    [
        (np.ones((2, 2)), np.ones((2, 2)), "one-dimensional"),
        (np.arange(3), np.arange(4), "equal shape"),
        (np.array([]), np.array([]), "non-empty"),
        (np.array([1.0, np.nan]), np.array([2.0, 3.0]), "finite"),
        (np.array([1.0, 2.0]), np.array([2.0, np.inf]), "finite"),
    ],
)
def test_npz_uses_series_pair_shape_and_finiteness_contract(
    tmp_path: pytest.TempPathFactory,
    source: np.ndarray,
    target: np.ndarray,
    message: str,
) -> None:
    path = tmp_path / "invalid-array.npz"
    np.savez(path, source=source, target=target)

    with pytest.raises(ValueError, match=message):
        load_npz(path, candidates=(1,))


def test_npz_requires_series_longer_than_maximum_candidate(
    tmp_path: pytest.TempPathFactory,
) -> None:
    path = tmp_path / "short.npz"
    np.savez(path, source=np.arange(3), target=np.arange(10, 13))

    with pytest.raises(ValueError, match=r"length|maximum|max"):
        load_npz(path, candidates=(3, 1))


@pytest.mark.parametrize("bad_candidates", [(1, 1), (True, 2), (), (0, 1)])
def test_loaders_share_candidate_validation(
    tmp_path: pytest.TempPathFactory,
    bad_candidates: tuple[int, ...],
) -> None:
    csv_path = tmp_path / "input.csv"
    npz_path = tmp_path / "input.npz"
    csv_path.write_bytes(_valid_csv_bytes())
    np.savez(npz_path, source=np.arange(4), target=np.arange(10, 14))

    with pytest.raises(ValueError, match="candidates"):
        load_csv(
            csv_path,
            source_column="source",
            target_column="target",
            candidates=bad_candidates,
        )
    with pytest.raises(ValueError, match="candidates"):
        load_npz(npz_path, candidates=bad_candidates)


def test_npz_does_not_reclassify_missing_file_as_analytic_value_error(
    tmp_path: pytest.TempPathFactory,
) -> None:
    with pytest.raises(FileNotFoundError):
        load_npz(tmp_path / "missing.npz", candidates=(1,))
