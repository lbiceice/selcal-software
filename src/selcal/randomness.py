"""Deterministic replicate-local random-index streams for SelCal v2."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any, NamedTuple, Protocol, cast, runtime_checkable

import numpy as np

from selcal.contracts_v2 import V2IntegrityError
from selcal.nulls.executable_base import _resolve_static_callable

NULL_TRANSFORM_STREAM = b"null_transform_v1"
_SEED_DOMAIN = b"SELCAL-SEED\x00\x01"
_MAX_REPLICATES = 1_000_000
_UINT64_LIMIT = 1 << 64


class _HashDigest(Protocol):
    def digest(self) -> bytes: ...


class _RandomCapsuleV2(NamedTuple):
    """Definition-time random kernel shared by execution and verification."""

    create_stream: Callable[[str, int, int], tuple[Any, str]]
    seed_digest: Callable[[tuple[Any, str]], str]
    randbelow: Callable[[tuple[Any, str], int], int]
    seed_digest_for_replicate: Callable[[str, int, int], str]
    external_leaves: tuple[object, ...]


def _freeze_random_capsule_v2(
    sha256: Callable[[bytes], _HashDigest],
    bytes_fromhex: Callable[[str], bytes],
    int_from_bytes: Callable[..., int],
    int_to_bytes: Callable[..., bytes],
    bytes_hex: Callable[[bytes], str],
    pcg64_constructor: Callable[[int], Any],
    pcg64_type: type[Any],
    pcg64_raw_descriptor: Callable[..., Any],
    exact_type: Callable[[Any], type[Any]],
    str_type: type[str],
    int_type: type[int],
    exact_len: Callable[[Any], int],
    exact_any: Callable[[Any], bool],
    range_value: type[range],
    integrity_error_type: type[V2IntegrityError],
    tuple_constructor: Callable[[Any], tuple[Any, str]],
    tuple_type: type[tuple[Any, ...]],
    tuple_getitem: Callable[[tuple[Any, ...], int], object],
    capsule_type: type[_RandomCapsuleV2],
    seed_domain: bytes,
    stream_domain: bytes,
    max_replicates: int,
    uint64_limit: int,
    /,
) -> _RandomCapsuleV2:
    """Freeze the production seed, raw-word, and rejection-sampling kernel."""

    stream_seed_getitem = cast(
        Callable[[tuple[Any, str], int], str],
        tuple_getitem,
    )

    def derive_seed_digest(
        scientific_plan_sha256: str,
        replicate_id: int,
        planned_replicates: int,
        /,
    ) -> bytes:
        if (
            exact_type(scientific_plan_sha256) is not str_type
            or exact_len(scientific_plan_sha256) != 64
            or exact_any(
                character not in "0123456789abcdef" for character in scientific_plan_sha256
            )
        ):
            raise integrity_error_type(
                "scientific_plan_sha256 must be exactly 64 lowercase hexadecimal characters"
            )
        if (
            exact_type(planned_replicates) is not int_type
            or not 1 <= planned_replicates <= max_replicates
        ):
            raise integrity_error_type("planned_replicates must be a built-in int in [1, 1000000]")
        if (
            exact_type(replicate_id) is not int_type
            or replicate_id < 0
            or replicate_id >= planned_replicates
            or replicate_id >= uint64_limit
        ):
            raise integrity_error_type(
                "replicate_id must be a built-in int in [0, planned_replicates) "
                "and fit unsigned uint64"
            )
        preimage = b"".join(
            (
                seed_domain,
                bytes_fromhex(scientific_plan_sha256),
                int_to_bytes(replicate_id, 8, "big", signed=False),
                int_to_bytes(exact_len(stream_domain), 2, "big", signed=False),
                stream_domain,
            )
        )
        return sha256(preimage).digest()

    def create_stream(
        scientific_plan_sha256: str,
        replicate_id: int,
        planned_replicates: int,
        /,
    ) -> tuple[Any, str]:
        seed_digest_bytes = derive_seed_digest(
            scientific_plan_sha256,
            replicate_id,
            planned_replicates,
        )
        seed_uint256 = int_from_bytes(seed_digest_bytes, "big")
        return tuple_constructor(
            (
                pcg64_constructor(seed_uint256),
                bytes_hex(seed_digest_bytes),
            )
        )

    def seed_digest(random_stream: tuple[Any, str], /) -> str:
        if exact_type(random_stream) is not tuple_type or exact_len(random_stream) != 2:
            raise integrity_error_type(
                "random stream must be an exact built-in tuple with exactly two fields"
            )
        digest = stream_seed_getitem(random_stream, 1)
        if (
            exact_type(digest) is not str_type
            or exact_len(digest) != 64
            or exact_any(character not in "0123456789abcdef" for character in digest)
        ):
            raise integrity_error_type("random stream seed digest drifted")
        return digest

    def randbelow(random_stream: tuple[Any, str], bound: int, /) -> int:
        if exact_type(random_stream) is not tuple_type or exact_len(random_stream) != 2:
            raise integrity_error_type(
                "random stream must be an exact built-in tuple with exactly two fields"
            )
        bit_generator = tuple_getitem(random_stream, 0)
        if exact_type(bit_generator) is not pcg64_type:
            raise integrity_error_type("random stream bit generator drifted")
        if exact_type(bound) is not int_type or bound < 1:
            raise integrity_error_type("randbelow bound must be a positive built-in int")
        if bound == 1:
            return 0

        bit_count = (bound - 1).bit_length()
        words_per_attempt = (bit_count + 63) // 64
        low_bit_mask = (1 << bit_count) - 1
        while True:
            candidate = 0
            for _ in range_value(words_per_attempt):
                word = int_type(pcg64_raw_descriptor(bit_generator))
                if word < 0 or word >= uint64_limit:
                    raise integrity_error_type("raw64 word must be a built-in int in [0, 2**64)")
                candidate = (candidate << 64) | word
            candidate &= low_bit_mask
            if candidate < bound:
                return candidate

    capsule: _RandomCapsuleV2

    def seed_digest_for_replicate(
        scientific_plan_sha256: str,
        replicate_id: int,
        planned_replicates: int,
        /,
    ) -> str:
        if (
            exact_type(capsule) is not capsule_type
            or tuple_getitem(capsule, 3) is not seed_digest_for_replicate
        ):
            raise integrity_error_type("random capsule seed oracle identity drifted")
        return bytes_hex(
            derive_seed_digest(
                scientific_plan_sha256,
                replicate_id,
                planned_replicates,
            )
        )

    external_leaves = (
        sha256,
        bytes_fromhex,
        int_from_bytes,
        int_to_bytes,
        bytes_hex,
        pcg64_constructor,
        pcg64_raw_descriptor,
        seed_domain,
        stream_domain,
        max_replicates,
        uint64_limit,
        range_value,
        tuple_constructor,
        tuple_type,
        tuple_getitem,
    )
    capsule = capsule_type(
        create_stream=create_stream,
        seed_digest=seed_digest,
        randbelow=randbelow,
        seed_digest_for_replicate=seed_digest_for_replicate,
        external_leaves=external_leaves,
    )
    return capsule


_RANDOM_CAPSULE_V2 = _freeze_random_capsule_v2(
    cast(Callable[[bytes], _HashDigest], hashlib.sha256),
    bytes.fromhex,
    int.from_bytes,
    int.to_bytes,
    bytes.hex,
    np.random.PCG64,
    np.random.PCG64,
    np.random.PCG64.random_raw,
    type,
    str,
    int,
    len,
    any,
    range,
    V2IntegrityError,
    cast(Callable[[Any], tuple[Any, str]], tuple),
    tuple,
    cast(Callable[[tuple[Any, ...], int], object], tuple.__getitem__),
    _RandomCapsuleV2,
    _SEED_DOMAIN,
    NULL_TRANSFORM_STREAM,
    _MAX_REPLICATES,
    _UINT64_LIMIT,
)
del _freeze_random_capsule_v2


@runtime_checkable
class Raw64Source(Protocol):
    """A source returning one raw word at a time for scripted conformance tests."""

    def random_raw(self) -> object: ...


def _require_plan_digest(value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise V2IntegrityError(
            "scientific_plan_sha256 must be exactly 64 lowercase hexadecimal characters"
        )
    return value


def _require_planned_replicates(value: object) -> int:
    if type(value) is not int or not 1 <= value <= _MAX_REPLICATES:
        raise V2IntegrityError(
            "planned_replicates must be a built-in int in [1, 1000000]"
        )
    return value


def _require_replicate_id(value: object, *, planned_replicates: int) -> int:
    if (
        type(value) is not int
        or value < 0
        or value >= planned_replicates
        or value >= _UINT64_LIMIT
    ):
        raise V2IntegrityError(
            "replicate_id must be a built-in int in [0, planned_replicates) "
            "and fit unsigned uint64"
        )
    return value


def _require_bound(value: object) -> int:
    if type(value) is not int or value < 1:
        raise V2IntegrityError("randbelow bound must be a positive built-in int")
    return value


def _require_raw64_word(value: object) -> int:
    if type(value) is not int or not 0 <= value < _UINT64_LIMIT:
        raise V2IntegrityError(
            "raw64 word must be a built-in int in [0, 2**64)"
        )
    return value


def _uniform_randbelow_from_next(
    next_word: Callable[[], object],
    bound: object,
) -> int:
    exact_bound = _require_bound(bound)
    if exact_bound == 1:
        return 0

    bit_count = (exact_bound - 1).bit_length()
    words_per_attempt = (bit_count + 63) // 64
    low_bit_mask = (1 << bit_count) - 1
    while True:
        candidate = 0
        for _ in range(words_per_attempt):
            word = _require_raw64_word(next_word())
            candidate = (candidate << 64) | word
        candidate &= low_bit_mask
        if candidate < exact_bound:
            return candidate


def uniform_randbelow(source: Raw64Source, bound: int, /) -> int:
    """Sample uniformly below an arbitrary positive bound by frozen rejection."""

    next_word = cast(
        Callable[[], object],
        _resolve_static_callable(
            source,
            method_name="random_raw",
            subject="source",
        ),
    )
    return _uniform_randbelow_from_next(next_word, bound)


class ReplicateRandomSource:
    """One deterministic PCG64 stream owned by exactly one replicate ID."""

    __slots__ = ("_bitgen", "_seed_digest_sha256")

    _bitgen: np.random.PCG64
    _seed_digest_sha256: str

    def __init__(
        self,
        scientific_plan_sha256: str,
        replicate_id: int,
        planned_replicates: int,
    ) -> None:
        plan_digest = _require_plan_digest(scientific_plan_sha256)
        replicate_count = _require_planned_replicates(planned_replicates)
        exact_replicate_id = _require_replicate_id(
            replicate_id,
            planned_replicates=replicate_count,
        )
        stream = NULL_TRANSFORM_STREAM
        preimage = b"".join(
            (
                _SEED_DOMAIN,
                bytes.fromhex(plan_digest),
                exact_replicate_id.to_bytes(8, "big", signed=False),
                len(stream).to_bytes(2, "big", signed=False),
                stream,
            )
        )
        seed_digest = hashlib.sha256(preimage).digest()
        self._seed_digest_sha256 = seed_digest.hex()
        seed_uint256 = int.from_bytes(seed_digest, "big")
        self._bitgen = np.random.PCG64(seed_uint256)

    @property
    def seed_digest_sha256(self) -> str:
        """Canonical persistent identity of this replicate's random stream."""

        return self._seed_digest_sha256

    def _next_raw64(self) -> int:
        word = int(self._bitgen.random_raw())
        return _require_raw64_word(word)

    def randbelow(self, bound: int, /) -> int:
        """Return one unbiased integer in ``range(bound)``."""

        return _uniform_randbelow_from_next(self._next_raw64, bound)


__all__ = [
    "NULL_TRANSFORM_STREAM",
    "Raw64Source",
    "ReplicateRandomSource",
    "uniform_randbelow",
]
