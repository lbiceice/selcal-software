from __future__ import annotations

import ast
import inspect
from collections.abc import Iterator

import numpy as np
import pytest

import selcal.calibration_v2 as calibration_v2
import selcal.randomness as randomness
from selcal.contracts_v2 import V2IntegrityError
from selcal.nulls.executable_base import UniformIndexSource
from selcal.randomness import (
    Raw64Source,
    ReplicateRandomSource,
    _RandomCapsuleV2,
    uniform_randbelow,
)

ZERO_PLAN_SHA256 = "0" * 64
REAL_PLAN_SHA256 = "dc7462598231451b36a94e2586037aa921a8159b2eabebb5fe5c81dcafa48931"
EXPECTED_COMPATIBILITY_RANDOM_VECTOR = (
    "a8dd219a1e348a103ff10bdad3b9ae8d3cff53032be90a3603734530f51fb82a",
    (1874999576356411599, 4645183139923492514, 4210660991089764159),
    (1, 2, 6, 17099354796136167930),
)


class ScriptedRaw64Source:
    def __init__(self, words: list[object]) -> None:
        self._words: Iterator[object] = iter(words)
        self.words_consumed = 0

    def random_raw(self) -> object:
        word = next(self._words)
        self.words_consumed += 1
        return word


def _compatibility_random_vector() -> (
    tuple[
        str,
        tuple[int, int, int],
        tuple[int, int, int, int],
    ]
):
    seed_source = ReplicateRandomSource(REAL_PLAN_SHA256, 0, 2)
    raw_source = ReplicateRandomSource(REAL_PLAN_SHA256, 0, 2)
    bounded_source = ReplicateRandomSource(REAL_PLAN_SHA256, 0, 2)
    return (
        seed_source.seed_digest_sha256,
        tuple(raw_source._next_raw64() for _ in range(3)),  # type: ignore[return-value]
        tuple(  # type: ignore[return-value]
            bounded_source.randbelow(bound) for bound in (2, 3, 17, 2**64 + 1)
        ),
    )


def test_compatibility_api_post_seal_seed_raw64_and_bounded_vector_is_committed() -> None:
    """Keep the compatibility KAT distinct from the production executor proof."""

    assert _compatibility_random_vector() == EXPECTED_COMPATIBILITY_RANDOM_VECTOR


def test_sealed_production_capsule_has_an_independent_primitive_known_answer_vector() -> None:
    executor_capsule = inspect.getclosurevars(
        calibration_v2._execute_replicates
    ).nonlocals["random_capsule"]
    create_stream = tuple.__getitem__(executor_capsule, 0)
    seed_digest = tuple.__getitem__(executor_capsule, 1)
    randbelow = tuple.__getitem__(executor_capsule, 2)
    seed_stream = create_stream(REAL_PLAN_SHA256, 0, 2)
    raw_stream = create_stream(REAL_PLAN_SHA256, 0, 2)
    bounded_stream = create_stream(REAL_PLAN_SHA256, 0, 2)

    assert type(executor_capsule) is _RandomCapsuleV2
    assert executor_capsule is randomness._RANDOM_CAPSULE_V2
    assert (
        seed_digest(seed_stream)
        == "a8dd219a1e348a103ff10bdad3b9ae8d3cff53032be90a3603734530f51fb82a"
    )
    assert randbelow(raw_stream, 2**64) == 1874999576356411599
    assert tuple(
        randbelow(bounded_stream, bound)
        for bound in (2, 3, 17, 2**64 + 1)
    ) == (1, 2, 6, 17099354796136167930)


def test_replicate_zero_primitive_known_answer_vector_is_literal() -> None:
    source = ReplicateRandomSource(
        scientific_plan_sha256=ZERO_PLAN_SHA256,
        replicate_id=0,
        planned_replicates=1,
    )

    assert (
        source.seed_digest_sha256
        == "b48487717b221abeca05be2756c8439153a3002b10af07825cf1947f2e8db1d0"
    )
    assert source._next_raw64() == 3260292358039148882


def test_sealed_capsule_randbelow_rejects_foreign_stream_shapes_and_bad_bounds() -> None:
    capsule = randomness._RANDOM_CAPSULE_V2
    create_stream = tuple.__getitem__(capsule, 0)
    seed_digest = tuple.__getitem__(capsule, 1)
    randbelow = tuple.__getitem__(capsule, 2)
    stream = create_stream(ZERO_PLAN_SHA256, 0, 1)

    with pytest.raises(V2IntegrityError, match="exact built-in tuple"):
        randbelow(object(), 2)
    for malformed_shape in ((), (object(),)):
        with pytest.raises(V2IntegrityError, match="two fields"):
            seed_digest(malformed_shape)
        with pytest.raises(V2IntegrityError, match="two fields"):
            randbelow(malformed_shape, 2)
    malformed_stream = (object(), tuple.__getitem__(stream, 1))
    with pytest.raises(V2IntegrityError, match="bit generator drifted"):
        randbelow(malformed_stream, 2)
    for invalid_bound in (0, True):
        with pytest.raises(V2IntegrityError, match="positive built-in int"):
            randbelow(stream, invalid_bound)
    assert randbelow(stream, 1) == 0


@pytest.mark.parametrize(
    ("replicate_id", "expected_seed", "expected_words"),
    [
        (
            0,
            "a8dd219a1e348a103ff10bdad3b9ae8d3cff53032be90a3603734530f51fb82a",
            (1874999576356411599, 4645183139923492514, 4210660991089764159),
        ),
        (
            1,
            "0fec37b5d56de9d84e86bb178f6715dcfb300db863201679e222a3e96240056e",
            (3445068325290577610, 519030488576969648, 7848502882682531391),
        ),
    ],
)
def test_real_plan_known_answer_vectors_are_literal(
    replicate_id: int,
    expected_seed: str,
    expected_words: tuple[int, int, int],
) -> None:
    source = ReplicateRandomSource(
        scientific_plan_sha256=REAL_PLAN_SHA256,
        replicate_id=replicate_id,
        planned_replicates=2,
    )

    assert source.seed_digest_sha256 == expected_seed
    assert tuple(source._next_raw64() for _ in range(3)) == expected_words


@pytest.mark.parametrize(
    ("bound", "words", "expected", "expected_consumed"),
    [
        (1, [], 0, 0),
        (2, [3260292358039148882], 0, 1),
        (3, [3, 2], 2, 2),
        (2**64, [3260292358039148882], 3260292358039148882, 1),
        (2**64 + 1, [3260292358039148882, 11609078438225695210], 11609078438225695210, 2),
        (2**65, [1, 2], 2**64 + 2, 2),
        (2**64 + 1, [1, 1, 0, 7], 7, 4),
    ],
)
def test_uniform_randbelow_boundary_vectors_are_literal(
    bound: int,
    words: list[object],
    expected: int,
    expected_consumed: int,
) -> None:
    source = ScriptedRaw64Source(words)

    assert uniform_randbelow(source, bound) == expected
    assert source.words_consumed == expected_consumed


def test_uniform_randbelow_repeats_after_rejection_until_source_is_exhausted() -> None:
    source = ScriptedRaw64Source([3])

    with pytest.raises(StopIteration):
        uniform_randbelow(source, 3)
    assert source.words_consumed == 1


def test_uniform_randbelow_reports_mid_attempt_exhaustion_after_consumed_words() -> None:
    source = ScriptedRaw64Source([1])

    with pytest.raises(StopIteration):
        uniform_randbelow(source, 2**65)
    assert source.words_consumed == 1


class MissingRaw64Source:
    pass


class NonCallableRaw64Source:
    random_raw = 7


class SideEffectRaw64Property:
    @property
    def random_raw(self) -> object:
        raise AssertionError("validation must not invoke a random_raw property")


class InstanceFieldRaw64Source:
    def __init__(self) -> None:
        self.random_raw = lambda: 1


class ClassMethodRaw64Source:
    @classmethod
    def random_raw(cls) -> int:
        return 1


class BrokenRaw64Descriptor:
    def __get__(self, instance: object, owner: type[object]) -> object:
        raise TypeError("broken binding")


class BrokenDescriptorRaw64Source:
    random_raw = BrokenRaw64Descriptor()


class RaisingRaw64Descriptor:
    def __init__(self, error: BaseException) -> None:
        self._error = error

    def __get__(self, instance: object, owner: type[object]) -> object:
        raise self._error


class StopIterationDescriptorRaw64Source:
    random_raw = RaisingRaw64Descriptor(StopIteration("binding stopped"))


class RuntimeErrorDescriptorRaw64Source:
    random_raw = RaisingRaw64Descriptor(RuntimeError("binding runtime failure"))


class OSErrorDescriptorRaw64Source:
    random_raw = RaisingRaw64Descriptor(OSError("binding operating-system failure"))


class MemoryErrorDescriptorRaw64Source:
    random_raw = RaisingRaw64Descriptor(MemoryError("binding memory failure"))


class SystemExitDescriptorRaw64Source:
    random_raw = RaisingRaw64Descriptor(SystemExit("binding system exit"))


class RaisingCallRaw64Source:
    def __init__(self, error: Exception) -> None:
        self._error = error

    def random_raw(self) -> object:
        raise self._error


@pytest.mark.parametrize(
    "source",
    [MissingRaw64Source(), NonCallableRaw64Source(), SideEffectRaw64Property()],
)
def test_uniform_randbelow_normalizes_missing_or_noncallable_raw_source(
    source: object,
) -> None:
    with pytest.raises(V2IntegrityError, match=r"Raw64Source|random_raw"):
        uniform_randbelow(source, 2)  # type: ignore[arg-type]


def test_uniform_randbelow_accepts_direct_fixed_seed_pcg64_source() -> None:
    source = np.random.PCG64(123456)
    expected_source = np.random.PCG64(123456)
    expected = int(expected_source.random_raw()) & 1

    assert uniform_randbelow(source, 2) == expected  # type: ignore[arg-type]


def test_uniform_randbelow_accepts_instance_field_callable_without_binding_it() -> None:
    source = InstanceFieldRaw64Source()

    assert uniform_randbelow(source, 2) == 1


def test_uniform_randbelow_binds_classmethod_descriptor() -> None:
    source = ClassMethodRaw64Source()

    assert uniform_randbelow(source, 2) == 1


def test_uniform_randbelow_normalizes_descriptor_binding_failure() -> None:
    with pytest.raises(V2IntegrityError, match=r"descriptor.*bound"):
        uniform_randbelow(BrokenDescriptorRaw64Source(), 2)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("source", "error_type", "message"),
    [
        (StopIterationDescriptorRaw64Source(), StopIteration, "binding stopped"),
        (RuntimeErrorDescriptorRaw64Source(), RuntimeError, "binding runtime failure"),
        (OSErrorDescriptorRaw64Source(), OSError, "binding operating-system failure"),
        (MemoryErrorDescriptorRaw64Source(), MemoryError, "binding memory failure"),
        (SystemExitDescriptorRaw64Source(), SystemExit, "binding system exit"),
    ],
)
def test_uniform_randbelow_preserves_non_shape_descriptor_binding_failures(
    source: object,
    error_type: type[BaseException],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        uniform_randbelow(source, 2)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("error", "error_type"),
    [
        (StopIteration("call stopped"), StopIteration),
        (RuntimeError("call runtime failure"), RuntimeError),
        (OSError("call operating-system failure"), OSError),
    ],
)
def test_uniform_randbelow_preserves_random_raw_execution_failures(
    error: Exception,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type, match=str(error)):
        uniform_randbelow(RaisingCallRaw64Source(error), 2)


@pytest.mark.parametrize("bad_word", [True, np.int64(1), -1, 2**64, 1.0, "1"])
def test_uniform_randbelow_rejects_non_exact_or_out_of_range_raw_words(
    bad_word: object,
) -> None:
    with pytest.raises(V2IntegrityError, match="raw64 word"):
        uniform_randbelow(ScriptedRaw64Source([bad_word]), 2)


@pytest.mark.parametrize("bad_bound", [True, np.int64(1), 0, -1, 1.0, "1"])
def test_uniform_randbelow_rejects_non_exact_or_nonpositive_bounds(bad_bound: object) -> None:
    with pytest.raises(V2IntegrityError, match="bound"):
        uniform_randbelow(ScriptedRaw64Source([]), bad_bound)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("plan_hash", "replicate_id", "planned_replicates"),
    [
        ("A" * 64, 0, 1),
        ("0" * 63, 0, 1),
        ("g" * 64, 0, 1),
        (0, 0, 1),
        (ZERO_PLAN_SHA256, True, 1),
        (ZERO_PLAN_SHA256, np.int64(0), 1),
        (ZERO_PLAN_SHA256, -1, 1),
        (ZERO_PLAN_SHA256, 1, 1),
        (ZERO_PLAN_SHA256, 2, 1),
        (ZERO_PLAN_SHA256, 2**64, 1),
        (ZERO_PLAN_SHA256, 0, True),
        (ZERO_PLAN_SHA256, 0, np.int64(1)),
        (ZERO_PLAN_SHA256, 0, 0),
        (ZERO_PLAN_SHA256, 0, 1_000_001),
    ],
)
def test_replicate_source_fails_closed_on_invalid_factory_inputs(
    plan_hash: object,
    replicate_id: object,
    planned_replicates: object,
) -> None:
    with pytest.raises(V2IntegrityError):
        ReplicateRandomSource(
            scientific_plan_sha256=plan_hash,  # type: ignore[arg-type]
            replicate_id=replicate_id,  # type: ignore[arg-type]
            planned_replicates=planned_replicates,  # type: ignore[arg-type]
        )


def test_factory_signature_has_only_the_three_frozen_scientific_inputs() -> None:
    signature = inspect.signature(ReplicateRandomSource)

    assert tuple(signature.parameters) == (
        "scientific_plan_sha256",
        "replicate_id",
        "planned_replicates",
    )


def test_sources_are_replicate_local_order_independent_and_retryable() -> None:
    baseline_zero = ReplicateRandomSource(REAL_PLAN_SHA256, 0, 2)
    baseline_one = ReplicateRandomSource(REAL_PLAN_SHA256, 1, 2)
    expected_zero = tuple(baseline_zero._next_raw64() for _ in range(3))
    expected_one = tuple(baseline_one._next_raw64() for _ in range(3))

    interleaved_zero = ReplicateRandomSource(REAL_PLAN_SHA256, 0, 2)
    interleaved_one = ReplicateRandomSource(REAL_PLAN_SHA256, 1, 2)
    actual_zero: list[int] = []
    actual_one: list[int] = []
    for _ in range(3):
        actual_one.append(interleaved_one._next_raw64())
        actual_zero.append(interleaved_zero._next_raw64())

    retry_zero = ReplicateRandomSource(REAL_PLAN_SHA256, 0, 2)
    assert tuple(actual_zero) == expected_zero
    assert tuple(actual_one) == expected_one
    assert tuple(retry_zero._next_raw64() for _ in range(3)) == expected_zero


def test_instances_do_not_share_consumption_state() -> None:
    first = ReplicateRandomSource(ZERO_PLAN_SHA256, 0, 1)
    second = ReplicateRandomSource(ZERO_PLAN_SHA256, 0, 1)

    first._next_raw64()
    assert second._next_raw64() == 3260292358039148882


def test_compatibility_source_satisfies_uniform_index_protocol() -> None:
    source = ReplicateRandomSource(ZERO_PLAN_SHA256, 0, 1)

    assert isinstance(source, UniformIndexSource)
    assert isinstance(ScriptedRaw64Source([]), Raw64Source)


def test_compatibility_source_converts_pcg64_word_immediately_to_builtin_int(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seeds: list[int] = []

    class FakePCG64:
        def __init__(self, seed: int) -> None:
            seeds.append(seed)

        def random_raw(self) -> np.uint64:
            return np.uint64(7)

    monkeypatch.setattr(randomness.np.random, "PCG64", FakePCG64)
    source = ReplicateRandomSource(ZERO_PLAN_SHA256, 0, 1)

    word = source._next_raw64()
    assert type(word) is int
    assert word == 7
    assert seeds == [
        int("b48487717b221abeca05be2756c8439153a3002b10af07825cf1947f2e8db1d0", 16)
    ]


def test_runtime_source_has_no_forbidden_random_api_calls() -> None:
    tree = ast.parse(inspect.getsource(randomness))
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    assert called_names.isdisjoint({"hash", "SeedSequence"})
    assert called_attributes.isdisjoint(
        {
            "SeedSequence",
            "choice",
            "default_rng",
            "integers",
            "permutation",
            "rand",
            "randint",
            "random",
            "shuffle",
            "spawn",
        }
    )
    assert "random_raw" in called_attributes

    def dotted_name(node: ast.expr) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            prefix = dotted_name(node.value)
            return None if prefix is None else f"{prefix}.{node.attr}"
        return None

    numpy_random_calls = {
        name
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (name := dotted_name(node.func)) is not None
        and name.startswith("np.random.")
    }
    assert numpy_random_calls == {"np.random.PCG64"}
