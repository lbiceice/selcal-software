from __future__ import annotations

import ast
import inspect
import itertools
from dataclasses import FrozenInstanceError, replace
from types import MappingProxyType
from typing import cast

import numpy as np
import pytest

from selcal.contracts import SeriesPair
from selcal.contracts_v2 import (
    BlockShuffleStateV2,
    CircularShiftStateV2,
    NullBindStatus,
    NullDisabledReason,
    NullTransformToken,
    V2IntegrityError,
)
from selcal.nulls import block_shuffle_v2 as block_module
from selcal.nulls.block_shuffle_v2 import BlockShuffleNullV2

SEMANTIC_SHA = "8" * 64
PLAN_SHA = "d" * 64


def pair(
    source: tuple[float, ...] = (0.0, 3.0, 1.0, 2.0, 4.0, 5.0),
    target: tuple[float, ...] = (0.0, 1.0, 3.0, 2.0, 5.0, 4.0),
) -> SeriesPair:
    return SeriesPair(
        source=np.asarray(source, dtype=np.float64),
        target=np.asarray(target, dtype=np.float64),
    )


class ScriptedIndexSource:
    def __init__(self, *indices: int) -> None:
        self._indices = iter(indices)
        self.bounds: list[int] = []

    def randbelow(self, bound: int, /) -> int:
        self.bounds.append(bound)
        return next(self._indices)


class EqualString(str):
    pass


class DerivedNullTransformToken(NullTransformToken):
    pass


def enabled_bound(
    observed: SeriesPair | None = None,
    *,
    block_length: int = 2,
    semantic_sha: str = SEMANTIC_SHA,
    plan_sha: str = PLAN_SHA,
):
    result = BlockShuffleNullV2(block_length=block_length).bind(
        observed if observed is not None else pair(),
        semantic_input_sha256=semantic_sha,
        scientific_plan_sha256=plan_sha,
    )
    assert result.status is NullBindStatus.ENABLED
    assert result.bound is not None
    return result.bound


def state_order(token: NullTransformToken) -> tuple[int, ...]:
    return cast(BlockShuffleStateV2, token.state).block_order


def test_nondivisible_tail_is_disabled_without_drop_pad_or_fixed_tail() -> None:
    observed = pair(
        source=(0.0, 1.0, 2.0, 3.0, 4.0),
        target=(5.0, 6.0, 7.0, 8.0, 9.0),
    )
    result = BlockShuffleNullV2(block_length=2).bind(
        observed,
        semantic_input_sha256=SEMANTIC_SHA,
        scientific_plan_sha256=PLAN_SHA,
    )

    assert result.status is NullBindStatus.DISABLED
    assert result.bound is None
    assert result.disabled_reason is NullDisabledReason.NON_DIVISIBLE_TAIL
    assert result.diagnostics == ("non_divisible_tail_v2",)


def test_fewer_than_two_complete_blocks_is_disabled() -> None:
    result = BlockShuffleNullV2(block_length=6).bind(
        pair(),
        semantic_input_sha256=SEMANTIC_SHA,
        scientific_plan_sha256=PLAN_SHA,
    )

    assert result.status is NullBindStatus.DISABLED
    assert result.bound is None
    assert result.disabled_reason is NullDisabledReason.FEWER_THAN_TWO_BLOCKS
    assert result.diagnostics == ("fewer_than_two_blocks_v2",)


def test_too_many_blocks_fails_before_label_or_factorial_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = SeriesPair(
        source=np.zeros(4097, dtype=np.float64),
        target=np.ones(4097, dtype=np.float64),
    )
    monkeypatch.setattr(
        block_module,
        "_factorial",
        lambda _: (_ for _ in ()).throw(AssertionError("factorial constructed")),
    )
    monkeypatch.setattr(
        block_module,
        "_ascending_labels",
        lambda _: (_ for _ in ()).throw(AssertionError("block list constructed")),
    )

    result = BlockShuffleNullV2(block_length=1).bind(
        observed,
        semantic_input_sha256=SEMANTIC_SHA,
        scientific_plan_sha256=PLAN_SHA,
    )

    assert result.status is NullBindStatus.DISABLED
    assert result.bound is None
    assert result.disabled_reason is NullDisabledReason.TOO_MANY_BLOCKS
    assert result.diagnostics == ("too_many_blocks_v2",)


def test_too_many_blocks_is_rejected_without_constructing_a_second_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = SeriesPair(
        source=np.zeros(4097, dtype=np.float64),
        target=np.ones(4097, dtype=np.float64),
    )

    def forbidden_pair_construction(self: object, *args: object, **kwargs: object) -> None:
        del self, args, kwargs
        raise AssertionError("disabled bind copied the complete pair")

    monkeypatch.setattr(SeriesPair, "__init__", forbidden_pair_construction)
    monkeypatch.setattr(
        block_module,
        "_owned_pair_content_guard_sha256",
        lambda *_: (_ for _ in ()).throw(AssertionError("disabled bind hashed pair")),
        raising=False,
    )

    result = BlockShuffleNullV2(block_length=1).bind(
        observed,
        semantic_input_sha256=SEMANTIC_SHA,
        scientific_plan_sha256=PLAN_SHA,
    )

    assert result.status is NullBindStatus.DISABLED
    assert result.disabled_reason is NullDisabledReason.TOO_MANY_BLOCKS


@pytest.mark.parametrize(
    "observed,block_length,reason",
    [
        (
            pair(
                source=(0.0, 1.0, 2.0, 3.0, 4.0),
                target=(5.0, 6.0, 7.0, 8.0, 9.0),
            ),
            2,
            NullDisabledReason.NON_DIVISIBLE_TAIL,
        ),
        (
            SeriesPair(
                source=np.zeros(1, dtype=np.float64),
                target=np.ones(1, dtype=np.float64),
            ),
            1,
            NullDisabledReason.FEWER_THAN_TWO_BLOCKS,
        ),
        (
            SeriesPair(
                source=np.zeros(4097, dtype=np.float64),
                target=np.ones(4097, dtype=np.float64),
            ),
            1,
            NullDisabledReason.TOO_MANY_BLOCKS,
        ),
    ],
)
def test_disabled_block_paths_do_not_copy_or_hash_complete_pair(
    monkeypatch: pytest.MonkeyPatch,
    observed: SeriesPair,
    block_length: int,
    reason: NullDisabledReason,
) -> None:
    def forbidden_pair_construction(
        self: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        del self, args, kwargs
        raise AssertionError("disabled bind copied the complete pair")

    monkeypatch.setattr(SeriesPair, "__init__", forbidden_pair_construction)
    monkeypatch.setattr(
        block_module,
        "_owned_pair_content_guard_sha256",
        lambda *_: (_ for _ in ()).throw(AssertionError("disabled bind hashed pair")),
    )

    result = BlockShuffleNullV2(block_length=block_length).bind(
        observed,
        semantic_input_sha256=SEMANTIC_SHA,
        scientific_plan_sha256=PLAN_SHA,
    )

    assert result.status is NullBindStatus.DISABLED
    assert result.disabled_reason is reason


def test_binding_4096_blocks_computes_factorial_only_after_the_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = SeriesPair(
        source=np.arange(4096, dtype=np.float64),
        target=np.arange(4096, dtype=np.float64),
    )
    calls: list[int] = []
    real_factorial = block_module._factorial

    def watched_factorial(value: int) -> int:
        calls.append(value)
        return real_factorial(value)

    monkeypatch.setattr(block_module, "_factorial", watched_factorial)
    result = BlockShuffleNullV2(block_length=1).bind(
        observed,
        semantic_input_sha256=SEMANTIC_SHA,
        scientific_plan_sha256=PLAN_SHA,
    )

    assert result.status is NullBindStatus.ENABLED
    assert result.bound is not None
    assert calls == [4096]
    assert result.bound.total_state_count > 2**64


def test_fisher_yates_uses_frozen_descending_bounds_and_approved_outputs() -> None:
    bound = enabled_bound()

    replicate_zero = ScriptedIndexSource(0, 1)
    replicate_one = ScriptedIndexSource(2, 0)
    token_zero = bound.sample_token(replicate_zero)
    token_one = bound.sample_token(replicate_one)

    assert state_order(token_zero) == (2, 1, 0)
    assert state_order(token_one) == (1, 0, 2)
    assert replicate_zero.bounds == [3, 2]
    assert replicate_one.bounds == [3, 2]


def test_all_labelled_permutations_are_reachable_once_from_fisher_yates_scripts() -> None:
    bound = enabled_bound()
    orders = {
        state_order(bound.sample_token(ScriptedIndexSource(j2, j1)))
        for j2 in range(3)
        for j1 in range(2)
    }

    assert orders == set(itertools.permutations((0, 1, 2)))
    assert len(orders) == bound.total_state_count == 6


def test_identity_is_accepted_without_retry() -> None:
    bound = enabled_bound()
    random = ScriptedIndexSource(2, 1)

    sampled = bound.sample_token(random)

    assert sampled == bound.identity_token()
    assert sampled.is_identity is True
    assert state_order(sampled) == (0, 1, 2)
    assert random.bounds == [3, 2]


def test_4096_block_sampling_is_streaming_with_exact_descending_bounds() -> None:
    observed = SeriesPair(
        source=np.arange(4096, dtype=np.float64),
        target=np.arange(4096, dtype=np.float64),
    )
    bound = enabled_bound(observed, block_length=1)

    class StreamingIdentitySource:
        def __init__(self) -> None:
            self.bounds: list[int] = []

        def randbelow(self, bound_value: int, /) -> int:
            self.bounds.append(bound_value)
            return bound_value - 1

    random = StreamingIdentitySource()
    token = bound.sample_token(random)

    assert token.is_identity is True
    assert len(state_order(token)) == 4096
    assert len(random.bounds) == 4095
    assert random.bounds == list(range(4096, 1, -1))


def test_sampling_does_not_materialize_the_factorial_state_space() -> None:
    source = inspect.getsource(block_module)
    tree = ast.parse(source)
    forbidden_calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute | ast.Name)
        and (node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id)
        in {"permutations", "product"}
    }
    assert forbidden_calls == set()

    bound = enabled_bound(
        SeriesPair(
            source=np.arange(18, dtype=np.float64),
            target=np.arange(18, dtype=np.float64),
        )
    )
    random = ScriptedIndexSource(*range(8, 0, -1))
    token = bound.sample_token(random)
    assert len(state_order(token)) == 9
    assert random.bounds == list(range(9, 1, -1))


def test_complete_source_block_mapping_preserves_internal_order_and_target_bytes() -> None:
    observed = pair()
    bound = enabled_bound(observed)
    token = bound.sample_token(ScriptedIndexSource(0, 1))

    transformed = bound.apply(token)

    assert np.array_equal(transformed.pair.source, (4.0, 5.0, 1.0, 2.0, 0.0, 3.0))
    assert transformed.pair.target.tobytes(order="C") == observed.target.tobytes(
        order="C"
    )
    assert transformed.source_changed is True
    assert transformed.diagnostics == ()


def test_repeated_numerical_blocks_retain_labelled_state_multiplicity() -> None:
    observed = pair(
        source=(1.0, 2.0, 1.0, 2.0, 3.0, 4.0),
        target=(6.0, 5.0, 4.0, 3.0, 2.0, 1.0),
    )
    bound = enabled_bound(observed)
    random = ScriptedIndexSource(2, 0, 2, 0)

    first = bound.sample_token(random)
    second = bound.sample_token(random)
    first_result = bound.apply(first)
    second_result = bound.apply(second)

    assert state_order(first) == state_order(second) == (1, 0, 2)
    assert first == second
    assert first.is_identity is False
    assert first_result.pair == observed
    assert first_result.source_changed is False
    assert first_result.diagnostics == ("nonidentity_data_equivalent_v2",)
    assert second_result.diagnostics == ("nonidentity_data_equivalent_v2",)
    assert random.bounds == [3, 2, 3, 2]


def test_result_owns_token_and_pair_snapshots() -> None:
    bound = enabled_bound()
    input_token = bound.sample_token(ScriptedIndexSource(0, 1))
    result = bound.apply(input_token)
    expected_source = result.pair.source.tobytes(order="C")
    expected_token_digest = block_module.block_token_sha256(result.token)

    object.__setattr__(input_token, "scientific_plan_sha256", "1" * 64)
    object.__setattr__(
        cast(BlockShuffleStateV2, input_token.state),
        "block_order",
        (0, 1, 2),
    )

    assert result.token is not input_token
    assert result.token.state is not input_token.state
    assert state_order(result.token) == (2, 1, 0)
    assert block_module.block_token_sha256(result.token) == expected_token_digest
    assert result.pair.source.tobytes(order="C") == expected_source
    assert result.pair.source.flags.writeable is False
    assert result.pair.target.flags.writeable is False


def test_apply_uses_one_token_snapshot_across_transform_injection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bound = enabled_bound()
    input_token = bound.sample_token(ScriptedIndexSource(0, 1))
    original_apply = block_module._apply_state

    def mutate_then_apply(
        pair_value: SeriesPair,
        order: tuple[int, ...],
        length: int,
    ) -> SeriesPair:
        object.__setattr__(input_token, "scientific_plan_sha256", "1" * 64)
        object.__setattr__(
            cast(BlockShuffleStateV2, input_token.state),
            "block_order",
            (0, 1, 2),
        )
        return original_apply(pair_value, order, length)

    monkeypatch.setattr(block_module, "_apply_state", mutate_then_apply)
    result = bound.apply(input_token)

    assert result.token.scientific_plan_sha256 == PLAN_SHA
    assert state_order(result.token) == (2, 1, 0)
    assert np.array_equal(result.pair.source, (4.0, 5.0, 1.0, 2.0, 0.0, 3.0))


def test_apply_rejects_hook_replacing_bound_pair_after_token_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bound = enabled_bound()
    token = bound.sample_token(ScriptedIndexSource(0, 1))
    replacement = pair(
        source=(10.0, 11.0, 12.0, 13.0, 14.0, 15.0),
        target=(20.0, 21.0, 22.0, 23.0, 24.0, 25.0),
    )
    original_apply = block_module._apply_state

    def replace_bound_then_transform(
        pair_value: SeriesPair,
        order: tuple[int, ...],
        length: int,
    ) -> SeriesPair:
        del pair_value
        object.__setattr__(bound, "_observed_pair", replacement)
        return original_apply(replacement, order, length)

    monkeypatch.setattr(block_module, "_apply_state", replace_bound_then_transform)

    with pytest.raises(V2IntegrityError):
        bound.apply(token)


def test_apply_rejects_hook_mutating_its_transform_input_in_place(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bound = enabled_bound()
    token = bound.sample_token(ScriptedIndexSource(0, 1))
    replacement = pair(
        source=(10.0, 11.0, 12.0, 13.0, 14.0, 15.0),
        target=(20.0, 21.0, 22.0, 23.0, 24.0, 25.0),
    )
    original_apply = block_module._apply_state

    def mutate_input_then_transform(
        pair_value: SeriesPair,
        order: tuple[int, ...],
        length: int,
    ) -> SeriesPair:
        object.__setattr__(pair_value, "source", replacement.source)
        object.__setattr__(pair_value, "target", replacement.target)
        return original_apply(pair_value, order, length)

    monkeypatch.setattr(block_module, "_apply_state", mutate_input_then_transform)

    with pytest.raises(V2IntegrityError):
        bound.apply(token)


@pytest.mark.parametrize("damage", ["replace_pair", "source", "target"])
def test_apply_rejects_same_length_bound_content_drift_before_transform(
    monkeypatch: pytest.MonkeyPatch,
    damage: str,
) -> None:
    bound = enabled_bound()
    token = bound.sample_token(ScriptedIndexSource(0, 1))
    replacement = pair(
        source=(10.0, 11.0, 12.0, 13.0, 14.0, 15.0),
        target=(20.0, 21.0, 22.0, 23.0, 24.0, 25.0),
    )
    if damage == "replace_pair":
        object.__setattr__(bound, "_observed_pair", replacement)
    else:
        observed = bound._observed_pair
        object.__setattr__(observed, damage, getattr(replacement, damage))
    monkeypatch.setattr(
        block_module,
        "_apply_state",
        lambda *_: (_ for _ in ()).throw(AssertionError("transform was called")),
    )

    with pytest.raises(V2IntegrityError, match="content guard"):
        bound.apply(token)


@pytest.mark.parametrize("guard", ["f" * 64, "F" * 64, object()])
def test_apply_rejects_tampered_owned_content_guard_before_transform(
    monkeypatch: pytest.MonkeyPatch,
    guard: object,
) -> None:
    bound = enabled_bound()
    token = bound.identity_token()
    object.__setattr__(bound, "_observed_content_guard_sha256", guard)
    monkeypatch.setattr(
        block_module,
        "_apply_state",
        lambda *_: (_ for _ in ()).throw(AssertionError("transform was called")),
    )

    with pytest.raises(V2IntegrityError):
        bound.apply(token)


def test_contract_and_bound_are_owned_immutable_snapshots() -> None:
    contract = BlockShuffleNullV2(block_length=2)
    source = np.asarray((0.0, 1.0, 2.0, 3.0), dtype=np.float64)
    target = np.asarray((3.0, 2.0, 1.0, 0.0), dtype=np.float64)
    observed = SeriesPair(source=source, target=target)
    result = contract.bind(
        observed,
        semantic_input_sha256=SEMANTIC_SHA,
        scientific_plan_sha256=PLAN_SHA,
    )
    assert result.bound is not None
    bound = result.bound

    source[:] = 99.0
    target[:] = 88.0
    object.__setattr__(observed, "source", np.full(4, 7.0))
    transformed = bound.apply(bound.identity_token())

    assert np.array_equal(transformed.pair.source, (0.0, 1.0, 2.0, 3.0))
    assert np.array_equal(transformed.pair.target, (3.0, 2.0, 1.0, 0.0))
    with pytest.raises(TypeError):
        bound.parameters["block_length"] = 3  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        contract.block_length = 3  # type: ignore[misc]


@pytest.mark.parametrize("value", [True, np.int64(1), 0, -1, 1.0, "1"])
def test_block_length_requires_exact_positive_builtin_int(value: object) -> None:
    with pytest.raises((TypeError, ValueError, V2IntegrityError)):
        BlockShuffleNullV2(block_length=value)  # type: ignore[arg-type]


def test_from_parameters_requires_the_exact_field_set() -> None:
    contract = BlockShuffleNullV2.from_parameters({"block_length": 2})
    assert contract.name == "block_shuffle_v2"
    assert contract.parameters == {"block_length": 2}
    assert len(contract.null_parameter_sha256) == 64

    for invalid in (
        {},
        {"block_length": 2, "extra": 1},
        {"block_length": True},
        MappingProxyType({"block_length": np.int64(2)}),
    ):
        with pytest.raises((TypeError, ValueError, V2IntegrityError)):
            BlockShuffleNullV2.from_parameters(invalid)


@pytest.mark.parametrize(
    "field,value",
    [
        ("semantic_input_sha256", "1" * 64),
        ("scientific_plan_sha256", "2" * 64),
        ("null_parameter_sha256", "3" * 64),
        ("bound_null_owner_sha256", "4" * 64),
    ],
)
def test_foreign_token_is_rejected_before_data_access_or_transform(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
) -> None:
    bound = enabled_bound()
    token = replace(bound.identity_token(), **{field: value})

    class ExplodingPair:
        @property
        def source(self) -> object:
            raise AssertionError("foreign token touched source")

        @property
        def target(self) -> object:
            raise AssertionError("foreign token touched target")

    object.__setattr__(bound, "_observed_pair", ExplodingPair())
    monkeypatch.setattr(
        block_module,
        "_apply_state",
        lambda *_: (_ for _ in ()).throw(AssertionError("transform called")),
    )

    with pytest.raises(V2IntegrityError):
        bound.apply(token)


@pytest.mark.parametrize(
    "damage",
    [
        "identity",
        "identity_type",
        "order_type",
        "wrong_length",
        "duplicate",
        "negative",
        "bool",
        "state_schema",
    ],
)
def test_invalid_state_or_identity_is_rejected_before_transform(
    monkeypatch: pytest.MonkeyPatch,
    damage: str,
) -> None:
    bound = enabled_bound()
    token = bound.identity_token()
    state = cast(BlockShuffleStateV2, token.state)
    if damage == "identity":
        object.__setattr__(token, "is_identity", False)
    elif damage == "identity_type":
        object.__setattr__(token, "is_identity", 1)
    elif damage == "order_type":
        object.__setattr__(state, "block_order", [0, 1, 2])
    elif damage == "wrong_length":
        object.__setattr__(state, "block_order", (0, 1))
    elif damage == "duplicate":
        object.__setattr__(state, "block_order", (0, 0, 2))
    elif damage == "negative":
        object.__setattr__(state, "block_order", (-1, 1, 2))
    elif damage == "bool":
        object.__setattr__(state, "block_order", (False, 1, 2))
    else:
        object.__setattr__(state, "schema", EqualString(state.schema))
    monkeypatch.setattr(
        block_module,
        "_apply_state",
        lambda *_: (_ for _ in ()).throw(AssertionError("transform called")),
    )

    with pytest.raises(V2IntegrityError):
        bound.apply(token)


@pytest.mark.parametrize(
    "field",
    [
        "schema",
        "null_name",
        "null_parameter_sha256",
        "semantic_input_sha256",
        "scientific_plan_sha256",
        "bound_null_owner_sha256",
    ],
)
def test_equal_string_subclasses_in_token_fields_are_rejected(field: str) -> None:
    bound = enabled_bound()
    token = bound.identity_token()
    object.__setattr__(token, field, EqualString(getattr(token, field)))
    with pytest.raises(V2IntegrityError):
        bound.apply(token)


@pytest.mark.parametrize(
    "boundary,owner,field",
    [
        ("apply", "token", "schema"),
        ("apply", "state", "block_order"),
        ("digest", "token", "null_name"),
        ("digest", "state", "schema"),
    ],
)
def test_deleted_slots_raise_typed_integrity_errors(
    boundary: str,
    owner: str,
    field: str,
) -> None:
    bound = enabled_bound()
    token = bound.sample_token(ScriptedIndexSource(0, 1))
    object.__delattr__(token if owner == "token" else token.state, field)

    with pytest.raises(V2IntegrityError):
        if boundary == "apply":
            bound.apply(token)
        else:
            block_module.block_token_sha256(token)


def test_apply_rejects_derived_token_and_foreign_state_types() -> None:
    bound = enabled_bound()
    token = bound.identity_token()
    derived = DerivedNullTransformToken(
        schema=token.schema,
        null_name=token.null_name,
        null_parameter_sha256=token.null_parameter_sha256,
        semantic_input_sha256=token.semantic_input_sha256,
        scientific_plan_sha256=token.scientific_plan_sha256,
        bound_null_owner_sha256=token.bound_null_owner_sha256,
        is_identity=token.is_identity,
        state=token.state,
    )
    with pytest.raises(V2IntegrityError, match="exact NullTransformToken"):
        bound.apply(derived)

    foreign = bound.identity_token()
    object.__setattr__(
        foreign,
        "state",
        CircularShiftStateV2(
            schema="selcal.circular-shift-state.v2",
            shift=0,
        ),
    )
    with pytest.raises(V2IntegrityError, match="state type"):
        bound.apply(foreign)


def test_bound_parameter_and_owner_drift_fail_before_transform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for field, value, match in (
        ("_block_length", 1, "parameter"),
        ("_null_parameter_sha256", "f" * 64, "parameter"),
        ("_bound_null_owner_sha256", "f" * 64, "owner"),
    ):
        bound = enabled_bound()
        token = bound.identity_token()
        object.__setattr__(bound, field, value)
        monkeypatch.setattr(
            block_module,
            "_apply_state",
            lambda *_: (_ for _ in ()).throw(AssertionError("transform called")),
        )
        with pytest.raises(V2IntegrityError, match=match):
            bound.apply(token)


@pytest.mark.parametrize("drift", ["shape", "target", "multiset", "mapping"])
def test_apply_rejects_hostile_transform_postcondition_drift(
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    observed = pair()
    bound = enabled_bound(observed)
    token = bound.sample_token(ScriptedIndexSource(0, 1))

    def hostile(pair_value: SeriesPair, order: tuple[int, ...], length: int) -> SeriesPair:
        del order, length
        if drift == "shape":
            return SeriesPair(source=pair_value.source[:-1], target=pair_value.target[:-1])
        if drift == "target":
            changed = pair_value.target.copy()
            changed[0] += 1.0
            return SeriesPair(source=pair_value.source, target=changed)
        if drift == "multiset":
            changed = pair_value.source.copy()
            changed[0] += 100.0
            return SeriesPair(source=changed, target=pair_value.target)
        # Legal different permutation: passes shape, target, and multiset.
        return SeriesPair(
            source=np.asarray((1.0, 2.0, 4.0, 5.0, 0.0, 3.0)),
            target=pair_value.target,
        )

    monkeypatch.setattr(block_module, "_apply_state", hostile)
    with pytest.raises(V2IntegrityError, match=drift):
        bound.apply(token)


def test_apply_rejects_non_pair_target_only_shape_and_identity_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bound = enabled_bound()
    identity = bound.identity_token()
    real_apply = block_module._apply_state
    monkeypatch.setattr(block_module, "_apply_state", lambda *_: object())
    with pytest.raises(V2IntegrityError, match="exact SeriesPair"):
        bound.apply(identity)

    def target_shape(pair_value: SeriesPair, order: tuple[int, ...], length: int) -> SeriesPair:
        del order, length
        result = SeriesPair(source=pair_value.source, target=pair_value.target)
        object.__setattr__(result, "target", result.target[:-1])
        return result

    monkeypatch.setattr(block_module, "_apply_state", target_shape)
    with pytest.raises(V2IntegrityError, match="shape"):
        bound.apply(identity)

    monkeypatch.setattr(
        block_module,
        "_apply_state",
        lambda pair_value, order, length: real_apply(pair_value, (1, 0, 2), length),
    )
    with pytest.raises(V2IntegrityError, match=r"identity.*drift"):
        bound.apply(identity)


@pytest.mark.parametrize("random_value", [-1, 3, True, np.int64(0), 0.0])
def test_sampler_rejects_nonexact_or_out_of_range_indices(random_value: object) -> None:
    bound = enabled_bound()

    class Invalid:
        def randbelow(self, bound_value: int, /) -> object:
            del bound_value
            return random_value

    with pytest.raises(V2IntegrityError, match="index"):
        bound.sample_token(Invalid())  # type: ignore[arg-type]


def test_sampler_uses_shared_static_resolver_without_invoking_properties() -> None:
    bound = enabled_bound()

    class PropertySource:
        def __init__(self) -> None:
            self.calls = 0

        @property
        def randbelow(self):  # type: ignore[no-untyped-def]
            self.calls += 1
            return lambda _: 0

    source = PropertySource()
    with pytest.raises(V2IntegrityError, match="property"):
        bound.sample_token(source)  # type: ignore[arg-type]
    assert source.calls == 0


def test_sampler_preserves_actual_execution_exceptions() -> None:
    bound = enabled_bound()

    class Source:
        def randbelow(self, bound_value: int, /) -> int:
            del bound_value
            raise RuntimeError("actual execution failure")

    with pytest.raises(RuntimeError, match="actual execution failure"):
        bound.sample_token(Source())


@pytest.mark.parametrize(
    "field,value",
    [
        ("semantic_input_sha256", "A" * 64),
        ("scientific_plan_sha256", "f" * 63),
        ("semantic_input_sha256", 1),
    ],
)
def test_bind_rejects_malformed_ownership_digests(field: str, value: object) -> None:
    kwargs: dict[str, object] = {
        "semantic_input_sha256": SEMANTIC_SHA,
        "scientific_plan_sha256": PLAN_SHA,
    }
    kwargs[field] = value
    with pytest.raises(V2IntegrityError):
        BlockShuffleNullV2(block_length=2).bind(pair(), **kwargs)  # type: ignore[arg-type]


def test_bind_requires_exact_series_pair() -> None:
    with pytest.raises(V2IntegrityError):
        BlockShuffleNullV2(block_length=2).bind(
            object(),  # type: ignore[arg-type]
            semantic_input_sha256=SEMANTIC_SHA,
            scientific_plan_sha256=PLAN_SHA,
        )


def test_bind_types_damaged_exact_pair_fields() -> None:
    observed = pair()
    object.__delattr__(observed, "source")

    with pytest.raises(V2IntegrityError, match="fields"):
        BlockShuffleNullV2(block_length=2).bind(
            observed,
            semantic_input_sha256=SEMANTIC_SHA,
            scientific_plan_sha256=PLAN_SHA,
        )


def test_private_bound_constructor_is_fail_closed() -> None:
    observed = pair()
    parameter_sha = block_module._parameter_digest(2)
    owner_sha = block_module._owner_digest(
        semantic_input_sha256=SEMANTIC_SHA,
        scientific_plan_sha256=PLAN_SHA,
        null_parameter_sha256=parameter_sha,
        observed_length=6,
    )
    common = {
        "_observed_pair": observed,
        "_block_length": 2,
        "_block_count": 3,
        "_semantic_input_sha256": SEMANTIC_SHA,
        "_scientific_plan_sha256": PLAN_SHA,
        "_null_parameter_sha256": parameter_sha,
        "_bound_null_owner_sha256": owner_sha,
        "_observed_content_guard_sha256": (
            block_module._owned_pair_content_guard_sha256(observed)
        ),
    }
    hostile_values = (
        {"_observed_pair": object()},
        {"_block_length": True},
        {"_block_count": 1},
        {"_block_count": 4097},
        {"_block_count": 2},
        {"_null_parameter_sha256": "f" * 64},
        {"_bound_null_owner_sha256": "f" * 64},
        {"_observed_content_guard_sha256": "f" * 64},
        {"_observed_content_guard_sha256": "F" * 64},
    )
    for changes in hostile_values:
        values = dict(common)
        values.update(changes)
        with pytest.raises(V2IntegrityError):
            block_module._BoundBlockShuffleV2(**values)


def test_private_bound_constructor_types_damaged_exact_pair_slots() -> None:
    observed = pair()
    content_guard = block_module._owned_pair_content_guard_sha256(observed)
    object.__delattr__(observed, "source")

    with pytest.raises(V2IntegrityError, match="snapshot"):
        block_module._BoundBlockShuffleV2(
            _observed_pair=observed,
            _block_length=2,
            _block_count=3,
            _semantic_input_sha256=SEMANTIC_SHA,
            _scientific_plan_sha256=PLAN_SHA,
            _null_parameter_sha256=block_module._parameter_digest(2),
            _bound_null_owner_sha256=block_module._owner_digest(
                semantic_input_sha256=SEMANTIC_SHA,
                scientific_plan_sha256=PLAN_SHA,
                null_parameter_sha256=block_module._parameter_digest(2),
                observed_length=6,
            ),
            _observed_content_guard_sha256=content_guard,
        )


def test_apply_types_invalid_bound_block_scalar_before_transform() -> None:
    bound = enabled_bound()
    token = bound.identity_token()
    object.__setattr__(bound, "_block_length", True)

    with pytest.raises(V2IntegrityError, match="parameter"):
        bound.apply(token)


def test_apply_rejects_owned_length_drift_against_golden_pair() -> None:
    bound = enabled_bound()
    token = bound.identity_token()
    drifted_length = 8
    drifted_count = 4
    drifted_owner = block_module._owner_digest(
        semantic_input_sha256=SEMANTIC_SHA,
        scientific_plan_sha256=PLAN_SHA,
        null_parameter_sha256=token.null_parameter_sha256,
        observed_length=drifted_length,
    )
    object.__setattr__(bound, "_observed_length", drifted_length)
    object.__setattr__(bound, "_block_count", drifted_count)
    object.__setattr__(bound, "_bound_null_owner_sha256", drifted_owner)
    token = replace(
        token,
        bound_null_owner_sha256=drifted_owner,
        state=BlockShuffleStateV2(
            schema="selcal.block-shuffle-state.v2",
            block_order=(0, 1, 2, 3),
        ),
    )

    with pytest.raises(V2IntegrityError, match="pair length"):
        bound.apply(token)


@pytest.mark.parametrize(
    "slot",
    [
        "_observed_pair",
        "_block_length",
        "_block_count",
        "_semantic_input_sha256",
        "_scientific_plan_sha256",
        "_null_parameter_sha256",
        "_bound_null_owner_sha256",
        "_observed_content_guard_sha256",
        "_observed_length",
    ],
)
def test_apply_types_deleted_bound_slots_as_integrity_failures(slot: str) -> None:
    bound = enabled_bound()
    token = bound.identity_token()
    object.__delattr__(bound, slot)

    with pytest.raises(V2IntegrityError):
        bound.apply(token)
