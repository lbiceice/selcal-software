from __future__ import annotations

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
from selcal.nulls import circular_shift_v2 as circular_module
from selcal.nulls.circular_shift_v2 import CircularShiftNullV2

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
    """A hostile string subclass equal to, but not identical with, built-in str."""


class DerivedNullTransformToken(NullTransformToken):
    pass


def enabled_bound(
    observed: SeriesPair | None = None,
    *,
    min_shift: int = 1,
    semantic_sha: str = SEMANTIC_SHA,
    plan_sha: str = PLAN_SHA,
):
    result = CircularShiftNullV2(min_shift=min_shift).bind(
        observed if observed is not None else pair(),
        semantic_input_sha256=semantic_sha,
        scientific_plan_sha256=plan_sha,
    )
    assert result.status is NullBindStatus.ENABLED
    assert result.bound is not None
    return result.bound


def test_complete_state_order_has_identity_then_symmetric_allowed_shifts() -> None:
    bound = enabled_bound(min_shift=2)
    random = ScriptedIndexSource(*range(bound.total_state_count))

    tokens = tuple(bound.sample_token(random) for _ in range(bound.total_state_count))
    shifts = tuple(cast(CircularShiftStateV2, token.state).shift for token in tokens)

    assert shifts == (0, 2, 3, 4)
    assert all(min(shift, bound.observed_length - shift) >= 2 for shift in shifts[1:])
    assert random.bounds == [4, 4, 4, 4]


def test_positive_shift_is_the_frozen_right_shift_and_target_is_byte_identical() -> None:
    observed = pair()
    bound = enabled_bound(observed)
    token = bound.sample_token(ScriptedIndexSource(2))

    transformed = bound.apply(token)

    expected = np.asarray((4.0, 5.0, 0.0, 3.0, 1.0, 2.0), dtype=np.float64)
    assert np.array_equal(transformed.pair.source, expected)
    assert transformed.pair.target.tobytes(order="C") == observed.target.tobytes(
        order="C"
    )
    assert transformed.source_changed is True
    assert transformed.diagnostics == ()


def test_identity_is_part_of_the_sample_space_and_applies_without_change() -> None:
    observed = pair()
    bound = enabled_bound(observed)
    sampled = bound.sample_token(ScriptedIndexSource(0))

    assert sampled == bound.identity_token()
    assert sampled.is_identity is True
    assert cast(CircularShiftStateV2, sampled.state).shift == 0
    transformed = bound.apply(sampled)
    assert transformed.pair == observed
    assert transformed.source_changed is False
    assert transformed.diagnostics == ()


def test_apply_returns_owned_token_and_arrays_unaffected_by_caller_token_mutation() -> None:
    bound = enabled_bound()
    input_token = bound.sample_token(ScriptedIndexSource(2))

    result = bound.apply(input_token)
    expected_source_bytes = result.pair.source.tobytes(order="C")
    expected_target_bytes = result.pair.target.tobytes(order="C")
    expected_result_digest = circular_module.circular_token_sha256(result.token)

    object.__setattr__(input_token, "scientific_plan_sha256", "1" * 64)
    object.__setattr__(cast(CircularShiftStateV2, input_token.state), "shift", 3)

    assert result.token is not input_token
    assert cast(CircularShiftStateV2, result.token.state).shift == 2
    assert circular_module.circular_token_sha256(result.token) == expected_result_digest
    assert result.pair.source.tobytes(order="C") == expected_source_bytes
    assert result.pair.target.tobytes(order="C") == expected_target_bytes


def test_apply_uses_one_canonical_token_snapshot_across_transform_injection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bound = enabled_bound()
    input_token = bound.sample_token(ScriptedIndexSource(2))
    original_apply_state = circular_module._apply_state

    def mutate_caller_token_then_transform(
        pair_value: SeriesPair,
        shift: int,
    ) -> SeriesPair:
        object.__setattr__(input_token, "scientific_plan_sha256", "1" * 64)
        object.__setattr__(cast(CircularShiftStateV2, input_token.state), "shift", 3)
        return original_apply_state(pair_value, shift)

    monkeypatch.setattr(
        circular_module,
        "_apply_state",
        mutate_caller_token_then_transform,
    )

    result = bound.apply(input_token)

    assert result.token is not input_token
    assert cast(CircularShiftStateV2, result.token.state).shift == 2
    assert result.token.scientific_plan_sha256 == PLAN_SHA
    assert np.array_equal(result.pair.source, (4.0, 5.0, 0.0, 3.0, 1.0, 2.0))


def test_apply_rejects_hook_replacing_bound_pair_after_token_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bound = enabled_bound()
    token = bound.sample_token(ScriptedIndexSource(2))
    replacement = pair(
        source=(10.0, 11.0, 12.0, 13.0, 14.0, 15.0),
        target=(20.0, 21.0, 22.0, 23.0, 24.0, 25.0),
    )
    original_apply = circular_module._apply_state

    def replace_bound_then_transform(
        pair_value: SeriesPair,
        shift: int,
    ) -> SeriesPair:
        del pair_value
        object.__setattr__(bound, "_observed_pair", replacement)
        return original_apply(replacement, shift)

    monkeypatch.setattr(circular_module, "_apply_state", replace_bound_then_transform)

    with pytest.raises(V2IntegrityError):
        bound.apply(token)


def test_apply_rejects_hook_mutating_its_transform_input_in_place(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bound = enabled_bound()
    token = bound.sample_token(ScriptedIndexSource(2))
    replacement = pair(
        source=(10.0, 11.0, 12.0, 13.0, 14.0, 15.0),
        target=(20.0, 21.0, 22.0, 23.0, 24.0, 25.0),
    )
    original_apply = circular_module._apply_state

    def mutate_input_then_transform(
        pair_value: SeriesPair,
        shift: int,
    ) -> SeriesPair:
        object.__setattr__(pair_value, "source", replacement.source)
        object.__setattr__(pair_value, "target", replacement.target)
        return original_apply(pair_value, shift)

    monkeypatch.setattr(circular_module, "_apply_state", mutate_input_then_transform)

    with pytest.raises(V2IntegrityError):
        bound.apply(token)


@pytest.mark.parametrize("damage", ["replace_pair", "source", "target"])
def test_apply_rejects_same_length_bound_content_drift_before_transform(
    monkeypatch: pytest.MonkeyPatch,
    damage: str,
) -> None:
    bound = enabled_bound()
    token = bound.sample_token(ScriptedIndexSource(2))
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
        circular_module,
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
        circular_module,
        "_apply_state",
        lambda *_: (_ for _ in ()).throw(AssertionError("transform was called")),
    )

    with pytest.raises(V2IntegrityError):
        bound.apply(token)


def test_apply_rejects_a_different_legal_shift_even_if_other_invariants_hold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bound = enabled_bound()
    token = bound.sample_token(ScriptedIndexSource(2))
    original_apply_state = circular_module._apply_state
    monkeypatch.setattr(
        circular_module,
        "_apply_state",
        lambda pair_value, shift: original_apply_state(pair_value, shift + 1),
    )

    with pytest.raises(V2IntegrityError, match="exact right shift"):
        bound.apply(token)


def test_periodic_nonidentity_state_is_retained_without_resampling() -> None:
    observed = pair(
        source=(1.0, 2.0, 1.0, 2.0),
        target=(9.0, 8.0, 7.0, 6.0),
    )
    bound = enabled_bound(observed)
    random = ScriptedIndexSource(2)

    token = bound.sample_token(random)
    transformed = bound.apply(token)

    assert cast(CircularShiftStateV2, token.state).shift == 2
    assert token.is_identity is False
    assert transformed.pair == observed
    assert transformed.source_changed is False
    assert transformed.diagnostics == ("nonidentity_data_equivalent_v2",)
    assert random.bounds == [4]


def test_sampling_is_with_replacement_and_does_not_deduplicate_states() -> None:
    bound = enabled_bound(
        pair(source=(1.0, 2.0, 1.0, 2.0), target=(4.0, 3.0, 2.0, 1.0))
    )
    random = ScriptedIndexSource(2, 2, 0, 2)

    tokens = tuple(bound.sample_token(random) for _ in range(4))

    assert tuple(cast(CircularShiftStateV2, token.state).shift for token in tokens) == (
        2,
        2,
        0,
        2,
    )
    assert tokens[0] == tokens[1] == tokens[3]
    assert random.bounds == [4, 4, 4, 4]


def test_empty_shift_space_returns_exact_disabled_sum_type() -> None:
    result = CircularShiftNullV2(min_shift=4).bind(
        pair(),
        semantic_input_sha256=SEMANTIC_SHA,
        scientific_plan_sha256=PLAN_SHA,
    )

    assert result.status is NullBindStatus.DISABLED
    assert result.bound is None
    assert result.disabled_reason is NullDisabledReason.SHIFT_SPACE_EMPTY
    assert result.diagnostics == ("shift_space_empty_v2",)


def test_empty_shift_space_does_not_copy_or_hash_complete_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = pair()

    def forbidden_pair_construction(
        self: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        del self, args, kwargs
        raise AssertionError("disabled bind copied the complete pair")

    monkeypatch.setattr(SeriesPair, "__init__", forbidden_pair_construction)
    monkeypatch.setattr(
        circular_module,
        "_owned_pair_content_guard_sha256",
        lambda *_: (_ for _ in ()).throw(AssertionError("disabled bind hashed pair")),
        raising=False,
    )

    result = CircularShiftNullV2(min_shift=4).bind(
        observed,
        semantic_input_sha256=SEMANTIC_SHA,
        scientific_plan_sha256=PLAN_SHA,
    )

    assert result.status is NullBindStatus.DISABLED
    assert result.disabled_reason is NullDisabledReason.SHIFT_SPACE_EMPTY


def test_half_length_shift_has_exactly_one_nonidentity_state_and_diagnostic() -> None:
    result = CircularShiftNullV2(min_shift=3).bind(
        pair(),
        semantic_input_sha256=SEMANTIC_SHA,
        scientific_plan_sha256=PLAN_SHA,
    )

    assert result.status is NullBindStatus.ENABLED
    assert result.bound is not None
    assert result.bound.total_state_count == 2
    assert result.diagnostics == ("circular_shift_orbit_resolution_v2",)
    token = result.bound.sample_token(ScriptedIndexSource(1))
    assert cast(CircularShiftStateV2, token.state).shift == 3


@pytest.mark.parametrize("value", [True, np.int64(1), 0, -1, 1.0, "1"])
def test_min_shift_requires_an_exact_positive_builtin_integer(value: object) -> None:
    with pytest.raises((TypeError, ValueError, V2IntegrityError)):
        CircularShiftNullV2(min_shift=value)  # type: ignore[arg-type]


def test_from_parameters_requires_the_exact_parameter_field_set() -> None:
    contract = CircularShiftNullV2.from_parameters({"min_shift": 2})
    assert contract.name == "circular_shift_v2"
    assert contract.parameters == {"min_shift": 2}
    assert len(contract.null_parameter_sha256) == 64

    for invalid in (
        {},
        {"min_shift": 1, "extra": 2},
        {"min_shift": True},
        MappingProxyType({"min_shift": np.int64(1)}),
    ):
        with pytest.raises((TypeError, ValueError, V2IntegrityError)):
            CircularShiftNullV2.from_parameters(invalid)


def test_bind_types_damaged_exact_pair_fields() -> None:
    observed = pair()
    object.__delattr__(observed, "source")

    with pytest.raises(V2IntegrityError, match="fields"):
        CircularShiftNullV2(min_shift=1).bind(
            observed,
            semantic_input_sha256=SEMANTIC_SHA,
            scientific_plan_sha256=PLAN_SHA,
        )


def test_contract_bound_parameters_and_arrays_are_frozen_owned_snapshots() -> None:
    contract = CircularShiftNullV2(min_shift=1)
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
    object.__setattr__(
        observed,
        "source",
        np.asarray((7.0, 7.0, 7.0, 7.0), dtype=np.float64),
    )
    transformed = bound.apply(bound.identity_token())

    assert np.array_equal(transformed.pair.source, (0.0, 1.0, 2.0, 3.0))
    assert np.array_equal(transformed.pair.target, (3.0, 2.0, 1.0, 0.0))
    assert transformed.pair.source.flags.writeable is False
    assert transformed.pair.target.flags.writeable is False
    with pytest.raises(TypeError):
        bound.parameters["min_shift"] = 3  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        contract.min_shift = 3  # type: ignore[misc]


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
            raise AssertionError("foreign token touched source data")

        @property
        def target(self) -> object:
            raise AssertionError("foreign token touched target data")

    object.__setattr__(bound, "_observed_pair", ExplodingPair())
    monkeypatch.setattr(
        circular_module,
        "_apply_state",
        lambda *_: (_ for _ in ()).throw(AssertionError("transform was called")),
    )

    with pytest.raises(V2IntegrityError):
        bound.apply(token)


def test_identity_flag_state_disagreement_is_rejected_before_transform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bound = enabled_bound()
    token = bound.identity_token()
    object.__setattr__(token, "is_identity", False)
    monkeypatch.setattr(
        circular_module,
        "_apply_state",
        lambda *_: (_ for _ in ()).throw(AssertionError("transform was called")),
    )

    with pytest.raises(V2IntegrityError, match="identity"):
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
def test_equal_string_subclass_in_token_fields_is_rejected_before_data_access(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    bound = enabled_bound()
    token = bound.identity_token()
    object.__setattr__(token, field, EqualString(getattr(token, field)))

    class ExplodingPair:
        @property
        def source(self) -> object:
            raise AssertionError("invalid token touched source data")

        @property
        def target(self) -> object:
            raise AssertionError("invalid token touched target data")

    object.__setattr__(bound, "_observed_pair", ExplodingPair())
    monkeypatch.setattr(
        circular_module,
        "_apply_state",
        lambda *_: (_ for _ in ()).throw(AssertionError("transform was called")),
    )

    with pytest.raises(V2IntegrityError):
        bound.apply(token)


def test_equal_string_subclass_in_state_schema_is_rejected_before_data_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bound = enabled_bound()
    token = bound.identity_token()
    state = cast(CircularShiftStateV2, token.state)
    object.__setattr__(state, "schema", EqualString(state.schema))

    class ExplodingPair:
        @property
        def source(self) -> object:
            raise AssertionError("invalid state touched source data")

        @property
        def target(self) -> object:
            raise AssertionError("invalid state touched target data")

    object.__setattr__(bound, "_observed_pair", ExplodingPair())
    monkeypatch.setattr(
        circular_module,
        "_apply_state",
        lambda *_: (_ for _ in ()).throw(AssertionError("transform was called")),
    )

    with pytest.raises(V2IntegrityError):
        bound.apply(token)


@pytest.mark.parametrize(
    ("boundary", "deleted_owner", "field"),
    [
        ("apply", "token", "schema"),
        ("apply", "state", "shift"),
        ("digest", "token", "null_name"),
        ("digest", "state", "schema"),
    ],
)
def test_deleted_token_or_state_slots_are_typed_integrity_failures(
    boundary: str,
    deleted_owner: str,
    field: str,
) -> None:
    bound = enabled_bound()
    token = bound.sample_token(ScriptedIndexSource(2))
    target = token if deleted_owner == "token" else token.state
    object.__delattr__(target, field)

    with pytest.raises(V2IntegrityError):
        if boundary == "apply":
            bound.apply(token)
        else:
            circular_module.circular_token_sha256(token)


def test_apply_rejects_bound_parameter_and_owner_drift_before_transform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for field, value, message in (
        ("_min_shift", 2, "parameter"),
        ("_bound_null_owner_sha256", "f" * 64, "owner"),
    ):
        bound = enabled_bound()
        token = bound.identity_token()
        object.__setattr__(bound, field, value)
        monkeypatch.setattr(
            circular_module,
            "_apply_state",
            lambda *_: (_ for _ in ()).throw(AssertionError("transform was called")),
        )
        with pytest.raises(V2IntegrityError, match=message):
            bound.apply(token)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("_min_shift", True, "parameter"),
        ("_observed_length", True, "length"),
        ("_min_shift", 4, "state space"),
    ],
)
def test_apply_types_invalid_bound_scalar_state_before_transform(
    field: str,
    value: object,
    message: str,
) -> None:
    bound = enabled_bound()
    token = bound.identity_token()
    object.__setattr__(bound, field, value)

    with pytest.raises(V2IntegrityError, match=message):
        bound.apply(token)


def test_apply_rejects_owned_length_drift_against_golden_pair() -> None:
    bound = enabled_bound()
    token = bound.identity_token()
    drifted_length = 8
    drifted_owner = circular_module._owner_digest(
        semantic_input_sha256=SEMANTIC_SHA,
        scientific_plan_sha256=PLAN_SHA,
        null_parameter_sha256=token.null_parameter_sha256,
        observed_length=drifted_length,
    )
    object.__setattr__(bound, "_observed_length", drifted_length)
    object.__setattr__(bound, "_bound_null_owner_sha256", drifted_owner)
    token = replace(token, bound_null_owner_sha256=drifted_owner)

    with pytest.raises(V2IntegrityError, match="pair length"):
        bound.apply(token)


def test_apply_rejects_derived_token_and_foreign_state_exact_types() -> None:
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

    foreign_state_token = bound.identity_token()
    object.__setattr__(
        foreign_state_token,
        "state",
        BlockShuffleStateV2(
            schema="selcal.block-shuffle-state.v2",
            block_order=(0, 1),
        ),
    )
    with pytest.raises(V2IntegrityError, match="state type"):
        bound.apply(foreign_state_token)


def test_out_of_space_state_is_rejected_before_transform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bound = enabled_bound(min_shift=2)
    token = replace(
        bound.identity_token(),
        is_identity=False,
        state=CircularShiftStateV2(
            schema="selcal.circular-shift-state.v2",
            shift=1,
        ),
    )
    monkeypatch.setattr(
        circular_module,
        "_apply_state",
        lambda *_: (_ for _ in ()).throw(AssertionError("transform was called")),
    )

    with pytest.raises(V2IntegrityError, match="state space"):
        bound.apply(token)


@pytest.mark.parametrize("drift", ["shape", "target", "multiset"])
def test_apply_rejects_hostile_transform_postcondition_drift(
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    observed = pair()
    bound = enabled_bound(observed)
    token = bound.sample_token(ScriptedIndexSource(1))

    def hostile_apply(pair_value: SeriesPair, shift: int) -> SeriesPair:
        del shift
        if drift == "shape":
            return SeriesPair(
                source=pair_value.source[:-1],
                target=pair_value.target[:-1],
            )
        if drift == "target":
            changed_target = pair_value.target.copy()
            changed_target[0] += 1.0
            return SeriesPair(source=pair_value.source, target=changed_target)
        changed_source = pair_value.source.copy()
        changed_source[0] += 100.0
        return SeriesPair(source=changed_source, target=pair_value.target)

    monkeypatch.setattr(circular_module, "_apply_state", hostile_apply)

    with pytest.raises(V2IntegrityError, match=drift):
        bound.apply(token)


def test_apply_rejects_non_series_pair_transform_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bound = enabled_bound()
    token = bound.identity_token()
    monkeypatch.setattr(circular_module, "_apply_state", lambda *_: object())

    with pytest.raises(V2IntegrityError, match="exact SeriesPair"):
        bound.apply(token)


def test_apply_rejects_target_only_shape_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bound = enabled_bound()
    token = bound.identity_token()

    def target_only_shape_drift(pair_value: SeriesPair, shift: int) -> SeriesPair:
        del shift
        transformed = SeriesPair(source=pair_value.source, target=pair_value.target)
        object.__setattr__(transformed, "target", transformed.target[:-1])
        return transformed

    monkeypatch.setattr(circular_module, "_apply_state", target_only_shape_drift)

    with pytest.raises(V2IntegrityError, match="shape"):
        bound.apply(token)


def test_apply_rejects_identity_source_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bound = enabled_bound()
    token = bound.identity_token()
    original_apply_state = circular_module._apply_state
    monkeypatch.setattr(
        circular_module,
        "_apply_state",
        lambda pair_value, shift: original_apply_state(pair_value, shift + 1),
    )

    with pytest.raises(V2IntegrityError, match=r"identity.*drift"):
        bound.apply(token)


@pytest.mark.parametrize(
    "random_value",
    [-1, 6, True, np.int64(0), 0.0],
)
def test_sampler_rejects_non_exact_or_out_of_range_indices(random_value: object) -> None:
    bound = enabled_bound()

    class InvalidIndexSource:
        def randbelow(self, bound_value: int, /) -> object:
            del bound_value
            return random_value

    with pytest.raises(V2IntegrityError, match="index"):
        bound.sample_token(InvalidIndexSource())  # type: ignore[arg-type]


def test_sampler_rejects_missing_and_noncallable_randbelow() -> None:
    bound = enabled_bound()

    class Missing:
        pass

    class NonCallable:
        randbelow = 0

    for source in (Missing(), NonCallable()):
        with pytest.raises(V2IntegrityError, match="randbelow"):
            bound.sample_token(source)  # type: ignore[arg-type]


def test_sampler_rejects_property_without_invoking_its_getter() -> None:
    bound = enabled_bound()

    class PropertySource:
        def __init__(self) -> None:
            self.getter_calls = 0

        @property
        def randbelow(self):  # type: ignore[no-untyped-def]
            self.getter_calls += 1
            return lambda bound_value: 0

    source = PropertySource()
    with pytest.raises(V2IntegrityError, match="property"):
        bound.sample_token(source)  # type: ignore[arg-type]
    assert source.getter_calls == 0


def test_sampler_accepts_supported_python_and_c_descriptor_shapes() -> None:
    bound = enabled_bound()

    class Ordinary:
        def randbelow(self, bound_value: int, /) -> int:
            assert bound_value == 6
            return 0

    class Static:
        @staticmethod
        def randbelow(bound_value: int, /) -> int:
            assert bound_value == 6
            return 0

    class Class:
        @classmethod
        def randbelow(cls, bound_value: int, /) -> int:
            assert cls is Class
            assert bound_value == 6
            return 0

    class CDescriptor(list[int]):
        randbelow = list.count

    class IndexCallable:
        def __call__(self, bound_value: int, /) -> int:
            assert bound_value == 6
            return 0

    class InstanceCallable:
        def __init__(self) -> None:
            self.randbelow = IndexCallable()

    sources = (Ordinary(), Static(), Class(), CDescriptor(), InstanceCallable())
    for source in sources:
        token = bound.sample_token(source)  # type: ignore[arg-type]
        assert token.is_identity is True


@pytest.mark.parametrize("binding_exception", [AttributeError, TypeError])
def test_sampler_types_descriptor_binding_failures(
    binding_exception: type[Exception],
) -> None:
    bound = enabled_bound()

    class BrokenDescriptor:
        def __get__(self, instance: object, owner: type[object]):
            del instance, owner
            raise binding_exception("cannot bind")

    class Source:
        randbelow = BrokenDescriptor()

    with pytest.raises(V2IntegrityError, match="could not be bound"):
        bound.sample_token(Source())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "execution_exception",
    [
        AttributeError,
        TypeError,
        StopIteration,
        RuntimeError,
        OSError,
        MemoryError,
        SystemExit,
    ],
)
def test_sampler_preserves_actual_randbelow_execution_exceptions(
    execution_exception: type[BaseException],
) -> None:
    bound = enabled_bound()

    class Source:
        def randbelow(self, bound_value: int, /) -> int:
            assert bound_value == 6
            raise execution_exception("actual execution failure")

    with pytest.raises(execution_exception, match="actual execution failure"):
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
        CircularShiftNullV2(min_shift=1).bind(pair(), **kwargs)  # type: ignore[arg-type]


def test_bind_requires_an_exact_series_pair() -> None:
    with pytest.raises(V2IntegrityError):
        CircularShiftNullV2(min_shift=1).bind(
            object(),  # type: ignore[arg-type]
            semantic_input_sha256=SEMANTIC_SHA,
            scientific_plan_sha256=PLAN_SHA,
        )


def test_private_bound_constructor_integrity_branches_are_fail_closed() -> None:
    observed = pair()
    parameter_sha = circular_module._parameter_digest(1)
    owner_sha = circular_module._owner_digest(
        semantic_input_sha256=SEMANTIC_SHA,
        scientific_plan_sha256=PLAN_SHA,
        null_parameter_sha256=parameter_sha,
        observed_length=6,
    )
    common = {
        "_observed_pair": observed,
        "_min_shift": 1,
        "_semantic_input_sha256": SEMANTIC_SHA,
        "_scientific_plan_sha256": PLAN_SHA,
        "_null_parameter_sha256": parameter_sha,
        "_bound_null_owner_sha256": owner_sha,
        "_observed_content_guard_sha256": (
            circular_module._owned_pair_content_guard_sha256(observed)
        ),
    }

    hostile_values = (
        {"_observed_pair": object()},
        {"_min_shift": True},
        {
            "_min_shift": 4,
            "_null_parameter_sha256": circular_module._parameter_digest(4),
        },
        {"_null_parameter_sha256": "f" * 64},
        {"_bound_null_owner_sha256": "f" * 64},
        {"_observed_content_guard_sha256": "f" * 64},
        {"_observed_content_guard_sha256": "F" * 64},
    )
    for changes in hostile_values:
        values = dict(common)
        values.update(changes)
        with pytest.raises(V2IntegrityError):
            circular_module._BoundCircularShiftV2(**values)


def test_private_bound_constructor_types_damaged_exact_pair_slots() -> None:
    observed = pair()
    content_guard = circular_module._owned_pair_content_guard_sha256(observed)
    object.__delattr__(observed, "source")

    with pytest.raises(V2IntegrityError, match="snapshot"):
        circular_module._BoundCircularShiftV2(
            _observed_pair=observed,
            _min_shift=1,
            _semantic_input_sha256=SEMANTIC_SHA,
            _scientific_plan_sha256=PLAN_SHA,
            _null_parameter_sha256=circular_module._parameter_digest(1),
            _bound_null_owner_sha256=circular_module._owner_digest(
                semantic_input_sha256=SEMANTIC_SHA,
                scientific_plan_sha256=PLAN_SHA,
                null_parameter_sha256=circular_module._parameter_digest(1),
                observed_length=6,
            ),
            _observed_content_guard_sha256=content_guard,
        )


@pytest.mark.parametrize(
    "slot",
    [
        "_observed_pair",
        "_min_shift",
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
