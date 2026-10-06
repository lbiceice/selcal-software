"""Exact-enumeration circular null: every non-identity state exactly once, never sampled."""

from __future__ import annotations

import numpy as np
import pytest

from selcal.contracts import SeriesPair
from selcal.contracts_v2 import CircularShiftStateV2, NullBindStatus, V2IntegrityError
from selcal.nulls.circular_shift_exact_v1 import CircularShiftExactNullV1
from selcal.nulls.circular_shift_v2 import CircularShiftNullV2

SEMANTIC_SHA = "8" * 64
PLAN_SHA = "d" * 64


def pair(n: int = 6) -> SeriesPair:
    return SeriesPair(
        source=np.arange(n, dtype=np.float64),
        target=np.asarray([0.0, 1.0, 3.0, 2.0, 5.0, 4.0] * ((n + 5) // 6), dtype=np.float64)[:n],
    )


def bound(observed: SeriesPair | None = None, *, min_shift: int = 1):
    result = CircularShiftExactNullV1(min_shift=min_shift).bind(
        observed if observed is not None else pair(),
        semantic_input_sha256=SEMANTIC_SHA,
        scientific_plan_sha256=PLAN_SHA,
    )
    assert result.status is NullBindStatus.ENABLED
    assert result.bound is not None
    return result.bound


def test_name_and_parameters_are_its_own():
    null = CircularShiftExactNullV1(min_shift=1)
    assert null.name == "circular_shift_exact_v1"
    assert dict(null.parameters) == {"min_shift": 1}


@pytest.mark.parametrize("min_shift", [0, 2, 3])
def test_only_the_full_group_is_accepted(min_shift):
    with pytest.raises(V2IntegrityError):
        CircularShiftExactNullV1(min_shift=min_shift)


def test_enumerate_token_covers_every_nonidentity_state_exactly_once():
    enumerating = bound()
    shifts = [enumerating.enumerate_token(index).state.shift for index in range(5)]
    assert shifts == [1, 2, 3, 4, 5]
    assert enumerating.total_state_count == 6
    assert enumerating.identity_token().state.shift == 0
    assert enumerating.identity_token().is_identity is True


@pytest.mark.parametrize("index", [-1, 5, 6, True, 1.0, "1"])
def test_enumerate_token_rejects_out_of_range_or_inexact_indexes(index):
    with pytest.raises(V2IntegrityError):
        bound().enumerate_token(index)


def test_sample_token_is_refused_because_this_null_never_samples():
    class Source:
        def randbelow(self, bound_: int, /) -> int:
            return 0

    with pytest.raises(V2IntegrityError):
        bound().sample_token(Source())


def test_apply_matches_the_sampling_null_state_for_state_for_the_same_shift():
    observed = pair(6)
    exact = bound(observed)
    sampling = (
        CircularShiftNullV2(min_shift=1)
        .bind(observed, semantic_input_sha256=SEMANTIC_SHA, scientific_plan_sha256=PLAN_SHA)
        .bound
    )
    assert sampling is not None
    for index in range(5):
        token = exact.enumerate_token(index)
        mirror = sampling._token_for_shift(token.state.shift)
        assert np.array_equal(exact.apply(token).pair.source, sampling.apply(mirror).pair.source)


def test_tokens_carry_this_null_name_and_its_own_state_schema():
    token = bound().enumerate_token(0)
    assert token.null_name == "circular_shift_exact_v1"
    assert type(token.state) is CircularShiftStateV2
    assert token.is_identity is False


def test_bind_disables_when_the_state_space_is_too_small():
    result = CircularShiftExactNullV1(min_shift=1).bind(
        SeriesPair(source=np.ones(1), target=np.zeros(1)),
        semantic_input_sha256=SEMANTIC_SHA,
        scientific_plan_sha256=PLAN_SHA,
    )
    assert result.status is NullBindStatus.DISABLED
