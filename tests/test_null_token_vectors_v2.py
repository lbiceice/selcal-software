from __future__ import annotations

import hashlib
from typing import cast

import numpy as np
import pytest

from selcal.canonical import canonical_json_bytes
from selcal.contracts import JsonValue, SeriesPair
from selcal.contracts_v2 import (
    BlockShuffleStateV2,
    CircularShiftStateV2,
    NullBindStatus,
    V2IntegrityError,
)
from selcal.nulls.block_shuffle_v2 import (
    BlockShuffleNullV2,
    block_token_sha256,
)
from selcal.nulls.circular_shift_v2 import (
    CircularShiftNullV2,
    circular_token_sha256,
)
from selcal.randomness import ReplicateRandomSource

SEMANTIC_SHA = "82aa122557ed48908df8bb6ce226f05bdb5c7e4cf4287a671f74ee1eee9af645"
PLAN_SHA = "dc7462598231451b36a94e2586037aa921a8159b2eabebb5fe5c81dcafa48931"
PARAMETER_SHA = "a2e6b608e05886b0c06388750e201c922c21d08e4ca6faf7668742798895dda3"
OWNER_SHA = "b5c33b348ba31477f0c085e7766eb39e1ce00b8a264fefa86eaf59d52c282831"
TOKEN_SHA = "b5fadafeee3a831b3fd3b4db0ed3851274a74a4bdeff2201757c6e96ed983486"
BLOCK_PLAN_SHA = "661d4b029d4f8dccc2545a8ad8776303ac06685e7410d48ffbe1484d41add810"
BLOCK_PARAMETER_SHA = "7e59ac4c97edabc3549303e6035f25a8c262f378e5440fa6ee8aeac5ef37c28a"
BLOCK_OWNER_SHA = "90be6919fffddc18211b062ba0f913320bc997888a8968f895a2bb02ec7346e2"
BLOCK_TOKEN_SHA = "7584cdb3acb9b4dae6a49303fa3bc3f96764f91db43db2cb2e4deaa609430670"


class IndexTwo:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def randbelow(self, bound: int, /) -> int:
        self.calls.append(bound)
        return 2


def normative_pair() -> SeriesPair:
    return SeriesPair(
        source=np.asarray((0.0, 3.0, 1.0, 2.0, 4.0, 5.0), dtype=np.float64),
        target=np.asarray((0.0, 1.0, 3.0, 2.0, 5.0, 4.0), dtype=np.float64),
    )


def test_approved_circular_parameter_owner_and_token_vectors() -> None:
    parameter_payload: JsonValue = {
        "schema": "selcal.null-parameters.v2",
        "name": "circular_shift_v2",
        "params": {"min_shift": 1},
    }
    owner_payload: JsonValue = {
        "schema": "selcal.bound-null-owner.v2",
        "semantic_input_sha256": SEMANTIC_SHA,
        "scientific_plan_sha256": PLAN_SHA,
        "null_parameter_sha256": PARAMETER_SHA,
        "observed_length": 6,
        "transformed_role": "source",
    }
    token_payload: JsonValue = {
        "schema": "selcal.null-transform-token.v2",
        "null_name": "circular_shift_v2",
        "null_parameter_sha256": PARAMETER_SHA,
        "semantic_input_sha256": SEMANTIC_SHA,
        "scientific_plan_sha256": PLAN_SHA,
        "bound_null_owner_sha256": OWNER_SHA,
        "is_identity": False,
        "state": {
            "schema": "selcal.circular-shift-state.v2",
            "shift": 2,
        },
    }
    expected_parameter_bytes = (
        b'{"name":"circular_shift_v2","params":{"min_shift":1},'
        b'"schema":"selcal.null-parameters.v2"}'
    )
    expected_owner_bytes = (
        b'{"null_parameter_sha256":"a2e6b608e05886b0c06388750e201c922c21d08e4ca6faf7668742798895dda3",'
        b'"observed_length":6,"schema":"selcal.bound-null-owner.v2",'
        b'"scientific_plan_sha256":"dc7462598231451b36a94e2586037aa921a8159b2eabebb5fe5c81dcafa48931",'
        b'"semantic_input_sha256":"82aa122557ed48908df8bb6ce226f05bdb5c7e4cf4287a671f74ee1eee9af645",'
        b'"transformed_role":"source"}'
    )
    expected_token_bytes = (
        b'{"bound_null_owner_sha256":"b5c33b348ba31477f0c085e7766eb39e1ce00b8a264fefa86eaf59d52c282831",'
        b'"is_identity":false,"null_name":"circular_shift_v2",'
        b'"null_parameter_sha256":"a2e6b608e05886b0c06388750e201c922c21d08e4ca6faf7668742798895dda3",'
        b'"schema":"selcal.null-transform-token.v2",'
        b'"scientific_plan_sha256":"dc7462598231451b36a94e2586037aa921a8159b2eabebb5fe5c81dcafa48931",'
        b'"semantic_input_sha256":"82aa122557ed48908df8bb6ce226f05bdb5c7e4cf4287a671f74ee1eee9af645",'
        b'"state":{"schema":"selcal.circular-shift-state.v2","shift":2}}'
    )

    assert canonical_json_bytes(parameter_payload) == expected_parameter_bytes
    assert canonical_json_bytes(owner_payload) == expected_owner_bytes
    assert canonical_json_bytes(token_payload) == expected_token_bytes
    assert hashlib.sha256(expected_parameter_bytes).hexdigest() == PARAMETER_SHA
    assert hashlib.sha256(expected_owner_bytes).hexdigest() == OWNER_SHA
    assert hashlib.sha256(expected_token_bytes).hexdigest() == TOKEN_SHA

    result = CircularShiftNullV2(min_shift=1).bind(
        normative_pair(),
        semantic_input_sha256=SEMANTIC_SHA,
        scientific_plan_sha256=PLAN_SHA,
    )
    assert result.status is NullBindStatus.ENABLED
    assert result.bound is not None
    bound = result.bound
    random = IndexTwo()
    token = bound.sample_token(random)

    assert bound.parameters == {"min_shift": 1}
    assert bound.null_parameter_sha256 == PARAMETER_SHA  # type: ignore[attr-defined]
    assert bound.bound_null_owner_sha256 == OWNER_SHA  # type: ignore[attr-defined]
    assert token.schema == "selcal.null-transform-token.v2"
    assert token.null_name == "circular_shift_v2"
    assert token.null_parameter_sha256 == PARAMETER_SHA
    assert token.semantic_input_sha256 == SEMANTIC_SHA
    assert token.scientific_plan_sha256 == PLAN_SHA
    assert token.bound_null_owner_sha256 == OWNER_SHA
    assert token.is_identity is False
    assert cast(CircularShiftStateV2, token.state).shift == 2
    assert circular_token_sha256(token) == TOKEN_SHA
    assert random.calls == [6]


def test_approved_block_parameter_owner_and_token_vectors() -> None:
    parameter_payload: JsonValue = {
        "schema": "selcal.null-parameters.v2",
        "name": "block_shuffle_v2",
        "params": {"block_length": 2},
    }
    owner_payload: JsonValue = {
        "schema": "selcal.bound-null-owner.v2",
        "semantic_input_sha256": SEMANTIC_SHA,
        "scientific_plan_sha256": BLOCK_PLAN_SHA,
        "null_parameter_sha256": BLOCK_PARAMETER_SHA,
        "observed_length": 6,
        "transformed_role": "source",
    }
    token_payload: JsonValue = {
        "schema": "selcal.null-transform-token.v2",
        "null_name": "block_shuffle_v2",
        "null_parameter_sha256": BLOCK_PARAMETER_SHA,
        "semantic_input_sha256": SEMANTIC_SHA,
        "scientific_plan_sha256": BLOCK_PLAN_SHA,
        "bound_null_owner_sha256": BLOCK_OWNER_SHA,
        "is_identity": False,
        "state": {
            "schema": "selcal.block-shuffle-state.v2",
            "block_order": [2, 1, 0],
        },
    }
    expected_parameter_bytes = (
        b'{"name":"block_shuffle_v2","params":{"block_length":2},'
        b'"schema":"selcal.null-parameters.v2"}'
    )
    expected_owner_bytes = (
        b'{"null_parameter_sha256":"7e59ac4c97edabc3549303e6035f25a8c262f378e5440fa6ee8aeac5ef37c28a",'
        b'"observed_length":6,"schema":"selcal.bound-null-owner.v2",'
        b'"scientific_plan_sha256":"661d4b029d4f8dccc2545a8ad8776303ac06685e7410d48ffbe1484d41add810",'
        b'"semantic_input_sha256":"82aa122557ed48908df8bb6ce226f05bdb5c7e4cf4287a671f74ee1eee9af645",'
        b'"transformed_role":"source"}'
    )
    expected_token_bytes = (
        b'{"bound_null_owner_sha256":"90be6919fffddc18211b062ba0f913320bc997888a8968f895a2bb02ec7346e2",'
        b'"is_identity":false,"null_name":"block_shuffle_v2",'
        b'"null_parameter_sha256":"7e59ac4c97edabc3549303e6035f25a8c262f378e5440fa6ee8aeac5ef37c28a",'
        b'"schema":"selcal.null-transform-token.v2",'
        b'"scientific_plan_sha256":"661d4b029d4f8dccc2545a8ad8776303ac06685e7410d48ffbe1484d41add810",'
        b'"semantic_input_sha256":"82aa122557ed48908df8bb6ce226f05bdb5c7e4cf4287a671f74ee1eee9af645",'
        b'"state":{"block_order":[2,1,0],"schema":"selcal.block-shuffle-state.v2"}}'
    )

    assert canonical_json_bytes(parameter_payload) == expected_parameter_bytes
    assert canonical_json_bytes(owner_payload) == expected_owner_bytes
    assert canonical_json_bytes(token_payload) == expected_token_bytes
    assert hashlib.sha256(expected_parameter_bytes).hexdigest() == BLOCK_PARAMETER_SHA
    assert hashlib.sha256(expected_owner_bytes).hexdigest() == BLOCK_OWNER_SHA
    assert hashlib.sha256(expected_token_bytes).hexdigest() == BLOCK_TOKEN_SHA

    result = BlockShuffleNullV2(block_length=2).bind(
        normative_pair(),
        semantic_input_sha256=SEMANTIC_SHA,
        scientific_plan_sha256=BLOCK_PLAN_SHA,
    )
    assert result.status is NullBindStatus.ENABLED
    assert result.bound is not None
    bound = result.bound
    random = ReplicateRandomSource(BLOCK_PLAN_SHA, 0, 999)
    token = bound.sample_token(random)
    replicate_one_token = bound.sample_token(
        ReplicateRandomSource(BLOCK_PLAN_SHA, 1, 999)
    )

    assert bound.parameters == {"block_length": 2}
    assert bound.null_parameter_sha256 == BLOCK_PARAMETER_SHA  # type: ignore[attr-defined]
    assert bound.bound_null_owner_sha256 == BLOCK_OWNER_SHA  # type: ignore[attr-defined]
    assert token.schema == "selcal.null-transform-token.v2"
    assert token.null_name == "block_shuffle_v2"
    assert token.null_parameter_sha256 == BLOCK_PARAMETER_SHA
    assert token.semantic_input_sha256 == SEMANTIC_SHA
    assert token.scientific_plan_sha256 == BLOCK_PLAN_SHA
    assert token.bound_null_owner_sha256 == BLOCK_OWNER_SHA
    assert token.is_identity is False
    assert cast(BlockShuffleStateV2, token.state).block_order == (2, 1, 0)
    assert cast(BlockShuffleStateV2, replicate_one_token.state).block_order == (1, 0, 2)
    assert block_token_sha256(token) == BLOCK_TOKEN_SHA


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "selcal.null-transform-token.v3"),
        ("null_parameter_sha256", "A" * 64),
        ("semantic_input_sha256", "f" * 63),
        ("scientific_plan_sha256", 1),
    ],
)
def test_token_evidence_digest_rejects_corrupted_frozen_fields(
    field: str,
    value: object,
) -> None:
    result = CircularShiftNullV2(min_shift=1).bind(
        normative_pair(),
        semantic_input_sha256=SEMANTIC_SHA,
        scientific_plan_sha256=PLAN_SHA,
    )
    assert result.bound is not None
    token = result.bound.sample_token(IndexTwo())
    object.__setattr__(token, field, value)

    with pytest.raises(V2IntegrityError):
        circular_token_sha256(token)


@pytest.mark.parametrize(
    "damage",
    ["type", "name", "identity", "identity_type", "state"],
)
def test_token_evidence_digest_covers_critical_integrity_damage(damage: str) -> None:
    result = CircularShiftNullV2(min_shift=1).bind(
        normative_pair(),
        semantic_input_sha256=SEMANTIC_SHA,
        scientific_plan_sha256=PLAN_SHA,
    )
    assert result.bound is not None
    token = result.bound.identity_token()

    if damage == "type":
        value: object = object()
    else:
        value = token
        if damage == "name":
            object.__setattr__(token, "null_name", "block_shuffle_v2")
        elif damage == "identity":
            object.__setattr__(token, "is_identity", False)
        elif damage == "identity_type":
            object.__setattr__(token, "is_identity", 1)
        else:
            object.__setattr__(
                token,
                "state",
                BlockShuffleStateV2(
                    schema="selcal.block-shuffle-state.v2",
                    block_order=(0, 1),
                ),
            )

    with pytest.raises(V2IntegrityError):
        circular_token_sha256(value)  # type: ignore[arg-type]
