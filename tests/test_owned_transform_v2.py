from __future__ import annotations

import inspect

import numpy as np
import pytest

from selcal.contracts import SeriesPair
from selcal.contracts_v2 import V2IntegrityError
from selcal.nulls import block_shuffle_v2, circular_shift_v2
from selcal.nulls.owned_transform_v2 import (
    _owned_pair_content_guard_sha256,
    prepare_owned_transform,
    require_sha256,
    snapshot_required_slots,
)


def test_circular_and_block_share_policy_free_ownership_facilities() -> None:
    circular_source = inspect.getsource(circular_shift_v2)
    block_source = inspect.getsource(block_shuffle_v2)

    for source in (circular_source, block_source):
        assert "from selcal.nulls.owned_transform_v2 import" in source
        assert "prepare_owned_transform" in source
        assert "finalize_transform_result" in source
        assert "def _require_sha256" not in source
        assert "def _owner_digest" not in source
        assert "def _value_multiset" not in source


@pytest.mark.parametrize("value", ["A" * 64, "f" * 63, 1, None])
def test_shared_sha_validator_is_exact_and_typed(value: object) -> None:
    with pytest.raises(V2IntegrityError):
        require_sha256(value, name="digest")


def test_shared_bound_slot_snapshot_types_missing_slots() -> None:
    class Slots:
        present = 1

    with pytest.raises(V2IntegrityError, match="missing"):
        snapshot_required_slots(
            Slots(),
            ("present", "missing"),
            subject="bound test null",
        )


def test_shared_owned_transform_rejects_wrong_pair_type() -> None:
    with pytest.raises(V2IntegrityError, match="exact SeriesPair"):
        prepare_owned_transform(
            object(),
            expected_content_guard_sha256="0" * 64,
        )


def test_shared_owned_transform_types_damaged_pair_slots() -> None:
    observed = SeriesPair(
        source=np.asarray((0.0, 1.0), dtype=np.float64),
        target=np.asarray((1.0, 0.0), dtype=np.float64),
    )
    object.__delattr__(observed, "source")

    with pytest.raises(V2IntegrityError, match="snapshot"):
        prepare_owned_transform(
            observed,
            expected_content_guard_sha256="0" * 64,
        )


def test_shared_content_guard_frames_source_and_target_roles() -> None:
    observed = SeriesPair(
        source=np.asarray((0.0, 1.0), dtype=np.float64),
        target=np.asarray((2.0, 3.0), dtype=np.float64),
    )
    swapped = SeriesPair(source=observed.target, target=observed.source)

    assert _owned_pair_content_guard_sha256(observed) != (
        _owned_pair_content_guard_sha256(swapped)
    )


def test_shared_owned_transform_rejects_content_guard_drift() -> None:
    observed = SeriesPair(
        source=np.asarray((0.0, 1.0), dtype=np.float64),
        target=np.asarray((2.0, 3.0), dtype=np.float64),
    )

    with pytest.raises(V2IntegrityError, match="content guard"):
        prepare_owned_transform(
            observed,
            expected_content_guard_sha256="0" * 64,
        )
