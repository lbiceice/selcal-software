from __future__ import annotations

import copy
import pickle
import sys
from dataclasses import FrozenInstanceError, replace
from typing import cast

import numpy as np
import pytest
from _random_behavior_graph_v2 import random_behavior_reachability_findings

import selcal.migration_v1_to_v2 as migration
import selcal.randomness as randomness
from selcal.canonical import scientific_plan_v1_sha256
from selcal.canonical_v2 import scientific_plan_v2_sha256
from selcal.contracts import JsonValue, PlanRequest
from selcal.contracts_v2 import PlanRequestV2, V2IntegrityError
from selcal.migration_v1_to_v2 import (
    MigrationFieldStatus,
    PlanMigrationChange,
    PlanMigrationRefusal,
    PlanMigrationV1ToV2,
    migrate_plan_v1_to_v2,
)
from selcal.nulls.block_shuffle_v2 import BlockShuffleNullV2
from selcal.nulls.circular_shift_v2 import CircularShiftNullV2
from selcal.resolution import resolve_plan
from selcal.resolution_v2 import PlanResolutionV2
from selcal.statistics.lagged_pearson import LaggedPearsonAdapter


def v1_request(**changes: object) -> PlanRequest:
    values: dict[str, object] = {
        "candidates": (3, 1, 2),
        "statistic_name": "lagged_pearson_v1",
        "statistic_params": {},
        "selection_rule": "max_upper",
        "null_name": "circular_shift_v1",
        "null_params": {"min_shift": 1},
        "replicates": 9,
        "alpha": 0.05,
        "tie_tolerance": 1e-12,
        "root_seed": 17,
        "failure_policy": "fail_closed_v1",
    }
    values.update(changes)
    return PlanRequest(**values)  # type: ignore[arg-type]


def construct_with_private_migration_seal(
    result: PlanMigrationV1ToV2,
    **changes: object,
) -> PlanMigrationV1ToV2:
    values: dict[str, object] = {
        "v1_scientific_plan_sha256": result.v1_scientific_plan_sha256,
        "v2_request": result.v2_request,
        "v2_resolution": result.v2_resolution,
        "v2_scientific_plan_sha256": result.v2_scientific_plan_sha256,
        "change_ledger": result.change_ledger,
        "semantic_equivalence": result.semantic_equivalence,
        "required_action": result.required_action,
    }
    values.update(changes)
    return PlanMigrationV1ToV2(
        **values,
        seal=migration._MIGRATION_SEAL,
    )  # type: ignore[arg-type]


LedgerRecord = tuple[
    str,
    bool,
    JsonValue,
    bool,
    JsonValue,
    MigrationFieldStatus,
]

_COMMON_EXPECTED_LEDGER: tuple[LedgerRecord, ...] = (
    ("alpha", True, 0.05, True, 0.05, MigrationFieldStatus.PRESERVED),
    (
        "calibration_contract",
        False,
        None,
        True,
        "full_reselection_global_mc_plus_one_v2",
        MigrationFieldStatus.ADDED,
    ),
    (
        "candidates",
        True,
        (1, 2, 3),
        True,
        (1, 2, 3),
        MigrationFieldStatus.PRESERVED,
    ),
    (
        "common_support",
        True,
        "max_candidate_lag_v1",
        True,
        "max_candidate_lag_v1",
        MigrationFieldStatus.PRESERVED,
    ),
    (
        "failure_contract",
        False,
        None,
        True,
        "exact_b_fail_closed_v2",
        MigrationFieldStatus.ADDED,
    ),
    (
        "failure_policy",
        True,
        "fail_closed_v1",
        False,
        None,
        MigrationFieldStatus.REMOVED,
    ),
    ("replicates", True, 9, True, 9, MigrationFieldStatus.PRESERVED),
    (
        "rng_contract",
        True,
        "sha256-to-pcg64-v1",
        True,
        "sha256_framed_plan_rid_stream_to_pcg64_raw64_v1",
        MigrationFieldStatus.CHANGED,
    ),
    ("root_seed", True, 17, True, 17, MigrationFieldStatus.PRESERVED),
    (
        "schema",
        True,
        "selcal.scientific-plan.v1",
        True,
        "selcal.scientific-plan.v2",
        MigrationFieldStatus.CHANGED,
    ),
    (
        "selection.decision_contract",
        False,
        None,
        True,
        "family_max_with_canonical_tie_label_v2",
        MigrationFieldStatus.ADDED,
    ),
    (
        "selection.rule",
        True,
        "max_upper",
        True,
        "max_upper",
        MigrationFieldStatus.PRESERVED,
    ),
    (
        "selection.tie_tolerance",
        True,
        1e-12,
        True,
        1e-12,
        MigrationFieldStatus.PRESERVED,
    ),
    (
        "statistic.name",
        True,
        "lagged_pearson_v1",
        True,
        "lagged_pearson_v1",
        MigrationFieldStatus.PRESERVED,
    ),
    (
        "statistic.params",
        True,
        {},
        True,
        {},
        MigrationFieldStatus.PRESERVED,
    ),
)


def expected_ledger_records(
    *,
    old_null_name: str,
    new_null_name: str,
    parameter_name: str,
    parameter_value: int,
) -> tuple[LedgerRecord, ...]:
    null_records: tuple[LedgerRecord, ...] = (
        (
            "null.name",
            True,
            old_null_name,
            True,
            new_null_name,
            MigrationFieldStatus.CHANGED,
        ),
        (
            f"null.params.{parameter_name}",
            True,
            parameter_value,
            True,
            parameter_value,
            MigrationFieldStatus.PRESERVED,
        ),
    )
    return tuple(sorted((*_COMMON_EXPECTED_LEDGER, *null_records)))


_MIGRATION_LEDGER_CASES = (
    (
        "circular_shift_v1",
        {"min_shift": 1},
        expected_ledger_records(
            old_null_name="circular_shift_v1",
            new_null_name="circular_shift_v2",
            parameter_name="min_shift",
            parameter_value=1,
        ),
    ),
    (
        "block_shuffle_v1",
        {"block_length": 2},
        expected_ledger_records(
            old_null_name="block_shuffle_v1",
            new_null_name="block_shuffle_v2",
            parameter_name="block_length",
            parameter_value=2,
        ),
    ),
)

_MIGRATION_LEDGER_DELETION_CASES = tuple(
    (old_null_name, null_params, record[0])
    for old_null_name, null_params, records in _MIGRATION_LEDGER_CASES
    for record in records
)


@pytest.mark.parametrize(
    ("old_name", "old_params", "new_name", "new_type"),
    (
        (
            "circular_shift_v1",
            {"min_shift": 1},
            "circular_shift_v2",
            CircularShiftNullV2,
        ),
        (
            "block_shuffle_v1",
            {"block_length": 2},
            "block_shuffle_v2",
            BlockShuffleNullV2,
        ),
    ),
)
def test_migration_is_explicit_non_equivalent_and_returns_both_identities(
    old_name: str,
    old_params: dict[str, int],
    new_name: str,
    new_type: type[object],
) -> None:
    source = v1_request(null_name=old_name, null_params=old_params)
    result = migrate_plan_v1_to_v2(source)

    assert type(result) is PlanMigrationV1ToV2
    assert type(result.v2_request) is PlanRequestV2
    assert type(result.v2_resolution) is PlanResolutionV2
    assert type(result.v2_resolution.adapters.null_model) is new_type
    assert result.v2_request.null_name == new_name
    assert result.v2_request.null_params == old_params
    assert result.v1_scientific_plan_sha256 == scientific_plan_v1_sha256(
        resolve_plan(source).plan
    )
    assert result.v2_scientific_plan_sha256 == scientific_plan_v2_sha256(
        result.v2_resolution.plan
    )
    assert result.v1_scientific_plan_sha256 != result.v2_scientific_plan_sha256
    assert result.semantic_equivalence == "NON_EQUIVALENT"
    assert result.required_action == "RECALIBRATION_REQUIRED"


def test_migration_ledger_is_complete_canonical_field_by_field_and_immutable() -> None:
    result = migrate_plan_v1_to_v2(v1_request())
    ledger = result.change_ledger

    assert type(ledger) is tuple
    assert tuple(entry.path for entry in ledger) == tuple(
        sorted(entry.path for entry in ledger)
    )
    assert len({entry.path for entry in ledger}) == len(ledger)
    by_path = {entry.path: entry for entry in ledger}
    assert by_path["schema"].source_value == "selcal.scientific-plan.v1"
    assert by_path["schema"].target_value == "selcal.scientific-plan.v2"
    assert by_path["schema"].status is MigrationFieldStatus.CHANGED
    assert by_path["null.name"].source_value == "circular_shift_v1"
    assert by_path["null.name"].target_value == "circular_shift_v2"
    assert by_path["null.params.min_shift"].status is MigrationFieldStatus.PRESERVED
    assert by_path["selection.decision_contract"].status is MigrationFieldStatus.ADDED
    assert by_path["failure_policy"].status is MigrationFieldStatus.REMOVED
    assert by_path["failure_contract"].status is MigrationFieldStatus.ADDED
    assert by_path["rng_contract"].status is MigrationFieldStatus.CHANGED
    assert by_path["calibration_contract"].status is MigrationFieldStatus.ADDED
    assert by_path["candidates"].status is MigrationFieldStatus.PRESERVED
    assert by_path["root_seed"].status is MigrationFieldStatus.PRESERVED
    with pytest.raises(FrozenInstanceError):
        ledger[0].path = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.semantic_equivalence = "EQUIVALENT"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("old_null_name", "null_params", "expected"),
    _MIGRATION_LEDGER_CASES,
)
def test_migration_ledger_freezes_every_expected_source_target_and_status(
    old_null_name: str,
    null_params: dict[str, int],
    expected: tuple[LedgerRecord, ...],
) -> None:
    result = migrate_plan_v1_to_v2(
        v1_request(null_name=old_null_name, null_params=null_params)
    )

    assert tuple(
        (
            entry.path,
            entry.source_present,
            entry.source_value,
            entry.target_present,
            entry.target_value,
            entry.status,
        )
        for entry in result.change_ledger
    ) == expected


@pytest.mark.parametrize(
    ("old_null_name", "null_params", "deleted_path"),
    _MIGRATION_LEDGER_DELETION_CASES,
)
def test_migration_factory_rejects_every_missing_expected_ledger_leaf(
    monkeypatch: pytest.MonkeyPatch,
    old_null_name: str,
    null_params: dict[str, int],
    deleted_path: str,
) -> None:
    original = migration._change_ledger

    def incomplete_ledger(
        source: JsonValue,
        target: JsonValue,
    ) -> tuple[PlanMigrationChange, ...]:
        return tuple(
            entry
            for entry in original(source, target)
            if entry.path != deleted_path
        )

    monkeypatch.setattr(migration, "_change_ledger", incomplete_ledger)

    with pytest.raises(ValueError, match=r"ledger|complete|expected"):
        migrate_plan_v1_to_v2(
            v1_request(null_name=old_null_name, null_params=null_params)
        )


@pytest.mark.parametrize(
    ("old_null_name", "null_params", "mutation"),
    tuple(
        (old_null_name, null_params, mutation)
        for old_null_name, null_params, _ in _MIGRATION_LEDGER_CASES
        for mutation in ("add", "replace")
    ),
)
def test_migration_factory_rejects_added_or_replaced_self_consistent_ledger_leaf(
    monkeypatch: pytest.MonkeyPatch,
    old_null_name: str,
    null_params: dict[str, int],
    mutation: str,
) -> None:
    original = migration._change_ledger

    def tampered_ledger(
        source: JsonValue,
        target: JsonValue,
    ) -> tuple[PlanMigrationChange, ...]:
        entries = original(source, target)
        if mutation == "add":
            extra = PlanMigrationChange(
                path="unexpected.extra",
                source_present=False,
                source_value=None,
                target_present=True,
                target_value=1,
                status=MigrationFieldStatus.ADDED,
            )
            return tuple(sorted((*entries, extra), key=lambda entry: entry.path))
        replacement = PlanMigrationChange(
            path="alpha",
            source_present=True,
            source_value=0.1,
            target_present=True,
            target_value=0.1,
            status=MigrationFieldStatus.PRESERVED,
        )
        return tuple(
            replacement if entry.path == "alpha" else entry
            for entry in entries
        )

    monkeypatch.setattr(migration, "_change_ledger", tampered_ledger)

    with pytest.raises(ValueError, match=r"ledger|complete|expected"):
        migrate_plan_v1_to_v2(
            v1_request(null_name=old_null_name, null_params=null_params)
        )


def test_migration_does_not_mutate_v1_or_implicitly_bind_execute_or_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = v1_request()
    snapshot = replace(source)

    def forbidden(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("migration must not bind, execute, sample, or calibrate")

    monkeypatch.setattr(LaggedPearsonAdapter, "bind", forbidden)
    monkeypatch.setattr(CircularShiftNullV2, "bind", forbidden)
    monkeypatch.setattr(randomness.ReplicateRandomSource, "__init__", forbidden)
    monkeypatch.setattr(randomness.ReplicateRandomSource, "randbelow", forbidden)
    monkeypatch.setattr(randomness.ReplicateRandomSource, "_next_raw64", forbidden)
    monkeypatch.setattr(
        randomness.ReplicateRandomSource,
        "seed_digest_sha256",
        property(forbidden),
    )
    monkeypatch.setattr(np.random, "PCG64", forbidden)
    capsule = randomness._RANDOM_CAPSULE_V2
    operation_codes = {
        tuple.__getitem__(capsule, index).__code__: name
        for index, name in enumerate(
            (
                "create_stream",
                "seed_digest",
                "randbelow",
                "seed_digest_for_replicate",
            )
        )
    }
    random_calls: list[str] = []

    def profile(frame: object, event: str, arg: object) -> None:
        del arg
        if event == "call" and frame.f_code in operation_codes:  # type: ignore[attr-defined]
            random_calls.append(operation_codes[frame.f_code])  # type: ignore[attr-defined]

    previous_profile = sys.getprofile()
    sys.setprofile(profile)
    try:
        result = migrate_plan_v1_to_v2(source)
    finally:
        sys.setprofile(previous_profile)

    assert source == snapshot
    assert random_calls == []
    assert result.v2_resolution.adapters.statistic.name == "lagged_pearson_v1"
    assert result.v2_resolution.adapters.null_model.name == "circular_shift_v2"


def test_migration_entrypoint_has_no_reachable_random_behavior_surface() -> None:
    capsule = randomness._RANDOM_CAPSULE_V2

    assert random_behavior_reachability_findings(
        migrate_plan_v1_to_v2,
        capsule_type=randomness._RandomCapsuleV2,
        canonical_capsule=capsule,
        public_random_source_type=randomness.ReplicateRandomSource,
        pcg64_type=np.random.PCG64,
        pcg64_raw_descriptor=np.random.PCG64.random_raw,
    ) == ()


@pytest.mark.parametrize(
    ("changes", "field"),
    (
        ({"root_seed": 2**64}, "root_seed"),
        ({"replicates": 1_000_001}, "replicates"),
    ),
)
def test_v2_invalid_v1_values_raise_typed_refusal_with_immutable_ledger(
    changes: dict[str, int], field: str
) -> None:
    source = v1_request(**changes)

    with pytest.raises(PlanMigrationRefusal) as caught:
        migrate_plan_v1_to_v2(source)

    refusal = caught.value
    assert refusal.reason == "V2_VALIDATION_REFUSAL"
    assert type(refusal.change_ledger) is tuple
    assert any(
        entry.path == field and entry.status is MigrationFieldStatus.REFUSED
        for entry in refusal.change_ledger
    )
    with pytest.raises(FrozenInstanceError):
        refusal.change_ledger[0].path = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    "null_name",
    ("custom_null_v1", "circular_shift_v3", "block_shuffle_v3"),
)
def test_unknown_or_custom_v1_null_has_no_implicit_migration(null_name: str) -> None:
    source = v1_request(null_name=null_name)

    with pytest.raises(PlanMigrationRefusal, match="supported v1 null") as caught:
        migrate_plan_v1_to_v2(source)
    assert any(
        entry.path == "null.name" and entry.status is MigrationFieldStatus.REFUSED
        for entry in caught.value.change_ledger
    )


def test_migration_requires_exact_v1_request() -> None:
    with pytest.raises(TypeError, match="exact PlanRequest"):
        migrate_plan_v1_to_v2(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="exact PlanRequest"):
        migrate_plan_v1_to_v2(  # type: ignore[arg-type]
            PlanRequestV2(
                candidates=(1,),
                statistic_name="lagged_pearson_v1",
                statistic_params={},
                selection_rule="max_upper",
                null_name="circular_shift_v2",
                null_params={"min_shift": 1},
                replicates=1,
                alpha=0.05,
                tie_tolerance=0.0,
                root_seed=0,
            )
        )


def test_migration_result_rejects_forged_resolution_and_identity_drift() -> None:
    result = migrate_plan_v1_to_v2(v1_request())
    forged = object.__new__(PlanResolutionV2)
    object.__setattr__(forged, "plan", result.v2_resolution.plan)
    object.__setattr__(forged, "adapters", result.v2_resolution.adapters)

    with pytest.raises(V2IntegrityError, match="resolver-owned"):
        construct_with_private_migration_seal(result, v2_resolution=forged)
    with pytest.raises(ValueError, match="v2 digest"):
        construct_with_private_migration_seal(
            result,
            v2_scientific_plan_sha256="0" * 64,
        )
    with pytest.raises(ValueError, match="v2 request"):
        construct_with_private_migration_seal(
            result,
            v2_request=PlanRequestV2(
                candidates=(1, 2),
                statistic_name="lagged_pearson_v1",
                statistic_params={},
                selection_rule="max_upper",
                null_name="circular_shift_v2",
                null_params={"min_shift": 1},
                replicates=9,
                alpha=0.05,
                tie_tolerance=1e-12,
                root_seed=17,
            ),
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"v1_scientific_plan_sha256": "bad"}, "v1_scientific_plan_sha256"),
        ({"v2_scientific_plan_sha256": "G" * 64}, "v2_scientific_plan_sha256"),
        ({"v2_request": object()}, "exact PlanRequestV2"),
        ({"v2_resolution": object()}, "exact PlanResolutionV2"),
        ({"change_ledger": []}, "change_ledger"),
        ({"change_ledger": ()}, "change_ledger"),
        ({"semantic_equivalence": "EQUIVALENT"}, "NON_EQUIVALENT"),
        ({"required_action": "NONE"}, "required_action"),
    ),
)
def test_migration_result_constructor_fails_closed(
    changes: dict[str, object], message: str
) -> None:
    result = migrate_plan_v1_to_v2(v1_request())
    with pytest.raises(ValueError, match=message):
        construct_with_private_migration_seal(result, **changes)


def test_migration_ledger_value_constructor_fails_closed() -> None:
    change = migrate_plan_v1_to_v2(v1_request()).change_ledger[0]
    invalid: tuple[dict[str, object], ...] = (
        {"path": ""},
        {"source_present": 1},
        {"status": "changed"},
        {"source_present": False, "source_value": 1},
        {"target_present": False, "target_value": 1},
    )
    for changes in invalid:
        with pytest.raises(ValueError):
            replace(change, **changes)


def test_migration_rejects_noncanonical_or_duplicate_ledger_paths() -> None:
    result = migrate_plan_v1_to_v2(v1_request())
    first = result.change_ledger[0]
    with pytest.raises(ValueError, match="unique and canonical"):
        construct_with_private_migration_seal(
            result,
            change_ledger=(first, first),
        )
    with pytest.raises(ValueError, match="unique and canonical"):
        construct_with_private_migration_seal(
            result,
            change_ledger=tuple(reversed(result.change_ledger)),
        )


def test_migration_hash_collision_guard_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(migration, "scientific_plan_v1_sha256", lambda plan: "0" * 64)
    monkeypatch.setattr(migration, "scientific_plan_v2_sha256", lambda plan: "0" * 64)
    with pytest.raises(RuntimeError, match="collide"):
        migrate_plan_v1_to_v2(v1_request())


def test_internal_ledger_helpers_reject_empty_root_and_freeze_tuple_values() -> None:
    with pytest.raises(ValueError, match="non-empty object"):
        migration._flatten_payload({})
    frozen = migration._freeze_ledger_value(
        cast(JsonValue, ({"nested": [1, 2]},)),
        name="test",
    )
    assert frozen == ({"nested": (1, 2)},)
    with pytest.raises(ValueError, match="finite exact JSON"):
        migration._freeze_ledger_value(float("nan"), name="test")
    with pytest.raises(ValueError, match="exact PlanMigrationChange"):
        migration._validate_change_entry(object(), allow_refused=False)


@pytest.mark.parametrize(
    ("source_present", "source_value", "target_present", "target_value", "status"),
    (
        (True, 1, True, 1, MigrationFieldStatus.CHANGED),
        (True, 1, True, 2, MigrationFieldStatus.PRESERVED),
        (False, None, True, 1, MigrationFieldStatus.REMOVED),
        (True, 1, False, None, MigrationFieldStatus.ADDED),
        (False, None, False, None, MigrationFieldStatus.ADDED),
        (False, None, True, 1, MigrationFieldStatus.REFUSED),
        (True, 1, False, None, MigrationFieldStatus.REFUSED),
    ),
)
def test_migration_change_status_must_be_derived_from_frozen_values(
    source_present: bool,
    source_value: JsonValue,
    target_present: bool,
    target_value: JsonValue,
    status: MigrationFieldStatus,
) -> None:
    with pytest.raises(ValueError, match=r"status|REFUSED|refusal"):
        PlanMigrationChange(
            path="ordinary.path",
            source_present=source_present,
            source_value=source_value,
            target_present=target_present,
            target_value=target_value,
            status=status,
        )


def test_refusal_change_requires_an_approved_refusal_field_and_shape() -> None:
    assert not migration._valid_refusal_value("ordinary.path", 1)
    with pytest.raises(ValueError, match=r"REFUSED|refusal"):
        PlanMigrationChange(
            path="ordinary.path",
            source_present=True,
            source_value=1,
            target_present=False,
            target_value=None,
            status=MigrationFieldStatus.REFUSED,
        )
    approved = PlanMigrationChange(
        path="root_seed",
        source_present=True,
        source_value=2**64,
        target_present=False,
        target_value=None,
        status=MigrationFieldStatus.REFUSED,
    )
    refusal = PlanMigrationRefusal("refused", change_ledger=(approved,))
    assert refusal.change_ledger == (approved,)
    with pytest.raises(ValueError, match="REFUSED"):
        PlanMigrationChange(
            path="null.name",
            source_present=True,
            source_value="circular_shift_v1",
            target_present=False,
            target_value=None,
            status=MigrationFieldStatus.REFUSED,
        )
    with pytest.raises(ValueError, match="refusal ledger"):
        PlanMigrationRefusal("invalid", change_ledger=())
    object.__setattr__(approved, "path", "ordinary.path")
    with pytest.raises(ValueError, match=r"REFUSED|refusal"):
        PlanMigrationRefusal("invalid", change_ledger=(approved,))


def test_successful_migration_has_private_factory_ownership() -> None:
    result = migrate_plan_v1_to_v2(v1_request())

    assert migration._require_resolver_owned_migration_v1_to_v2(result) is result
    with pytest.raises(ValueError, match=r"migration|factory|seal"):
        replace(result)

    copied = copy.copy(result)
    with pytest.raises(V2IntegrityError, match=r"migration.*owned|factory-owned"):
        migration._require_resolver_owned_migration_v1_to_v2(copied)
    with pytest.raises((pickle.PicklingError, TypeError)):
        pickle.dumps(result)

    naked = object.__new__(PlanMigrationV1ToV2)
    for field_name in (
        "v1_scientific_plan_sha256",
        "v2_request",
        "v2_resolution",
        "v2_scientific_plan_sha256",
        "change_ledger",
        "semantic_equivalence",
        "required_action",
    ):
        object.__setattr__(naked, field_name, getattr(result, field_name))
    with pytest.raises(V2IntegrityError, match=r"migration.*owned|factory-owned"):
        migration._require_resolver_owned_migration_v1_to_v2(naked)
    with pytest.raises(V2IntegrityError, match="exact factory-owned"):
        migration._require_resolver_owned_migration_v1_to_v2(object())


def test_private_migration_seal_still_rejects_subclasses() -> None:
    class MigrationSubclass(PlanMigrationV1ToV2):
        pass

    result = migrate_plan_v1_to_v2(v1_request())
    with pytest.raises(ValueError, match="exact PlanMigrationV1ToV2"):
        MigrationSubclass(
            v1_scientific_plan_sha256=result.v1_scientific_plan_sha256,
            v2_request=result.v2_request,
            v2_resolution=result.v2_resolution,
            v2_scientific_plan_sha256=result.v2_scientific_plan_sha256,
            change_ledger=result.change_ledger,
            semantic_equivalence="NON_EQUIVALENT",
            required_action="RECALIBRATION_REQUIRED",
            seal=migration._MIGRATION_SEAL,
        )


def test_migration_snapshot_preserves_typed_integrity_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = migrate_plan_v1_to_v2(v1_request())

    def typed_failure(value: PlanMigrationV1ToV2) -> migration._MigrationOwnershipSnapshot:
        del value
        raise V2IntegrityError("typed failure")

    monkeypatch.setattr(migration, "_capture_migration_snapshot", typed_failure)
    with pytest.raises(V2IntegrityError, match="typed failure"):
        migration._capture_migration_snapshot_fail_closed(result)


@pytest.mark.parametrize(
    ("field_name", "mutated"),
    (
        ("v1_scientific_plan_sha256", "0" * 64),
        ("v2_scientific_plan_sha256", "0" * 64),
        ("semantic_equivalence", "EQUIVALENT"),
        ("required_action", "NONE"),
    ),
)
def test_migration_ownership_freezes_digest_and_literal_fields(
    field_name: str, mutated: object
) -> None:
    result = migrate_plan_v1_to_v2(v1_request())
    object.__setattr__(result, field_name, mutated)

    with pytest.raises(V2IntegrityError, match=r"migration|digest|literal|identity"):
        migration._require_resolver_owned_migration_v1_to_v2(result)


@pytest.mark.parametrize("field_name", ("semantic_equivalence", "required_action"))
def test_migration_literals_reject_string_subclasses(field_name: str) -> None:
    class StringSubclass(str):
        pass

    result = migrate_plan_v1_to_v2(v1_request())
    object.__setattr__(result, field_name, StringSubclass(getattr(result, field_name)))
    with pytest.raises(V2IntegrityError, match=r"migration|identity|literal"):
        migration._require_resolver_owned_migration_v1_to_v2(result)


def test_migration_ownership_freezes_request_and_resolution_identities() -> None:
    request_replaced = migrate_plan_v1_to_v2(v1_request())
    object.__setattr__(
        request_replaced,
        "v2_request",
        PlanRequestV2(
            candidates=(1, 2, 3),
            statistic_name="lagged_pearson_v1",
            statistic_params={},
            selection_rule="max_upper",
            null_name="circular_shift_v2",
            null_params={"min_shift": 1},
            replicates=9,
            alpha=0.05,
            tie_tolerance=1e-12,
            root_seed=17,
        ),
    )
    with pytest.raises(V2IntegrityError, match=r"request.*identity"):
        migration._require_resolver_owned_migration_v1_to_v2(request_replaced)

    resolution_replaced = migrate_plan_v1_to_v2(v1_request())
    other_resolution = migrate_plan_v1_to_v2(v1_request()).v2_resolution
    object.__setattr__(resolution_replaced, "v2_resolution", other_resolution)
    with pytest.raises(V2IntegrityError, match=r"resolution.*identity"):
        migration._require_resolver_owned_migration_v1_to_v2(resolution_replaced)


def test_migration_ownership_freezes_request_fields_and_full_ledger_fingerprint() -> None:
    request_drift = migrate_plan_v1_to_v2(v1_request())
    object.__setattr__(request_drift.v2_request, "root_seed", 18)
    with pytest.raises(V2IntegrityError, match=r"request|identity|canonical"):
        migration._require_resolver_owned_migration_v1_to_v2(request_drift)

    for mutation in ("delete", "add", "change", "status", "refused"):
        result = migrate_plan_v1_to_v2(v1_request())
        ledger = result.change_ledger
        if mutation == "delete":
            object.__setattr__(result, "change_ledger", ledger[1:])
        elif mutation == "add":
            object.__setattr__(result, "change_ledger", (*ledger, ledger[0]))
        elif mutation == "change":
            object.__setattr__(ledger[0], "target_value", "tampered")
        elif mutation == "status":
            object.__setattr__(ledger[0], "status", MigrationFieldStatus.CHANGED)
        else:
            object.__setattr__(ledger[0], "status", MigrationFieldStatus.REFUSED)
        with pytest.raises(V2IntegrityError, match=r"ledger|REFUSED|fingerprint|status"):
            migration._require_resolver_owned_migration_v1_to_v2(result)


def test_success_ledger_rejects_refused_entry_even_if_fingerprint_is_rebuilt() -> None:
    result = migrate_plan_v1_to_v2(v1_request())
    refused = PlanMigrationChange(
        path="root_seed",
        source_present=True,
        source_value=2**64,
        target_present=False,
        target_value=None,
        status=MigrationFieldStatus.REFUSED,
    )
    with pytest.raises(ValueError, match="REFUSED"):
        construct_with_private_migration_seal(result, change_ledger=(refused,))


def test_migration_ownership_detects_registered_fingerprint_drift() -> None:
    result = migrate_plan_v1_to_v2(v1_request())
    registered = migration._MIGRATION_OWNED[result]
    object.__setattr__(registered, "ledger_sha256", "0" * 64)

    with pytest.raises(V2IntegrityError, match="fingerprint drifted"):
        migration._require_resolver_owned_migration_v1_to_v2(result)


def test_expected_migration_identity_rejects_unknown_v2_null() -> None:
    result = migrate_plan_v1_to_v2(v1_request())
    object.__setattr__(result.v2_request, "null_name", "unknown_null_v2")

    with pytest.raises(ValueError, match="no canonical v1 migration source"):
        migration._expected_success_migration_identity(
            result.v2_request,
            result.v2_resolution,
        )
