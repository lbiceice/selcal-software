from __future__ import annotations

from dataclasses import replace

import pytest

from selcal.canonical_v2 import scientific_plan_v2_payload, scientific_plan_v2_sha256
from selcal.contracts import PlanRequest, ResolvedScientificPlan
from selcal.contracts_v2 import PlanRequestV2, ResolvedScientificPlanV2
from selcal.resolution_v2 import resolve_plan_v2

EXPECTED_BYTES = (
    b'{"alpha":{"$float64":"0x1.999999999999ap-5"},'
    b'"calibration_contract":"full_reselection_global_mc_plus_one_v2",'
    b'"candidates":[1,2,3],"common_support":"max_candidate_lag_v1",'
    b'"failure_contract":"exact_b_fail_closed_v2",'
    b'"null":{"name":"circular_shift_v2","params":{"min_shift":1}},'
    b'"replicates":9,'
    b'"rng_contract":"sha256_framed_plan_rid_stream_to_pcg64_raw64_v1",'
    b'"root_seed":17,"schema":"selcal.scientific-plan.v2",'
    b'"selection":{"decision_contract":"family_max_with_canonical_tie_label_v2",'
    b'"rule":"max_upper",'
    b'"tie_tolerance":{"$float64":"0x1.19799812dea11p-40"}},'
    b'"statistic":{"name":"lagged_pearson_v1","params":{}}}'
)
EXPECTED_DIGEST = "dc7462598231451b36a94e2586037aa921a8159b2eabebb5fe5c81dcafa48931"


def normative_request() -> PlanRequestV2:
    return PlanRequestV2(
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
    )


def test_normative_v2_payload_has_exact_fields_bytes_and_digest() -> None:
    resolution = resolve_plan_v2(normative_request())
    payload = scientific_plan_v2_payload(resolution.plan)

    assert type(resolution.plan) is ResolvedScientificPlanV2
    assert set(payload) == {
        "schema",
        "statistic",
        "candidates",
        "selection",
        "null",
        "common_support",
        "replicates",
        "alpha",
        "root_seed",
        "rng_contract",
        "calibration_contract",
        "failure_contract",
    }
    assert set(payload["statistic"]) == {"name", "params"}  # type: ignore[arg-type]
    assert set(payload["selection"]) == {  # type: ignore[arg-type]
        "rule",
        "tie_tolerance",
        "decision_contract",
    }
    assert set(payload["null"]) == {"name", "params"}  # type: ignore[arg-type]
    from selcal.canonical import canonical_json_bytes

    assert canonical_json_bytes(payload) == EXPECTED_BYTES
    assert scientific_plan_v2_sha256(resolution.plan) == EXPECTED_DIGEST


def test_v2_hash_is_order_independent_but_commits_every_scientific_field() -> None:
    first = resolve_plan_v2(normative_request()).plan
    reordered = resolve_plan_v2(
        PlanRequestV2(
            candidates=(3, 1, 2),
            statistic_name="lagged_pearson_v1",
            statistic_params={},
            selection_rule="max_upper",
            null_name="circular_shift_v2",
            null_params={"min_shift": 1},
            replicates=9,
            alpha=0.05,
            tie_tolerance=1e-12,
            root_seed=17,
        )
    ).plan
    assert scientific_plan_v2_sha256(first) == scientific_plan_v2_sha256(reordered)

    changes: tuple[dict[str, object], ...] = (
        {"candidates": (1, 2)},
        {"selection_rule": "max_absolute"},
        {"null_params": {"min_shift": 2}},
        {"replicates": 10},
        {"alpha": 0.1},
        {"tie_tolerance": 0.0},
        {"root_seed": 18},
    )
    for change in changes:
        values = {
            "candidates": (1, 2, 3),
            "statistic_name": "lagged_pearson_v1",
            "statistic_params": {},
            "selection_rule": "max_upper",
            "null_name": "circular_shift_v2",
            "null_params": {"min_shift": 1},
            "replicates": 9,
            "alpha": 0.05,
            "tie_tolerance": 1e-12,
            "root_seed": 17,
        }
        values.update(change)
        changed = resolve_plan_v2(PlanRequestV2(**values)).plan  # type: ignore[arg-type]
        assert scientific_plan_v2_sha256(changed) != EXPECTED_DIGEST


def test_v2_hash_commits_registered_statistic_and_null_identity_fields() -> None:
    alternatives = (
        PlanRequestV2(
            candidates=(1, 2, 3),
            statistic_name="equal_width_binned_nette_v1",
            statistic_params={"bins": 3},
            selection_rule="max_upper",
            null_name="circular_shift_v2",
            null_params={"min_shift": 1},
            replicates=9,
            alpha=0.05,
            tie_tolerance=1e-12,
            root_seed=17,
        ),
        PlanRequestV2(
            candidates=(1, 2, 3),
            statistic_name="equal_width_binned_nette_v1",
            statistic_params={"bins": 4},
            selection_rule="max_upper",
            null_name="circular_shift_v2",
            null_params={"min_shift": 1},
            replicates=9,
            alpha=0.05,
            tie_tolerance=1e-12,
            root_seed=17,
        ),
        PlanRequestV2(
            candidates=(1, 2, 3),
            statistic_name="lagged_pearson_v1",
            statistic_params={},
            selection_rule="max_upper",
            null_name="block_shuffle_v2",
            null_params={"block_length": 2},
            replicates=9,
            alpha=0.05,
            tie_tolerance=1e-12,
            root_seed=17,
        ),
    )

    digests = tuple(
        scientific_plan_v2_sha256(resolve_plan_v2(request).plan)
        for request in alternatives
    )
    assert all(digest != EXPECTED_DIGEST for digest in digests)
    assert len(set(digests)) == len(digests)


def test_only_exact_resolved_v2_plan_crosses_payload_and_hash_gate() -> None:
    resolved = resolve_plan_v2(normative_request()).plan
    v1_request = PlanRequest(
        candidates=(1,),
        statistic_name="lagged_pearson_v1",
        statistic_params={},
        selection_rule="max_upper",
        null_name="circular_shift_v1",
        null_params={"min_shift": 1},
        replicates=1,
        alpha=0.05,
        tie_tolerance=0.0,
        root_seed=0,
        failure_policy="fail_closed_v1",
    )

    class ResolvedV2Subclass(ResolvedScientificPlanV2):
        pass

    class ResolvedV1Subclass(ResolvedScientificPlan):
        pass

    for value in (
        normative_request(),
        v1_request,
        object(),
        object.__new__(ResolvedV2Subclass),
        object.__new__(ResolvedV1Subclass),
    ):
        with pytest.raises(TypeError, match="exact ResolvedScientificPlanV2"):
            scientific_plan_v2_payload(value)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="exact ResolvedScientificPlanV2"):
            scientific_plan_v2_sha256(value)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match=r"resolver|resolution"):
        replace(resolved)


def test_resolved_v2_plan_equality_is_exact_and_field_sensitive() -> None:
    first = resolve_plan_v2(normative_request()).plan
    second = resolve_plan_v2(normative_request()).plan
    changed = resolve_plan_v2(
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
            root_seed=18,
        )
    ).plan

    assert first == second
    assert first != changed
    assert first != normative_request()
