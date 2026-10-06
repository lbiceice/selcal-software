from __future__ import annotations

import pytest

from selcal.canonical import scientific_plan_sha256, scientific_plan_v1_sha256
from selcal.contracts import PlanRequest, ResolvedScientificPlan
from selcal.resolution import resolve_plan


def _legacy_plan(null_name: str, null_params: dict[str, int]) -> ResolvedScientificPlan:
    request = PlanRequest(
        candidates=(3, 1, 2),
        statistic_name="equal_width_binned_nette_v1",
        statistic_params={"bins": 3},
        selection_rule="max_upper",
        null_name=null_name,
        null_params=null_params,
        replicates=9,
        alpha=0.05,
        tie_tolerance=1e-12,
        root_seed=17,
        failure_policy="fail_closed_v1",
    )
    return resolve_plan(request).plan


@pytest.mark.parametrize(
    ("null_name", "null_params", "expected"),
    [
        (
            "circular_shift_v1",
            {"min_shift": 1},
            "41089d28130ba5e8f42b3205618c147d7dde3d7ebf32173c750d2c545f0db150",
        ),
        (
            "block_shuffle_v1",
            {"block_length": 2},
            "43430161e87194afef9cebb3c5b278e5784632690e7d7a8a8ccf5932870b9cdf",
        ),
    ],
)
def test_legacy_scientific_plan_v1_digest_is_literal(
    null_name: str,
    null_params: dict[str, int],
    expected: str,
) -> None:
    plan = _legacy_plan(null_name, null_params)

    assert scientific_plan_v1_sha256(plan) == expected
    assert scientific_plan_sha256(plan) == expected


def test_explicit_v1_hash_alias_preserves_strict_resolved_plan_identity() -> None:
    class ResolvedSubclass(ResolvedScientificPlan):
        pass

    request = PlanRequest(
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
    forged_subclass = object.__new__(ResolvedSubclass)

    for value in (request, forged_subclass, object()):
        with pytest.raises(TypeError, match="exact ResolvedScientificPlan"):
            scientific_plan_v1_sha256(value)  # type: ignore[arg-type]
