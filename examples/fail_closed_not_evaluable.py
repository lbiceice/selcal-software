from __future__ import annotations

import json

import numpy as np

from selcal import (
    PlanRequestV2,
    calibrate_selected_family,
    resolve_plan_v2,
    verify_calibration_result,
)
from selcal.contracts import SeriesPair


def run_example() -> dict[str, bool | float | int | str | list[str] | None]:
    pair = SeriesPair(
        source=np.asarray((0.0, 1.0, 2.0, 3.0, 4.0), dtype=np.float64),
        target=np.asarray((4.0, 3.0, 2.0, 1.0, 0.0), dtype=np.float64),
    )
    resolution = resolve_plan_v2(
        PlanRequestV2(
            candidates=(1, 3),
            statistic_name="lagged_pearson_v1",
            statistic_params={},
            selection_rule="max_upper",
            # Five samples form one block, so block shuffling has nothing to permute. (A
            # restricted circular shift once served here; it is now refused before running.)
            null_name="block_shuffle_v2",
            null_params={"block_length": 5},
            replicates=7,
            alpha=0.05,
            tie_tolerance=1e-12,
            root_seed=17,
        )
    )
    result = calibrate_selected_family(pair, resolution)
    verify_calibration_result(result, resolution)
    failure_stage = result.failure_stage
    if failure_stage is None:
        raise RuntimeError("verified not-evaluable result did not contain a failure stage")

    return {
        "schema": "selcal.example.fail-closed-not-evaluable.v1",
        "status": result.status.value,
        "failure_stage": failure_stage.value,
        "scientific_plan_sha256": result.scientific_plan_sha256,
        "planned_replicates": result.planned_replicates,
        "observed_result_count": len(result.observed_results),
        "replicate_count": len(result.replicates),
        "p_value": result.p_value,
        "reject_null": result.reject_null,
        "diagnostics": list(result.diagnostics),
    }


def main() -> int:
    print(json.dumps(run_example(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
