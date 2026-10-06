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


def run_example() -> dict[str, bool | float | int | str | None]:
    pair = SeriesPair(
        source=np.asarray((0, 1, 0, 1, 2, 1, 2, 0, 2, 1), dtype=np.float64),
        target=np.asarray((1, 0, 1, 2, 1, 0, 2, 1, 2, 0), dtype=np.float64),
    )
    resolution = resolve_plan_v2(
        PlanRequestV2(
            candidates=(1, 2),
            statistic_name="equal_width_binned_nette_v1",
            statistic_params={"bins": 3},
            selection_rule="max_absolute",
            null_name="block_shuffle_v2",
            null_params={"block_length": 2},
            replicates=9,
            alpha=0.05,
            tie_tolerance=1e-12,
            root_seed=23,
        )
    )
    result = calibrate_selected_family(pair, resolution)
    verify_calibration_result(result, resolution)
    selection = result.observed_selection
    if selection is None:
        raise RuntimeError("verified complete result did not contain an observed selection")

    return {
        "schema": "selcal.example.binned-nette-block-shuffle.v1",
        "status": result.status.value,
        "scientific_plan_sha256": result.scientific_plan_sha256,
        "selected_candidate": selection.selected_candidate,
        "decision_statistic": round(selection.decision_statistic, 12),
        "planned_replicates": result.planned_replicates,
        "replicate_count": len(result.replicates),
        "exceedance_count": result.exceedance_count,
        "failure_count": result.failure_count,
        "p_value": result.p_value,
        "reject_null": result.reject_null,
    }


def main() -> int:
    print(json.dumps(run_example(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
