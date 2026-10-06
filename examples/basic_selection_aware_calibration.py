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
        source=np.asarray((0.0, 3.0, 1.0, 2.0, 4.0, 5.0), dtype=np.float64),
        target=np.asarray((0.0, 1.0, 3.0, 2.0, 5.0, 4.0), dtype=np.float64),
    )
    resolution = resolve_plan_v2(
        PlanRequestV2(
            candidates=(1, 2, 3),
            statistic_name="lagged_pearson_v1",
            statistic_params={},
            selection_rule="max_absolute",
            null_name="circular_shift_v2",
            null_params={"min_shift": 1},
            replicates=9,
            alpha=0.05,
            tie_tolerance=1e-12,
            root_seed=17,
        )
    )
    result = calibrate_selected_family(pair, resolution)
    verify_calibration_result(result, resolution)
    selection = result.observed_selection
    if selection is None:
        raise RuntimeError("verified complete result did not contain an observed selection")

    return {
        "schema": "selcal.example.basic-selection-aware-calibration.v1",
        "status": result.status.value,
        "scientific_plan_sha256": result.scientific_plan_sha256,
        "selected_candidate": selection.selected_candidate,
        "decision_statistic": round(selection.decision_statistic, 12),
        "planned_replicates": result.planned_replicates,
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
