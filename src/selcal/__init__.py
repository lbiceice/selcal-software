"""SelCal pre-release scientific-plan and statistic core."""
from selcal.calibration_v2 import verify_calibration_result
from selcal.contracts import PlanRequest, ResolvedScientificPlan
from selcal.contracts_v2 import PlanRequestV2, ResolvedScientificPlanV2
from selcal.inference import InvalidNullForInferenceError, calibrate_selected_family
from selcal.migration_v1_to_v2 import PlanMigrationV1ToV2, migrate_plan_v1_to_v2
from selcal.resolution import PlanResolution, resolve_plan
from selcal.resolution_v2 import PlanResolutionV2, resolve_plan_v2

__version__ = "0.1.0"
__all__ = [
    "InvalidNullForInferenceError",
    "PlanMigrationV1ToV2",
    "PlanRequest",
    "PlanRequestV2",
    "PlanResolution",
    "PlanResolutionV2",
    "ResolvedScientificPlan",
    "ResolvedScientificPlanV2",
    "calibrate_selected_family",
    "migrate_plan_v1_to_v2",
    "resolve_plan",
    "resolve_plan_v2",
    "verify_calibration_result",
]
