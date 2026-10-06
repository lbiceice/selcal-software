"""Public analysis entry: one admission rule before the verified calibration kernel.

R11 Windows return (2026-10-03, R11-01): the file workflow refused circular shifts with
min_shift > 1, but ``selcal.calibrate_selected_family`` still returned a complete result for
them. This module is that public name. The kernel ``selcal.calibration_v2`` is unchanged and
stays unrestricted for record replay and internal tests; it is not the public analysis entry.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from selcal.calibration_v2 import calibrate_selected_family as _kernel
from selcal.contracts import JsonValue, SeriesPair
from selcal.contracts_v2 import CalibrationResult, SelCalV2Error
from selcal.resolution_v2 import PlanResolutionV2

__all__ = ["InvalidNullForInferenceError", "calibrate_selected_family", "null_supports_inference"]


class InvalidNullForInferenceError(SelCalV2Error):
    """A plan whose null has no validity guarantee was offered for a new analysis."""

    code = "invalid_null_for_inference"


def null_supports_inference(null_name: str, null_params: Mapping[str, JsonValue]) -> bool:
    """Whether SelCal supports this null for new inference (one rule for every entry point).

    Circular shifts are supported only as the complete cyclic group (min_shift = 1). For
    1 < min_shift < n/2 the shift set contains two consecutive shifts and is not closed, so it
    is not a group; other values are degenerate. Restricted shifts are therefore unsupported as
    a policy, not because every such set fails to be a group. Nulls other than circular shifts
    are not judged here (workflow.null_validity reports NOT_ASSESSED for them).
    """
    if null_name == "circular_shift_v2":
        return null_params.get("min_shift") == 1
    return True


def _freeze_entry(
    kernel: Callable[[SeriesPair, PlanResolutionV2], CalibrationResult],
    supports: Callable[[str, Mapping[str, JsonValue]], bool],
    error: type[InvalidNullForInferenceError],
    /,
) -> Callable[[SeriesPair, PlanResolutionV2], CalibrationResult]:
    def calibrate_selected_family(
        pair: SeriesPair,
        resolution: PlanResolutionV2,
        /,
    ) -> CalibrationResult:
        """Run a new selected-family analysis: admission check, then the verified kernel.

        Raises InvalidNullForInferenceError (``code == "invalid_null_for_inference"``) before
        any computation when the null has no validity guarantee, such as circular shifts with
        min_shift > 1. Otherwise identical to the kernel: exact-B calibration, verified result.
        """
        plan = getattr(resolution, "plan", None)
        if plan is not None and not supports(plan.null_name, plan.null_params):
            raise error(
                "this null is not supported for inference: circular shifts require min_shift = 1"
            )
        return kernel(pair, resolution)

    calibrate_selected_family.__module__ = __name__
    calibrate_selected_family.__qualname__ = "calibrate_selected_family"
    return calibrate_selected_family


calibrate_selected_family = _freeze_entry(
    _kernel, null_supports_inference, InvalidNullForInferenceError
)
del _freeze_entry
