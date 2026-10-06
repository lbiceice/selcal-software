"""Canonical scientific identity for resolved SelCal v2 plans."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from types import MemberDescriptorType
from typing import cast

from selcal.canonical import canonical_json_bytes
from selcal.contracts import JsonValue, SelectionRule
from selcal.contracts_v2 import ResolvedScientificPlanV2


def scientific_plan_v2_payload(plan: ResolvedScientificPlanV2, /) -> JsonValue:
    """Return the exact canonical payload committed by scientific-plan v2."""

    if type(plan) is not ResolvedScientificPlanV2:
        raise TypeError("plan must be an exact ResolvedScientificPlanV2")
    return {
        "schema": "selcal.scientific-plan.v2",
        "statistic": {
            "name": plan.statistic_name,
            "params": plan.statistic_params,
        },
        "candidates": list(plan.candidates),
        "selection": {
            "rule": SelectionRule(plan.selection_rule).value,
            "tie_tolerance": plan.tie_tolerance,
            "decision_contract": "family_max_with_canonical_tie_label_v2",
        },
        "null": {
            "name": plan.null_name,
            "params": plan.null_params,
        },
        "common_support": "max_candidate_lag_v1",
        "replicates": plan.replicates,
        "alpha": plan.alpha,
        "root_seed": plan.root_seed,
        "rng_contract": "sha256_framed_plan_rid_stream_to_pcg64_raw64_v1",
        "calibration_contract": "full_reselection_global_mc_plus_one_v2",
        "failure_contract": "exact_b_fail_closed_v2",
    }


def scientific_plan_v2_sha256(plan: ResolvedScientificPlanV2, /) -> str:
    """Hash only an exact resolver-created scientific-plan v2 value."""

    return hashlib.sha256(canonical_json_bytes(scientific_plan_v2_payload(plan))).hexdigest()


def _freeze_result_verifier_plan_identity_v2() -> (
    tuple[
        Callable[[object], bytes],
        Callable[[object], str],
    ]
):
    """Build verifier plan bytes/hash leaves without project-global lookups."""

    exact_type = type
    plan_type = ResolvedScientificPlanV2
    selection_rule_type = SelectionRule
    selection_max_upper = SelectionRule.MAX_UPPER
    selection_max_absolute = SelectionRule.MAX_ABSOLUTE
    mapping_type = Mapping
    string_type = str
    boolean_type = bool
    integer_type = int
    float_type = float
    list_type = list
    tuple_type = tuple
    isinstance_value = isinstance
    cast_value = cast
    member_descriptor_get = MemberDescriptorType.__get__
    isfinite = math.isfinite
    value_error_type = ValueError
    type_error_type = TypeError
    json_dumps = json.dumps
    sha256 = hashlib.sha256
    field_names = (
        "candidates",
        "statistic_name",
        "statistic_params",
        "selection_rule",
        "null_name",
        "null_params",
        "replicates",
        "alpha",
        "tie_tolerance",
        "root_seed",
    )
    slot_descriptors = tuple(
        (name, cast(MemberDescriptorType, plan_type.__dict__[name])) for name in field_names
    )

    def normalize_json(value: object) -> object:
        if value is None:
            return None
        if exact_type(value) is string_type:
            return value
        if exact_type(value) is boolean_type:
            return value
        if exact_type(value) is integer_type:
            return value
        if exact_type(value) is float_type:
            float_value = cast_value(float_type, value)
            if not isfinite(float_value):
                raise value_error_type("float values must be finite")
            return {"$float64": float_value.hex()}
        if isinstance_value(value, mapping_type):
            mapping = cast_value(mapping_type, value)
            normalized: dict[str, object] = {}
            for key, nested_value in mapping.items():
                if exact_type(key) is not string_type:
                    raise value_error_type("mapping keys must be strings")
                if key == "$float64":
                    raise value_error_type("$float64 is reserved for canonical float encoding")
                normalized[key] = normalize_json(nested_value)
            return normalized
        if exact_type(value) is list_type or exact_type(value) is tuple_type:
            sequence = cast_value(tuple_type, value)
            return [normalize_json(item) for item in sequence]
        raise value_error_type(f"unsupported JSON value type: {exact_type(value).__name__}")

    def plan_canonical_bytes(plan: object, /) -> bytes:
        if exact_type(plan) is not plan_type:
            raise type_error_type("plan must be an exact ResolvedScientificPlanV2")
        values: dict[str, object] = {}
        for name, descriptor in slot_descriptors:
            if descriptor is None:
                raise type_error_type(f"scientific plan slot descriptor is missing: {name}")
            if plan_type.__dict__.get(name) is not descriptor:
                raise type_error_type(f"scientific plan slot descriptor drifted: {name}")
            values[name] = member_descriptor_get(descriptor, plan, plan_type)
        rule = values["selection_rule"]
        candidates = cast_value(tuple_type, values["candidates"])
        exact_rule = cast_value(selection_rule_type, rule)
        if exact_type(exact_rule) is not selection_rule_type:
            raise type_error_type("selection_rule must be canonical")
        if exact_rule is selection_max_upper:
            rule_value = "max_upper"
        elif exact_rule is selection_max_absolute:
            rule_value = "max_absolute"
        else:
            raise type_error_type("selection_rule must be canonical")
        payload = {
            "schema": "selcal.scientific-plan.v2",
            "statistic": {
                "name": values["statistic_name"],
                "params": values["statistic_params"],
            },
            "candidates": list_type(candidates),
            "selection": {
                "rule": rule_value,
                "tie_tolerance": values["tie_tolerance"],
                "decision_contract": "family_max_with_canonical_tie_label_v2",
            },
            "null": {
                "name": values["null_name"],
                "params": values["null_params"],
            },
            "common_support": "max_candidate_lag_v1",
            "replicates": values["replicates"],
            "alpha": values["alpha"],
            "root_seed": values["root_seed"],
            "rng_contract": "sha256_framed_plan_rid_stream_to_pcg64_raw64_v1",
            "calibration_contract": "full_reselection_global_mc_plus_one_v2",
            "failure_contract": "exact_b_fail_closed_v2",
        }
        normalized = normalize_json(payload)
        return json_dumps(
            normalized,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    def plan_sha256(plan: object, /) -> str:
        return sha256(plan_canonical_bytes(plan)).hexdigest()

    return plan_canonical_bytes, plan_sha256


(
    _RESULT_VERIFIER_PLAN_CANONICAL_BYTES_V2,
    _RESULT_VERIFIER_PLAN_SHA256_V2,
) = _freeze_result_verifier_plan_identity_v2()
del _freeze_result_verifier_plan_identity_v2
