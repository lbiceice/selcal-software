"""Canonical byte encodings and scientific identity hashes."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from typing import cast

import numpy as np

from selcal.contracts import (
    JsonValue,
    ResolvedScientificPlan,
    SelectionRule,
    SeriesPair,
    canonical_candidates,
)


def _normalize_json(value: object) -> JsonValue:
    if isinstance(value, np.generic):
        raise ValueError("unsupported JSON value type: NumPy scalar")
    if value is None:
        return None
    if type(value) is str:
        return value
    if type(value) is bool:
        return value
    if type(value) is int:
        return value
    if type(value) is float:
        float_value = value
        if not math.isfinite(float_value):
            raise ValueError("float values must be finite")
        return {"$float64": float_value.hex()}
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        normalized: dict[str, JsonValue] = {}
        for key, nested_value in mapping.items():
            if type(key) is not str:
                raise ValueError("mapping keys must be strings")
            if key == "$float64":
                raise ValueError("$float64 is reserved for canonical float encoding")
            normalized[key] = _normalize_json(nested_value)
        return normalized
    if type(value) is list or type(value) is tuple:
        return [_normalize_json(item) for item in value]
    raise ValueError(f"unsupported JSON value type: {type(value).__name__}")


def canonical_json_bytes(value: JsonValue) -> bytes:
    normalized = _normalize_json(value)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def raw_input_sha256(payload: bytes) -> str:
    if type(payload) is not bytes:
        raise TypeError("payload must be actual bytes")
    return hashlib.sha256(payload).hexdigest()


def semantic_input_sha256(pair: SeriesPair, candidates: tuple[int, ...]) -> str:
    candidates = canonical_candidates(candidates)
    digest = hashlib.sha256()
    digest.update(b"selcal.semantic-input.v1\0")
    digest.update(
        canonical_json_bytes(
            {
                "roles": ["source", "target"],
                "shape": list(pair.source.shape),
                "candidates": list(candidates),
            }
        )
    )
    digest.update(pair.source.astype("<f8", copy=False).tobytes(order="C"))
    digest.update(pair.target.astype("<f8", copy=False).tobytes(order="C"))
    return digest.hexdigest()


def scientific_plan_v1_sha256(plan: ResolvedScientificPlan) -> str:
    """Hash the exact historical scientific-plan v1 payload."""

    if type(plan) is not ResolvedScientificPlan:
        raise TypeError("plan must be an exact ResolvedScientificPlan")
    payload: JsonValue = {
        "schema": "selcal.scientific-plan.v1",
        "candidates": list(canonical_candidates(plan.candidates)),
        "statistic": {
            "name": plan.statistic_name,
            "params": plan.statistic_params,
        },
        "selection": {
            "rule": SelectionRule(plan.selection_rule).value,
            "tie_tolerance": plan.tie_tolerance,
        },
        "null": {
            "name": plan.null_name,
            "params": plan.null_params,
        },
        "replicates": plan.replicates,
        "alpha": plan.alpha,
        "root_seed": plan.root_seed,
        "failure_policy": plan.failure_policy,
        "rng_contract": "sha256-to-pcg64-v1",
        "common_support": "max_candidate_lag_v1",
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def scientific_plan_sha256(plan: ResolvedScientificPlan) -> str:
    """Compatibility entry point for the historical scientific-plan v1 hash."""

    return scientific_plan_v1_sha256(plan)


def _actual_string(value: object, *, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be an actual string")
    return value


def _implementation_component(value: object, *, name: str) -> str:
    component = _actual_string(value, name=name)
    if not component:
        raise ValueError(f"{name} must be nonempty")
    if component != component.strip():
        raise ValueError(f"{name} must not have leading or trailing whitespace")
    return component


def _sha256_component(value: object, *, name: str) -> str:
    component = _actual_string(value, name=name)
    if re.fullmatch(r"[0-9a-f]{64}", component) is None:
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256 hex string")
    return component


def implementation_sha256(
    package_version: str,
    source_revision: str,
    backend_identity: str,
) -> str:
    payload: JsonValue = {
        "schema": "selcal.implementation.v1",
        "package_version": _implementation_component(
            package_version, name="package_version"
        ),
        "source_revision": _implementation_component(
            source_revision, name="source_revision"
        ),
        "backend_identity": _implementation_component(
            backend_identity, name="backend_identity"
        ),
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def run_id(
    semantic_hash: str,
    plan_hash: str,
    implementation_hash: str,
) -> str:
    payload: JsonValue = {
        "semantic_input_sha256": _sha256_component(
            semantic_hash, name="semantic_input_sha256"
        ),
        "scientific_plan_sha256": _sha256_component(
            plan_hash, name="scientific_plan_sha256"
        ),
        "implementation_sha256": _sha256_component(
            implementation_hash, name="implementation_sha256"
        ),
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
