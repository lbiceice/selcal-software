from __future__ import annotations

import gc
import hashlib
import inspect
import json
import math
import os
import pickle
import platform
import subprocess
import sys
import tracemalloc
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field, fields
from enum import StrEnum
from pathlib import Path
from types import CellType, FrameType, FunctionType

import numpy as np
import pytest

import selcal
import selcal.calibration_v2 as calibration_v2
import selcal.contracts_v2 as contracts_v2
from selcal.contracts import (
    ReplicateStatus,
    RunStatus,
    SelectionResult,
    StatisticResult,
    Validity,
)
from selcal.contracts_v2 import (
    BlockShuffleStateV2,
    CalibrationResult,
    CircularShiftStateV2,
    NullTransformToken,
    PlanRequestV2,
    ReplicateFailureStage,
    ReplicateOutcome,
    RunFailureStage,
    V2IntegrityError,
)
from selcal.resolution_v2 import PlanResolutionV2, resolve_plan_v2

_SEMANTIC_INPUT_SHA256 = "db4106a281134389e084ce9b90d6ba1ddf29e6cf4a6d3e698d85028c1533b362"
_SCIENTIFIC_PLAN_SHA256 = "fafe09523db114bf7926ad0f242795affe34e098c001b64286a9e234e375fd0f"
_SEED_DIGEST_0 = "9666fb826c0f31c2bfd2820097b5a13f18002e8b0d2e6761ec2bf6b62f99f656"
_SEED_DIGEST_1 = "3b4b13085cd1722e80d7d036ae92e07b7ad4d2758e5fed7e94e8d5dd9f0c37f3"
_SEED_DIGEST_2 = "ac30fe17506254372666a5675fe433502ee77bcb2f70d552b6a680edbb41b66b"
_ROW17_PLAN_SHA256 = "a83bd08f667d8e8491b056a828a26283e393dbdffc3cca9078372aad7b36b2b8"
_ROW17_BOUND_OWNER_SHA256 = (
    "2185de667ebf828767625e23dd07438e5247080341227b66d36878463aa99858"
)
_ROW17_SEED_DIGESTS = (
    "da2652d3ff63ae540b854110ef0e9a0095a67c5988892d012e6c102a207d01d8",
    "7eab0499410d3585e221ad8f198907afdb8ce7518a9b550661bc1fef1854dd8b",
    "13dd61231777e6a4c5007a592be40c581c0299eda21a6b3278e38e01f8618a24",
)
# Backend identity is an environment-bound verifier input, not a scientific expected value.
_BACKEND_IDENTITY = (
    f"selcal.lagged_pearson.v1|numpy={np.__version__}|units=pearson_correlation"
)
_PREPROCESSING_IDENTITY = (
    "no_hidden_transform|formula_two_pass_centered_after_power_of_two_scaling"
    "|correctly_rounded_sums|common_support_max_lag"
)


def _literal_resolution() -> PlanResolutionV2:
    return resolve_plan_v2(
        PlanRequestV2(
            candidates=(1, 2),
            statistic_name="lagged_pearson_v1",
            statistic_params={},
            selection_rule="max_upper",
            null_name="circular_shift_v2",
            null_params={"min_shift": 1},
            replicates=3,
            alpha=0.75,
            tie_tolerance=1e-12,
            root_seed=17,
        )
    )


def _literal_identity_token() -> NullTransformToken:
    state = object.__new__(CircularShiftStateV2)
    object.__setattr__(state, "schema", "selcal.circular-shift-state.v2")
    object.__setattr__(state, "shift", 0)

    token = object.__new__(NullTransformToken)
    object.__setattr__(token, "schema", "selcal.null-transform-token.v2")
    object.__setattr__(token, "null_name", "circular_shift_v2")
    object.__setattr__(
        token,
        "null_parameter_sha256",
        "a2e6b608e05886b0c06388750e201c922c21d08e4ca6faf7668742798895dda3",
    )
    object.__setattr__(
        token,
        "semantic_input_sha256",
        "db4106a281134389e084ce9b90d6ba1ddf29e6cf4a6d3e698d85028c1533b362",
    )
    object.__setattr__(
        token,
        "scientific_plan_sha256",
        "fafe09523db114bf7926ad0f242795affe34e098c001b64286a9e234e375fd0f",
    )
    object.__setattr__(
        token,
        "bound_null_owner_sha256",
        "6863e569187ddc8983aa51f622ce96d6d644842ae76a116f6c31c9e5338befba",
    )
    object.__setattr__(token, "is_identity", True)
    object.__setattr__(token, "state", state)
    return token


def _literal_replicate_execution_token(
    *,
    shift: int,
    is_identity: bool,
) -> NullTransformToken:
    state = object.__new__(CircularShiftStateV2)
    object.__setattr__(state, "schema", "selcal.circular-shift-state.v2")
    object.__setattr__(state, "shift", shift)

    token = object.__new__(NullTransformToken)
    object.__setattr__(token, "schema", "selcal.null-transform-token.v2")
    object.__setattr__(token, "null_name", "circular_shift_v2")
    object.__setattr__(
        token,
        "null_parameter_sha256",
        "a2e6b608e05886b0c06388750e201c922c21d08e4ca6faf7668742798895dda3",
    )
    object.__setattr__(token, "semantic_input_sha256", _SEMANTIC_INPUT_SHA256)
    object.__setattr__(token, "scientific_plan_sha256", _SCIENTIFIC_PLAN_SHA256)
    object.__setattr__(
        token,
        "bound_null_owner_sha256",
        "6863e569187ddc8983aa51f622ce96d6d644842ae76a116f6c31c9e5338befba",
    )
    object.__setattr__(token, "is_identity", is_identity)
    object.__setattr__(token, "state", state)
    return token


def _literal_statistic_result(
    *,
    candidate_id: int,
    estimate: float | None,
    selection_score: float | None,
    validity: Validity,
    diagnostics: tuple[str, ...] = (),
) -> StatisticResult:
    result = object.__new__(StatisticResult)
    object.__setattr__(result, "candidate_id", candidate_id)
    object.__setattr__(result, "estimate", estimate)
    object.__setattr__(result, "selection_score", selection_score)
    object.__setattr__(result, "support_n", 4)
    object.__setattr__(result, "validity", validity)
    object.__setattr__(result, "diagnostics", diagnostics)
    object.__setattr__(result, "backend_identity", _BACKEND_IDENTITY)
    object.__setattr__(result, "preprocessing_identity", _PREPROCESSING_IDENTITY)
    return result


def _literal_selection(
    *,
    selected_candidate: int,
    selected_index: int,
    decision_statistic: float,
    tied_candidates: tuple[int, ...],
) -> SelectionResult:
    selection = object.__new__(SelectionResult)
    object.__setattr__(selection, "selected_candidate", selected_candidate)
    object.__setattr__(selection, "selected_index", selected_index)
    object.__setattr__(selection, "decision_statistic", decision_statistic)
    object.__setattr__(selection, "tied_candidates", tied_candidates)
    return selection


def _literal_replicate(
    *,
    replicate_id: int,
    seed_digest_sha256: str,
    status: ReplicateStatus,
    failure_stage: ReplicateFailureStage | None,
    transform_token: NullTransformToken,
    statistic_results: tuple[StatisticResult, ...],
    selection: SelectionResult | None,
    diagnostics: tuple[str, ...] = (),
) -> ReplicateOutcome:
    outcome = object.__new__(ReplicateOutcome)
    object.__setattr__(outcome, "replicate_id", replicate_id)
    object.__setattr__(outcome, "seed_digest_sha256", seed_digest_sha256)
    object.__setattr__(outcome, "status", status)
    object.__setattr__(outcome, "failure_stage", failure_stage)
    object.__setattr__(outcome, "transform_token", transform_token)
    object.__setattr__(outcome, "statistic_results", statistic_results)
    object.__setattr__(outcome, "selection", selection)
    object.__setattr__(outcome, "diagnostics", diagnostics)
    return outcome


def _literal_calibration_result(
    *,
    status: RunStatus,
    failure_stage: RunFailureStage | None,
    observed_results: tuple[StatisticResult, ...],
    observed_selection: SelectionResult,
    replicates: tuple[ReplicateOutcome, ...],
    exceedance_count: int,
    failure_count: int,
    p_value: float | None,
    bound_low: float | None,
    bound_high: float | None,
    reject_null: bool | None,
    diagnostics: tuple[str, ...] = (),
) -> CalibrationResult:
    result = object.__new__(CalibrationResult)
    object.__setattr__(result, "status", status)
    object.__setattr__(result, "failure_stage", failure_stage)
    object.__setattr__(result, "semantic_input_sha256", _SEMANTIC_INPUT_SHA256)
    object.__setattr__(result, "scientific_plan_sha256", _SCIENTIFIC_PLAN_SHA256)
    object.__setattr__(result, "planned_replicates", 3)
    object.__setattr__(result, "alpha", 0.75)
    object.__setattr__(result, "observed_results", observed_results)
    object.__setattr__(result, "observed_selection", observed_selection)
    object.__setattr__(result, "replicates", replicates)
    object.__setattr__(result, "exceedance_count", exceedance_count)
    object.__setattr__(result, "failure_count", failure_count)
    object.__setattr__(result, "p_value", p_value)
    object.__setattr__(result, "exceedance_bound_low", bound_low)
    object.__setattr__(result, "exceedance_bound_high", bound_high)
    object.__setattr__(result, "reject_null", reject_null)
    object.__setattr__(result, "diagnostics", diagnostics)
    return result


def _literal_zero_sign_characterization_case(
    *,
    estimate_zero: float,
    score_zero: float,
    decision_zero: float,
) -> tuple[CalibrationResult, PlanResolutionV2, StatisticResult, SelectionResult]:
    resolution = _literal_resolution()
    token = _literal_identity_token()
    observed_results = (
        _literal_statistic_result(
            candidate_id=1,
            estimate=3.0,
            selection_score=3.0,
            validity=Validity.VALID,
        ),
        _literal_statistic_result(
            candidate_id=2,
            estimate=1.0,
            selection_score=1.0,
            validity=Validity.VALID,
        ),
    )
    observed_selection = _literal_selection(
        selected_candidate=1,
        selected_index=0,
        decision_statistic=3.0,
        tied_candidates=(1,),
    )
    exceedance_results = (
        _literal_statistic_result(
            candidate_id=1,
            estimate=4.0,
            selection_score=4.0,
            validity=Validity.VALID,
        ),
        _literal_statistic_result(
            candidate_id=2,
            estimate=2.0,
            selection_score=2.0,
            validity=Validity.VALID,
        ),
    )
    exceedance_selection = _literal_selection(
        selected_candidate=1,
        selected_index=0,
        decision_statistic=4.0,
        tied_candidates=(1,),
    )
    boundary_results = (
        _literal_statistic_result(
            candidate_id=1,
            estimate=0.0,
            selection_score=0.0,
            validity=Validity.VALID,
        ),
        _literal_statistic_result(
            candidate_id=2,
            estimate=1e-12,
            selection_score=1e-12,
            validity=Validity.VALID,
        ),
    )
    boundary_selection = _literal_selection(
        selected_candidate=1,
        selected_index=0,
        decision_statistic=1e-12,
        tied_candidates=(1, 2),
    )
    zero_result = _literal_statistic_result(
        candidate_id=1,
        estimate=estimate_zero,
        selection_score=score_zero,
        validity=Validity.VALID,
    )
    zero_results = (
        zero_result,
        _literal_statistic_result(
            candidate_id=2,
            estimate=-1.0,
            selection_score=-1.0,
            validity=Validity.VALID,
        ),
    )
    zero_selection = _literal_selection(
        selected_candidate=1,
        selected_index=0,
        decision_statistic=decision_zero,
        tied_candidates=(1,),
    )
    result = _literal_calibration_result(
        status=RunStatus.COMPLETE,
        failure_stage=None,
        observed_results=observed_results,
        observed_selection=observed_selection,
        replicates=(
            _literal_replicate(
                replicate_id=0,
                seed_digest_sha256=_SEED_DIGEST_0,
                status=ReplicateStatus.COMPLETE,
                failure_stage=None,
                transform_token=token,
                statistic_results=exceedance_results,
                selection=exceedance_selection,
            ),
            _literal_replicate(
                replicate_id=1,
                seed_digest_sha256=_SEED_DIGEST_1,
                status=ReplicateStatus.COMPLETE,
                failure_stage=None,
                transform_token=token,
                statistic_results=boundary_results,
                selection=boundary_selection,
            ),
            _literal_replicate(
                replicate_id=2,
                seed_digest_sha256=_SEED_DIGEST_2,
                status=ReplicateStatus.COMPLETE,
                failure_stage=None,
                transform_token=token,
                statistic_results=zero_results,
                selection=zero_selection,
            ),
        ),
        exceedance_count=1,
        failure_count=0,
        p_value=0.5,
        bound_low=None,
        bound_high=None,
        reject_null=True,
    )
    return result, resolution, zero_result, zero_selection


def _row17_literal_resolution() -> PlanResolutionV2:
    return resolve_plan_v2(
        PlanRequestV2(
            candidates=(1, 2),
            statistic_name="lagged_pearson_v1",
            statistic_params={},
            selection_rule="max_upper",
            null_name="circular_shift_v2",
            null_params={"min_shift": 1},
            replicates=3,
            alpha=0.05,
            tie_tolerance=1e-12,
            root_seed=17,
        )
    )


def _row17_literal_identity_token() -> NullTransformToken:
    state = object.__new__(CircularShiftStateV2)
    object.__setattr__(state, "schema", "selcal.circular-shift-state.v2")
    object.__setattr__(state, "shift", 0)

    token = object.__new__(NullTransformToken)
    object.__setattr__(token, "schema", "selcal.null-transform-token.v2")
    object.__setattr__(token, "null_name", "circular_shift_v2")
    object.__setattr__(
        token,
        "null_parameter_sha256",
        "a2e6b608e05886b0c06388750e201c922c21d08e4ca6faf7668742798895dda3",
    )
    object.__setattr__(token, "semantic_input_sha256", _SEMANTIC_INPUT_SHA256)
    object.__setattr__(token, "scientific_plan_sha256", _ROW17_PLAN_SHA256)
    object.__setattr__(
        token,
        "bound_null_owner_sha256",
        _ROW17_BOUND_OWNER_SHA256,
    )
    object.__setattr__(token, "is_identity", True)
    object.__setattr__(token, "state", state)
    return token


def _row17_literal_vector(
    *,
    decision: float,
    runner_up: float,
) -> tuple[tuple[StatisticResult, ...], SelectionResult]:
    vector = (
        _literal_statistic_result(
            candidate_id=1,
            estimate=decision,
            selection_score=decision,
            validity=Validity.VALID,
        ),
        _literal_statistic_result(
            candidate_id=2,
            estimate=runner_up,
            selection_score=runner_up,
            validity=Validity.VALID,
        ),
    )
    selection = _literal_selection(
        selected_candidate=1,
        selected_index=0,
        decision_statistic=decision,
        tied_candidates=(1,),
    )
    return vector, selection


def _row17_dataflow_literal_case() -> tuple[CalibrationResult, PlanResolutionV2]:
    resolution = _row17_literal_resolution()
    token = _row17_literal_identity_token()
    observed_results, observed_selection = _row17_literal_vector(
        decision=0.75,
        runner_up=0.25,
    )
    replicate_0_results, replicate_0_selection = _row17_literal_vector(
        decision=0.5,
        runner_up=0.1,
    )
    replicate_1_results, replicate_1_selection = _row17_literal_vector(
        decision=0.8,
        runner_up=0.2,
    )
    replicate_2_results, replicate_2_selection = _row17_literal_vector(
        decision=0.2,
        runner_up=-0.1,
    )
    result = _literal_calibration_result(
        status=RunStatus.COMPLETE,
        failure_stage=None,
        observed_results=observed_results,
        observed_selection=observed_selection,
        replicates=(
            _literal_replicate(
                replicate_id=0,
                seed_digest_sha256=_ROW17_SEED_DIGESTS[0],
                status=ReplicateStatus.COMPLETE,
                failure_stage=None,
                transform_token=token,
                statistic_results=replicate_0_results,
                selection=replicate_0_selection,
            ),
            _literal_replicate(
                replicate_id=1,
                seed_digest_sha256=_ROW17_SEED_DIGESTS[1],
                status=ReplicateStatus.COMPLETE,
                failure_stage=None,
                transform_token=token,
                statistic_results=replicate_1_results,
                selection=replicate_1_selection,
            ),
            _literal_replicate(
                replicate_id=2,
                seed_digest_sha256=_ROW17_SEED_DIGESTS[2],
                status=ReplicateStatus.COMPLETE,
                failure_stage=None,
                transform_token=token,
                statistic_results=replicate_2_results,
                selection=replicate_2_selection,
            ),
        ),
        exceedance_count=1,
        failure_count=0,
        p_value=0.5,
        bound_low=None,
        bound_high=None,
        reject_null=False,
    )
    object.__setattr__(result, "scientific_plan_sha256", _ROW17_PLAN_SHA256)
    object.__setattr__(result, "alpha", 0.05)
    return result, resolution


_ResultFaultV2 = Callable[[CalibrationResult], None]
_ResultFaultAssertionV2 = Callable[[CalibrationResult], None]
_ContextFaultV2 = Callable[[tuple[object, ...]], tuple[object, ...]]
_ContextFaultAssertionV2 = Callable[[tuple[object, ...]], None]
_TokenFaultV2 = Callable[[CalibrationResult, NullTransformToken], None]
_TokenFaultAssertionV2 = Callable[[CalibrationResult, NullTransformToken], None]
_VerifierV2 = Callable[[object, object], None]


@dataclass(frozen=True, slots=True)
class PreCallFaultV2:
    fault_id: str
    apply: _ResultFaultV2
    assert_applied: _ResultFaultAssertionV2


@dataclass(frozen=True, slots=True)
class ContextPhaseFaultV2:
    fault_id: str
    read_index: int
    apply: _ContextFaultV2
    assert_applied: _ContextFaultAssertionV2


@dataclass(frozen=True, slots=True)
class BeforeRealTokenFaultV2:
    fault_id: str
    call_index: int
    apply: _TokenFaultV2
    assert_applied: _TokenFaultAssertionV2


@dataclass(frozen=True, slots=True)
class AfterSuccessfulTokenReturnFaultV2:
    fault_id: str
    call_index: int
    apply: _TokenFaultV2
    assert_applied: _TokenFaultAssertionV2


@dataclass(frozen=True, slots=True)
class FaultInjectionV2:
    label: str
    pre_call: PreCallFaultV2 | None = None
    context_phase: ContextPhaseFaultV2 | None = None
    before_real_token: BeforeRealTokenFaultV2 | None = None
    after_token_return: AfterSuccessfulTokenReturnFaultV2 | None = None


@dataclass(frozen=True, slots=True)
class AdjacentFaultScenarioV2:
    row_number: int
    row_id: str
    left: FaultInjectionV2
    right: FaultInjectionV2
    message: str
    has_cause: bool
    right_message: str
    right_has_cause: bool
    double_pre_call_fault_ids: tuple[str, ...] = ()
    right_pre_call_fault_ids: tuple[str, ...] = ()
    double_context_fault_ids: tuple[str, ...] = ()
    right_context_fault_ids: tuple[str, ...] = ()
    double_token_fault_ids: tuple[str, ...] = ()
    right_token_fault_ids: tuple[str, ...] = ()
    double_token_completion_ids: tuple[str, ...] = ()
    right_token_completion_ids: tuple[str, ...] = ()
    double_after_token_fault_ids: tuple[str, ...] = ()
    right_after_token_fault_ids: tuple[str, ...] = ()
    double_context_fault_calls: int = 0
    right_context_fault_calls: int = 0
    double_token_verifier_calls: int = 0
    right_token_verifier_calls: int = 0
    double_token_fault_calls: int = 0
    right_token_fault_calls: int = 0
    double_token_completion_calls: int = 0
    right_token_completion_calls: int = 0
    double_after_token_fault_calls: int = 0
    right_after_token_fault_calls: int = 0
    double_context_reads: int = 1
    right_context_reads: int = 1


class _FaultReceiptPhaseV2(StrEnum):
    PRE_CALL = "pre_call"
    CONTEXT_READ = "context_read"


@dataclass(frozen=True, slots=True)
class _FaultReceiptV2:
    phase: _FaultReceiptPhaseV2
    fault_id: str
    phase_index: int


class _TraversalReceiptPhaseV2(StrEnum):
    OBSERVED_STATISTIC = "observed_statistic"
    OBSERVED_SELECTION_REREAD = "observed_selection_reread"
    REPLICATE = "replicate"


@dataclass(frozen=True, slots=True)
class _TraversalReceiptV2:
    phase: _TraversalReceiptPhaseV2
    subject: str


@dataclass(frozen=True, slots=True)
class _IntegrityErrorExpectationV2:
    message: str
    has_cause: bool


@dataclass(frozen=True, slots=True)
class NullBindAdjacentFaultScenarioV2:
    row_id: str
    left: FaultInjectionV2
    right: FaultInjectionV2
    double_expectation: _IntegrityErrorExpectationV2
    right_expectation: _IntegrityErrorExpectationV2
    double_receipts: tuple[_FaultReceiptV2, ...]
    right_receipts: tuple[_FaultReceiptV2, ...]
    double_context_reads: int
    right_context_reads: int
    double_context_fault_calls: int
    right_context_fault_calls: int


@dataclass(frozen=True, slots=True)
class ObservedScanAdjacentFaultScenarioV2:
    row_id: str
    left: FaultInjectionV2
    right: FaultInjectionV2
    double_expectation: _IntegrityErrorExpectationV2
    right_expectation: _IntegrityErrorExpectationV2
    double_receipts: tuple[_FaultReceiptV2, ...]
    right_receipts: tuple[_FaultReceiptV2, ...]
    double_context_reads: int
    right_context_reads: int
    double_context_fault_calls: int
    right_context_fault_calls: int


@dataclass(frozen=True, slots=True)
class ReplicateExecutionAdjacentFaultScenarioV2:
    row_id: str
    left: FaultInjectionV2
    right: FaultInjectionV2
    double_expectation: _IntegrityErrorExpectationV2
    right_expectation: _IntegrityErrorExpectationV2
    double_receipts: tuple[_FaultReceiptV2, ...]
    right_receipts: tuple[_FaultReceiptV2, ...]
    double_context_reads: int
    right_context_reads: int
    double_context_fault_calls: int
    right_context_fault_calls: int
    double_token_verifier_calls: int
    right_token_verifier_calls: int
    double_token_completion_calls: int
    right_token_completion_calls: int


class _FourTerminalMatrixStateV2(StrEnum):
    COMPLETE = "COMPLETE"
    REPLICATE_EXECUTION = "REPLICATE_EXECUTION"
    NULL_BIND = "NULL_BIND"
    OBSERVED_STATISTIC_SCAN = "OBSERVED_STATISTIC_SCAN"


class _FourTerminalBoundaryKindV2(StrEnum):
    COUNT_TERMINAL = "COUNT_TERMINAL"
    TERMINAL_CONTEXT_READ_2 = "TERMINAL_CONTEXT_READ_2"


class _FourTerminalFixtureV2(StrEnum):
    COMPLETE = "COMPLETE"
    REPLICATE_EXECUTION = "REPLICATE_EXECUTION"
    NULL_BIND = "NULL_BIND"
    OBSERVED_STATISTIC_SCAN = "OBSERVED_STATISTIC_SCAN"


class _FourTerminalRunKindV2(StrEnum):
    DOUBLE = "DOUBLE"
    RIGHT_ONLY = "RIGHT_ONLY"


@dataclass(frozen=True, slots=True)
class _FourTerminalStimulusV2:
    factory: _FourTerminalFixtureV2
    scenario_id: str
    run_kind: _FourTerminalRunKindV2


@dataclass(frozen=True, slots=True)
class _TokenCallbackReceiptV2:
    call_index: int
    token_schema: str
    null_name: str
    null_parameter_sha256: str
    semantic_input_sha256: str
    scientific_plan_sha256: str
    bound_null_owner_sha256: str
    is_identity: bool
    state_schema: str
    state_payload: int | tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _FourTerminalObservedV2:
    status: RunStatus
    failure_stage: RunFailureStage | None
    exception_type: type[BaseException]
    message: str
    has_cause: bool
    context_reads: int
    read_2_actions: int
    fault_receipts: tuple[_FaultReceiptV2, ...]
    token_receipts: tuple[_TokenCallbackReceiptV2, ...]


@dataclass(frozen=True, slots=True)
class _FourTerminalOracleCellV2:
    state: _FourTerminalMatrixStateV2
    boundary: _FourTerminalBoundaryKindV2
    stimulus: _FourTerminalStimulusV2
    expected: _FourTerminalObservedV2


@dataclass(slots=True)
class _PhaseProbeV2:
    fault_receipts: list[_FaultReceiptV2] = field(default_factory=list)
    pre_call_fault_ids: list[str] = field(default_factory=list)
    context_fault_ids: list[str] = field(default_factory=list)
    token_fault_ids: list[str] = field(default_factory=list)
    token_completion_ids: list[str] = field(default_factory=list)
    token_completion_receipts: list[_TokenCallbackReceiptV2] = field(
        default_factory=list
    )
    after_token_fault_ids: list[str] = field(default_factory=list)
    context_reads: int = 0
    context_fault_calls: int = 0
    token_verifier_calls: int = 0
    token_fault_calls: int = 0
    token_completion_calls: int = 0
    after_token_fault_calls: int = 0


def _adjacent_fault_complete_case() -> tuple[CalibrationResult, PlanResolutionV2]:
    result, resolution, _, _ = _literal_zero_sign_characterization_case(
        estimate_zero=0.0,
        score_zero=0.0,
        decision_zero=0.0,
    )
    return result, resolution


def _null_bind_short_path_literal_case() -> tuple[
    CalibrationResult,
    PlanResolutionV2,
]:
    result = object.__new__(CalibrationResult)
    object.__setattr__(result, "status", RunStatus.NOT_EVALUABLE)
    object.__setattr__(result, "failure_stage", RunFailureStage.NULL_BIND)
    object.__setattr__(result, "semantic_input_sha256", _SEMANTIC_INPUT_SHA256)
    object.__setattr__(result, "scientific_plan_sha256", _SCIENTIFIC_PLAN_SHA256)
    object.__setattr__(result, "planned_replicates", 3)
    object.__setattr__(result, "alpha", 0.75)
    object.__setattr__(result, "observed_results", ())
    object.__setattr__(result, "observed_selection", None)
    object.__setattr__(result, "replicates", ())
    object.__setattr__(result, "exceedance_count", 0)
    object.__setattr__(result, "failure_count", 0)
    object.__setattr__(result, "p_value", None)
    object.__setattr__(result, "exceedance_bound_low", None)
    object.__setattr__(result, "exceedance_bound_high", None)
    object.__setattr__(result, "reject_null", None)
    object.__setattr__(result, "diagnostics", ("literal_null_bind_disabled",))
    return result, _literal_resolution()


def _observed_statistic_scan_short_path_literal_case() -> tuple[
    CalibrationResult,
    PlanResolutionV2,
]:
    observed_results = (
        _literal_statistic_result(
            candidate_id=1,
            estimate=2.0,
            selection_score=None,
            validity=Validity.VALID,
        ),
        _literal_statistic_result(
            candidate_id=2,
            estimate=None,
            selection_score=None,
            validity=Validity.ANALYTIC_FAILURE,
            diagnostics=("literal_analytic_failure",),
        ),
    )
    result = object.__new__(CalibrationResult)
    object.__setattr__(result, "status", RunStatus.NOT_EVALUABLE)
    object.__setattr__(
        result,
        "failure_stage",
        RunFailureStage.OBSERVED_STATISTIC_SCAN,
    )
    object.__setattr__(result, "semantic_input_sha256", _SEMANTIC_INPUT_SHA256)
    object.__setattr__(result, "scientific_plan_sha256", _SCIENTIFIC_PLAN_SHA256)
    object.__setattr__(result, "planned_replicates", 3)
    object.__setattr__(result, "alpha", 0.75)
    object.__setattr__(result, "observed_results", observed_results)
    object.__setattr__(result, "observed_selection", None)
    object.__setattr__(result, "replicates", ())
    object.__setattr__(result, "exceedance_count", 0)
    object.__setattr__(result, "failure_count", 0)
    object.__setattr__(result, "p_value", None)
    object.__setattr__(result, "exceedance_bound_low", None)
    object.__setattr__(result, "exceedance_bound_high", None)
    object.__setattr__(result, "reject_null", None)
    object.__setattr__(result, "diagnostics", ("literal_observed_scan_failure",))
    return result, _literal_resolution()


def _replicate_execution_terminal_literal_case() -> tuple[
    CalibrationResult,
    PlanResolutionV2,
]:
    resolution = _literal_resolution()
    tokens = (
        _literal_replicate_execution_token(shift=0, is_identity=True),
        _literal_replicate_execution_token(shift=1, is_identity=False),
        _literal_replicate_execution_token(shift=2, is_identity=False),
    )
    observed_results = (
        _literal_statistic_result(
            candidate_id=1,
            estimate=3.0,
            selection_score=3.0,
            validity=Validity.VALID,
        ),
        _literal_statistic_result(
            candidate_id=2,
            estimate=1.0,
            selection_score=1.0,
            validity=Validity.VALID,
        ),
    )
    observed_selection = _literal_selection(
        selected_candidate=1,
        selected_index=0,
        decision_statistic=3.0,
        tied_candidates=(1,),
    )
    exceedance_results = (
        _literal_statistic_result(
            candidate_id=1,
            estimate=4.0,
            selection_score=4.0,
            validity=Validity.VALID,
        ),
        _literal_statistic_result(
            candidate_id=2,
            estimate=2.0,
            selection_score=2.0,
            validity=Validity.VALID,
        ),
    )
    exceedance_selection = _literal_selection(
        selected_candidate=1,
        selected_index=0,
        decision_statistic=4.0,
        tied_candidates=(1,),
    )
    analytic_failure_results = (
        _literal_statistic_result(
            candidate_id=1,
            estimate=None,
            selection_score=None,
            validity=Validity.ANALYTIC_FAILURE,
            diagnostics=("literal_replicate_failure_candidate_1",),
        ),
        _literal_statistic_result(
            candidate_id=2,
            estimate=None,
            selection_score=None,
            validity=Validity.ANALYTIC_FAILURE,
            diagnostics=("literal_replicate_failure_candidate_2",),
        ),
    )
    nonexceedance_results = (
        _literal_statistic_result(
            candidate_id=1,
            estimate=2.0,
            selection_score=2.0,
            validity=Validity.VALID,
        ),
        _literal_statistic_result(
            candidate_id=2,
            estimate=1.0,
            selection_score=1.0,
            validity=Validity.VALID,
        ),
    )
    nonexceedance_selection = _literal_selection(
        selected_candidate=1,
        selected_index=0,
        decision_statistic=2.0,
        tied_candidates=(1,),
    )
    replicates = (
        _literal_replicate(
            replicate_id=0,
            seed_digest_sha256=_SEED_DIGEST_0,
            status=ReplicateStatus.COMPLETE,
            failure_stage=None,
            transform_token=tokens[0],
            statistic_results=exceedance_results,
            selection=exceedance_selection,
        ),
        _literal_replicate(
            replicate_id=1,
            seed_digest_sha256=_SEED_DIGEST_1,
            status=ReplicateStatus.ANALYTIC_FAILURE,
            failure_stage=ReplicateFailureStage.STATISTIC_SCAN,
            transform_token=tokens[1],
            statistic_results=analytic_failure_results,
            selection=None,
            diagnostics=("literal_replicate_execution_failure",),
        ),
        _literal_replicate(
            replicate_id=2,
            seed_digest_sha256=_SEED_DIGEST_2,
            status=ReplicateStatus.COMPLETE,
            failure_stage=None,
            transform_token=tokens[2],
            statistic_results=nonexceedance_results,
            selection=nonexceedance_selection,
        ),
    )
    result = object.__new__(CalibrationResult)
    object.__setattr__(result, "status", RunStatus.NOT_EVALUABLE)
    object.__setattr__(
        result,
        "failure_stage",
        RunFailureStage.REPLICATE_EXECUTION,
    )
    object.__setattr__(result, "semantic_input_sha256", _SEMANTIC_INPUT_SHA256)
    object.__setattr__(result, "scientific_plan_sha256", _SCIENTIFIC_PLAN_SHA256)
    object.__setattr__(result, "planned_replicates", 3)
    object.__setattr__(result, "alpha", 0.75)
    object.__setattr__(result, "observed_results", observed_results)
    object.__setattr__(result, "observed_selection", observed_selection)
    object.__setattr__(result, "replicates", replicates)
    object.__setattr__(result, "exceedance_count", 1)
    object.__setattr__(result, "failure_count", 1)
    object.__setattr__(result, "p_value", None)
    object.__setattr__(result, "exceedance_bound_low", 0.5)
    object.__setattr__(result, "exceedance_bound_high", 0.75)
    object.__setattr__(result, "reject_null", None)
    object.__setattr__(result, "diagnostics", ("literal_replicate_execution_failure",))
    return result, resolution


def _assert_replicate_execution_terminal_literal_contract(
    result: CalibrationResult,
) -> None:
    assert result.status is RunStatus.NOT_EVALUABLE
    assert result.failure_stage is RunFailureStage.REPLICATE_EXECUTION
    assert type(result.planned_replicates) is int
    assert result.planned_replicates == 3
    assert type(result.replicates) is tuple
    assert len(result.replicates) == 3
    assert tuple(outcome.replicate_id for outcome in result.replicates) == (0, 1, 2)
    assert tuple(outcome.status for outcome in result.replicates) == (
        ReplicateStatus.COMPLETE,
        ReplicateStatus.ANALYTIC_FAILURE,
        ReplicateStatus.COMPLETE,
    )
    assert tuple(outcome.failure_stage for outcome in result.replicates) == (
        None,
        ReplicateFailureStage.STATISTIC_SCAN,
        None,
    )
    tokens = tuple(outcome.transform_token for outcome in result.replicates)
    assert all(type(token) is NullTransformToken for token in tokens)
    assert len({id(token) for token in tokens}) == 3
    states = tuple(token.state for token in tokens)
    assert all(type(state) is CircularShiftStateV2 for state in states)
    assert len({id(state) for state in states}) == 3
    assert tuple(
        (
            token.schema,
            token.null_name,
            token.null_parameter_sha256,
            token.semantic_input_sha256,
            token.scientific_plan_sha256,
            token.bound_null_owner_sha256,
            token.is_identity,
            state.schema,
            state.shift,
        )
        for token, state in zip(tokens, states, strict=True)
    ) == (
        (
            "selcal.null-transform-token.v2",
            "circular_shift_v2",
            "a2e6b608e05886b0c06388750e201c922c21d08e4ca6faf7668742798895dda3",
            _SEMANTIC_INPUT_SHA256,
            _SCIENTIFIC_PLAN_SHA256,
            "6863e569187ddc8983aa51f622ce96d6d644842ae76a116f6c31c9e5338befba",
            True,
            "selcal.circular-shift-state.v2",
            0,
        ),
        (
            "selcal.null-transform-token.v2",
            "circular_shift_v2",
            "a2e6b608e05886b0c06388750e201c922c21d08e4ca6faf7668742798895dda3",
            _SEMANTIC_INPUT_SHA256,
            _SCIENTIFIC_PLAN_SHA256,
            "6863e569187ddc8983aa51f622ce96d6d644842ae76a116f6c31c9e5338befba",
            False,
            "selcal.circular-shift-state.v2",
            1,
        ),
        (
            "selcal.null-transform-token.v2",
            "circular_shift_v2",
            "a2e6b608e05886b0c06388750e201c922c21d08e4ca6faf7668742798895dda3",
            _SEMANTIC_INPUT_SHA256,
            _SCIENTIFIC_PLAN_SHA256,
            "6863e569187ddc8983aa51f622ce96d6d644842ae76a116f6c31c9e5338befba",
            False,
            "selcal.circular-shift-state.v2",
            2,
        ),
    )
    observed_selection = result.observed_selection
    assert type(observed_selection) is SelectionResult
    assert type(observed_selection.decision_statistic) is float
    assert observed_selection.decision_statistic == 3.0
    first_selection = result.replicates[0].selection
    assert type(first_selection) is SelectionResult
    assert type(first_selection.decision_statistic) is float
    assert first_selection.decision_statistic == 4.0
    assert result.replicates[1].selection is None
    third_selection = result.replicates[2].selection
    assert type(third_selection) is SelectionResult
    assert type(third_selection.decision_statistic) is float
    assert third_selection.decision_statistic == 2.0
    failure_vector = result.replicates[1].statistic_results
    assert type(failure_vector) is tuple
    assert len(failure_vector) == 2
    assert all(
        item.validity is Validity.ANALYTIC_FAILURE for item in failure_vector
    )
    assert all(item.estimate is None for item in failure_vector)
    assert all(item.selection_score is None for item in failure_vector)
    assert type(result.exceedance_count) is int
    assert result.exceedance_count == 1
    assert type(result.failure_count) is int
    assert result.failure_count == 1
    assert result.p_value is None
    assert type(result.exceedance_bound_low) is float
    assert result.exceedance_bound_low == 0.5
    assert type(result.exceedance_bound_high) is float
    assert result.exceedance_bound_high == 0.75
    assert result.reject_null is None


def _assert_observed_statistic_scan_literal_contract(
    result: CalibrationResult,
) -> None:
    assert result.status is RunStatus.NOT_EVALUABLE
    assert result.failure_stage is RunFailureStage.OBSERVED_STATISTIC_SCAN
    observed_results: object = result.observed_results
    assert type(observed_results) is tuple
    assert len(observed_results) == 2
    first, second = observed_results
    assert type(first) is StatisticResult
    assert first.candidate_id == 1
    assert type(first.estimate) is float and first.estimate == 2.0
    assert first.selection_score is None
    assert first.validity is Validity.VALID
    assert type(second) is StatisticResult
    assert second.candidate_id == 2
    assert second.estimate is None
    assert second.selection_score is None
    assert second.validity is Validity.ANALYTIC_FAILURE
    assert result.observed_selection is None
    assert result.replicates == ()
    assert type(result.exceedance_count) is int
    assert result.exceedance_count == 0
    assert type(result.failure_count) is int
    assert result.failure_count == 0
    assert result.p_value is None
    assert result.exceedance_bound_low is None
    assert result.exceedance_bound_high is None
    assert result.reject_null is None


def _closure_cell(value: object) -> CellType:
    def capture() -> object:
        return value

    closure = capture.__closure__
    assert closure is not None
    return closure[0]


def _token_callback_receipt(
    call_index: int,
    token: NullTransformToken,
) -> _TokenCallbackReceiptV2:
    state = token.state
    if type(state) is CircularShiftStateV2:
        state_payload: int | tuple[int, ...] = state.shift
        state_schema = state.schema
    else:
        assert type(state) is BlockShuffleStateV2
        state_payload = state.block_order
        state_schema = state.schema
    return _TokenCallbackReceiptV2(
        call_index=call_index,
        token_schema=token.schema,
        null_name=token.null_name,
        null_parameter_sha256=token.null_parameter_sha256,
        semantic_input_sha256=token.semantic_input_sha256,
        scientific_plan_sha256=token.scientific_plan_sha256,
        bound_null_owner_sha256=token.bound_null_owner_sha256,
        is_identity=token.is_identity,
        state_schema=state_schema,
        state_payload=state_payload,
    )


def _verifier_with_phase_profile(
    result: CalibrationResult,
    injections: tuple[FaultInjectionV2, ...],
    probe: _PhaseProbeV2,
) -> _VerifierV2:
    original = contracts_v2.verify_calibration_result
    original_closure = original.__closure__
    assert original_closure is not None
    closure_by_name = dict(
        zip(original.__code__.co_freevars, original_closure, strict=True)
    )
    canonical_context_reader = closure_by_name["resolution_context"].cell_contents
    assert callable(canonical_context_reader)

    canonical_token_verifier: Callable[..., None] | None = None

    def profiled_token_verifier(
        null_adapter: object,
        transform_token: object,
        semantic_digest: object,
        plan_digest: object,
        observed_length: object,
        *,
        replicate_id: object = None,
        planned_replicates: object = None,
    ) -> None:
        probe.token_verifier_calls += 1
        for injection in injections:
            token_phase = injection.before_real_token
            if (
                token_phase is not None
                and token_phase.call_index == probe.token_verifier_calls
            ):
                assert type(transform_token) is NullTransformToken
                probe.token_fault_calls += 1
                token_phase.apply(result, transform_token)
                token_phase.assert_applied(result, transform_token)
                probe.token_fault_ids.append(token_phase.fault_id)
        assert canonical_token_verifier is not None
        canonical_token_verifier(
            null_adapter,
            transform_token,
            semantic_digest,
            plan_digest,
            observed_length,
            replicate_id=replicate_id,
            planned_replicates=planned_replicates,
        )
        probe.token_completion_calls += 1
        probe.token_completion_ids.append(
            f"phase.token.real-return.call-{probe.token_verifier_calls}"
        )
        assert type(transform_token) is NullTransformToken
        probe.token_completion_receipts.append(
            _token_callback_receipt(
                probe.token_verifier_calls,
                transform_token,
            )
        )
        for injection in injections:
            after_token_return = injection.after_token_return
            if (
                after_token_return is not None
                and after_token_return.call_index == probe.token_verifier_calls
            ):
                assert type(transform_token) is NullTransformToken
                probe.after_token_fault_calls += 1
                after_token_return.apply(result, transform_token)
                after_token_return.assert_applied(result, transform_token)
                probe.after_token_fault_ids.append(after_token_return.fault_id)

    def profiled_context_reader(resolution: object) -> object:
        nonlocal canonical_token_verifier
        probe.context_reads += 1
        context = canonical_context_reader(resolution)
        assert type(context) is tuple and len(context) == 11
        current_token_verifier = context[10]
        assert callable(current_token_verifier)
        if canonical_token_verifier is None:
            canonical_token_verifier = current_token_verifier
        else:
            assert canonical_token_verifier is current_token_verifier
        profiled_context = (*context[:10], profiled_token_verifier)
        for injection in injections:
            context_phase = injection.context_phase
            if (
                context_phase is not None
                and context_phase.read_index == probe.context_reads
            ):
                probe.context_fault_calls += 1
                profiled_context = context_phase.apply(profiled_context)
                context_phase.assert_applied(profiled_context)
                probe.context_fault_ids.append(context_phase.fault_id)
                probe.fault_receipts.append(
                    _FaultReceiptV2(
                        phase=_FaultReceiptPhaseV2.CONTEXT_READ,
                        fault_id=context_phase.fault_id,
                        phase_index=probe.context_reads,
                    )
                )
        return profiled_context

    profiled_closure = tuple(
        _closure_cell(profiled_context_reader)
        if name == "resolution_context"
        else cell
        for name, cell in zip(
            original.__code__.co_freevars,
            original_closure,
            strict=True,
        )
    )
    return FunctionType(
        original.__code__,
        original.__globals__,
        original.__name__,
        original.__defaults__,
        profiled_closure,
    )


def _apply_before_call_faults(
    result: CalibrationResult,
    injections: tuple[FaultInjectionV2, ...],
    probe: _PhaseProbeV2,
) -> None:
    for injection in injections:
        if injection.pre_call is not None:
            injection.pre_call.apply(result)
    for injection in injections:
        if injection.pre_call is not None:
            injection.pre_call.assert_applied(result)
            probe.pre_call_fault_ids.append(injection.pre_call.fault_id)
            probe.fault_receipts.append(
                _FaultReceiptV2(
                    phase=_FaultReceiptPhaseV2.PRE_CALL,
                    fault_id=injection.pre_call.fault_id,
                    phase_index=0,
                )
            )


def _first_adjacent_replicate(result: CalibrationResult) -> ReplicateOutcome:
    replicates = result.replicates
    assert type(replicates) is tuple and replicates
    outcome = replicates[0]
    assert type(outcome) is ReplicateOutcome
    return outcome


def _first_adjacent_selection(result: CalibrationResult) -> SelectionResult:
    selection = _first_adjacent_replicate(result).selection
    assert type(selection) is SelectionResult
    return selection


def _fault_context_read_1(
    context: tuple[object, ...],
) -> tuple[object, ...]:
    return context[:-1]


def _fault_result_slots(result: CalibrationResult) -> None:
    object.__delattr__(result, "status")


def _fault_result_diagnostics(result: CalibrationResult) -> None:
    object.__setattr__(result, "diagnostics", [])


def _fault_exact_replicate_tuple(result: CalibrationResult) -> None:
    object.__setattr__(result, "replicates", list(result.replicates))


def _fault_semantic_digest(result: CalibrationResult) -> None:
    object.__setattr__(result, "semantic_input_sha256", "invalid-semantic-digest")


def _fault_plan_digest(result: CalibrationResult) -> None:
    object.__setattr__(result, "scientific_plan_sha256", "invalid-plan-digest")


def _fault_result_scalars(result: CalibrationResult) -> None:
    object.__setattr__(result, "planned_replicates", True)


def _fault_sealed_selection_contract(
    context: tuple[object, ...],
) -> tuple[object, ...]:
    return (*context[:2], (), *context[3:])


def _fault_observed_vector_structure(result: CalibrationResult) -> None:
    object.__setattr__(result, "observed_results", list(result.observed_results))


def _fault_observed_selection(result: CalibrationResult) -> None:
    object.__setattr__(result, "observed_selection", object())


def _fault_observed_selection_decision(result: CalibrationResult) -> None:
    object.__setattr__(result.observed_selection, "decision_statistic", 2.0)


def _fault_exact_b_retention(result: CalibrationResult) -> None:
    object.__setattr__(result, "replicates", result.replicates[:2])


def _fault_replicate_slots(result: CalibrationResult) -> None:
    object.__delattr__(_first_adjacent_replicate(result), "status")


def _fault_replicate_diagnostics(result: CalibrationResult) -> None:
    object.__setattr__(_first_adjacent_replicate(result), "diagnostics", [])


def _fault_replicate_identity_and_seed(result: CalibrationResult) -> None:
    object.__setattr__(
        _first_adjacent_replicate(result),
        "seed_digest_sha256",
        "0000000000000000000000000000000000000000000000000000000000000000",
    )


def _fault_replicate_vector(result: CalibrationResult) -> None:
    outcome = _first_adjacent_replicate(result)
    object.__setattr__(outcome, "statistic_results", list(outcome.statistic_results))


def _fault_common_support(result: CalibrationResult) -> None:
    statistic_results = _first_adjacent_replicate(result).statistic_results
    for statistic_result in statistic_results:
        object.__setattr__(statistic_result, "support_n", 3)


def _fault_sealed_token_verification(
    _result: CalibrationResult,
    token: NullTransformToken,
) -> None:
    object.__setattr__(token, "schema", "invalid-token-schema")


def _fault_status_stage_pairing(result: CalibrationResult) -> None:
    object.__setattr__(
        _first_adjacent_replicate(result),
        "failure_stage",
        ReplicateFailureStage.STATISTIC_SCAN,
    )


def _fault_post_token_selection_reread(
    result: CalibrationResult,
    _token: NullTransformToken,
) -> None:
    object.__delattr__(_first_adjacent_selection(result), "decision_statistic")


def _fault_post_token_decision(
    result: CalibrationResult,
    _token: NullTransformToken,
) -> None:
    selection = _first_adjacent_selection(result)
    decision: object = selection.decision_statistic
    assert type(decision) is float
    assert decision == 0.5
    object.__setattr__(selection, "decision_statistic", 0.75)


class _Row18AccumulationComparisonOperand:
    """Test-only propagated phase probe, never a public/scientific contract."""

    __slots__ = ("addition_calls", "comparison_calls")

    def __init__(self) -> None:
        self.comparison_calls = 0
        self.addition_calls = 0

    def __ge__(self, other: object) -> _Row18AccumulationAdditionSentinel:
        assert type(other) is float
        assert other == 3.0
        self.comparison_calls += 1
        return _Row18AccumulationAdditionSentinel(self)


class _Row18AccumulationAdditionSentinel:
    """Raise only when production propagates the comparison into E accumulation."""

    __slots__ = ("owner",)

    def __init__(self, owner: _Row18AccumulationComparisonOperand) -> None:
        self.owner = owner

    def __radd__(self, other: object) -> int:
        assert type(other) is int
        assert other == 0
        self.owner.addition_calls += 1
        raise V2IntegrityError("TEST-ONLY E-F accumulation phase probe")


def _fault_e_f_accumulation_probe(
    result: CalibrationResult,
    _token: NullTransformToken,
) -> None:
    selection = _first_adjacent_selection(result)
    decision: object = selection.decision_statistic
    assert type(decision) is float
    assert decision == 4.0
    object.__setattr__(
        selection,
        "decision_statistic",
        _Row18AccumulationComparisonOperand(),
    )


def _fault_exact_count_types(result: CalibrationResult) -> None:
    object.__setattr__(result, "exceedance_count", True)


def _fault_terminal_arithmetic(result: CalibrationResult) -> None:
    object.__setattr__(result, "p_value", 0.25)


def _fault_context_read_2_drift(
    context: tuple[object, ...],
) -> tuple[object, ...]:
    return (*context[:6], 0.25, *context[7:])


class _AdverseNullBindObservedResultDecoyV2:
    """Invalid observed-vector member that NULL_BIND must not traverse."""

    __slots__ = ()


class _AdverseNullBindObservedSelectionDecoyV2:
    """Invalid observed selection that NULL_BIND must not traverse."""

    __slots__ = ()


def _install_null_bind_adverse_observed_decoys(
    result: CalibrationResult,
) -> None:
    object.__setattr__(
        result,
        "observed_results",
        (
            _AdverseNullBindObservedResultDecoyV2(),
            _AdverseNullBindObservedResultDecoyV2(),
        ),
    )
    object.__setattr__(
        result,
        "observed_selection",
        _AdverseNullBindObservedSelectionDecoyV2(),
    )


def _fault_null_bind_exact_count_types(result: CalibrationResult) -> None:
    object.__setattr__(result, "exceedance_count", True)


def _fault_null_bind_terminal_record(result: CalibrationResult) -> None:
    object.__setattr__(result, "p_value", 0.25)


def _fault_null_bind_context_read_2_drift(
    context: tuple[object, ...],
) -> tuple[object, ...]:
    return (*context[:6], 0.25, *context[7:])


def _fault_observed_scan_failure_vector(result: CalibrationResult) -> None:
    observed_results = result.observed_results
    assert type(observed_results) is tuple and len(observed_results) == 2
    object.__setattr__(observed_results[1], "estimate", 0.25)


def _fault_observed_scan_exact_count_types(result: CalibrationResult) -> None:
    object.__setattr__(result, "exceedance_count", True)


def _fault_observed_scan_terminal_record(result: CalibrationResult) -> None:
    object.__setattr__(result, "p_value", 0.25)


def _fault_observed_scan_context_read_2_drift(
    context: tuple[object, ...],
) -> tuple[object, ...]:
    return (*context[:6], 0.25, *context[7:])


def _fault_replicate_execution_exact_count_types(
    result: CalibrationResult,
) -> None:
    object.__setattr__(result, "failure_count", True)


def _fault_replicate_execution_terminal_bounds(
    result: CalibrationResult,
) -> None:
    object.__setattr__(result, "exceedance_bound_high", 0.5)


def _fault_replicate_execution_context_read_2_drift(
    context: tuple[object, ...],
) -> tuple[object, ...]:
    return (*context[:6], 0.25, *context[7:])


def _assert_context_read_1_fault_applied(context: tuple[object, ...]) -> None:
    assert len(context) == 10


def _assert_sealed_selection_contract_fault_applied(
    context: tuple[object, ...],
) -> None:
    assert len(context) == 11
    assert context[2] == ()


def _assert_context_read_2_drift_fault_applied(
    context: tuple[object, ...],
) -> None:
    assert len(context) == 11
    assert type(context[6]) is float
    assert context[6] == 0.25


def _assert_result_slots_fault_applied(result: CalibrationResult) -> None:
    assert not hasattr(result, "status")


def _assert_result_diagnostics_fault_applied(result: CalibrationResult) -> None:
    corrupted_diagnostics: object = result.diagnostics
    assert type(corrupted_diagnostics) is list and corrupted_diagnostics == []


def _assert_exact_replicate_tuple_fault_applied(result: CalibrationResult) -> None:
    corrupted_replicates: object = result.replicates
    assert type(corrupted_replicates) is list and len(corrupted_replicates) == 3


def _assert_semantic_digest_fault_applied(result: CalibrationResult) -> None:
    assert result.semantic_input_sha256 == "invalid-semantic-digest"


def _assert_plan_digest_fault_applied(result: CalibrationResult) -> None:
    assert result.scientific_plan_sha256 == "invalid-plan-digest"


def _assert_result_scalars_fault_applied(result: CalibrationResult) -> None:
    assert type(result.planned_replicates) is bool
    assert result.planned_replicates is True


def _assert_observed_vector_structure_fault_applied(
    result: CalibrationResult,
) -> None:
    corrupted_observed_results: object = result.observed_results
    assert type(corrupted_observed_results) is list
    assert len(corrupted_observed_results) == 2


def _assert_observed_selection_fault_applied(result: CalibrationResult) -> None:
    assert type(result.observed_selection) is object


def _assert_observed_selection_decision_fault_applied(
    result: CalibrationResult,
) -> None:
    assert type(result.observed_selection) is SelectionResult
    assert result.observed_selection.decision_statistic == 2.0


def _assert_exact_b_retention_fault_applied(result: CalibrationResult) -> None:
    assert type(result.replicates) is tuple
    assert tuple(outcome.replicate_id for outcome in result.replicates) == (0, 1)


def _assert_replicate_slots_fault_applied(result: CalibrationResult) -> None:
    assert not hasattr(_first_adjacent_replicate(result), "status")
    assert hasattr(result.replicates[1], "status")


def _assert_replicate_diagnostics_fault_applied(result: CalibrationResult) -> None:
    corrupted_diagnostics: object = _first_adjacent_replicate(result).diagnostics
    assert type(corrupted_diagnostics) is list
    assert corrupted_diagnostics == []
    assert type(result.diagnostics) is tuple


def _assert_replicate_identity_and_seed_fault_applied(
    result: CalibrationResult,
) -> None:
    assert _first_adjacent_replicate(result).seed_digest_sha256 == (
        "0000000000000000000000000000000000000000000000000000000000000000"
    )
    assert result.replicates[1].seed_digest_sha256 == _SEED_DIGEST_1


def _assert_replicate_vector_fault_applied(result: CalibrationResult) -> None:
    corrupted_results: object = _first_adjacent_replicate(result).statistic_results
    assert type(corrupted_results) is list
    assert type(result.replicates[1].statistic_results) is tuple


def _assert_common_support_fault_applied(result: CalibrationResult) -> None:
    replicate_results = _first_adjacent_replicate(result).statistic_results
    assert all(item.support_n == 3 for item in replicate_results)
    assert all(item.support_n == 4 for item in result.observed_results)


def _assert_status_stage_pairing_fault_applied(result: CalibrationResult) -> None:
    outcome = _first_adjacent_replicate(result)
    assert outcome.status is ReplicateStatus.COMPLETE
    assert outcome.failure_stage is ReplicateFailureStage.STATISTIC_SCAN


def _assert_sealed_token_verification_fault_applied(
    _result: CalibrationResult,
    token: NullTransformToken,
) -> None:
    corrupted_schema: object = token.schema
    assert corrupted_schema == "invalid-token-schema"


def _assert_post_token_selection_reread_fault_applied(
    result: CalibrationResult,
    _token: NullTransformToken,
) -> None:
    assert not hasattr(_first_adjacent_selection(result), "decision_statistic")


def _assert_post_token_decision_fault_applied(
    result: CalibrationResult,
    _token: NullTransformToken,
) -> None:
    decision: object = _first_adjacent_selection(result).decision_statistic
    assert type(decision) is float
    assert decision == 0.75


def _assert_e_f_accumulation_probe_applied(
    result: CalibrationResult,
    _token: NullTransformToken,
) -> None:
    operand: object = _first_adjacent_selection(result).decision_statistic
    assert type(operand) is _Row18AccumulationComparisonOperand
    assert operand.comparison_calls == 0
    assert operand.addition_calls == 0


def _assert_exact_count_types_fault_applied(result: CalibrationResult) -> None:
    exceedance_count: object = result.exceedance_count
    assert type(exceedance_count) is bool
    assert exceedance_count is True


def _assert_terminal_arithmetic_fault_applied(result: CalibrationResult) -> None:
    p_value: object = result.p_value
    assert type(p_value) is float
    assert p_value == 0.25


def _assert_null_bind_adverse_observed_decoys_installed(
    result: CalibrationResult,
) -> None:
    adverse_observed: object = result.observed_results
    assert type(adverse_observed) is tuple
    assert len(adverse_observed) == 2
    assert all(
        type(item) is _AdverseNullBindObservedResultDecoyV2
        for item in adverse_observed
    )
    adverse_selection: object = result.observed_selection
    assert type(adverse_selection) is _AdverseNullBindObservedSelectionDecoyV2
    assert type(result.replicates) is tuple
    assert result.replicates == ()


def _assert_null_bind_exact_count_types_fault_applied(
    result: CalibrationResult,
) -> None:
    exceedance_count: object = result.exceedance_count
    assert type(exceedance_count) is bool
    assert exceedance_count is True


def _assert_null_bind_terminal_record_fault_applied(
    result: CalibrationResult,
) -> None:
    p_value: object = result.p_value
    assert type(p_value) is float
    assert p_value == 0.25


def _assert_null_bind_context_read_2_drift_fault_applied(
    context: tuple[object, ...],
) -> None:
    assert len(context) == 11
    assert type(context[6]) is float
    assert context[6] == 0.25


def _assert_observed_scan_failure_vector_fault_applied(
    result: CalibrationResult,
) -> None:
    observed_results: object = result.observed_results
    assert type(observed_results) is tuple
    assert len(observed_results) == 2
    first, second = observed_results
    assert type(first) is StatisticResult
    assert first.validity is Validity.VALID
    assert type(first.estimate) is float and first.estimate == 2.0
    assert first.selection_score is None
    assert type(second) is StatisticResult
    assert second.validity is Validity.ANALYTIC_FAILURE
    assert type(second.estimate) is float and second.estimate == 0.25
    assert second.selection_score is None
    assert result.observed_selection is None


def _assert_observed_scan_exact_count_types_fault_applied(
    result: CalibrationResult,
) -> None:
    exceedance_count: object = result.exceedance_count
    assert type(exceedance_count) is bool
    assert exceedance_count is True


def _assert_observed_scan_terminal_record_fault_applied(
    result: CalibrationResult,
) -> None:
    p_value: object = result.p_value
    assert type(p_value) is float
    assert p_value == 0.25


def _assert_observed_scan_context_read_2_drift_fault_applied(
    context: tuple[object, ...],
) -> None:
    assert len(context) == 11
    assert type(context[6]) is float
    assert context[6] == 0.25


def _assert_replicate_execution_exact_count_types_fault_applied(
    result: CalibrationResult,
) -> None:
    failure_count: object = result.failure_count
    assert type(failure_count) is bool
    assert failure_count is True


def _assert_replicate_execution_terminal_bounds_fault_applied(
    result: CalibrationResult,
) -> None:
    bound_high: object = result.exceedance_bound_high
    assert type(bound_high) is float
    assert bound_high == 0.5


def _assert_replicate_execution_context_read_2_drift_fault_applied(
    context: tuple[object, ...],
) -> None:
    assert len(context) == 11
    assert type(context[6]) is float
    assert context[6] == 0.25


_CONTEXT_READ_1_FAULT = FaultInjectionV2(
    label="context read #1",
    context_phase=ContextPhaseFaultV2(
        fault_id="phase.context.read-1.invalid-shape",
        read_index=1,
        apply=_fault_context_read_1,
        assert_applied=_assert_context_read_1_fault_applied,
    ),
)
_RESULT_SLOTS_FAULT = FaultInjectionV2(
    label="result slots",
    pre_call=PreCallFaultV2(
        fault_id="pre.result.slots.missing-status",
        apply=_fault_result_slots,
        assert_applied=_assert_result_slots_fault_applied,
    ),
)
_RESULT_DIAGNOSTICS_FAULT = FaultInjectionV2(
    label="result diagnostics",
    pre_call=PreCallFaultV2(
        fault_id="pre.result.diagnostics.list",
        apply=_fault_result_diagnostics,
        assert_applied=_assert_result_diagnostics_fault_applied,
    ),
)
_EXACT_REPLICATE_TUPLE_FAULT = FaultInjectionV2(
    label="exact replicate tuple",
    pre_call=PreCallFaultV2(
        fault_id="pre.result.replicates.list",
        apply=_fault_exact_replicate_tuple,
        assert_applied=_assert_exact_replicate_tuple_fault_applied,
    ),
)
_SEMANTIC_DIGEST_FAULT = FaultInjectionV2(
    label="semantic digest",
    pre_call=PreCallFaultV2(
        fault_id="pre.result.semantic-digest.invalid",
        apply=_fault_semantic_digest,
        assert_applied=_assert_semantic_digest_fault_applied,
    ),
)
_PLAN_DIGEST_FAULT = FaultInjectionV2(
    label="plan digest",
    pre_call=PreCallFaultV2(
        fault_id="pre.result.plan-digest.invalid",
        apply=_fault_plan_digest,
        assert_applied=_assert_plan_digest_fault_applied,
    ),
)
_RESULT_SCALARS_FAULT = FaultInjectionV2(
    label="result scalars",
    pre_call=PreCallFaultV2(
        fault_id="pre.result.planned-replicates.bool",
        apply=_fault_result_scalars,
        assert_applied=_assert_result_scalars_fault_applied,
    ),
)
_SEALED_SELECTION_CONTRACT_FAULT = FaultInjectionV2(
    label="sealed selection contract",
    context_phase=ContextPhaseFaultV2(
        fault_id="phase.context.selection-contract.empty-candidates",
        read_index=1,
        apply=_fault_sealed_selection_contract,
        assert_applied=_assert_sealed_selection_contract_fault_applied,
    ),
)
_OBSERVED_VECTOR_STRUCTURE_FAULT = FaultInjectionV2(
    label="observed vector structure",
    pre_call=PreCallFaultV2(
        fault_id="pre.result.observed-vector.list",
        apply=_fault_observed_vector_structure,
        assert_applied=_assert_observed_vector_structure_fault_applied,
    ),
)
_OBSERVED_SELECTION_FAULT = FaultInjectionV2(
    label="observed selection",
    pre_call=PreCallFaultV2(
        fault_id="pre.result.observed-selection.foreign-object",
        apply=_fault_observed_selection,
        assert_applied=_assert_observed_selection_fault_applied,
    ),
)
_OBSERVED_SELECTION_DECISION_FAULT = FaultInjectionV2(
    label="observed selection decision",
    pre_call=PreCallFaultV2(
        fault_id="pre.result.observed-selection.decision-2",
        apply=_fault_observed_selection_decision,
        assert_applied=_assert_observed_selection_decision_fault_applied,
    ),
)
_EXACT_B_RETENTION_FAULT = FaultInjectionV2(
    label="exact-B retention",
    pre_call=PreCallFaultV2(
        fault_id="pre.result.replicates.truncated-to-2",
        apply=_fault_exact_b_retention,
        assert_applied=_assert_exact_b_retention_fault_applied,
    ),
)
_REPLICATE_SLOTS_FAULT = FaultInjectionV2(
    label="replicate slots",
    pre_call=PreCallFaultV2(
        fault_id="pre.replicate-0.slots.missing-status",
        apply=_fault_replicate_slots,
        assert_applied=_assert_replicate_slots_fault_applied,
    ),
)
_REPLICATE_DIAGNOSTICS_FAULT = FaultInjectionV2(
    label="replicate diagnostics",
    pre_call=PreCallFaultV2(
        fault_id="pre.replicate-0.diagnostics.list",
        apply=_fault_replicate_diagnostics,
        assert_applied=_assert_replicate_diagnostics_fault_applied,
    ),
)
_REPLICATE_IDENTITY_AND_SEED_FAULT = FaultInjectionV2(
    label="replicate identity and seed",
    pre_call=PreCallFaultV2(
        fault_id="pre.replicate-0.seed.zero-digest",
        apply=_fault_replicate_identity_and_seed,
        assert_applied=_assert_replicate_identity_and_seed_fault_applied,
    ),
)
_REPLICATE_VECTOR_FAULT = FaultInjectionV2(
    label="replicate vector",
    pre_call=PreCallFaultV2(
        fault_id="pre.replicate-0.vector.list",
        apply=_fault_replicate_vector,
        assert_applied=_assert_replicate_vector_fault_applied,
    ),
)
_COMMON_SUPPORT_FAULT = FaultInjectionV2(
    label="common support",
    pre_call=PreCallFaultV2(
        fault_id="pre.replicate-0.support.3",
        apply=_fault_common_support,
        assert_applied=_assert_common_support_fault_applied,
    ),
)
_SEALED_TOKEN_VERIFICATION_FAULT = FaultInjectionV2(
    label="sealed token verification",
    before_real_token=BeforeRealTokenFaultV2(
        fault_id="phase.token.schema.invalid",
        call_index=1,
        apply=_fault_sealed_token_verification,
        assert_applied=_assert_sealed_token_verification_fault_applied,
    ),
)
_STATUS_STAGE_PAIRING_FAULT = FaultInjectionV2(
    label="status-stage pairing",
    pre_call=PreCallFaultV2(
        fault_id="pre.replicate-0.failure-stage.statistic-scan",
        apply=_fault_status_stage_pairing,
        assert_applied=_assert_status_stage_pairing_fault_applied,
    ),
)
_POST_TOKEN_SELECTION_REREAD_FAULT = FaultInjectionV2(
    label="post-token selection reread",
    after_token_return=AfterSuccessfulTokenReturnFaultV2(
        fault_id="phase.token.after-return.replicate-0-selection.missing-decision",
        call_index=1,
        apply=_fault_post_token_selection_reread,
        assert_applied=_assert_post_token_selection_reread_fault_applied,
    ),
)
_POST_TOKEN_DECISION_FAULT = FaultInjectionV2(
    label="post-token decision",
    after_token_return=AfterSuccessfulTokenReturnFaultV2(
        fault_id="phase.token.after-return.replicate-0-selection.decision-0.75",
        call_index=1,
        apply=_fault_post_token_decision,
        assert_applied=_assert_post_token_decision_fault_applied,
    ),
)
_ROW17_COMPANION_CLASSIFICATION = "DATAFLOW_ONLY_NOT_ERROR_BOUNDARY"
_E_F_ACCUMULATION_FAULT = FaultInjectionV2(
    label="E-F accumulation",
    after_token_return=AfterSuccessfulTokenReturnFaultV2(
        fault_id=(
            "phase.token.after-return.replicate-0-selection.accumulation-sentinel"
        ),
        call_index=1,
        apply=_fault_e_f_accumulation_probe,
        assert_applied=_assert_e_f_accumulation_probe_applied,
    ),
)
_EXACT_COUNT_TYPES_FAULT = FaultInjectionV2(
    label="exact count types",
    pre_call=PreCallFaultV2(
        fault_id="pre.result.exceedance-count.bool",
        apply=_fault_exact_count_types,
        assert_applied=_assert_exact_count_types_fault_applied,
    ),
)
_TERMINAL_ARITHMETIC_FAULT = FaultInjectionV2(
    label="terminal arithmetic",
    pre_call=PreCallFaultV2(
        fault_id="pre.result.p-value.0.25",
        apply=_fault_terminal_arithmetic,
        assert_applied=_assert_terminal_arithmetic_fault_applied,
    ),
)
_CONTEXT_READ_2_DRIFT_FAULT = FaultInjectionV2(
    label="context read #2 drift",
    context_phase=ContextPhaseFaultV2(
        fault_id="phase.context.read-2.tie-tolerance-0.25",
        read_index=2,
        apply=_fault_context_read_2_drift,
        assert_applied=_assert_context_read_2_drift_fault_applied,
    ),
)


_NULL_BIND_SEALED_SELECTION_CONTRACT_FAULT = FaultInjectionV2(
    label="sealed selection contract",
    context_phase=ContextPhaseFaultV2(
        fault_id="null-bind.context.read-1.selection-contract.empty-candidates",
        read_index=1,
        apply=_fault_sealed_selection_contract,
        assert_applied=_assert_sealed_selection_contract_fault_applied,
    ),
)
_NULL_BIND_EXACT_COUNT_TYPES_FAULT = FaultInjectionV2(
    label="exact count types",
    pre_call=PreCallFaultV2(
        fault_id="null-bind.pre.exceedance-count.bool",
        apply=_fault_null_bind_exact_count_types,
        assert_applied=_assert_null_bind_exact_count_types_fault_applied,
    ),
)
_NULL_BIND_TERMINAL_RECORD_FAULT = FaultInjectionV2(
    label="exact NOT_EVALUABLE-NULL_BIND terminal",
    pre_call=PreCallFaultV2(
        fault_id="null-bind.pre.terminal.p-value.0.25",
        apply=_fault_null_bind_terminal_record,
        assert_applied=_assert_null_bind_terminal_record_fault_applied,
    ),
)
_NULL_BIND_CONTEXT_READ_2_DRIFT_FAULT = FaultInjectionV2(
    label="context read #2 drift",
    context_phase=ContextPhaseFaultV2(
        fault_id="null-bind.context.read-2.tie-tolerance.0.25",
        read_index=2,
        apply=_fault_null_bind_context_read_2_drift,
        assert_applied=_assert_null_bind_context_read_2_drift_fault_applied,
    ),
)


_OBSERVED_SCAN_SEALED_SELECTION_CONTRACT_FAULT = FaultInjectionV2(
    label="sealed selection contract",
    context_phase=ContextPhaseFaultV2(
        fault_id="observed-scan.context.read-1.selection-contract.empty-candidates",
        read_index=1,
        apply=_fault_sealed_selection_contract,
        assert_applied=_assert_sealed_selection_contract_fault_applied,
    ),
)
_OBSERVED_SCAN_FAILURE_VECTOR_FAULT = FaultInjectionV2(
    label="observed analytic-failure vector",
    pre_call=PreCallFaultV2(
        fault_id="observed-scan.pre.failure-vector.analytic-estimate.0.25",
        apply=_fault_observed_scan_failure_vector,
        assert_applied=_assert_observed_scan_failure_vector_fault_applied,
    ),
)
_OBSERVED_SCAN_EXACT_COUNT_TYPES_FAULT = FaultInjectionV2(
    label="exact count types",
    pre_call=PreCallFaultV2(
        fault_id="observed-scan.pre.exceedance-count.bool",
        apply=_fault_observed_scan_exact_count_types,
        assert_applied=_assert_observed_scan_exact_count_types_fault_applied,
    ),
)
_OBSERVED_SCAN_TERMINAL_RECORD_FAULT = FaultInjectionV2(
    label="exact NOT_EVALUABLE-OBSERVED_STATISTIC_SCAN terminal",
    pre_call=PreCallFaultV2(
        fault_id="observed-scan.pre.terminal.p-value.0.25",
        apply=_fault_observed_scan_terminal_record,
        assert_applied=_assert_observed_scan_terminal_record_fault_applied,
    ),
)
_OBSERVED_SCAN_CONTEXT_READ_2_DRIFT_FAULT = FaultInjectionV2(
    label="context read #2 drift",
    context_phase=ContextPhaseFaultV2(
        fault_id="observed-scan.context.read-2.tie-tolerance.0.25",
        read_index=2,
        apply=_fault_observed_scan_context_read_2_drift,
        assert_applied=_assert_observed_scan_context_read_2_drift_fault_applied,
    ),
)


_REPLICATE_EXECUTION_EXACT_COUNT_TYPES_FAULT = FaultInjectionV2(
    label="exact count types",
    pre_call=PreCallFaultV2(
        fault_id="replicate-execution.pre.failure-count.bool",
        apply=_fault_replicate_execution_exact_count_types,
        assert_applied=(
            _assert_replicate_execution_exact_count_types_fault_applied
        ),
    ),
)
_REPLICATE_EXECUTION_TERMINAL_BOUNDS_FAULT = FaultInjectionV2(
    label="exact NOT_EVALUABLE-REPLICATE_EXECUTION bounds",
    pre_call=PreCallFaultV2(
        fault_id="replicate-execution.pre.bound-high.0.5",
        apply=_fault_replicate_execution_terminal_bounds,
        assert_applied=_assert_replicate_execution_terminal_bounds_fault_applied,
    ),
)
_REPLICATE_EXECUTION_CONTEXT_READ_2_DRIFT_FAULT = FaultInjectionV2(
    label="context read #2 drift",
    context_phase=ContextPhaseFaultV2(
        fault_id="replicate-execution.context.read-2.tie-tolerance.0.25",
        read_index=2,
        apply=_fault_replicate_execution_context_read_2_drift,
        assert_applied=(
            _assert_replicate_execution_context_read_2_drift_fault_applied
        ),
    ),
)


REPLICATE_EXECUTION_ADJACENT_LEDGER_DOUBLE_FAULTS: tuple[
    ReplicateExecutionAdjacentFaultScenarioV2,
    ...,
] = (
    ReplicateExecutionAdjacentFaultScenarioV2(
        row_id="R1-exact-count-types__replicate-execution-terminal-bounds",
        left=_REPLICATE_EXECUTION_EXACT_COUNT_TYPES_FAULT,
        right=_REPLICATE_EXECUTION_TERMINAL_BOUNDS_FAULT,
        double_expectation=_IntegrityErrorExpectationV2(
            message="calibration counts must be exact built-in integers",
            has_cause=False,
        ),
        right_expectation=_IntegrityErrorExpectationV2(
            message="replicate-execution bounds contradict E, F, and B",
            has_cause=False,
        ),
        double_receipts=(
            _FaultReceiptV2(
                phase=_FaultReceiptPhaseV2.PRE_CALL,
                fault_id="replicate-execution.pre.failure-count.bool",
                phase_index=0,
            ),
            _FaultReceiptV2(
                phase=_FaultReceiptPhaseV2.PRE_CALL,
                fault_id="replicate-execution.pre.bound-high.0.5",
                phase_index=0,
            ),
        ),
        right_receipts=(
            _FaultReceiptV2(
                phase=_FaultReceiptPhaseV2.PRE_CALL,
                fault_id="replicate-execution.pre.bound-high.0.5",
                phase_index=0,
            ),
        ),
        double_context_reads=1,
        right_context_reads=1,
        double_context_fault_calls=0,
        right_context_fault_calls=0,
        double_token_verifier_calls=3,
        right_token_verifier_calls=3,
        double_token_completion_calls=3,
        right_token_completion_calls=3,
    ),
    ReplicateExecutionAdjacentFaultScenarioV2(
        row_id="R2-replicate-execution-terminal-bounds__context-read-2-drift",
        left=_REPLICATE_EXECUTION_TERMINAL_BOUNDS_FAULT,
        right=_REPLICATE_EXECUTION_CONTEXT_READ_2_DRIFT_FAULT,
        double_expectation=_IntegrityErrorExpectationV2(
            message="replicate-execution bounds contradict E, F, and B",
            has_cause=False,
        ),
        right_expectation=_IntegrityErrorExpectationV2(
            message="scientific plan changed during result verification",
            has_cause=False,
        ),
        double_receipts=(
            _FaultReceiptV2(
                phase=_FaultReceiptPhaseV2.PRE_CALL,
                fault_id="replicate-execution.pre.bound-high.0.5",
                phase_index=0,
            ),
        ),
        right_receipts=(
            _FaultReceiptV2(
                phase=_FaultReceiptPhaseV2.CONTEXT_READ,
                fault_id=(
                    "replicate-execution.context.read-2.tie-tolerance.0.25"
                ),
                phase_index=2,
            ),
        ),
        double_context_reads=1,
        right_context_reads=2,
        double_context_fault_calls=0,
        right_context_fault_calls=1,
        double_token_verifier_calls=3,
        right_token_verifier_calls=3,
        double_token_completion_calls=3,
        right_token_completion_calls=3,
    ),
)


NULL_BIND_ADJACENT_LEDGER_DOUBLE_FAULTS: tuple[
    NullBindAdjacentFaultScenarioV2,
    ...,
] = (
    NullBindAdjacentFaultScenarioV2(
        row_id="N1-sealed-selection-contract__exact-count-types",
        left=_NULL_BIND_SEALED_SELECTION_CONTRACT_FAULT,
        right=_NULL_BIND_EXACT_COUNT_TYPES_FAULT,
        double_expectation=_IntegrityErrorExpectationV2(
            message="sealed plan selection contract is invalid",
            has_cause=False,
        ),
        right_expectation=_IntegrityErrorExpectationV2(
            message="calibration counts must be exact built-in integers",
            has_cause=False,
        ),
        double_receipts=(
            _FaultReceiptV2(
                phase=_FaultReceiptPhaseV2.PRE_CALL,
                fault_id="null-bind.pre.exceedance-count.bool",
                phase_index=0,
            ),
            _FaultReceiptV2(
                phase=_FaultReceiptPhaseV2.CONTEXT_READ,
                fault_id=(
                    "null-bind.context.read-1.selection-contract.empty-candidates"
                ),
                phase_index=1,
            ),
        ),
        right_receipts=(
            _FaultReceiptV2(
                phase=_FaultReceiptPhaseV2.PRE_CALL,
                fault_id="null-bind.pre.exceedance-count.bool",
                phase_index=0,
            ),
        ),
        double_context_reads=1,
        right_context_reads=1,
        double_context_fault_calls=1,
        right_context_fault_calls=0,
    ),
    NullBindAdjacentFaultScenarioV2(
        row_id="N2-exact-count-types__exact-null-bind-terminal",
        left=_NULL_BIND_EXACT_COUNT_TYPES_FAULT,
        right=_NULL_BIND_TERMINAL_RECORD_FAULT,
        double_expectation=_IntegrityErrorExpectationV2(
            message="calibration counts must be exact built-in integers",
            has_cause=False,
        ),
        right_expectation=_IntegrityErrorExpectationV2(
            message="NULL_BIND terminal result is invalid",
            has_cause=False,
        ),
        double_receipts=(
            _FaultReceiptV2(
                phase=_FaultReceiptPhaseV2.PRE_CALL,
                fault_id="null-bind.pre.exceedance-count.bool",
                phase_index=0,
            ),
            _FaultReceiptV2(
                phase=_FaultReceiptPhaseV2.PRE_CALL,
                fault_id="null-bind.pre.terminal.p-value.0.25",
                phase_index=0,
            ),
        ),
        right_receipts=(
            _FaultReceiptV2(
                phase=_FaultReceiptPhaseV2.PRE_CALL,
                fault_id="null-bind.pre.terminal.p-value.0.25",
                phase_index=0,
            ),
        ),
        double_context_reads=1,
        right_context_reads=1,
        double_context_fault_calls=0,
        right_context_fault_calls=0,
    ),
    NullBindAdjacentFaultScenarioV2(
        row_id="N3-exact-null-bind-terminal__context-read-2-drift",
        left=_NULL_BIND_TERMINAL_RECORD_FAULT,
        right=_NULL_BIND_CONTEXT_READ_2_DRIFT_FAULT,
        double_expectation=_IntegrityErrorExpectationV2(
            message="NULL_BIND terminal result is invalid",
            has_cause=False,
        ),
        right_expectation=_IntegrityErrorExpectationV2(
            message="scientific plan changed during result verification",
            has_cause=False,
        ),
        double_receipts=(
            _FaultReceiptV2(
                phase=_FaultReceiptPhaseV2.PRE_CALL,
                fault_id="null-bind.pre.terminal.p-value.0.25",
                phase_index=0,
            ),
        ),
        right_receipts=(
            _FaultReceiptV2(
                phase=_FaultReceiptPhaseV2.CONTEXT_READ,
                fault_id="null-bind.context.read-2.tie-tolerance.0.25",
                phase_index=2,
            ),
        ),
        double_context_reads=1,
        right_context_reads=2,
        double_context_fault_calls=0,
        right_context_fault_calls=1,
    ),
)


OBSERVED_SCAN_ADJACENT_LEDGER_DOUBLE_FAULTS: tuple[
    ObservedScanAdjacentFaultScenarioV2,
    ...,
] = (
    ObservedScanAdjacentFaultScenarioV2(
        row_id="O1-sealed-selection-contract__observed-failure-vector",
        left=_OBSERVED_SCAN_SEALED_SELECTION_CONTRACT_FAULT,
        right=_OBSERVED_SCAN_FAILURE_VECTOR_FAULT,
        double_expectation=_IntegrityErrorExpectationV2(
            message="sealed plan selection contract is invalid",
            has_cause=False,
        ),
        right_expectation=_IntegrityErrorExpectationV2(
            message="observed result vector analytical failure must remain raw",
            has_cause=False,
        ),
        double_receipts=(
            _FaultReceiptV2(
                phase=_FaultReceiptPhaseV2.PRE_CALL,
                fault_id="observed-scan.pre.failure-vector.analytic-estimate.0.25",
                phase_index=0,
            ),
            _FaultReceiptV2(
                phase=_FaultReceiptPhaseV2.CONTEXT_READ,
                fault_id="observed-scan.context.read-1.selection-contract.empty-candidates",
                phase_index=1,
            ),
        ),
        right_receipts=(
            _FaultReceiptV2(
                phase=_FaultReceiptPhaseV2.PRE_CALL,
                fault_id="observed-scan.pre.failure-vector.analytic-estimate.0.25",
                phase_index=0,
            ),
        ),
        double_context_reads=1,
        right_context_reads=1,
        double_context_fault_calls=1,
        right_context_fault_calls=0,
    ),
    ObservedScanAdjacentFaultScenarioV2(
        row_id="O2-observed-failure-vector__exact-count-types",
        left=_OBSERVED_SCAN_FAILURE_VECTOR_FAULT,
        right=_OBSERVED_SCAN_EXACT_COUNT_TYPES_FAULT,
        double_expectation=_IntegrityErrorExpectationV2(
            message="observed result vector analytical failure must remain raw",
            has_cause=False,
        ),
        right_expectation=_IntegrityErrorExpectationV2(
            message="calibration counts must be exact built-in integers",
            has_cause=False,
        ),
        double_receipts=(
            _FaultReceiptV2(
                phase=_FaultReceiptPhaseV2.PRE_CALL,
                fault_id="observed-scan.pre.failure-vector.analytic-estimate.0.25",
                phase_index=0,
            ),
            _FaultReceiptV2(
                phase=_FaultReceiptPhaseV2.PRE_CALL,
                fault_id="observed-scan.pre.exceedance-count.bool",
                phase_index=0,
            ),
        ),
        right_receipts=(
            _FaultReceiptV2(
                phase=_FaultReceiptPhaseV2.PRE_CALL,
                fault_id="observed-scan.pre.exceedance-count.bool",
                phase_index=0,
            ),
        ),
        double_context_reads=1,
        right_context_reads=1,
        double_context_fault_calls=0,
        right_context_fault_calls=0,
    ),
    ObservedScanAdjacentFaultScenarioV2(
        row_id="O3-exact-count-types__exact-observed-scan-terminal",
        left=_OBSERVED_SCAN_EXACT_COUNT_TYPES_FAULT,
        right=_OBSERVED_SCAN_TERMINAL_RECORD_FAULT,
        double_expectation=_IntegrityErrorExpectationV2(
            message="calibration counts must be exact built-in integers",
            has_cause=False,
        ),
        right_expectation=_IntegrityErrorExpectationV2(
            message="OBSERVED_STATISTIC_SCAN terminal result is invalid",
            has_cause=False,
        ),
        double_receipts=(
            _FaultReceiptV2(
                phase=_FaultReceiptPhaseV2.PRE_CALL,
                fault_id="observed-scan.pre.exceedance-count.bool",
                phase_index=0,
            ),
            _FaultReceiptV2(
                phase=_FaultReceiptPhaseV2.PRE_CALL,
                fault_id="observed-scan.pre.terminal.p-value.0.25",
                phase_index=0,
            ),
        ),
        right_receipts=(
            _FaultReceiptV2(
                phase=_FaultReceiptPhaseV2.PRE_CALL,
                fault_id="observed-scan.pre.terminal.p-value.0.25",
                phase_index=0,
            ),
        ),
        double_context_reads=1,
        right_context_reads=1,
        double_context_fault_calls=0,
        right_context_fault_calls=0,
    ),
    ObservedScanAdjacentFaultScenarioV2(
        row_id="O4-exact-observed-scan-terminal__context-read-2-drift",
        left=_OBSERVED_SCAN_TERMINAL_RECORD_FAULT,
        right=_OBSERVED_SCAN_CONTEXT_READ_2_DRIFT_FAULT,
        double_expectation=_IntegrityErrorExpectationV2(
            message="OBSERVED_STATISTIC_SCAN terminal result is invalid",
            has_cause=False,
        ),
        right_expectation=_IntegrityErrorExpectationV2(
            message="scientific plan changed during result verification",
            has_cause=False,
        ),
        double_receipts=(
            _FaultReceiptV2(
                phase=_FaultReceiptPhaseV2.PRE_CALL,
                fault_id="observed-scan.pre.terminal.p-value.0.25",
                phase_index=0,
            ),
        ),
        right_receipts=(
            _FaultReceiptV2(
                phase=_FaultReceiptPhaseV2.CONTEXT_READ,
                fault_id="observed-scan.context.read-2.tie-tolerance.0.25",
                phase_index=2,
            ),
        ),
        double_context_reads=1,
        right_context_reads=2,
        double_context_fault_calls=0,
        right_context_fault_calls=1,
    ),
)


ADJACENT_LEDGER_DOUBLE_FAULTS: tuple[AdjacentFaultScenarioV2, ...] = (
    AdjacentFaultScenarioV2(
        row_number=1,
        row_id="row01-context-read-1__result-slots",
        left=_CONTEXT_READ_1_FAULT,
        right=_RESULT_SLOTS_FAULT,
        message="sealed resolution context is invalid",
        has_cause=False,
        right_message="calibration result slot snapshot is invalid",
        right_has_cause=True,
        double_pre_call_fault_ids=("pre.result.slots.missing-status",),
        right_pre_call_fault_ids=("pre.result.slots.missing-status",),
        double_context_fault_ids=("phase.context.read-1.invalid-shape",),
        double_context_fault_calls=1,
    ),
    AdjacentFaultScenarioV2(
        row_number=2,
        row_id="row02-result-slots__result-diagnostics",
        left=_RESULT_SLOTS_FAULT,
        right=_RESULT_DIAGNOSTICS_FAULT,
        message="calibration result slot snapshot is invalid",
        has_cause=True,
        right_message=(
            "calibration result diagnostics must be an exact string tuple"
        ),
        right_has_cause=False,
        double_pre_call_fault_ids=(
            "pre.result.slots.missing-status",
            "pre.result.diagnostics.list",
        ),
        right_pre_call_fault_ids=("pre.result.diagnostics.list",),
    ),
    AdjacentFaultScenarioV2(
        row_number=3,
        row_id="row03-result-diagnostics__exact-replicate-tuple",
        left=_RESULT_DIAGNOSTICS_FAULT,
        right=_EXACT_REPLICATE_TUPLE_FAULT,
        message="calibration result diagnostics must be an exact string tuple",
        has_cause=False,
        right_message="calibration replicates must be an exact tuple",
        right_has_cause=False,
        double_pre_call_fault_ids=(
            "pre.result.diagnostics.list",
            "pre.result.replicates.list",
        ),
        right_pre_call_fault_ids=("pre.result.replicates.list",),
    ),
    AdjacentFaultScenarioV2(
        row_number=4,
        row_id="row04-exact-replicate-tuple__semantic-digest",
        left=_EXACT_REPLICATE_TUPLE_FAULT,
        right=_SEMANTIC_DIGEST_FAULT,
        message="calibration replicates must be an exact tuple",
        has_cause=False,
        right_message=(
            "semantic_input_sha256 must be a lowercase SHA-256 hex digest"
        ),
        right_has_cause=False,
        double_pre_call_fault_ids=(
            "pre.result.replicates.list",
            "pre.result.semantic-digest.invalid",
        ),
        right_pre_call_fault_ids=("pre.result.semantic-digest.invalid",),
    ),
    AdjacentFaultScenarioV2(
        row_number=5,
        row_id="row05-semantic-digest__plan-digest",
        left=_SEMANTIC_DIGEST_FAULT,
        right=_PLAN_DIGEST_FAULT,
        message="semantic_input_sha256 must be a lowercase SHA-256 hex digest",
        has_cause=False,
        right_message=(
            "scientific_plan_sha256 must be a lowercase SHA-256 hex digest"
        ),
        right_has_cause=False,
        double_pre_call_fault_ids=(
            "pre.result.semantic-digest.invalid",
            "pre.result.plan-digest.invalid",
        ),
        right_pre_call_fault_ids=("pre.result.plan-digest.invalid",),
    ),
    AdjacentFaultScenarioV2(
        row_number=6,
        row_id="row06-plan-digest__result-scalars",
        left=_PLAN_DIGEST_FAULT,
        right=_RESULT_SCALARS_FAULT,
        message="scientific_plan_sha256 must be a lowercase SHA-256 hex digest",
        has_cause=False,
        right_message="calibration result scalars contradict the sealed plan",
        right_has_cause=False,
        double_pre_call_fault_ids=(
            "pre.result.plan-digest.invalid",
            "pre.result.planned-replicates.bool",
        ),
        right_pre_call_fault_ids=("pre.result.planned-replicates.bool",),
    ),
    AdjacentFaultScenarioV2(
        row_number=7,
        row_id="row07-result-scalars__sealed-selection-contract",
        left=_RESULT_SCALARS_FAULT,
        right=_SEALED_SELECTION_CONTRACT_FAULT,
        message="calibration result scalars contradict the sealed plan",
        has_cause=False,
        right_message="sealed plan selection contract is invalid",
        right_has_cause=False,
        double_pre_call_fault_ids=(
            "pre.result.planned-replicates.bool",
        ),
        double_context_fault_ids=(
            "phase.context.selection-contract.empty-candidates",
        ),
        right_context_fault_ids=(
            "phase.context.selection-contract.empty-candidates",
        ),
        double_context_fault_calls=1,
        right_context_fault_calls=1,
    ),
    AdjacentFaultScenarioV2(
        row_number=8,
        row_id="row08-observed-vector-structure__observed-selection",
        left=_OBSERVED_VECTOR_STRUCTURE_FAULT,
        right=_OBSERVED_SELECTION_FAULT,
        message="observed result vector must be an exact tuple",
        has_cause=False,
        right_message=(
            "observed result vector selection must have its exact canonical type"
        ),
        right_has_cause=False,
        double_pre_call_fault_ids=(
            "pre.result.observed-vector.list",
            "pre.result.observed-selection.foreign-object",
        ),
        right_pre_call_fault_ids=(
            "pre.result.observed-selection.foreign-object",
        ),
    ),
    AdjacentFaultScenarioV2(
        row_number=9,
        row_id="row09-observed-selection-decision__exact-b-retention",
        left=_OBSERVED_SELECTION_DECISION_FAULT,
        right=_EXACT_B_RETENTION_FAULT,
        message=(
            "observed result vector selection contradicts central reselection"
        ),
        has_cause=False,
        right_message="calibration result does not retain exact B outcomes",
        right_has_cause=False,
        double_pre_call_fault_ids=(
            "pre.result.observed-selection.decision-2",
            "pre.result.replicates.truncated-to-2",
        ),
        right_pre_call_fault_ids=(
            "pre.result.replicates.truncated-to-2",
        ),
    ),
    AdjacentFaultScenarioV2(
        row_number=10,
        row_id="row10-replicate-slots__replicate-diagnostics",
        left=_REPLICATE_SLOTS_FAULT,
        right=_REPLICATE_DIAGNOSTICS_FAULT,
        message="replicate 0 slot snapshot is invalid",
        has_cause=True,
        right_message="replicate 0 diagnostics must be an exact string tuple",
        right_has_cause=False,
        double_pre_call_fault_ids=(
            "pre.replicate-0.slots.missing-status",
            "pre.replicate-0.diagnostics.list",
        ),
        right_pre_call_fault_ids=(
            "pre.replicate-0.diagnostics.list",
        ),
    ),
    AdjacentFaultScenarioV2(
        row_number=11,
        row_id="row11-replicate-diagnostics__replicate-identity-and-seed",
        left=_REPLICATE_DIAGNOSTICS_FAULT,
        right=_REPLICATE_IDENTITY_AND_SEED_FAULT,
        message="replicate 0 diagnostics must be an exact string tuple",
        has_cause=False,
        right_message="replicate 0 identity drifted",
        right_has_cause=False,
        double_pre_call_fault_ids=(
            "pre.replicate-0.diagnostics.list",
            "pre.replicate-0.seed.zero-digest",
        ),
        right_pre_call_fault_ids=(
            "pre.replicate-0.seed.zero-digest",
        ),
    ),
    AdjacentFaultScenarioV2(
        row_number=12,
        row_id="row12-replicate-identity__replicate-vector",
        left=_REPLICATE_IDENTITY_AND_SEED_FAULT,
        right=_REPLICATE_VECTOR_FAULT,
        message="replicate 0 identity drifted",
        has_cause=False,
        right_message="replicate 0 result vector must be an exact tuple",
        right_has_cause=False,
        double_pre_call_fault_ids=(
            "pre.replicate-0.seed.zero-digest",
            "pre.replicate-0.vector.list",
        ),
        right_pre_call_fault_ids=("pre.replicate-0.vector.list",),
    ),
    AdjacentFaultScenarioV2(
        row_number=13,
        row_id="row13-replicate-vector__common-support",
        left=_REPLICATE_VECTOR_FAULT,
        right=_COMMON_SUPPORT_FAULT,
        message="replicate 0 result vector must be an exact tuple",
        has_cause=False,
        right_message="all result vectors must use one common support",
        right_has_cause=False,
        double_pre_call_fault_ids=(
            "pre.replicate-0.vector.list",
            "pre.replicate-0.support.3",
        ),
        right_pre_call_fault_ids=("pre.replicate-0.support.3",),
    ),
    AdjacentFaultScenarioV2(
        row_number=14,
        row_id="row14-common-support__sealed-token-verification",
        left=_COMMON_SUPPORT_FAULT,
        right=_SEALED_TOKEN_VERIFICATION_FAULT,
        message="all result vectors must use one common support",
        has_cause=False,
        right_message=(
            "result token ownership contradicts the sealed resolution"
        ),
        right_has_cause=False,
        double_pre_call_fault_ids=("pre.replicate-0.support.3",),
        right_token_fault_ids=("phase.token.schema.invalid",),
        right_token_verifier_calls=1,
        right_token_fault_calls=1,
    ),
    AdjacentFaultScenarioV2(
        row_number=15,
        row_id="row15-sealed-token-verification__status-stage-pairing",
        left=_SEALED_TOKEN_VERIFICATION_FAULT,
        right=_STATUS_STAGE_PAIRING_FAULT,
        message="result token ownership contradicts the sealed resolution",
        has_cause=False,
        right_message="complete replicate has a failure stage",
        right_has_cause=False,
        double_pre_call_fault_ids=(
            "pre.replicate-0.failure-stage.statistic-scan",
        ),
        right_pre_call_fault_ids=(
            "pre.replicate-0.failure-stage.statistic-scan",
        ),
        double_token_fault_ids=("phase.token.schema.invalid",),
        double_token_verifier_calls=1,
        right_token_verifier_calls=1,
        double_token_fault_calls=1,
        right_token_completion_ids=("phase.token.real-return.call-1",),
        right_token_completion_calls=1,
    ),
    AdjacentFaultScenarioV2(
        row_number=16,
        row_id="row16-status-stage-pairing__post-token-selection-reread",
        left=_STATUS_STAGE_PAIRING_FAULT,
        right=_POST_TOKEN_SELECTION_REREAD_FAULT,
        message="complete replicate has a failure stage",
        has_cause=False,
        right_message="replicate 0 selection slot snapshot is invalid",
        right_has_cause=True,
        double_pre_call_fault_ids=(
            "pre.replicate-0.failure-stage.statistic-scan",
        ),
        double_after_token_fault_ids=(
            "phase.token.after-return.replicate-0-selection.missing-decision",
        ),
        right_after_token_fault_ids=(
            "phase.token.after-return.replicate-0-selection.missing-decision",
        ),
        double_token_verifier_calls=1,
        right_token_verifier_calls=1,
        double_token_completion_ids=("phase.token.real-return.call-1",),
        right_token_completion_ids=("phase.token.real-return.call-1",),
        double_token_completion_calls=1,
        right_token_completion_calls=1,
        double_after_token_fault_calls=1,
        right_after_token_fault_calls=1,
    ),
    AdjacentFaultScenarioV2(
        row_number=18,
        row_id="row18-e-f-accumulation__exact-count-types",
        left=_E_F_ACCUMULATION_FAULT,
        right=_EXACT_COUNT_TYPES_FAULT,
        message="TEST-ONLY E-F accumulation phase probe",
        has_cause=False,
        right_message="calibration counts must be exact built-in integers",
        right_has_cause=False,
        double_pre_call_fault_ids=("pre.result.exceedance-count.bool",),
        right_pre_call_fault_ids=("pre.result.exceedance-count.bool",),
        double_token_completion_ids=("phase.token.real-return.call-1",),
        right_token_completion_ids=(
            "phase.token.real-return.call-1",
            "phase.token.real-return.call-2",
            "phase.token.real-return.call-3",
        ),
        double_after_token_fault_ids=(
            "phase.token.after-return.replicate-0-selection.accumulation-sentinel",
        ),
        double_token_verifier_calls=1,
        right_token_verifier_calls=3,
        double_token_completion_calls=1,
        right_token_completion_calls=3,
        double_after_token_fault_calls=1,
    ),
    AdjacentFaultScenarioV2(
        row_number=19,
        row_id="row19-exact-count-types__terminal-arithmetic",
        left=_EXACT_COUNT_TYPES_FAULT,
        right=_TERMINAL_ARITHMETIC_FAULT,
        message="calibration counts must be exact built-in integers",
        has_cause=False,
        right_message="COMPLETE result decision arithmetic is invalid",
        right_has_cause=False,
        double_pre_call_fault_ids=(
            "pre.result.exceedance-count.bool",
            "pre.result.p-value.0.25",
        ),
        right_pre_call_fault_ids=("pre.result.p-value.0.25",),
        double_token_verifier_calls=3,
        right_token_verifier_calls=3,
        double_token_completion_ids=(
            "phase.token.real-return.call-1",
            "phase.token.real-return.call-2",
            "phase.token.real-return.call-3",
        ),
        right_token_completion_ids=(
            "phase.token.real-return.call-1",
            "phase.token.real-return.call-2",
            "phase.token.real-return.call-3",
        ),
        double_token_completion_calls=3,
        right_token_completion_calls=3,
    ),
    AdjacentFaultScenarioV2(
        row_number=20,
        row_id="row20-terminal-arithmetic__context-read-2-drift",
        left=_TERMINAL_ARITHMETIC_FAULT,
        right=_CONTEXT_READ_2_DRIFT_FAULT,
        message="COMPLETE result decision arithmetic is invalid",
        has_cause=False,
        right_message="scientific plan changed during result verification",
        right_has_cause=False,
        double_pre_call_fault_ids=("pre.result.p-value.0.25",),
        right_context_fault_ids=(
            "phase.context.read-2.tie-tolerance-0.25",
        ),
        right_context_fault_calls=1,
        double_token_verifier_calls=3,
        right_token_verifier_calls=3,
        double_token_completion_ids=(
            "phase.token.real-return.call-1",
            "phase.token.real-return.call-2",
            "phase.token.real-return.call-3",
        ),
        right_token_completion_ids=(
            "phase.token.real-return.call-1",
            "phase.token.real-return.call-2",
            "phase.token.real-return.call-3",
        ),
        double_token_completion_calls=3,
        right_token_completion_calls=3,
        right_context_reads=2,
    ),
)


_EXPECTED_ADJACENT_ERROR_ROWS = (
    (1, "context read #1", "result slots"),
    (2, "result slots", "result diagnostics"),
    (3, "result diagnostics", "exact replicate tuple"),
    (4, "exact replicate tuple", "semantic digest"),
    (5, "semantic digest", "plan digest"),
    (6, "plan digest", "result scalars"),
    (7, "result scalars", "sealed selection contract"),
    (8, "observed vector structure", "observed selection"),
    (9, "observed selection decision", "exact-B retention"),
    (10, "replicate slots", "replicate diagnostics"),
    (11, "replicate diagnostics", "replicate identity and seed"),
    (12, "replicate identity and seed", "replicate vector"),
    (13, "replicate vector", "common support"),
    (14, "common support", "sealed token verification"),
    (15, "sealed token verification", "status-stage pairing"),
    (16, "status-stage pairing", "post-token selection reread"),
    (18, "E-F accumulation", "exact count types"),
    (19, "exact count types", "terminal arithmetic"),
    (20, "terminal arithmetic", "context read #2 drift"),
)


def test_adjacent_fault_table_covers_exact_error_rows_01_to_16_and_18_to_20() -> None:
    assert tuple(
        (scenario.row_number, scenario.left.label, scenario.right.label)
        for scenario in ADJACENT_LEDGER_DOUBLE_FAULTS
    ) == _EXPECTED_ADJACENT_ERROR_ROWS
    assert tuple(
        scenario.row_number for scenario in ADJACENT_LEDGER_DOUBLE_FAULTS
    ) == (*range(1, 17), *range(18, 21))
    assert len({scenario.row_id for scenario in ADJACENT_LEDGER_DOUBLE_FAULTS}) == 19
    assert all(
        scenario.row_id.startswith(f"row{scenario.row_number:02d}-")
        for scenario in ADJACENT_LEDGER_DOUBLE_FAULTS
    )


@pytest.mark.parametrize(
    "scenario",
    ADJACENT_LEDGER_DOUBLE_FAULTS,
    ids=lambda scenario: scenario.row_id,
)
def test_adjacent_error_rows_01_to_16_and_18_to_20_retain_exact_priority(
    scenario: AdjacentFaultScenarioV2,
) -> None:
    result, resolution = _adjacent_fault_complete_case()
    injections = (scenario.left, scenario.right)
    probe = _PhaseProbeV2()
    _apply_before_call_faults(result, injections, probe)
    assert tuple(probe.pre_call_fault_ids) == scenario.double_pre_call_fault_ids
    assert probe.context_fault_ids == []
    assert probe.token_fault_ids == []
    assert probe.token_completion_ids == []
    assert probe.after_token_fault_ids == []
    verifier = _verifier_with_phase_profile(result, injections, probe)

    with pytest.raises(V2IntegrityError) as captured:
        verifier(result, resolution)

    assert type(captured.value) is V2IntegrityError
    assert str(captured.value) == scenario.message
    assert (captured.value.__cause__ is not None) is scenario.has_cause
    assert probe.context_reads == scenario.double_context_reads
    assert probe.context_fault_calls == scenario.double_context_fault_calls
    assert tuple(probe.context_fault_ids) == scenario.double_context_fault_ids
    assert probe.token_verifier_calls == scenario.double_token_verifier_calls
    assert probe.token_fault_calls == scenario.double_token_fault_calls
    assert tuple(probe.token_fault_ids) == scenario.double_token_fault_ids
    assert (
        probe.token_completion_calls == scenario.double_token_completion_calls
    )
    assert tuple(probe.token_completion_ids) == scenario.double_token_completion_ids
    assert (
        probe.after_token_fault_calls == scenario.double_after_token_fault_calls
    )
    assert tuple(probe.after_token_fault_ids) == (
        scenario.double_after_token_fault_ids
    )


def _row17_decision_snapshot(
    result: CalibrationResult,
) -> tuple[float, tuple[float, float, float]]:
    observed_selection = result.observed_selection
    assert type(observed_selection) is SelectionResult
    observed_decision: object = observed_selection.decision_statistic
    assert type(observed_decision) is float
    decisions: list[float] = []
    for outcome in result.replicates:
        selection = outcome.selection
        assert type(selection) is SelectionResult
        decision: object = selection.decision_statistic
        assert type(decision) is float
        decisions.append(decision)
    assert len(decisions) == 3
    return observed_decision, (decisions[0], decisions[1], decisions[2])


def _assert_row17_three_successful_token_returns(probe: _PhaseProbeV2) -> None:
    assert probe.token_verifier_calls == 3
    assert probe.token_completion_calls == 3
    assert probe.token_completion_ids == [
        "phase.token.real-return.call-1",
        "phase.token.real-return.call-2",
        "phase.token.real-return.call-3",
    ]
    assert probe.token_fault_calls == 0
    assert probe.token_fault_ids == []


def _assert_row17_call_1_after_return_mutation(probe: _PhaseProbeV2) -> None:
    assert probe.after_token_fault_calls == 1
    assert probe.after_token_fault_ids == [
        "phase.token.after-return.replicate-0-selection.decision-0.75"
    ]


def test_row17_dataflow_only_baseline_literal_is_valid() -> None:
    assert _ROW17_COMPANION_CLASSIFICATION == "DATAFLOW_ONLY_NOT_ERROR_BOUNDARY"
    assert all(scenario.row_number != 17 for scenario in ADJACENT_LEDGER_DOUBLE_FAULTS)
    result, resolution = _row17_dataflow_literal_case()
    observed_decision, decisions = _row17_decision_snapshot(result)
    assert observed_decision == 0.75
    assert decisions == (0.5, 0.8, 0.2)
    derived_e = sum(decision >= observed_decision for decision in decisions)
    assert derived_e == 1
    expected_p = (1 + derived_e) / 4
    assert expected_p == 0.5
    assert type(result.alpha) is float
    assert result.alpha == 0.05
    assert type(result.exceedance_count) is int
    assert result.exceedance_count == 1
    assert type(result.p_value) is float
    assert result.p_value == 0.5
    assert type(result.reject_null) is bool
    assert result.reject_null is False
    probe = _PhaseProbeV2()
    verifier = _verifier_with_phase_profile(result, (), probe)

    verifier(result, resolution)

    assert probe.context_reads == 2
    _assert_row17_three_successful_token_returns(probe)
    assert probe.after_token_fault_calls == 0
    assert probe.after_token_fault_ids == []


def test_row17_dataflow_only_stale_terminal_is_rejected_after_mutation() -> None:
    result, resolution = _row17_dataflow_literal_case()
    injections = (_POST_TOKEN_DECISION_FAULT,)
    probe = _PhaseProbeV2()
    verifier = _verifier_with_phase_profile(result, injections, probe)

    with pytest.raises(V2IntegrityError) as captured:
        verifier(result, resolution)

    assert type(captured.value) is V2IntegrityError
    assert str(captured.value) == "COMPLETE result decision arithmetic is invalid"
    assert captured.value.__cause__ is None
    observed_decision, decisions = _row17_decision_snapshot(result)
    assert observed_decision == 0.75
    assert decisions == (0.75, 0.8, 0.2)
    derived_e = sum(decision >= observed_decision for decision in decisions)
    assert derived_e == 2
    expected_p = (1 + derived_e) / 4
    assert expected_p == 0.75
    assert type(result.exceedance_count) is int
    assert result.exceedance_count == 1
    assert type(result.p_value) is float
    assert result.p_value == 0.5
    assert type(result.reject_null) is bool
    assert result.reject_null is False
    assert probe.context_reads == 1
    _assert_row17_three_successful_token_returns(probe)
    _assert_row17_call_1_after_return_mutation(probe)


def test_row17_dataflow_only_corrected_terminal_is_valid_after_mutation() -> None:
    result, resolution = _row17_dataflow_literal_case()
    observed_decision, baseline_decisions = _row17_decision_snapshot(result)
    assert observed_decision == 0.75
    assert baseline_decisions == (0.5, 0.8, 0.2)
    post_mutation_decisions = (0.75, 0.8, 0.2)
    derived_e = sum(
        decision >= observed_decision for decision in post_mutation_decisions
    )
    assert derived_e == 2
    expected_p = (1 + derived_e) / 4
    assert expected_p == 0.75
    assert (expected_p <= 0.05) is False
    object.__setattr__(result, "exceedance_count", 2)
    object.__setattr__(result, "p_value", 0.75)
    object.__setattr__(result, "reject_null", False)
    assert type(result.exceedance_count) is int
    assert result.exceedance_count == 2
    assert type(result.p_value) is float
    assert result.p_value == 0.75
    assert type(result.reject_null) is bool
    assert result.reject_null is False
    assert type(result.alpha) is float
    assert result.alpha == 0.05
    injections = (_POST_TOKEN_DECISION_FAULT,)
    probe = _PhaseProbeV2()
    verifier = _verifier_with_phase_profile(result, injections, probe)

    verifier(result, resolution)

    observed_decision, decisions = _row17_decision_snapshot(result)
    assert observed_decision == 0.75
    assert decisions == (0.75, 0.8, 0.2)
    assert probe.context_reads == 2
    _assert_row17_three_successful_token_returns(probe)
    _assert_row17_call_1_after_return_mutation(probe)


def test_row18_test_only_accumulation_probe_precedes_native_count_fault() -> None:
    scenario = ADJACENT_LEDGER_DOUBLE_FAULTS[16]
    assert scenario.row_number == 18
    result, resolution = _adjacent_fault_complete_case()
    injections = (scenario.left, scenario.right)
    probe = _PhaseProbeV2()
    _apply_before_call_faults(result, injections, probe)
    assert probe.pre_call_fault_ids == ["pre.result.exceedance-count.bool"]
    assert type(result.exceedance_count) is bool
    assert result.exceedance_count is True
    verifier = _verifier_with_phase_profile(result, injections, probe)

    with pytest.raises(V2IntegrityError) as captured:
        verifier(result, resolution)

    # The exact error below is emitted by the test-only propagated phase probe.
    # It is not a native/public/scientific SelCal error-contract assertion.
    assert type(captured.value) is V2IntegrityError
    assert str(captured.value) == "TEST-ONLY E-F accumulation phase probe"
    assert captured.value.__cause__ is None
    assert probe.context_reads == 1
    assert probe.token_verifier_calls == 1
    assert probe.token_completion_calls == 1
    assert probe.token_completion_ids == ["phase.token.real-return.call-1"]
    assert probe.token_fault_calls == 0
    assert probe.token_fault_ids == []
    assert probe.after_token_fault_calls == 1
    assert probe.after_token_fault_ids == [
        "phase.token.after-return.replicate-0-selection.accumulation-sentinel"
    ]
    operand: object = _first_adjacent_selection(result).decision_statistic
    assert type(operand) is _Row18AccumulationComparisonOperand
    assert operand.comparison_calls == 1
    assert operand.addition_calls == 1
    assert type(result.exceedance_count) is bool
    assert result.exceedance_count is True


def test_row18_after_return_probe_does_not_run_when_real_token_raises() -> None:
    result, resolution = _adjacent_fault_complete_case()
    injections = (_SEALED_TOKEN_VERIFICATION_FAULT, _E_F_ACCUMULATION_FAULT)
    probe = _PhaseProbeV2()
    verifier = _verifier_with_phase_profile(result, injections, probe)

    with pytest.raises(V2IntegrityError) as captured:
        verifier(result, resolution)

    assert type(captured.value) is V2IntegrityError
    assert str(captured.value) == (
        "result token ownership contradicts the sealed resolution"
    )
    assert captured.value.__cause__ is None
    assert probe.context_reads == 1
    assert probe.token_verifier_calls == 1
    assert probe.token_fault_calls == 1
    assert probe.token_fault_ids == ["phase.token.schema.invalid"]
    assert probe.token_completion_calls == 0
    assert probe.token_completion_ids == []
    assert probe.after_token_fault_calls == 0
    assert probe.after_token_fault_ids == []
    decision: object = _first_adjacent_selection(result).decision_statistic
    assert type(decision) is float
    assert decision == 4.0


@pytest.mark.parametrize(
    "scenario",
    ADJACENT_LEDGER_DOUBLE_FAULTS,
    ids=lambda scenario: scenario.row_id,
)
def test_adjacent_error_rows_01_to_16_and_18_to_20_right_only_is_live(
    scenario: AdjacentFaultScenarioV2,
) -> None:
    result, resolution = _adjacent_fault_complete_case()
    injections = (scenario.right,)
    probe = _PhaseProbeV2()
    _apply_before_call_faults(result, injections, probe)
    assert tuple(probe.pre_call_fault_ids) == scenario.right_pre_call_fault_ids
    assert probe.context_fault_ids == []
    assert probe.token_fault_ids == []
    assert probe.token_completion_ids == []
    assert probe.after_token_fault_ids == []
    verifier = _verifier_with_phase_profile(result, injections, probe)

    with pytest.raises(V2IntegrityError) as captured:
        verifier(result, resolution)

    assert type(captured.value) is V2IntegrityError
    assert str(captured.value) == scenario.right_message
    assert (captured.value.__cause__ is not None) is scenario.right_has_cause
    assert probe.context_reads == scenario.right_context_reads
    assert probe.context_fault_calls == scenario.right_context_fault_calls
    assert tuple(probe.context_fault_ids) == scenario.right_context_fault_ids
    assert probe.token_verifier_calls == scenario.right_token_verifier_calls
    assert probe.token_fault_calls == scenario.right_token_fault_calls
    assert tuple(probe.token_fault_ids) == scenario.right_token_fault_ids
    assert probe.token_completion_calls == scenario.right_token_completion_calls
    assert tuple(probe.token_completion_ids) == scenario.right_token_completion_ids
    assert (
        probe.after_token_fault_calls == scenario.right_after_token_fault_calls
    )
    assert tuple(probe.after_token_fault_ids) == scenario.right_after_token_fault_ids


_NULL_BIND_ROUTING_CLASSIFICATION = (
    "NULL_BIND_ROUTING_DATAFLOW_ONLY_NOT_ERROR_BOUNDARY"
)
_EXPECTED_NULL_BIND_ADJACENT_ROWS = (
    (
        "N1-sealed-selection-contract__exact-count-types",
        "sealed selection contract",
        "exact count types",
    ),
    (
        "N2-exact-count-types__exact-null-bind-terminal",
        "exact count types",
        "exact NOT_EVALUABLE-NULL_BIND terminal",
    ),
    (
        "N3-exact-null-bind-terminal__context-read-2-drift",
        "exact NOT_EVALUABLE-NULL_BIND terminal",
        "context read #2 drift",
    ),
)

_OBSERVED_SCAN_ROUTING_CLASSIFICATION = (
    "OBSERVED_STATISTIC_SCAN_ROUTING_DATAFLOW_ONLY_NOT_ERROR_BOUNDARY"
)
_EXPECTED_OBSERVED_SCAN_ADJACENT_ROWS = (
    (
        "O1-sealed-selection-contract__observed-failure-vector",
        "sealed selection contract",
        "observed analytic-failure vector",
    ),
    (
        "O2-observed-failure-vector__exact-count-types",
        "observed analytic-failure vector",
        "exact count types",
    ),
    (
        "O3-exact-count-types__exact-observed-scan-terminal",
        "exact count types",
        "exact NOT_EVALUABLE-OBSERVED_STATISTIC_SCAN terminal",
    ),
    (
        "O4-exact-observed-scan-terminal__context-read-2-drift",
        "exact NOT_EVALUABLE-OBSERVED_STATISTIC_SCAN terminal",
        "context read #2 drift",
    ),
)


_LITERAL_COMPLETE_TOKEN_CALLBACK_RECEIPTS = (
    _TokenCallbackReceiptV2(
        call_index=1,
        token_schema="selcal.null-transform-token.v2",
        null_name="circular_shift_v2",
        null_parameter_sha256=(
            "a2e6b608e05886b0c06388750e201c922c21d08e4ca6faf7668742798895dda3"
        ),
        semantic_input_sha256=_SEMANTIC_INPUT_SHA256,
        scientific_plan_sha256=_SCIENTIFIC_PLAN_SHA256,
        bound_null_owner_sha256=(
            "6863e569187ddc8983aa51f622ce96d6d644842ae76a116f6c31c9e5338befba"
        ),
        is_identity=True,
        state_schema="selcal.circular-shift-state.v2",
        state_payload=0,
    ),
    _TokenCallbackReceiptV2(
        call_index=2,
        token_schema="selcal.null-transform-token.v2",
        null_name="circular_shift_v2",
        null_parameter_sha256=(
            "a2e6b608e05886b0c06388750e201c922c21d08e4ca6faf7668742798895dda3"
        ),
        semantic_input_sha256=_SEMANTIC_INPUT_SHA256,
        scientific_plan_sha256=_SCIENTIFIC_PLAN_SHA256,
        bound_null_owner_sha256=(
            "6863e569187ddc8983aa51f622ce96d6d644842ae76a116f6c31c9e5338befba"
        ),
        is_identity=True,
        state_schema="selcal.circular-shift-state.v2",
        state_payload=0,
    ),
    _TokenCallbackReceiptV2(
        call_index=3,
        token_schema="selcal.null-transform-token.v2",
        null_name="circular_shift_v2",
        null_parameter_sha256=(
            "a2e6b608e05886b0c06388750e201c922c21d08e4ca6faf7668742798895dda3"
        ),
        semantic_input_sha256=_SEMANTIC_INPUT_SHA256,
        scientific_plan_sha256=_SCIENTIFIC_PLAN_SHA256,
        bound_null_owner_sha256=(
            "6863e569187ddc8983aa51f622ce96d6d644842ae76a116f6c31c9e5338befba"
        ),
        is_identity=True,
        state_schema="selcal.circular-shift-state.v2",
        state_payload=0,
    ),
)
_LITERAL_REPLICATE_EXECUTION_TOKEN_CALLBACK_RECEIPTS = (
    _TokenCallbackReceiptV2(
        call_index=1,
        token_schema="selcal.null-transform-token.v2",
        null_name="circular_shift_v2",
        null_parameter_sha256=(
            "a2e6b608e05886b0c06388750e201c922c21d08e4ca6faf7668742798895dda3"
        ),
        semantic_input_sha256=_SEMANTIC_INPUT_SHA256,
        scientific_plan_sha256=_SCIENTIFIC_PLAN_SHA256,
        bound_null_owner_sha256=(
            "6863e569187ddc8983aa51f622ce96d6d644842ae76a116f6c31c9e5338befba"
        ),
        is_identity=True,
        state_schema="selcal.circular-shift-state.v2",
        state_payload=0,
    ),
    _TokenCallbackReceiptV2(
        call_index=2,
        token_schema="selcal.null-transform-token.v2",
        null_name="circular_shift_v2",
        null_parameter_sha256=(
            "a2e6b608e05886b0c06388750e201c922c21d08e4ca6faf7668742798895dda3"
        ),
        semantic_input_sha256=_SEMANTIC_INPUT_SHA256,
        scientific_plan_sha256=_SCIENTIFIC_PLAN_SHA256,
        bound_null_owner_sha256=(
            "6863e569187ddc8983aa51f622ce96d6d644842ae76a116f6c31c9e5338befba"
        ),
        is_identity=False,
        state_schema="selcal.circular-shift-state.v2",
        state_payload=1,
    ),
    _TokenCallbackReceiptV2(
        call_index=3,
        token_schema="selcal.null-transform-token.v2",
        null_name="circular_shift_v2",
        null_parameter_sha256=(
            "a2e6b608e05886b0c06388750e201c922c21d08e4ca6faf7668742798895dda3"
        ),
        semantic_input_sha256=_SEMANTIC_INPUT_SHA256,
        scientific_plan_sha256=_SCIENTIFIC_PLAN_SHA256,
        bound_null_owner_sha256=(
            "6863e569187ddc8983aa51f622ce96d6d644842ae76a116f6c31c9e5338befba"
        ),
        is_identity=False,
        state_schema="selcal.circular-shift-state.v2",
        state_payload=2,
    ),
)


_FOUR_TERMINAL_ORACLE_CELLS: tuple[_FourTerminalOracleCellV2, ...] = (
    _FourTerminalOracleCellV2(
        state=_FourTerminalMatrixStateV2.COMPLETE,
        boundary=_FourTerminalBoundaryKindV2.COUNT_TERMINAL,
        stimulus=_FourTerminalStimulusV2(
            factory=_FourTerminalFixtureV2.COMPLETE,
            scenario_id="row19-exact-count-types__terminal-arithmetic",
            run_kind=_FourTerminalRunKindV2.DOUBLE,
        ),
        expected=_FourTerminalObservedV2(
            status=RunStatus.COMPLETE,
            failure_stage=None,
            exception_type=V2IntegrityError,
            message="calibration counts must be exact built-in integers",
            has_cause=False,
            context_reads=1,
            read_2_actions=0,
            fault_receipts=(
                _FaultReceiptV2(
                    phase=_FaultReceiptPhaseV2.PRE_CALL,
                    fault_id="pre.result.exceedance-count.bool",
                    phase_index=0,
                ),
                _FaultReceiptV2(
                    phase=_FaultReceiptPhaseV2.PRE_CALL,
                    fault_id="pre.result.p-value.0.25",
                    phase_index=0,
                ),
            ),
            token_receipts=_LITERAL_COMPLETE_TOKEN_CALLBACK_RECEIPTS,
        ),
    ),
    _FourTerminalOracleCellV2(
        state=_FourTerminalMatrixStateV2.COMPLETE,
        boundary=_FourTerminalBoundaryKindV2.COUNT_TERMINAL,
        stimulus=_FourTerminalStimulusV2(
            factory=_FourTerminalFixtureV2.COMPLETE,
            scenario_id="row19-exact-count-types__terminal-arithmetic",
            run_kind=_FourTerminalRunKindV2.RIGHT_ONLY,
        ),
        expected=_FourTerminalObservedV2(
            status=RunStatus.COMPLETE,
            failure_stage=None,
            exception_type=V2IntegrityError,
            message="COMPLETE result decision arithmetic is invalid",
            has_cause=False,
            context_reads=1,
            read_2_actions=0,
            fault_receipts=(
                _FaultReceiptV2(
                    phase=_FaultReceiptPhaseV2.PRE_CALL,
                    fault_id="pre.result.p-value.0.25",
                    phase_index=0,
                ),
            ),
            token_receipts=_LITERAL_COMPLETE_TOKEN_CALLBACK_RECEIPTS,
        ),
    ),
    _FourTerminalOracleCellV2(
        state=_FourTerminalMatrixStateV2.COMPLETE,
        boundary=_FourTerminalBoundaryKindV2.TERMINAL_CONTEXT_READ_2,
        stimulus=_FourTerminalStimulusV2(
            factory=_FourTerminalFixtureV2.COMPLETE,
            scenario_id="row20-terminal-arithmetic__context-read-2-drift",
            run_kind=_FourTerminalRunKindV2.DOUBLE,
        ),
        expected=_FourTerminalObservedV2(
            status=RunStatus.COMPLETE,
            failure_stage=None,
            exception_type=V2IntegrityError,
            message="COMPLETE result decision arithmetic is invalid",
            has_cause=False,
            context_reads=1,
            read_2_actions=0,
            fault_receipts=(
                _FaultReceiptV2(
                    phase=_FaultReceiptPhaseV2.PRE_CALL,
                    fault_id="pre.result.p-value.0.25",
                    phase_index=0,
                ),
            ),
            token_receipts=_LITERAL_COMPLETE_TOKEN_CALLBACK_RECEIPTS,
        ),
    ),
    _FourTerminalOracleCellV2(
        state=_FourTerminalMatrixStateV2.COMPLETE,
        boundary=_FourTerminalBoundaryKindV2.TERMINAL_CONTEXT_READ_2,
        stimulus=_FourTerminalStimulusV2(
            factory=_FourTerminalFixtureV2.COMPLETE,
            scenario_id="row20-terminal-arithmetic__context-read-2-drift",
            run_kind=_FourTerminalRunKindV2.RIGHT_ONLY,
        ),
        expected=_FourTerminalObservedV2(
            status=RunStatus.COMPLETE,
            failure_stage=None,
            exception_type=V2IntegrityError,
            message="scientific plan changed during result verification",
            has_cause=False,
            context_reads=2,
            read_2_actions=1,
            fault_receipts=(
                _FaultReceiptV2(
                    phase=_FaultReceiptPhaseV2.CONTEXT_READ,
                    fault_id="phase.context.read-2.tie-tolerance-0.25",
                    phase_index=2,
                ),
            ),
            token_receipts=_LITERAL_COMPLETE_TOKEN_CALLBACK_RECEIPTS,
        ),
    ),
    _FourTerminalOracleCellV2(
        state=_FourTerminalMatrixStateV2.REPLICATE_EXECUTION,
        boundary=_FourTerminalBoundaryKindV2.COUNT_TERMINAL,
        stimulus=_FourTerminalStimulusV2(
            factory=_FourTerminalFixtureV2.REPLICATE_EXECUTION,
            scenario_id="R1-exact-count-types__replicate-execution-terminal-bounds",
            run_kind=_FourTerminalRunKindV2.DOUBLE,
        ),
        expected=_FourTerminalObservedV2(
            status=RunStatus.NOT_EVALUABLE,
            failure_stage=RunFailureStage.REPLICATE_EXECUTION,
            exception_type=V2IntegrityError,
            message="calibration counts must be exact built-in integers",
            has_cause=False,
            context_reads=1,
            read_2_actions=0,
            fault_receipts=(
                _FaultReceiptV2(
                    phase=_FaultReceiptPhaseV2.PRE_CALL,
                    fault_id="replicate-execution.pre.failure-count.bool",
                    phase_index=0,
                ),
                _FaultReceiptV2(
                    phase=_FaultReceiptPhaseV2.PRE_CALL,
                    fault_id="replicate-execution.pre.bound-high.0.5",
                    phase_index=0,
                ),
            ),
            token_receipts=_LITERAL_REPLICATE_EXECUTION_TOKEN_CALLBACK_RECEIPTS,
        ),
    ),
    _FourTerminalOracleCellV2(
        state=_FourTerminalMatrixStateV2.REPLICATE_EXECUTION,
        boundary=_FourTerminalBoundaryKindV2.COUNT_TERMINAL,
        stimulus=_FourTerminalStimulusV2(
            factory=_FourTerminalFixtureV2.REPLICATE_EXECUTION,
            scenario_id="R1-exact-count-types__replicate-execution-terminal-bounds",
            run_kind=_FourTerminalRunKindV2.RIGHT_ONLY,
        ),
        expected=_FourTerminalObservedV2(
            status=RunStatus.NOT_EVALUABLE,
            failure_stage=RunFailureStage.REPLICATE_EXECUTION,
            exception_type=V2IntegrityError,
            message="replicate-execution bounds contradict E, F, and B",
            has_cause=False,
            context_reads=1,
            read_2_actions=0,
            fault_receipts=(
                _FaultReceiptV2(
                    phase=_FaultReceiptPhaseV2.PRE_CALL,
                    fault_id="replicate-execution.pre.bound-high.0.5",
                    phase_index=0,
                ),
            ),
            token_receipts=_LITERAL_REPLICATE_EXECUTION_TOKEN_CALLBACK_RECEIPTS,
        ),
    ),
    _FourTerminalOracleCellV2(
        state=_FourTerminalMatrixStateV2.REPLICATE_EXECUTION,
        boundary=_FourTerminalBoundaryKindV2.TERMINAL_CONTEXT_READ_2,
        stimulus=_FourTerminalStimulusV2(
            factory=_FourTerminalFixtureV2.REPLICATE_EXECUTION,
            scenario_id=(
                "R2-replicate-execution-terminal-bounds__context-read-2-drift"
            ),
            run_kind=_FourTerminalRunKindV2.DOUBLE,
        ),
        expected=_FourTerminalObservedV2(
            status=RunStatus.NOT_EVALUABLE,
            failure_stage=RunFailureStage.REPLICATE_EXECUTION,
            exception_type=V2IntegrityError,
            message="replicate-execution bounds contradict E, F, and B",
            has_cause=False,
            context_reads=1,
            read_2_actions=0,
            fault_receipts=(
                _FaultReceiptV2(
                    phase=_FaultReceiptPhaseV2.PRE_CALL,
                    fault_id="replicate-execution.pre.bound-high.0.5",
                    phase_index=0,
                ),
            ),
            token_receipts=_LITERAL_REPLICATE_EXECUTION_TOKEN_CALLBACK_RECEIPTS,
        ),
    ),
    _FourTerminalOracleCellV2(
        state=_FourTerminalMatrixStateV2.REPLICATE_EXECUTION,
        boundary=_FourTerminalBoundaryKindV2.TERMINAL_CONTEXT_READ_2,
        stimulus=_FourTerminalStimulusV2(
            factory=_FourTerminalFixtureV2.REPLICATE_EXECUTION,
            scenario_id=(
                "R2-replicate-execution-terminal-bounds__context-read-2-drift"
            ),
            run_kind=_FourTerminalRunKindV2.RIGHT_ONLY,
        ),
        expected=_FourTerminalObservedV2(
            status=RunStatus.NOT_EVALUABLE,
            failure_stage=RunFailureStage.REPLICATE_EXECUTION,
            exception_type=V2IntegrityError,
            message="scientific plan changed during result verification",
            has_cause=False,
            context_reads=2,
            read_2_actions=1,
            fault_receipts=(
                _FaultReceiptV2(
                    phase=_FaultReceiptPhaseV2.CONTEXT_READ,
                    fault_id=(
                        "replicate-execution.context.read-2.tie-tolerance.0.25"
                    ),
                    phase_index=2,
                ),
            ),
            token_receipts=_LITERAL_REPLICATE_EXECUTION_TOKEN_CALLBACK_RECEIPTS,
        ),
    ),
    _FourTerminalOracleCellV2(
        state=_FourTerminalMatrixStateV2.NULL_BIND,
        boundary=_FourTerminalBoundaryKindV2.COUNT_TERMINAL,
        stimulus=_FourTerminalStimulusV2(
            factory=_FourTerminalFixtureV2.NULL_BIND,
            scenario_id="N2-exact-count-types__exact-null-bind-terminal",
            run_kind=_FourTerminalRunKindV2.DOUBLE,
        ),
        expected=_FourTerminalObservedV2(
            status=RunStatus.NOT_EVALUABLE,
            failure_stage=RunFailureStage.NULL_BIND,
            exception_type=V2IntegrityError,
            message="calibration counts must be exact built-in integers",
            has_cause=False,
            context_reads=1,
            read_2_actions=0,
            fault_receipts=(
                _FaultReceiptV2(
                    phase=_FaultReceiptPhaseV2.PRE_CALL,
                    fault_id="null-bind.pre.exceedance-count.bool",
                    phase_index=0,
                ),
                _FaultReceiptV2(
                    phase=_FaultReceiptPhaseV2.PRE_CALL,
                    fault_id="null-bind.pre.terminal.p-value.0.25",
                    phase_index=0,
                ),
            ),
            token_receipts=(),
        ),
    ),
    _FourTerminalOracleCellV2(
        state=_FourTerminalMatrixStateV2.NULL_BIND,
        boundary=_FourTerminalBoundaryKindV2.COUNT_TERMINAL,
        stimulus=_FourTerminalStimulusV2(
            factory=_FourTerminalFixtureV2.NULL_BIND,
            scenario_id="N2-exact-count-types__exact-null-bind-terminal",
            run_kind=_FourTerminalRunKindV2.RIGHT_ONLY,
        ),
        expected=_FourTerminalObservedV2(
            status=RunStatus.NOT_EVALUABLE,
            failure_stage=RunFailureStage.NULL_BIND,
            exception_type=V2IntegrityError,
            message="NULL_BIND terminal result is invalid",
            has_cause=False,
            context_reads=1,
            read_2_actions=0,
            fault_receipts=(
                _FaultReceiptV2(
                    phase=_FaultReceiptPhaseV2.PRE_CALL,
                    fault_id="null-bind.pre.terminal.p-value.0.25",
                    phase_index=0,
                ),
            ),
            token_receipts=(),
        ),
    ),
    _FourTerminalOracleCellV2(
        state=_FourTerminalMatrixStateV2.NULL_BIND,
        boundary=_FourTerminalBoundaryKindV2.TERMINAL_CONTEXT_READ_2,
        stimulus=_FourTerminalStimulusV2(
            factory=_FourTerminalFixtureV2.NULL_BIND,
            scenario_id="N3-exact-null-bind-terminal__context-read-2-drift",
            run_kind=_FourTerminalRunKindV2.DOUBLE,
        ),
        expected=_FourTerminalObservedV2(
            status=RunStatus.NOT_EVALUABLE,
            failure_stage=RunFailureStage.NULL_BIND,
            exception_type=V2IntegrityError,
            message="NULL_BIND terminal result is invalid",
            has_cause=False,
            context_reads=1,
            read_2_actions=0,
            fault_receipts=(
                _FaultReceiptV2(
                    phase=_FaultReceiptPhaseV2.PRE_CALL,
                    fault_id="null-bind.pre.terminal.p-value.0.25",
                    phase_index=0,
                ),
            ),
            token_receipts=(),
        ),
    ),
    _FourTerminalOracleCellV2(
        state=_FourTerminalMatrixStateV2.NULL_BIND,
        boundary=_FourTerminalBoundaryKindV2.TERMINAL_CONTEXT_READ_2,
        stimulus=_FourTerminalStimulusV2(
            factory=_FourTerminalFixtureV2.NULL_BIND,
            scenario_id="N3-exact-null-bind-terminal__context-read-2-drift",
            run_kind=_FourTerminalRunKindV2.RIGHT_ONLY,
        ),
        expected=_FourTerminalObservedV2(
            status=RunStatus.NOT_EVALUABLE,
            failure_stage=RunFailureStage.NULL_BIND,
            exception_type=V2IntegrityError,
            message="scientific plan changed during result verification",
            has_cause=False,
            context_reads=2,
            read_2_actions=1,
            fault_receipts=(
                _FaultReceiptV2(
                    phase=_FaultReceiptPhaseV2.CONTEXT_READ,
                    fault_id="null-bind.context.read-2.tie-tolerance.0.25",
                    phase_index=2,
                ),
            ),
            token_receipts=(),
        ),
    ),
    _FourTerminalOracleCellV2(
        state=_FourTerminalMatrixStateV2.OBSERVED_STATISTIC_SCAN,
        boundary=_FourTerminalBoundaryKindV2.COUNT_TERMINAL,
        stimulus=_FourTerminalStimulusV2(
            factory=_FourTerminalFixtureV2.OBSERVED_STATISTIC_SCAN,
            scenario_id="O3-exact-count-types__exact-observed-scan-terminal",
            run_kind=_FourTerminalRunKindV2.DOUBLE,
        ),
        expected=_FourTerminalObservedV2(
            status=RunStatus.NOT_EVALUABLE,
            failure_stage=RunFailureStage.OBSERVED_STATISTIC_SCAN,
            exception_type=V2IntegrityError,
            message="calibration counts must be exact built-in integers",
            has_cause=False,
            context_reads=1,
            read_2_actions=0,
            fault_receipts=(
                _FaultReceiptV2(
                    phase=_FaultReceiptPhaseV2.PRE_CALL,
                    fault_id="observed-scan.pre.exceedance-count.bool",
                    phase_index=0,
                ),
                _FaultReceiptV2(
                    phase=_FaultReceiptPhaseV2.PRE_CALL,
                    fault_id="observed-scan.pre.terminal.p-value.0.25",
                    phase_index=0,
                ),
            ),
            token_receipts=(),
        ),
    ),
    _FourTerminalOracleCellV2(
        state=_FourTerminalMatrixStateV2.OBSERVED_STATISTIC_SCAN,
        boundary=_FourTerminalBoundaryKindV2.COUNT_TERMINAL,
        stimulus=_FourTerminalStimulusV2(
            factory=_FourTerminalFixtureV2.OBSERVED_STATISTIC_SCAN,
            scenario_id="O3-exact-count-types__exact-observed-scan-terminal",
            run_kind=_FourTerminalRunKindV2.RIGHT_ONLY,
        ),
        expected=_FourTerminalObservedV2(
            status=RunStatus.NOT_EVALUABLE,
            failure_stage=RunFailureStage.OBSERVED_STATISTIC_SCAN,
            exception_type=V2IntegrityError,
            message="OBSERVED_STATISTIC_SCAN terminal result is invalid",
            has_cause=False,
            context_reads=1,
            read_2_actions=0,
            fault_receipts=(
                _FaultReceiptV2(
                    phase=_FaultReceiptPhaseV2.PRE_CALL,
                    fault_id="observed-scan.pre.terminal.p-value.0.25",
                    phase_index=0,
                ),
            ),
            token_receipts=(),
        ),
    ),
    _FourTerminalOracleCellV2(
        state=_FourTerminalMatrixStateV2.OBSERVED_STATISTIC_SCAN,
        boundary=_FourTerminalBoundaryKindV2.TERMINAL_CONTEXT_READ_2,
        stimulus=_FourTerminalStimulusV2(
            factory=_FourTerminalFixtureV2.OBSERVED_STATISTIC_SCAN,
            scenario_id="O4-exact-observed-scan-terminal__context-read-2-drift",
            run_kind=_FourTerminalRunKindV2.DOUBLE,
        ),
        expected=_FourTerminalObservedV2(
            status=RunStatus.NOT_EVALUABLE,
            failure_stage=RunFailureStage.OBSERVED_STATISTIC_SCAN,
            exception_type=V2IntegrityError,
            message="OBSERVED_STATISTIC_SCAN terminal result is invalid",
            has_cause=False,
            context_reads=1,
            read_2_actions=0,
            fault_receipts=(
                _FaultReceiptV2(
                    phase=_FaultReceiptPhaseV2.PRE_CALL,
                    fault_id="observed-scan.pre.terminal.p-value.0.25",
                    phase_index=0,
                ),
            ),
            token_receipts=(),
        ),
    ),
    _FourTerminalOracleCellV2(
        state=_FourTerminalMatrixStateV2.OBSERVED_STATISTIC_SCAN,
        boundary=_FourTerminalBoundaryKindV2.TERMINAL_CONTEXT_READ_2,
        stimulus=_FourTerminalStimulusV2(
            factory=_FourTerminalFixtureV2.OBSERVED_STATISTIC_SCAN,
            scenario_id="O4-exact-observed-scan-terminal__context-read-2-drift",
            run_kind=_FourTerminalRunKindV2.RIGHT_ONLY,
        ),
        expected=_FourTerminalObservedV2(
            status=RunStatus.NOT_EVALUABLE,
            failure_stage=RunFailureStage.OBSERVED_STATISTIC_SCAN,
            exception_type=V2IntegrityError,
            message="scientific plan changed during result verification",
            has_cause=False,
            context_reads=2,
            read_2_actions=1,
            fault_receipts=(
                _FaultReceiptV2(
                    phase=_FaultReceiptPhaseV2.CONTEXT_READ,
                    fault_id=(
                        "observed-scan.context.read-2.tie-tolerance.0.25"
                    ),
                    phase_index=2,
                ),
            ),
            token_receipts=(),
        ),
    ),
)


def _assert_exact_integrity_error(
    captured: pytest.ExceptionInfo[V2IntegrityError],
    expectation: _IntegrityErrorExpectationV2,
) -> None:
    assert type(captured.value) is V2IntegrityError
    assert str(captured.value) == expectation.message
    assert (captured.value.__cause__ is not None) is expectation.has_cause


def _verify_with_replicate_path_profile(
    verifier: _VerifierV2,
    result: CalibrationResult,
    resolution: PlanResolutionV2,
    *,
    replicate_receipt: list[str],
    sentinel_armed: bool,
) -> object:
    assert replicate_receipt == []
    real_verifier = contracts_v2.verify_calibration_result
    closure_vars = inspect.getclosurevars(real_verifier)
    exact_slot_reader = closure_vars.nonlocals["primitive_ops"].read_exact_slots
    assert type(exact_slot_reader) is FunctionType
    assert exact_slot_reader.__module__ == "selcal._verifier_primitives_v2"
    assert exact_slot_reader.__qualname__ == (
        "_freeze_slot_reader.<locals>.read_exact_slots"
    )
    exact_slot_reader_code = exact_slot_reader.__code__
    assert (
        inspect.getclosurevars(real_verifier)
        .nonlocals["primitive_ops"].read_exact_slots.__code__
        is exact_slot_reader_code
    )

    def replicate_path_profile(
        frame: FrameType,
        event: str,
        _arg: object,
    ) -> None:
        if event != "call" or frame.f_code is not exact_slot_reader_code:
            return
        subject = frame.f_locals.get("subject")
        if type(subject) is not str or not subject.startswith("replicate "):
            return
        replicate_id = subject.removeprefix("replicate ")
        if not replicate_id.isascii() or not replicate_id.isdecimal():
            return
        assert frame.f_code is exact_slot_reader_code
        replicate_receipt.append(subject)
        if sentinel_armed:
            raise AssertionError(
                "TEST-ONLY replicate validation traversal sentinel"
            )

    previous_profiler = sys.getprofile()
    sys.setprofile(replicate_path_profile)
    runtime_verifier: Callable[[object, object], object] = verifier
    try:
        returned = runtime_verifier(result, resolution)
    finally:
        sys.setprofile(previous_profiler)
        assert sys.getprofile() is previous_profiler
    return returned


def _verify_with_observed_short_path_profile(
    verifier: _VerifierV2,
    result: CalibrationResult,
    resolution: PlanResolutionV2,
    *,
    traversal_receipts: list[_TraversalReceiptV2],
) -> object:
    assert traversal_receipts == []
    real_verifier = contracts_v2.verify_calibration_result
    closure_vars = inspect.getclosurevars(real_verifier)
    exact_slot_reader = closure_vars.nonlocals["primitive_ops"].read_exact_slots
    assert type(exact_slot_reader) is FunctionType
    assert exact_slot_reader.__module__ == "selcal._verifier_primitives_v2"
    assert exact_slot_reader.__qualname__ == (
        "_freeze_slot_reader.<locals>.read_exact_slots"
    )
    exact_slot_reader_code = exact_slot_reader.__code__
    assert (
        inspect.getclosurevars(real_verifier)
        .nonlocals["primitive_ops"].read_exact_slots.__code__
        is exact_slot_reader_code
    )

    def observed_short_path_profile(
        frame: FrameType,
        event: str,
        _arg: object,
    ) -> None:
        if event != "call" or frame.f_code is not exact_slot_reader_code:
            return
        subject = frame.f_locals.get("subject")
        if type(subject) is not str:
            return
        if subject == "observed result vector statistic":
            traversal_receipts.append(
                _TraversalReceiptV2(
                    phase=_TraversalReceiptPhaseV2.OBSERVED_STATISTIC,
                    subject="observed result vector statistic",
                )
            )
            return
        if subject == "observed selection":
            traversal_receipts.append(
                _TraversalReceiptV2(
                    phase=_TraversalReceiptPhaseV2.OBSERVED_SELECTION_REREAD,
                    subject="observed selection",
                )
            )
            return
        if not subject.startswith("replicate "):
            return
        replicate_id = subject.removeprefix("replicate ")
        if not replicate_id.isascii() or not replicate_id.isdecimal():
            return
        traversal_receipts.append(
            _TraversalReceiptV2(
                phase=_TraversalReceiptPhaseV2.REPLICATE,
                subject=subject,
            )
        )

    previous_profiler = sys.getprofile()
    sys.setprofile(observed_short_path_profile)
    runtime_verifier: Callable[[object, object], object] = verifier
    try:
        returned = runtime_verifier(result, resolution)
    finally:
        sys.setprofile(previous_profiler)
        assert sys.getprofile() is previous_profiler
    return returned


def _assert_replicate_execution_three_token_receipts(
    probe: _PhaseProbeV2,
) -> None:
    assert probe.token_verifier_calls == 3
    assert probe.token_completion_calls == 3
    assert tuple(probe.token_completion_ids) == (
        "phase.token.real-return.call-1",
        "phase.token.real-return.call-2",
        "phase.token.real-return.call-3",
    )
    assert probe.token_fault_calls == 0
    assert probe.token_fault_ids == []
    assert probe.after_token_fault_calls == 0
    assert probe.after_token_fault_ids == []
    assert tuple(probe.token_completion_receipts) == (
        _TokenCallbackReceiptV2(
            call_index=1,
            token_schema="selcal.null-transform-token.v2",
            null_name="circular_shift_v2",
            null_parameter_sha256=(
                "a2e6b608e05886b0c06388750e201c922c21d08e4ca6faf7668742798895dda3"
            ),
            semantic_input_sha256=_SEMANTIC_INPUT_SHA256,
            scientific_plan_sha256=_SCIENTIFIC_PLAN_SHA256,
            bound_null_owner_sha256=(
                "6863e569187ddc8983aa51f622ce96d6d644842ae76a116f6c31c9e5338befba"
            ),
            is_identity=True,
            state_schema="selcal.circular-shift-state.v2",
            state_payload=0,
        ),
        _TokenCallbackReceiptV2(
            call_index=2,
            token_schema="selcal.null-transform-token.v2",
            null_name="circular_shift_v2",
            null_parameter_sha256=(
                "a2e6b608e05886b0c06388750e201c922c21d08e4ca6faf7668742798895dda3"
            ),
            semantic_input_sha256=_SEMANTIC_INPUT_SHA256,
            scientific_plan_sha256=_SCIENTIFIC_PLAN_SHA256,
            bound_null_owner_sha256=(
                "6863e569187ddc8983aa51f622ce96d6d644842ae76a116f6c31c9e5338befba"
            ),
            is_identity=False,
            state_schema="selcal.circular-shift-state.v2",
            state_payload=1,
        ),
        _TokenCallbackReceiptV2(
            call_index=3,
            token_schema="selcal.null-transform-token.v2",
            null_name="circular_shift_v2",
            null_parameter_sha256=(
                "a2e6b608e05886b0c06388750e201c922c21d08e4ca6faf7668742798895dda3"
            ),
            semantic_input_sha256=_SEMANTIC_INPUT_SHA256,
            scientific_plan_sha256=_SCIENTIFIC_PLAN_SHA256,
            bound_null_owner_sha256=(
                "6863e569187ddc8983aa51f622ce96d6d644842ae76a116f6c31c9e5338befba"
            ),
            is_identity=False,
            state_schema="selcal.circular-shift-state.v2",
            state_payload=2,
        ),
    )


def test_replicate_execution_adjacent_ledger_has_exact_two_priority_rows() -> None:
    assert tuple(
        (scenario.row_id, scenario.left.label, scenario.right.label)
        for scenario in REPLICATE_EXECUTION_ADJACENT_LEDGER_DOUBLE_FAULTS
    ) == (
        (
            "R1-exact-count-types__replicate-execution-terminal-bounds",
            "exact count types",
            "exact NOT_EVALUABLE-REPLICATE_EXECUTION bounds",
        ),
        (
            "R2-replicate-execution-terminal-bounds__context-read-2-drift",
            "exact NOT_EVALUABLE-REPLICATE_EXECUTION bounds",
            "context read #2 drift",
        ),
    )
    assert len(REPLICATE_EXECUTION_ADJACENT_LEDGER_DOUBLE_FAULTS) == 2
    assert len(
        {
            scenario.row_id
            for scenario in REPLICATE_EXECUTION_ADJACENT_LEDGER_DOUBLE_FAULTS
        }
    ) == 2


def test_replicate_execution_exact_valid_acceptance_traverses_failure_and_tokens() -> None:
    result, resolution = _replicate_execution_terminal_literal_case()
    _assert_replicate_execution_terminal_literal_contract(result)
    probe = _PhaseProbeV2()
    verifier = _verifier_with_phase_profile(result, (), probe)
    replicate_receipt: list[str] = []

    returned = _verify_with_replicate_path_profile(
        verifier,
        result,
        resolution,
        replicate_receipt=replicate_receipt,
        sentinel_armed=False,
    )

    assert returned is None
    assert tuple(replicate_receipt) == (
        "replicate 0",
        "replicate 1",
        "replicate 2",
    )
    assert probe.context_reads == 2
    assert probe.context_fault_calls == 0
    _assert_replicate_execution_three_token_receipts(probe)
    # Two failed candidate positions are one failed replicate under the terminal contract.
    assert sum(
        item.validity is Validity.ANALYTIC_FAILURE
        for item in result.replicates[1].statistic_results
    ) == 2
    assert result.failure_count == 1


def test_replicate_execution_priority_row_r1_retains_exact_double_priority() -> None:
    scenario = REPLICATE_EXECUTION_ADJACENT_LEDGER_DOUBLE_FAULTS[0]
    assert scenario.row_id.startswith("R1-")
    result, resolution = _replicate_execution_terminal_literal_case()
    _assert_replicate_execution_terminal_literal_contract(result)
    injections = (scenario.left, scenario.right)
    probe = _PhaseProbeV2()
    _apply_before_call_faults(result, injections, probe)
    verifier = _verifier_with_phase_profile(result, injections, probe)

    with pytest.raises(V2IntegrityError) as captured:
        verifier(result, resolution)

    _assert_exact_integrity_error(captured, scenario.double_expectation)
    assert tuple(probe.fault_receipts) == scenario.double_receipts
    assert probe.context_reads == scenario.double_context_reads == 1
    assert probe.context_fault_calls == scenario.double_context_fault_calls == 0
    assert probe.token_verifier_calls == scenario.double_token_verifier_calls == 3
    assert probe.token_completion_calls == scenario.double_token_completion_calls == 3
    _assert_replicate_execution_three_token_receipts(probe)


def test_replicate_execution_priority_row_r1_fresh_right_only_is_live() -> None:
    scenario = REPLICATE_EXECUTION_ADJACENT_LEDGER_DOUBLE_FAULTS[0]
    assert scenario.row_id.startswith("R1-")
    result, resolution = _replicate_execution_terminal_literal_case()
    _assert_replicate_execution_terminal_literal_contract(result)
    injections = (scenario.right,)
    probe = _PhaseProbeV2()
    _apply_before_call_faults(result, injections, probe)
    verifier = _verifier_with_phase_profile(result, injections, probe)

    with pytest.raises(V2IntegrityError) as captured:
        verifier(result, resolution)

    _assert_exact_integrity_error(captured, scenario.right_expectation)
    assert tuple(probe.fault_receipts) == scenario.right_receipts
    assert probe.context_reads == scenario.right_context_reads == 1
    assert probe.context_fault_calls == scenario.right_context_fault_calls == 0
    assert probe.token_verifier_calls == scenario.right_token_verifier_calls == 3
    assert probe.token_completion_calls == scenario.right_token_completion_calls == 3
    _assert_replicate_execution_three_token_receipts(probe)


def test_replicate_execution_priority_row_r2_double_stops_before_context_read_2() -> None:
    scenario = REPLICATE_EXECUTION_ADJACENT_LEDGER_DOUBLE_FAULTS[1]
    assert scenario.row_id.startswith("R2-")
    result, resolution = _replicate_execution_terminal_literal_case()
    _assert_replicate_execution_terminal_literal_contract(result)
    injections = (scenario.left, scenario.right)
    probe = _PhaseProbeV2()
    _apply_before_call_faults(result, injections, probe)
    verifier = _verifier_with_phase_profile(result, injections, probe)

    with pytest.raises(V2IntegrityError) as captured:
        verifier(result, resolution)

    _assert_exact_integrity_error(captured, scenario.double_expectation)
    assert tuple(probe.fault_receipts) == scenario.double_receipts
    assert probe.context_reads == scenario.double_context_reads == 1
    assert probe.context_fault_calls == scenario.double_context_fault_calls == 0
    assert probe.context_fault_ids == []
    assert probe.token_verifier_calls == scenario.double_token_verifier_calls == 3
    assert probe.token_completion_calls == scenario.double_token_completion_calls == 3
    _assert_replicate_execution_three_token_receipts(probe)


def test_replicate_execution_priority_row_r2_fresh_right_only_read_2_is_live() -> None:
    scenario = REPLICATE_EXECUTION_ADJACENT_LEDGER_DOUBLE_FAULTS[1]
    assert scenario.row_id.startswith("R2-")
    result, resolution = _replicate_execution_terminal_literal_case()
    _assert_replicate_execution_terminal_literal_contract(result)
    injections = (scenario.right,)
    probe = _PhaseProbeV2()
    _apply_before_call_faults(result, injections, probe)
    verifier = _verifier_with_phase_profile(result, injections, probe)

    with pytest.raises(V2IntegrityError) as captured:
        verifier(result, resolution)

    _assert_exact_integrity_error(captured, scenario.right_expectation)
    assert tuple(probe.fault_receipts) == scenario.right_receipts
    assert probe.context_reads == scenario.right_context_reads == 2
    assert probe.context_fault_calls == scenario.right_context_fault_calls == 1
    assert probe.context_fault_ids == [
        "replicate-execution.context.read-2.tie-tolerance.0.25"
    ]
    assert probe.token_verifier_calls == scenario.right_token_verifier_calls == 3
    assert probe.token_completion_calls == scenario.right_token_completion_calls == 3
    _assert_replicate_execution_three_token_receipts(probe)


def _fresh_four_terminal_stimulus_inputs(
    stimulus: _FourTerminalStimulusV2,
) -> tuple[
    CalibrationResult,
    PlanResolutionV2,
    FaultInjectionV2,
    FaultInjectionV2,
]:
    if stimulus.factory is _FourTerminalFixtureV2.COMPLETE:
        complete_matches = tuple(
            scenario
            for scenario in ADJACENT_LEDGER_DOUBLE_FAULTS
            if scenario.row_id == stimulus.scenario_id
        )
        assert len(complete_matches) == 1
        complete_result, complete_resolution = _adjacent_fault_complete_case()
        return (
            complete_result,
            complete_resolution,
            complete_matches[0].left,
            complete_matches[0].right,
        )
    if stimulus.factory is _FourTerminalFixtureV2.REPLICATE_EXECUTION:
        replicate_matches = tuple(
            scenario
            for scenario in REPLICATE_EXECUTION_ADJACENT_LEDGER_DOUBLE_FAULTS
            if scenario.row_id == stimulus.scenario_id
        )
        assert len(replicate_matches) == 1
        replicate_result, replicate_resolution = (
            _replicate_execution_terminal_literal_case()
        )
        return (
            replicate_result,
            replicate_resolution,
            replicate_matches[0].left,
            replicate_matches[0].right,
        )
    if stimulus.factory is _FourTerminalFixtureV2.NULL_BIND:
        null_matches = tuple(
            scenario
            for scenario in NULL_BIND_ADJACENT_LEDGER_DOUBLE_FAULTS
            if scenario.row_id == stimulus.scenario_id
        )
        assert len(null_matches) == 1
        null_result, null_resolution = _null_bind_short_path_literal_case()
        return (
            null_result,
            null_resolution,
            null_matches[0].left,
            null_matches[0].right,
        )
    assert stimulus.factory is _FourTerminalFixtureV2.OBSERVED_STATISTIC_SCAN
    observed_matches = tuple(
        scenario
        for scenario in OBSERVED_SCAN_ADJACENT_LEDGER_DOUBLE_FAULTS
        if scenario.row_id == stimulus.scenario_id
    )
    assert len(observed_matches) == 1
    observed_result, observed_resolution = (
        _observed_statistic_scan_short_path_literal_case()
    )
    return (
        observed_result,
        observed_resolution,
        observed_matches[0].left,
        observed_matches[0].right,
    )


def _observe_four_terminal_matrix_call(
    stimulus: _FourTerminalStimulusV2,
) -> _FourTerminalObservedV2:
    result, resolution, left, right = _fresh_four_terminal_stimulus_inputs(
        stimulus
    )
    injections: tuple[FaultInjectionV2, ...]
    if stimulus.run_kind is _FourTerminalRunKindV2.DOUBLE:
        injections = (left, right)
    else:
        assert stimulus.run_kind is _FourTerminalRunKindV2.RIGHT_ONLY
        injections = (right,)
    probe = _PhaseProbeV2()
    _apply_before_call_faults(result, injections, probe)
    verifier = _verifier_with_phase_profile(result, injections, probe)

    try:
        verifier(result, resolution)
    except Exception as captured:
        return _FourTerminalObservedV2(
            status=result.status,
            failure_stage=result.failure_stage,
            exception_type=type(captured),
            message=str(captured),
            has_cause=captured.__cause__ is not None,
            context_reads=probe.context_reads,
            read_2_actions=probe.context_fault_calls,
            fault_receipts=tuple(probe.fault_receipts),
            token_receipts=tuple(probe.token_completion_receipts),
        )
    raise AssertionError("TEST-ONLY four-terminal stimulus did not raise")


def test_four_terminal_oracle_has_exact_sixteen_independent_cells() -> None:
    exact_keys = tuple(
        (cell.state, cell.boundary, cell.stimulus.run_kind)
        for cell in _FOUR_TERMINAL_ORACLE_CELLS
    )
    assert exact_keys == tuple(
        (
            state,
            boundary,
            run_kind,
        )
        for state in (
            _FourTerminalMatrixStateV2.COMPLETE,
            _FourTerminalMatrixStateV2.REPLICATE_EXECUTION,
            _FourTerminalMatrixStateV2.NULL_BIND,
            _FourTerminalMatrixStateV2.OBSERVED_STATISTIC_SCAN,
        )
        for boundary in (
            _FourTerminalBoundaryKindV2.COUNT_TERMINAL,
            _FourTerminalBoundaryKindV2.TERMINAL_CONTEXT_READ_2,
        )
        for run_kind in (
            _FourTerminalRunKindV2.DOUBLE,
            _FourTerminalRunKindV2.RIGHT_ONLY,
        )
    )
    assert len(exact_keys) == 16
    assert len(set(exact_keys)) == 16
    assert {cell.state for cell in _FOUR_TERMINAL_ORACLE_CELLS} == {
        _FourTerminalMatrixStateV2.COMPLETE,
        _FourTerminalMatrixStateV2.REPLICATE_EXECUTION,
        _FourTerminalMatrixStateV2.NULL_BIND,
        _FourTerminalMatrixStateV2.OBSERVED_STATISTIC_SCAN,
    }
    assert {cell.boundary for cell in _FOUR_TERMINAL_ORACLE_CELLS} == {
        _FourTerminalBoundaryKindV2.COUNT_TERMINAL,
        _FourTerminalBoundaryKindV2.TERMINAL_CONTEXT_READ_2,
    }
    assert {cell.stimulus.run_kind for cell in _FOUR_TERMINAL_ORACLE_CELLS} == {
        _FourTerminalRunKindV2.DOUBLE,
        _FourTerminalRunKindV2.RIGHT_ONLY,
    }
    assert tuple(
        (cell.state.value, cell.stimulus.factory.value)
        for cell in _FOUR_TERMINAL_ORACLE_CELLS
    ) == (
        ("COMPLETE", "COMPLETE"),
        ("COMPLETE", "COMPLETE"),
        ("COMPLETE", "COMPLETE"),
        ("COMPLETE", "COMPLETE"),
        ("REPLICATE_EXECUTION", "REPLICATE_EXECUTION"),
        ("REPLICATE_EXECUTION", "REPLICATE_EXECUTION"),
        ("REPLICATE_EXECUTION", "REPLICATE_EXECUTION"),
        ("REPLICATE_EXECUTION", "REPLICATE_EXECUTION"),
        ("NULL_BIND", "NULL_BIND"),
        ("NULL_BIND", "NULL_BIND"),
        ("NULL_BIND", "NULL_BIND"),
        ("NULL_BIND", "NULL_BIND"),
        ("OBSERVED_STATISTIC_SCAN", "OBSERVED_STATISTIC_SCAN"),
        ("OBSERVED_STATISTIC_SCAN", "OBSERVED_STATISTIC_SCAN"),
        ("OBSERVED_STATISTIC_SCAN", "OBSERVED_STATISTIC_SCAN"),
        ("OBSERVED_STATISTIC_SCAN", "OBSERVED_STATISTIC_SCAN"),
    )


@pytest.mark.parametrize(
    "cell",
    _FOUR_TERMINAL_ORACLE_CELLS,
    ids=lambda cell: (
        f"{cell.state.value.lower()}-"
        f"{cell.boundary.value.lower()}-"
        f"{cell.stimulus.run_kind.value.lower()}"
    ),
)
def test_four_terminal_oracle_observes_fresh_literal_behavior(
    cell: _FourTerminalOracleCellV2,
) -> None:
    observed = _observe_four_terminal_matrix_call(cell.stimulus)

    assert observed == cell.expected
def test_null_bind_adjacent_ledger_has_exact_three_genuine_priority_rows() -> None:
    assert tuple(
        (scenario.row_id, scenario.left.label, scenario.right.label)
        for scenario in NULL_BIND_ADJACENT_LEDGER_DOUBLE_FAULTS
    ) == _EXPECTED_NULL_BIND_ADJACENT_ROWS
    assert len(NULL_BIND_ADJACENT_LEDGER_DOUBLE_FAULTS) == 3
    assert len(
        {scenario.row_id for scenario in NULL_BIND_ADJACENT_LEDGER_DOUBLE_FAULTS}
    ) == 3
    assert all(
        "branch-validation" not in scenario.row_id
        for scenario in NULL_BIND_ADJACENT_LEDGER_DOUBLE_FAULTS
    )
    assert _NULL_BIND_ROUTING_CLASSIFICATION not in {
        scenario.row_id for scenario in NULL_BIND_ADJACENT_LEDGER_DOUBLE_FAULTS
    }


@pytest.mark.parametrize(
    "scenario",
    NULL_BIND_ADJACENT_LEDGER_DOUBLE_FAULTS[:2],
    ids=lambda scenario: scenario.row_id,
)
def test_null_bind_priority_rows_n1_n2_retain_exact_double_fault_priority(
    scenario: NullBindAdjacentFaultScenarioV2,
) -> None:
    result, resolution = _null_bind_short_path_literal_case()
    injections = (scenario.left, scenario.right)
    probe = _PhaseProbeV2()
    _apply_before_call_faults(result, injections, probe)
    verifier = _verifier_with_phase_profile(result, injections, probe)

    with pytest.raises(V2IntegrityError) as captured:
        verifier(result, resolution)

    _assert_exact_integrity_error(captured, scenario.double_expectation)
    assert tuple(probe.fault_receipts) == scenario.double_receipts
    assert probe.context_reads == scenario.double_context_reads
    assert probe.context_fault_calls == scenario.double_context_fault_calls
    assert probe.token_verifier_calls == 0
    assert probe.token_fault_calls == 0
    assert probe.token_completion_calls == 0


@pytest.mark.parametrize(
    "scenario",
    NULL_BIND_ADJACENT_LEDGER_DOUBLE_FAULTS[:2],
    ids=lambda scenario: scenario.row_id,
)
def test_null_bind_priority_rows_n1_n2_fresh_right_only_fault_is_live(
    scenario: NullBindAdjacentFaultScenarioV2,
) -> None:
    result, resolution = _null_bind_short_path_literal_case()
    injections = (scenario.right,)
    probe = _PhaseProbeV2()
    _apply_before_call_faults(result, injections, probe)
    verifier = _verifier_with_phase_profile(result, injections, probe)

    with pytest.raises(V2IntegrityError) as captured:
        verifier(result, resolution)

    _assert_exact_integrity_error(captured, scenario.right_expectation)
    assert tuple(probe.fault_receipts) == scenario.right_receipts
    assert probe.context_reads == scenario.right_context_reads
    assert probe.context_fault_calls == scenario.right_context_fault_calls
    assert probe.token_verifier_calls == 0
    assert probe.token_fault_calls == 0
    assert probe.token_completion_calls == 0


def test_null_bind_priority_row_n3_double_stops_before_context_read_2() -> None:
    scenario = NULL_BIND_ADJACENT_LEDGER_DOUBLE_FAULTS[2]
    assert scenario.row_id.startswith("N3-")

    double_result, double_resolution = _null_bind_short_path_literal_case()
    double_injections = (scenario.left, scenario.right)
    double_probe = _PhaseProbeV2()
    _apply_before_call_faults(double_result, double_injections, double_probe)
    double_verifier = _verifier_with_phase_profile(
        double_result,
        double_injections,
        double_probe,
    )
    with pytest.raises(V2IntegrityError) as double_captured:
        double_verifier(double_result, double_resolution)

    _assert_exact_integrity_error(double_captured, scenario.double_expectation)
    assert tuple(double_probe.fault_receipts) == scenario.double_receipts
    assert double_probe.context_reads == scenario.double_context_reads == 1
    assert double_probe.context_fault_calls == scenario.double_context_fault_calls == 0
    assert double_probe.context_fault_ids == []
    assert double_probe.token_verifier_calls == 0


def test_null_bind_priority_row_n3_fresh_right_only_context_read_2_is_live() -> None:
    scenario = NULL_BIND_ADJACENT_LEDGER_DOUBLE_FAULTS[2]
    assert scenario.row_id.startswith("N3-")

    right_result, right_resolution = _null_bind_short_path_literal_case()
    right_injections = (scenario.right,)
    right_probe = _PhaseProbeV2()
    _apply_before_call_faults(right_result, right_injections, right_probe)
    right_verifier = _verifier_with_phase_profile(
        right_result,
        right_injections,
        right_probe,
    )
    with pytest.raises(V2IntegrityError) as right_captured:
        right_verifier(right_result, right_resolution)

    _assert_exact_integrity_error(right_captured, scenario.right_expectation)
    assert tuple(right_probe.fault_receipts) == scenario.right_receipts
    assert right_probe.context_reads == scenario.right_context_reads == 2
    assert right_probe.context_fault_calls == scenario.right_context_fault_calls == 1
    assert right_probe.context_fault_ids == [
        "null-bind.context.read-2.tie-tolerance.0.25"
    ]
    assert right_probe.token_verifier_calls == 0


def test_null_bind_routing_profiler_has_live_complete_positive_control() -> None:
    result, resolution = _adjacent_fault_complete_case()
    replicate_receipt: list[str] = []

    returned = _verify_with_replicate_path_profile(
        contracts_v2.verify_calibration_result,
        result,
        resolution,
        replicate_receipt=replicate_receipt,
        sentinel_armed=False,
    )

    assert returned is None
    assert tuple(replicate_receipt) == (
        "replicate 0",
        "replicate 1",
        "replicate 2",
    )


def test_null_bind_routing_dataflow_only_reaches_later_count_fault() -> None:
    assert _NULL_BIND_ROUTING_CLASSIFICATION == (
        "NULL_BIND_ROUTING_DATAFLOW_ONLY_NOT_ERROR_BOUNDARY"
    )
    result, resolution = _null_bind_short_path_literal_case()
    _install_null_bind_adverse_observed_decoys(result)
    _assert_null_bind_adverse_observed_decoys_installed(result)
    _fault_null_bind_exact_count_types(result)
    _assert_null_bind_exact_count_types_fault_applied(result)
    replicate_receipt: list[str] = []
    probe = _PhaseProbeV2()
    verifier = _verifier_with_phase_profile(result, (), probe)

    with pytest.raises(V2IntegrityError) as captured:
        _verify_with_replicate_path_profile(
            verifier,
            result,
            resolution,
            replicate_receipt=replicate_receipt,
            sentinel_armed=True,
        )

    assert type(captured.value) is V2IntegrityError
    assert str(captured.value) == (
        "calibration counts must be exact built-in integers"
    )
    assert captured.value.__cause__ is None
    assert replicate_receipt == []
    assert probe.context_reads == 1
    assert probe.token_verifier_calls == 0


def test_null_bind_adverse_decoys_are_skipped_then_rejected_by_terminal() -> None:
    result, resolution = _null_bind_short_path_literal_case()
    _install_null_bind_adverse_observed_decoys(result)
    _assert_null_bind_adverse_observed_decoys_installed(result)
    assert type(result.exceedance_count) is int
    assert result.exceedance_count == 0
    assert type(result.failure_count) is int
    assert result.failure_count == 0
    replicate_receipt: list[str] = []
    probe = _PhaseProbeV2()
    verifier = _verifier_with_phase_profile(result, (), probe)

    with pytest.raises(V2IntegrityError) as captured:
        _verify_with_replicate_path_profile(
            verifier,
            result,
            resolution,
            replicate_receipt=replicate_receipt,
            sentinel_armed=True,
        )

    assert type(captured.value) is V2IntegrityError
    assert str(captured.value) == "NULL_BIND terminal result is invalid"
    assert captured.value.__cause__ is None
    assert replicate_receipt == []
    assert probe.context_reads == 1
    assert probe.token_verifier_calls == 0


def test_null_bind_exact_valid_terminal_control_returns_without_scans() -> None:
    result, resolution = _null_bind_short_path_literal_case()
    assert result.status is RunStatus.NOT_EVALUABLE
    assert result.failure_stage is RunFailureStage.NULL_BIND
    assert result.observed_results == ()
    assert result.observed_selection is None
    assert result.replicates == ()
    assert type(result.exceedance_count) is int
    assert result.exceedance_count == 0
    assert type(result.failure_count) is int
    assert result.failure_count == 0
    assert result.p_value is None
    assert result.exceedance_bound_low is None
    assert result.exceedance_bound_high is None
    assert result.reject_null is None
    replicate_receipt: list[str] = []
    probe = _PhaseProbeV2()
    verifier = _verifier_with_phase_profile(result, (), probe)

    returned = _verify_with_replicate_path_profile(
        verifier,
        result,
        resolution,
        replicate_receipt=replicate_receipt,
        sentinel_armed=True,
    )

    assert returned is None
    assert probe.context_reads == 2
    assert probe.context_fault_calls == 0
    assert probe.token_verifier_calls == 0
    assert probe.token_fault_calls == 0
    assert probe.token_completion_calls == 0
    assert replicate_receipt == []


def test_observed_scan_adjacent_ledger_has_exact_four_genuine_priority_rows() -> None:
    assert (
        tuple(
            (scenario.row_id, scenario.left.label, scenario.right.label)
            for scenario in OBSERVED_SCAN_ADJACENT_LEDGER_DOUBLE_FAULTS
        )
        == _EXPECTED_OBSERVED_SCAN_ADJACENT_ROWS
    )
    assert len(OBSERVED_SCAN_ADJACENT_LEDGER_DOUBLE_FAULTS) == 4
    assert len({scenario.row_id for scenario in OBSERVED_SCAN_ADJACENT_LEDGER_DOUBLE_FAULTS}) == 4
    assert all(
        scenario.left is not scenario.right
        for scenario in OBSERVED_SCAN_ADJACENT_LEDGER_DOUBLE_FAULTS
    )
    assert _OBSERVED_SCAN_ROUTING_CLASSIFICATION not in {
        scenario.row_id for scenario in OBSERVED_SCAN_ADJACENT_LEDGER_DOUBLE_FAULTS
    }


@pytest.mark.parametrize(
    "scenario",
    OBSERVED_SCAN_ADJACENT_LEDGER_DOUBLE_FAULTS[:3],
    ids=lambda scenario: scenario.row_id,
)
def test_observed_scan_priority_rows_o1_o2_o3_retain_exact_double_fault_priority(
    scenario: ObservedScanAdjacentFaultScenarioV2,
) -> None:
    result, resolution = _observed_statistic_scan_short_path_literal_case()
    _assert_observed_statistic_scan_literal_contract(result)
    injections = (scenario.left, scenario.right)
    probe = _PhaseProbeV2()
    _apply_before_call_faults(result, injections, probe)
    verifier = _verifier_with_phase_profile(result, injections, probe)

    with pytest.raises(V2IntegrityError) as captured:
        verifier(result, resolution)

    _assert_exact_integrity_error(captured, scenario.double_expectation)
    assert tuple(probe.fault_receipts) == scenario.double_receipts
    assert probe.context_reads == scenario.double_context_reads
    assert probe.context_fault_calls == scenario.double_context_fault_calls
    assert probe.token_verifier_calls == 0
    assert probe.token_fault_calls == 0
    assert probe.token_completion_calls == 0


@pytest.mark.parametrize(
    "scenario",
    OBSERVED_SCAN_ADJACENT_LEDGER_DOUBLE_FAULTS[:3],
    ids=lambda scenario: scenario.row_id,
)
def test_observed_scan_priority_rows_o1_o2_o3_fresh_right_only_fault_is_live(
    scenario: ObservedScanAdjacentFaultScenarioV2,
) -> None:
    result, resolution = _observed_statistic_scan_short_path_literal_case()
    _assert_observed_statistic_scan_literal_contract(result)
    injections = (scenario.right,)
    probe = _PhaseProbeV2()
    _apply_before_call_faults(result, injections, probe)
    verifier = _verifier_with_phase_profile(result, injections, probe)

    with pytest.raises(V2IntegrityError) as captured:
        verifier(result, resolution)

    _assert_exact_integrity_error(captured, scenario.right_expectation)
    assert tuple(probe.fault_receipts) == scenario.right_receipts
    assert probe.context_reads == scenario.right_context_reads
    assert probe.context_fault_calls == scenario.right_context_fault_calls
    assert probe.token_verifier_calls == 0
    assert probe.token_fault_calls == 0
    assert probe.token_completion_calls == 0


def test_observed_scan_priority_row_o4_double_stops_before_context_read_2() -> None:
    scenario = OBSERVED_SCAN_ADJACENT_LEDGER_DOUBLE_FAULTS[3]
    assert scenario.row_id.startswith("O4-")
    result, resolution = _observed_statistic_scan_short_path_literal_case()
    _assert_observed_statistic_scan_literal_contract(result)
    injections = (scenario.left, scenario.right)
    probe = _PhaseProbeV2()
    _apply_before_call_faults(result, injections, probe)
    verifier = _verifier_with_phase_profile(result, injections, probe)

    with pytest.raises(V2IntegrityError) as captured:
        verifier(result, resolution)

    _assert_exact_integrity_error(captured, scenario.double_expectation)
    assert tuple(probe.fault_receipts) == scenario.double_receipts
    assert probe.context_reads == scenario.double_context_reads == 1
    assert probe.context_fault_calls == scenario.double_context_fault_calls == 0
    assert probe.context_fault_ids == []
    assert probe.token_verifier_calls == 0


def test_observed_scan_priority_row_o4_fresh_right_only_context_read_2_is_live() -> None:
    scenario = OBSERVED_SCAN_ADJACENT_LEDGER_DOUBLE_FAULTS[3]
    assert scenario.row_id.startswith("O4-")
    result, resolution = _observed_statistic_scan_short_path_literal_case()
    _assert_observed_statistic_scan_literal_contract(result)
    injections = (scenario.right,)
    probe = _PhaseProbeV2()
    _apply_before_call_faults(result, injections, probe)
    verifier = _verifier_with_phase_profile(result, injections, probe)

    with pytest.raises(V2IntegrityError) as captured:
        verifier(result, resolution)

    _assert_exact_integrity_error(captured, scenario.right_expectation)
    assert tuple(probe.fault_receipts) == scenario.right_receipts
    assert probe.context_reads == scenario.right_context_reads == 2
    assert probe.context_fault_calls == scenario.right_context_fault_calls == 1
    assert probe.context_fault_ids == ["observed-scan.context.read-2.tie-tolerance.0.25"]
    assert probe.token_verifier_calls == 0


def test_observed_scan_exact_valid_acceptance_profiles_only_failure_vector() -> None:
    assert _OBSERVED_SCAN_ROUTING_CLASSIFICATION == (
        "OBSERVED_STATISTIC_SCAN_ROUTING_DATAFLOW_ONLY_NOT_ERROR_BOUNDARY"
    )
    complete_result, complete_resolution = _adjacent_fault_complete_case()
    complete_probe = _PhaseProbeV2()
    complete_verifier = _verifier_with_phase_profile(
        complete_result,
        (),
        complete_probe,
    )
    complete_receipts: list[_TraversalReceiptV2] = []

    complete_returned = _verify_with_observed_short_path_profile(
        complete_verifier,
        complete_result,
        complete_resolution,
        traversal_receipts=complete_receipts,
    )

    assert complete_returned is None
    assert tuple(complete_receipts) == (
        _TraversalReceiptV2(
            phase=_TraversalReceiptPhaseV2.OBSERVED_STATISTIC,
            subject="observed result vector statistic",
        ),
        _TraversalReceiptV2(
            phase=_TraversalReceiptPhaseV2.OBSERVED_STATISTIC,
            subject="observed result vector statistic",
        ),
        _TraversalReceiptV2(
            phase=_TraversalReceiptPhaseV2.OBSERVED_SELECTION_REREAD,
            subject="observed selection",
        ),
        _TraversalReceiptV2(
            phase=_TraversalReceiptPhaseV2.REPLICATE,
            subject="replicate 0",
        ),
        _TraversalReceiptV2(
            phase=_TraversalReceiptPhaseV2.REPLICATE,
            subject="replicate 1",
        ),
        _TraversalReceiptV2(
            phase=_TraversalReceiptPhaseV2.REPLICATE,
            subject="replicate 2",
        ),
    )
    assert complete_probe.context_reads == 2
    assert complete_probe.token_verifier_calls == 3
    assert complete_probe.token_completion_calls == 3

    observed_result, observed_resolution = _observed_statistic_scan_short_path_literal_case()
    _assert_observed_statistic_scan_literal_contract(observed_result)
    observed_probe = _PhaseProbeV2()
    observed_verifier = _verifier_with_phase_profile(
        observed_result,
        (),
        observed_probe,
    )
    observed_receipts: list[_TraversalReceiptV2] = []

    observed_returned = _verify_with_observed_short_path_profile(
        observed_verifier,
        observed_result,
        observed_resolution,
        traversal_receipts=observed_receipts,
    )

    assert observed_returned is None
    assert tuple(observed_receipts) == (
        _TraversalReceiptV2(
            phase=_TraversalReceiptPhaseV2.OBSERVED_STATISTIC,
            subject="observed result vector statistic",
        ),
        _TraversalReceiptV2(
            phase=_TraversalReceiptPhaseV2.OBSERVED_STATISTIC,
            subject="observed result vector statistic",
        ),
    )
    assert observed_result.observed_selection is None
    assert observed_result.replicates == ()
    assert observed_probe.context_reads == 2
    assert observed_probe.token_verifier_calls == 0
    assert observed_probe.token_fault_calls == 0
    assert observed_probe.token_completion_calls == 0


def test_public_verifier_has_exact_picklable_function_identity() -> None:
    verifier = contracts_v2.verify_calibration_result
    signature = inspect.signature(verifier)

    assert str(signature) == "(result: 'object', resolution: 'object', /) -> 'None'"
    assert tuple(parameter.kind for parameter in signature.parameters.values()) == (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_ONLY,
    )
    assert verifier.__annotations__ == {
        "result": "object",
        "resolution": "object",
        "return": "None",
    }
    assert verifier.__module__ == "selcal.contracts_v2"
    assert verifier.__name__ == "verify_calibration_result"
    assert verifier.__qualname__ == "verify_calibration_result"
    assert verifier.__defaults__ is None
    assert verifier.__kwdefaults__ is None
    assert verifier.__doc__ is not None
    assert verifier.__doc__.strip()
    assert pickle.loads(pickle.dumps(verifier)) is verifier
    assert selcal.verify_calibration_result is verifier
    assert calibration_v2.verify_calibration_result is verifier
    assert contracts_v2.verify_calibration_result is verifier


def test_verifier_result_dataclasses_have_exact_field_order_and_counts() -> None:
    calibration_result_fields = tuple(
        field.name for field in fields(contracts_v2.CalibrationResult)
    )
    replicate_outcome_fields = tuple(
        field.name for field in fields(contracts_v2.ReplicateOutcome)
    )
    statistic_result_fields = tuple(
        field.name for field in fields(contracts_v2.StatisticResult)
    )
    selection_result_fields = tuple(
        field.name for field in fields(contracts_v2.SelectionResult)
    )

    assert calibration_result_fields == (
        "status",
        "failure_stage",
        "semantic_input_sha256",
        "scientific_plan_sha256",
        "planned_replicates",
        "alpha",
        "observed_results",
        "observed_selection",
        "replicates",
        "exceedance_count",
        "failure_count",
        "p_value",
        "exceedance_bound_low",
        "exceedance_bound_high",
        "reject_null",
        "diagnostics",
    )
    assert replicate_outcome_fields == (
        "replicate_id",
        "seed_digest_sha256",
        "status",
        "failure_stage",
        "transform_token",
        "statistic_results",
        "selection",
        "diagnostics",
    )
    assert statistic_result_fields == (
        "candidate_id",
        "estimate",
        "selection_score",
        "support_n",
        "validity",
        "diagnostics",
        "backend_identity",
        "preprocessing_identity",
    )
    assert selection_result_fields == (
        "selected_candidate",
        "selected_index",
        "decision_statistic",
        "tied_candidates",
    )
    assert (
        len(calibration_result_fields),
        len(replicate_outcome_fields),
        len(statistic_result_fields),
        len(selection_result_fields),
    ) == (16, 8, 8, 4)


def test_public_verifier_accepts_complete_literal_selection_and_plus_one_kat() -> None:
    resolution = _literal_resolution()
    token = _literal_identity_token()

    observed_results = (
        _literal_statistic_result(
            candidate_id=1,
            estimate=3.0,
            selection_score=3.0,
            validity=Validity.VALID,
        ),
        _literal_statistic_result(
            candidate_id=2,
            estimate=1.0,
            selection_score=1.0,
            validity=Validity.VALID,
        ),
    )
    observed_selection = _literal_selection(
        selected_candidate=1,
        selected_index=0,
        decision_statistic=3.0,
        tied_candidates=(1,),
    )

    strict_results = (
        _literal_statistic_result(
            candidate_id=1,
            estimate=4.0,
            selection_score=4.0,
            validity=Validity.VALID,
        ),
        _literal_statistic_result(
            candidate_id=2,
            estimate=2.0,
            selection_score=2.0,
            validity=Validity.VALID,
        ),
    )
    strict_selection = _literal_selection(
        selected_candidate=1,
        selected_index=0,
        decision_statistic=4.0,
        tied_candidates=(1,),
    )

    tolerance_boundary_results = (
        _literal_statistic_result(
            candidate_id=1,
            estimate=0.0,
            selection_score=0.0,
            validity=Validity.VALID,
        ),
        _literal_statistic_result(
            candidate_id=2,
            estimate=1e-12,
            selection_score=1e-12,
            validity=Validity.VALID,
        ),
    )
    tolerance_boundary_selection = _literal_selection(
        selected_candidate=1,
        selected_index=0,
        decision_statistic=1e-12,
        tied_candidates=(1, 2),
    )

    zero_results = (
        _literal_statistic_result(
            candidate_id=1,
            estimate=0.0,
            selection_score=0.0,
            validity=Validity.VALID,
        ),
        _literal_statistic_result(
            candidate_id=2,
            estimate=-1.0,
            selection_score=-1.0,
            validity=Validity.VALID,
        ),
    )
    zero_selection = _literal_selection(
        selected_candidate=1,
        selected_index=0,
        decision_statistic=0.0,
        tied_candidates=(1,),
    )

    result = _literal_calibration_result(
        status=RunStatus.COMPLETE,
        failure_stage=None,
        observed_results=observed_results,
        observed_selection=observed_selection,
        replicates=(
            _literal_replicate(
                replicate_id=0,
                seed_digest_sha256=_SEED_DIGEST_0,
                status=ReplicateStatus.COMPLETE,
                failure_stage=None,
                transform_token=token,
                statistic_results=strict_results,
                selection=strict_selection,
            ),
            _literal_replicate(
                replicate_id=1,
                seed_digest_sha256=_SEED_DIGEST_1,
                status=ReplicateStatus.COMPLETE,
                failure_stage=None,
                transform_token=token,
                statistic_results=tolerance_boundary_results,
                selection=tolerance_boundary_selection,
            ),
            _literal_replicate(
                replicate_id=2,
                seed_digest_sha256=_SEED_DIGEST_2,
                status=ReplicateStatus.COMPLETE,
                failure_stage=None,
                transform_token=token,
                statistic_results=zero_results,
                selection=zero_selection,
            ),
        ),
        exceedance_count=1,
        failure_count=0,
        p_value=0.5,
        bound_low=None,
        bound_high=None,
        reject_null=True,
    )

    assert strict_selection.selected_index == 0
    assert strict_selection.tied_candidates == (1,)
    assert (
        tolerance_boundary_results[1].selection_score
        - tolerance_boundary_results[0].selection_score
        == resolution.plan.tie_tolerance
        == 1e-12
    )
    assert tolerance_boundary_selection.selected_index == 0
    assert tolerance_boundary_selection.selected_candidate == 1
    assert tolerance_boundary_selection.tied_candidates == (1, 2)
    assert zero_results[0].estimate is not None
    assert zero_results[0].selection_score is not None
    assert math.copysign(1.0, zero_results[0].estimate) == 1.0
    assert math.copysign(1.0, zero_results[0].selection_score) == 1.0
    assert math.copysign(1.0, zero_selection.decision_statistic) == 1.0
    assert result.exceedance_count == 1
    assert result.p_value == 0.5 == (1 + 1) / (3 + 1)

    contracts_v2.verify_calibration_result(result, resolution)


@pytest.mark.parametrize(
    ("estimate_zero", "score_zero", "decision_zero", "expected_signs"),
    (
        pytest.param(-0.0, 0.0, 0.0, (-1.0, 1.0, 1.0), id="negative-estimate-only"),
        pytest.param(0.0, -0.0, 0.0, (1.0, -1.0, 1.0), id="negative-score-only"),
        pytest.param(0.0, 0.0, -0.0, (1.0, 1.0, -1.0), id="negative-decision-only"),
        pytest.param(-0.0, -0.0, -0.0, (-1.0, -1.0, -1.0), id="all-negative-zero"),
    ),
)
def test_public_verifier_characterizes_zero_sign_as_unobservable(
    estimate_zero: float,
    score_zero: float,
    decision_zero: float,
    expected_signs: tuple[float, float, float],
) -> None:
    result, resolution, zero_result, zero_selection = _literal_zero_sign_characterization_case(
        estimate_zero=estimate_zero,
        score_zero=score_zero,
        decision_zero=decision_zero,
    )
    assert zero_result.estimate is not None
    assert zero_result.selection_score is not None
    assert (
        math.copysign(1.0, zero_result.estimate),
        math.copysign(1.0, zero_result.selection_score),
        math.copysign(1.0, zero_selection.decision_statistic),
    ) == expected_signs

    # Adverse characterization: the two-argument verifier returns no evidence and
    # float equality cannot distinguish signed zero, so sign normalization is not
    # observable on this public surface.  A +0 returned-evidence KAT belongs in the
    # primitive selection-leaf RED/GREEN slice.
    assert contracts_v2.verify_calibration_result(result, resolution) is None


def test_public_verifier_accepts_mixed_failure_literal_bounds_kat() -> None:
    resolution = _literal_resolution()
    token = _literal_identity_token()

    observed_results = (
        _literal_statistic_result(
            candidate_id=1,
            estimate=3.0,
            selection_score=3.0,
            validity=Validity.VALID,
        ),
        _literal_statistic_result(
            candidate_id=2,
            estimate=1.0,
            selection_score=1.0,
            validity=Validity.VALID,
        ),
    )
    observed_selection = _literal_selection(
        selected_candidate=1,
        selected_index=0,
        decision_statistic=3.0,
        tied_candidates=(1,),
    )

    exceedance_results = (
        _literal_statistic_result(
            candidate_id=1,
            estimate=4.0,
            selection_score=4.0,
            validity=Validity.VALID,
        ),
        _literal_statistic_result(
            candidate_id=2,
            estimate=2.0,
            selection_score=2.0,
            validity=Validity.VALID,
        ),
    )
    exceedance_selection = _literal_selection(
        selected_candidate=1,
        selected_index=0,
        decision_statistic=4.0,
        tied_candidates=(1,),
    )

    mixed_failure_results = (
        _literal_statistic_result(
            candidate_id=1,
            estimate=2.0,
            selection_score=None,
            validity=Validity.VALID,
        ),
        _literal_statistic_result(
            candidate_id=2,
            estimate=None,
            selection_score=None,
            validity=Validity.ANALYTIC_FAILURE,
            diagnostics=("literal_mixed_analytic_failure",),
        ),
    )

    nonexceedance_results = (
        _literal_statistic_result(
            candidate_id=1,
            estimate=2.0,
            selection_score=2.0,
            validity=Validity.VALID,
        ),
        _literal_statistic_result(
            candidate_id=2,
            estimate=1.0,
            selection_score=1.0,
            validity=Validity.VALID,
        ),
    )
    nonexceedance_selection = _literal_selection(
        selected_candidate=1,
        selected_index=0,
        decision_statistic=2.0,
        tied_candidates=(1,),
    )

    result = _literal_calibration_result(
        status=RunStatus.NOT_EVALUABLE,
        failure_stage=RunFailureStage.REPLICATE_EXECUTION,
        observed_results=observed_results,
        observed_selection=observed_selection,
        replicates=(
            _literal_replicate(
                replicate_id=0,
                seed_digest_sha256=_SEED_DIGEST_0,
                status=ReplicateStatus.COMPLETE,
                failure_stage=None,
                transform_token=token,
                statistic_results=exceedance_results,
                selection=exceedance_selection,
            ),
            _literal_replicate(
                replicate_id=1,
                seed_digest_sha256=_SEED_DIGEST_1,
                status=ReplicateStatus.ANALYTIC_FAILURE,
                failure_stage=ReplicateFailureStage.STATISTIC_SCAN,
                transform_token=token,
                statistic_results=mixed_failure_results,
                selection=None,
                diagnostics=("literal_mixed_analytic_failure",),
            ),
            _literal_replicate(
                replicate_id=2,
                seed_digest_sha256=_SEED_DIGEST_2,
                status=ReplicateStatus.COMPLETE,
                failure_stage=None,
                transform_token=token,
                statistic_results=nonexceedance_results,
                selection=nonexceedance_selection,
            ),
        ),
        exceedance_count=1,
        failure_count=1,
        p_value=None,
        bound_low=0.5,
        bound_high=0.75,
        reject_null=None,
        diagnostics=("literal_replicate_execution_failure",),
    )

    assert sum(
        item.validity is Validity.ANALYTIC_FAILURE for item in mixed_failure_results
    ) == 1
    assert result.exceedance_count == 1
    assert result.failure_count == 1
    assert result.exceedance_bound_low == 0.5 == (1 + 1) / (3 + 1)
    assert result.exceedance_bound_high == 0.75 == (1 + 1 + 1) / (3 + 1)

    contracts_v2.verify_calibration_result(result, resolution)


def test_public_verifier_accepts_all_positions_failure_literal_bounds_kat() -> None:
    resolution = _literal_resolution()
    token = _literal_identity_token()

    observed_results = (
        _literal_statistic_result(
            candidate_id=1,
            estimate=3.0,
            selection_score=3.0,
            validity=Validity.VALID,
        ),
        _literal_statistic_result(
            candidate_id=2,
            estimate=1.0,
            selection_score=1.0,
            validity=Validity.VALID,
        ),
    )
    observed_selection = _literal_selection(
        selected_candidate=1,
        selected_index=0,
        decision_statistic=3.0,
        tied_candidates=(1,),
    )

    first_nonexceedance_results = (
        _literal_statistic_result(
            candidate_id=1,
            estimate=2.0,
            selection_score=2.0,
            validity=Validity.VALID,
        ),
        _literal_statistic_result(
            candidate_id=2,
            estimate=1.0,
            selection_score=1.0,
            validity=Validity.VALID,
        ),
    )
    first_nonexceedance_selection = _literal_selection(
        selected_candidate=1,
        selected_index=0,
        decision_statistic=2.0,
        tied_candidates=(1,),
    )

    all_positions_failure_results = (
        _literal_statistic_result(
            candidate_id=1,
            estimate=None,
            selection_score=None,
            validity=Validity.ANALYTIC_FAILURE,
            diagnostics=("literal_all_positions_analytic_failure",),
        ),
        _literal_statistic_result(
            candidate_id=2,
            estimate=None,
            selection_score=None,
            validity=Validity.ANALYTIC_FAILURE,
            diagnostics=("literal_all_positions_analytic_failure",),
        ),
    )

    second_nonexceedance_results = (
        _literal_statistic_result(
            candidate_id=1,
            estimate=1.0,
            selection_score=1.0,
            validity=Validity.VALID,
        ),
        _literal_statistic_result(
            candidate_id=2,
            estimate=0.0,
            selection_score=0.0,
            validity=Validity.VALID,
        ),
    )
    second_nonexceedance_selection = _literal_selection(
        selected_candidate=1,
        selected_index=0,
        decision_statistic=1.0,
        tied_candidates=(1,),
    )

    result = _literal_calibration_result(
        status=RunStatus.NOT_EVALUABLE,
        failure_stage=RunFailureStage.REPLICATE_EXECUTION,
        observed_results=observed_results,
        observed_selection=observed_selection,
        replicates=(
            _literal_replicate(
                replicate_id=0,
                seed_digest_sha256=_SEED_DIGEST_0,
                status=ReplicateStatus.COMPLETE,
                failure_stage=None,
                transform_token=token,
                statistic_results=first_nonexceedance_results,
                selection=first_nonexceedance_selection,
            ),
            _literal_replicate(
                replicate_id=1,
                seed_digest_sha256=_SEED_DIGEST_1,
                status=ReplicateStatus.ANALYTIC_FAILURE,
                failure_stage=ReplicateFailureStage.STATISTIC_SCAN,
                transform_token=token,
                statistic_results=all_positions_failure_results,
                selection=None,
                diagnostics=("literal_all_positions_analytic_failure",),
            ),
            _literal_replicate(
                replicate_id=2,
                seed_digest_sha256=_SEED_DIGEST_2,
                status=ReplicateStatus.COMPLETE,
                failure_stage=None,
                transform_token=token,
                statistic_results=second_nonexceedance_results,
                selection=second_nonexceedance_selection,
            ),
        ),
        exceedance_count=0,
        failure_count=1,
        p_value=None,
        bound_low=0.25,
        bound_high=0.5,
        reject_null=None,
        diagnostics=("literal_replicate_execution_failure",),
    )

    assert all(
        item.validity is Validity.ANALYTIC_FAILURE for item in all_positions_failure_results
    )
    assert len(all_positions_failure_results) == 2
    assert result.failure_count == 1
    assert result.exceedance_count == 0
    assert result.exceedance_bound_low == 0.25 == (1 + 0) / (3 + 1)
    assert result.exceedance_bound_high == 0.5 == (1 + 0 + 1) / (3 + 1)

    contracts_v2.verify_calibration_result(result, resolution)


def test_public_verifier_rejects_independent_valid_but_scored_failure_literal() -> None:
    resolution = _literal_resolution()
    token = _literal_identity_token()

    observed_results = (
        _literal_statistic_result(
            candidate_id=1,
            estimate=3.0,
            selection_score=3.0,
            validity=Validity.VALID,
        ),
        _literal_statistic_result(
            candidate_id=2,
            estimate=1.0,
            selection_score=1.0,
            validity=Validity.VALID,
        ),
    )
    observed_selection = _literal_selection(
        selected_candidate=1,
        selected_index=0,
        decision_statistic=3.0,
        tied_candidates=(1,),
    )

    complete_exceedance_results = (
        _literal_statistic_result(
            candidate_id=1,
            estimate=4.0,
            selection_score=4.0,
            validity=Validity.VALID,
        ),
        _literal_statistic_result(
            candidate_id=2,
            estimate=2.0,
            selection_score=2.0,
            validity=Validity.VALID,
        ),
    )
    complete_exceedance_selection = _literal_selection(
        selected_candidate=1,
        selected_index=0,
        decision_statistic=4.0,
        tied_candidates=(1,),
    )

    valid_but_scored_failure_results = (
        _literal_statistic_result(
            candidate_id=1,
            estimate=2.0,
            selection_score=2.0,
            validity=Validity.VALID,
        ),
        _literal_statistic_result(
            candidate_id=2,
            estimate=None,
            selection_score=None,
            validity=Validity.ANALYTIC_FAILURE,
            diagnostics=("literal_analytic_failure",),
        ),
    )

    complete_nonexceedance_results = (
        _literal_statistic_result(
            candidate_id=1,
            estimate=2.0,
            selection_score=2.0,
            validity=Validity.VALID,
        ),
        _literal_statistic_result(
            candidate_id=2,
            estimate=1.0,
            selection_score=1.0,
            validity=Validity.VALID,
        ),
    )
    complete_nonexceedance_selection = _literal_selection(
        selected_candidate=1,
        selected_index=0,
        decision_statistic=2.0,
        tied_candidates=(1,),
    )

    result = _literal_calibration_result(
        status=RunStatus.NOT_EVALUABLE,
        failure_stage=RunFailureStage.REPLICATE_EXECUTION,
        observed_results=observed_results,
        observed_selection=observed_selection,
        replicates=(
            _literal_replicate(
                replicate_id=0,
                seed_digest_sha256=_SEED_DIGEST_0,
                status=ReplicateStatus.COMPLETE,
                failure_stage=None,
                transform_token=token,
                statistic_results=complete_exceedance_results,
                selection=complete_exceedance_selection,
            ),
            _literal_replicate(
                replicate_id=1,
                seed_digest_sha256=_SEED_DIGEST_1,
                status=ReplicateStatus.ANALYTIC_FAILURE,
                failure_stage=ReplicateFailureStage.STATISTIC_SCAN,
                transform_token=token,
                statistic_results=valid_but_scored_failure_results,
                selection=None,
                diagnostics=("literal_analytic_failure",),
            ),
            _literal_replicate(
                replicate_id=2,
                seed_digest_sha256=_SEED_DIGEST_2,
                status=ReplicateStatus.COMPLETE,
                failure_stage=None,
                transform_token=token,
                statistic_results=complete_nonexceedance_results,
                selection=complete_nonexceedance_selection,
            ),
        ),
        exceedance_count=1,
        failure_count=1,
        p_value=None,
        bound_low=0.5,
        bound_high=0.75,
        reject_null=None,
        diagnostics=("literal_replicate_execution_failure",),
    )

    with pytest.raises(V2IntegrityError) as captured:
        contracts_v2.verify_calibration_result(result, resolution)

    assert type(captured.value) is V2IntegrityError
    assert str(captured.value) == (
        "replicate 1 result vector failure vector must remain unscored"
    )
    assert captured.value.__cause__ is None


def test_public_verifier_accepts_circular_shift_raw_literal_token_owner_kat() -> None:
    resolution = resolve_plan_v2(
        PlanRequestV2(
            candidates=(1, 2),
            statistic_name="lagged_pearson_v1",
            statistic_params={},
            selection_rule="max_upper",
            null_name="circular_shift_v2",
            null_params={"min_shift": 1},
            replicates=1,
            alpha=0.75,
            tie_tolerance=1e-12,
            root_seed=101,
        )
    )

    observed_candidate_1 = object.__new__(StatisticResult)
    object.__setattr__(observed_candidate_1, "candidate_id", 1)
    object.__setattr__(observed_candidate_1, "estimate", 3.0)
    object.__setattr__(observed_candidate_1, "selection_score", 3.0)
    object.__setattr__(observed_candidate_1, "support_n", 4)
    object.__setattr__(observed_candidate_1, "validity", Validity.VALID)
    object.__setattr__(observed_candidate_1, "diagnostics", ())
    object.__setattr__(observed_candidate_1, "backend_identity", _BACKEND_IDENTITY)
    object.__setattr__(
        observed_candidate_1,
        "preprocessing_identity",
        _PREPROCESSING_IDENTITY,
    )
    observed_candidate_2 = object.__new__(StatisticResult)
    object.__setattr__(observed_candidate_2, "candidate_id", 2)
    object.__setattr__(observed_candidate_2, "estimate", 1.0)
    object.__setattr__(observed_candidate_2, "selection_score", 1.0)
    object.__setattr__(observed_candidate_2, "support_n", 4)
    object.__setattr__(observed_candidate_2, "validity", Validity.VALID)
    object.__setattr__(observed_candidate_2, "diagnostics", ())
    object.__setattr__(observed_candidate_2, "backend_identity", _BACKEND_IDENTITY)
    object.__setattr__(
        observed_candidate_2,
        "preprocessing_identity",
        _PREPROCESSING_IDENTITY,
    )
    observed_results = (observed_candidate_1, observed_candidate_2)

    observed_selection = object.__new__(SelectionResult)
    object.__setattr__(observed_selection, "selected_candidate", 1)
    object.__setattr__(observed_selection, "selected_index", 0)
    object.__setattr__(observed_selection, "decision_statistic", 3.0)
    object.__setattr__(observed_selection, "tied_candidates", (1,))

    replicate_candidate_1 = object.__new__(StatisticResult)
    object.__setattr__(replicate_candidate_1, "candidate_id", 1)
    object.__setattr__(replicate_candidate_1, "estimate", 4.0)
    object.__setattr__(replicate_candidate_1, "selection_score", 4.0)
    object.__setattr__(replicate_candidate_1, "support_n", 4)
    object.__setattr__(replicate_candidate_1, "validity", Validity.VALID)
    object.__setattr__(replicate_candidate_1, "diagnostics", ())
    object.__setattr__(replicate_candidate_1, "backend_identity", _BACKEND_IDENTITY)
    object.__setattr__(
        replicate_candidate_1,
        "preprocessing_identity",
        _PREPROCESSING_IDENTITY,
    )
    replicate_candidate_2 = object.__new__(StatisticResult)
    object.__setattr__(replicate_candidate_2, "candidate_id", 2)
    object.__setattr__(replicate_candidate_2, "estimate", 2.0)
    object.__setattr__(replicate_candidate_2, "selection_score", 2.0)
    object.__setattr__(replicate_candidate_2, "support_n", 4)
    object.__setattr__(replicate_candidate_2, "validity", Validity.VALID)
    object.__setattr__(replicate_candidate_2, "diagnostics", ())
    object.__setattr__(replicate_candidate_2, "backend_identity", _BACKEND_IDENTITY)
    object.__setattr__(
        replicate_candidate_2,
        "preprocessing_identity",
        _PREPROCESSING_IDENTITY,
    )
    replicate_results = (replicate_candidate_1, replicate_candidate_2)

    replicate_selection = object.__new__(SelectionResult)
    object.__setattr__(replicate_selection, "selected_candidate", 1)
    object.__setattr__(replicate_selection, "selected_index", 0)
    object.__setattr__(replicate_selection, "decision_statistic", 4.0)
    object.__setattr__(replicate_selection, "tied_candidates", (1,))

    state = object.__new__(CircularShiftStateV2)
    object.__setattr__(state, "schema", "selcal.circular-shift-state.v2")
    object.__setattr__(state, "shift", 4)

    token = object.__new__(NullTransformToken)
    object.__setattr__(token, "schema", "selcal.null-transform-token.v2")
    object.__setattr__(token, "null_name", "circular_shift_v2")
    object.__setattr__(
        token,
        "null_parameter_sha256",
        "a2e6b608e05886b0c06388750e201c922c21d08e4ca6faf7668742798895dda3",
    )
    object.__setattr__(
        token,
        "semantic_input_sha256",
        "db4106a281134389e084ce9b90d6ba1ddf29e6cf4a6d3e698d85028c1533b362",
    )
    object.__setattr__(
        token,
        "scientific_plan_sha256",
        "99c0890f44bd465f5fc8e9ecf6230da22acd38d1d1edf97795f982cc192f81b0",
    )
    object.__setattr__(
        token,
        "bound_null_owner_sha256",
        "57afd8c1b6d900e647a19d8fd530449a1d7a689e7013b98b6cfaa6dbb71a0ff8",
    )
    object.__setattr__(token, "is_identity", False)
    object.__setattr__(token, "state", state)

    replicate = object.__new__(ReplicateOutcome)
    object.__setattr__(replicate, "replicate_id", 0)
    object.__setattr__(
        replicate,
        "seed_digest_sha256",
        "f1672a719447f6cdb2dbf040550d5f20644e861b6d6a78354416d211c22f7f02",
    )
    object.__setattr__(replicate, "status", ReplicateStatus.COMPLETE)
    object.__setattr__(replicate, "failure_stage", None)
    object.__setattr__(replicate, "transform_token", token)
    object.__setattr__(replicate, "statistic_results", replicate_results)
    object.__setattr__(replicate, "selection", replicate_selection)
    object.__setattr__(replicate, "diagnostics", ())

    result = object.__new__(CalibrationResult)
    object.__setattr__(result, "status", RunStatus.COMPLETE)
    object.__setattr__(result, "failure_stage", None)
    object.__setattr__(
        result,
        "semantic_input_sha256",
        "db4106a281134389e084ce9b90d6ba1ddf29e6cf4a6d3e698d85028c1533b362",
    )
    object.__setattr__(
        result,
        "scientific_plan_sha256",
        "99c0890f44bd465f5fc8e9ecf6230da22acd38d1d1edf97795f982cc192f81b0",
    )
    object.__setattr__(result, "planned_replicates", 1)
    object.__setattr__(result, "alpha", 0.75)
    object.__setattr__(result, "observed_results", observed_results)
    object.__setattr__(result, "observed_selection", observed_selection)
    object.__setattr__(result, "replicates", (replicate,))
    object.__setattr__(result, "exceedance_count", 1)
    object.__setattr__(result, "failure_count", 0)
    object.__setattr__(result, "p_value", 1.0)
    object.__setattr__(result, "exceedance_bound_low", None)
    object.__setattr__(result, "exceedance_bound_high", None)
    object.__setattr__(result, "reject_null", False)
    object.__setattr__(result, "diagnostics", ())

    assert (
        token.schema,
        token.null_name,
        token.null_parameter_sha256,
        token.semantic_input_sha256,
        token.scientific_plan_sha256,
        token.bound_null_owner_sha256,
        token.is_identity,
    ) == (
        "selcal.null-transform-token.v2",
        "circular_shift_v2",
        "a2e6b608e05886b0c06388750e201c922c21d08e4ca6faf7668742798895dda3",
        "db4106a281134389e084ce9b90d6ba1ddf29e6cf4a6d3e698d85028c1533b362",
        "99c0890f44bd465f5fc8e9ecf6230da22acd38d1d1edf97795f982cc192f81b0",
        "57afd8c1b6d900e647a19d8fd530449a1d7a689e7013b98b6cfaa6dbb71a0ff8",
        False,
    )
    assert token.state is state
    assert (state.schema, state.shift) == ("selcal.circular-shift-state.v2", 4)
    assert replicate.seed_digest_sha256 == (
        "f1672a719447f6cdb2dbf040550d5f20644e861b6d6a78354416d211c22f7f02"
    )
    assert result.exceedance_count == 1
    assert result.failure_count == 0
    assert result.p_value == 1.0 == (1 + 1) / (1 + 1)
    assert result.reject_null is False

    contracts_v2.verify_calibration_result(result, resolution)


def test_public_verifier_accepts_block_shuffle_raw_literal_token_owner_kat() -> None:
    resolution = resolve_plan_v2(
        PlanRequestV2(
            candidates=(1, 2),
            statistic_name="lagged_pearson_v1",
            statistic_params={},
            selection_rule="max_upper",
            null_name="block_shuffle_v2",
            null_params={"block_length": 2},
            replicates=1,
            alpha=0.75,
            tie_tolerance=1e-12,
            root_seed=200,
        )
    )

    observed_candidate_1 = object.__new__(StatisticResult)
    object.__setattr__(observed_candidate_1, "candidate_id", 1)
    object.__setattr__(observed_candidate_1, "estimate", 3.0)
    object.__setattr__(observed_candidate_1, "selection_score", 3.0)
    object.__setattr__(observed_candidate_1, "support_n", 4)
    object.__setattr__(observed_candidate_1, "validity", Validity.VALID)
    object.__setattr__(observed_candidate_1, "diagnostics", ())
    object.__setattr__(observed_candidate_1, "backend_identity", _BACKEND_IDENTITY)
    object.__setattr__(
        observed_candidate_1,
        "preprocessing_identity",
        _PREPROCESSING_IDENTITY,
    )
    observed_candidate_2 = object.__new__(StatisticResult)
    object.__setattr__(observed_candidate_2, "candidate_id", 2)
    object.__setattr__(observed_candidate_2, "estimate", 1.0)
    object.__setattr__(observed_candidate_2, "selection_score", 1.0)
    object.__setattr__(observed_candidate_2, "support_n", 4)
    object.__setattr__(observed_candidate_2, "validity", Validity.VALID)
    object.__setattr__(observed_candidate_2, "diagnostics", ())
    object.__setattr__(observed_candidate_2, "backend_identity", _BACKEND_IDENTITY)
    object.__setattr__(
        observed_candidate_2,
        "preprocessing_identity",
        _PREPROCESSING_IDENTITY,
    )
    observed_results = (observed_candidate_1, observed_candidate_2)

    observed_selection = object.__new__(SelectionResult)
    object.__setattr__(observed_selection, "selected_candidate", 1)
    object.__setattr__(observed_selection, "selected_index", 0)
    object.__setattr__(observed_selection, "decision_statistic", 3.0)
    object.__setattr__(observed_selection, "tied_candidates", (1,))

    replicate_candidate_1 = object.__new__(StatisticResult)
    object.__setattr__(replicate_candidate_1, "candidate_id", 1)
    object.__setattr__(replicate_candidate_1, "estimate", 2.0)
    object.__setattr__(replicate_candidate_1, "selection_score", 2.0)
    object.__setattr__(replicate_candidate_1, "support_n", 4)
    object.__setattr__(replicate_candidate_1, "validity", Validity.VALID)
    object.__setattr__(replicate_candidate_1, "diagnostics", ())
    object.__setattr__(replicate_candidate_1, "backend_identity", _BACKEND_IDENTITY)
    object.__setattr__(
        replicate_candidate_1,
        "preprocessing_identity",
        _PREPROCESSING_IDENTITY,
    )
    replicate_candidate_2 = object.__new__(StatisticResult)
    object.__setattr__(replicate_candidate_2, "candidate_id", 2)
    object.__setattr__(replicate_candidate_2, "estimate", 1.0)
    object.__setattr__(replicate_candidate_2, "selection_score", 1.0)
    object.__setattr__(replicate_candidate_2, "support_n", 4)
    object.__setattr__(replicate_candidate_2, "validity", Validity.VALID)
    object.__setattr__(replicate_candidate_2, "diagnostics", ())
    object.__setattr__(replicate_candidate_2, "backend_identity", _BACKEND_IDENTITY)
    object.__setattr__(
        replicate_candidate_2,
        "preprocessing_identity",
        _PREPROCESSING_IDENTITY,
    )
    replicate_results = (replicate_candidate_1, replicate_candidate_2)

    replicate_selection = object.__new__(SelectionResult)
    object.__setattr__(replicate_selection, "selected_candidate", 1)
    object.__setattr__(replicate_selection, "selected_index", 0)
    object.__setattr__(replicate_selection, "decision_statistic", 2.0)
    object.__setattr__(replicate_selection, "tied_candidates", (1,))

    state = object.__new__(BlockShuffleStateV2)
    object.__setattr__(state, "schema", "selcal.block-shuffle-state.v2")
    object.__setattr__(state, "block_order", (1, 2, 0))

    token = object.__new__(NullTransformToken)
    object.__setattr__(token, "schema", "selcal.null-transform-token.v2")
    object.__setattr__(token, "null_name", "block_shuffle_v2")
    object.__setattr__(
        token,
        "null_parameter_sha256",
        "7e59ac4c97edabc3549303e6035f25a8c262f378e5440fa6ee8aeac5ef37c28a",
    )
    object.__setattr__(
        token,
        "semantic_input_sha256",
        "36d4e2d541d4613cc8cb0efd339ce41ff9ff88d00567ed889ba7d91935a695f2",
    )
    object.__setattr__(
        token,
        "scientific_plan_sha256",
        "5b06cb033cd0aec74934512ad254614781899a0d3336f41c7a527471677d5b58",
    )
    object.__setattr__(
        token,
        "bound_null_owner_sha256",
        "6483201ebfd69c11409c5dc81ee1286cea636274d2ffb0f3a4b27bc9acdf5c52",
    )
    object.__setattr__(token, "is_identity", False)
    object.__setattr__(token, "state", state)

    replicate = object.__new__(ReplicateOutcome)
    object.__setattr__(replicate, "replicate_id", 0)
    object.__setattr__(
        replicate,
        "seed_digest_sha256",
        "d2cc9b99619f121b33bfe737a7b49fb6810bda7b95f32f7baaebc3b05316e2b1",
    )
    object.__setattr__(replicate, "status", ReplicateStatus.COMPLETE)
    object.__setattr__(replicate, "failure_stage", None)
    object.__setattr__(replicate, "transform_token", token)
    object.__setattr__(replicate, "statistic_results", replicate_results)
    object.__setattr__(replicate, "selection", replicate_selection)
    object.__setattr__(replicate, "diagnostics", ())

    result = object.__new__(CalibrationResult)
    object.__setattr__(result, "status", RunStatus.COMPLETE)
    object.__setattr__(result, "failure_stage", None)
    object.__setattr__(
        result,
        "semantic_input_sha256",
        "36d4e2d541d4613cc8cb0efd339ce41ff9ff88d00567ed889ba7d91935a695f2",
    )
    object.__setattr__(
        result,
        "scientific_plan_sha256",
        "5b06cb033cd0aec74934512ad254614781899a0d3336f41c7a527471677d5b58",
    )
    object.__setattr__(result, "planned_replicates", 1)
    object.__setattr__(result, "alpha", 0.75)
    object.__setattr__(result, "observed_results", observed_results)
    object.__setattr__(result, "observed_selection", observed_selection)
    object.__setattr__(result, "replicates", (replicate,))
    object.__setattr__(result, "exceedance_count", 0)
    object.__setattr__(result, "failure_count", 0)
    object.__setattr__(result, "p_value", 0.5)
    object.__setattr__(result, "exceedance_bound_low", None)
    object.__setattr__(result, "exceedance_bound_high", None)
    object.__setattr__(result, "reject_null", True)
    object.__setattr__(result, "diagnostics", ())

    assert (
        token.schema,
        token.null_name,
        token.null_parameter_sha256,
        token.semantic_input_sha256,
        token.scientific_plan_sha256,
        token.bound_null_owner_sha256,
        token.is_identity,
    ) == (
        "selcal.null-transform-token.v2",
        "block_shuffle_v2",
        "7e59ac4c97edabc3549303e6035f25a8c262f378e5440fa6ee8aeac5ef37c28a",
        "36d4e2d541d4613cc8cb0efd339ce41ff9ff88d00567ed889ba7d91935a695f2",
        "5b06cb033cd0aec74934512ad254614781899a0d3336f41c7a527471677d5b58",
        "6483201ebfd69c11409c5dc81ee1286cea636274d2ffb0f3a4b27bc9acdf5c52",
        False,
    )
    assert token.state is state
    assert (state.schema, state.block_order) == (
        "selcal.block-shuffle-state.v2",
        (1, 2, 0),
    )
    assert replicate.seed_digest_sha256 == (
        "d2cc9b99619f121b33bfe737a7b49fb6810bda7b95f32f7baaebc3b05316e2b1"
    )
    assert result.exceedance_count == 0
    assert result.failure_count == 0
    assert result.p_value == 0.5 == (1 + 0) / (1 + 1)
    assert result.reject_null is True

    contracts_v2.verify_calibration_result(result, resolution)


_MEMORY_SEMANTIC_INPUT_SHA256 = (
    "8a9e9223f06d57b234b268d045bf1e0fd40d4e4e22f2b7a1bd21cd947f95463f"
)
_MEMORY_CIRCULAR_PARAMETER_SHA256 = (
    "a2e6b608e05886b0c06388750e201c922c21d08e4ca6faf7668742798895dda3"
)
_MEMORY_SUPPORT_N = 2048
_MEMORY_ALPHA = 0.0001
_MEMORY_COMPLETE_P_VALUES = {
    32: 0.030303030303030304,
    64: 0.015384615384615385,
    1000: 0.000999000999000999,
    2000: 0.0004997501249375312,
    4000: 0.00024993751562109475,
}
_MEMORY_ISOLATED_SAMPLES_PER_CASE = 5
_MEMORY_B_AXIS_CANDIDATES = 32
_MEMORY_B_AXIS_SMALL_REPLICATES = 32
_MEMORY_B_AXIS_LARGE_REPLICATES = 1000
_MEMORY_B_AXIS_SATURATION_REPLICATES = 2000
_MEMORY_B_AXIS_FAR_REPLICATES = 4000
_MEMORY_B_AXIS_ABSOLUTE_PEAK_LIMIT_BYTES = 262_144
_MEMORY_B_AXIS_DELTA_LIMIT_BYTES = 163_840
_MEMORY_B_AXIS_SATURATION_DELTA_LIMIT_BYTES = 8192
_MEMORY_C_AXIS_REPLICATES = 64
_MEMORY_C_AXIS_SMALL_CANDIDATES = 8
_MEMORY_C_AXIS_LARGE_CANDIDATES = 128
_MEMORY_C_AXIS_ABSOLUTE_PEAK_LIMIT_BYTES = 36_864
_MEMORY_C_AXIS_DELTA_LIMIT_BYTES = 12_288
_ISOLATED_MEMORY_WORKER_CODE = (
    "import runpy,sys;"
    "namespace=runpy.run_path(sys.argv[1],run_name='_selcal_memory_worker');"
    "namespace['_emit_isolated_verifier_memory_sample']("
    "int(sys.argv[2]),int(sys.argv[3]),sys.argv[4])"
)


def _memory_literal_seed_digest(
    scientific_plan_sha256: str,
    replicate_id: int,
) -> str:
    stream_domain = b"null_transform_v1"
    preimage = b"".join(
        (
            b"SELCAL-SEED\x00\x01",
            bytes.fromhex(scientific_plan_sha256),
            replicate_id.to_bytes(8, "big", signed=False),
            len(stream_domain).to_bytes(2, "big", signed=False),
            stream_domain,
        )
    )
    return hashlib.sha256(preimage).hexdigest()


def _memory_literal_owner_digest(
    scientific_plan_sha256: str,
    observed_length: int,
) -> str:
    canonical = json.dumps(
        {
            "schema": "selcal.bound-null-owner.v2",
            "semantic_input_sha256": _MEMORY_SEMANTIC_INPUT_SHA256,
            "scientific_plan_sha256": scientific_plan_sha256,
            "null_parameter_sha256": _MEMORY_CIRCULAR_PARAMETER_SHA256,
            "observed_length": observed_length,
            "transformed_role": "source",
        },
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _memory_literal_statistic(
    *,
    candidate_id: int,
    score: float,
) -> StatisticResult:
    result = object.__new__(StatisticResult)
    object.__setattr__(result, "candidate_id", candidate_id)
    object.__setattr__(result, "estimate", score)
    object.__setattr__(result, "selection_score", score)
    object.__setattr__(result, "support_n", _MEMORY_SUPPORT_N)
    object.__setattr__(result, "validity", Validity.VALID)
    object.__setattr__(result, "diagnostics", ())
    object.__setattr__(result, "backend_identity", _BACKEND_IDENTITY)
    object.__setattr__(result, "preprocessing_identity", _PREPROCESSING_IDENTITY)
    return result


def _memory_literal_selection(
    *,
    decision_statistic: float,
) -> SelectionResult:
    selection = object.__new__(SelectionResult)
    object.__setattr__(selection, "selected_candidate", 1)
    object.__setattr__(selection, "selected_index", 0)
    object.__setattr__(selection, "decision_statistic", decision_statistic)
    object.__setattr__(selection, "tied_candidates", (1,))
    return selection


def _memory_literal_token(
    *,
    scientific_plan_sha256: str,
    observed_length: int,
    replicate_id: int,
) -> NullTransformToken:
    state = object.__new__(CircularShiftStateV2)
    object.__setattr__(state, "schema", "selcal.circular-shift-state.v2")
    object.__setattr__(state, "shift", replicate_id % (observed_length - 1) + 1)

    token = object.__new__(NullTransformToken)
    object.__setattr__(token, "schema", "selcal.null-transform-token.v2")
    object.__setattr__(token, "null_name", "circular_shift_v2")
    object.__setattr__(
        token,
        "null_parameter_sha256",
        _MEMORY_CIRCULAR_PARAMETER_SHA256,
    )
    object.__setattr__(
        token,
        "semantic_input_sha256",
        _MEMORY_SEMANTIC_INPUT_SHA256,
    )
    object.__setattr__(
        token,
        "scientific_plan_sha256",
        scientific_plan_sha256,
    )
    object.__setattr__(
        token,
        "bound_null_owner_sha256",
        _memory_literal_owner_digest(scientific_plan_sha256, observed_length),
    )
    object.__setattr__(token, "is_identity", False)
    object.__setattr__(token, "state", state)
    return token


def _memory_literal_complete_graph(
    *,
    replicate_count: int,
    candidate_count: int,
) -> tuple[CalibrationResult, PlanResolutionV2]:
    candidates = tuple(range(1, candidate_count + 1))
    resolution = resolve_plan_v2(
        PlanRequestV2(
            candidates=candidates,
            statistic_name="lagged_pearson_v1",
            statistic_params={},
            selection_rule="max_upper",
            null_name="circular_shift_v2",
            null_params={"min_shift": 1},
            replicates=replicate_count,
            alpha=_MEMORY_ALPHA,
            tie_tolerance=1e-12,
            root_seed=20260901,
        )
    )
    verifier_context = inspect.getclosurevars(
        contracts_v2.verify_calibration_result
    ).nonlocals["resolution_context"]
    assert type(verifier_context) is FunctionType
    context = verifier_context(resolution)
    assert type(context) is tuple
    assert len(context) == 11
    scientific_plan_sha256 = context[1]
    assert type(scientific_plan_sha256) is str
    observed_length = _MEMORY_SUPPORT_N + candidate_count

    observed_results = tuple(
        _memory_literal_statistic(
            candidate_id=candidate_id,
            score=1.0 if candidate_id == 1 else 0.0,
        )
        for candidate_id in candidates
    )
    observed_selection = _memory_literal_selection(decision_statistic=1.0)

    replicates: list[ReplicateOutcome] = []
    for replicate_id in range(replicate_count):
        statistic_results = tuple(
            _memory_literal_statistic(
                candidate_id=candidate_id,
                score=0.5 if candidate_id == 1 else 0.0,
            )
            for candidate_id in candidates
        )
        selection = _memory_literal_selection(decision_statistic=0.5)
        outcome = object.__new__(ReplicateOutcome)
        object.__setattr__(outcome, "replicate_id", replicate_id)
        object.__setattr__(
            outcome,
            "seed_digest_sha256",
            _memory_literal_seed_digest(scientific_plan_sha256, replicate_id),
        )
        object.__setattr__(outcome, "status", ReplicateStatus.COMPLETE)
        object.__setattr__(outcome, "failure_stage", None)
        object.__setattr__(
            outcome,
            "transform_token",
            _memory_literal_token(
                scientific_plan_sha256=scientific_plan_sha256,
                observed_length=observed_length,
                replicate_id=replicate_id,
            ),
        )
        object.__setattr__(outcome, "statistic_results", statistic_results)
        object.__setattr__(outcome, "selection", selection)
        object.__setattr__(outcome, "diagnostics", ())
        replicates.append(outcome)

    result = object.__new__(CalibrationResult)
    object.__setattr__(result, "status", RunStatus.COMPLETE)
    object.__setattr__(result, "failure_stage", None)
    object.__setattr__(
        result,
        "semantic_input_sha256",
        _MEMORY_SEMANTIC_INPUT_SHA256,
    )
    object.__setattr__(
        result,
        "scientific_plan_sha256",
        scientific_plan_sha256,
    )
    object.__setattr__(result, "planned_replicates", replicate_count)
    object.__setattr__(result, "alpha", _MEMORY_ALPHA)
    object.__setattr__(result, "observed_results", observed_results)
    object.__setattr__(result, "observed_selection", observed_selection)
    object.__setattr__(result, "replicates", tuple(replicates))
    object.__setattr__(result, "exceedance_count", 0)
    object.__setattr__(result, "failure_count", 0)
    object.__setattr__(result, "p_value", _MEMORY_COMPLETE_P_VALUES[replicate_count])
    object.__setattr__(result, "exceedance_bound_low", None)
    object.__setattr__(result, "exceedance_bound_high", None)
    object.__setattr__(result, "reject_null", False)
    object.__setattr__(result, "diagnostics", ())
    return result, resolution


def _isolated_verifier_memory_sample(
    *,
    replicate_count: int,
    candidate_count: int,
) -> dict[str, object]:
    if sys.gettrace() is not None or sys.getprofile() is not None or tracemalloc.is_tracing():
        raise RuntimeError("measurement environment is dirty: tracing or profiling is active")
    tracing_before_construction = tracemalloc.is_tracing()
    result, resolution = _memory_literal_complete_graph(
        replicate_count=replicate_count,
        candidate_count=candidate_count,
    )
    tracing_after_construction = tracemalloc.is_tracing()
    assert tracing_after_construction is False
    gc.collect()
    tracing_after_gc = tracemalloc.is_tracing()
    assert tracing_after_gc is False

    tracemalloc.start()
    tracing_during_verifier = tracemalloc.is_tracing()
    assert tracing_during_verifier is True
    returned = contracts_v2.verify_calibration_result(result, resolution)
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.clear_traces()
    tracemalloc.stop()
    tracing_after_stop = tracemalloc.is_tracing()
    assert tracing_after_stop is False
    return {
        "replicate_count": replicate_count,
        "candidate_count": candidate_count,
        "peak_bytes": peak,
        "outcome_is_none": returned is None,
        "tracing_before_construction": tracing_before_construction,
        "tracing_after_construction": tracing_after_construction,
        "tracing_after_gc": tracing_after_gc,
        "tracing_during_verifier": tracing_during_verifier,
        "tracing_after_stop": tracing_after_stop,
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "process_id": os.getpid(),
    }


def _emit_isolated_verifier_memory_sample(
    replicate_count: int,
    candidate_count: int,
    launch_nonce: str = "",
) -> None:
    sample = _isolated_verifier_memory_sample(
        replicate_count=replicate_count,
        candidate_count=candidate_count,
    )
    print(json.dumps({**sample, "launch_nonce": launch_nonce}, sort_keys=True))


def _run_isolated_verifier_memory_sample(
    *,
    replicate_count: int,
    candidate_count: int,
) -> tuple[int, str]:
    # A fresh nonce per launch proves that each sample came from its own new process. Process
    # ids are not used for that: the operating system may reuse the id of a finished process
    # (observed on Windows, 2026-09-24).
    launch_nonce = uuid.uuid4().hex
    test_path = Path(__file__).resolve()
    completed = subprocess.run(
        (
            sys.executable,
            "-c",
            _ISOLATED_MEMORY_WORKER_CODE,
            str(test_path),
            str(replicate_count),
            str(candidate_count),
            launch_nonce,
        ),
        cwd=test_path.parent.parent,
        check=False,
        capture_output=True,
        text=True,
        errors="replace",
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert type(payload) is dict
    assert payload["replicate_count"] == replicate_count
    assert payload["candidate_count"] == candidate_count
    assert payload["outcome_is_none"] is True
    assert payload["tracing_before_construction"] is False
    assert payload["tracing_after_construction"] is False
    assert payload["tracing_after_gc"] is False
    assert payload["tracing_during_verifier"] is True
    assert payload["tracing_after_stop"] is False
    peak = payload["peak_bytes"]
    assert type(peak) is int
    assert peak > 0
    process_id = payload["process_id"]
    assert type(process_id) is int
    assert process_id > 0
    assert process_id != os.getpid()
    assert payload["launch_nonce"] == launch_nonce
    return peak, launch_nonce


def _five_isolated_verifier_memory_samples(
    *,
    replicate_count: int,
    candidate_count: int,
) -> tuple[tuple[int, str], ...]:
    return tuple(
        _run_isolated_verifier_memory_sample(
            replicate_count=replicate_count,
            candidate_count=candidate_count,
        )
        for _ in range(_MEMORY_ISOLATED_SAMPLES_PER_CASE)
    )


def _median_of_five(values: tuple[int, ...]) -> int:
    assert len(values) == _MEMORY_ISOLATED_SAMPLES_PER_CASE == 5
    return sorted(values)[2]


# PRE-PRODUCTION-REFACTOR memory rationale, frozen 2026-09-01 before the primitive
# module existed.  Exact environment: CPython 3.11.12, NumPy 2.4.6,
# macOS-26.6.2-arm64-arm-64bit.  Production SHA-256 was
# 9e4cb4a945602c3ca8e299aefa73b01e753d3b90bdaa465b849d81bcc2b80ef5;
# handoff test SHA-256 was
# 33e021f359ba80e5120904863cc05a0c62250ec5064e01a33ada958394667259;
# measurement-helper test SHA-256 before adding the limits was
# ad03c2488e496d14fad3bc3674944b36a2ac79325492137124e4c988c5359954.
# Each raw sample
# is one fresh, cold process; the measured call is the only code inside the
# tracemalloc window after construction and GC.  Baseline peaks (bytes): B axis
# C=32: B=32 [20149, 20149, 20149, 20149, 20149], max/median=20149/20149;
# B=1000 [174871, 174871, 174871, 174871, 174871], max/median=174871/174871;
# B=2000 [238487, 238487, 238487, 238487, 238487], max/median=238487/238487;
# B=4000 [238487, 238487, 238487, 238487, 238487], max/median=238487/238487.
# C axis B=64: C=8 [23008, 23008, 23008, 23008, 23008], max/median=23008/23008;
# C=128 [33804, 33804, 33804, 33804, 33804], max/median=33804/33804.
# B=32/1000/2000/4000 has no B-sized semantic container.  The only linear-looking
# live trace before saturation was fixed-cap CPython freelist retention from the
# real sealed token slot reader.  The B=4000
# far probe and its 8192 saturation delta prevent accepting continued O(B) growth
# hidden under the B=1000 delta.  The literal B limits (absolute 262144;
# required-pair max-to-min and median delta 163840) are independent of the already-
# stored B graph.  The literal C limits (absolute 36864;
# max-to-min and median delta 12288) deliberately permit bounded O(C) scratch.
# Exact worker command used for each listed literal B/C case:
# python
# -c "import runpy,sys;namespace=runpy.run_path(sys.argv[1],
# run_name='_selcal_memory_worker');namespace[
# '_emit_isolated_verifier_memory_sample'](int(sys.argv[2]),int(sys.argv[3]))"
# tests/test_verifier_integration_v2.py <literal-B> <literal-C>.
# tracemalloc measures Python-tracked allocations only,
# including CPython freelists, not native allocator/RSS/NumPy buffers.  This is a
# finite same-runtime peak characterization, not semantic live-set proof, a cross-
# platform memory guarantee, or an asymptotic theorem.  Task 8 may rerun but must
# not increase these thresholds.


def test_public_verifier_scratch_peak_is_b_bounded_with_fixed_c() -> None:
    small_samples = _five_isolated_verifier_memory_samples(
        replicate_count=_MEMORY_B_AXIS_SMALL_REPLICATES,
        candidate_count=_MEMORY_B_AXIS_CANDIDATES,
    )
    large_samples = _five_isolated_verifier_memory_samples(
        replicate_count=_MEMORY_B_AXIS_LARGE_REPLICATES,
        candidate_count=_MEMORY_B_AXIS_CANDIDATES,
    )
    saturation_samples = _five_isolated_verifier_memory_samples(
        replicate_count=_MEMORY_B_AXIS_SATURATION_REPLICATES,
        candidate_count=_MEMORY_B_AXIS_CANDIDATES,
    )
    far_samples = _five_isolated_verifier_memory_samples(
        replicate_count=_MEMORY_B_AXIS_FAR_REPLICATES,
        candidate_count=_MEMORY_B_AXIS_CANDIDATES,
    )
    all_launches = tuple(
        launch_nonce
        for _peak, launch_nonce in (
            small_samples + large_samples + saturation_samples + far_samples
        )
    )
    assert len(set(all_launches)) == (4 * _MEMORY_ISOLATED_SAMPLES_PER_CASE)

    small_peaks = tuple(peak for peak, _process_id in small_samples)
    large_peaks = tuple(peak for peak, _process_id in large_samples)
    saturation_peaks = tuple(peak for peak, _process_id in saturation_samples)
    far_peaks = tuple(peak for peak, _process_id in far_samples)
    assert max(small_peaks) <= _MEMORY_B_AXIS_ABSOLUTE_PEAK_LIMIT_BYTES
    assert max(large_peaks) <= _MEMORY_B_AXIS_ABSOLUTE_PEAK_LIMIT_BYTES
    assert max(saturation_peaks) <= _MEMORY_B_AXIS_ABSOLUTE_PEAK_LIMIT_BYTES
    assert max(far_peaks) <= _MEMORY_B_AXIS_ABSOLUTE_PEAK_LIMIT_BYTES
    assert (
        max(large_peaks) - min(small_peaks)
        <= _MEMORY_B_AXIS_DELTA_LIMIT_BYTES
    )
    assert (
        _median_of_five(large_peaks) - _median_of_five(small_peaks)
        <= _MEMORY_B_AXIS_DELTA_LIMIT_BYTES
    )
    assert (
        max(far_peaks) - min(saturation_peaks)
        <= _MEMORY_B_AXIS_SATURATION_DELTA_LIMIT_BYTES
    )
    assert (
        abs(_median_of_five(far_peaks) - _median_of_five(saturation_peaks))
        <= _MEMORY_B_AXIS_SATURATION_DELTA_LIMIT_BYTES
    )


def test_public_verifier_scratch_peak_has_bounded_c_scaling_with_fixed_b() -> None:
    small_samples = _five_isolated_verifier_memory_samples(
        replicate_count=_MEMORY_C_AXIS_REPLICATES,
        candidate_count=_MEMORY_C_AXIS_SMALL_CANDIDATES,
    )
    large_samples = _five_isolated_verifier_memory_samples(
        replicate_count=_MEMORY_C_AXIS_REPLICATES,
        candidate_count=_MEMORY_C_AXIS_LARGE_CANDIDATES,
    )
    all_process_ids = tuple(
        process_id for _peak, process_id in small_samples + large_samples
    )
    assert len(set(all_process_ids)) == 2 * _MEMORY_ISOLATED_SAMPLES_PER_CASE

    small_peaks = tuple(peak for peak, _process_id in small_samples)
    large_peaks = tuple(peak for peak, _process_id in large_samples)
    assert max(small_peaks) <= _MEMORY_C_AXIS_ABSOLUTE_PEAK_LIMIT_BYTES
    assert max(large_peaks) <= _MEMORY_C_AXIS_ABSOLUTE_PEAK_LIMIT_BYTES
    assert (
        max(large_peaks) - min(small_peaks)
        <= _MEMORY_C_AXIS_DELTA_LIMIT_BYTES
    )
    assert (
        _median_of_five(large_peaks) - _median_of_five(small_peaks)
        <= _MEMORY_C_AXIS_DELTA_LIMIT_BYTES
    )
