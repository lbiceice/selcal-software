from __future__ import annotations

import inspect
import os
import subprocess
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import FunctionType
from typing import cast

import numpy as np
import pytest

import selcal.calibration_v2 as calibration_v2
import selcal.nulls.block_shuffle_v2 as block_shuffle_v2
from selcal.calibration_v2 import _prepare_calibration, _PreparedCalibration
from selcal.contracts import RunStatus, SeriesPair
from selcal.contracts_v2 import (
    PlanRequestV2,
    ResourceLimitError,
    RunFailureStage,
    V2IntegrityError,
)
from selcal.resolution_v2 import PlanResolutionV2, resolve_plan_v2

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CAPS = {
    "_MAX_IN_MEMORY_REPLICATES_V2": 1_000,
    "_MAX_IN_MEMORY_CANDIDATE_EVALUATIONS_V2": 5_000,
    "_MAX_IN_MEMORY_TOKEN_STATE_UNITS_V2": 25_000,
    "_MAX_IN_MEMORY_WORK_UNITS_V2": 2_500_000,
}
EXPECTED_OVERSIZED_INTEGER_MESSAGE = (
    "IN_MEMORY_EXECUTION_BUDGET_EXCEEDED_V2: "
    "integer magnitude exceeds safe diagnostic envelope; "
    "caps: B<=1000, BC<=5000, BS<=25000, work<=2500000"
)


def _pair(length: int = 26, *, analytical_failure: bool = False) -> SeriesPair:
    if analytical_failure:
        source = np.ones(length, dtype=np.float64)
        target = np.ones(length, dtype=np.float64)
    else:
        source = np.arange(length, dtype=np.float64)
        target = np.asarray((*range(1, length), 0), dtype=np.float64)
    return SeriesPair(source=source, target=target)


def _resolution(
    *,
    replicates: int,
    candidates: tuple[int, ...] = (1,),
    null_name: str = "circular_shift_v2",
    null_params: dict[str, int] | None = None,
) -> PlanResolutionV2:
    if null_params is None:
        null_params = {"min_shift": 1}
    return resolve_plan_v2(
        PlanRequestV2(
            candidates=candidates,
            statistic_name="lagged_pearson_v1",
            statistic_params={},
            selection_rule="max_upper",
            null_name=null_name,
            null_params=null_params,
            replicates=replicates,
            alpha=0.05,
            tie_tolerance=1e-12,
            root_seed=17,
        )
    )


def _budget_guard() -> Callable[..., None]:
    candidate = getattr(calibration_v2, "_require_in_memory_execution_budget_values", None)
    assert callable(candidate), "Task10 quality fix must define the pure integer budget gate"
    return cast(Callable[..., None], candidate)


@contextmanager
def _trace_capsule_stream_creation(
    executor: Callable[..., object],
) -> Iterator[list[int]]:
    """Trace the actual frozen capsule constructor captured by an executor."""

    closure = inspect.getclosurevars(executor).nonlocals
    create_stream = closure.get("create_random_stream")
    assert type(create_stream) is FunctionType
    calls: list[int] = []

    def profile(frame: object, event: str, arg: object) -> None:
        del arg
        if event == "call" and frame.f_code is create_stream.__code__:  # type: ignore[attr-defined]
            calls.append(1)

    previous_profile = sys.getprofile()
    sys.setprofile(profile)
    try:
        yield calls
    finally:
        sys.setprofile(previous_profile)


def _expected_message(*, B: int, C: int, N: int, S: int) -> str:
    return (
        "IN_MEMORY_EXECUTION_BUDGET_EXCEEDED_V2: "
        f"B={B}, C={C}, N={N}, S={S}, BC={B * C}, BS={B * S}, "
        f"work={B * (C * N + S + N)}; "
        "caps: B<=1000, BC<=5000, BS<=25000, work<=2500000"
    )


def _replace_budget_module_globals_with_bypasses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def prepared_noop(value: object, /) -> object:
        return value

    def values_noop(*, B: int, C: int, N: int, S: int) -> None:
        del B, C, N, S

    monkeypatch.setattr(
        calibration_v2,
        "_require_prepared_in_memory_execution_budget",
        prepared_noop,
    )
    monkeypatch.setattr(
        calibration_v2,
        "_require_in_memory_execution_budget_values",
        values_noop,
    )
    for name in EXPECTED_CAPS:
        monkeypatch.setattr(calibration_v2, name, 10**30)


def _closure_reachable_functions(root: object) -> tuple[FunctionType, ...]:
    pending = [root]
    reachable: list[FunctionType] = []
    seen: set[int] = set()
    while pending:
        candidate = pending.pop()
        if not inspect.isfunction(candidate) or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        reachable.append(candidate)
        pending.extend(inspect.getclosurevars(candidate).nonlocals.values())
    return tuple(reachable)


def test_versioned_in_memory_budget_constants_are_exact() -> None:
    assert {
        name: getattr(calibration_v2, name, None) for name in EXPECTED_CAPS
    } == EXPECTED_CAPS


@pytest.mark.parametrize(
    ("B", "C", "N", "S"),
    (
        (1_000, 1, 1, 1),
        (1_000, 5, 1, 1),
        (1_000, 1, 1, 25),
        (1, 4, 499_999, 5),
    ),
    ids=("B_cap", "BC_cap", "BS_cap", "work_cap"),
)
def test_each_in_memory_budget_cap_itself_passes(B: int, C: int, N: int, S: int) -> None:
    _budget_guard()(B=B, C=C, N=N, S=S)


@pytest.mark.parametrize(
    ("B", "C", "N", "S"),
    (
        (1_001, 1, 1, 1),
        (1, 5_001, 1, 1),
        (1, 1, 1, 25_001),
        (1, 4, 500_000, 1),
    ),
    ids=("B_plus_one", "BC_plus_one", "BS_plus_one", "work_plus_one"),
)
def test_each_in_memory_budget_cap_plus_one_has_stable_typed_failure(
    B: int,
    C: int,
    N: int,
    S: int,
) -> None:
    with pytest.raises(ResourceLimitError) as captured:
        _budget_guard()(B=B, C=C, N=N, S=S)

    assert str(captured.value) == _expected_message(B=B, C=C, N=N, S=S)


def test_huge_builtin_integer_fails_before_product_or_decimal_materialization() -> None:
    with pytest.raises(ResourceLimitError) as captured:
        _budget_guard()(B=10**5000, C=1, N=1, S=1)

    assert str(captured.value) == EXPECTED_OVERSIZED_INTEGER_MESSAGE


@pytest.mark.parametrize("field_name", ("B", "C", "N", "S"))
@pytest.mark.parametrize(
    ("invalid_value", "case_name"),
    (
        (0, "zero"),
        (-1, "negative"),
        (True, "bool"),
        (1.0, "non_int"),
    ),
    ids=lambda value: str(value),
)
def test_each_budget_operand_rejects_every_invalid_integer_shape(
    field_name: str,
    invalid_value: object,
    case_name: str,
) -> None:
    del case_name
    values: dict[str, object] = {"B": 1, "C": 1, "N": 1, "S": 1}
    values[field_name] = invalid_value

    with pytest.raises(V2IntegrityError) as captured:
        _budget_guard()(**values)

    assert str(captured.value) == f"{field_name} must be a positive built-in integer"


def test_bound_null_snapshot_exposes_generic_retained_token_state_units() -> None:
    circular = _prepare_calibration(_pair(), _resolution(replicates=1))
    block = _prepare_calibration(
        _pair(),
        _resolution(
            replicates=1,
            null_name="block_shuffle_v2",
            null_params={"block_length": 1},
        ),
    )
    assert type(circular) is _PreparedCalibration
    assert type(block) is _PreparedCalibration

    assert circular.null_snapshot.execution.retained_token_state_units == 1
    assert block.null_snapshot.execution.retained_token_state_units == 26


def test_same_b_c_n_can_pass_or_fail_only_from_generic_s() -> None:
    circular = _prepare_calibration(_pair(), _resolution(replicates=1_000))
    block = _prepare_calibration(
        _pair(),
        _resolution(
            replicates=1_000,
            null_name="block_shuffle_v2",
            null_params={"block_length": 1},
        ),
    )
    assert type(circular) is _PreparedCalibration
    assert type(block) is _PreparedCalibration

    guard = getattr(calibration_v2, "_require_prepared_in_memory_execution_budget", None)
    assert callable(guard)
    guard(circular)
    with pytest.raises(ResourceLimitError) as captured:
        guard(block)
    assert str(captured.value) == _expected_message(B=1_000, C=1, N=26, S=26)


def test_million_replicate_plan_resolves_but_public_executor_fails_before_execute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolution = _resolution(replicates=1_000_000)
    assert resolution.plan.replicates == 1_000_000
    calls = 0

    def execution_bomb(prepared: object, /) -> object:
        nonlocal calls
        del prepared
        calls += 1
        raise AssertionError("_execute_replicates ran before the public budget gate")

    monkeypatch.setattr(calibration_v2, "_execute_replicates", execution_bomb)
    with pytest.raises(ResourceLimitError) as captured:
        calibration_v2.calibrate_selected_family(_pair(6), resolution)

    assert str(captured.value) == _expected_message(B=1_000_000, C=1, N=6, S=1)
    assert calls == 0


def test_internal_execute_cannot_bypass_budget_or_allocate_replicate_state(
) -> None:
    prepared = _prepare_calibration(_pair(6), _resolution(replicates=1_000_000))
    assert type(prepared) is _PreparedCalibration
    with _trace_capsule_stream_creation(calibration_v2._execute_replicates) as calls:
        with pytest.raises(ResourceLimitError) as captured:
            calibration_v2._execute_replicates(prepared, observer=None)

    assert str(captured.value) == _expected_message(B=1_000_000, C=1, N=6, S=1)
    assert calls == []


def test_public_budget_cannot_be_bypassed_by_ordinary_module_rebinding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolution = _resolution(replicates=1_000_000)
    calls = 0

    def execution_bomb(prepared: object, /) -> object:
        nonlocal calls
        del prepared
        calls += 1
        raise AssertionError("replaced public executor ran after budget guard rebinding")

    _replace_budget_module_globals_with_bypasses(monkeypatch)
    monkeypatch.setattr(calibration_v2, "_execute_replicates", execution_bomb)

    signature = inspect.signature(calibration_v2.calibrate_selected_family)
    assert tuple(signature.parameters) == ("pair", "resolution")
    assert all(
        parameter.kind is inspect.Parameter.POSITIONAL_ONLY
        for parameter in signature.parameters.values()
    )
    with pytest.raises(ResourceLimitError) as captured:
        calibration_v2.calibrate_selected_family(_pair(6), resolution)

    assert str(captured.value) == _expected_message(B=1_000_000, C=1, N=6, S=1)
    assert calls == 0


def test_private_budget_cannot_be_bypassed_by_ordinary_module_rebinding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepare_calibration(_pair(6), _resolution(replicates=1_000_000))
    assert type(prepared) is _PreparedCalibration

    _replace_budget_module_globals_with_bypasses(monkeypatch)

    signature = inspect.signature(calibration_v2._execute_replicates)
    assert tuple(signature.parameters) == ("prepared", "observer")
    assert next(iter(signature.parameters.values())).kind is inspect.Parameter.POSITIONAL_ONLY
    assert signature.parameters["observer"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["observer"].default is inspect.Parameter.empty
    with _trace_capsule_stream_creation(calibration_v2._execute_replicates) as calls:
        with pytest.raises(ResourceLimitError) as captured:
            calibration_v2._execute_replicates(prepared, observer=None)

    assert str(captured.value) == _expected_message(B=1_000_000, C=1, N=6, S=1)
    assert calls == []


def test_no_module_or_public_closure_exposes_an_injectable_calibration_core() -> None:
    candidates = {
        candidate
        for candidate in vars(calibration_v2).values()
        if inspect.isfunction(candidate)
    }
    candidates.update(
        _closure_reachable_functions(calibration_v2.calibrate_selected_family)
    )
    injectable = {
        candidate.__qualname__
        for candidate in candidates
        if {"require_budget", "execute_replicates"}
        <= set(inspect.signature(candidate).parameters)
    }

    assert not injectable


def test_every_closure_reachable_executor_rejects_before_rng_allocation(
) -> None:
    prepared = _prepare_calibration(_pair(6), _resolution(replicates=1_000_000))
    assert type(prepared) is _PreparedCalibration
    executors = {
        candidate
        for root in (
            calibration_v2._execute_replicates,
            calibration_v2.calibrate_selected_family,
        )
        for candidate in _closure_reachable_functions(root)
        if candidate.__name__ == "_execute_replicates"
    }
    assert executors
    for executor in executors:
        with _trace_capsule_stream_creation(executor) as allocation_calls:
            with pytest.raises(ResourceLimitError):
                executor(prepared, observer=None)
        assert allocation_calls == []


def test_maximum_block_count_bs_overflow_precedes_every_replicate_allocation_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepare_calibration(
        _pair(4_096),
        _resolution(
            replicates=7,
            null_name="block_shuffle_v2",
            null_params={"block_length": 1},
        ),
    )
    assert type(prepared) is _PreparedCalibration
    calls: list[str] = []

    class AllocationBomb:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs
            calls.append("random_or_outcome")
            raise AssertionError("replicate object allocated before the B*S budget gate")

    def call_bomb(*args: object, **kwargs: object) -> object:
        del args, kwargs
        calls.append("token_apply_history_or_signature")
        raise AssertionError("B-scale replicate path ran before the B*S budget gate")

    monkeypatch.setattr(calibration_v2, "ReplicateOutcome", AllocationBomb)
    monkeypatch.setattr(calibration_v2, "_ReplicateHistorySnapshot", call_bomb)
    monkeypatch.setattr(block_shuffle_v2, "_ascending_labels", call_bomb)
    monkeypatch.setattr(block_shuffle_v2, "_apply_state", call_bomb)

    with _trace_capsule_stream_creation(calibration_v2._execute_replicates) as random_calls:
        with pytest.raises(ResourceLimitError) as captured:
            calibration_v2._execute_replicates(prepared, observer=None)

    assert str(captured.value) == _expected_message(B=7, C=1, N=4_096, S=4_096)
    assert random_calls == []
    assert calls == []


def test_budget_does_not_preempt_null_disabled_or_observed_failure_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def execution_bomb(prepared: object, /) -> object:
        del prepared
        raise AssertionError("terminal paths must not enter replicate execution")

    _replace_budget_module_globals_with_bypasses(monkeypatch)
    monkeypatch.setattr(calibration_v2, "_execute_replicates", execution_bomb)
    disabled = calibration_v2.calibrate_selected_family(
        _pair(6),
        _resolution(
            replicates=1_000_000,
            null_params={"min_shift": 4},
        ),
    )
    observed_failure = calibration_v2.calibrate_selected_family(
        _pair(6, analytical_failure=True),
        _resolution(replicates=1_000_000),
    )

    assert disabled.status is RunStatus.NOT_EVALUABLE
    assert disabled.failure_stage is RunFailureStage.NULL_BIND
    assert observed_failure.status is RunStatus.NOT_EVALUABLE
    assert observed_failure.failure_stage is RunFailureStage.OBSERVED_STATISTIC_SCAN


def test_budget_error_is_cross_process_stable_with_explicit_source_and_hash_seed() -> None:
    script = """
import selcal.calibration_v2 as module
try:
    module._require_in_memory_execution_budget_values(B=1001, C=1, N=1, S=1)
except Exception as error:
    print(type(error).__name__ + ':' + str(error))
else:
    raise AssertionError('budget unexpectedly passed')
"""
    outputs: list[str] = []
    for hash_seed in ("1", "987654"):
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(REPOSITORY_ROOT / "src")
        environment["PYTHONHASHSEED"] = hash_seed
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=REPOSITORY_ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=10,
        )
        outputs.append(completed.stdout.strip())

    expected = "ResourceLimitError:" + _expected_message(B=1_001, C=1, N=1, S=1)
    assert outputs == [expected, expected]
