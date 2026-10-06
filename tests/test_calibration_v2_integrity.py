from __future__ import annotations

import dis
import gc
import importlib
import inspect
import math
import os
import subprocess
import sys
import tracemalloc
import weakref
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from types import CodeType, FrameType, FunctionType, MappingProxyType, SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from _random_behavior_graph_v2 import behavior_capture_graph

import selcal.calibration_v2 as calibration_v2
import selcal.contracts_v2 as contracts_v2
import selcal.nulls.executable_base as executable_base
import selcal.randomness as randomness
import selcal.resolution_v2 as resolution_v2
import selcal.result_wire as result_wire
from selcal.calibration_v2 import (
    _execute_replicates,
    _prepare_calibration,
    _PreparedCalibration,
)
from selcal.canonical import canonical_json_bytes
from selcal.canonical_v2 import (
    _RESULT_VERIFIER_PLAN_CANONICAL_BYTES_V2,
    scientific_plan_v2_payload,
    scientific_plan_v2_sha256,
)
from selcal.contracts import (
    RunStatus,
    SelectionRule,
    SeriesPair,
    StatisticResult,
    Validity,
)
from selcal.contracts_v2 import (
    _V2_RESOLUTION_SEAL,
    BlockShuffleStateV2,
    CalibrationResult,
    CircularShiftStateV2,
    NullTransformToken,
    PlanRequestV2,
    ReplicateOutcome,
    ResolvedScientificPlanV2,
    RunFailureStage,
    V2IntegrityError,
)
from selcal.nulls.block_shuffle_v2 import BlockShuffleNullV2
from selcal.nulls.circular_shift_v2 import CircularShiftNullV2, _BoundCircularShiftV2
from selcal.nulls.owned_transform_v2 import owner_digest
from selcal.randomness import ReplicateRandomSource as RealReplicateRandomSource
from selcal.randomness import _RandomCapsuleV2
from selcal.resolution_v2 import PlanResolutionV2, resolve_plan_v2
from selcal.statistics.binned_nette import BinnedNetTEAdapter
from selcal.statistics.lagged_pearson import LaggedPearsonAdapter, _BoundLaggedPearsonAdapter


def _pair() -> SeriesPair:
    return SeriesPair(
        source=np.asarray([0, 1, 4, 2, 5, 3], dtype=np.float64),
        target=np.asarray([4, 1, 3, 0, 5, 2], dtype=np.float64),
    )


def _resolution(
    *,
    candidates: tuple[int, ...] = (1, 2),
    replicates: int = 3,
    min_shift: int = 1,
) -> PlanResolutionV2:
    return resolve_plan_v2(
        PlanRequestV2(
            candidates=candidates,
            statistic_name="lagged_pearson_v1",
            statistic_params={},
            selection_rule="max_upper",
            null_name="circular_shift_v2",
            null_params={"min_shift": min_shift},
            replicates=replicates,
            alpha=0.05,
            tie_tolerance=1e-12,
            root_seed=17,
        )
    )


def _verifier() -> Callable[[CalibrationResult, PlanResolutionV2], None]:
    candidate = getattr(contracts_v2, "verify_calibration_result", None)
    assert callable(candidate), "Task 8C must add verify_calibration_result(result, resolution, /)"
    return cast(Callable[[CalibrationResult, PlanResolutionV2], None], candidate)


def _published_seed_digest_operation() -> Callable[[object, object, object], str]:
    candidate = inspect.getclosurevars(_verifier()).nonlocals["externals"].seed_digest
    assert inspect.isfunction(candidate)
    return cast(Callable[[object, object, object], str], candidate)


def _seed_operation_project_code_findings() -> tuple[str, ...]:
    """Audit only project code transitively retained by the published seed oracle."""

    pending: list[tuple[str, object]] = [
        ("seed_digest_operation", _published_seed_digest_operation())
    ]
    seen: set[int] = set()
    findings: list[str] = []
    while pending:
        path, candidate = pending.pop()
        if id(candidate) in seen:
            continue
        seen.add(id(candidate))
        assert len(seen) <= 64, "seed operation graph exceeded its exact bound"

        if isinstance(candidate, type):
            if not candidate.__module__.startswith("selcal."):
                continue
            constructor = inspect.getattr_static(candidate, "__init__", None)
            if type(constructor) is FunctionType:
                pending.append((f"{path}.__init__", constructor))
            continue
        if type(candidate) is not FunctionType or not candidate.__module__.startswith("selcal."):
            continue

        codes = [candidate.__code__]
        visited_codes: set[int] = set()
        while codes:
            code = codes.pop()
            if id(code) in visited_codes:
                continue
            visited_codes.add(id(code))
            for instruction in dis.get_instructions(code):
                if instruction.opname in {"LOAD_GLOBAL", "IMPORT_NAME", "IMPORT_FROM"}:
                    findings.append(f"{path}:{code.co_name}:{instruction.opname}:{instruction.argval}")
            codes.extend(
                constant for constant in code.co_consts if type(constant) is CodeType
            )

        if candidate.__closure__ is not None:
            pending.extend(
                (
                    f"{path}.closure[{name}]",
                    cell.cell_contents,
                )
                for name, cell in zip(
                    candidate.__code__.co_freevars,
                    candidate.__closure__,
                    strict=True,
                )
            )

    return tuple(sorted(findings))


def _calibrator() -> Callable[[SeriesPair, PlanResolutionV2], CalibrationResult]:
    candidate = getattr(calibration_v2, "calibrate_selected_family", None)
    assert callable(candidate), "Task 8C must add calibrate_selected_family(pair, resolution, /)"
    return cast(Callable[[SeriesPair, PlanResolutionV2], CalibrationResult], candidate)


def _ownership_records() -> dict[int, Any]:
    require = resolution_v2._require_resolver_owned_resolution_v2
    ownership_lookup = inspect.getclosurevars(require).nonlocals["ownership_lookup"]
    records_get = inspect.getclosurevars(ownership_lookup).nonlocals["records_get"]
    records = getattr(records_get, "__self__", None)
    assert type(records) is dict
    return records


def test_resolution_ownership_records_are_not_exposed_by_the_module() -> None:
    records = _ownership_records()
    exposed_names = tuple(
        sorted(name for name, value in vars(resolution_v2).items() if value is records)
    )

    assert (
        exposed_names,
        getattr(resolution_v2, "_RESOLVER_OWNED_BY_ID", None) is records,
    ) == ((), False)


def _complete_result() -> tuple[CalibrationResult, PlanResolutionV2]:
    """Build constructor-valid Task 8B evidence without presuming Task 8C exists."""

    exact = _resolution()
    prepared = _prepare_calibration(_pair(), exact)
    assert type(prepared) is _PreparedCalibration
    outcomes = _execute_replicates(prepared, observer=None)
    observed_a = prepared.observed_selection.decision_statistic
    exceedances = sum(
        outcome.selection is not None and outcome.selection.decision_statistic >= observed_a
        for outcome in outcomes
    )
    p_value = (1 + exceedances) / (exact.plan.replicates + 1)
    return (
        CalibrationResult(
            status=RunStatus.COMPLETE,
            failure_stage=None,
            semantic_input_sha256=prepared.semantic_input_sha256,
            scientific_plan_sha256=prepared.scientific_plan_sha256,
            planned_replicates=exact.plan.replicates,
            alpha=exact.plan.alpha,
            observed_results=prepared.observed_results,
            observed_selection=prepared.observed_selection,
            replicates=outcomes,
            exceedance_count=exceedances,
            failure_count=0,
            p_value=p_value,
            exceedance_bound_low=None,
            exceedance_bound_high=None,
            reject_null=p_value <= exact.plan.alpha,
            diagnostics=prepared.diagnostics,
        ),
        exact,
    )


_EXPECTED_PRODUCTION_RANDOM_VECTOR = (
    "456f8ccf9fe4666c96f58f903bc7d2750961b532d115d10983d34b062c61eb82",
    (
        (
            "e295b627109e737811afffa5a84a5bad928ad1de7dca3332e4a9340ffdee9d8c",
            2,
            (0.17541160386140586, -0.3971869795707654),
            1,
            0.17541160386140586,
        ),
        (
            "6e5dcdd9768a9e6546c233fbe94159459b9d148b9d304fccc8f8019fb53fbb1a",
            5,
            (0.9922778767136676, -0.6139406135149204),
            1,
            0.9922778767136676,
        ),
    ),
)
_EXPECTED_COMPATIBILITY_RANDOM_VECTOR = (
    "a8dd219a1e348a103ff10bdad3b9ae8d3cff53032be90a3603734530f51fb82a",
    (1874999576356411599, 4645183139923492514, 4210660991089764159),
    (1, 2, 6, 17099354796136167930),
)


def _compatibility_random_vector() -> (
    tuple[
        str,
        tuple[int, ...],
        tuple[int, ...],
    ]
):
    plan_digest = "dc7462598231451b36a94e2586037aa921a8159b2eabebb5fe5c81dcafa48931"
    seed_source = RealReplicateRandomSource(plan_digest, 0, 2)
    raw_source = RealReplicateRandomSource(plan_digest, 0, 2)
    bounded_source = RealReplicateRandomSource(plan_digest, 0, 2)
    return (
        seed_source.seed_digest_sha256,
        tuple(raw_source._next_raw64() for _ in range(3)),
        tuple(bounded_source.randbelow(bound) for bound in (2, 3, 17, 2**64 + 1)),
    )


def _production_random_vector(
    result: CalibrationResult,
) -> tuple[str, tuple[tuple[str, int, tuple[float, ...], int, float], ...]]:
    assert result.status is RunStatus.COMPLETE
    rows: list[tuple[str, int, tuple[float, ...], int, float]] = []
    for outcome in result.replicates:
        assert type(outcome.transform_token.state) is CircularShiftStateV2
        assert outcome.selection is not None
        scores = tuple(item.selection_score for item in outcome.statistic_results)
        assert all(type(score) is float for score in scores)
        rows.append(
            (
                outcome.seed_digest_sha256,
                outcome.transform_token.state.shift,
                cast(tuple[float, ...], scores),
                outcome.selection.selected_candidate,
                outcome.selection.decision_statistic,
            )
        )
    return result.scientific_plan_sha256, tuple(rows)


# The expected vector was recorded on macOS arm64. Seeds, shifts and selections must match
# exactly on every platform; statistic values may differ in the last bits (Linux: up to 1 ULP,
# docs/status/evidence/linux_20260923/summary.md), so they are compared within 4 ULP.
_RECORDED_STATISTIC_MAX_ULP = 4


def _within_ulp(actual: float, expected: float) -> bool:
    return abs(actual - expected) <= _RECORDED_STATISTIC_MAX_ULP * math.ulp(expected)


def _assert_matches_recorded_vector(
    actual: tuple[str, tuple[tuple[str, int, tuple[float, ...], int, float], ...]],
) -> None:
    plan_digest, rows = actual
    expected_digest, expected_rows = _EXPECTED_PRODUCTION_RANDOM_VECTOR
    assert plan_digest == expected_digest
    assert len(rows) == len(expected_rows)
    for row, expected in zip(rows, expected_rows, strict=True):
        seed, shift, scores, selected, statistic = row
        assert (seed, shift, selected, len(scores)) == (
            expected[0],
            expected[1],
            expected[3],
            len(expected[2]),
        )
        assert all(
            _within_ulp(value, target) for value, target in zip(scores, expected[2], strict=True)
        )
        assert _within_ulp(statistic, expected[4])


def _production_random_result() -> tuple[CalibrationResult, PlanResolutionV2]:
    exact = _resolution(replicates=2)
    result = _calibrator()(_pair(), exact)
    _assert_matches_recorded_vector(_production_random_vector(result))
    return result, exact


def _install_randomness_module_bomb(
    patch: pytest.MonkeyPatch,
    target: str,
    bomb: Callable[..., object],
) -> object:
    if target == "hashlib":
        original = randomness.hashlib
        patch.setattr(randomness, "hashlib", SimpleNamespace(sha256=bomb))
        return original
    if target == "np":
        original = randomness.np
        patch.setattr(
            randomness,
            "np",
            SimpleNamespace(random=SimpleNamespace(PCG64=bomb)),
        )
        return original
    original = getattr(randomness, target)
    patch.setattr(randomness, target, bomb)
    return original


@pytest.mark.parametrize(
    "target",
    (
        "_require_plan_digest",
        "_require_planned_replicates",
        "_require_replicate_id",
        "_require_raw64_word",
        "_uniform_randbelow_from_next",
        "hashlib",
        "np",
    ),
)
def test_post_seal_production_random_module_replacement_matrix_is_inert(
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    compatibility_baseline = _compatibility_random_vector()
    assert compatibility_baseline == _EXPECTED_COMPATIBILITY_RANDOM_VECTOR
    baseline, exact = _production_random_result()
    baseline_vector = _production_random_vector(baseline)
    replacement_calls = 0
    attack_error: AssertionError | None = None
    attacked: CalibrationResult | None = None

    def bomb(*_args: object, **_kwargs: object) -> object:
        nonlocal replacement_calls
        replacement_calls += 1
        raise AssertionError(f"replacement randomness module target executed: {target}")

    original: object
    with monkeypatch.context() as patch:
        original = _install_randomness_module_bomb(patch, target, bomb)
        try:
            attacked = _calibrator()(_pair(), exact)
        except AssertionError as error:
            attack_error = error

    assert getattr(randomness, target) is original
    assert _compatibility_random_vector() == compatibility_baseline
    restored, _ = _production_random_result()
    assert _production_random_vector(restored) == baseline_vector
    if attack_error is not None:
        raise attack_error
    assert replacement_calls == 0
    assert attacked == baseline
    assert _production_random_vector(cast(CalibrationResult, attacked)) == baseline_vector


@pytest.mark.parametrize(
    "descriptor_name",
    ("__init__", "randbelow", "_next_raw64", "seed_digest_sha256"),
)
def test_public_random_source_descriptor_replacement_is_inert_or_typed(
    monkeypatch: pytest.MonkeyPatch,
    descriptor_name: str,
) -> None:
    baseline, exact = _production_random_result()
    original_descriptor = inspect.getattr_static(
        RealReplicateRandomSource,
        descriptor_name,
    )
    replacement_calls = 0
    attack_error: AssertionError | None = None
    typed_error: V2IntegrityError | None = None
    attacked: CalibrationResult | None = None

    def bomb(*_args: object, **_kwargs: object) -> object:
        nonlocal replacement_calls
        replacement_calls += 1
        raise AssertionError(f"replacement random descriptor executed: {descriptor_name}")

    replacement: object = property(bomb) if descriptor_name == "seed_digest_sha256" else bomb
    with monkeypatch.context() as patch:
        patch.setattr(RealReplicateRandomSource, descriptor_name, replacement)
        assert inspect.getattr_static(RealReplicateRandomSource, descriptor_name) is replacement
        try:
            attacked = _calibrator()(_pair(), exact)
        except AssertionError as error:
            attack_error = error
        except V2IntegrityError as error:
            typed_error = error

    assert inspect.getattr_static(RealReplicateRandomSource, descriptor_name) is original_descriptor
    restored, _ = _production_random_result()
    assert restored == baseline
    if attack_error is not None:
        raise attack_error
    assert replacement_calls == 0
    assert typed_error is not None or attacked == baseline


@pytest.mark.parametrize(
    "descriptor_name",
    (
        "create_stream",
        "seed_digest",
        "randbelow",
        "seed_digest_for_replicate",
    ),
)
@pytest.mark.parametrize("replacement_kind", ("property", "function"))
def test_private_random_capsule_field_descriptors_are_not_runtime_execution_surfaces(
    monkeypatch: pytest.MonkeyPatch,
    descriptor_name: str,
    replacement_kind: str,
) -> None:
    baseline, exact = _production_random_result()
    original_descriptor = inspect.getattr_static(_RandomCapsuleV2, descriptor_name)
    replacement_calls = 0

    def bomb(*_args: object, **_kwargs: object) -> object:
        nonlocal replacement_calls
        replacement_calls += 1
        raise AssertionError(
            f"replacement random capsule descriptor executed: {descriptor_name}"
        )

    replacement: object = property(bomb) if replacement_kind == "property" else bomb
    with monkeypatch.context() as patch:
        patch.setattr(_RandomCapsuleV2, descriptor_name, replacement)
        assert inspect.getattr_static(_RandomCapsuleV2, descriptor_name) is replacement
        attacked = _calibrator()(_pair(), exact)
        _verifier()(attacked, exact)

    assert inspect.getattr_static(_RandomCapsuleV2, descriptor_name) is original_descriptor
    assert replacement_calls == 0
    assert attacked == baseline


def test_private_random_stream_is_an_exact_builtin_tuple_without_a_project_class() -> None:
    capsule = randomness._RANDOM_CAPSULE_V2
    create_stream = cast(
        Callable[[str, int, int], tuple[object, str]],
        tuple.__getitem__(capsule, 0),
    )
    stream = create_stream("0" * 64, 0, 1)

    assert not hasattr(randomness, "_RandomStreamV2")
    assert type(stream) is tuple
    assert inspect.getattr_static(type(stream), "__new__") is tuple.__new__
    assert inspect.getattr_static(type(stream), "__getitem__") is tuple.__getitem__


def test_random_descriptor_replace_use_restore_inside_real_null_callback_is_inert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline, exact = _production_random_result()
    original_sample = _BoundCircularShiftV2.sample_token
    original_randbelow = inspect.getattr_static(
        RealReplicateRandomSource,
        "randbelow",
    )
    callback_calls = 0
    active_windows = 0
    replacement_calls = 0
    attack_error: AssertionError | None = None
    typed_error: V2IntegrityError | None = None
    attacked: CalibrationResult | None = None

    def bomb(*_args: object, **_kwargs: object) -> object:
        nonlocal replacement_calls
        replacement_calls += 1
        raise AssertionError("temporary ReplicateRandomSource.randbelow executed")

    def replace_use_restore(
        self: _BoundCircularShiftV2,
        random: object,
        /,
    ) -> object:
        nonlocal active_windows, callback_calls
        callback_calls += 1
        with monkeypatch.context() as patch:
            patch.setattr(RealReplicateRandomSource, "randbelow", bomb)
            assert inspect.getattr_static(RealReplicateRandomSource, "randbelow") is bomb
            active_windows += 1
            token = original_sample(self, random)  # type: ignore[arg-type]
            assert inspect.getattr_static(RealReplicateRandomSource, "randbelow") is bomb
        assert inspect.getattr_static(RealReplicateRandomSource, "randbelow") is original_randbelow
        return token

    with monkeypatch.context() as patch:
        patch.setattr(_BoundCircularShiftV2, "sample_token", replace_use_restore)
        try:
            attacked = _calibrator()(_pair(), exact)
        except AssertionError as error:
            attack_error = error
        except V2IntegrityError as error:
            typed_error = error

    assert _BoundCircularShiftV2.sample_token is original_sample
    assert inspect.getattr_static(RealReplicateRandomSource, "randbelow") is original_randbelow
    restored, _ = _production_random_result()
    assert restored == baseline
    if attack_error is not None:
        raise attack_error
    assert callback_calls == exact.plan.replicates
    assert active_windows == exact.plan.replicates
    assert replacement_calls == 0
    assert typed_error is not None or attacked == baseline


def _callable_capture_graph(
    root: object,
    *,
    terminal_identities: tuple[object, ...] = (),
) -> tuple[tuple[str, object], ...]:
    return behavior_capture_graph(
        root,
        follow_function_globals=False,
        terminal_identities=terminal_identities,
        node_limit=2048,
        edge_limit=8192,
        depth_limit=24,
    ).occurrences


def _assert_only_canonical_random_dependencies(
    graph: tuple[tuple[str, object], ...],
) -> None:
    capsule = randomness._RANDOM_CAPSULE_V2
    operations = tuple(tuple.__getitem__(capsule, index) for index in range(4))
    operation_by_code = {operation.__code__: operation for operation in operations}
    for path, candidate in sorted(graph, key=lambda item: item[0]):
        if type(candidate) is _RandomCapsuleV2:
            assert candidate is capsule, f"alternate random capsule reached at {path}"
        if type(candidate) is FunctionType and candidate.__code__ in operation_by_code:
            assert candidate is operation_by_code[candidate.__code__], (
                f"alternate random operation reached at {path}"
            )


def test_callable_capture_graph_exposes_alternate_capsule_behind_builder_state() -> None:
    canonical_capsule = randomness._RANDOM_CAPSULE_V2
    canonical_randbelow = tuple.__getitem__(canonical_capsule, 2)
    alternate_randbelow = FunctionType(
        canonical_randbelow.__code__,
        canonical_randbelow.__globals__,
        name="alternate_randbelow",
        argdefs=canonical_randbelow.__defaults__,
        closure=canonical_randbelow.__closure__,
    )
    alternate_capsule = _RandomCapsuleV2(
        create_stream=tuple.__getitem__(canonical_capsule, 0),
        seed_digest=tuple.__getitem__(canonical_capsule, 1),
        randbelow=alternate_randbelow,
        seed_digest_for_replicate=tuple.__getitem__(canonical_capsule, 3),
        external_leaves=(),
    )

    class OpaqueBuilder:
        __slots__ = ("alternate_capsule",)

        def __init__(self, capsule: _RandomCapsuleV2) -> None:
            self.alternate_capsule = capsule

        def __call__(self, _operation: object, _stream: object, /) -> tuple[object, ...]:
            return (self.alternate_capsule,)

    builder = OpaqueBuilder(alternate_capsule)

    def counterfeit_executor() -> tuple[object, object]:
        return canonical_capsule, builder

    graph = _callable_capture_graph(counterfeit_executor)

    with pytest.raises(AssertionError, match="alternate random capsule reached"):
        _assert_only_canonical_random_dependencies(graph)
    assert any(
        path.endswith("nonlocal[builder].slot[alternate_capsule]")
        and candidate is alternate_capsule
        for path, candidate in graph
    )
    assert any(
        "nonlocal[builder].slot[alternate_capsule]" in path
        and candidate is alternate_randbelow
        for path, candidate in graph
    )


def test_executor_uses_exact_canonical_random_bridge_builder_schema() -> None:
    executor_nonlocals = inspect.getclosurevars(calibration_v2._execute_replicates).nonlocals
    builder = executor_nonlocals["build_random_capability"]
    builder_nonlocals = inspect.getclosurevars(builder).nonlocals

    assert builder is executable_base._build_random_index_capability_v2
    assert frozenset(builder_nonlocals) == frozenset(
        {"callable_value", "integrity_error_type", "seal", "tuple_type"}
    )
    assert builder_nonlocals["callable_value"] is callable
    assert builder_nonlocals["integrity_error_type"] is V2IntegrityError
    assert builder_nonlocals["seal"] is executable_base._RANDOM_INDEX_CAPABILITY_SEAL_V2
    assert builder_nonlocals["tuple_type"] is tuple
    assert builder.__defaults__ is None
    assert builder.__kwdefaults__ is None

    builder_graph = behavior_capture_graph(
        builder,
        follow_function_globals=False,
        node_limit=32,
        edge_limit=64,
        depth_limit=8,
    )
    nested_codes = tuple(
        (index, value)
        for index, value in enumerate(builder.__code__.co_consts)
        if type(value) is CodeType
    )
    assert len(nested_codes) == 1
    nested_index, nested_code = nested_codes[0]
    assert nested_code.co_name == "randbelow"
    assert builder_graph.opaque_callables == ()
    assert tuple(path for path, _ in builder_graph.occurrences) == (
        "root",
        "root.nonlocal[tuple_type]",
        "root.nonlocal[seal]",
        "root.nonlocal[integrity_error_type]",
        "root.nonlocal[callable_value]",
        "root.code",
        f"root.code.nested_code[{nested_index}]",
    )


def test_executor_and_verifier_share_one_frozen_random_capsule_identity() -> None:
    capsule = randomness._RANDOM_CAPSULE_V2
    canonical_operations = tuple(
        tuple.__getitem__(capsule, index) for index in range(4)
    )
    create_stream, seed_digest, randbelow, seed_digest_for_replicate = (
        canonical_operations
    )
    executor_nonlocals = inspect.getclosurevars(calibration_v2._execute_replicates).nonlocals
    verifier_nonlocals = inspect.getclosurevars(_verifier()).nonlocals
    assert frozenset(executor_nonlocals) == frozenset(
        {
            "analytical_failure_status",
            "analytical_failure_validity",
            "any_value",
            "bool_type",
            "build_random_capability",
            "callable_value",
            "cast_value",
            "complete_status",
            "create_random_stream",
            "encode_outcome",
            "exact_length",
            "exact_type",
            "finite_value",
            "float_type",
            "history_integrity_ops",
            "history_snapshot_type",
            "int_type",
            "integrity_error_type",
            "light_ops",
            "module_globals",
            "null_transform_token_type",
            "random_capsule",
            "random_capsule_type",
            "random_randbelow",
            "read_random_seed_digest",
            "replicate_ids",
            "require_budget",
            "require_outcome_type",
            "require_owned_prepared",
            "require_prepared_type",
            "result_vector_snapshot",
            "seed_digest_for_replicate",
            "selection_rule_type",
            "signature_link_type",
            "snapshot_selection",
            "statistic_scan_stage",
            "tuple_getitem",
            "tuple_type",
            "type_error_type",
            "validate_raw_replicate_vector",
            "validate_selector_output",
            "validate_transform_result",
            "verify_behavior_globals",
            "verify_codec_globals",
        }
    )
    assert frozenset(verifier_nonlocals) == frozenset(
        {
            "check_context",
            "check_final_context",
            "construct_counts",
            "construct_expectation",
            "counts_type",
            "expectation_type",
            "externals",
            "primitive_ops",
            "reaches",
            "resolution_context",
            "result_prefix",
            "schemas",
            "tuple_getitem",
            "types",
            "verify_observed",
        }
    )
    # The verifier counts exceedances with the one frozen tie rule shared with the kernel.
    from selcal.contracts_v2 import reaches_observed_decision

    assert verifier_nonlocals["reaches"] is reaches_observed_decision
    assert calibration_v2._execute_replicates.__defaults__ is None
    assert calibration_v2._execute_replicates.__kwdefaults__ is None
    assert executor_nonlocals["callable_value"] is callable
    assert executor_nonlocals["type_error_type"] is TypeError
    assert executor_nonlocals["encode_outcome"] is result_wire._encode_replicate_outcome
    codec_guard_nonlocals = inspect.getclosurevars(
        executor_nonlocals["verify_codec_globals"]
    ).nonlocals
    assert codec_guard_nonlocals["get_global"].__self__ is vars(result_wire)
    codec_globals = dict(codec_guard_nonlocals["expected_globals"])
    assert codec_globals["_project_outcome"] is result_wire._project_outcome
    assert codec_globals["canonical_json_bytes"] is canonical_json_bytes
    assert _verifier().__defaults__ is None
    assert _verifier().__kwdefaults__ is None

    behavior_guard_nonlocals = inspect.getclosurevars(
        executor_nonlocals["verify_behavior_globals"]
    ).nonlocals
    assert behavior_guard_nonlocals["get_global"].__self__ is executor_nonlocals[
        "module_globals"
    ]
    executor_graph = _callable_capture_graph(
        calibration_v2._execute_replicates,
        terminal_identities=(
            executor_nonlocals["module_globals"],
            behavior_guard_nonlocals["expected_globals"],
        ),
    )
    verifier_graph = _callable_capture_graph(_verifier())
    _assert_only_canonical_random_dependencies(executor_graph)
    _assert_only_canonical_random_dependencies(verifier_graph)
    operation_names = (
        "create_stream",
        "seed_digest",
        "randbelow",
        "seed_digest_for_replicate",
    )
    operation_by_code = {
        operation.__code__: (name, operation)
        for name, operation in zip(
            operation_names,
            canonical_operations,
            strict=True,
        )
    }
    def random_capture_paths(
        graph: tuple[tuple[str, object], ...],
    ) -> tuple[tuple[str, str, str], ...]:
        paths: list[tuple[str, str, str]] = []
        for path, candidate in graph:
            if type(candidate) is _RandomCapsuleV2:
                paths.append(
                    (
                        path,
                        "capsule",
                        "canonical" if candidate is capsule else "alternate",
                    )
                )
            elif type(candidate) is FunctionType and candidate.__code__ in operation_by_code:
                name, operation = operation_by_code[candidate.__code__]
                paths.append(
                    (
                        path,
                        name,
                        "canonical" if candidate is operation else "alternate",
                    )
                )
        return tuple(sorted(paths))

    assert random_capture_paths(executor_graph) == (
        ("root.nonlocal[create_random_stream]", "create_stream", "canonical"),
        ("root.nonlocal[random_capsule]", "capsule", "canonical"),
        ("root.nonlocal[random_randbelow]", "randbelow", "canonical"),
        ("root.nonlocal[read_random_seed_digest]", "seed_digest", "canonical"),
        (
            "root.nonlocal[seed_digest_for_replicate]",
            "seed_digest_for_replicate",
            "canonical",
        ),
        (
            "root.nonlocal[seed_digest_for_replicate].nonlocal[capsule]",
            "capsule",
            "canonical",
        ),
        (
            "root.nonlocal[seed_digest_for_replicate].nonlocal[capsule]."
            "tuple[create_stream]",
            "create_stream",
            "canonical",
        ),
        (
            "root.nonlocal[seed_digest_for_replicate].nonlocal[capsule]."
            "tuple[randbelow]",
            "randbelow",
            "canonical",
        ),
        (
            "root.nonlocal[seed_digest_for_replicate].nonlocal[capsule]."
            "tuple[seed_digest]",
            "seed_digest",
            "canonical",
        ),
        (
            "root.nonlocal[seed_digest_for_replicate].nonlocal[capsule]."
            "tuple[seed_digest_for_replicate]",
            "seed_digest_for_replicate",
            "canonical",
        ),
        (
            "root.nonlocal[seed_digest_for_replicate]."
            "nonlocal[seed_digest_for_replicate]",
            "seed_digest_for_replicate",
            "canonical",
        ),
    )
    # The identity walk now first reaches the same sealed seed operation through
    # the typed operation record. Preserve the complete canonical suffix graph.
    verifier_seed_path = (
        "root.nonlocal[verify_observed].nonlocal[ops].tuple[verify_terminal]."
        "nonlocal[observed].nonlocal[externals].tuple[seed_digest]"
    )
    assert tuple(
        (path.replace(verifier_seed_path, "root.nonlocal[seed_digest_operation]"), name, identity)
        for path, name, identity in random_capture_paths(verifier_graph)
    ) == (
        (
            "root.nonlocal[seed_digest_operation]",
            "seed_digest_for_replicate",
            "canonical",
        ),
        (
            "root.nonlocal[seed_digest_operation].nonlocal[capsule]",
            "capsule",
            "canonical",
        ),
        (
            "root.nonlocal[seed_digest_operation].nonlocal[capsule]."
            "tuple[create_stream]",
            "create_stream",
            "canonical",
        ),
        (
            "root.nonlocal[seed_digest_operation].nonlocal[capsule].tuple[randbelow]",
            "randbelow",
            "canonical",
        ),
        (
            "root.nonlocal[seed_digest_operation].nonlocal[capsule].tuple[seed_digest]",
            "seed_digest",
            "canonical",
        ),
        (
            "root.nonlocal[seed_digest_operation].nonlocal[capsule]."
            "tuple[seed_digest_for_replicate]",
            "seed_digest_for_replicate",
            "canonical",
        ),
        (
            "root.nonlocal[seed_digest_operation]."
            "nonlocal[seed_digest_for_replicate]",
            "seed_digest_for_replicate",
            "canonical",
        ),
    )
    assert type(capsule) is _RandomCapsuleV2
    assert executor_nonlocals["random_capsule"] is capsule
    assert executor_nonlocals["random_capsule_type"] is _RandomCapsuleV2
    assert executor_nonlocals["tuple_getitem"] is tuple.__getitem__
    assert executor_nonlocals["create_random_stream"] is create_stream
    assert executor_nonlocals["read_random_seed_digest"] is seed_digest
    assert executor_nonlocals["random_randbelow"] is randbelow
    assert executor_nonlocals["seed_digest_for_replicate"] is seed_digest_for_replicate
    assert (
        executor_nonlocals["build_random_capability"]
        is executable_base._build_random_index_capability_v2
    )
    assert verifier_nonlocals["externals"].seed_digest is seed_digest_for_replicate
    create_stream_nonlocals = inspect.getclosurevars(create_stream).nonlocals
    seed_digest_nonlocals = inspect.getclosurevars(seed_digest).nonlocals
    seed_oracle_nonlocals = inspect.getclosurevars(seed_digest_for_replicate).nonlocals
    assert create_stream_nonlocals["tuple_constructor"] is tuple
    assert seed_digest_nonlocals["stream_seed_getitem"] is tuple.__getitem__
    assert seed_oracle_nonlocals["tuple_getitem"] is tuple.__getitem__
    assert seed_oracle_nonlocals["capsule_type"] is _RandomCapsuleV2
    assert create_stream_nonlocals["derive_seed_digest"] is seed_oracle_nonlocals[
        "derive_seed_digest"
    ]
    assert seed_oracle_nonlocals["capsule"] is capsule
    assert "random_source_type" not in executor_nonlocals


def test_shared_random_capsule_uses_no_empty_local_identity_anchor() -> None:
    executor_deletes = {
        instruction.argval
        for instruction in dis.get_instructions(calibration_v2._execute_replicates)
        if instruction.opname == "DELETE_FAST"
    }
    oracle_deletes = {
        instruction.argval
        for instruction in dis.get_instructions(
            tuple.__getitem__(randomness._RANDOM_CAPSULE_V2, 3)
        )
        if instruction.opname == "DELETE_FAST"
    }

    assert "random_capsule_identity" not in executor_deletes
    assert "capsule_identity" not in oracle_deletes


def test_published_verifier_ignores_post_init_plan_digest_validator_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, exact = _complete_result()
    replacement_calls = 0

    def bomb(_value: object) -> str:
        nonlocal replacement_calls
        replacement_calls += 1
        raise AssertionError("replacement randomness._require_plan_digest executed")

    with monkeypatch.context() as patch:
        patch.setattr(randomness, "_require_plan_digest", bomb)
        _verifier()(result, exact)

    assert replacement_calls == 0


def test_published_verifier_ignores_post_init_math_isfinite_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, exact = _complete_result()
    replacement_calls = 0

    def bomb(_value: object) -> bool:
        nonlocal replacement_calls
        replacement_calls += 1
        raise AssertionError("replacement math.isfinite executed")

    with monkeypatch.context() as patch:
        patch.setattr(math, "isfinite", bomb)
        _verifier()(result, exact)

    assert replacement_calls == 0


@pytest.mark.parametrize(
    "dependency",
    (
        "_require_plan_digest",
        "_require_planned_replicates",
        "_require_replicate_id",
        "_SEED_DOMAIN",
        "NULL_TRANSFORM_STREAM",
        "hashlib.sha256",
        "np.random.PCG64",
    ),
)
def test_published_seed_oracle_ignores_exact_post_init_randomness_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    dependency: str,
) -> None:
    result, exact = _complete_result()
    replacement_calls = 0

    def bomb(*_args: object, **_kwargs: object) -> object:
        nonlocal replacement_calls
        replacement_calls += 1
        raise AssertionError(f"replacement randomness dependency executed: {dependency}")

    with monkeypatch.context() as patch:
        if dependency in {
            "_require_plan_digest",
            "_require_planned_replicates",
            "_require_replicate_id",
        }:
            patch.setattr(randomness, dependency, bomb)
        elif dependency == "_SEED_DOMAIN":
            patch.setattr(randomness, dependency, b"POST-INIT-SEED-DOMAIN")
        elif dependency == "NULL_TRANSFORM_STREAM":
            patch.setattr(randomness, dependency, b"post_init_stream")
        elif dependency == "hashlib.sha256":
            patch.setattr(randomness, "hashlib", SimpleNamespace(sha256=bomb))
        else:
            assert dependency == "np.random.PCG64"
            patch.setattr(
                randomness,
                "np",
                SimpleNamespace(random=SimpleNamespace(PCG64=bomb)),
            )
        _verifier()(result, exact)

    assert replacement_calls == 0


def test_published_seed_oracle_project_code_has_no_runtime_global_or_import_dispatch() -> None:
    assert _seed_operation_project_code_findings() == ()


def test_published_verifier_captures_isfinite_callable_without_math_module() -> None:
    verifier_leaves = inspect.getclosurevars(_verifier()).nonlocals

    assert "math" not in verifier_leaves
    assert verifier_leaves["externals"].isfinite is math.isfinite


def test_published_seed_oracle_matches_known_answer_and_multiple_replicate_ids() -> None:
    seed_digest = _published_seed_digest_operation()
    zero_plan = "0" * 64

    assert (
        seed_digest(zero_plan, 0, 1)
        == "b48487717b221abeca05be2756c8439153a3002b10af07825cf1947f2e8db1d0"
    )
    assert tuple(seed_digest(zero_plan, replicate_id, 4) for replicate_id in range(4)) == tuple(
        RealReplicateRandomSource(zero_plan, replicate_id, 4).seed_digest_sha256
        for replicate_id in range(4)
    )


@pytest.mark.parametrize(
    ("scientific_plan_sha256", "replicate_id", "planned_replicates"),
    (
        ("A" * 64, 0, 1),
        ("0" * 63, 0, 1),
        (0, 0, 1),
        ("0" * 64, 0, True),
        ("0" * 64, 0, 0),
        ("0" * 64, 0, 1_000_001),
        ("0" * 64, True, 1),
        ("0" * 64, -1, 1),
        ("0" * 64, 1, 1),
        ("0" * 64, 1 << 64, 1_000_000),
    ),
)
def test_published_seed_oracle_preserves_randomness_validation_contract(
    scientific_plan_sha256: object,
    replicate_id: object,
    planned_replicates: object,
) -> None:
    seed_digest = _published_seed_digest_operation()

    with pytest.raises(V2IntegrityError) as canonical_error:
        RealReplicateRandomSource(
            scientific_plan_sha256,  # type: ignore[arg-type]
            replicate_id,  # type: ignore[arg-type]
            planned_replicates,  # type: ignore[arg-type]
        )
    with pytest.raises(V2IntegrityError) as published_error:
        seed_digest(scientific_plan_sha256, replicate_id, planned_replicates)

    assert str(published_error.value) == str(canonical_error.value)


def _configured_resolution(
    statistic_name: str,
    null_name: str,
    terminal: str,
) -> PlanResolutionV2:
    statistic_params = {} if statistic_name == "lagged_pearson_v1" else {"bins": 3}
    null_params = (
        {"min_shift": 3 if terminal == "null_bind" else 1}
        if null_name == "circular_shift_v2"
        else {"block_length": 2}
    )
    return resolve_plan_v2(
        PlanRequestV2(
            candidates=(1, 2),
            statistic_name=statistic_name,
            statistic_params=statistic_params,
            selection_rule="max_upper",
            null_name=null_name,
            null_params=null_params,
            replicates=1,
            alpha=0.05,
            tie_tolerance=1e-12,
            root_seed=17,
        )
    )


def _terminal_result(
    statistic_name: str,
    null_name: str,
    terminal: str,
) -> tuple[CalibrationResult, PlanResolutionV2]:
    resolution = _configured_resolution(statistic_name, null_name, terminal)
    if terminal == "null_bind":
        pair = SeriesPair(
            source=np.arange(5, dtype=np.float64),
            target=np.arange(4, -1, -1, dtype=np.float64),
        )
    elif terminal == "observed_statistic_scan":
        pair = SeriesPair(
            source=np.ones(6, dtype=np.float64),
            target=np.asarray([0, 1, 3, 2, 5, 4], dtype=np.float64),
        )
    else:
        assert terminal == "complete"
        pair = _pair()
    return _calibrator()(pair, resolution), resolution


def _constructor_copy_with_plan_scalars(
    result: CalibrationResult,
    *,
    planned_replicates: int | None = None,
    alpha: float | None = None,
) -> CalibrationResult:
    exact_replicates = (
        result.planned_replicates if planned_replicates is None else planned_replicates
    )
    exact_alpha = result.alpha if alpha is None else alpha
    reject_null = None if result.p_value is None else result.p_value <= exact_alpha
    return CalibrationResult(
        status=result.status,
        failure_stage=result.failure_stage,
        semantic_input_sha256=result.semantic_input_sha256,
        scientific_plan_sha256=result.scientific_plan_sha256,
        planned_replicates=exact_replicates,
        alpha=exact_alpha,
        observed_results=result.observed_results,
        observed_selection=result.observed_selection,
        replicates=result.replicates,
        exceedance_count=result.exceedance_count,
        failure_count=result.failure_count,
        p_value=result.p_value,
        exceedance_bound_low=result.exceedance_bound_low,
        exceedance_bound_high=result.exceedance_bound_high,
        reject_null=reject_null,
        diagnostics=result.diagnostics,
    )


@contextmanager
def _temporarily_replaced_attribute(
    owner: object,
    name: str,
    replacement: object,
) -> Iterator[None]:
    original = inspect.getattr_static(owner, name)
    setattr(owner, name, replacement)
    try:
        yield
    finally:
        setattr(owner, name, original)


@contextmanager
def _temporarily_replaced_inherited_class_attribute(
    owner: type[object],
    name: str,
    replacement: object,
) -> Iterator[None]:
    was_defined = name in owner.__dict__
    original = inspect.getattr_static(owner, name)
    setattr(owner, name, replacement)
    try:
        yield
    finally:
        if was_defined:
            setattr(owner, name, original)
        else:
            delattr(owner, name)


class _DelegatingDescriptor:
    def __init__(self, original: object, *, returned: object | None = None) -> None:
        self._original = original
        self._returned = returned
        self.read_count = 0

    def __get__(self, instance: object, owner: type[object] | None = None) -> object:
        if instance is None:
            return self
        self.read_count += 1
        if self._returned is not None:
            return self._returned
        getter = self._original.__get__
        return getter(instance, owner)

    def __set__(self, instance: object, value: object) -> None:
        setter = self._original.__set__
        setter(instance, value)


def _bomb_property(counter: list[int], message: str) -> property:
    def bomb(_instance: object) -> object:
        counter[0] += 1
        raise AssertionError(message)

    return property(bomb)


def _constructor_valid_but_resolution_inconsistent_copy(
    result: CalibrationResult,
) -> CalibrationResult:
    """Use only ordinary constructors to forge a resolution-inconsistent vector."""

    original = result.observed_results[0]
    inconsistent = StatisticResult(
        candidate_id=original.candidate_id,
        estimate=original.estimate,
        selection_score=original.selection_score,
        support_n=original.support_n,
        validity=original.validity,
        diagnostics=original.diagnostics,
        backend_identity="constructor-valid-foreign-backend",
        preprocessing_identity=original.preprocessing_identity,
    )
    return CalibrationResult(
        status=result.status,
        failure_stage=result.failure_stage,
        semantic_input_sha256=result.semantic_input_sha256,
        scientific_plan_sha256=result.scientific_plan_sha256,
        planned_replicates=result.planned_replicates,
        alpha=result.alpha,
        observed_results=(inconsistent, *result.observed_results[1:]),
        observed_selection=result.observed_selection,
        replicates=result.replicates,
        exceedance_count=result.exceedance_count,
        failure_count=result.failure_count,
        p_value=result.p_value,
        exceedance_bound_low=result.exceedance_bound_low,
        exceedance_bound_high=result.exceedance_bound_high,
        reject_null=result.reject_null,
        diagnostics=result.diagnostics,
    )


def _constructor_valid_foreign_backend_copy(
    result: CalibrationResult,
    foreign_backend: str = "constructor-valid-foreign-backend",
) -> CalibrationResult:
    """Rebuild every statistic through public constructors under a foreign backend."""

    def rebuild_statistic(statistic: StatisticResult) -> StatisticResult:
        return StatisticResult(
            candidate_id=statistic.candidate_id,
            estimate=statistic.estimate,
            selection_score=statistic.selection_score,
            support_n=statistic.support_n,
            validity=statistic.validity,
            diagnostics=statistic.diagnostics,
            backend_identity=foreign_backend,
            preprocessing_identity=statistic.preprocessing_identity,
        )

    rebuilt_outcomes = tuple(
        ReplicateOutcome(
            replicate_id=outcome.replicate_id,
            seed_digest_sha256=outcome.seed_digest_sha256,
            status=outcome.status,
            failure_stage=outcome.failure_stage,
            transform_token=outcome.transform_token,
            statistic_results=tuple(
                rebuild_statistic(statistic) for statistic in outcome.statistic_results
            ),
            selection=outcome.selection,
            diagnostics=outcome.diagnostics,
        )
        for outcome in result.replicates
    )
    return CalibrationResult(
        status=result.status,
        failure_stage=result.failure_stage,
        semantic_input_sha256=result.semantic_input_sha256,
        scientific_plan_sha256=result.scientific_plan_sha256,
        planned_replicates=result.planned_replicates,
        alpha=result.alpha,
        observed_results=tuple(
            rebuild_statistic(statistic) for statistic in result.observed_results
        ),
        observed_selection=result.observed_selection,
        replicates=rebuilt_outcomes,
        exceedance_count=result.exceedance_count,
        failure_count=result.failure_count,
        p_value=result.p_value,
        exceedance_bound_low=result.exceedance_bound_low,
        exceedance_bound_high=result.exceedance_bound_high,
        reject_null=result.reject_null,
        diagnostics=result.diagnostics,
    )


def _import_target(target: str) -> tuple[object, str]:
    module_name, attribute = target.rsplit(".", maxsplit=1)
    return importlib.import_module(module_name), attribute


def _replace_all_result_identities(
    result: CalibrationResult,
    *,
    field: str,
    value: str,
) -> None:
    for statistic in result.observed_results:
        object.__setattr__(statistic, field, value)
    for outcome in result.replicates:
        for statistic in outcome.statistic_results:
            object.__setattr__(statistic, field, value)


def _replace_result_semantic_identity(
    result: CalibrationResult,
    resolution: PlanResolutionV2,
    semantic_digest: str,
) -> None:
    """Build a public-information-consistent but provenance-free digest variant."""

    object.__setattr__(result, "semantic_input_sha256", semantic_digest)
    if not result.replicates:
        return
    assert result.observed_results
    observed_length = result.observed_results[0].support_n + max(resolution.plan.candidates)
    for outcome in result.replicates:
        token = outcome.transform_token
        object.__setattr__(token, "semantic_input_sha256", semantic_digest)
        object.__setattr__(
            token,
            "bound_null_owner_sha256",
            owner_digest(
                semantic_input_sha256=semantic_digest,
                scientific_plan_sha256=token.scientific_plan_sha256,
                null_parameter_sha256=token.null_parameter_sha256,
                observed_length=observed_length,
            ),
        )


def _apply_integrity_drift(
    case: str,
    result: CalibrationResult,
    exact: PlanResolutionV2,
) -> None:
    if case == "stored_B":
        object.__setattr__(result, "planned_replicates", result.planned_replicates + 1)
    elif case == "replicate_id_duplicate":
        object.__setattr__(result.replicates[1], "replicate_id", 0)
    elif case == "replicate_id_missing":
        object.__setattr__(result, "replicates", result.replicates[:-1])
    elif case == "replicate_id_out_of_range":
        object.__setattr__(result.replicates[-1], "replicate_id", result.planned_replicates)
    elif case == "replicate_id_unordered":
        object.__setattr__(
            result,
            "replicates",
            (result.replicates[1], result.replicates[0], *result.replicates[2:]),
        )
    elif case == "candidate_id_duplicate":
        object.__setattr__(result.observed_results[1], "candidate_id", 1)
    elif case == "candidate_id_missing":
        object.__setattr__(result, "observed_results", result.observed_results[:-1])
    elif case == "candidate_id_out_of_range":
        object.__setattr__(result.observed_results[0], "candidate_id", 99)
    elif case == "candidate_id_unordered":
        object.__setattr__(result, "observed_results", tuple(reversed(result.observed_results)))
    elif case == "planned_candidates":
        object.__setattr__(exact.plan, "candidates", (1, 3))
    elif case == "selection_rule":
        object.__setattr__(exact.plan, "selection_rule", SelectionRule.MAX_ABSOLUTE)
    elif case == "tie_tolerance":
        object.__setattr__(exact.plan, "tie_tolerance", 0.5)
    elif case == "statistic_adapter":
        object.__setattr__(exact.adapters, "statistic", object())
    elif case == "null_adapter":
        object.__setattr__(exact.adapters, "null_model", object())
    elif case == "backend_all_vectors":
        _replace_all_result_identities(result, field="backend_identity", value="forged-backend")
    elif case == "preprocessing_all_vectors":
        _replace_all_result_identities(
            result,
            field="preprocessing_identity",
            value="forged-preprocessing",
        )
    elif case == "support_n_one_vector":
        statistic = result.observed_results[0]
        object.__setattr__(statistic, "support_n", statistic.support_n + 1)
    elif case == "support_n_all_vectors":
        for statistic in result.observed_results:
            object.__setattr__(statistic, "support_n", statistic.support_n + 1)
        for outcome in result.replicates:
            for statistic in outcome.statistic_results:
                object.__setattr__(statistic, "support_n", statistic.support_n + 1)
    elif case == "selection_score":
        object.__setattr__(result.observed_results[0], "selection_score", -999.0)
    elif case == "tie_set":
        assert result.observed_selection is not None
        object.__setattr__(result.observed_selection, "tied_candidates", (1, 2))
    elif case == "decision_statistic":
        assert result.observed_selection is not None
        object.__setattr__(result.observed_selection, "decision_statistic", -999.0)
    elif case == "seed_digest":
        object.__setattr__(result.replicates[-1], "seed_digest_sha256", "f" * 64)
    elif case == "token_semantic_input":
        object.__setattr__(
            result.replicates[-1].transform_token,
            "semantic_input_sha256",
            "f" * 64,
        )
    elif case == "token_plan":
        object.__setattr__(
            result.replicates[-1].transform_token,
            "scientific_plan_sha256",
            "f" * 64,
        )
    elif case == "token_null_name":
        object.__setattr__(
            result.replicates[-1].transform_token,
            "null_name",
            "block_shuffle_v2",
        )
    elif case == "token_null_parameters":
        object.__setattr__(
            result.replicates[-1].transform_token,
            "null_parameter_sha256",
            "f" * 64,
        )
    elif case == "token_bound_owner":
        object.__setattr__(
            result.replicates[-1].transform_token,
            "bound_null_owner_sha256",
            "f" * 64,
        )
    elif case == "result_semantic_input":
        object.__setattr__(result, "semantic_input_sha256", "f" * 64)
    elif case == "result_plan_hash":
        object.__setattr__(result, "scientific_plan_sha256", "f" * 64)
    elif case == "alpha":
        object.__setattr__(result, "alpha", 0.25)
    elif case == "final_plan_rehash":
        object.__setattr__(exact.plan, "root_seed", 18)
    else:
        raise AssertionError(f"unknown integrity drift: {case}")


def test_resolution_bound_verifier_accepts_exact_task8b_evidence() -> None:
    result, exact = _complete_result()
    _verifier()(result, exact)


@pytest.mark.parametrize(
    "case",
    [
        "stored_B",
        "replicate_id_duplicate",
        "replicate_id_missing",
        "replicate_id_out_of_range",
        "replicate_id_unordered",
        "candidate_id_duplicate",
        "candidate_id_missing",
        "candidate_id_out_of_range",
        "candidate_id_unordered",
        "planned_candidates",
        "selection_rule",
        "tie_tolerance",
        "statistic_adapter",
        "null_adapter",
        "backend_all_vectors",
        "preprocessing_all_vectors",
        "support_n_one_vector",
        "support_n_all_vectors",
        "selection_score",
        "tie_set",
        "decision_statistic",
        "seed_digest",
        "token_semantic_input",
        "token_plan",
        "token_null_name",
        "token_null_parameters",
        "token_bound_owner",
        "result_semantic_input",
        "result_plan_hash",
        "alpha",
        "final_plan_rehash",
    ],
)
def test_resolution_bound_verifier_rejects_every_identity_drift(case: str) -> None:
    result, exact = _complete_result()
    _apply_integrity_drift(case, result, exact)

    with pytest.raises(V2IntegrityError):
        _verifier()(result, exact)


def test_verifier_rejects_a_structurally_valid_token_outside_the_bound_state_universe() -> None:
    result, exact = _complete_result()
    original = result.replicates[0]
    token = original.transform_token
    support_sizes = {
        statistic.support_n
        for vector in (
            result.observed_results,
            *(outcome.statistic_results for outcome in result.replicates),
        )
        for statistic in vector
    }
    assert len(support_sizes) == 1
    # The frozen max-candidate common-support contract makes this derivation unique.
    observed_length = support_sizes.pop() + max(exact.plan.candidates)
    out_of_universe = NullTransformToken(
        schema=token.schema,
        null_name=token.null_name,
        null_parameter_sha256=token.null_parameter_sha256,
        semantic_input_sha256=token.semantic_input_sha256,
        scientific_plan_sha256=token.scientific_plan_sha256,
        bound_null_owner_sha256=token.bound_null_owner_sha256,
        is_identity=False,
        state=CircularShiftStateV2(
            schema="selcal.circular-shift-state.v2",
            shift=observed_length,
        ),
    )
    object.__setattr__(original, "transform_token", out_of_universe)

    with pytest.raises(V2IntegrityError, match=r"token|state|shift|universe|range"):
        _verifier()(result, exact)


def test_verifier_uses_only_the_sealed_resolution_and_result_without_rebinding() -> None:
    result, exact = _complete_result()
    bind_codes = {
        type(exact.adapters.statistic).bind.__code__,
        type(exact.adapters.null_model).bind.__code__,
    }
    bind_calls: list[FrameType] = []

    def profile(frame: FrameType, event: str, _arg: object) -> None:
        if event == "call" and frame.f_code in bind_codes:
            bind_calls.append(frame)

    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        _verifier()(result, exact)
    finally:
        sys.setprofile(previous)
    assert bind_calls == []


def test_verifier_does_not_trust_a_rebound_post_init_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, exact = _complete_result()
    object.__setattr__(result, "p_value", 0.123)
    monkeypatch.setattr(
        contracts_v2,
        "_CALIBRATION_RESULT_POST_INIT",
        lambda _result: None,
        raising=False,
    )

    with pytest.raises(V2IntegrityError):
        _verifier()(result, exact)


def test_resolution_ownership_rejects_coordinated_result_callback_rebinding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, exact = _complete_result()
    registration = resolution_v2._STATISTIC_REGISTRATIONS_V2[exact.plan.statistic_name]
    canonical = resolution_v2._CANONICAL_STATISTIC_REGISTRATIONS_V2[exact.plan.statistic_name]

    def forged_identity(_adapter: object) -> tuple[str, str]:
        return "forged-backend", "forged-preprocessing"

    _replace_all_result_identities(
        result,
        field="backend_identity",
        value="forged-backend",
    )
    _replace_all_result_identities(
        result,
        field="preprocessing_identity",
        value="forged-preprocessing",
    )
    original_identity = registration.statistic_result_identity
    coordinated = dict(resolution_v2._CANONICAL_STATISTIC_REGISTRATIONS_V2)
    coordinated[exact.plan.statistic_name] = canonical._replace(
        statistic_result_identity=forged_identity
    )
    monkeypatch.setattr(
        resolution_v2,
        "_CANONICAL_STATISTIC_REGISTRATIONS_V2",
        MappingProxyType(coordinated),
    )
    try:
        object.__setattr__(
            registration,
            "statistic_result_identity",
            forged_identity,
        )
        with pytest.raises(V2IntegrityError):
            _verifier()(result, exact)
    finally:
        object.__setattr__(
            registration,
            "statistic_result_identity",
            original_identity,
        )


def _install_replicate_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    indices = (1, 2, 3)
    original_sample = _BoundCircularShiftV2.sample_token
    sample_count = 0

    class ScriptedIndexCapability:
        def __init__(self, index: int) -> None:
            self._index = index

        def randbelow(self, bound: int, /) -> int:
            assert 0 <= self._index < bound
            return self._index

    def scripted_sample(
        self: _BoundCircularShiftV2,
        random: object,
        /,
    ) -> NullTransformToken:
        del random
        nonlocal sample_count
        replicate_id = sample_count
        sample_count += 1
        assert replicate_id < len(indices)
        return original_sample(self, ScriptedIndexCapability(indices[replicate_id]))

    # Slice 2 transitional null-kernel seam; preserve the production token
    # constructor and range checks while keeping the random capsule sealed.
    monkeypatch.setattr(_BoundCircularShiftV2, "sample_token", scripted_sample)
    call_count = 0

    def evaluate(
        bound: _BoundLaggedPearsonAdapter,
        supplied: SeriesPair,
        /,
    ) -> tuple[StatisticResult, ...]:
        nonlocal call_count
        del supplied
        current = call_count
        call_count += 1
        failure = current == 2
        return tuple(
            StatisticResult(
                candidate_id=candidate,
                estimate=None if failure else float(candidate + current),
                selection_score=None,
                support_n=bound.observed_length - max(bound.candidates),
                validity=Validity.ANALYTIC_FAILURE if failure else Validity.VALID,
                diagnostics=("scripted_task8c_failure",) if failure else (),
                backend_identity=bound.backend_identity,
                preprocessing_identity=bound.preprocessing_identity,
            )
            for candidate in bound.candidates
        )

    monkeypatch.setattr(_BoundLaggedPearsonAdapter, "evaluate_all", evaluate)


def _terminal_case(
    monkeypatch: pytest.MonkeyPatch,
    path: str,
) -> tuple[SeriesPair, PlanResolutionV2, RunStatus, RunFailureStage | None]:
    if path == "complete":
        return _pair(), _resolution(), RunStatus.COMPLETE, None
    if path == "null_bind":
        pair = SeriesPair(
            source=np.asarray([0, 1, 2, 3, 4], dtype=np.float64),
            target=np.asarray([4, 3, 2, 1, 0], dtype=np.float64),
        )
        return pair, _resolution(min_shift=3), RunStatus.NOT_EVALUABLE, RunFailureStage.NULL_BIND
    if path == "observed_statistic_scan":
        pair = SeriesPair(
            source=np.asarray([1, 1, 1, 1, 1, 1], dtype=np.float64),
            target=np.asarray([0, 1, 3, 2, 5, 4], dtype=np.float64),
        )
        return (
            pair,
            _resolution(),
            RunStatus.NOT_EVALUABLE,
            RunFailureStage.OBSERVED_STATISTIC_SCAN,
        )
    if path == "replicate_execution":
        _install_replicate_failure(monkeypatch)
        return (
            _pair(),
            _resolution(),
            RunStatus.NOT_EVALUABLE,
            RunFailureStage.REPLICATE_EXECUTION,
        )
    raise AssertionError(f"unknown terminal path: {path}")


@pytest.mark.parametrize(
    "path",
    ["complete", "null_bind", "observed_statistic_scan", "replicate_execution"],
)
def test_every_public_terminal_path_verifies_exactly_once_before_return(
    monkeypatch: pytest.MonkeyPatch,
    path: str,
) -> None:
    supplied, exact, expected_status, expected_stage = _terminal_case(monkeypatch, path)
    verified: list[CalibrationResult] = []

    def spy(result: CalibrationResult, resolution: PlanResolutionV2, /) -> None:
        del resolution
        verified.append(result)

    terminal = calibration_v2._verify_terminal_result_before_return
    terminal_nonlocals = inspect.getclosurevars(terminal).nonlocals
    sealed_verifier = cast(Callable[..., object], terminal_nonlocals["real_verifier"])
    verifier_nonlocals = inspect.getclosurevars(sealed_verifier).nonlocals
    context_reader = cast(Callable[..., object], verifier_nonlocals["resolution_context"])
    final_resolution_reader = cast(Callable[..., object], terminal_nonlocals["require_resolution"])
    events: list[str] = []

    def observe(frame: FrameType, event: str, _arg: object) -> None:
        if event != "call":
            return
        if frame.f_code is spy.__code__:
            events.append("public")
        elif frame.f_code is sealed_verifier.__code__:
            events.append("sealed")
        elif frame.f_code is context_reader.__code__:
            events.append("sealed_context_reread")
        elif (
            frame.f_code is final_resolution_reader.__code__
            and frame.f_back is not None
            and frame.f_back.f_code is terminal.__code__
        ):
            events.append("final_reread")

    monkeypatch.setattr(calibration_v2, "verify_calibration_result", spy)
    sys.setprofile(observe)
    try:
        result = _calibrator()(supplied, exact)
    finally:
        sys.setprofile(None)

    assert result.status is expected_status
    assert result.failure_stage is expected_stage
    assert verified == [result]
    assert verified[0] is result
    assert events == [
        "public",
        "sealed",
        "sealed_context_reread",
        "sealed_context_reread",
        "final_reread",
    ]


@pytest.mark.parametrize(
    "target",
    (
        "selcal.contracts_v2._verify_result_vector_against_plan",
        "selcal.canonical_v2.scientific_plan_v2_sha256",
        "selcal.resolution_v2._snapshot_result_verification_context_v2",
        "selcal.randomness.ReplicateRandomSource",
    ),
)
def test_post_seal_cross_module_verifier_rebinding_cannot_accept_forgery(
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    result, exact = _complete_result()
    forged = _constructor_valid_but_resolution_inconsistent_copy(result)
    module, attribute = _import_target(target)
    replacement_calls = 0

    def bomb(*args: object, **kwargs: object) -> object:
        nonlocal replacement_calls
        del args, kwargs
        replacement_calls += 1
        raise AssertionError(f"rebound verifier dependency executed: {target}")

    monkeypatch.setattr(module, attribute, bomb)
    with pytest.raises(V2IntegrityError):
        _verifier()(forged, exact)
    assert replacement_calls == 0

    _verifier()(result, exact)
    assert replacement_calls == 0


@pytest.mark.parametrize(
    "target",
    ("_read_verifier_slots", "_raise_integrity", "math"),
)
def test_post_seal_active_verifier_dependency_rebinding_cannot_execute(
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    result, exact = _complete_result()
    forged = _constructor_valid_but_resolution_inconsistent_copy(result)
    replacement_calls = 0

    def bomb(*args: object, **kwargs: object) -> object:
        nonlocal replacement_calls
        del args, kwargs
        replacement_calls += 1
        raise AssertionError(f"rebound active verifier dependency executed: {target}")

    replacement: object
    if target == "math":
        replacement = SimpleNamespace(isfinite=bomb)
    else:
        replacement = bomb
    monkeypatch.setattr(contracts_v2, target, replacement)

    with pytest.raises(V2IntegrityError):
        _verifier()(forged, exact)
    assert replacement_calls == 0


@pytest.mark.parametrize("plan_field", ("alpha", "replicates"))
def test_plan_fields_are_not_read_through_a_replaced_getattribute(
    plan_field: str,
) -> None:
    if plan_field == "alpha":
        result, exact = _complete_result()
        forged_value: object = 0.9
        forged = _constructor_copy_with_plan_scalars(result, alpha=0.9)
    else:
        result, exact = _terminal_result(
            "lagged_pearson_v1",
            "circular_shift_v2",
            "null_bind",
        )
        forged_value = result.planned_replicates + 1
        forged = _constructor_copy_with_plan_scalars(
            result,
            planned_replicates=cast(int, forged_value),
        )
    original_getattribute = inspect.getattr_static(
        ResolvedScientificPlanV2,
        "__getattribute__",
    )
    replacement_calls = 0

    def replaced_getattribute(self: object, name: str) -> object:
        nonlocal replacement_calls
        if name == plan_field:
            replacement_calls += 1
            return forged_value
        return original_getattribute(self, name)

    with _temporarily_replaced_attribute(
        ResolvedScientificPlanV2,
        "__getattribute__",
        replaced_getattribute,
    ):
        with pytest.raises(V2IntegrityError):
            _verifier()(forged, exact)

    assert replacement_calls == 0


def test_context_backend_descriptor_replacement_cannot_accept_foreign_backend() -> None:
    result, exact = _complete_result()
    foreign_backend = "constructor-valid-foreign-backend"
    forged = _constructor_valid_foreign_backend_copy(result, foreign_backend)
    context_type = resolution_v2._ResultVerificationContextV2
    original = inspect.getattr_static(context_type, "backend_identity")
    replacement = _DelegatingDescriptor(original, returned=foreign_backend)

    with _temporarily_replaced_attribute(
        context_type,
        "backend_identity",
        replacement,
    ):
        with pytest.raises(V2IntegrityError):
            _verifier()(forged, exact)

    assert replacement.read_count == 0


def test_legacy_context_plan_property_is_outside_the_sealed_reader_graph() -> None:
    result, exact = _complete_result()
    sealed_context = _sealed_resolution_context_reader()(exact)
    assert type(sealed_context) is tuple
    assert len(sealed_context) == 11
    assert all(
        type(item) is not resolution_v2._ResultVerificationContextV2 for item in sealed_context
    )
    replacement_calls = [0]
    replacement = _bomb_property(
        replacement_calls,
        "replacement context plan getter executed",
    )

    with _temporarily_replaced_attribute(
        resolution_v2._ResultVerificationContextV2,
        "plan",
        replacement,
    ):
        _verifier()(result, exact)

    assert replacement_calls == [0]


@pytest.mark.parametrize(
    ("owner_name", "field_name"),
    (
        ("_ResolutionOwnershipSnapshotV2", "plan"),
        ("_ResolutionOwnershipSnapshotV2", "result_token_verifier"),
        ("_AdapterRegistryIdentityV2", "factories"),
        ("_AdapterRegistryIdentityV2", "factories_mapping"),
        ("_AdapterRegistrationV2", "entry"),
    ),
)
def test_context_metadata_descriptor_replacement_is_typed_without_execution(
    owner_name: str,
    field_name: str,
) -> None:
    result, exact = _complete_result()
    owner = cast(type[object], getattr(resolution_v2, owner_name))
    replacement_calls = [0]
    replacement = _bomb_property(
        replacement_calls,
        f"replacement {owner_name}.{field_name} getter executed",
    )

    with _temporarily_replaced_attribute(owner, field_name, replacement):
        with pytest.raises(V2IntegrityError):
            _verifier()(result, exact)

    assert replacement_calls == [0]


@pytest.mark.parametrize(
    ("statistic_name", "expected_backend"),
    (
        (
            "lagged_pearson_v1",
            "selcal.lagged_pearson.v1|numpy=review-forged|units=pearson_correlation",
        ),
        (
            "equal_width_binned_nette_v1",
            "selcal.equal_width_binned_nette.v1|numpy=review-forged|bins=3|units=nats",
        ),
    ),
)
def test_statistic_identity_does_not_read_rebound_numpy_version(
    statistic_name: str,
    expected_backend: str,
) -> None:
    result, exact = _terminal_result(
        statistic_name,
        "circular_shift_v2",
        "complete",
    )
    forged = _constructor_valid_foreign_backend_copy(result, expected_backend)

    class ForeignNumpy:
        def __init__(self) -> None:
            self.version_reads = 0

        @property
        def __version__(self) -> str:
            self.version_reads += 1
            return "review-forged"

    replacement = ForeignNumpy()
    with _temporarily_replaced_attribute(resolution_v2, "np", replacement):
        with pytest.raises(V2IntegrityError):
            _verifier()(forged, exact)

    assert replacement.version_reads == 0


@pytest.mark.parametrize(
    ("statistic_name", "helper_name"),
    (
        ("lagged_pearson_v1", "_pearson_identity_snapshot"),
        ("equal_width_binned_nette_v1", "_binned_identity_snapshot"),
    ),
)
def test_statistic_identity_does_not_execute_rebound_snapshot_helper(
    statistic_name: str,
    helper_name: str,
) -> None:
    result, exact = _terminal_result(
        statistic_name,
        "circular_shift_v2",
        "complete",
    )
    forged = _constructor_valid_foreign_backend_copy(result)
    replacement_calls = 0

    def bomb(*args: object, **kwargs: object) -> object:
        nonlocal replacement_calls
        del args, kwargs
        replacement_calls += 1
        raise AssertionError(f"rebound {helper_name} executed")

    with _temporarily_replaced_attribute(resolution_v2, helper_name, bomb):
        with pytest.raises(V2IntegrityError):
            _verifier()(forged, exact)

    assert replacement_calls == 0


@pytest.mark.parametrize(
    ("null_name", "terminal", "null_type", "field_name"),
    (
        (
            "circular_shift_v2",
            "null_bind",
            CircularShiftNullV2,
            "min_shift",
        ),
        (
            "circular_shift_v2",
            "observed_statistic_scan",
            CircularShiftNullV2,
            "min_shift",
        ),
        (
            "block_shuffle_v2",
            "null_bind",
            BlockShuffleNullV2,
            "block_length",
        ),
        (
            "block_shuffle_v2",
            "observed_statistic_scan",
            BlockShuffleNullV2,
            "block_length",
        ),
    ),
)
def test_zero_replicate_terminal_verification_detects_null_descriptor_drift(
    null_name: str,
    terminal: str,
    null_type: type[object],
    field_name: str,
) -> None:
    result, exact = _terminal_result("lagged_pearson_v1", null_name, terminal)
    assert result.replicates == ()
    replacement_calls = [0]
    replacement = _bomb_property(
        replacement_calls,
        f"replacement {null_name}.{field_name} getter executed",
    )

    with _temporarily_replaced_attribute(null_type, field_name, replacement):
        with pytest.raises(V2IntegrityError):
            _verifier()(result, exact)

    assert replacement_calls == [0]


def _sealed_resolution_context_reader() -> Callable[[object], object]:
    candidate = inspect.getclosurevars(_verifier()).nonlocals["resolution_context"]
    assert callable(candidate)
    return cast(Callable[[object], object], candidate)


def _sealed_context_values(resolution: PlanResolutionV2) -> tuple[object, ...]:
    context = _sealed_resolution_context_reader()(resolution)
    assert type(context) is tuple
    assert len(context) == 11
    return context


def _sealed_science_leaf_maps() -> tuple[object, object, object, object]:
    nonlocals = inspect.getclosurevars(_sealed_resolution_context_reader()).nonlocals
    return (
        nonlocals["statistic_snapshots"],
        nonlocals["statistic_identities"],
        nonlocals["null_snapshots"],
        nonlocals["token_verifiers"],
    )


@pytest.mark.parametrize("root_seed", (0, 17, 2**64 - 1))
@pytest.mark.parametrize("replicates", (1, 1_000, 1_000_000))
@pytest.mark.parametrize(
    "selection_rule",
    (SelectionRule.MAX_UPPER, SelectionRule.MAX_ABSOLUTE),
)
@pytest.mark.parametrize(
    ("null_name", "null_params"),
    (
        ("circular_shift_v2", {"min_shift": 1}),
        ("block_shuffle_v2", {"block_length": 2}),
    ),
)
@pytest.mark.parametrize(
    ("statistic_name", "statistic_params"),
    (
        ("lagged_pearson_v1", {}),
        ("equal_width_binned_nette_v1", {"bins": 3}),
    ),
)
def test_sealed_context_matches_legacy_plan_and_pure_context_across_72_valid_plans(
    statistic_name: str,
    statistic_params: dict[str, int],
    null_name: str,
    null_params: dict[str, int],
    selection_rule: SelectionRule,
    replicates: int,
    root_seed: int,
) -> None:
    resolution = resolve_plan_v2(
        PlanRequestV2(
            candidates=(1, 2),
            statistic_name=statistic_name,
            statistic_params=statistic_params,
            selection_rule=selection_rule,
            null_name=null_name,
            null_params=null_params,
            replicates=replicates,
            alpha=0.05,
            tie_tolerance=1e-12,
            root_seed=root_seed,
        )
    )
    legacy_context = resolution_v2._snapshot_result_verification_context_v2(resolution)
    legacy_plan_bytes = canonical_json_bytes(scientific_plan_v2_payload(resolution.plan))
    legacy_plan_digest = scientific_plan_v2_sha256(resolution.plan)
    _, ownership = _ownership_records()[id(resolution)]
    (
        sealed_plan,
        sealed_plan_digest,
        sealed_candidates,
        sealed_rule,
        sealed_replicates,
        sealed_alpha,
        sealed_tie_tolerance,
        sealed_backend,
        sealed_preprocessing,
        sealed_null,
        sealed_token_verifier,
    ) = _sealed_context_values(resolution)
    _, _, _, token_verifiers = _sealed_science_leaf_maps()

    assert ownership.plan_canonical_bytes == legacy_plan_bytes
    assert ownership.plan_sha256 == legacy_plan_digest
    assert legacy_context.plan is resolution.plan
    assert legacy_context.plan_sha256 == legacy_plan_digest
    assert sealed_plan is legacy_context.plan
    assert sealed_plan_digest == legacy_context.plan_sha256
    assert sealed_candidates == resolution.plan.candidates
    assert sealed_rule is resolution.plan.selection_rule
    assert sealed_replicates == resolution.plan.replicates
    assert sealed_alpha == resolution.plan.alpha
    assert sealed_tie_tolerance == resolution.plan.tie_tolerance
    assert sealed_backend == legacy_context.backend_identity
    assert sealed_preprocessing == legacy_context.preprocessing_identity
    assert sealed_null is legacy_context.null_adapter
    exact_token_verifiers = cast(Mapping[type[object], object], token_verifiers)
    assert sealed_token_verifier is exact_token_verifiers[type(sealed_null)]


@pytest.mark.parametrize("method_name", ("__eq__", "__hash__"))
def test_resolution_identity_lookup_never_executes_replaced_eq_or_hash(
    method_name: str,
) -> None:
    result, exact = _complete_result()
    replacement_calls = 0

    def bomb(*_args: object, **_kwargs: object) -> object:
        nonlocal replacement_calls
        replacement_calls += 1
        raise AssertionError(f"replacement PlanResolutionV2.{method_name} executed")

    with _temporarily_replaced_inherited_class_attribute(
        PlanResolutionV2,
        method_name,
        bomb,
    ):
        try:
            _verifier()(result, exact)
        except V2IntegrityError:
            outcome = "typed fail-closed"
        else:
            outcome = "unchanged acceptance"

    assert outcome in {"unchanged acceptance", "typed fail-closed"}
    assert replacement_calls == 0


def test_resolution_identity_record_requires_the_original_live_object() -> None:
    result, exact = _complete_result()
    other = _resolution(candidates=(1, 3))
    exact_key = id(exact)
    original_record = _ownership_records()[exact_key]
    _ownership_records()[exact_key] = _ownership_records()[id(other)]
    try:
        with pytest.raises(V2IntegrityError, match="resolution identity drifted"):
            _verifier()(result, exact)
    finally:
        _ownership_records()[exact_key] = original_record


def test_resolution_identity_record_is_removed_after_collection() -> None:
    resolution = _resolution(candidates=(1, 4))
    resolution_key = id(resolution)
    resolution_reference = weakref.ref(resolution)
    assert resolution_key in _ownership_records()

    del resolution
    gc.collect()

    assert resolution_reference() is None
    assert resolution_key not in _ownership_records()


def test_frozen_identity_entrypoints_reject_wrong_public_input_types() -> None:
    outsider = object()

    with pytest.raises(V2IntegrityError, match="exact resolver-owned PlanResolutionV2"):
        resolution_v2._require_resolver_owned_resolution_v2(outsider)
    with pytest.raises(TypeError, match="exact PlanRequestV2"):
        resolve_plan_v2(cast(Any, outsider))


def test_resolution_cleanup_is_silent_when_the_record_was_already_removed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolution = _resolution(candidates=(1, 5))
    resolution_key = id(resolution)
    record = _ownership_records().pop(resolution_key)
    resolution_reference, _ = record
    unraisable: list[object] = []
    monkeypatch.setattr(sys, "unraisablehook", unraisable.append)

    del resolution
    gc.collect()

    assert resolution_reference() is None
    assert unraisable == []
    assert resolution_key not in _ownership_records()


def test_resolution_cleanup_leaves_an_unidentifiable_empty_record_safely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolution = _resolution(candidates=(1, 6))
    resolution_key = id(resolution)
    record = _ownership_records()[resolution_key]
    resolution_reference, _ = record
    unraisable: list[object] = []
    monkeypatch.setattr(sys, "unraisablehook", unraisable.append)
    _ownership_records()[resolution_key] = cast(Any, ())

    del resolution
    gc.collect()

    try:
        assert resolution_reference() is None
        assert unraisable == []
        assert _ownership_records()[resolution_key] == ()
    finally:
        _ownership_records().pop(resolution_key, None)


def test_stale_resolution_cleanup_cannot_delete_a_reused_identity_record() -> None:
    stale = _resolution(candidates=(2, 3))
    current = _resolution(candidates=(2, 4))
    stale_key = id(stale)
    current_key = id(current)
    stale_record = _ownership_records()[stale_key]
    current_record = _ownership_records()[current_key]
    stale_reference, _ = stale_record
    _ownership_records()[stale_key] = current_record

    del stale
    gc.collect()

    try:
        assert stale_reference() is None
        assert _ownership_records()[stale_key] is current_record
    finally:
        _ownership_records().pop(stale_key, None)


class _BombCallable:
    def __init__(self, message: str) -> None:
        self.message = message
        self.calls = 0

    def __call__(self, *_args: object, **_kwargs: object) -> object:
        self.calls += 1
        raise AssertionError(self.message)


def test_resolution_identity_record_never_executes_a_replacement_callable() -> None:
    result, exact = _complete_result()
    key = id(exact)
    original_record = _ownership_records()[key]
    _, snapshot = original_record
    bomb = _BombCallable("replacement resolution reference executed")
    _ownership_records()[key] = (cast(Any, bomb), snapshot)
    try:
        with pytest.raises(V2IntegrityError):
            _verifier()(result, exact)
    finally:
        _ownership_records()[key] = original_record

    assert bomb.calls == 0


@pytest.mark.parametrize(
    "record_kind",
    (
        "non_tuple",
        "length_one",
        "length_three",
        "plain_reference",
        "callable_reference",
        "wrong_snapshot_type",
    ),
)
def test_resolution_identity_record_shape_and_field_types_fail_closed(
    record_kind: str,
) -> None:
    result, exact = _complete_result()
    key = id(exact)
    original_record = _ownership_records()[key]
    reference, snapshot = original_record
    bomb = _BombCallable("malformed identity-record callable executed")
    malformed: object
    if record_kind == "non_tuple":
        malformed = object()
    elif record_kind == "length_one":
        malformed = (reference,)
    elif record_kind == "length_three":
        malformed = (reference, snapshot, object())
    elif record_kind == "plain_reference":
        malformed = (object(), snapshot)
    elif record_kind == "callable_reference":
        malformed = (bomb, snapshot)
    else:
        assert record_kind == "wrong_snapshot_type"
        malformed = (reference, object())
    _ownership_records()[key] = cast(Any, malformed)
    try:
        with pytest.raises(V2IntegrityError):
            _verifier()(result, exact)
    finally:
        _ownership_records()[key] = original_record

    assert bomb.calls == 0


@pytest.mark.parametrize("malformed_shape", ("length_one", "length_three"))
def test_malformed_identity_record_cleanup_is_silent_and_removes_the_record(
    monkeypatch: pytest.MonkeyPatch,
    malformed_shape: str,
) -> None:
    resolution = _resolution(candidates=(3, 4))
    key = id(resolution)
    reference, snapshot = _ownership_records()[key]
    malformed = (reference,) if malformed_shape == "length_one" else (reference, snapshot, object())
    _ownership_records()[key] = cast(Any, malformed)
    unraisable: list[object] = []
    monkeypatch.setattr(sys, "unraisablehook", unraisable.append)

    del resolution
    gc.collect()

    try:
        assert reference() is None
        assert unraisable == []
        assert key not in _ownership_records()
    finally:
        _ownership_records().pop(key, None)


@pytest.mark.parametrize("record_length", (0, 1, 3))
def test_exact_identity_record_wrong_lengths_fail_closed_on_lookup(
    record_length: int,
) -> None:
    result, exact = _complete_result()
    key = id(exact)
    original_record = _ownership_records()[key]
    reference, snapshot = original_record
    values = (reference, snapshot, object())[:record_length]
    malformed = tuple.__new__(resolution_v2._ResolutionIdentityRecordV2, values)
    assert type(malformed) is resolution_v2._ResolutionIdentityRecordV2
    _ownership_records()[key] = cast(Any, malformed)

    try:
        with pytest.raises(V2IntegrityError, match="ownership record is invalid"):
            _verifier()(result, exact)
    finally:
        _ownership_records()[key] = original_record


@pytest.mark.parametrize("record_length", (0, 1, 3))
def test_exact_identity_record_wrong_lengths_have_silent_gc_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    record_length: int,
) -> None:
    resolution = _resolution(candidates=(2, 3))
    key = id(resolution)
    original_record = _ownership_records()[key]
    reference, snapshot = original_record
    values = (reference, snapshot, object())[:record_length]
    malformed = tuple.__new__(resolution_v2._ResolutionIdentityRecordV2, values)
    assert type(malformed) is resolution_v2._ResolutionIdentityRecordV2
    _ownership_records()[key] = cast(Any, malformed)
    unraisable: list[object] = []
    monkeypatch.setattr(sys, "unraisablehook", unraisable.append)

    del resolution
    gc.collect()

    try:
        assert reference() is None
        assert unraisable == []
        if record_length == 0:
            assert _ownership_records()[key] is malformed
        else:
            assert key not in _ownership_records()
    finally:
        _ownership_records().pop(key, None)


@pytest.mark.parametrize("method_name", ("__eq__", "__hash__"))
@pytest.mark.parametrize(
    "consumer",
    (
        "require_owned",
        "require_owned_pure",
        "legacy_context",
        "bound_statistic",
        "bound_null",
        "public_calibrator",
    ),
)
def test_every_ownership_consumer_avoids_replaced_resolution_identity_methods(
    method_name: str,
    consumer: str,
) -> None:
    supplied = _pair()
    exact = _resolution()
    prepared = _prepare_calibration(supplied, exact)
    expected_result = _calibrator()(supplied, exact)
    replacement_calls = 0

    def bomb(*_args: object, **_kwargs: object) -> object:
        nonlocal replacement_calls
        replacement_calls += 1
        raise AssertionError(f"replacement PlanResolutionV2.{method_name} executed")

    def invoke() -> object:
        if consumer == "require_owned":
            return resolution_v2._require_resolver_owned_resolution_v2(exact)
        if consumer == "require_owned_pure":
            return resolution_v2._require_resolver_owned_resolution_v2_pure(exact)
        if consumer == "legacy_context":
            return resolution_v2._snapshot_result_verification_context_v2(exact)
        if consumer == "bound_statistic":
            return resolution_v2._snapshot_registered_bound_statistic_v2(
                exact,
                prepared.bound_statistic,
                expected_pair=supplied,
            )
        if consumer == "bound_null":
            return resolution_v2._snapshot_registered_bound_null_v2(
                exact,
                prepared.bound_null,
                expected_pair=supplied,
            )
        assert consumer == "public_calibrator"
        return _calibrator()(supplied, exact)

    with _temporarily_replaced_inherited_class_attribute(
        PlanResolutionV2,
        method_name,
        bomb,
    ):
        if consumer == "public_calibrator":
            assert invoke() == expected_result
        else:
            try:
                invoke()
            except V2IntegrityError:
                pass

    assert replacement_calls == 0


@pytest.mark.parametrize("method_name", ("__eq__", "__hash__"))
def test_resolve_under_resolution_identity_method_drift_never_executes_replacement(
    method_name: str,
) -> None:
    replacement_calls = 0

    def bomb(*_args: object, **_kwargs: object) -> object:
        nonlocal replacement_calls
        replacement_calls += 1
        raise AssertionError(f"replacement PlanResolutionV2.{method_name} executed")

    with _temporarily_replaced_inherited_class_attribute(
        PlanResolutionV2,
        method_name,
        bomb,
    ):
        try:
            exact = _resolution(candidates=(2, 4))
        except V2IntegrityError:
            exact = None
        if exact is not None:
            result = _calibrator()(_pair(), exact)
            _verifier()(result, exact)

    assert replacement_calls == 0


def test_resolution_identity_record_iteration_replacement_is_never_executed() -> None:
    result, exact = _complete_result()
    replacement_calls = 0

    def bomb(_record: object) -> Iterator[object]:
        nonlocal replacement_calls
        replacement_calls += 1
        raise AssertionError("replacement identity-record __iter__ executed")

    with _temporarily_replaced_inherited_class_attribute(
        resolution_v2._ResolutionIdentityRecordV2,
        "__iter__",
        bomb,
    ):
        try:
            _verifier()(result, exact)
        except V2IntegrityError:
            pass

    assert replacement_calls == 0


def test_resolution_identity_record_constructor_replacement_is_never_executed() -> None:
    replacement_calls = 0

    def bomb(*_args: object, **_kwargs: object) -> object:
        nonlocal replacement_calls
        replacement_calls += 1
        raise AssertionError("replacement identity-record __new__ executed")

    with _temporarily_replaced_inherited_class_attribute(
        resolution_v2._ResolutionIdentityRecordV2,
        "__new__",
        bomb,
    ):
        try:
            exact = _resolution(candidates=(2, 4))
        except V2IntegrityError:
            exact = None
        if exact is not None:
            result = _calibrator()(_pair(), exact)
            _verifier()(result, exact)

    assert replacement_calls == 0


def test_public_hook_cannot_use_temporary_identity_record_iteration_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected, exact = _complete_result()
    real_public_verifier = _verifier()
    replacement_calls = 0
    events: list[str] = []

    def bomb(_record: object) -> Iterator[object]:
        nonlocal replacement_calls
        replacement_calls += 1
        raise AssertionError("temporary identity-record __iter__ executed")

    def replace_use_restore(
        result: CalibrationResult,
        resolution: PlanResolutionV2,
        /,
    ) -> None:
        events.append("public")
        with _temporarily_replaced_inherited_class_attribute(
            resolution_v2._ResolutionIdentityRecordV2,
            "__iter__",
            bomb,
        ):
            try:
                real_public_verifier(result, resolution)
            except V2IntegrityError:
                pass
        events.append("restored")

    monkeypatch.setattr(calibration_v2, "verify_calibration_result", replace_use_restore)
    attacked = _calibrator()(_pair(), exact)

    assert attacked == expected
    assert events == ["public", "restored"]
    assert replacement_calls == 0


@pytest.mark.parametrize(
    "consumer",
    (
        "require_owned",
        "require_owned_pure",
        "legacy_context",
        "bound_statistic",
        "bound_null",
        "public_calibrator",
    ),
)
def test_ownership_consumers_do_not_dispatch_through_rebound_lookup_alias(
    consumer: str,
) -> None:
    supplied = _pair()
    exact = _resolution()
    prepared = _prepare_calibration(supplied, exact)
    expected_result = _calibrator()(supplied, exact)
    replacement_calls = 0

    def bomb(*_args: object, **_kwargs: object) -> object:
        nonlocal replacement_calls
        replacement_calls += 1
        raise AssertionError("rebound ownership lookup alias executed")

    def invoke() -> object:
        if consumer == "require_owned":
            return resolution_v2._require_resolver_owned_resolution_v2(exact)
        if consumer == "require_owned_pure":
            return resolution_v2._require_resolver_owned_resolution_v2_pure(exact)
        if consumer == "legacy_context":
            return resolution_v2._snapshot_result_verification_context_v2(exact)
        if consumer == "bound_statistic":
            return resolution_v2._snapshot_registered_bound_statistic_v2(
                exact,
                prepared.bound_statistic,
                expected_pair=supplied,
            )
        if consumer == "bound_null":
            return resolution_v2._snapshot_registered_bound_null_v2(
                exact,
                prepared.bound_null,
                expected_pair=supplied,
            )
        assert consumer == "public_calibrator"
        return _calibrator()(supplied, exact)

    with _temporarily_replaced_attribute(
        resolution_v2,
        "_LOOKUP_RESOLUTION_OWNERSHIP_V2",
        bomb,
    ):
        try:
            outcome = invoke()
        except V2IntegrityError:
            outcome = None
        if consumer == "public_calibrator" and outcome is not None:
            assert outcome == expected_result

    assert replacement_calls == 0


def test_resolver_does_not_dispatch_through_rebound_register_alias() -> None:
    replacement_calls = 0

    def bomb(*_args: object, **_kwargs: object) -> object:
        nonlocal replacement_calls
        replacement_calls += 1
        raise AssertionError("rebound ownership register alias executed")

    with _temporarily_replaced_attribute(
        resolution_v2,
        "_REGISTER_RESOLUTION_OWNERSHIP_V2",
        bomb,
    ):
        try:
            exact = _resolution(candidates=(3, 4))
        except V2IntegrityError:
            exact = None
        if exact is not None:
            result = _calibrator()(_pair(), exact)
            _verifier()(result, exact)

    assert replacement_calls == 0


@pytest.mark.parametrize(
    "alias_name",
    ("_LOOKUP_RESOLUTION_OWNERSHIP_V2", "_REGISTER_RESOLUTION_OWNERSHIP_V2"),
)
def test_public_hook_cannot_use_temporary_ownership_operation_alias_replacement(
    monkeypatch: pytest.MonkeyPatch,
    alias_name: str,
) -> None:
    expected, exact = _complete_result()
    real_public_verifier = _verifier()
    replacement_calls = 0
    events: list[str] = []

    def bomb(*_args: object, **_kwargs: object) -> object:
        nonlocal replacement_calls
        replacement_calls += 1
        raise AssertionError(f"temporary ownership operation executed: {alias_name}")

    def replace_use_restore(
        result: CalibrationResult,
        resolution: PlanResolutionV2,
        /,
    ) -> None:
        events.append("public")
        with _temporarily_replaced_attribute(resolution_v2, alias_name, bomb):
            try:
                if alias_name == "_LOOKUP_RESOLUTION_OWNERSHIP_V2":
                    real_public_verifier(result, resolution)
                else:
                    _resolution(candidates=(4, 5))
            except V2IntegrityError:
                pass
        events.append("restored")

    monkeypatch.setattr(calibration_v2, "verify_calibration_result", replace_use_restore)
    attacked = _calibrator()(_pair(), exact)

    assert attacked == expected
    assert events == ["public", "restored"]
    assert replacement_calls == 0


def test_ownership_consumers_and_resolver_do_not_load_operation_alias_globals() -> None:
    root_names = (
        "_require_resolver_owned_resolution_v2",
        "_require_resolver_owned_resolution_v2_pure",
        "_snapshot_result_verification_context_v2",
        "_snapshot_registered_bound_statistic_v2",
        "_snapshot_registered_bound_null_v2",
        "resolve_plan_v2",
    )
    forbidden_aliases = {
        "_RESOLVER_OWNED",
        "_LOOKUP_RESOLUTION_OWNERSHIP_V2",
        "_REGISTER_RESOLUTION_OWNERSHIP_V2",
    }
    roots = tuple((name, getattr(resolution_v2, name)) for name in root_names)
    assert all(callable(root) and not isinstance(root, type) for _, root in roots)

    pending: list[tuple[str, object, int]] = [
        (f"resolution_v2.{name}", root, 0) for name, root in roots
    ]
    seen: set[int] = set()
    findings: list[str] = []
    edge_count = 0

    def schedule(path: str, candidate: object, depth: int) -> None:
        nonlocal edge_count
        edge_count += 1
        assert edge_count <= 2048, "ownership operation graph exceeded edge bound"
        pending.append((path, candidate, depth))

    while pending:
        path, candidate, depth = pending.pop()
        assert depth <= 24, f"ownership operation graph exceeded depth bound at {path}"
        if id(candidate) in seen:
            continue
        seen.add(id(candidate))
        assert len(seen) <= 512, f"ownership operation graph exceeded node bound at {path}"

        if type(candidate) is tuple:
            for index in range(tuple.__len__(candidate)):
                schedule(f"{path}[{index}]", tuple.__getitem__(candidate, index), depth + 1)
            continue
        if type(candidate) is dict:
            for index, value in enumerate(dict.values(candidate)):
                schedule(f"{path}.value[{index}]", value, depth + 1)
            continue

        if type(candidate) is not FunctionType:
            module_name = str(getattr(type(candidate), "__module__", ""))
            if (
                callable(candidate)
                and not isinstance(candidate, type)
                and module_name.startswith("selcal.")
            ):
                call_descriptor = inspect.getattr_static(type(candidate), "__call__", None)
                if type(call_descriptor) is FunctionType:
                    schedule(f"{path}.__call__", call_descriptor, depth + 1)
            continue

        if not candidate.__module__.startswith("selcal."):
            continue

        codes = [candidate.__code__]
        visited_codes: set[int] = set()
        while codes:
            code = codes.pop()
            if id(code) in visited_codes:
                continue
            visited_codes.add(id(code))
            for instruction in dis.get_instructions(code):
                if instruction.opname != "LOAD_GLOBAL":
                    continue
                global_name = str(instruction.argval)
                if global_name in forbidden_aliases:
                    findings.append(f"{path}:{code.co_name}:{global_name}")
                resolved = candidate.__globals__.get(global_name)
                if type(resolved) is FunctionType and resolved.__module__.startswith("selcal."):
                    schedule(f"{path}.global[{global_name}]", resolved, depth + 1)
                elif (
                    callable(resolved)
                    and not isinstance(resolved, type)
                    and str(getattr(type(resolved), "__module__", "")).startswith("selcal.")
                ):
                    schedule(f"{path}.global[{global_name}]", resolved, depth + 1)
            codes.extend(constant for constant in code.co_consts if type(constant) is CodeType)

        if candidate.__closure__ is not None:
            for name, cell in zip(
                candidate.__code__.co_freevars,
                candidate.__closure__,
                strict=True,
            ):
                schedule(f"{path}.closure[{name}]", cell.cell_contents, depth + 1)
        if candidate.__defaults__ is not None:
            schedule(f"{path}.defaults", candidate.__defaults__, depth + 1)
        if candidate.__kwdefaults__ is not None:
            schedule(f"{path}.kwdefaults", candidate.__kwdefaults__, depth + 1)

    assert findings == []


@pytest.mark.parametrize(
    "alias_name",
    ("_LOOKUP_RESOLUTION_OWNERSHIP_V2", "_REGISTER_RESOLUTION_OWNERSHIP_V2"),
)
def test_bootstrap_ownership_operation_aliases_are_not_callable(alias_name: str) -> None:
    assert not callable(getattr(resolution_v2, alias_name, None))


def test_closure_reachable_ownership_lookup_rejects_unowned_inputs() -> None:
    consumer_names = (
        "_require_resolver_owned_resolution_v2",
        "_require_resolver_owned_resolution_v2_pure",
        "_snapshot_result_verification_context_v2",
        "_snapshot_registered_bound_statistic_v2",
        "_snapshot_registered_bound_null_v2",
    )
    lookups = tuple(
        inspect.getclosurevars(getattr(resolution_v2, name)).nonlocals["ownership_lookup"]
        for name in consumer_names
    )
    assert all(lookup is lookups[0] for lookup in lookups)
    ownership_lookup = cast(Callable[[object], object], lookups[0])
    exact = _resolution()
    unowned = PlanResolutionV2(
        plan=exact.plan,
        adapters=exact.adapters,
        seal=_V2_RESOLUTION_SEAL,
    )

    with pytest.raises(V2IntegrityError, match="must be an exact resolver-owned"):
        ownership_lookup(object())
    with pytest.raises(V2IntegrityError, match="not resolver-owned"):
        ownership_lookup(unowned)


def test_resolve_exposes_no_registration_callable_and_rejects_before_atomic_write() -> None:
    nonlocals = inspect.getclosurevars(resolve_plan_v2).nonlocals
    assert "ownership_register" not in nonlocals
    records_before = tuple(
        (key, id(record)) for key, record in _ownership_records().items()
    )

    with pytest.raises(TypeError, match="request must be an exact PlanRequestV2"):
        resolve_plan_v2(cast(Any, object()))

    assert (
        tuple((key, id(record)) for key, record in _ownership_records().items())
        == records_before
    )


def test_resolve_uses_the_definition_time_resolution_snapshot_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = _ownership_records()
    real_capture = resolution_v2._capture_resolution_snapshot
    replacement_calls = 0

    def coordinated_wrong_snapshot(exact: PlanResolutionV2) -> object:
        nonlocal replacement_calls
        replacement_calls += 1
        return real_capture(exact)._replace(plan_sha256="0" * 64)

    with monkeypatch.context() as context:
        context.setattr(
            resolution_v2,
            "_capture_resolution_snapshot",
            coordinated_wrong_snapshot,
        )
        resolved = _resolution(replicates=1)

    key = id(resolved)
    try:
        stored_snapshot = records[key].snapshot
        ownership_error: str | None = None
        owned: PlanResolutionV2 | None = None
        try:
            owned = resolution_v2._require_resolver_owned_resolution_v2(resolved)
        except V2IntegrityError as error:
            ownership_error = str(error)

        assert (
            replacement_calls,
            stored_snapshot.plan_sha256,
            ownership_error,
            owned is resolved,
        ) == (
            0,
            scientific_plan_v2_sha256(resolved.plan),
            None,
            True,
        )
    finally:
        records.pop(key, None)


def test_resolve_does_not_dispatch_to_a_runtime_snapshot_bomb_or_publish_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = _ownership_records()
    records_before = tuple((key, id(record)) for key, record in records.items())
    replacement_calls = 0
    resolved: PlanResolutionV2 | None = None
    escaped_runtime_error: str | None = None

    def runtime_snapshot_bomb(exact: PlanResolutionV2) -> object:
        del exact
        nonlocal replacement_calls
        replacement_calls += 1
        raise RuntimeError("runtime resolution snapshot replacement executed")

    with monkeypatch.context() as context:
        context.setattr(
            resolution_v2,
            "_capture_resolution_snapshot",
            runtime_snapshot_bomb,
        )
        try:
            resolved = _resolution(replicates=1)
        except RuntimeError as error:
            escaped_runtime_error = str(error)

    try:
        if escaped_runtime_error is not None:
            assert tuple((key, id(record)) for key, record in records.items()) == records_before
        assert (replacement_calls, escaped_runtime_error) == (0, None)
        assert resolved is not None
        assert resolution_v2._require_resolver_owned_resolution_v2(resolved) is resolved
    finally:
        if resolved is not None:
            records.pop(id(resolved), None)


def test_resolve_types_captured_snapshot_runtime_failure_without_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = _ownership_records()
    records_before = tuple((key, id(record)) for key, record in records.items())
    real_validation = resolution_v2._validate_resolution_fields
    validation_calls = 0
    caught: Exception | None = None

    def validation_runtime_failure(value: PlanResolutionV2) -> object:
        nonlocal validation_calls
        validation_calls += 1
        if validation_calls == 1:
            return real_validation(value)
        raise RuntimeError("captured snapshot validation failed")

    monkeypatch.setattr(
        resolution_v2,
        "_validate_resolution_fields",
        validation_runtime_failure,
    )
    try:
        _resolution(replicates=1)
    except Exception as error:
        caught = error

    assert validation_calls == 2
    assert tuple((key, id(record)) for key, record in records.items()) == records_before
    assert type(caught) is V2IntegrityError
    assert str(caught) == "resolution identity snapshot is invalid"


def test_exact_oracle_consumer_uses_the_definition_time_ownership_snapshot_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from selcal.exact_oracle_v0 import ExactOracleResultV0, exact_state_oracle_v0

    exact = _resolution(candidates=(1,), replicates=1)
    replacement_calls = 0
    escaped_runtime_error: str | None = None
    oracle_result: ExactOracleResultV0 | None = None

    def ownership_snapshot_bomb(value: PlanResolutionV2) -> object:
        del value
        nonlocal replacement_calls
        replacement_calls += 1
        raise RuntimeError("runtime ownership snapshot replacement executed")

    # The exact oracle is a real ownership consumer and does not spend replicate budget B.
    with monkeypatch.context() as context:
        context.setattr(
            resolution_v2,
            "_capture_resolution_snapshot_fail_closed",
            ownership_snapshot_bomb,
        )
        try:
            oracle_result = exact_state_oracle_v0(_pair(), exact)
        except RuntimeError as error:
            escaped_runtime_error = str(error)

    assert (replacement_calls, escaped_runtime_error) == (0, None)
    assert type(oracle_result) is ExactOracleResultV0


def test_resolve_closure_exposes_no_callable_that_can_promote_an_unowned_resolution() -> None:
    legitimate = _resolution()
    unowned = PlanResolutionV2(
        plan=legitimate.plan,
        adapters=legitimate.adapters,
        seal=_V2_RESOLUTION_SEAL,
    )
    key = id(unowned)
    assert key not in _ownership_records()
    callable_leaves: list[tuple[str, Callable[[object], object]]] = []
    table_mutators: list[str] = []
    pending: list[tuple[str, object, int]] = [("resolve_plan_v2", resolve_plan_v2, 0)]
    seen: set[int] = set()
    # Select by callable behavior and signature, not by a registration-like name.
    while pending:
        path, candidate, depth = pending.pop()
        assert depth <= 16, f"resolve closure graph exceeded depth bound at {path}"
        if id(candidate) in seen:
            continue
        seen.add(id(candidate))
        assert len(seen) <= 64, f"resolve closure graph exceeded node bound at {path}"
        bound_owner = getattr(candidate, "__self__", None)
        mutator_name = getattr(candidate, "__name__", None)
        if candidate is dict.__setitem__ or (
            type(bound_owner) is dict and mutator_name in {"__setitem__", "update", "setdefault"}
        ):
            table_mutators.append(path)
        if type(candidate) is not FunctionType:
            continue
        if not candidate.__module__.startswith("selcal."):
            continue
        if candidate is not resolve_plan_v2 and not inspect.isbuiltin(candidate):
            try:
                inspect.signature(candidate).bind(unowned)
            except TypeError:
                pass
            else:
                callable_leaves.append((path, cast(Callable[[object], object], candidate)))
        if candidate.__closure__ is not None:
            for name, cell in zip(
                candidate.__code__.co_freevars,
                candidate.__closure__,
                strict=True,
            ):
                pending.append((f"{path}.closure[{name}]", cell.cell_contents, depth + 1))

    assert 0 < len(callable_leaves) <= 32
    promoters: list[str] = []
    try:
        for path, leaf in callable_leaves:
            try:
                leaf(unowned)
            except Exception:
                pass
            try:
                resolution_v2._require_resolver_owned_resolution_v2(unowned)
            except V2IntegrityError:
                pass
            else:
                promoters.append(path)
            finally:
                _ownership_records().pop(key, None)
    finally:
        _ownership_records().pop(key, None)

    assert (promoters, table_mutators) == ([], [])


def test_bound_snapshot_entrypoints_have_exact_static_call_contracts(
    tmp_path: Path,
) -> None:
    probe = tmp_path / "bound_snapshot_keyword_probe.py"
    probe.write_text(
        """
from selcal.contracts import SeriesPair
from selcal.resolution_v2 import (
    PlanResolutionV2,
    _snapshot_registered_bound_null_v2,
    _snapshot_registered_bound_statistic_v2,
)


def probe_bound_snapshot_keywords(
    resolution: PlanResolutionV2,
    bound: object,
    pair: SeriesPair,
) -> None:
    _snapshot_registered_bound_statistic_v2(
        resolution,
        bound,
        expected_pair=pair,
    )
    _snapshot_registered_bound_null_v2(
        resolution,
        bound,
        expected_pair=pair,
    )
    _snapshot_registered_bound_statistic_v2(
        resolution,
        bound,
        expected_pair=pair,
        arbitrary_keyword=True,
    )
    _snapshot_registered_bound_null_v2(
        resolution,
        bound,
        expected_pair=pair,
        arbitrary_keyword=True,
    )
    _snapshot_registered_bound_statistic_v2(resolution, bound)
    _snapshot_registered_bound_null_v2(resolution, bound)
    _snapshot_registered_bound_statistic_v2(
        object(),
        bound,
        expected_pair=pair,
    )
    _snapshot_registered_bound_null_v2(
        object(),
        bound,
        expected_pair=pair,
    )
""".lstrip(),
        encoding="utf-8",
        newline="\n",
    )
    environment = os.environ.copy()
    environment["MYPYPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--strict",
            "--no-incremental",
            str(probe),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=20,
        check=False,
    )
    output = completed.stdout + completed.stderr

    assert completed.returncode != 0, output
    error_lines = [line for line in output.splitlines() if ": error:" in line]
    assert len(error_lines) == 6, output
    assert output.count('Unexpected keyword argument "arbitrary_keyword"') == 2
    assert output.count('Missing named argument "expected_pair"') == 2
    assert output.count('Argument 1 to "__call__"') == 2
    assert output.count('has incompatible type "object"') == 2


@pytest.mark.parametrize(
    ("drift", "message"),
    (
        ("missing_identity_record", "not resolver-owned"),
        ("stored_plan_digest", "plan drifted"),
        ("registry_identity", "registry identity drifted"),
        ("registration_association", "bound registration snapshot drifted"),
        ("sealed_leaf", "verification leaves drifted"),
        ("pure_snapshot", "pure identity snapshot drifted"),
    ),
)
def test_sealed_reader_rejects_identity_and_ownership_snapshot_drift(
    drift: str,
    message: str,
) -> None:
    result, exact = _complete_result()
    key = id(exact)
    original_record = _ownership_records()[key]
    _, original_snapshot = original_record

    if drift == "missing_identity_record":
        _ownership_records().pop(key)
    elif drift == "stored_plan_digest":
        replacement = original_snapshot._replace(
            plan_sha256="0" * 64,
        )
        _ownership_records()[key] = original_record._replace(
            snapshot=replacement,
        )
    else:
        if drift == "registry_identity":
            replacement = original_snapshot._replace(
                statistic_registry_identity=original_snapshot.statistic_registry_identity._replace(
                    factories=(),
                ),
            )
        elif drift == "registration_association":
            replacement = original_snapshot._replace(
                statistic_registration=original_snapshot.null_registration,
            )
        elif drift == "sealed_leaf":
            replacement = original_snapshot._replace(
                statistic_result_identity=object(),
            )
        else:
            assert drift == "pure_snapshot"
            replacement = original_snapshot._replace(
                statistic_parameters_bytes=b'{"foreign":true}',
            )
        _ownership_records()[key] = original_record._replace(
            snapshot=replacement,
        )

    try:
        with pytest.raises(V2IntegrityError, match=message):
            _verifier()(result, exact)
    finally:
        _ownership_records()[key] = original_record


@pytest.mark.parametrize("invalid", (object(), ()))
def test_public_verifier_rejects_wrong_exact_argument_types(invalid: object) -> None:
    result, exact = _complete_result()
    with pytest.raises(V2IntegrityError, match="exact canonical type"):
        if invalid == ():
            _verifier()(result, invalid)  # type: ignore[arg-type]
        else:
            _verifier()(invalid, exact)  # type: ignore[arg-type]


def test_frozen_plan_bytes_match_public_canonicalization_for_all_json_kinds() -> None:
    plan = ResolvedScientificPlanV2(
        candidates=(1, 2),
        statistic_name="synthetic_statistic_v2",
        statistic_params={
            "none": None,
            "boolean": True,
            "integer": 7,
            "float": 1.25,
            "mapping": {"nested": "value"},
            "sequence": [1, False, None],
        },
        selection_rule="max_absolute",
        null_name="synthetic_null_v2",
        null_params={"sequence": ["value", 2.5]},
        replicates=3,
        alpha=0.05,
        tie_tolerance=1e-12,
        root_seed=17,
        seal=_V2_RESOLUTION_SEAL,
    )

    assert _RESULT_VERIFIER_PLAN_CANONICAL_BYTES_V2(plan) == canonical_json_bytes(
        scientific_plan_v2_payload(plan)
    )


def test_frozen_plan_bytes_reject_wrong_exact_type() -> None:
    with pytest.raises(TypeError, match="exact ResolvedScientificPlanV2"):
        _RESULT_VERIFIER_PLAN_CANONICAL_BYTES_V2(object())


def test_frozen_plan_bytes_reject_descriptor_drift_without_executing_it() -> None:
    plan = _resolution().plan
    replacement_calls = 0

    def bomb(_instance: object) -> object:
        nonlocal replacement_calls
        replacement_calls += 1
        raise AssertionError("replacement ResolvedScientificPlanV2.alpha executed")

    with _temporarily_replaced_attribute(ResolvedScientificPlanV2, "alpha", property(bomb)):
        with pytest.raises(TypeError, match="slot descriptor drifted: alpha"):
            _RESULT_VERIFIER_PLAN_CANONICAL_BYTES_V2(plan)

    assert replacement_calls == 0


def test_result_slot_descriptor_drift_fails_before_replacement_execution() -> None:
    result, exact = _complete_result()
    original = inspect.getattr_static(CalibrationResult, "status")
    replacement_calls = 0

    def bomb(_instance: object) -> object:
        nonlocal replacement_calls
        replacement_calls += 1
        raise AssertionError("replacement CalibrationResult.status executed")

    with _temporarily_replaced_attribute(CalibrationResult, "status", property(bomb)):
        with pytest.raises(V2IntegrityError, match="slot descriptor drifted"):
            _verifier()(result, exact)

    assert inspect.getattr_static(CalibrationResult, "status") is original
    assert replacement_calls == 0


@pytest.mark.parametrize(
    ("drift", "message"),
    (
        ("ownership_type", "ownership record fields are invalid"),
        ("missing_identity_record", "not resolver-owned"),
        ("registry_factory_type", "registry identity drifted"),
    ),
)
def test_sealed_reader_normalizes_additional_ownership_failures(
    drift: str,
    message: str,
) -> None:
    result, exact = _complete_result()
    key = id(exact)
    original_record = _ownership_records()[key]
    _, original_snapshot = original_record

    if drift == "ownership_type":
        invalid_snapshot = cast(Any, ())
        _ownership_records()[key] = original_record._replace(
            snapshot=invalid_snapshot,
        )
    elif drift == "missing_identity_record":
        _ownership_records().pop(key)
    else:
        assert drift == "registry_factory_type"
        registry_identity = original_snapshot.statistic_registry_identity
        name, _factory = registry_identity.factories[0]
        replacement = original_snapshot._replace(
            statistic_registry_identity=registry_identity._replace(
                factories=((name, object()), *registry_identity.factories[1:]),
            ),
        )
        _ownership_records()[key] = original_record._replace(
            snapshot=replacement,
        )

    try:
        with pytest.raises(V2IntegrityError, match=message):
            _verifier()(result, exact)
    finally:
        _ownership_records()[key] = original_record


@pytest.mark.parametrize(
    ("leaf_kind", "adapter_type"),
    (
        ("statistic", LaggedPearsonAdapter),
        ("statistic", BinnedNetTEAdapter),
        ("null", CircularShiftNullV2),
        ("null", BlockShuffleNullV2),
    ),
)
def test_sealed_adapter_snapshot_rejects_wrong_exact_type(
    leaf_kind: str,
    adapter_type: type[object],
) -> None:
    statistic_snapshots, _, null_snapshots, _ = _sealed_science_leaf_maps()
    snapshot_maps = {
        "statistic": cast(Mapping[type[object], Callable[[object], object]], statistic_snapshots),
        "null": cast(Mapping[type[object], Callable[[object], object]], null_snapshots),
    }

    with pytest.raises(V2IntegrityError, match="wrong exact type"):
        snapshot_maps[leaf_kind][adapter_type](object())


@pytest.mark.parametrize(
    ("statistic_name", "null_name", "leaf_kind", "field_name"),
    (
        ("lagged_pearson_v1", "circular_shift_v2", "statistic", "_parameters"),
        ("equal_width_binned_nette_v1", "circular_shift_v2", "statistic", "bins"),
        ("lagged_pearson_v1", "circular_shift_v2", "null", "_null_parameter_sha256"),
        ("lagged_pearson_v1", "block_shuffle_v2", "null", "_parameters"),
    ),
)
def test_sealed_adapter_snapshot_rejects_slot_or_digest_descriptor_drift_without_execution(
    statistic_name: str,
    null_name: str,
    leaf_kind: str,
    field_name: str,
) -> None:
    resolution = _configured_resolution(statistic_name, null_name, "complete")
    adapter = (
        resolution.adapters.statistic
        if leaf_kind == "statistic"
        else resolution.adapters.null_model
    )
    statistic_snapshots, _, null_snapshots, _ = _sealed_science_leaf_maps()
    snapshot_maps = {
        "statistic": cast(Mapping[type[object], Callable[[object], object]], statistic_snapshots),
        "null": cast(Mapping[type[object], Callable[[object], object]], null_snapshots),
    }
    snapshot = snapshot_maps[leaf_kind][type(adapter)]
    replacement_calls = [0]
    replacement = _bomb_property(
        replacement_calls,
        f"replacement {type(adapter).__name__}.{field_name} executed",
    )

    with _temporarily_replaced_attribute(type(adapter), field_name, replacement):
        with pytest.raises(V2IntegrityError, match="descriptor drifted"):
            snapshot(adapter)

    assert replacement_calls == [0]


@pytest.mark.parametrize(
    ("factory", "parameters"),
    (
        (LaggedPearsonAdapter.from_parameters, {"unexpected": 1}),
        (BinnedNetTEAdapter.from_parameters, {"bins": 1}),
        (CircularShiftNullV2.from_parameters, {"min_shift": 0}),
        (BlockShuffleNullV2.from_parameters, {"block_length": 0}),
    ),
)
def test_public_adapter_constructors_make_invalid_parameter_snapshots_unreachable(
    factory: Callable[[object], object],
    parameters: dict[str, int],
) -> None:
    with pytest.raises((TypeError, ValueError, V2IntegrityError)):
        factory(parameters)


@pytest.mark.parametrize(
    ("statistic_name", "null_name"),
    (
        ("lagged_pearson_v1", "circular_shift_v2"),
        ("equal_width_binned_nette_v1", "block_shuffle_v2"),
    ),
)
def test_sealed_adapter_snapshots_preserve_constructor_owned_parameter_digests(
    statistic_name: str,
    null_name: str,
) -> None:
    resolution = _configured_resolution(statistic_name, null_name, "complete")
    statistic_snapshots, statistic_identities, null_snapshots, _ = _sealed_science_leaf_maps()
    exact_statistic_snapshots = cast(
        Mapping[type[object], Callable[[object], tuple[object, ...]]],
        statistic_snapshots,
    )
    exact_statistic_identities = cast(
        Mapping[type[object], Callable[[object], tuple[str, str]]],
        statistic_identities,
    )
    exact_null_snapshots = cast(
        Mapping[type[object], Callable[[object], tuple[object, ...]]],
        null_snapshots,
    )
    statistic = resolution.adapters.statistic
    null_adapter = resolution.adapters.null_model
    (
        statistic_parameter_bytes,
        _statistic_operational,
        _statistic_descriptors,
        backend_identity,
        preprocessing_identity,
    ) = exact_statistic_snapshots[type(statistic)](statistic)
    null_parameter_bytes, null_operational, _null_descriptors = exact_null_snapshots[
        type(null_adapter)
    ](null_adapter)
    identity_backend, identity_preprocessing = exact_statistic_identities[type(statistic)](
        statistic
    )
    _null_scalar, null_parameter_digest = cast(tuple[object, str], null_operational)

    assert statistic_parameter_bytes == canonical_json_bytes(resolution.plan.statistic_params)
    assert null_parameter_bytes == canonical_json_bytes(resolution.plan.null_params)
    assert null_parameter_digest == null_adapter.null_parameter_sha256
    assert (backend_identity, preprocessing_identity) == (
        identity_backend,
        identity_preprocessing,
    )


def _observed_length(
    result: CalibrationResult,
    resolution: PlanResolutionV2,
) -> int:
    assert result.observed_results
    return result.observed_results[0].support_n + max(resolution.plan.candidates)


def _sealed_token_verifier(
    resolution: PlanResolutionV2,
) -> Callable[[object, object, str, str, int], None]:
    (
        _plan,
        _plan_digest,
        _candidates,
        _rule,
        _replicates,
        _alpha,
        _tie_tolerance,
        _backend,
        _preprocessing,
        _null_adapter,
        token_verifier,
    ) = _sealed_context_values(resolution)
    assert callable(token_verifier)
    return cast(Callable[[object, object, str, str, int], None], token_verifier)


@pytest.mark.parametrize(
    ("statistic_name", "null_name"),
    (
        ("lagged_pearson_v1", "circular_shift_v2"),
        ("equal_width_binned_nette_v1", "block_shuffle_v2"),
    ),
)
def test_common_token_contract_rejects_nonpositive_observed_length(
    statistic_name: str,
    null_name: str,
) -> None:
    result, resolution = _terminal_result(statistic_name, null_name, "complete")
    token = result.replicates[0].transform_token
    verifier = _sealed_token_verifier(resolution)
    verifier_nonlocals = inspect.getclosurevars(verifier).nonlocals
    verify_common_token = cast(
        Callable[[tuple[object, ...], object, str, str, str, int], object],
        verifier_nonlocals["verify_common_token"],
    )
    adapter_snapshotter = cast(
        Callable[[object], tuple[object, ...]],
        verifier_nonlocals[
            "_circular_adapter_snapshot"
            if null_name == "circular_shift_v2"
            else "_block_adapter_snapshot"
        ],
    )

    with pytest.raises(V2IntegrityError, match="observed length"):
        verify_common_token(
            adapter_snapshotter(resolution.adapters.null_model),
            token,
            null_name,
            result.semantic_input_sha256,
            result.scientific_plan_sha256,
            0,
        )


def test_circular_token_verifier_rejects_constructor_valid_out_of_universe_shift() -> None:
    result, resolution = _terminal_result(
        "lagged_pearson_v1",
        "circular_shift_v2",
        "complete",
    )
    original = result.replicates[0].transform_token
    observed_length = _observed_length(result, resolution)
    token = NullTransformToken(
        schema=original.schema,
        null_name=original.null_name,
        null_parameter_sha256=original.null_parameter_sha256,
        semantic_input_sha256=original.semantic_input_sha256,
        scientific_plan_sha256=original.scientific_plan_sha256,
        bound_null_owner_sha256=original.bound_null_owner_sha256,
        is_identity=False,
        state=CircularShiftStateV2(
            schema="selcal.circular-shift-state.v2",
            shift=observed_length,
        ),
    )

    with pytest.raises(V2IntegrityError, match="state universe"):
        _sealed_token_verifier(resolution)(
            resolution.adapters.null_model,
            token,
            result.semantic_input_sha256,
            result.scientific_plan_sha256,
            observed_length,
        )


@pytest.mark.parametrize(
    ("observed_length", "message"),
    (
        (5, "state space"),
        (2, "state space"),
    ),
)
def test_block_token_verifier_rejects_nondivisible_or_single_block_universe(
    observed_length: int,
    message: str,
) -> None:
    result, resolution = _terminal_result(
        "lagged_pearson_v1",
        "block_shuffle_v2",
        "complete",
    )

    with pytest.raises(V2IntegrityError, match=message):
        _sealed_token_verifier(resolution)(
            resolution.adapters.null_model,
            result.replicates[0].transform_token,
            result.semantic_input_sha256,
            result.scientific_plan_sha256,
            observed_length,
        )


def test_block_token_verifier_rejects_constructor_valid_wrong_permutation_size() -> None:
    result, resolution = _terminal_result(
        "lagged_pearson_v1",
        "block_shuffle_v2",
        "complete",
    )
    original = result.replicates[0].transform_token
    observed_length = 8
    token = NullTransformToken(
        schema=original.schema,
        null_name=original.null_name,
        null_parameter_sha256=original.null_parameter_sha256,
        semantic_input_sha256=original.semantic_input_sha256,
        scientific_plan_sha256=original.scientific_plan_sha256,
        bound_null_owner_sha256=owner_digest(
            semantic_input_sha256=original.semantic_input_sha256,
            scientific_plan_sha256=original.scientific_plan_sha256,
            null_parameter_sha256=original.null_parameter_sha256,
            observed_length=observed_length,
        ),
        is_identity=True,
        state=BlockShuffleStateV2(
            schema="selcal.block-shuffle-state.v2",
            block_order=(0, 1, 2),
        ),
    )

    with pytest.raises(V2IntegrityError, match="block order"):
        _sealed_token_verifier(resolution)(
            resolution.adapters.null_model,
            token,
            result.semantic_input_sha256,
            result.scientific_plan_sha256,
            observed_length,
        )


@pytest.mark.parametrize(
    ("owner_name", "field_name"),
    (
        ("PlanResolutionV2", "adapters"),
        ("_AdapterRegistryIdentityV2", "registry"),
        ("_AdapterRegistrationV2", "factory"),
        ("_ResolutionOwnershipSnapshotV2", "plan_canonical_bytes"),
        ("_ResolutionOwnershipSnapshotV2", "statistic_result_identity"),
    ),
)
def test_sealed_reader_normalizes_registry_registration_leaf_and_ownership_descriptor_drift(
    owner_name: str,
    field_name: str,
) -> None:
    resolution = _resolution()
    owner = cast(type[object], getattr(resolution_v2, owner_name))
    replacement_calls = [0]
    replacement = _bomb_property(
        replacement_calls,
        f"replacement {owner_name}.{field_name} executed",
    )

    with _temporarily_replaced_attribute(owner, field_name, replacement):
        with pytest.raises(V2IntegrityError):
            _sealed_resolution_context_reader()(resolution)

    assert replacement_calls == [0]


def _typed_context_outcome(
    reader: Callable[[object], object],
    resolution: PlanResolutionV2,
) -> str:
    try:
        reader(resolution)
    except V2IntegrityError:
        return "V2IntegrityError"
    except BaseException as error:
        return f"unexpected:{type(error).__name__}"
    return "accept"


@contextmanager
def _ordinary_context_drift(
    resolution: PlanResolutionV2,
    drift: str,
) -> Iterator[None]:
    if drift == "plan":
        alternate = _resolution(
            replicates=resolution.plan.replicates + 1,
        )
        owner = PlanResolutionV2
        field_name = "plan"
        replacement = _DelegatingDescriptor(
            inspect.getattr_static(owner, field_name),
            returned=alternate.plan,
        )
    elif drift == "registration":
        owner = resolution_v2._AdapterRegistrationEntryV2
        field_name = "statistic_result_identity"
        replacement = _DelegatingDescriptor(inspect.getattr_static(owner, field_name))
    elif drift == "adapter_parameters":
        owner = type(resolution.adapters.statistic)
        field_name = "_parameters"
        replacement = _DelegatingDescriptor(inspect.getattr_static(owner, field_name))
    elif drift == "adapter_operational":
        if resolution.plan.statistic_name == "equal_width_binned_nette_v1":
            owner = type(resolution.adapters.statistic)
            field_name = "bins"
        else:
            owner = type(resolution.adapters.null_model)
            field_name = (
                "min_shift" if resolution.plan.null_name == "circular_shift_v2" else "block_length"
            )
        replacement = _DelegatingDescriptor(inspect.getattr_static(owner, field_name))
    elif drift == "adapter_descriptor":
        owner = type(resolution.adapters.statistic)
        field_name = "name"
        replacement = _DelegatingDescriptor(inspect.getattr_static(owner, field_name))
    else:
        raise AssertionError(f"unknown context drift: {drift}")

    with _temporarily_replaced_attribute(owner, field_name, replacement):
        yield


@pytest.mark.parametrize(
    "statistic_name",
    ("lagged_pearson_v1", "equal_width_binned_nette_v1"),
)
@pytest.mark.parametrize(
    "null_name",
    ("circular_shift_v2", "block_shuffle_v2"),
)
@pytest.mark.parametrize(
    "terminal",
    ("complete", "null_bind", "observed_statistic_scan"),
)
@pytest.mark.parametrize(
    "drift",
    (
        "plan",
        "registration",
        "adapter_parameters",
        "adapter_operational",
        "adapter_descriptor",
    ),
)
def test_sealed_context_reader_matches_existing_pure_ownership_validator(
    statistic_name: str,
    null_name: str,
    terminal: str,
    drift: str,
) -> None:
    result, exact = _terminal_result(statistic_name, null_name, terminal)
    expected_stage = {
        "complete": None,
        "null_bind": RunFailureStage.NULL_BIND,
        "observed_statistic_scan": RunFailureStage.OBSERVED_STATISTIC_SCAN,
    }[terminal]
    assert result.failure_stage is expected_stage
    sealed_reader = _sealed_resolution_context_reader()
    pure_reader = cast(
        Callable[[object], object],
        resolution_v2._snapshot_result_verification_context_v2,
    )

    with _ordinary_context_drift(exact, drift):
        pure_outcome = _typed_context_outcome(pure_reader, exact)
        sealed_outcome = _typed_context_outcome(sealed_reader, exact)

    assert pure_outcome in {"accept", "V2IntegrityError"}
    assert sealed_outcome in {"accept", "V2IntegrityError"}
    assert sealed_outcome == pure_outcome


def _reachable_science_leaf_cores() -> tuple[Callable[..., object], ...]:
    forbidden = {"plan_sha256", "resolution_context", "seed_digest_operation"}
    nonlocals = inspect.getclosurevars(_verifier()).nonlocals
    vulnerable: list[Callable[..., object]] = []
    for value in nonlocals.values():
        if not callable(value) or isinstance(value, type):
            continue
        module_name = getattr(value, "__module__", type(value).__module__)
        if not str(module_name).startswith("selcal."):
            continue
        try:
            parameters = set(inspect.signature(value).parameters)
        except (TypeError, ValueError):
            call_descriptor = inspect.getattr_static(type(value), "__call__", None)
            if type(call_descriptor) is not FunctionType:
                continue
            parameters = set(inspect.signature(call_descriptor).parameters)
            parameters.discard("self")
        if forbidden <= parameters:
            vulnerable.append(cast(Callable[..., object], value))
    return tuple(vulnerable)


def test_public_verifier_closure_exposes_no_science_leaf_injectable_core() -> None:
    assert _reachable_science_leaf_cores() == ()


def test_reachable_verifier_core_cannot_accept_replacement_context_directly() -> None:
    vulnerable = _reachable_science_leaf_cores()
    if not vulnerable:
        return

    result, exact = _complete_result()
    forged = _constructor_valid_foreign_backend_copy(result)
    verifier_leaves = inspect.getclosurevars(_verifier()).nonlocals
    original_context = resolution_v2._snapshot_result_verification_context_v2(exact)
    replacement_context = SimpleNamespace(
        plan=original_context.plan,
        plan_sha256=original_context.plan_sha256,
        backend_identity="constructor-valid-foreign-backend",
        preprocessing_identity=original_context.preprocessing_identity,
        null_adapter=original_context.null_adapter,
        verify_token=original_context.verify_token,
    )
    vulnerable[0](
        forged,
        exact,
        plan_sha256=verifier_leaves["plan_sha256"],
        resolution_context=lambda _resolution: replacement_context,
        seed_digest_operation=verifier_leaves["seed_digest"],
    )
    pytest.fail("reachable verifier core accepted a replacement scientific context")


def test_dynamic_public_hook_cross_module_verifier_replace_use_restore_rejects_forgery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected, exact = _complete_result()
    forged = _constructor_valid_but_resolution_inconsistent_copy(expected)
    original = contracts_v2._verify_result_vector_against_plan
    events: list[str] = []
    replacement_calls = 0

    def permissive_vector_verifier(*args: object, **kwargs: object) -> int:
        nonlocal replacement_calls
        del args, kwargs
        replacement_calls += 1
        return expected.observed_results[0].support_n

    def replace_use_restore(
        result: CalibrationResult,
        resolution: PlanResolutionV2,
        /,
    ) -> None:
        del result
        events.append("public")
        monkeypatch.setattr(
            contracts_v2,
            "_verify_result_vector_against_plan",
            permissive_vector_verifier,
        )
        try:
            try:
                _verifier()(forged, resolution)
            except V2IntegrityError:
                events.append("forgery_rejected")
            else:
                events.append("forgery_accepted")
        finally:
            monkeypatch.setattr(
                contracts_v2,
                "_verify_result_vector_against_plan",
                original,
            )

    monkeypatch.setattr(calibration_v2, "verify_calibration_result", replace_use_restore)
    attacked = _calibrator()(_pair(), exact)

    assert attacked == expected
    assert events == ["public", "forgery_rejected"]
    assert replacement_calls == 0


@pytest.mark.parametrize("method_name", ("__eq__", "__hash__"))
def test_public_hook_cannot_use_temporary_resolution_eq_or_hash_replacement(
    monkeypatch: pytest.MonkeyPatch,
    method_name: str,
) -> None:
    expected, exact = _complete_result()
    forged = _constructor_valid_but_resolution_inconsistent_copy(expected)
    real_public_verifier = _verifier()
    replacement_calls = 0
    events: list[str] = []

    def bomb(*_args: object, **_kwargs: object) -> object:
        nonlocal replacement_calls
        replacement_calls += 1
        raise AssertionError(f"replacement PlanResolutionV2.{method_name} executed")

    def replace_use_restore(
        result: CalibrationResult,
        resolution: PlanResolutionV2,
        /,
    ) -> None:
        events.append("public")
        with _temporarily_replaced_inherited_class_attribute(
            PlanResolutionV2,
            method_name,
            bomb,
        ):
            with pytest.raises(V2IntegrityError):
                real_public_verifier(forged, resolution)
            real_public_verifier(result, resolution)
        events.append("restored")

    monkeypatch.setattr(calibration_v2, "verify_calibration_result", replace_use_restore)
    attacked = _calibrator()(_pair(), exact)

    assert attacked == expected
    assert events == ["public", "restored"]
    assert replacement_calls == 0


def test_rebound_transform_validator_cannot_change_a_verified_complete_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supplied = _pair()
    exact = _resolution()
    expected = _calibrator()(supplied, exact)
    assert expected.status is RunStatus.COMPLETE

    def return_observed_pair(
        result: object,
        *,
        sampled_token: object,
        sampled_token_snapshot: tuple[object, ...],
        prepared: _PreparedCalibration,
    ) -> tuple[SeriesPair, tuple[str, ...]]:
        del result, sampled_token, sampled_token_snapshot
        return prepared.observed_pair, ()

    monkeypatch.setattr(calibration_v2, "_validate_transform_result", return_observed_pair)

    attacked = _calibrator()(supplied, exact)

    assert attacked.status is RunStatus.COMPLETE
    assert attacked == expected


@pytest.mark.parametrize(
    "helper_name",
    (
        "_validate_transform_result",
        "_validate_raw_replicate_vector",
        "_require_frozen_outcome_type",
    ),
)
def test_rebound_direct_executor_helpers_cannot_change_a_verified_complete_result(
    monkeypatch: pytest.MonkeyPatch,
    helper_name: str,
) -> None:
    supplied = _pair()
    exact = _resolution()
    expected = _calibrator()(supplied, exact)

    def bomb(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError(f"rebound direct helper executed: {helper_name}")

    monkeypatch.setattr(calibration_v2, helper_name, bomb)
    attacked = _calibrator()(supplied, exact)

    assert attacked == expected


@pytest.mark.parametrize(
    "helper_name",
    (
        "_snapshot_input_pair",
        "_snapshot_replicate_token",
        "_snapshot_statistic_result",
        "_result_vector_snapshot",
        "_validate_selector_output",
        "_snapshot_selection",
        "_require_preparer_owned_prepared",
    ),
)
def test_rebound_transitive_helper_fails_closed_before_attack_execution(
    monkeypatch: pytest.MonkeyPatch,
    helper_name: str,
) -> None:
    supplied = _pair()
    exact = _resolution()

    def bomb(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError(f"rebound transitive helper executed: {helper_name}")

    monkeypatch.setattr(calibration_v2, helper_name, bomb)
    with pytest.raises(
        V2IntegrityError,
        match=rf"behavior-critical module global drifted: {helper_name}",
    ):
        _calibrator()(supplied, exact)


def test_terminal_helper_has_no_injectable_attestor_and_freezes_the_real_verifier() -> None:
    helper = calibration_v2._verify_terminal_result_before_return
    signature = inspect.signature(helper)

    assert "_attest_result" not in signature.parameters
    assert helper.__defaults__ is None
    assert helper.__kwdefaults__ is None
    frozen = inspect.getclosurevars(helper).nonlocals
    assert frozen["real_verifier"] is contracts_v2.verify_calibration_result
    assert frozen["module_globals"] is vars(calibration_v2)
    assert callable(frozen["verify_behavior_globals"])
    assert "_attest_result" not in frozen


def test_terminal_helper_rejects_a_noncallable_public_verifier_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(calibration_v2, "verify_calibration_result", object())

    with pytest.raises(
        V2IntegrityError,
        match="public calibration result verifier is not callable",
    ):
        _calibrator()(_pair(), _resolution())


@pytest.mark.parametrize(
    ("expected_field", "match"),
    (
        ("expected_semantic_input_sha256", "semantic input identity drifted"),
        ("expected_scientific_plan_sha256", "scientific plan identity drifted"),
    ),
)
def test_terminal_helper_rejects_wrong_caller_bound_identity_before_public_hook(
    expected_field: str,
    match: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, exact = _complete_result()
    public_calls = 0

    def public_bomb(*args: object, **kwargs: object) -> None:
        nonlocal public_calls
        del args, kwargs
        public_calls += 1
        raise AssertionError("wrong caller-bound identity reached the public verifier")

    monkeypatch.setattr(calibration_v2, "verify_calibration_result", public_bomb)
    expected = {
        "expected_semantic_input_sha256": result.semantic_input_sha256,
        "expected_scientific_plan_sha256": result.scientific_plan_sha256,
    }
    expected[expected_field] = "f" * 64

    with pytest.raises(V2IntegrityError, match=match):
        calibration_v2._verify_terminal_result_before_return(
            result,
            exact,
            expected_semantic_input_sha256=expected[
                "expected_semantic_input_sha256"
            ],
            expected_scientific_plan_sha256=expected[
                "expected_scientific_plan_sha256"
            ],
        )
    assert public_calls == 0


@pytest.mark.parametrize(
    "path",
    ["complete", "null_bind", "observed_statistic_scan", "replicate_execution"],
)
def test_rebound_terminal_helper_cannot_substitute_a_foreign_result(
    monkeypatch: pytest.MonkeyPatch,
    path: str,
) -> None:
    supplied, exact, expected_status, expected_stage = _terminal_case(monkeypatch, path)
    real_verify = _verifier()
    verified: list[CalibrationResult] = []
    foreign = object()

    def spy(result: CalibrationResult, resolution: PlanResolutionV2, /) -> None:
        real_verify(result, resolution)
        verified.append(result)

    monkeypatch.setattr(calibration_v2, "verify_calibration_result", spy)
    monkeypatch.setattr(
        calibration_v2,
        "_verify_terminal_result_before_return",
        lambda *args, **kwargs: foreign,
    )
    result = _calibrator()(supplied, exact)

    assert result is not foreign
    assert result.status is expected_status
    assert result.failure_stage is expected_stage
    assert verified == [result]


@pytest.mark.parametrize(
    "path",
    ["complete", "null_bind", "observed_statistic_scan", "replicate_execution"],
)
def test_frozen_real_verifier_rejects_hook_mutation_with_terminal_global_rebound(
    monkeypatch: pytest.MonkeyPatch,
    path: str,
) -> None:
    supplied, exact, _, _ = _terminal_case(monkeypatch, path)
    real_verify = _verifier()
    verification_calls = 0

    def verify_then_mutate(
        result: CalibrationResult,
        resolution: PlanResolutionV2,
        /,
    ) -> None:
        nonlocal verification_calls
        verification_calls += 1
        real_verify(result, resolution)
        object.__setattr__(result, "p_value", 0.123)

    monkeypatch.setattr(calibration_v2, "verify_calibration_result", verify_then_mutate)
    monkeypatch.setattr(
        calibration_v2,
        "_verify_terminal_result_before_return",
        lambda *args, **kwargs: object(),
    )

    with pytest.raises(V2IntegrityError):
        _calibrator()(supplied, exact)
    assert verification_calls == 1


@pytest.mark.parametrize(
    "path",
    ["complete", "null_bind", "observed_statistic_scan", "replicate_execution"],
)
def test_plan_drift_injected_at_each_terminal_verification_produces_no_result(
    monkeypatch: pytest.MonkeyPatch,
    path: str,
) -> None:
    supplied, exact, _, _ = _terminal_case(monkeypatch, path)
    real_verify = _verifier()
    verification_calls = 0

    def drift_then_verify(
        result: CalibrationResult,
        resolution: PlanResolutionV2,
        /,
    ) -> None:
        nonlocal verification_calls
        verification_calls += 1
        object.__setattr__(resolution.plan, "root_seed", resolution.plan.root_seed + 1)
        real_verify(result, resolution)

    monkeypatch.setattr(calibration_v2, "verify_calibration_result", drift_then_verify)

    with pytest.raises(V2IntegrityError):
        _calibrator()(supplied, exact)
    assert verification_calls == 1


def test_public_calibrator_rejects_mutation_after_the_verifier_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact = _resolution()
    real_verify = _verifier()

    def verify_then_mutate(
        result: CalibrationResult,
        resolution: PlanResolutionV2,
        /,
    ) -> None:
        real_verify(result, resolution)
        object.__setattr__(result, "p_value", 0.123)

    monkeypatch.setattr(calibration_v2, "verify_calibration_result", verify_then_mutate)

    with pytest.raises(V2IntegrityError):
        _calibrator()(_pair(), exact)


def test_terminal_attestation_ignores_rebound_signature_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact = _resolution()
    real_verify = _verifier()
    verification_calls = 0

    def verify_then_mutate(
        result: CalibrationResult,
        resolution: PlanResolutionV2,
        /,
    ) -> None:
        nonlocal verification_calls
        verification_calls += 1
        real_verify(result, resolution)
        object.__setattr__(result, "p_value", 0.123)

    monkeypatch.setattr(
        calibration_v2,
        "_calibration_result_exact_signature",
        lambda _result, /: ("constant-forged-signature",),
        raising=False,
    )
    monkeypatch.setattr(calibration_v2, "verify_calibration_result", verify_then_mutate)

    with pytest.raises(V2IntegrityError):
        _calibrator()(_pair(), exact)
    assert verification_calls == 1


def test_public_calibrator_ignores_rebound_preparation_and_binds_entry_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supplied, exact, _, _ = _terminal_case(monkeypatch, "null_bind")
    terminal = _prepare_calibration(supplied, exact)
    assert type(terminal) is CalibrationResult
    object.__setattr__(terminal, "semantic_input_sha256", "f" * 64)
    rebound_calls = 0

    def forged_preparation(
        _pair: SeriesPair,
        _resolution: PlanResolutionV2,
        /,
    ) -> CalibrationResult:
        nonlocal rebound_calls
        rebound_calls += 1
        return terminal

    monkeypatch.setattr(
        calibration_v2,
        "_prepare_calibration",
        forged_preparation,
    )

    result = _calibrator()(supplied, exact)

    assert rebound_calls == 0
    assert result.status is RunStatus.NOT_EVALUABLE
    assert result.failure_stage is RunFailureStage.NULL_BIND
    assert result.semantic_input_sha256 != "f" * 64


def test_bare_verifier_does_not_claim_null_bind_semantic_input_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supplied, exact, _, _ = _terminal_case(monkeypatch, "null_bind")
    terminal = _prepare_calibration(supplied, exact)
    assert type(terminal) is CalibrationResult
    object.__setattr__(terminal, "semantic_input_sha256", "f" * 64)

    # The two-argument contract has no pair or token from which to authenticate
    # NULL_BIND provenance.  The public calibrator, tested above, closes this gate.
    _verifier()(terminal, exact)


def test_bare_verifier_complete_semantic_provenance_claim_ceiling_characterization() -> None:
    result, exact = _complete_result()
    _replace_result_semantic_identity(result, exact, "f" * 64)

    # CHARACTERIZATION, not a blind or security proof: every owner field is public and
    # structurally recomputable, so the two-argument verifier cannot recover the pair.
    _verifier()(result, exact)


@pytest.mark.parametrize(
    "path",
    ["complete", "null_bind", "observed_statistic_scan", "replicate_execution"],
)
def test_public_calibrator_rejects_semantic_drift_after_public_verification(
    monkeypatch: pytest.MonkeyPatch,
    path: str,
) -> None:
    supplied, exact, _, _ = _terminal_case(monkeypatch, path)
    real_verify = _verifier()

    def verify_then_replace_public_semantic_identity(
        result: CalibrationResult,
        resolution: PlanResolutionV2,
        /,
    ) -> None:
        real_verify(result, resolution)
        _replace_result_semantic_identity(result, resolution, "f" * 64)

    monkeypatch.setattr(
        calibration_v2,
        "verify_calibration_result",
        verify_then_replace_public_semantic_identity,
    )

    with pytest.raises(V2IntegrityError, match=r"semantic|input|drift"):
        _calibrator()(supplied, exact)


@pytest.mark.parametrize(
    "path",
    ["complete", "null_bind", "observed_statistic_scan", "replicate_execution"],
)
def test_public_calibrator_rejects_resolution_drift_after_public_verification(
    monkeypatch: pytest.MonkeyPatch,
    path: str,
) -> None:
    supplied, exact, _, _ = _terminal_case(monkeypatch, path)
    real_verify = _verifier()

    def verify_then_replace_resolution(
        result: CalibrationResult,
        resolution: PlanResolutionV2,
        /,
    ) -> None:
        real_verify(result, resolution)
        object.__setattr__(resolution.plan, "root_seed", resolution.plan.root_seed + 1)

    monkeypatch.setattr(
        calibration_v2,
        "verify_calibration_result",
        verify_then_replace_resolution,
    )

    with pytest.raises(V2IntegrityError, match=r"plan|resolution|drift"):
        _calibrator()(supplied, exact)


def _terminal_attestation_peak_bytes(
    result: CalibrationResult,
    resolution: PlanResolutionV2,
) -> int:
    gc.collect()
    tracemalloc.start()
    try:
        calibration_v2._verify_terminal_result_before_return(
            result,
            resolution,
            expected_semantic_input_sha256=result.semantic_input_sha256,
            expected_scientific_plan_sha256=result.scientific_plan_sha256,
        )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak


def test_terminal_attestation_does_not_materialize_a_second_result_tree() -> None:
    pair = SeriesPair(
        source=np.sin(np.arange(96, dtype=np.float64) / 7.0),
        target=np.cos(np.arange(96, dtype=np.float64) / 9.0),
    )
    small_resolution = _resolution(candidates=(1, 2, 3, 4), replicates=32)
    large_resolution = _resolution(candidates=tuple(range(1, 33)), replicates=128)
    small_result = _calibrator()(pair, small_resolution)
    large_result = _calibrator()(pair, large_resolution)

    small_peak = _terminal_attestation_peak_bytes(small_result, small_resolution)
    large_peak = _terminal_attestation_peak_bytes(large_result, large_resolution)

    # Runtime evidence with platform headroom: the result trees already exist before
    # tracing.  Terminal checking may retain O(C) scratch state, but not O(B*C) mirrors.
    assert large_peak < 8 * 1024 * 1024
    assert large_peak <= small_peak + 2 * 1024 * 1024
