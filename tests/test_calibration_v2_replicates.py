from __future__ import annotations

import ast
import dis
import inspect
import sys
from dataclasses import fields, replace
from pathlib import Path
from types import FrameType, SimpleNamespace

import numpy as np
import pytest

import selcal.calibration_v2 as calibration_v2
import selcal.contracts_v2 as contracts_v2
import selcal.nulls.executable_base as executable_base
import selcal.randomness as randomness
from selcal.calibration_v2 import (
    _execute_replicates,
    _prepare_calibration,
    _PreparedCalibration,
)
from selcal.contracts import (
    ReplicateStatus,
    RunStatus,
    SelectionResult,
    SelectionRule,
    SeriesPair,
    StatisticResult,
    Validity,
)
from selcal.contracts_v2 import (
    BlockShuffleStateV2,
    CircularShiftStateV2,
    NullTransformResult,
    NullTransformToken,
    PlanRequestV2,
    ReplicateFailureStage,
    V2IntegrityError,
)
from selcal.nulls.block_shuffle_v2 import _BoundBlockShuffleV2
from selcal.nulls.circular_shift_v2 import _BoundCircularShiftV2
from selcal.randomness import ReplicateRandomSource as RealReplicateRandomSource
from selcal.resolution_v2 import PlanResolutionV2, resolve_plan_v2
from selcal.selection_v2 import select_family_v2 as real_select_family_v2
from selcal.statistics.binned_nette import _BoundBinnedNetTEAdapter
from selcal.statistics.lagged_pearson import _BoundLaggedPearsonAdapter


def test_calibration_core_does_not_name_concrete_v2_nulls_or_states() -> None:
    source = Path(calibration_v2.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    string_literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and type(node.value) is str
    }
    referenced_names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    imported_names = {
        alias.name.rsplit(".", maxsplit=1)[-1]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import | ast.ImportFrom)
        for alias in node.names
    }

    assert string_literals.isdisjoint({"circular_shift_v2", "block_shuffle_v2"})
    assert (referenced_names | imported_names).isdisjoint(
        {"CircularShiftStateV2", "BlockShuffleStateV2"}
    )


def test_task8b_outcome_snapshot_type_has_no_ellipsis_or_arg_type_suppression() -> None:
    source = Path(calibration_v2.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    ellipsis_callables = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == "Callable"
        and isinstance(node.slice, ast.Tuple)
        and any(
            isinstance(item, ast.Constant) and item.value is Ellipsis for item in node.slice.elts
        )
    ]
    aggregate = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_task8b_terminal_history_snapshot"
    )
    aggregate_source = "\n".join(source.splitlines()[aggregate.lineno - 1 : aggregate.end_lineno])

    assert not ellipsis_callables
    assert "type: ignore[arg-type]" not in aggregate_source


def pair(source: list[float], target: list[float]) -> SeriesPair:
    return SeriesPair(
        source=np.asarray(source, dtype=np.float64),
        target=np.asarray(target, dtype=np.float64),
    )


def observed_pair() -> SeriesPair:
    return pair(
        [0, 1, 4, 2, 5, 3],
        [4, 1, 3, 0, 5, 2],
    )


def resolution(*, replicates: int = 5) -> PlanResolutionV2:
    return resolve_plan_v2(
        PlanRequestV2(
            candidates=(1, 2),
            statistic_name="lagged_pearson_v1",
            statistic_params={},
            selection_rule="max_upper",
            null_name="circular_shift_v2",
            null_params={"min_shift": 1},
            replicates=replicates,
            alpha=0.05,
            tie_tolerance=1e-12,
            root_seed=17,
        )
    )


def prepare(exact: PlanResolutionV2 | None = None) -> _PreparedCalibration:
    prepared = _prepare_calibration(observed_pair(), exact or resolution())
    assert type(prepared) is _PreparedCalibration
    return prepared


def test_public_calibration_calls_the_sealed_capsule_with_exact_replicate_contract() -> None:
    exact = resolution(replicates=3)
    expected = prepare(exact)
    capsule = randomness._RANDOM_CAPSULE_V2
    create_code = tuple.__getitem__(capsule, 0).__code__
    randbelow_code = tuple.__getitem__(capsule, 2).__code__
    create_calls: list[tuple[str, int, int]] = []
    created_streams: list[object] = []
    randbelow_calls: list[tuple[object, int]] = []

    def trace_random_capsule(frame: FrameType, event: str, arg: object) -> None:
        if frame.f_code is create_code:
            if event == "call":
                create_calls.append(
                    (
                        frame.f_locals["scientific_plan_sha256"],
                        frame.f_locals["replicate_id"],
                        frame.f_locals["planned_replicates"],
                    )
                )
            elif event == "return":
                created_streams.append(arg)
        elif event == "call" and frame.f_code is randbelow_code:
            randbelow_calls.append(
                (frame.f_locals["random_stream"], frame.f_locals["bound"])
            )

    previous_profile = sys.getprofile()
    sys.setprofile(trace_random_capsule)
    try:
        result = calibration_v2.calibrate_selected_family(observed_pair(), exact)
    finally:
        sys.setprofile(previous_profile)

    assert result.status is RunStatus.COMPLETE
    assert create_calls == [
        (expected.scientific_plan_sha256, replicate_id, exact.plan.replicates)
        for replicate_id in range(exact.plan.replicates)
    ]
    assert len(created_streams) == exact.plan.replicates
    assert len(randbelow_calls) == exact.plan.replicates
    assert all(
        supplied_stream is created_stream
        for (supplied_stream, _), created_stream in zip(
            randbelow_calls, created_streams, strict=True
        )
    )
    assert tuple(bound for _, bound in randbelow_calls) == (
        expected.bound_null.total_state_count,
    ) * exact.plan.replicates


def install_scripted_rng(
    monkeypatch: pytest.MonkeyPatch,
    indices: tuple[int, ...],
) -> None:
    """Drive downstream null-token/history branches, not the RNG call contract.

    This transitional Slice 2 seam deliberately substitutes only the null callback's
    private capability. It must not synthesize replicate metadata or be used as
    evidence about the sealed production capsule; Slice 3 will replace this seam.
    """

    original_circular_sample = _BoundCircularShiftV2.sample_token
    original_block_sample = _BoundBlockShuffleV2.sample_token
    next_index = 0

    class ScriptedIndexCapability:
        def __init__(self, index: int) -> None:
            self._index = index

        def randbelow(self, bound: int, /) -> int:
            assert 0 <= self._index < bound
            return self._index

    def next_capability() -> ScriptedIndexCapability:
        nonlocal next_index
        assert next_index < len(indices)
        capability = ScriptedIndexCapability(indices[next_index])
        next_index += 1
        return capability

    def scripted_circular_sample(
        self: _BoundCircularShiftV2,
        random: object,
        /,
    ) -> NullTransformToken:
        del random
        return original_circular_sample(self, next_capability())

    def scripted_block_sample(
        self: _BoundBlockShuffleV2,
        random: object,
        /,
    ) -> NullTransformToken:
        del random
        return original_block_sample(self, next_capability())

    # Slice 2 test seam: drive the real null callback/pure token kernel without
    # reopening the sealed production random capsule. Slice 3 will own this seam.
    monkeypatch.setattr(_BoundCircularShiftV2, "sample_token", scripted_circular_sample)
    monkeypatch.setattr(_BoundBlockShuffleV2, "sample_token", scripted_block_sample)


def test_private_random_capability_bridge_rejects_forged_shapes() -> None:
    with pytest.raises(V2IntegrityError, match="operation must be callable"):
        executable_base._build_random_index_capability_v2(object(), object())  # type: ignore[arg-type]

    stream = object()
    capability = executable_base._build_random_index_capability_v2(
        lambda supplied_stream, bound: bound - int(supplied_stream is stream),
        stream,
    )
    resolved = executable_base._resolve_static_callable(
        capability,
        method_name="randbelow",
        subject="test random capability",
    )
    assert resolved(4) == 3
    with pytest.raises(V2IntegrityError, match="sample_token"):
        executable_base._resolve_static_callable(
            capability,
            method_name="sample_token",
            subject="test random capability",
        )
    with pytest.raises(V2IntegrityError, match="randbelow must be callable"):
        executable_base._resolve_static_callable(
            (capability[0], object()),
            method_name="randbelow",
            subject="test random capability",
        )


def analytic_failure_vector(
    bound: _BoundLaggedPearsonAdapter,
) -> tuple[StatisticResult, ...]:
    return tuple(
        StatisticResult(
            candidate_id=candidate,
            estimate=None,
            selection_score=None,
            support_n=bound.observed_length - max(bound.candidates),
            validity=Validity.ANALYTIC_FAILURE,
            diagnostics=("scripted_replicate_analytic_failure",),
            backend_identity=bound.backend_identity,
            preprocessing_identity=bound.preprocessing_identity,
        )
        for candidate in bound.candidates
    )


def rule_discriminator_probe_vector(
    prepared: _PreparedCalibration,
) -> tuple[StatisticResult, ...]:
    return (
        StatisticResult(
            candidate_id=1,
            estimate=0.25,
            selection_score=None,
            support_n=4,
            validity=Validity.VALID,
            diagnostics=(),
            backend_identity=prepared.statistic_snapshot.backend_identity,
            preprocessing_identity=prepared.statistic_snapshot.preprocessing_identity,
        ),
        StatisticResult(
            candidate_id=2,
            estimate=-0.75,
            selection_score=None,
            support_n=4,
            validity=Validity.VALID,
            diagnostics=(),
            backend_identity=prepared.statistic_snapshot.backend_identity,
            preprocessing_identity=prepared.statistic_snapshot.preprocessing_identity,
        ),
    )


class OneIndex:
    def __init__(self, index: int) -> None:
        self.index = index

    def randbelow(self, bound: int, /) -> int:
        assert 0 <= self.index < bound
        return self.index


class TransparentSlotReplacement:
    def __init__(self, descriptor: object, owner: type[object]) -> None:
        self.descriptor = descriptor
        self.owner = owner

    def __get__(self, instance: object, owner: type[object] | None = None) -> object:
        del owner
        if instance is None:
            return self
        return self.descriptor.__get__(instance, self.owner)  # type: ignore[attr-defined]

    def __set__(self, instance: object, value: object) -> None:
        self.descriptor.__set__(instance, value)  # type: ignore[attr-defined]


@pytest.mark.parametrize("bound_type", [_BoundCircularShiftV2, _BoundBlockShuffleV2])
def test_snapshot_token_does_not_branch_on_exception_text(
    bound_type: type[object],
) -> None:
    source = inspect.getsource(bound_type.snapshot_token)  # type: ignore[attr-defined]

    assert "str(error)" not in source


def block_prepared() -> _PreparedCalibration:
    exact = resolve_plan_v2(
        PlanRequestV2(
            candidates=(1,),
            statistic_name="lagged_pearson_v1",
            statistic_params={},
            selection_rule="max_upper",
            null_name="block_shuffle_v2",
            null_params={"block_length": 2},
            replicates=2,
            alpha=0.05,
            tie_tolerance=1e-12,
            root_seed=17,
        )
    )
    prepared_value = _prepare_calibration(
        pair([0, 1, 2, 4], [2, 0, 4, 1]),
        exact,
    )
    assert type(prepared_value) is _PreparedCalibration
    return prepared_value


def test_execute_signature_has_prepared_and_required_keyword_only_observer() -> None:
    signature = inspect.signature(_execute_replicates)
    parameters = tuple(signature.parameters.values())

    assert len(parameters) == 2
    assert parameters[0].name == "prepared"
    assert parameters[0].kind is inspect.Parameter.POSITIONAL_ONLY
    assert parameters[0].default is inspect.Parameter.empty
    assert parameters[1].name == "observer"
    assert parameters[1].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters[1].default is inspect.Parameter.empty


def test_execute_rejects_extra_positional_and_keyword_arguments() -> None:
    prepared = prepare(resolution(replicates=1))

    with pytest.raises(TypeError):
        _execute_replicates(prepared, object())  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        _execute_replicates(prepared=prepared)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        _execute_replicates(prepared, injected=object())  # type: ignore[call-arg]


def test_execute_rejects_custom_snapshot_ops_that_would_bypass_final_integrity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact = resolution(replicates=1)
    install_scripted_rng(monkeypatch, (1,))
    prepared = prepare(exact)
    attack_fired = False

    def polluting_aggregate(
        supplied: _PreparedCalibration,
        history: tuple[object, ...],
    ) -> tuple[object, ...]:
        nonlocal attack_fired
        attack_fired = True
        outcome = history[0].outcome  # type: ignore[attr-defined]
        assert outcome.selection is not None
        object.__setattr__(outcome.selection, "decision_statistic", -999.0)
        object.__setattr__(supplied.resolution.plan, "root_seed", 999)
        return ()

    with pytest.raises(TypeError):
        _execute_replicates(prepared, polluting_aggregate)  # type: ignore[call-arg]
    assert not attack_fired


@pytest.mark.parametrize(
    "forbidden",
    ["CodeType", "co_consts", "code.replace", "_freeze_task8b_replicate_executor"],
)
def test_replicate_executor_uses_no_bytecode_binding_mechanism(forbidden: str) -> None:
    source = Path(calibration_v2.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    findings = {
        "CodeType": any(
            isinstance(node, ast.Name) and node.id == "CodeType" for node in ast.walk(tree)
        ),
        "co_consts": any(
            isinstance(node, ast.Attribute) and node.attr == "co_consts"
            for node in ast.walk(tree)
        ),
        "code.replace": any(
            isinstance(node, ast.Attribute)
            and node.attr == "replace"
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "__code__"
            for node in ast.walk(tree)
        ),
        "_freeze_task8b_replicate_executor": any(
            isinstance(node, ast.FunctionDef)
            and node.name == "_freeze_task8b_replicate_executor"
            for node in ast.walk(tree)
        ),
    }

    assert not findings[forbidden]


def test_execute_is_one_full_body_guarded_closure_without_injectable_defaults() -> None:
    source = Path(calibration_v2.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    definitions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_execute_replicates"
    ]

    assert len(definitions) == 1
    assert definitions[0] not in tree.body
    executor = calibration_v2._execute_replicates
    assert inspect.isfunction(executor)
    assert executor.__defaults__ is None
    assert executor.__kwdefaults__ is None
    frozen = inspect.getclosurevars(executor).nonlocals
    assert {
        "history_integrity_ops",
        "light_ops",
        "module_globals",
        "require_budget",
        "verify_behavior_globals",
    } <= set(frozen)
    assert frozen["history_integrity_ops"] is calibration_v2._TASK8B_HISTORY_INTEGRITY_OPS
    assert frozen["light_ops"] is calibration_v2._TASK8B_LIGHT_OPS
    assert frozen["require_budget"] is calibration_v2._require_prepared_in_memory_execution_budget
    assert frozen["module_globals"] is vars(calibration_v2)
    assert callable(frozen["verify_behavior_globals"])


def test_execute_closure_contains_no_separately_callable_raw_executor() -> None:
    frozen = inspect.getclosurevars(_execute_replicates).nonlocals

    assert "canonical_executor" not in frozen
    assert not {
        name
        for name, value in frozen.items()
        if inspect.isfunction(value) and value.__name__ == "_execute_replicates"
    }


@pytest.mark.parametrize(
    "callable_name",
    (
        "_execute_replicates",
        "calibrate_selected_family",
        "_verify_terminal_result_before_return",
    ),
)
def test_trusted_call_boundaries_have_no_behavior_critical_load_globals(
    callable_name: str,
) -> None:
    candidate = getattr(calibration_v2, callable_name)
    loaded_globals = {
        instruction.argval
        for instruction in dis.get_instructions(candidate)
        if instruction.opname == "LOAD_GLOBAL"
    }

    assert not loaded_globals


def test_only_executor_and_terminal_capture_the_module_globals_mapping() -> None:
    executor_closure = inspect.getclosurevars(calibration_v2._execute_replicates).nonlocals
    calibrator_closure = inspect.getclosurevars(
        calibration_v2.calibrate_selected_family
    ).nonlocals
    terminal_closure = inspect.getclosurevars(
        calibration_v2._verify_terminal_result_before_return
    ).nonlocals

    assert executor_closure["module_globals"] is vars(calibration_v2)
    assert "module_globals" not in calibrator_closure
    assert terminal_closure["module_globals"] is vars(calibration_v2)


def _reachable_calibration_callables(root: object) -> tuple[object, ...]:
    pending = [root]
    reachable: list[object] = []
    seen: set[int] = set()
    while pending:
        candidate = pending.pop()
        if id(candidate) in seen:
            continue
        seen.add(id(candidate))
        if inspect.isfunction(candidate):
            if candidate.__globals__ is not vars(calibration_v2):
                continue
            reachable.append(candidate)
            closure = inspect.getclosurevars(candidate)
            pending.extend(closure.nonlocals.values())
            pending.extend(closure.globals.values())
        elif isinstance(candidate, tuple):
            pending.extend(candidate)
    return tuple(reachable)


def _recursive_load_globals(candidate: object) -> set[str]:
    assert inspect.isfunction(candidate)
    code_type = type(candidate.__code__)
    pending = [candidate.__code__]
    seen: set[int] = set()
    loaded: set[str] = set()
    while pending:
        code = pending.pop()
        if id(code) in seen:
            continue
        seen.add(id(code))
        loaded.update(
            instruction.argval
            for instruction in dis.get_instructions(code)
            if instruction.opname == "LOAD_GLOBAL"
        )
        pending.extend(constant for constant in code.co_consts if type(constant) is code_type)
    return loaded


@pytest.mark.parametrize(
    "callable_name",
    (
        "_execute_replicates",
        "calibrate_selected_family",
        "_verify_terminal_result_before_return",
    ),
)
def test_recursive_trusted_callable_globals_are_covered_by_a_frozen_identity_manifest(
    callable_name: str,
) -> None:
    candidate = getattr(calibration_v2, callable_name)
    closure = inspect.getclosurevars(candidate).nonlocals
    guard = closure["verify_behavior_globals"]
    guard_closure = inspect.getclosurevars(guard).nonlocals
    expected_globals = dict(guard_closure["expected_globals"])
    referenced_globals = {
        global_name
        for reachable in _reachable_calibration_callables(candidate)
        for global_name in _recursive_load_globals(reachable)
    }

    assert referenced_globals <= expected_globals.keys()


def test_execute_captures_history_integrity_operations_before_any_callback() -> None:
    instructions = tuple(dis.get_instructions(_execute_replicates))
    budget_load = next(
        instruction
        for instruction in instructions
        if instruction.opname == "LOAD_DEREF" and instruction.argval == "require_budget"
    )
    capsule_load = next(
        instruction
        for instruction in instructions
        if instruction.opname == "LOAD_DEREF" and instruction.argval == "random_capsule"
    )
    bundle_loads = tuple(
        instruction
        for instruction in instructions
        if instruction.opname == "LOAD_DEREF"
        and instruction.argval == "history_integrity_ops"
    )
    create_stream_loads = tuple(
        instruction
        for instruction in instructions
        if instruction.opname == "LOAD_DEREF"
        and instruction.argval == "create_random_stream"
    )
    late_bound_helpers = {
        "_register_task8b_outcome_signature",
        "_task8b_read_exact_slots",
        "_capture_task8b_terminal_token_snapshots",
        "_task8b_terminal_history_snapshot",
        "_exact_snapshot_value_matches",
        "_task8b_outcome_signature_from_fields",
        "_task8b_token_signature",
        "_task8b_result_vector_signature",
        "_task8b_selection_signature",
        "_task8b_exact_value_snapshot",
        "_snapshot_replicate_token",
    }
    forbidden_loads = {
        instruction.argval
        for instruction in instructions
        if instruction.opname == "LOAD_GLOBAL" and instruction.argval in late_bound_helpers
    }

    first_deref = next(
        instruction for instruction in instructions if instruction.opname == "LOAD_DEREF"
    )
    assert first_deref.argval == "require_budget"
    assert len(bundle_loads) == 1
    assert budget_load.offset < capsule_load.offset < bundle_loads[0].offset
    assert create_stream_loads[-1].offset > bundle_loads[0].offset
    assert not forbidden_loads


def test_callback_cannot_replace_terminal_history_integrity_operations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_scripted_rng(monkeypatch, (1, 2))
    original_sample = _BoundCircularShiftV2.sample_token
    sample_calls = 0
    attack_fired = False

    def poison_history_and_replace_integrity_operations(
        self: _BoundCircularShiftV2,
        random: object,
        /,
    ) -> NullTransformToken:
        nonlocal attack_fired, sample_calls
        sample_calls += 1
        if sample_calls == 2:
            frame = inspect.currentframe()
            while frame is not None and frame.f_code.co_name != "_execute_replicates":
                frame = frame.f_back
            assert frame is not None
            history = frame.f_locals["history"]
            first_record = history[0]
            object.__setattr__(
                first_record.outcome.selection,
                "decision_statistic",
                -999.0,
            )

            def forged_terminal_snapshot(
                supplied_history: tuple[object, ...],
                _token_snapshots: tuple[object, ...],
            ) -> tuple[tuple[object, ...], tuple[object, ...]]:
                return (
                    tuple(record.outcome for record in supplied_history),  # type: ignore[attr-defined]
                    tuple(record.signature for record in supplied_history),  # type: ignore[attr-defined]
                )

            monkeypatch.setattr(
                calibration_v2,
                "_task8b_terminal_history_snapshot",
                forged_terminal_snapshot,
            )
            monkeypatch.setattr(
                calibration_v2,
                "_task8b_outcome_signature_from_fields",
                lambda *args, **kwargs: first_record.signature,
            )
            monkeypatch.setattr(
                calibration_v2,
                "_exact_snapshot_value_matches",
                lambda _current, _initial: True,
            )
            attack_fired = True
        return original_sample(self, random)  # type: ignore[arg-type]

    monkeypatch.setattr(
        _BoundCircularShiftV2,
        "sample_token",
        poison_history_and_replace_integrity_operations,
    )

    with pytest.raises(V2IntegrityError, match=r"history|outcome|drift"):
        returned = _execute_replicates(prepare(resolution(replicates=2)), observer=None)
        assert returned[0].selection is not None
        assert returned[0].selection.decision_statistic == -999.0
    assert attack_fired


def test_sample_callback_cannot_replace_the_frozen_token_snapshot_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_scripted_rng(monkeypatch, (1,))
    original_sample = _BoundCircularShiftV2.sample_token
    original_snapshot = calibration_v2._snapshot_replicate_token
    replacement_installed = False

    def mutate_shift_then_snapshot(
        token: object,
        prepared: _PreparedCalibration,
        include_identity: bool = False,
    ) -> tuple[object, ...]:
        caller = inspect.currentframe()
        assert caller is not None and caller.f_back is not None
        if caller.f_back.f_code.co_name == "_execute_replicates":
            object.__setattr__(token.state, "shift", 2)  # type: ignore[attr-defined]
        return original_snapshot(token, prepared, include_identity)

    def replace_snapshot_after_sampling(
        self: _BoundCircularShiftV2,
        random: object,
        /,
    ) -> NullTransformToken:
        nonlocal replacement_installed
        token = original_sample(self, random)  # type: ignore[arg-type]
        monkeypatch.setattr(
            calibration_v2,
            "_snapshot_replicate_token",
            mutate_shift_then_snapshot,
        )
        replacement_installed = True
        return token

    monkeypatch.setattr(
        _BoundCircularShiftV2,
        "sample_token",
        replace_snapshot_after_sampling,
    )

    with pytest.raises(
        V2IntegrityError,
        match="behavior-critical module global drifted: _snapshot_replicate_token",
    ):
        _execute_replicates(prepare(resolution(replicates=1)), observer=None)
    assert replacement_installed


@pytest.mark.parametrize(
    ("replicates", "analytical_failure_ids"),
    [
        (1, ()),
        (4, ()),
        (4, (1, 3)),
    ],
)
def test_execute_uses_fixed_full_and_light_checkpoint_schedule(
    replicates: int,
    analytical_failure_ids: tuple[int, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact = resolution(replicates=replicates)
    install_scripted_rng(monkeypatch, tuple(range(1, replicates + 1)))
    original_evaluate = _BoundLaggedPearsonAdapter.evaluate_all
    evaluate_call = 0

    def maybe_fail(
        self: _BoundLaggedPearsonAdapter,
        supplied: SeriesPair,
        /,
    ) -> tuple[StatisticResult, ...]:
        nonlocal evaluate_call
        evaluate_call += 1
        replicate_id = evaluate_call - 2
        if replicate_id in analytical_failure_ids:
            return analytic_failure_vector(self)
        return original_evaluate(self, supplied)

    monkeypatch.setattr(_BoundLaggedPearsonAdapter, "evaluate_all", maybe_fail)
    prepared = prepare(exact)
    tracked_codes = {
        calibration_v2._require_preparer_owned_prepared.__code__: "require",
        calibration_v2._verify_prepared_light_checkpoint.__code__: "light",
        calibration_v2._capture_prepared_light_fingerprint.__code__: "light_capture",
        calibration_v2._capture_prepared_ownership.__code__: "full_capture",
    }
    calls = {label: 0 for label in tracked_codes.values()}

    def count_calls(frame: object, event: str, _arg: object) -> None:
        if event != "call":
            return
        code = frame.f_code  # type: ignore[attr-defined]
        label = tracked_codes.get(code)
        if label is not None:
            calls[label] += 1

    previous_profile = sys.getprofile()
    sys.setprofile(count_calls)
    try:
        outcomes = _execute_replicates(prepared, observer=None)
    finally:
        sys.setprofile(previous_profile)

    assert len(outcomes) == replicates
    assert calls["require"] == 2
    assert calls["light"] == 3 * replicates + 2
    assert calls["light_capture"] == 1
    assert calls["full_capture"] == 2


def test_recursive_snapshot_matching_is_not_on_the_replicate_hot_path() -> None:
    """History checks may be O(B), but the generic Python walker must stay O(1)."""

    matcher_code = calibration_v2._exact_snapshot_value_matches.__code__

    def matcher_calls(replicates: int) -> int:
        calls = 0

        def count_calls(frame: object, event: str, _arg: object) -> None:
            nonlocal calls
            if event == "call" and frame.f_code is matcher_code:  # type: ignore[attr-defined]
                calls += 1

        prepared = prepare(resolution(replicates=replicates))
        previous_profile = sys.getprofile()
        sys.setprofile(count_calls)
        try:
            _execute_replicates(prepared, observer=None)
        finally:
            sys.setprofile(previous_profile)
        return calls

    assert matcher_calls(4) == matcher_calls(1)


def test_each_outcome_builds_one_registered_and_one_terminal_signature() -> None:
    signature_code = calibration_v2._task8b_outcome_signature_from_fields.__code__
    calls = 0

    def count_calls(frame: object, event: str, _arg: object) -> None:
        nonlocal calls
        if event == "call" and frame.f_code is signature_code:  # type: ignore[attr-defined]
            calls += 1

    replicates = 4
    prepared = prepare(resolution(replicates=replicates))
    previous_profile = sys.getprofile()
    sys.setprofile(count_calls)
    try:
        _execute_replicates(prepared, observer=None)
    finally:
        sys.setprofile(previous_profile)

    assert calls == 2 * replicates


@pytest.mark.parametrize(
    ("attack_stage", "attack_target"),
    [
        ("sample", "plan"),
        ("apply", "adapter"),
        ("evaluate", "bound_statistic"),
    ],
)
def test_light_checkpoint_preserves_live_component_drift_detection(
    attack_stage: str,
    attack_target: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact = resolution(replicates=2)
    install_scripted_rng(monkeypatch, (1, 2))
    original_sample = _BoundCircularShiftV2.sample_token
    original_apply = _BoundCircularShiftV2.apply
    original_evaluate = _BoundLaggedPearsonAdapter.evaluate_all
    prepared_holder: list[_PreparedCalibration] = []
    sample_calls = 0
    apply_calls = 0
    evaluate_calls = 0

    def mutate_target() -> None:
        prepared = prepared_holder[0]
        if attack_target == "plan":
            object.__setattr__(exact.plan, "root_seed", 18)
        elif attack_target == "adapter":
            object.__setattr__(exact.adapters, "statistic", object())
        elif attack_target == "bound_statistic":
            object.__setattr__(prepared.bound_statistic, "candidates", (1,))
        else:
            raise AssertionError(f"unknown attack target: {attack_target}")

    def attack_sample(
        self: _BoundCircularShiftV2,
        random: object,
        /,
    ) -> object:
        nonlocal sample_calls
        sample_calls += 1
        result = original_sample(self, random)  # type: ignore[arg-type]
        if attack_stage == "sample":
            mutate_target()
        return result

    def attack_apply(
        self: _BoundCircularShiftV2,
        token: object,
        /,
    ) -> object:
        nonlocal apply_calls
        apply_calls += 1
        result = original_apply(self, token)  # type: ignore[arg-type]
        if attack_stage == "apply":
            mutate_target()
        return result

    def attack_evaluate(
        self: _BoundLaggedPearsonAdapter,
        supplied: SeriesPair,
        /,
    ) -> object:
        nonlocal evaluate_calls
        evaluate_calls += 1
        result = original_evaluate(self, supplied)
        if attack_stage == "evaluate" and evaluate_calls == 2:
            mutate_target()
        return result

    monkeypatch.setattr(_BoundCircularShiftV2, "sample_token", attack_sample)
    monkeypatch.setattr(_BoundCircularShiftV2, "apply", attack_apply)
    monkeypatch.setattr(_BoundLaggedPearsonAdapter, "evaluate_all", attack_evaluate)
    prepared = prepare(exact)
    prepared_holder.append(prepared)

    with pytest.raises(V2IntegrityError, match=r"plan|resolution|adapter|bound|snapshot|drift"):
        _execute_replicates(prepared, observer=None)

    assert sample_calls == 1
    if attack_stage == "sample":
        assert apply_calls == 0
    elif attack_stage == "apply":
        assert evaluate_calls == 1


def test_post_scan_light_checkpoint_precedes_central_reselection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact = resolution(replicates=1)
    install_scripted_rng(monkeypatch, (2,))
    original_evaluate = _BoundLaggedPearsonAdapter.evaluate_all
    original_post_init = calibration_v2.ReplicateOutcome.__post_init__
    evaluate_calls = 0
    outcome_calls = 0

    def flip_rule_after_scan(
        self: _BoundLaggedPearsonAdapter,
        supplied: SeriesPair,
        /,
    ) -> tuple[StatisticResult, ...]:
        nonlocal evaluate_calls
        evaluate_calls += 1
        results = original_evaluate(self, supplied)
        if evaluate_calls == 2:
            object.__setattr__(exact.plan, "selection_rule", SelectionRule.MAX_ABSOLUTE)
        return results

    def restore_rule_from_outcome(self: object) -> None:
        nonlocal outcome_calls
        outcome_calls += 1
        original_post_init(self)  # type: ignore[arg-type]
        object.__setattr__(exact.plan, "selection_rule", SelectionRule.MAX_UPPER)

    monkeypatch.setattr(_BoundLaggedPearsonAdapter, "evaluate_all", flip_rule_after_scan)
    prepared = prepare(exact)
    monkeypatch.setattr(
        calibration_v2.ReplicateOutcome,
        "__post_init__",
        restore_rule_from_outcome,
    )

    with pytest.raises(V2IntegrityError, match=r"light|plan|fingerprint|drift"):
        _execute_replicates(prepared, observer=None)

    assert evaluate_calls == 2
    assert outcome_calls == 0


def test_replicate_selection_is_validated_against_the_entry_contract() -> None:
    exact = resolution(replicates=1)
    prepared = prepare(exact)
    raw_results = rule_discriminator_probe_vector(prepared)
    raw_snapshot = calibration_v2._result_vector_snapshot(
        raw_results,
        subject="direct frozen-rule probe",
    )
    wrong_selection, wrong_scored = real_select_family_v2(
        raw_results,
        exact.plan.candidates,
        SelectionRule.MAX_ABSOLUTE,
        exact.plan.tie_tolerance,
    )
    assert wrong_selection.selected_candidate == 2

    with pytest.raises(V2IntegrityError, match=r"selection|scored"):
        calibration_v2._validate_selector_output(
            raw_results,
            raw_snapshot,
            wrong_selection,
            wrong_scored,
            exact.plan.candidates,
            True,
            exact.plan.tie_tolerance,
        )


def test_replicate_validator_does_not_reinterpret_a_frozen_rule_through_live_enum() -> None:
    exact = resolution(replicates=1)
    prepared = prepare(exact)
    original_max_upper = SelectionRule.MAX_UPPER
    max_absolute = SelectionRule.MAX_ABSOLUTE
    raw_results = rule_discriminator_probe_vector(prepared)
    raw_snapshot = calibration_v2._result_vector_snapshot(
        raw_results,
        subject="direct live-enum probe",
    )
    wrong_selection, wrong_scored = real_select_family_v2(
        raw_results,
        exact.plan.candidates,
        max_absolute,
        exact.plan.tie_tolerance,
    )
    assert wrong_selection.selected_candidate == 2

    type.__setattr__(SelectionRule, "MAX_UPPER", max_absolute)
    try:
        with pytest.raises(V2IntegrityError, match=r"selection|scored"):
            calibration_v2._validate_selector_output(
                raw_results,
                raw_snapshot,
                wrong_selection,
                wrong_scored,
                exact.plan.candidates,
                True,
                exact.plan.tie_tolerance,
            )
    finally:
        type.__setattr__(SelectionRule, "MAX_UPPER", original_max_upper)


def test_entry_rule_classification_ignores_a_live_enum_value_descriptor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact = resolution(replicates=1)
    install_scripted_rng(monkeypatch, (2,))
    prepared = prepare(exact)
    original_max_upper = SelectionRule.MAX_UPPER
    max_absolute = SelectionRule.MAX_ABSOLUTE
    original_value_descriptor = inspect.getattr_static(SelectionRule, "value")
    original_post_init = calibration_v2.ReplicateOutcome.__post_init__

    def targeted_value(self: SelectionRule) -> object:
        caller = inspect.currentframe()
        assert caller is not None and caller.f_back is not None
        if caller.f_back.f_code.co_name == "_execute_replicates":
            type.__setattr__(SelectionRule, "MAX_UPPER", max_absolute)
            return "max_absolute"
        return original_value_descriptor.__get__(self, SelectionRule)  # type: ignore[attr-defined]

    def restore_enum_from_outcome(self: object) -> None:
        original_post_init(self)  # type: ignore[arg-type]
        type.__setattr__(SelectionRule, "MAX_UPPER", original_max_upper)
        if "value" in SelectionRule.__dict__:
            type.__delattr__(SelectionRule, "value")

    type.__setattr__(SelectionRule, "value", property(targeted_value))
    monkeypatch.setattr(
        calibration_v2.ReplicateOutcome,
        "__post_init__",
        restore_enum_from_outcome,
    )
    try:
        outcomes = _execute_replicates(prepared, observer=None)
    finally:
        type.__setattr__(SelectionRule, "MAX_UPPER", original_max_upper)
        if "value" in SelectionRule.__dict__:
            type.__delattr__(SelectionRule, "value")

    assert outcomes[0].selection is not None
    assert outcomes[0].selection.selected_candidate == 1


def test_replaced_selector_alias_is_rejected_before_replicate_callbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact = resolution(replicates=1)
    install_scripted_rng(monkeypatch, (2,))
    prepared = prepare(exact)
    original_post_init = calibration_v2.ReplicateOutcome.__post_init__
    selector_calls = 0
    outcome_calls = 0

    def drift_after_selection(*args: object, **kwargs: object) -> object:
        nonlocal selector_calls
        selector_calls += 1
        result = real_select_family_v2(*args, **kwargs)  # type: ignore[arg-type]
        object.__setattr__(exact.plan, "root_seed", 18)
        return result

    def restore_plan_from_outcome(self: object) -> None:
        nonlocal outcome_calls
        outcome_calls += 1
        original_post_init(self)  # type: ignore[arg-type]
        object.__setattr__(exact.plan, "root_seed", 17)

    monkeypatch.setattr(calibration_v2, "select_family_v2", drift_after_selection)
    monkeypatch.setattr(
        calibration_v2.ReplicateOutcome,
        "__post_init__",
        restore_plan_from_outcome,
    )

    with pytest.raises(V2IntegrityError, match=r"selector|plan|resolution|drift"):
        _execute_replicates(prepared, observer=None)

    assert selector_calls == 0
    assert outcome_calls == 0
    assert exact.plan.root_seed == 17


@pytest.mark.parametrize("attack_stage", ["sample", "apply"])
def test_frozen_light_gate_cannot_be_replaced_to_hide_transient_plan_drift(
    attack_stage: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact = resolution(replicates=1)
    install_scripted_rng(monkeypatch, (2,))
    original_sample = _BoundCircularShiftV2.sample_token
    original_apply = _BoundCircularShiftV2.apply
    original_evaluate = _BoundLaggedPearsonAdapter.evaluate_all
    evaluate_calls = 0

    def attack_sample(
        self: _BoundCircularShiftV2,
        random: object,
        /,
    ) -> object:
        token = original_sample(self, random)  # type: ignore[arg-type]
        if attack_stage == "sample":
            object.__setattr__(exact.plan, "root_seed", 18)
        return token

    def attack_apply(
        self: _BoundCircularShiftV2,
        token: object,
        /,
    ) -> object:
        if attack_stage == "sample":
            object.__setattr__(exact.plan, "root_seed", 17)
        result = original_apply(self, token)  # type: ignore[arg-type]
        if attack_stage == "apply":
            object.__setattr__(exact.plan, "root_seed", 18)
        return result

    def restore_before_evaluate(
        self: _BoundLaggedPearsonAdapter,
        supplied: SeriesPair,
        /,
    ) -> tuple[StatisticResult, ...]:
        nonlocal evaluate_calls
        evaluate_calls += 1
        if attack_stage == "apply" and evaluate_calls == 2:
            object.__setattr__(exact.plan, "root_seed", 17)
        return original_evaluate(self, supplied)

    monkeypatch.setattr(_BoundCircularShiftV2, "sample_token", attack_sample)
    monkeypatch.setattr(_BoundCircularShiftV2, "apply", attack_apply)
    monkeypatch.setattr(
        _BoundLaggedPearsonAdapter,
        "evaluate_all",
        restore_before_evaluate,
    )
    prepared = prepare(exact)
    monkeypatch.setattr(
        calibration_v2,
        "_verify_prepared_light_checkpoint",
        lambda *args: None,
    )

    with pytest.raises(V2IntegrityError, match=r"light|plan|fingerprint|drift"):
        _execute_replicates(prepared, observer=None)


def test_light_fingerprint_cannot_be_mutated_from_the_executor_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact = resolution(replicates=1)
    install_scripted_rng(monkeypatch, (2,))
    original_sample = _BoundCircularShiftV2.sample_token
    original_apply = _BoundCircularShiftV2.apply
    prepared_holder: list[_PreparedCalibration] = []
    fingerprint_holder: list[object] = []

    def synchronize_fingerprint() -> None:
        current = calibration_v2._capture_prepared_light_fingerprint(prepared_holder[0])
        replacement = getattr(current, "signature", current)
        try:
            object.__setattr__(fingerprint_holder[0], "signature", replacement)
        except (AttributeError, TypeError):
            pass

    def corrupt_plan_and_synchronize_fingerprint(
        self: _BoundCircularShiftV2,
        random: object,
        /,
    ) -> object:
        token = original_sample(self, random)  # type: ignore[arg-type]
        frame = inspect.currentframe()
        while frame is not None and frame.f_code.co_name != "_execute_replicates":
            frame = frame.f_back
        assert frame is not None
        fingerprint_holder.append(frame.f_locals["light_fingerprint"])
        object.__setattr__(exact.plan, "root_seed", 18)
        synchronize_fingerprint()
        return token

    def restore_plan_and_synchronize_fingerprint(
        self: _BoundCircularShiftV2,
        token: object,
        /,
    ) -> object:
        object.__setattr__(exact.plan, "root_seed", 17)
        synchronize_fingerprint()
        return original_apply(self, token)  # type: ignore[arg-type]

    monkeypatch.setattr(
        _BoundCircularShiftV2,
        "sample_token",
        corrupt_plan_and_synchronize_fingerprint,
    )
    monkeypatch.setattr(
        _BoundCircularShiftV2,
        "apply",
        restore_plan_and_synchronize_fingerprint,
    )
    prepared = prepare(exact)
    prepared_holder.append(prepared)

    with pytest.raises(V2IntegrityError, match=r"light|plan|fingerprint|drift"):
        _execute_replicates(prepared, observer=None)


def test_light_fingerprint_type_alias_cannot_supply_a_live_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact = resolution(replicates=1)
    install_scripted_rng(monkeypatch, (2,))
    original_sample = _BoundCircularShiftV2.sample_token
    original_apply = _BoundCircularShiftV2.apply
    prepared_holder: list[_PreparedCalibration] = []
    live_signature = calibration_v2._prepared_light_signature

    class ForgedFingerprint:
        def __init__(self, *, signature: object) -> None:
            del signature

        @property
        def signature(self) -> object:
            return live_signature(prepared_holder[0])

    def drift_sample(
        self: _BoundCircularShiftV2,
        random: object,
        /,
    ) -> object:
        token = original_sample(self, random)  # type: ignore[arg-type]
        object.__setattr__(exact.plan, "root_seed", 18)
        return token

    def restore_apply(
        self: _BoundCircularShiftV2,
        token: object,
        /,
    ) -> object:
        object.__setattr__(exact.plan, "root_seed", 17)
        return original_apply(self, token)  # type: ignore[arg-type]

    monkeypatch.setattr(_BoundCircularShiftV2, "sample_token", drift_sample)
    monkeypatch.setattr(_BoundCircularShiftV2, "apply", restore_apply)
    prepared = prepare(exact)
    prepared_holder.append(prepared)
    monkeypatch.setattr(
        calibration_v2,
        "_PreparedLightFingerprint",
        ForgedFingerprint,
    )

    with pytest.raises(V2IntegrityError, match=r"light|plan|fingerprint|drift"):
        _execute_replicates(prepared, observer=None)


def test_execute_runs_exact_b_full_reselection_replicates_with_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact = resolution(replicates=7)
    # Six circular states; B exceeds the state count, and 0/1 are repeated.
    install_scripted_rng(monkeypatch, (0, 1, 1, 2, 0, 3, 1))
    evaluate_calls: list[bytes] = []
    selection_calls = 0
    applied_tokens: list[object] = []
    original_evaluate = _BoundLaggedPearsonAdapter.evaluate_all
    original_apply = _BoundCircularShiftV2.apply

    def spy_evaluate(
        self: _BoundLaggedPearsonAdapter,
        supplied: SeriesPair,
        /,
    ) -> object:
        evaluate_calls.append(supplied.source.tobytes(order="C"))
        return original_evaluate(self, supplied)

    def count_selector(frame: object, event: str, _arg: object) -> None:
        nonlocal selection_calls
        if (
            event == "call" and frame.f_code is real_select_family_v2.__code__  # type: ignore[attr-defined]
        ):
            selection_calls += 1

    def spy_apply(
        self: _BoundCircularShiftV2,
        token: object,
        /,
    ) -> object:
        applied_tokens.append(token)
        return original_apply(self, token)  # type: ignore[arg-type]

    monkeypatch.setattr(_BoundLaggedPearsonAdapter, "evaluate_all", spy_evaluate)
    monkeypatch.setattr(_BoundCircularShiftV2, "apply", spy_apply)
    previous_profile = sys.getprofile()
    sys.setprofile(count_selector)
    try:
        prepared = prepare(exact)
        outcomes = _execute_replicates(prepared, observer=None)
    finally:
        sys.setprofile(previous_profile)

    assert type(outcomes) is tuple
    assert len(outcomes) == exact.plan.replicates == 7
    assert tuple(outcome.replicate_id for outcome in outcomes) == tuple(range(7))
    assert all(outcome.status is ReplicateStatus.COMPLETE for outcome in outcomes)
    assert all(outcome.failure_stage is None for outcome in outcomes)
    assert tuple(outcome.seed_digest_sha256 for outcome in outcomes) == tuple(
        RealReplicateRandomSource(
            prepared.scientific_plan_sha256,
            replicate_id,
            7,
        ).seed_digest_sha256
        for replicate_id in range(7)
    )
    assert len(set(outcome.seed_digest_sha256 for outcome in outcomes)) == 7
    assert tuple(outcome.transform_token.state.shift for outcome in outcomes) == (
        0,
        1,
        1,
        2,
        0,
        3,
        1,
    )
    assert outcomes[0].transform_token.is_identity
    assert outcomes[4].transform_token.is_identity
    assert outcomes[1].transform_token == outcomes[2].transform_token
    assert outcomes[2].transform_token == outcomes[6].transform_token
    assert all(
        outcome.transform_token is applied_token
        for outcome, applied_token in zip(outcomes, applied_tokens, strict=True)
    )
    assert len(evaluate_calls) == 1 + 7
    assert len(set(evaluate_calls[1:])) >= 4
    assert selection_calls == 1 + 7
    for outcome in outcomes:
        assert tuple(result.candidate_id for result in outcome.statistic_results) == (1, 2)
        assert all(result.validity is Validity.VALID for result in outcome.statistic_results)
        assert all(result.selection_score is not None for result in outcome.statistic_results)
        assert outcome.selection is not None
        assert outcome.selection.decision_statistic == max(
            result.selection_score
            for result in outcome.statistic_results
            if result.selection_score is not None
        )
    for identity in (outcomes[0], outcomes[4]):
        assert identity.selection is not None
        assert (
            identity.selection.decision_statistic == prepared.observed_selection.decision_statistic
        )


def test_execute_retains_failures_and_continues_every_planned_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact = resolution(replicates=5)
    # No identity at the deliberately failing IDs 1 and 3.
    install_scripted_rng(monkeypatch, (1, 2, 3, 4, 5))
    original_evaluate = _BoundLaggedPearsonAdapter.evaluate_all
    evaluate_call = 0
    selection_calls = 0

    def scripted_evaluate(
        self: _BoundLaggedPearsonAdapter,
        supplied: SeriesPair,
        /,
    ) -> tuple[StatisticResult, ...]:
        nonlocal evaluate_call
        evaluate_call += 1
        if evaluate_call in {3, 5}:  # observed is call 1; failures are IDs 1 and 3
            return analytic_failure_vector(self)
        return original_evaluate(self, supplied)

    def count_selector(frame: object, event: str, _arg: object) -> None:
        nonlocal selection_calls
        if (
            event == "call" and frame.f_code is real_select_family_v2.__code__  # type: ignore[attr-defined]
        ):
            selection_calls += 1

    monkeypatch.setattr(
        _BoundLaggedPearsonAdapter,
        "evaluate_all",
        scripted_evaluate,
    )
    previous_profile = sys.getprofile()
    sys.setprofile(count_selector)
    try:
        prepared = prepare(exact)
        outcomes = _execute_replicates(prepared, observer=None)
    finally:
        sys.setprofile(previous_profile)

    assert tuple(outcome.replicate_id for outcome in outcomes) == (0, 1, 2, 3, 4)
    assert tuple(outcome.status for outcome in outcomes) == (
        ReplicateStatus.COMPLETE,
        ReplicateStatus.ANALYTIC_FAILURE,
        ReplicateStatus.COMPLETE,
        ReplicateStatus.ANALYTIC_FAILURE,
        ReplicateStatus.COMPLETE,
    )
    assert evaluate_call == 1 + exact.plan.replicates
    assert selection_calls == 1 + 3
    for replicate_id in (1, 3):
        outcome = outcomes[replicate_id]
        assert outcome.failure_stage is ReplicateFailureStage.STATISTIC_SCAN
        assert outcome.selection is None
        assert len(outcome.statistic_results) == len(exact.plan.candidates)
        assert all(
            result.validity is Validity.ANALYTIC_FAILURE for result in outcome.statistic_results
        )
        assert all(result.selection_score is None for result in outcome.statistic_results)
        assert outcome.transform_token.state.shift in {2, 4}
        assert outcome.diagnostics[-1] == "replicate_statistic_scan_analytical_failure_v2"


def test_sampled_identity_score_mismatch_is_integrity_not_a_replicate_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact = resolution(replicates=2)
    install_scripted_rng(monkeypatch, (0, 1))
    original_evaluate = _BoundLaggedPearsonAdapter.evaluate_all
    evaluate_call = 0
    outcome_constructions = 0

    def mismatching_identity(
        self: _BoundLaggedPearsonAdapter,
        supplied: SeriesPair,
        /,
    ) -> tuple[StatisticResult, ...]:
        nonlocal evaluate_call
        evaluate_call += 1
        results = original_evaluate(self, supplied)
        if evaluate_call == 2:
            results = (
                replace(results[0], estimate=10.0),
                results[1],
            )
        return results

    def bomb_outcome(self: object) -> None:
        del self
        nonlocal outcome_constructions
        outcome_constructions += 1
        raise AssertionError("identity mismatch must not become any outcome")

    monkeypatch.setattr(
        _BoundLaggedPearsonAdapter,
        "evaluate_all",
        mismatching_identity,
    )
    prepared = prepare(exact)
    monkeypatch.setattr(calibration_v2.ReplicateOutcome, "__post_init__", bomb_outcome)

    with pytest.raises(V2IntegrityError, match="identity"):
        _execute_replicates(prepared, observer=None)
    assert evaluate_call == 2
    assert outcome_constructions == 0


def test_sampled_identity_analytical_failure_is_integrity_not_statistical_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact = resolution(replicates=2)
    install_scripted_rng(monkeypatch, (0, 1))
    original_evaluate = _BoundLaggedPearsonAdapter.evaluate_all
    evaluate_call = 0

    def failing_identity(
        self: _BoundLaggedPearsonAdapter,
        supplied: SeriesPair,
        /,
    ) -> tuple[StatisticResult, ...]:
        nonlocal evaluate_call
        evaluate_call += 1
        if evaluate_call == 2:
            return analytic_failure_vector(self)
        return original_evaluate(self, supplied)

    monkeypatch.setattr(_BoundLaggedPearsonAdapter, "evaluate_all", failing_identity)
    prepared = prepare(exact)

    with pytest.raises(V2IntegrityError, match="identity"):
        _execute_replicates(prepared, observer=None)
    assert evaluate_call == 2


def test_nonidentity_data_equivalence_is_retained_without_resampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact = resolution(replicates=2)
    install_scripted_rng(monkeypatch, (2, 2))
    periodic = pair(
        [0, 1, 0, 1, 0, 1],
        [1, 2, 3, 2, 4, 3],
    )
    prepared_value = _prepare_calibration(periodic, exact)
    assert type(prepared_value) is _PreparedCalibration

    outcomes = _execute_replicates(prepared_value, observer=None)

    assert tuple(outcome.replicate_id for outcome in outcomes) == (0, 1)
    assert all(not outcome.transform_token.is_identity for outcome in outcomes)
    assert all("nonidentity_data_equivalent_v2" in outcome.diagnostics for outcome in outcomes)


def test_execute_supports_complete_block_space_with_identity_and_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact = resolve_plan_v2(
        PlanRequestV2(
            candidates=(1,),
            statistic_name="lagged_pearson_v1",
            statistic_params={},
            selection_rule="max_upper",
            null_name="block_shuffle_v2",
            null_params={"block_length": 2},
            replicates=5,
            alpha=0.05,
            tie_tolerance=1e-12,
            root_seed=17,
        )
    )
    install_scripted_rng(monkeypatch, (0, 1, 0, 1, 1))
    prepared_value = _prepare_calibration(
        pair([0, 1, 2, 4], [2, 0, 4, 1]),
        exact,
    )
    assert type(prepared_value) is _PreparedCalibration

    outcomes = _execute_replicates(prepared_value, observer=None)

    assert len(outcomes) == 5 > prepared_value.bound_null.total_state_count == 2
    assert tuple(outcome.transform_token.state.block_order for outcome in outcomes) == (
        (1, 0),
        (0, 1),
        (1, 0),
        (0, 1),
        (0, 1),
    )
    assert tuple(outcome.transform_token.is_identity for outcome in outcomes) == (
        False,
        True,
        False,
        True,
        True,
    )
    assert all(outcome.status is ReplicateStatus.COMPLETE for outcome in outcomes)
    for outcome in outcomes:
        if outcome.transform_token.is_identity:
            assert outcome.selection is not None
            assert (
                outcome.selection.decision_statistic
                == prepared_value.observed_selection.decision_statistic
            )


@pytest.mark.parametrize(
    (
        "null_name",
        "null_params",
        "scripted_indices",
        "expected_states",
        "identity_ids",
        "duplicate_ids",
    ),
    [
        (
            "circular_shift_v2",
            {"min_shift": 1},
            (0, 1, 2, 2, 3),
            (0, 1, 2, 2, 3),
            (0,),
            (2, 3),
        ),
        (
            "block_shuffle_v2",
            {"block_length": 5},
            (1, 0, 0, 1, 0),
            ((0, 1), (1, 0), (1, 0), (0, 1), (1, 0)),
            (0, 3),
            (1, 2),
        ),
    ],
)
def test_binned_nette_two_registered_nulls_execute_exact_b_full_reselection(
    monkeypatch: pytest.MonkeyPatch,
    null_name: str,
    null_params: dict[str, int],
    scripted_indices: tuple[int, ...],
    expected_states: tuple[object, ...],
    identity_ids: tuple[int, ...],
    duplicate_ids: tuple[int, int],
) -> None:
    """Exercise Binned NetTE through both owned nulls, including one failure."""

    planned_replicates = len(scripted_indices)
    assert planned_replicates >= 3
    exact = resolve_plan_v2(
        PlanRequestV2(
            candidates=(1, 2),
            statistic_name="equal_width_binned_nette_v1",
            statistic_params={"bins": 3},
            selection_rule="max_absolute",
            null_name=null_name,
            null_params=null_params,
            replicates=planned_replicates,
            alpha=0.05,
            tie_tolerance=1e-12,
            root_seed=17,
        )
    )
    install_scripted_rng(monkeypatch, scripted_indices)
    evaluated_vectors: list[tuple[StatisticResult, ...]] = []
    selection_calls = 0
    original_evaluate = _BoundBinnedNetTEAdapter.evaluate_all

    def scripted_evaluate(
        self: _BoundBinnedNetTEAdapter,
        supplied: SeriesPair,
        /,
    ) -> tuple[StatisticResult, ...]:
        raw = original_evaluate(self, supplied)
        if len(evaluated_vectors) == 2:  # observed is 0; replicate ID 1 is 2.
            raw = tuple(
                StatisticResult(
                    candidate_id=candidate,
                    estimate=None,
                    selection_score=None,
                    support_n=self.observed_length - max(self.candidates),
                    validity=Validity.ANALYTIC_FAILURE,
                    diagnostics=("scripted_binned_nette_analytic_failure",),
                    backend_identity=self.backend_identity,
                    preprocessing_identity=self.preprocessing_identity,
                )
                for candidate in self.candidates
            )
        evaluated_vectors.append(raw)
        return raw

    def count_selector(frame: object, event: str, _arg: object) -> None:
        nonlocal selection_calls
        if event == "call" and frame.f_code is real_select_family_v2.__code__:  # type: ignore[attr-defined]
            selection_calls += 1

    monkeypatch.setattr(_BoundBinnedNetTEAdapter, "evaluate_all", scripted_evaluate)
    previous_profile = sys.getprofile()
    sys.setprofile(count_selector)
    try:
        prepared = _prepare_calibration(
            pair(
                [0, 1, 0, 1, 2, 1, 2, 0, 2, 1],
                [1, 0, 1, 2, 1, 0, 2, 1, 2, 0],
            ),
            exact,
        )
        outcomes = _execute_replicates(prepared, observer=None)
    finally:
        sys.setprofile(previous_profile)

    assert type(outcomes) is tuple
    assert len(outcomes) == planned_replicates
    assert tuple(outcome.replicate_id for outcome in outcomes) == tuple(
        range(planned_replicates)
    )
    assert len(evaluated_vectors) == 1 + planned_replicates
    assert selection_calls == 1 + planned_replicates - 1
    assert tuple(outcome.status for outcome in outcomes) == (
        ReplicateStatus.COMPLETE,
        ReplicateStatus.ANALYTIC_FAILURE,
        ReplicateStatus.COMPLETE,
        ReplicateStatus.COMPLETE,
        ReplicateStatus.COMPLETE,
    )
    assert outcomes[1].failure_stage is ReplicateFailureStage.STATISTIC_SCAN
    assert outcomes[1].selection is None
    assert outcomes[-1].status is ReplicateStatus.COMPLETE

    if null_name == "circular_shift_v2":
        actual_states = tuple(
            outcome.transform_token.state.shift for outcome in outcomes
        )
    else:
        actual_states = tuple(
            outcome.transform_token.state.block_order for outcome in outcomes
        )
    assert actual_states == expected_states
    assert tuple(
        outcome.replicate_id
        for outcome in outcomes
        if outcome.transform_token.is_identity
    ) == identity_ids
    left_duplicate, right_duplicate = duplicate_ids
    assert actual_states[left_duplicate] == actual_states[right_duplicate]
    assert (
        outcomes[left_duplicate].transform_token
        is not outcomes[right_duplicate].transform_token
    )

    for replicate_id, (raw, outcome) in enumerate(
        zip(evaluated_vectors[1:], outcomes, strict=True)
    ):
        assert tuple(result.candidate_id for result in raw) == (1, 2)
        assert tuple(result.candidate_id for result in outcome.statistic_results) == (1, 2)
        assert len(outcome.statistic_results) == len(exact.plan.candidates)
        if replicate_id == 1:
            assert outcome.statistic_results is raw
            assert all(
                result.validity is Validity.ANALYTIC_FAILURE
                for result in outcome.statistic_results
            )
            assert all(
                result.selection_score is None for result in outcome.statistic_results
            )
            assert (
                outcome.diagnostics[-1]
                == "replicate_statistic_scan_analytical_failure_v2"
            )
            continue

        assert all(result.validity is Validity.VALID for result in raw)
        assert all(result.selection_score is None for result in raw)
        assert all(
            result.validity is Validity.VALID for result in outcome.statistic_results
        )
        assert all(
            result.selection_score is not None for result in outcome.statistic_results
        )
        assert tuple(result.estimate for result in outcome.statistic_results) == tuple(
            result.estimate for result in raw
        )
        assert outcome.selection is not None
        expected_selection, expected_scored = real_select_family_v2(
            raw,
            exact.plan.candidates,
            exact.plan.selection_rule,
            exact.plan.tie_tolerance,
        )
        assert outcome.selection == expected_selection
        assert outcome.statistic_results == expected_scored

    for identity_id in identity_ids:
        identity = outcomes[identity_id]
        assert identity.selection is not None
        assert (
            identity.selection.decision_statistic
            == prepared.observed_selection.decision_statistic
        )
        assert identity.statistic_results == prepared.observed_results


def test_transform_result_source_changed_must_match_the_actual_source() -> None:
    prepared = prepare()
    token = prepared.null_snapshot.execution.identity_token.callable()
    token_snapshot = calibration_v2._snapshot_replicate_token(token, prepared)
    result = prepared.null_snapshot.execution.apply.callable(token)
    object.__setattr__(result, "source_changed", True)

    with pytest.raises(V2IntegrityError, match="source_changed"):
        calibration_v2._validate_transform_result(
            result,
            sampled_token=token,
            sampled_token_snapshot=token_snapshot,
            prepared=prepared,
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda raw: list(raw), "exact result tuple"),
        (lambda raw: raw[:-1], "match the plan"),
        (lambda raw: (replace(raw[0], selection_score=0.1), raw[1]), "unscored"),
        (
            lambda raw: (replace(raw[0], backend_identity="foreign"), raw[1]),
            "implementation identity",
        ),
        (
            lambda raw: (replace(raw[0], preprocessing_identity="foreign"), raw[1]),
            "implementation identity",
        ),
    ],
)
def test_replicate_raw_vector_guards_fail_closed(
    mutation: object,
    match: str,
) -> None:
    prepared = prepare()
    malformed = mutation(prepared.raw_observed_results)  # type: ignore[operator]
    with pytest.raises(V2IntegrityError, match=match):
        calibration_v2._validate_raw_replicate_vector(
            malformed,
            prepared,
            replicate_id=0,
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("schema", "foreign", "schema"),
        ("null_name", "block_shuffle_v2", "null identity"),
        ("semantic_input_sha256", "a" * 64, "ownership"),
        ("is_identity", 1, "identity flag"),
        (
            "state",
            BlockShuffleStateV2(
                schema="selcal.block-shuffle-state.v2",
                block_order=(1, 0),
            ),
            "circular token state",
        ),
    ],
)
def test_circular_sampled_token_common_guards_fail_closed(
    field: str,
    value: object,
    match: str,
) -> None:
    prepared = prepare()
    token = prepared.null_snapshot.execution.identity_token.callable()
    object.__setattr__(token, field, value)
    with pytest.raises(V2IntegrityError, match=match):
        calibration_v2._snapshot_replicate_token(token, prepared)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("schema", "foreign", "circular token state"),
        ("shift", np.int64(0), "circular token state"),
    ],
)
def test_circular_sampled_state_guards_fail_closed(
    field: str,
    value: object,
    match: str,
) -> None:
    prepared = prepare()
    token = prepared.null_snapshot.execution.identity_token.callable()
    object.__setattr__(token.state, field, value)
    with pytest.raises(V2IntegrityError, match=match):
        calibration_v2._snapshot_replicate_token(token, prepared)


def test_circular_sampled_token_rejects_identity_state_contradiction() -> None:
    prepared = prepare()
    token = prepared.null_snapshot.execution.identity_token.callable()
    object.__setattr__(token, "is_identity", False)
    with pytest.raises(V2IntegrityError, match="contradicts"):
        calibration_v2._snapshot_replicate_token(token, prepared)


def test_circular_sampled_state_normalizes_structural_property_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = prepare()
    token = prepared.null_snapshot.execution.identity_token.callable()

    def explode(self: CircularShiftStateV2) -> str:
        del self
        raise ValueError("structural")

    monkeypatch.setattr(CircularShiftStateV2, "schema", property(explode))
    with pytest.raises(V2IntegrityError, match="circular token state is invalid"):
        calibration_v2._snapshot_replicate_token(token, prepared)


def test_sampled_token_requires_an_exact_token() -> None:
    with pytest.raises(V2IntegrityError, match="exact NullTransformToken"):
        calibration_v2._snapshot_replicate_token(object(), prepare())


def test_sampled_token_normalizes_structural_property_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = prepare()
    token = prepared.null_snapshot.execution.identity_token.callable()

    def explode(self: NullTransformToken) -> str:
        del self
        raise ValueError("structural")

    monkeypatch.setattr(NullTransformToken, "schema", property(explode))
    with pytest.raises(V2IntegrityError, match="snapshot is invalid"):
        calibration_v2._snapshot_replicate_token(token, prepared)


def test_sampled_token_does_not_swallow_generic_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = prepare()
    token = prepared.null_snapshot.execution.identity_token.callable()

    def explode(self: NullTransformToken) -> str:
        del self
        raise RuntimeError("token infrastructure exploded")

    monkeypatch.setattr(NullTransformToken, "schema", property(explode))
    with pytest.raises(RuntimeError, match="token infrastructure exploded"):
        calibration_v2._snapshot_replicate_token(token, prepared)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        (
            "state",
            CircularShiftStateV2(
                schema="selcal.circular-shift-state.v2",
                shift=1,
            ),
            "block token state",
        ),
    ],
)
def test_block_sampled_token_requires_the_exact_state(
    field: str,
    value: object,
    match: str,
) -> None:
    prepared = block_prepared()
    token = prepared.null_snapshot.execution.identity_token.callable()
    object.__setattr__(token, field, value)
    with pytest.raises(V2IntegrityError, match=match):
        calibration_v2._snapshot_replicate_token(token, prepared)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema", "foreign"),
        ("block_order", [0, 1]),
        ("block_order", (np.int64(0), 1)),
    ],
)
def test_block_sampled_state_guards_fail_closed(field: str, value: object) -> None:
    prepared = block_prepared()
    token = prepared.null_snapshot.execution.identity_token.callable()
    object.__setattr__(token.state, field, value)
    with pytest.raises(V2IntegrityError, match="block token state"):
        calibration_v2._snapshot_replicate_token(token, prepared)


def test_block_sampled_state_normalizes_structural_property_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = block_prepared()
    token = prepared.null_snapshot.execution.identity_token.callable()

    def explode(self: BlockShuffleStateV2) -> str:
        del self
        raise ValueError("structural")

    monkeypatch.setattr(BlockShuffleStateV2, "schema", property(explode))
    with pytest.raises(V2IntegrityError, match="block token state is invalid"):
        calibration_v2._snapshot_replicate_token(token, prepared)


@pytest.mark.parametrize(
    ("role", "contract_type", "field_name"),
    [
        ("circular", NullTransformToken, "schema"),
        ("circular", CircularShiftStateV2, "shift"),
        ("block", BlockShuffleStateV2, "block_order"),
    ],
)
def test_execute_rejects_transparent_token_state_descriptor_replacement(
    role: str,
    contract_type: type[object],
    field_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = prepare(resolution(replicates=1)) if role == "circular" else block_prepared()
    original = inspect.getattr_static(contract_type, field_name)
    monkeypatch.setattr(
        contract_type,
        field_name,
        TransparentSlotReplacement(original, contract_type),
    )

    with pytest.raises(V2IntegrityError, match=r"descriptor|drift|snapshot"):
        _execute_replicates(prepared, observer=None)


def test_sampled_token_rejects_an_unsupported_equal_null_name() -> None:
    prepared = prepare()
    token = prepared.null_snapshot.execution.identity_token.callable()
    object.__setattr__(prepared.null_snapshot, "name", "unsupported_null_v2")
    object.__setattr__(token, "null_name", "unsupported_null_v2")
    with pytest.raises(V2IntegrityError, match="unsupported"):
        calibration_v2._snapshot_replicate_token(token, prepared)


def test_capsule_seed_digest_rejects_foreign_streams_and_bad_digest_storage() -> None:
    capsule = inspect.getclosurevars(_execute_replicates).nonlocals["random_capsule"]
    create_stream = tuple.__getitem__(capsule, 0)
    seed_digest = tuple.__getitem__(capsule, 1)
    with pytest.raises(V2IntegrityError, match="random stream"):
        seed_digest(object())
    valid_stream = create_stream("0" * 64, 0, 1)
    malformed_stream = (tuple.__getitem__(valid_stream, 0), "not-a-digest")
    with pytest.raises(V2IntegrityError, match="seed digest drifted"):
        seed_digest(malformed_stream)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda result, prepared: object(), "exact NullTransformResult"),
        (
            lambda result, prepared: replace(
                result,
                token=prepared.null_snapshot.execution.sample_token.callable(OneIndex(1)),
            ),
            "token drifted",
        ),
        (
            lambda result, prepared: _mutate_result(result, "source_changed", 1),
            "source_changed",
        ),
        (
            lambda result, prepared: _mutate_result(result, "diagnostics", []),
            "diagnostics",
        ),
        (
            lambda result, prepared: _mutate_result(
                result,
                "pair",
                pair([0, 1, 2, 3, 4], [4, 3, 2, 1, 0]),
            ),
            "shape",
        ),
        (
            lambda result, prepared: _mutate_result(
                result,
                "pair",
                pair(
                    list(result.pair.source),
                    list(reversed(result.pair.target)),
                ),
            ),
            "target identity",
        ),
    ],
)
def test_transform_result_guards_fail_closed(
    mutation: object,
    match: str,
) -> None:
    prepared = prepare()
    token = prepared.null_snapshot.execution.identity_token.callable()
    token_snapshot = calibration_v2._snapshot_replicate_token(token, prepared)
    result = prepared.null_snapshot.execution.apply.callable(token)
    malformed = mutation(result, prepared)  # type: ignore[operator]
    with pytest.raises(V2IntegrityError, match=match):
        calibration_v2._validate_transform_result(
            malformed,
            sampled_token=token,
            sampled_token_snapshot=token_snapshot,
            prepared=prepared,
        )


def _mutate_result(result: object, field: str, value: object) -> object:
    object.__setattr__(result, field, value)
    return result


def test_transform_result_normalizes_structural_property_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = prepare()
    token = prepared.null_snapshot.execution.identity_token.callable()
    token_snapshot = calibration_v2._snapshot_replicate_token(token, prepared)
    result = prepared.null_snapshot.execution.apply.callable(token)

    def explode(self: NullTransformResult) -> SeriesPair:
        del self
        raise ValueError("structural")

    monkeypatch.setattr(NullTransformResult, "pair", property(explode))
    with pytest.raises(V2IntegrityError, match="snapshot is invalid"):
        calibration_v2._validate_transform_result(
            result,
            sampled_token=token,
            sampled_token_snapshot=token_snapshot,
            prepared=prepared,
        )


@pytest.mark.parametrize("replacement_digest", ["f" * 64, "", object()])
def test_public_seed_digest_descriptor_drift_is_inert_to_the_sealed_executor(
    replacement_digest: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact = resolution(replicates=1)
    baseline = _execute_replicates(prepare(exact), observer=None)
    descriptor_reads = 0

    def drifting_digest(self: RealReplicateRandomSource) -> object:
        del self
        nonlocal descriptor_reads
        descriptor_reads += 1
        return replacement_digest

    monkeypatch.setattr(RealReplicateRandomSource, "seed_digest_sha256", property(drifting_digest))
    attacked = _execute_replicates(prepare(exact), observer=None)

    assert attacked == baseline
    assert descriptor_reads == 0


@pytest.mark.parametrize("drift_stage", ["evaluate", "select"])
def test_replicate_token_drift_is_rejected_after_each_scan_stage(
    drift_stage: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_scripted_rng(monkeypatch, (1,))
    sampled: list[NullTransformToken] = []
    original_token_post_init = NullTransformToken.__post_init__
    original_result_post_init = StatisticResult.__post_init__

    def capture_sampled_token(self: NullTransformToken) -> None:
        original_token_post_init(self)
        frame = inspect.currentframe()
        assert frame is not None and frame.f_back is not None
        constructor_caller = frame.f_back.f_back
        if (
            constructor_caller is not None
            and constructor_caller.f_code.co_name == "_token_for_shift"
        ):
            sampled.append(self)

    def drift_during_scored_result_construction(self: StatisticResult) -> None:
        original_result_post_init(self)
        if sampled and (
            (drift_stage == "evaluate" and self.selection_score is None)
            or (drift_stage == "select" and self.selection_score is not None)
        ):
            object.__setattr__(sampled[-1].state, "shift", 2)

    prepared = prepare(resolution(replicates=1))
    monkeypatch.setattr(
        NullTransformToken,
        "__post_init__",
        capture_sampled_token,
    )
    monkeypatch.setattr(
        StatisticResult,
        "__post_init__",
        drift_during_scored_result_construction,
    )
    with pytest.raises(V2IntegrityError, match="token drifted"):
        _execute_replicates(prepared, observer=None)


def test_replicate_selector_requires_an_exact_two_item_tuple(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_scripted_rng(monkeypatch, (1,))
    prepared = prepare(resolution(replicates=1))
    raw_snapshot = calibration_v2._result_vector_snapshot(
        prepared.raw_observed_results,
        subject="direct malformed selector vector",
    )
    with pytest.raises(V2IntegrityError, match="scored output"):
        calibration_v2._validate_selector_output(
            prepared.raw_observed_results,
            raw_snapshot,
            prepared.observed_selection,
            object(),
            prepared.resolution.plan.candidates,
            True,
            prepared.resolution.plan.tie_tolerance,
        )

    monkeypatch.setattr(calibration_v2, "select_family_v2", lambda *args: object())
    with pytest.raises(V2IntegrityError, match=r"selector.*drift"):
        _execute_replicates(prepared, observer=None)


def test_replicate_outcome_must_be_the_exact_contract_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_scripted_rng(monkeypatch, (1,))
    prepared = prepare(resolution(replicates=1))
    monkeypatch.setattr(calibration_v2, "ReplicateOutcome", lambda **kwargs: object())
    with pytest.raises(
        V2IntegrityError,
        match="behavior-critical module global drifted: ReplicateOutcome",
    ):
        _execute_replicates(prepared, observer=None)


def test_replicate_outcome_is_revalidated_after_its_constructor_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_scripted_rng(monkeypatch, (1,))
    prepared = prepare(resolution(replicates=1))
    canonical_type = calibration_v2.ReplicateOutcome
    original_post_init = canonical_type.__post_init__

    def corrupt_status_after_validation(self: object) -> None:
        original_post_init(self)  # type: ignore[arg-type]
        object.__setattr__(self, "status", "complete")

    monkeypatch.setattr(canonical_type, "__post_init__", corrupt_status_after_validation)

    with pytest.raises(V2IntegrityError, match=r"replicate outcome|status|contract"):
        returned = _execute_replicates(prepared, observer=None)
        assert type(returned[0]) is canonical_type
        assert type(returned[0].status) is str
        assert returned[0].status == "complete"


def test_complete_outcome_final_expected_check_follows_contract_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_scripted_rng(monkeypatch, (1,))
    prepared = prepare(resolution(replicates=1))
    original_validator = contracts_v2._validate_complete_scored_vector
    validation_calls = 0

    def poison_on_revalidation(*args: object, **kwargs: object) -> object:
        nonlocal validation_calls
        validation_calls += 1
        result = original_validator(*args, **kwargs)  # type: ignore[arg-type]
        if validation_calls == 2:
            statistic_results = args[0]
            object.__setattr__(statistic_results[0], "estimate", 999.0)  # type: ignore[index]
        return result

    monkeypatch.setattr(
        contracts_v2,
        "_validate_complete_scored_vector",
        poison_on_revalidation,
    )

    with pytest.raises(V2IntegrityError, match=r"replicate outcome|result vector|drift"):
        returned = _execute_replicates(prepared, observer=None)
        assert returned[0].statistic_results[0].estimate == 999.0
    assert validation_calls == 2


def test_analytical_outcome_final_expected_check_follows_contract_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_scripted_rng(monkeypatch, (1,))
    original_evaluate = _BoundLaggedPearsonAdapter.evaluate_all
    original_validator = contracts_v2._validate_raw_failure_vector
    evaluate_calls = 0
    validation_calls = 0

    def fail_replicate_scan(
        self: _BoundLaggedPearsonAdapter,
        supplied: SeriesPair,
        /,
    ) -> tuple[StatisticResult, ...]:
        nonlocal evaluate_calls
        evaluate_calls += 1
        if evaluate_calls == 2:
            return analytic_failure_vector(self)
        return original_evaluate(self, supplied)

    def poison_on_revalidation(*args: object, **kwargs: object) -> object:
        nonlocal validation_calls
        validation_calls += 1
        result = original_validator(*args, **kwargs)  # type: ignore[arg-type]
        if validation_calls == 2:
            statistic_results = args[0]
            object.__setattr__(statistic_results[0], "support_n", 999)  # type: ignore[index]
        return result

    monkeypatch.setattr(_BoundLaggedPearsonAdapter, "evaluate_all", fail_replicate_scan)
    prepared = prepare(resolution(replicates=1))
    monkeypatch.setattr(
        contracts_v2,
        "_validate_raw_failure_vector",
        poison_on_revalidation,
    )

    with pytest.raises(V2IntegrityError, match=r"replicate outcome|result vector|drift"):
        returned = _execute_replicates(prepared, observer=None)
        assert returned[0].statistic_results[0].support_n == 999
    assert validation_calls == 2


@pytest.mark.parametrize("analytical_failure", [False, True])
def test_aggregate_token_snapshot_cannot_mutate_an_outcome_after_field_capture(
    analytical_failure: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_scripted_rng(monkeypatch, (1,))
    original_evaluate = _BoundLaggedPearsonAdapter.evaluate_all
    original_snapshot_token = _BoundCircularShiftV2.snapshot_token
    original_post_init = calibration_v2.ReplicateOutcome.__post_init__
    evaluate_calls = 0
    aggregate_snapshot_calls = 0
    captured_outcomes: list[object] = []

    def maybe_fail_replicate_scan(
        self: _BoundLaggedPearsonAdapter,
        supplied: SeriesPair,
        /,
    ) -> tuple[StatisticResult, ...]:
        nonlocal evaluate_calls
        evaluate_calls += 1
        if analytical_failure and evaluate_calls == 2:
            return analytic_failure_vector(self)
        return original_evaluate(self, supplied)

    def capture_outcome(self: object) -> None:
        original_post_init(self)  # type: ignore[arg-type]
        captured_outcomes.append(self)

    def mutate_during_aggregate_snapshot(
        self: _BoundCircularShiftV2,
        token: object,
        /,
    ) -> tuple[object, ...]:
        nonlocal aggregate_snapshot_calls
        snapshot = original_snapshot_token(self, token)
        frame = inspect.currentframe()
        while frame is not None and frame.f_code.co_name != (
            "_capture_task8b_terminal_token_snapshots"
        ):
            frame = frame.f_back
        if frame is not None:
            aggregate_snapshot_calls += 1
            outcome = captured_outcomes[0]
            status = outcome.status  # type: ignore[attr-defined]
            object.__setattr__(outcome, "status", status.value)
        return snapshot

    monkeypatch.setattr(
        _BoundLaggedPearsonAdapter,
        "evaluate_all",
        maybe_fail_replicate_scan,
    )
    monkeypatch.setattr(
        _BoundCircularShiftV2,
        "snapshot_token",
        mutate_during_aggregate_snapshot,
    )
    prepared = prepare(resolution(replicates=1))
    monkeypatch.setattr(
        calibration_v2.ReplicateOutcome,
        "__post_init__",
        capture_outcome,
    )

    with pytest.raises(V2IntegrityError, match=r"aggregate|history|outcome|drift"):
        returned = _execute_replicates(prepared, observer=None)
        assert type(returned[0].status) is str
    assert aggregate_snapshot_calls == 1


def test_terminal_token_hooks_all_finish_before_history_is_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact = resolution(replicates=2)
    install_scripted_rng(monkeypatch, (1, 2))
    original_snapshot_token = _BoundCircularShiftV2.snapshot_token
    original_post_init = calibration_v2.ReplicateOutcome.__post_init__
    captured_outcomes: list[object] = []
    terminal_snapshot_calls = 0

    def capture_outcome(self: object) -> None:
        original_post_init(self)  # type: ignore[arg-type]
        captured_outcomes.append(self)

    def pollute_earlier_outcome_from_the_last_terminal_token_hook(
        self: _BoundCircularShiftV2,
        token: object,
        /,
    ) -> tuple[object, ...]:
        nonlocal terminal_snapshot_calls
        snapshot = original_snapshot_token(self, token)
        frame = inspect.currentframe()
        terminal_frame_names = {
            "aggregate_snapshot",
            "_capture_task8b_terminal_token_snapshots",
        }
        while frame is not None and frame.f_code.co_name not in terminal_frame_names:
            frame = frame.f_back
        if frame is not None:
            terminal_snapshot_calls += 1
            if terminal_snapshot_calls == 2:
                first = captured_outcomes[0]
                object.__setattr__(first.selection, "decision_statistic", -999.0)
        return snapshot

    monkeypatch.setattr(
        _BoundCircularShiftV2,
        "snapshot_token",
        pollute_earlier_outcome_from_the_last_terminal_token_hook,
    )
    prepared = prepare(exact)
    monkeypatch.setattr(
        calibration_v2.ReplicateOutcome,
        "__post_init__",
        capture_outcome,
    )

    with pytest.raises(V2IntegrityError, match=r"history|outcome|drift"):
        _execute_replicates(prepared, observer=None)
    assert terminal_snapshot_calls == 2


@pytest.mark.parametrize("analytical_failure", [False, True])
def test_expected_outcome_comparison_never_calls_custom_scalar_equality(
    analytical_failure: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_scripted_rng(monkeypatch, (1,))
    original_evaluate = _BoundLaggedPearsonAdapter.evaluate_all
    original_snapshot_token = _BoundCircularShiftV2.snapshot_token
    original_post_init = calibration_v2.ReplicateOutcome.__post_init__
    evaluate_calls = 0
    snapshot_injections = 0
    equality_calls = 0
    captured_outcomes: list[object] = []

    class MutatingFloat(float):
        def __eq__(self, other: object) -> bool:
            nonlocal equality_calls
            equality_calls += 1
            outcome = captured_outcomes[0]
            result = outcome.statistic_results[0]  # type: ignore[attr-defined]
            object.__setattr__(result, "support_n", 999)
            return bool(super().__eq__(other))

    class MutatingInt(int):
        def __eq__(self, other: object) -> bool:
            nonlocal equality_calls
            equality_calls += 1
            outcome = captured_outcomes[0]
            result = outcome.statistic_results[0]  # type: ignore[attr-defined]
            object.__setattr__(result, "diagnostics", ("custom_equality_pollution",))
            return bool(super().__eq__(other))

    def maybe_fail_replicate_scan(
        self: _BoundLaggedPearsonAdapter,
        supplied: SeriesPair,
        /,
    ) -> tuple[StatisticResult, ...]:
        nonlocal evaluate_calls
        evaluate_calls += 1
        if analytical_failure and evaluate_calls == 2:
            return analytic_failure_vector(self)
        return original_evaluate(self, supplied)

    def capture_outcome(self: object) -> None:
        original_post_init(self)  # type: ignore[arg-type]
        captured_outcomes.append(self)

    def inject_equal_valued_scalar(
        self: _BoundCircularShiftV2,
        token: object,
        /,
    ) -> tuple[object, ...]:
        nonlocal snapshot_injections
        snapshot = original_snapshot_token(self, token)
        frame = inspect.currentframe()
        while frame is not None and frame.f_code.co_name != (
            "_register_task8b_outcome_signature"
        ):
            frame = frame.f_back
        if (
            frame is not None
            and frame.f_locals.get("expected_contract") is not None
            and snapshot_injections == 0
        ):
            snapshot_injections += 1
            result = captured_outcomes[0].statistic_results[0]  # type: ignore[attr-defined]
            if analytical_failure:
                object.__setattr__(result, "support_n", MutatingInt(result.support_n))
            else:
                object.__setattr__(result, "estimate", MutatingFloat(result.estimate))
        return snapshot

    monkeypatch.setattr(
        _BoundLaggedPearsonAdapter,
        "evaluate_all",
        maybe_fail_replicate_scan,
    )
    monkeypatch.setattr(
        _BoundCircularShiftV2,
        "snapshot_token",
        inject_equal_valued_scalar,
    )
    prepared = prepare(resolution(replicates=1))
    monkeypatch.setattr(
        calibration_v2.ReplicateOutcome,
        "__post_init__",
        capture_outcome,
    )

    with pytest.raises(V2IntegrityError, match=r"replicate outcome|result vector|contract"):
        returned = _execute_replicates(prepared, observer=None)
        assert equality_calls >= 1
        if analytical_failure:
            assert returned[0].statistic_results[0].diagnostics == ("custom_equality_pollution",)
        else:
            assert returned[0].statistic_results[0].support_n == 999
    assert snapshot_injections == 1
    assert equality_calls == 0


@pytest.mark.parametrize("target", ["selection", "diagnostics"])
def test_registration_rejects_valid_fields_changed_by_the_token_snapshot_hook(
    target: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_snapshot_token = _BoundCircularShiftV2.snapshot_token
    attack_fired = False

    def mutate_after_outcome_validation(
        self: _BoundCircularShiftV2,
        token: object,
        /,
    ) -> tuple[object, ...]:
        nonlocal attack_fired
        snapshot = original_snapshot_token(self, token)
        frame = inspect.currentframe()
        while frame is not None and frame.f_code.co_name != (
            "_register_task8b_outcome_signature"
        ):
            frame = frame.f_back
        if frame is not None and not attack_fired:
            outcome = frame.f_locals["outcome"]
            if target == "selection":
                selection = outcome.selection
                object.__setattr__(
                    selection,
                    "decision_statistic",
                    selection.decision_statistic + 0.125,
                )
            else:
                object.__setattr__(outcome, "diagnostics", ("post_validation_drift",))
            attack_fired = True
        return snapshot

    monkeypatch.setattr(
        _BoundCircularShiftV2,
        "snapshot_token",
        mutate_after_outcome_validation,
    )

    with pytest.raises(V2IntegrityError, match=r"selection|outcome|contract|drift"):
        _execute_replicates(prepare(resolution(replicates=1)), observer=None)
    assert attack_fired


def test_terminal_gate_rejects_a_replaced_registered_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_snapshot_token = _BoundCircularShiftV2.snapshot_token
    attack_fired = False

    def replace_signature_during_terminal_token_hooks(
        self: _BoundCircularShiftV2,
        token: object,
        /,
    ) -> tuple[object, ...]:
        nonlocal attack_fired
        snapshot = original_snapshot_token(self, token)
        frame = inspect.currentframe()
        while frame is not None and frame.f_code.co_name != (
            "_capture_task8b_terminal_token_snapshots"
        ):
            frame = frame.f_back
        if frame is not None and not attack_fired:
            record = frame.f_locals["record"]
            object.__setattr__(record, "signature", ())
            attack_fired = True
        return snapshot

    monkeypatch.setattr(
        _BoundCircularShiftV2,
        "snapshot_token",
        replace_signature_during_terminal_token_hooks,
    )

    with pytest.raises(V2IntegrityError, match=r"history|signature|drift"):
        _execute_replicates(prepare(resolution(replicates=1)), observer=None)
    assert attack_fired


@pytest.mark.parametrize("replacement", ["bool", "custom_eq"])
def test_terminal_gate_rejects_typed_substitution_inside_a_registered_signature(
    replacement: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_snapshot_token = _BoundCircularShiftV2.snapshot_token
    attack_fired = False
    equality_calls = 0

    class UserEqualityInt(int):
        def __eq__(self, other: object) -> bool:
            nonlocal equality_calls
            equality_calls += 1
            return bool(super().__eq__(other))

    def replace_int_with_equal_bool_during_terminal_hooks(
        self: _BoundCircularShiftV2,
        token: object,
        /,
    ) -> tuple[object, ...]:
        nonlocal attack_fired
        snapshot = original_snapshot_token(self, token)
        frame = inspect.currentframe()
        while frame is not None and frame.f_code.co_name != (
            "_capture_task8b_terminal_token_snapshots"
        ):
            frame = frame.f_back
        if frame is not None and not attack_fired:
            record = frame.f_locals["record"]
            signature = record.signature
            replicate_id = signature[1]
            forged_value = False if replacement == "bool" else UserEqualityInt(0)
            forged_replicate_id = (replicate_id[0], forged_value)
            forged = (signature[0], forged_replicate_id, *signature[2:])
            object.__setattr__(record, "signature", forged)
            attack_fired = True
        return snapshot

    monkeypatch.setattr(
        _BoundCircularShiftV2,
        "snapshot_token",
        replace_int_with_equal_bool_during_terminal_hooks,
    )

    with pytest.raises(V2IntegrityError, match=r"history|signature|drift"):
        _execute_replicates(prepare(resolution(replicates=1)), observer=None)
    assert attack_fired
    assert equality_calls == 0


def test_later_callback_cannot_forge_history_to_match_a_polluted_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_scripted_rng(monkeypatch, (1, 2))
    original_sample = _BoundCircularShiftV2.sample_token
    attack_fired = False

    def pollute_outcome_and_forge_its_record(
        self: _BoundCircularShiftV2,
        random: object,
        /,
    ) -> NullTransformToken:
        nonlocal attack_fired
        frame = inspect.currentframe()
        while frame is not None and frame.f_code.co_name != "_execute_replicates":
            frame = frame.f_back
        if frame is not None and len(frame.f_locals["history"]) == 1 and not attack_fired:
            record = frame.f_locals["history"][0]
            ops = frame.f_locals["history_integrity_ops"]
            selection = record.outcome.selection
            object.__setattr__(selection, "decision_statistic", -999.0)
            fields_value = ops.read_exact_slots(
                selection,
                ops.selection_slot_spec,
                ops.integrity_error,
            )
            selection_snapshot = ops.exact_value_snapshot(
                fields_value,
                ops.enum_members,
                ops.integrity_error,
            )
            forged = (
                *record.signature[:8],
                (id(selection), selection_snapshot),
                record.signature[9],
            )
            object.__setattr__(record, "signature", forged)
            attack_fired = True
        return original_sample(self, random)

    monkeypatch.setattr(
        _BoundCircularShiftV2,
        "sample_token",
        pollute_outcome_and_forge_its_record,
    )

    with pytest.raises(V2IntegrityError, match=r"history|signature|drift"):
        _execute_replicates(prepare(resolution(replicates=2)), observer=None)
    assert attack_fired


def test_prepare_rejects_a_subclass_replacing_the_prepared_type_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = calibration_v2._PreparedCalibration

    class DerivedPrepared(canonical):
        pass

    monkeypatch.setattr(calibration_v2, "_PreparedCalibration", DerivedPrepared)
    with pytest.raises(V2IntegrityError, match=r"canonical|type alias|drift"):
        _prepare_calibration(observed_pair(), resolution(replicates=1))


def test_execute_rejects_a_registered_subclass_after_prepared_alias_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical_value = prepare(resolution(replicates=1))
    canonical_type = calibration_v2._PreparedCalibration

    class DerivedPrepared(canonical_type):
        pass

    values = {field.name: getattr(canonical_value, field.name) for field in fields(canonical_value)}
    forged = DerivedPrepared(**values, seal=calibration_v2._PREPARATION_SEAL)
    monkeypatch.setattr(calibration_v2, "_PreparedCalibration", DerivedPrepared)
    with pytest.raises(V2IntegrityError, match=r"canonical|type alias|drift"):
        calibration_v2._capture_prepared_ownership(forged)
    calibration_v2._PREPARER_OWNED[forged] = calibration_v2._PREPARER_OWNED[canonical_value]

    with pytest.raises(V2IntegrityError, match=r"canonical|type alias|drift"):
        _execute_replicates(forged, observer=None)


def test_execute_rejects_a_direct_unowned_prepared_subclass() -> None:
    canonical_value = prepare(resolution(replicates=1))
    canonical_type = calibration_v2._PreparedCalibration

    class DerivedPrepared(canonical_type):
        pass

    values = {field.name: getattr(canonical_value, field.name) for field in fields(canonical_value)}
    forged = DerivedPrepared(**values, seal=calibration_v2._PREPARATION_SEAL)
    with pytest.raises(V2IntegrityError, match=r"canonical|type alias|drift"):
        calibration_v2._capture_prepared_ownership(forged)
    with pytest.raises(V2IntegrityError, match="exact preparer-owned"):
        _execute_replicates(forged, observer=None)


def test_execute_rejects_a_subclass_replacing_the_outcome_type_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = prepare(resolution(replicates=1))
    canonical = calibration_v2.ReplicateOutcome

    class DerivedOutcome(canonical):
        pass

    monkeypatch.setattr(calibration_v2, "ReplicateOutcome", DerivedOutcome)
    with pytest.raises(
        V2IntegrityError,
        match="behavior-critical module global drifted: ReplicateOutcome",
    ):
        _execute_replicates(prepared, observer=None)


def test_execute_revalidates_the_frozen_outcome_type_after_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = prepare(resolution(replicates=1))
    canonical_type = calibration_v2._require_frozen_outcome_type()
    checks = 0

    def drifting_type_gate() -> type[object]:
        nonlocal checks
        checks += 1
        return canonical_type if checks == 1 else object

    monkeypatch.setattr(
        calibration_v2,
        "_require_frozen_outcome_type",
        drifting_type_gate,
    )
    outcomes = _execute_replicates(prepared, observer=None)
    assert all(type(outcome) is canonical_type for outcome in outcomes)
    assert checks == 0


def test_outcome_history_rejects_a_direct_contract_subclass() -> None:
    prepared = prepare(resolution(replicates=1))
    outcome = _execute_replicates(prepared, observer=None)[0]
    canonical_type = type(outcome)

    class DerivedOutcome(canonical_type):
        pass

    values = {field.name: getattr(outcome, field.name) for field in fields(outcome)}
    derived = DerivedOutcome(**values)
    with pytest.raises(V2IntegrityError, match=r"canonical|type|drift|exact"):
        calibration_v2._task8b_read_exact_slots(
            derived,
            calibration_v2._TASK8B_OUTCOME_SLOT_SPEC,
            V2IntegrityError,
        )


def test_registered_outcome_snapshot_rejects_replaced_public_field_getters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcome = _execute_replicates(prepare(resolution(replicates=1)), observer=None)[0]
    getter_calls = 0

    def inaccessible(_self: object) -> int:
        nonlocal getter_calls
        getter_calls += 1
        raise AttributeError("injected inaccessible slot")

    monkeypatch.setattr(type(outcome), "replicate_id", property(inaccessible))
    with pytest.raises(V2IntegrityError, match=r"descriptor|drift"):
        calibration_v2._task8b_read_exact_slots(
            outcome,
            calibration_v2._TASK8B_OUTCOME_SLOT_SPEC,
            V2IntegrityError,
        )
    assert getter_calls == 0


def test_task8b_slot_walker_rejects_a_nonmember_contract_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcome = _execute_replicates(prepare(resolution(replicates=1)), observer=None)[0]
    assert outcome.selection is not None
    monkeypatch.setattr(
        SelectionResult,
        "decision_statistic",
        property(lambda _self: 0.0),
    )
    with pytest.raises(V2IntegrityError, match=r"descriptor|drift"):
        calibration_v2._task8b_read_exact_slots(
            outcome.selection,
            calibration_v2._TASK8B_SELECTION_SLOT_SPEC,
            V2IntegrityError,
        )
    with pytest.raises(RuntimeError, match="canonical slot"):
        calibration_v2._freeze_task8b_slot_spec(
            SelectionResult,
            calibration_v2._TASK8B_DATACLASS_FIELDS,
        )


@pytest.mark.parametrize(
    ("state_type", "match"),
    [
        ("not-a-state-type", "state type"),
        (str, "state slot"),
    ],
)
def test_task8b_token_guard_requires_an_exact_slotted_dataclass_state(
    state_type: object,
    match: str,
) -> None:
    prepared = prepare(resolution(replicates=1))
    token = prepared.null_snapshot.execution.identity_token.callable()
    valid_snapshot = prepared.null_snapshot.execution.snapshot_token.callable(token)
    malformed_snapshot = (*valid_snapshot[:7], state_type, *valid_snapshot[8:])
    fake_prepared = SimpleNamespace(
        semantic_input_sha256=prepared.semantic_input_sha256,
        scientific_plan_sha256=prepared.scientific_plan_sha256,
        null_snapshot=SimpleNamespace(
            name=prepared.null_snapshot.name,
            null_parameter_sha256=prepared.null_snapshot.null_parameter_sha256,
            bound_null_owner_sha256=prepared.null_snapshot.bound_null_owner_sha256,
            execution=SimpleNamespace(
                snapshot_token=SimpleNamespace(callable=lambda _token: malformed_snapshot),
            ),
        ),
    )

    with pytest.raises(V2IntegrityError, match=match):
        calibration_v2._capture_task8b_token_guard(
            token,
            fake_prepared,  # type: ignore[arg-type]
            calibration_v2._TASK8B_HISTORY_INTEGRITY_OPS,
        )


def test_task8b_compact_walker_defensive_contract_guards() -> None:
    prepared = prepare(resolution(replicates=1))
    outcome = _execute_replicates(prepared, observer=None)[0]
    token = outcome.transform_token
    dummy_guard = ((), (), calibration_v2._TASK8B_TOKEN_SLOT_SPEC)

    class DerivedStatisticResult(StatisticResult):
        pass

    sample = outcome.statistic_results[0]
    derived = DerivedStatisticResult(
        **{field.name: getattr(sample, field.name) for field in fields(sample)}
    )
    with pytest.raises(V2IntegrityError, match="subclass"):
        calibration_v2._task8b_result_vector_signature(
            (derived,),
            None,
            calibration_v2._TASK8B_HISTORY_INTEGRITY_OPS,
        )
    with pytest.raises(V2IntegrityError, match="frozen result vector"):
        calibration_v2._task8b_result_vector_signature(
            (),
            frozen_fields=[],  # type: ignore[arg-type]
            ops=calibration_v2._TASK8B_HISTORY_INTEGRITY_OPS,
        )
    with pytest.raises(V2IntegrityError, match="absent selection"):
        calibration_v2._task8b_selection_signature(
            None,
            (),
            calibration_v2._TASK8B_HISTORY_INTEGRITY_OPS,
        )
    with pytest.raises(V2IntegrityError, match="outcome field"):
        calibration_v2._task8b_outcome_signature_from_fields(
            outcome,
            (),
            (),
            dummy_guard,
            None,
            None,
            None,
            calibration_v2._TASK8B_HISTORY_INTEGRITY_OPS,
        )

    common_fields = (
        0,
        "a" * 64,
        ReplicateStatus.COMPLETE,
        None,
    )
    with pytest.raises(V2IntegrityError, match="token must be exact"):
        calibration_v2._task8b_outcome_signature_from_fields(
            outcome,
            (),
            (*common_fields, object(), (), None, ()),
            dummy_guard,
            None,
            None,
            None,
            calibration_v2._TASK8B_HISTORY_INTEGRITY_OPS,
        )
    with pytest.raises(V2IntegrityError, match="diagnostics"):
        calibration_v2._task8b_outcome_signature_from_fields(
            outcome,
            (),
            (*common_fields, token, (), None, []),
            dummy_guard,
            None,
            None,
            None,
            calibration_v2._TASK8B_HISTORY_INTEGRITY_OPS,
        )
    with pytest.raises(V2IntegrityError, match="expected contract"):
        calibration_v2._register_task8b_outcome_signature(
            outcome,
            (),
            prepared,
            (),
            calibration_v2._TASK8B_HISTORY_INTEGRITY_OPS,
        )
    with pytest.raises(V2IntegrityError, match="containers"):
        calibration_v2._task8b_terminal_history_snapshot(
            [],  # type: ignore[arg-type]
            (),
            (),
            calibration_v2._TASK8B_HISTORY_INTEGRITY_OPS,
        )
    with pytest.raises(V2IntegrityError, match="incomplete"):
        calibration_v2._task8b_terminal_history_snapshot(
            (),
            (dummy_guard,),
            (),
            calibration_v2._TASK8B_HISTORY_INTEGRITY_OPS,
        )


def test_registered_outcome_rejects_a_valid_but_different_token_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_scripted_rng(monkeypatch, (1,))
    original_post_init = calibration_v2.ReplicateOutcome.__post_init__

    def change_valid_shift_after_construction(self: object) -> None:
        original_post_init(self)  # type: ignore[arg-type]
        object.__setattr__(self.transform_token.state, "shift", 2)  # type: ignore[attr-defined]

    prepared = prepare(resolution(replicates=1))
    monkeypatch.setattr(
        calibration_v2.ReplicateOutcome,
        "__post_init__",
        change_valid_shift_after_construction,
    )

    with pytest.raises(V2IntegrityError, match="token contradicts"):
        _execute_replicates(prepared, observer=None)


def test_exact_snapshot_matcher_rejects_a_container_type_change() -> None:
    assert not calibration_v2._exact_snapshot_value_matches((), [])


def test_task8b_canonical_matcher_is_type_sensitive_without_user_equality() -> None:
    equality_calls = 0

    class UserEquality:
        def __eq__(self, other: object) -> bool:
            nonlocal equality_calls
            equality_calls += 1
            return other == 0

    assert calibration_v2._task8b_canonical_snapshot_matches((1,), (1,))
    assert not calibration_v2._task8b_canonical_snapshot_matches((1,), (2,))
    assert not calibration_v2._task8b_canonical_snapshot_matches((False,), (0,))
    assert not calibration_v2._task8b_canonical_snapshot_matches((UserEquality(),), (0,))
    assert not calibration_v2._task8b_canonical_snapshot_matches(
        (UserEquality(),),
        (UserEquality(),),
    )
    assert equality_calls == 0


class OutcomeTupleSubclass(tuple[StatisticResult, ...]):
    pass


@pytest.mark.parametrize(
    "pollution",
    [
        "selection_decision",
        "scored_score",
        "raw_result",
        "token",
        "replicate_id",
        "seed",
        "diagnostics",
        "container_type",
        "status_type",
        "failure_stage_type",
        "diagnostics_item_type",
        "complete_failure_stage",
        "analytic_failure_stage",
        "analytic_selection",
        "analytic_raw_identity",
        "analytic_without_failure",
        "token_state_type",
        "deleted_outcome_slot",
    ],
)
def test_final_history_check_rejects_later_callback_pollution(
    pollution: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact = resolution(replicates=2)
    install_scripted_rng(monkeypatch, (1, 2))
    captured_outcomes: list[object] = []
    original_post_init = calibration_v2.ReplicateOutcome.__post_init__
    original_sample = _BoundCircularShiftV2.sample_token
    original_evaluate = _BoundLaggedPearsonAdapter.evaluate_all
    evaluate_calls = 0
    sample_calls = 0
    raw_vectors: list[tuple[StatisticResult, ...]] = []

    def capture_outcome(self: object) -> None:
        original_post_init(self)  # type: ignore[arg-type]
        captured_outcomes.append(self)

    def maybe_failed_evaluate(
        self: _BoundLaggedPearsonAdapter,
        supplied: SeriesPair,
        /,
    ) -> tuple[StatisticResult, ...]:
        nonlocal evaluate_calls
        evaluate_calls += 1
        if (
            pollution
            in {
                "raw_result",
                "analytic_failure_stage",
                "analytic_selection",
                "analytic_raw_identity",
            }
            and evaluate_calls == 2
        ):
            results = analytic_failure_vector(self)
        else:
            results = original_evaluate(self, supplied)
        if evaluate_calls > 1:
            raw_vectors.append(results)
        return results

    def polluting_sample(
        self: _BoundCircularShiftV2,
        random: object,
        /,
    ) -> NullTransformToken:
        nonlocal sample_calls
        sample_calls += 1
        if sample_calls == 2:
            previous = captured_outcomes[0]
            if pollution == "selection_decision":
                object.__setattr__(previous.selection, "decision_statistic", -1.0)
            elif pollution == "scored_score":
                object.__setattr__(
                    previous.statistic_results[0],
                    "selection_score",
                    999.0,
                )
            elif pollution == "raw_result":
                object.__setattr__(
                    previous.statistic_results[0],
                    "diagnostics",
                    ("polluted_raw",),
                )
            elif pollution == "token":
                object.__setattr__(previous.transform_token.state, "shift", 3)
            elif pollution == "replicate_id":
                object.__setattr__(previous, "replicate_id", 99)
            elif pollution == "seed":
                object.__setattr__(previous, "seed_digest_sha256", "f" * 64)
            elif pollution == "diagnostics":
                object.__setattr__(previous, "diagnostics", ("polluted",))
            elif pollution == "container_type":
                object.__setattr__(
                    previous,
                    "statistic_results",
                    OutcomeTupleSubclass(previous.statistic_results),
                )
            elif pollution == "status_type":
                object.__setattr__(previous, "status", "complete")
            elif pollution == "failure_stage_type":
                object.__setattr__(previous, "failure_stage", "statistic_scan")
            elif pollution == "diagnostics_item_type":
                object.__setattr__(previous, "diagnostics", (1,))
            elif pollution == "complete_failure_stage":
                object.__setattr__(
                    previous,
                    "failure_stage",
                    ReplicateFailureStage.STATISTIC_SCAN,
                )
            elif pollution == "analytic_failure_stage":
                object.__setattr__(previous, "failure_stage", None)
            elif pollution == "analytic_selection":
                object.__setattr__(previous, "selection", prepared.observed_selection)
            elif pollution == "analytic_raw_identity":
                object.__setattr__(
                    previous,
                    "statistic_results",
                    tuple(list(previous.statistic_results)),
                )
            elif pollution == "analytic_without_failure":
                object.__setattr__(previous, "status", ReplicateStatus.ANALYTIC_FAILURE)
                object.__setattr__(
                    previous,
                    "failure_stage",
                    ReplicateFailureStage.STATISTIC_SCAN,
                )
                object.__setattr__(previous, "selection", None)
                object.__setattr__(previous, "statistic_results", raw_vectors[0])
            elif pollution == "token_state_type":
                NullTransformToken.__dict__["state"].__set__(
                    previous.transform_token,
                    object(),
                )
            elif pollution == "deleted_outcome_slot":
                calibration_v2.ReplicateOutcome.__dict__["diagnostics"].__delete__(previous)
            else:
                raise AssertionError(f"unknown pollution: {pollution}")
        return original_sample(self, random)  # type: ignore[arg-type]

    monkeypatch.setattr(
        _BoundLaggedPearsonAdapter,
        "evaluate_all",
        maybe_failed_evaluate,
    )
    monkeypatch.setattr(_BoundCircularShiftV2, "sample_token", polluting_sample)
    prepared = prepare(exact)
    monkeypatch.setattr(
        calibration_v2.ReplicateOutcome,
        "__post_init__",
        capture_outcome,
    )

    with pytest.raises(V2IntegrityError, match=r"replicate|outcome|history|drift"):
        _execute_replicates(prepared, observer=None)
    assert len(captured_outcomes) == 2


def test_history_check_occurs_after_the_last_outcome_constructor_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact = resolution(replicates=2)
    install_scripted_rng(monkeypatch, (1, 2))
    captured_outcomes: list[object] = []
    original_post_init = calibration_v2.ReplicateOutcome.__post_init__
    checks_after_two_outcomes = 0

    def capture_outcome(self: object) -> None:
        nonlocal checks_after_two_outcomes
        original_post_init(self)  # type: ignore[arg-type]
        captured_outcomes.append(self)
        if len(captured_outcomes) == 2:
            checks_after_two_outcomes += 1
            if checks_after_two_outcomes == 1:
                object.__setattr__(
                    captured_outcomes[0].selection,
                    "decision_statistic",
                    -1.0,
                )

    prepared = prepare(exact)
    monkeypatch.setattr(
        calibration_v2.ReplicateOutcome,
        "__post_init__",
        capture_outcome,
    )
    with pytest.raises(V2IntegrityError, match=r"replicate|outcome|history|drift"):
        _execute_replicates(prepared, observer=None)
    assert checks_after_two_outcomes == 1


def test_final_validation_rejects_an_invalid_prepared_array_after_outcome_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact = resolution(replicates=2)
    install_scripted_rng(monkeypatch, (1, 2))
    captured_outcomes: list[object] = []
    original_post_init = calibration_v2.ReplicateOutcome.__post_init__
    checks_after_two_outcomes = 0

    def capture_outcome(self: object) -> None:
        nonlocal checks_after_two_outcomes
        original_post_init(self)  # type: ignore[arg-type]
        captured_outcomes.append(self)
        if len(captured_outcomes) == 2:
            checks_after_two_outcomes += 1
            if checks_after_two_outcomes == 1:
                observed_pair_value = prepared.observed_pair
                SeriesPair.__dict__["source"].__set__(observed_pair_value, object())

    prepared = prepare(exact)
    monkeypatch.setattr(
        calibration_v2.ReplicateOutcome,
        "__post_init__",
        capture_outcome,
    )
    with pytest.raises(V2IntegrityError, match="SeriesPair source must be an exact ndarray"):
        _execute_replicates(prepared, observer=None)
    assert checks_after_two_outcomes == 1


def test_last_outcome_callback_cannot_mutate_frozen_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact = resolution(replicates=2)
    install_scripted_rng(monkeypatch, (1, 2))
    selection_descriptor = SelectionResult.__dict__["decision_statistic"]
    score_descriptor = StatisticResult.__dict__["selection_score"]
    original_post_init = calibration_v2.ReplicateOutcome.__post_init__
    captured_outcomes: list[object] = []
    attack_fired = False

    def capture_outcome(self: object) -> None:
        nonlocal attack_fired
        original_post_init(self)  # type: ignore[arg-type]
        captured_outcomes.append(self)
        if len(captured_outcomes) == 2 and not attack_fired:
            first = captured_outcomes[0]
            selection_descriptor.__set__(first.selection, -1.0)  # type: ignore[attr-defined]
            score_descriptor.__set__(first.statistic_results[0], 999.0)  # type: ignore[attr-defined]
            attack_fired = True

    prepared = prepare(exact)
    monkeypatch.setattr(
        calibration_v2.ReplicateOutcome,
        "__post_init__",
        capture_outcome,
    )
    with pytest.raises(V2IntegrityError, match=r"history|selection|result|drift"):
        _execute_replicates(prepared, observer=None)
    assert attack_fired


def test_last_outcome_callback_cannot_mutate_a_cached_token_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact = resolution(replicates=1)
    install_scripted_rng(monkeypatch, (1,))
    original_post_init = calibration_v2.ReplicateOutcome.__post_init__
    captured_outcomes: list[object] = []
    attack_fired = False

    def capture_outcome(self: object) -> None:
        nonlocal attack_fired
        original_post_init(self)  # type: ignore[arg-type]
        captured_outcomes.append(self)
        if captured_outcomes and not attack_fired:
            token = captured_outcomes[0].transform_token  # type: ignore[attr-defined]
            object.__setattr__(token.state, "shift", 3)
            attack_fired = True

    prepared = prepare(exact)
    monkeypatch.setattr(
        calibration_v2.ReplicateOutcome,
        "__post_init__",
        capture_outcome,
    )
    with pytest.raises(V2IntegrityError, match=r"token|history|drift"):
        _execute_replicates(prepared, observer=None)
    assert attack_fired


def test_execute_rejects_unowned_or_drifted_prepared_before_rng(
) -> None:
    create_stream = inspect.getclosurevars(_execute_replicates).nonlocals[
        "create_random_stream"
    ]
    rng_calls: list[int] = []

    def profile(frame: object, event: str, arg: object) -> None:
        del arg
        if event == "call" and frame.f_code is create_stream.__code__:  # type: ignore[attr-defined]
            rng_calls.append(1)

    previous_profile = sys.getprofile()
    sys.setprofile(profile)
    try:
        with pytest.raises(V2IntegrityError, match="exact preparer-owned"):
            _execute_replicates(object(), observer=None)  # type: ignore[arg-type]

        prepared = prepare()
        object.__setattr__(prepared, "diagnostics", ("drifted",))
        with pytest.raises(V2IntegrityError, match=r"prepared|preparer|snapshot|association"):
            _execute_replicates(prepared, observer=None)
    finally:
        sys.setprofile(previous_profile)
    assert rng_calls == []


@pytest.mark.parametrize("attack_stage", ["sample", "apply", "evaluate"])
def test_each_replicate_checkpoint_rejects_live_identity_drift(
    attack_stage: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact = resolution(replicates=2)
    install_scripted_rng(monkeypatch, (1, 2))
    original_sample = _BoundCircularShiftV2.sample_token
    original_apply = _BoundCircularShiftV2.apply
    original_evaluate = _BoundLaggedPearsonAdapter.evaluate_all
    evaluate_calls = 0

    def attack_sample(
        self: _BoundCircularShiftV2,
        random: object,
        /,
    ) -> object:
        result = original_sample(self, random)  # type: ignore[arg-type]
        object.__setattr__(exact.plan, "root_seed", 18)
        return result

    def attack_apply(
        self: _BoundCircularShiftV2,
        token: object,
        /,
    ) -> object:
        result = original_apply(self, token)  # type: ignore[arg-type]
        object.__setattr__(exact.plan, "root_seed", 18)
        return result

    def attack_evaluate(
        self: _BoundLaggedPearsonAdapter,
        supplied: SeriesPair,
        /,
    ) -> object:
        nonlocal evaluate_calls
        evaluate_calls += 1
        result = original_evaluate(self, supplied)
        if evaluate_calls == 2:
            object.__setattr__(exact.plan, "root_seed", 18)
        return result

    if attack_stage == "sample":
        monkeypatch.setattr(_BoundCircularShiftV2, "sample_token", attack_sample)
    elif attack_stage == "apply":
        monkeypatch.setattr(_BoundCircularShiftV2, "apply", attack_apply)
    else:
        monkeypatch.setattr(_BoundLaggedPearsonAdapter, "evaluate_all", attack_evaluate)

    prepared = prepare(exact)
    with pytest.raises(V2IntegrityError, match=r"plan|resolution|drift"):
        _execute_replicates(prepared, observer=None)


def test_generic_replicate_runtime_error_propagates_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact = resolution(replicates=3)
    install_scripted_rng(monkeypatch, (1, 2, 3))
    original_evaluate = _BoundLaggedPearsonAdapter.evaluate_all
    evaluate_calls = 0

    def explode_on_first_replicate(
        self: _BoundLaggedPearsonAdapter,
        supplied: SeriesPair,
        /,
    ) -> object:
        nonlocal evaluate_calls
        evaluate_calls += 1
        if evaluate_calls == 2:
            raise RuntimeError("replicate infrastructure exploded")
        return original_evaluate(self, supplied)

    monkeypatch.setattr(
        _BoundLaggedPearsonAdapter,
        "evaluate_all",
        explode_on_first_replicate,
    )
    prepared = prepare(exact)

    with pytest.raises(RuntimeError, match="replicate infrastructure exploded"):
        _execute_replicates(prepared, observer=None)
    assert evaluate_calls == 2


def test_task_8b_executor_remains_internal_and_does_not_compute_a_final_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact = resolution(replicates=2)
    install_scripted_rng(monkeypatch, (1, 2))

    outcomes = _execute_replicates(prepare(exact), observer=None)

    assert type(outcomes) is tuple
    assert all(type(outcome).__name__ == "ReplicateOutcome" for outcome in outcomes)
    assert "_execute_replicates" not in calibration_v2.__all__
    assert calibration_v2.__all__ == [
        "calibrate_selected_family",
        "verify_calibration_result",
    ]
