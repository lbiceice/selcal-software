from __future__ import annotations

import ast
import dis
import importlib.util
import inspect
import os
import subprocess
import sys
from collections.abc import Iterator, Mapping
from dataclasses import FrozenInstanceError, dataclass, field
from pathlib import Path
from types import CodeType, FunctionType, MappingProxyType, ModuleType

import numpy as np
import pytest

import selcal.contracts_v2 as contracts_v2
import selcal.nulls.executable_base as executable_base
from selcal.contracts import JsonValue, SeriesPair
from selcal.contracts_v2 import (
    BlockShuffleStateV2,
    CircularShiftStateV2,
    NullBindStatus,
    NullDisabledReason,
    NullTransformResult,
    NullTransformToken,
    PlanRequestV2,
    PlanVersionError,
    ReplicateFailureStage,
    ResourceLimitError,
    RunFailureStage,
    SelCalV2Error,
    V2IntegrityError,
)
from selcal.nulls.executable_base import (
    BoundNullModel,
    NullBindResult,
    UniformIndexSource,
)
from selcal.randomness import _RandomCapsuleV2
from selcal.resolution_v2 import resolve_plan_v2

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def _recursive_instruction_names(candidate: object) -> set[str]:
    code = getattr(candidate, "__code__", None)
    assert type(code) is CodeType
    pending = [code]
    seen: set[int] = set()
    opnames: set[str] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        opnames.update(instruction.opname for instruction in dis.get_instructions(current))
        pending.extend(
            constant for constant in current.co_consts if type(constant) is CodeType
        )
    return opnames


def _trusted_project_call_graph_findings(
    root: object,
) -> tuple[list[str], list[str], list[str]]:
    """Walk the bounded verifier graph through ordinary reachable Python values."""

    package_root = Path(contracts_v2.__file__).resolve().parent
    pending: list[tuple[str, object, int]] = [
        ("verify_calibration_result", root, 0)
    ]
    seen: set[int] = set()
    edge_count = 0
    runtime_imports: list[str] = []
    behavior_globals: list[str] = []
    runtime_frontier: list[str] = []

    def schedule(edges: Iterator[tuple[str, object, int]]) -> None:
        nonlocal edge_count
        materialized = tuple(edges)
        edge_count += len(materialized)
        assert edge_count <= 4096, "trusted graph exceeded edge bound"
        pending.extend(materialized)

    while pending:
        path, candidate, depth = pending.pop()
        assert depth <= 32, f"trusted graph exceeded depth bound at {path}"
        if id(candidate) in seen:
            continue
        seen.add(id(candidate))
        # The extracted typed-record graph measures 553 nodes / 853 edges with
        # all four retained frontiers. This audit traversal cap is not a
        # production memory allowance; depth and edge caps are unchanged.
        assert len(seen) <= 1024, f"trusted graph exceeded node bound at {path}"
        if isinstance(candidate, tuple):
            named_fields = getattr(type(candidate), "_fields", ())
            if type(candidate) is _RandomCapsuleV2:
                assert type(named_fields) is tuple
                assert named_fields == (
                    "create_stream",
                    "seed_digest",
                    "randbelow",
                    "seed_digest_for_replicate",
                    "external_leaves",
                )
                schedule(
                    iter(
                        (
                            (
                                f"{path}.seed_digest_for_replicate",
                                candidate[3],
                                depth + 1,
                            ),
                        )
                    )
                )
                continue
            if (
                type(named_fields) is tuple
                and type(candidate).__module__ == "selcal.resolution_v2"
            ):
                verifier_fields = {
                    "statistic_result_identity",
                    "result_token_verifier",
                }
                schedule(
                    (f"{path}.{name}", candidate[index], depth + 1)
                    for index, name in enumerate(named_fields)
                    if name in verifier_fields
                )
                continue
            schedule(
                (f"{path}[{index}]", value, depth + 1)
                for index, value in enumerate(candidate)
            )
            continue
        if isinstance(candidate, Mapping):
            schedule(
                (f"{path}.value[{index}]", value, depth + 1)
                for index, value in enumerate(candidate.values())
            )
            continue
        if callable(candidate) and not isinstance(candidate, type):
            call_descriptor = inspect.getattr_static(
                type(candidate),
                "__call__",
                None,
            )
            if type(candidate) is not FunctionType and type(call_descriptor) is FunctionType:
                schedule(
                    iter(
                        (
                            (
                                f"{path}.__call__",
                                call_descriptor,
                                depth + 1,
                            ),
                        )
                    )
                )
        if type(candidate) is not FunctionType:
            continue
        module_name = candidate.__module__
        source_file = inspect.getsourcefile(candidate)
        if (
            not module_name.startswith("selcal.")
            or source_file is None
            or not Path(source_file).resolve().is_relative_to(package_root)
        ):
            continue
        if candidate.__name__ in {
            "_pearson_result_identity",
            "_binned_result_identity",
            "_verify_circular_result_token",
            "_verify_block_result_token",
        }:
            runtime_frontier.append(f"{path}:{candidate.__name__}")

        codes = [candidate.__code__]
        while codes:
            code = codes.pop()
            for instruction in dis.get_instructions(code):
                if instruction.opname in {"IMPORT_NAME", "IMPORT_FROM"}:
                    runtime_imports.append(f"{path}:{instruction.opname}")
                if instruction.opname == "LOAD_GLOBAL":
                    name = str(instruction.argval)
                    behavior_globals.append(f"{path}:{name}")
                    resolved = candidate.__globals__.get(name)
                    if type(resolved) is FunctionType:
                        schedule(
                            iter(((f"{path}.global[{name}]", resolved, depth + 1),))
                        )
            codes.extend(
                constant for constant in code.co_consts if type(constant) is CodeType
            )

        if candidate.__closure__ is not None:
            schedule(
                (f"{path}.closure[{name}]", cell.cell_contents, depth + 1)
                for name, cell in zip(
                    candidate.__code__.co_freevars,
                    candidate.__closure__,
                    strict=True,
                )
            )
        if candidate.__defaults__ is not None:
            schedule(
                iter(((f"{path}.defaults", candidate.__defaults__, depth + 1),))
            )
        if candidate.__kwdefaults__ is not None:
            schedule(
                (f"{path}.kwdefault[{name}]", value, depth + 1)
                for name, value in candidate.__kwdefaults__.items()
            )

    return (
        sorted(runtime_imports),
        sorted(behavior_globals),
        sorted(runtime_frontier),
    )


def test_trusted_verifier_has_no_recursive_runtime_import_instructions() -> None:
    opnames = _recursive_instruction_names(contracts_v2.verify_calibration_result)

    assert "IMPORT_NAME" not in opnames
    assert "IMPORT_FROM" not in opnames


def test_trusted_verifier_graph_still_rejects_node_budget_overflow() -> None:
    oversized = tuple(object() for _ in range(1024))
    with pytest.raises(AssertionError, match="trusted graph exceeded node bound"):
        _trusted_project_call_graph_findings(oversized)


def test_trusted_verifier_reachable_project_graph_has_no_behavior_globals() -> None:
    retained_resolutions = (
        resolve_plan_v2(
            PlanRequestV2(
                candidates=(1, 2),
                statistic_name="lagged_pearson_v1",
                statistic_params={},
                selection_rule="max_upper",
                null_name="circular_shift_v2",
                null_params={"min_shift": 1},
                replicates=1,
                alpha=0.05,
                tie_tolerance=1e-12,
                root_seed=17,
            )
        ),
        resolve_plan_v2(
            PlanRequestV2(
                candidates=(1, 2),
                statistic_name="equal_width_binned_nette_v1",
                statistic_params={"bins": 3},
                selection_rule="max_upper",
                null_name="block_shuffle_v2",
                null_params={"block_length": 2},
                replicates=1,
                alpha=0.05,
                tie_tolerance=1e-12,
                root_seed=17,
            )
        ),
    )
    runtime_imports, behavior_globals, runtime_frontier = (
        _trusted_project_call_graph_findings(
            contracts_v2.verify_calibration_result
        )
    )

    assert retained_resolutions
    assert {
        "_pearson_result_identity",
        "_binned_result_identity",
        "_verify_circular_result_token",
        "_verify_block_result_token",
    } <= {finding.rsplit(":", maxsplit=1)[1] for finding in runtime_frontier}
    assert runtime_imports == []
    assert behavior_globals == []


def _load_isolated_contracts_before_bootstrap() -> ModuleType:
    module_name = f"selcal._contracts_v2_prebootstrap_{id(object())}"
    source = Path(contracts_v2.__file__).resolve()
    specification = importlib.util.spec_from_file_location(module_name, source)
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    try:
        specification.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def test_isolated_prebootstrap_verifier_read_fails_with_typed_integrity_error() -> None:
    isolated = _load_isolated_contracts_before_bootstrap()
    try:
        with pytest.raises(isolated.V2IntegrityError, match="not initialized"):
            _uninitialized_verifier = isolated.verify_calibration_result
    finally:
        sys.modules.pop(isolated.__name__, None)


def test_isolated_verifier_bootstrap_is_exactly_once_and_typed() -> None:
    isolated = _load_isolated_contracts_before_bootstrap()
    try:
        bootstrap = isolated._bootstrap_result_verifier_v2
        leaves = (
            lambda _plan: SHA_A,
            lambda resolution: resolution,
            lambda _digest, _replicate_id, _planned: SHA_B,
        )
        bootstrap(*leaves)
        with pytest.raises(
            isolated.V2IntegrityError,
            match="already bootstrapped",
        ):
            bootstrap(*leaves)
    finally:
        sys.modules.pop(isolated.__name__, None)


def test_isolated_verifier_bootstrap_rejects_invalid_capsule_atomically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    isolated = _load_isolated_contracts_before_bootstrap()
    try:
        monkeypatch.setattr(
            isolated,
            "_freeze_result_verifier_capsule_v2",
            lambda *_leaves: (object(), ()),
        )
        with pytest.raises(isolated.V2IntegrityError, match="capsule"):
            isolated._bootstrap_result_verifier_v2(
                lambda _plan: SHA_A,
                lambda resolution: resolution,
                lambda _digest, _replicate_id, _planned: SHA_B,
            )
        assert isolated._RESULT_VERIFIER_BOOTSTRAPPED_V2 is False
        with pytest.raises(isolated.V2IntegrityError, match="not initialized"):
            _uninitialized_verifier = isolated.verify_calibration_result
    finally:
        sys.modules.pop(isolated.__name__, None)


def test_isolated_verifier_bootstrap_rejects_non_function_capsule_atomically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    isolated = _load_isolated_contracts_before_bootstrap()

    class CallableVerifier:
        def __call__(self, _result: object, _resolution: object, /) -> None:
            return None

    try:
        capsule = isolated._ResultVerifierCapsuleV2(
            verify=CallableVerifier(),
            external_leaves=(),
        )
        monkeypatch.setattr(
            isolated,
            "_freeze_result_verifier_capsule_v2",
            lambda *_leaves: capsule,
        )
        with pytest.raises(isolated.V2IntegrityError, match="capsule"):
            isolated._bootstrap_result_verifier_v2(
                lambda _plan: SHA_A,
                lambda resolution: resolution,
                lambda _digest, _replicate_id, _planned: SHA_B,
            )
        assert isolated._RESULT_VERIFIER_BOOTSTRAPPED_V2 is False
        with pytest.raises(isolated.V2IntegrityError, match="not initialized"):
            _uninitialized_verifier = isolated.verify_calibration_result
    finally:
        sys.modules.pop(isolated.__name__, None)


@pytest.mark.parametrize(
    "first_import",
    ("selcal", "selcal.contracts_v2", "selcal.calibration_v2"),
)
def test_verifier_capsule_bootstraps_in_fresh_process_import_orders(
    tmp_path: Path,
    first_import: str,
) -> None:
    source_root = Path(__file__).resolve().parents[1] / "src"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(source_root)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "\n".join(
                (
                    "import importlib, inspect, pickle, sys",
                    f"importlib.import_module({first_import!r})",
                    "import selcal",
                    "import selcal.calibration_v2 as calibration_v2",
                    "import selcal.contracts_v2 as contracts_v2",
                    "assert sys.modules['selcal'] is selcal",
                    "verifier = contracts_v2.verify_calibration_result",
                    "assert verifier.__module__ == 'selcal.contracts_v2'",
                    "assert verifier.__name__ == 'verify_calibration_result'",
                    "assert verifier.__qualname__ == 'verify_calibration_result'",
                    "signature = inspect.signature(verifier)",
                    "assert str(signature) == "
                    "\"(result: 'object', resolution: 'object', /) -> 'None'\"",
                    "assert tuple(signature.parameters) == ('result', 'resolution')",
                    "assert all(parameter.kind is inspect.Parameter.POSITIONAL_ONLY "
                    "for parameter in signature.parameters.values())",
                    "assert verifier.__annotations__ == "
                    "{'result': 'object', 'resolution': 'object', 'return': 'None'}",
                    "assert verifier.__doc__",
                    "assert verifier.__defaults__ is None",
                    "assert verifier.__kwdefaults__ is None",
                    "assert pickle.loads(pickle.dumps(verifier)) is verifier",
                    "assert selcal.verify_calibration_result is verifier",
                    "assert calibration_v2.verify_calibration_result is verifier",
                    "assert not hasattr(contracts_v2, '_bootstrap_result_verifier_v2')",
                    "assert not hasattr(contracts_v2, '_freeze_result_verifier_capsule_v2')",
                    "assert not hasattr(contracts_v2, '_verify_calibration_result_core')",
                )
            ),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=10,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def pair() -> SeriesPair:
    return SeriesPair(
        source=np.array([1.0, 2.0, 3.0, 4.0]),
        target=np.array([4.0, 3.0, 2.0, 1.0]),
    )


def circular_state(shift: int = 0) -> CircularShiftStateV2:
    return CircularShiftStateV2(
        schema="selcal.circular-shift-state.v2",
        shift=shift,
    )


def circular_token(shift: int = 0) -> NullTransformToken:
    return NullTransformToken(
        schema="selcal.null-transform-token.v2",
        null_name="circular_shift_v2",
        null_parameter_sha256=SHA_A,
        semantic_input_sha256=SHA_B,
        scientific_plan_sha256=SHA_C,
        bound_null_owner_sha256=SHA_D,
        is_identity=shift == 0,
        state=circular_state(shift),
    )


def block_state(order: tuple[int, ...] = (0, 1)) -> BlockShuffleStateV2:
    return BlockShuffleStateV2(
        schema="selcal.block-shuffle-state.v2",
        block_order=order,
    )


def block_token(order: tuple[int, ...] = (0, 1)) -> NullTransformToken:
    return NullTransformToken(
        schema="selcal.null-transform-token.v2",
        null_name="block_shuffle_v2",
        null_parameter_sha256=SHA_A,
        semantic_input_sha256=SHA_B,
        scientific_plan_sha256=SHA_C,
        bound_null_owner_sha256=SHA_D,
        is_identity=order == tuple(range(len(order))),
        state=block_state(order),
    )


class ZeroIndexSource:
    def randbelow(self, bound: int, /) -> int:
        if type(bound) is not int or bound < 1:
            raise ValueError
        return 0


@dataclass(frozen=True, slots=True)
class BoundCircular:
    @property
    def name(self) -> str:
        return "circular_shift_v2"

    @property
    def parameters(self) -> MappingProxyType[str, JsonValue]:
        return MappingProxyType({"min_shift": 1})

    @property
    def observed_length(self) -> int:
        return 4

    @property
    def total_state_count(self) -> int:
        return 4

    def identity_token(self) -> NullTransformToken:
        return circular_token()

    def sample_token(self, random: UniformIndexSource, /) -> NullTransformToken:
        random.randbelow(self.total_state_count)
        return circular_token()

    def snapshot_token(self, token: object, /) -> tuple[object, ...]:
        del token
        return ()

    def apply(self, token: NullTransformToken, /) -> NullTransformResult:
        return NullTransformResult(
            pair=pair(),
            token=token,
            source_changed=False,
            diagnostics=(),
        )


class ExplodingCallable:
    def __call__(self, *_args: object, **_kwargs: object) -> None:
        raise AssertionError("NullBindResult must not execute bound methods")


@dataclass(frozen=True, slots=True)
class SlottedBoundSpoof:
    name: object = "circular_shift_v2"
    parameters: object = field(
        default_factory=lambda: MappingProxyType({"min_shift": 1})
    )
    observed_length: object = 4
    total_state_count: object = 4
    identity_token: object = field(default_factory=ExplodingCallable)
    sample_token: object = field(default_factory=ExplodingCallable)
    snapshot_token: object = field(default_factory=ExplodingCallable)
    apply: object = field(default_factory=ExplodingCallable)


class ImmutableParameterMapping(Mapping[str, JsonValue]):
    __slots__ = ("_items",)

    def __init__(self, items: tuple[tuple[str, JsonValue], ...]) -> None:
        self._items = items

    def __getitem__(self, key: str) -> JsonValue:
        for candidate, value in self._items:
            if candidate == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _value in self._items)

    def __len__(self) -> int:
        return len(self._items)


class PlainStructuralBound:
    def __init__(self, parameters: Mapping[str, JsonValue]) -> None:
        self.name = "circular_shift_v2"
        self.parameters = parameters
        self.observed_length = 4
        self.total_state_count = 4

    def identity_token(self) -> NullTransformToken:
        return circular_token()

    def sample_token(self, random: UniformIndexSource, /) -> NullTransformToken:
        return circular_token()

    def snapshot_token(self, token: object, /) -> tuple[object, ...]:
        del token
        return ()

    def apply(self, token: NullTransformToken, /) -> NullTransformResult:
        return NullTransformResult(pair(), token, False, ())


class BoundCircularWithoutSnapshot:
    name = BoundCircular.name
    parameters = BoundCircular.parameters
    observed_length = BoundCircular.observed_length
    total_state_count = BoundCircular.total_state_count
    identity_token = BoundCircular.identity_token
    sample_token = BoundCircular.sample_token
    apply = BoundCircular.apply


class SyntheticThirdBound(BoundCircular):
    @property
    def name(self) -> str:
        return "synthetic_third_v2"


@dataclass(frozen=True)
class ForeignState:
    schema: str
    shift: int


class DerivedCircularState(CircularShiftStateV2):
    pass


class DerivedSeriesPair(SeriesPair):
    pass


def test_v2_error_taxonomy_is_explicit() -> None:
    assert issubclass(V2IntegrityError, SelCalV2Error)
    assert issubclass(PlanVersionError, SelCalV2Error)
    assert issubclass(ResourceLimitError, SelCalV2Error)
    assert V2IntegrityError is not RuntimeError


def test_v2_enums_have_only_the_approved_values() -> None:
    assert tuple(NullBindStatus) == (
        NullBindStatus.ENABLED,
        NullBindStatus.DISABLED,
    )
    assert tuple(reason.value for reason in NullDisabledReason) == (
        "shift_space_empty",
        "non_divisible_tail",
        "fewer_than_two_blocks",
        "too_many_blocks",
    )
    assert tuple(stage.value for stage in RunFailureStage) == (
        "null_bind",
        "observed_statistic_scan",
        "replicate_execution",
    )
    assert tuple(ReplicateFailureStage) == (ReplicateFailureStage.STATISTIC_SCAN,)


@pytest.mark.parametrize(
    ("schema", "shift"),
    [
        ("wrong", 0),
        ("selcal.circular-shift-state.v2", -1),
        ("selcal.circular-shift-state.v2", True),
        ("selcal.circular-shift-state.v2", np.int64(1)),
    ],
)
def test_circular_state_rejects_non_exact_fields(schema: object, shift: object) -> None:
    with pytest.raises(V2IntegrityError):
        CircularShiftStateV2(schema=schema, shift=shift)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("schema", "order"),
    [
        ("wrong", (0, 1)),
        ("selcal.block-shuffle-state.v2", [0, 1]),
        ("selcal.block-shuffle-state.v2", (0,)),
        ("selcal.block-shuffle-state.v2", (0, True)),
        ("selcal.block-shuffle-state.v2", (np.int64(0), 1)),
        ("selcal.block-shuffle-state.v2", (0, 0)),
        ("selcal.block-shuffle-state.v2", (0, 2)),
    ],
)
def test_block_state_requires_an_exact_labelled_permutation(
    schema: object, order: object
) -> None:
    with pytest.raises(V2IntegrityError):
        BlockShuffleStateV2(schema=schema, block_order=order)  # type: ignore[arg-type]


def test_states_are_immutable_nonhashable_exact_values() -> None:
    first = circular_state(2)
    second = circular_state(2)
    block = block_state((1, 0))

    assert first == second
    assert first != circular_state(1)
    assert block == block_state((1, 0))
    with pytest.raises(FrozenInstanceError):
        first.shift = 3  # type: ignore[misc]
    with pytest.raises(TypeError):
        hash(first)
    with pytest.raises(TypeError):
        hash(block)


@pytest.mark.parametrize(
    ("changes",),
    [
        ({"schema": "wrong"},),
        ({"null_name": "circular_shift_v1"},),
        ({"null_parameter_sha256": "A" * 64},),
        ({"semantic_input_sha256": "a" * 63},),
        ({"scientific_plan_sha256": 1},),
        ({"bound_null_owner_sha256": "g" * 64},),
        ({"is_identity": 1},),
        ({"state": ForeignState("selcal.circular-shift-state.v2", 0)},),
        ({"state": DerivedCircularState("selcal.circular-shift-state.v2", 0)},),
        ({"null_name": "block_shuffle_v2"},),
        ({"is_identity": False},),
    ],
)
def test_null_token_rejects_malformed_or_contradictory_fields(
    changes: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "schema": "selcal.null-transform-token.v2",
        "null_name": "circular_shift_v2",
        "null_parameter_sha256": SHA_A,
        "semantic_input_sha256": SHA_B,
        "scientific_plan_sha256": SHA_C,
        "bound_null_owner_sha256": SHA_D,
        "is_identity": True,
        "state": circular_state(),
    }
    values.update(changes)
    with pytest.raises(V2IntegrityError):
        NullTransformToken(**values)  # type: ignore[arg-type]


def test_block_token_requires_state_name_and_identity_agreement() -> None:
    assert block_token() == block_token()
    assert block_token((1, 0)).is_identity is False

    with pytest.raises(V2IntegrityError):
        NullTransformToken(
            schema="selcal.null-transform-token.v2",
            null_name="block_shuffle_v2",
            null_parameter_sha256=SHA_A,
            semantic_input_sha256=SHA_B,
            scientific_plan_sha256=SHA_C,
            bound_null_owner_sha256=SHA_D,
            is_identity=True,
            state=block_state((1, 0)),
        )


def test_token_is_immutable_nonhashable() -> None:
    token = circular_token()
    with pytest.raises(FrozenInstanceError):
        token.is_identity = False  # type: ignore[misc]
    with pytest.raises(TypeError):
        hash(token)


def test_null_transform_result_is_an_exact_immutable_value() -> None:
    result = NullTransformResult(
        pair=pair(),
        token=circular_token(),
        source_changed=False,
        diagnostics=("sampled_identity",),
    )
    same = NullTransformResult(
        pair=pair(),
        token=circular_token(),
        source_changed=False,
        diagnostics=("sampled_identity",),
    )
    assert result == same
    with pytest.raises(FrozenInstanceError):
        result.source_changed = True  # type: ignore[misc]
    with pytest.raises(TypeError):
        hash(result)


@pytest.mark.parametrize(
    "changes",
    [
        {"pair": DerivedSeriesPair(np.array([1.0]), np.array([1.0]))},
        {"token": object()},
        {"source_changed": np.bool_(False)},
        {"diagnostics": ["x"]},
        {"diagnostics": (1,)},
        {"source_changed": True},
    ],
)
def test_null_transform_result_rejects_non_exact_or_identity_contradictions(
    changes: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "pair": pair(),
        "token": circular_token(),
        "source_changed": False,
        "diagnostics": (),
    }
    values.update(changes)
    with pytest.raises(V2IntegrityError):
        NullTransformResult(**values)  # type: ignore[arg-type]


def test_runtime_protocols_accept_the_approved_structures() -> None:
    assert isinstance(ZeroIndexSource(), UniformIndexSource)
    assert isinstance(BoundCircular(), BoundNullModel)


def test_bound_null_structure_accepts_a_registered_third_null_shape() -> None:
    bound = SyntheticThirdBound()

    assert isinstance(bound, BoundNullModel)
    result = NullBindResult(
        status=NullBindStatus.ENABLED,
        bound=bound,
        disabled_reason=None,
        diagnostics=(),
    )

    assert result.bound is bound


def test_executable_base_does_not_name_concrete_v2_nulls() -> None:
    source = Path(executable_base.__file__).read_text(encoding="utf-8")
    string_literals = {
        node.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Constant) and type(node.value) is str
    }

    assert string_literals.isdisjoint(
        {"circular_shift_v2", "block_shuffle_v2"}
    )


def test_bound_null_protocol_rejects_a_missing_snapshot_token() -> None:
    assert not isinstance(BoundCircularWithoutSnapshot(), BoundNullModel)


def test_enabled_bind_rejects_a_missing_snapshot_token() -> None:
    with pytest.raises(V2IntegrityError, match="structurally valid bound null"):
        NullBindResult(
            status=NullBindStatus.ENABLED,
            bound=BoundCircularWithoutSnapshot(),  # type: ignore[arg-type]
            disabled_reason=None,
            diagnostics=(),
        )


def test_null_bind_result_enforces_the_sum_type() -> None:
    bound = BoundCircular()
    enabled = NullBindResult(
        status=NullBindStatus.ENABLED,
        bound=bound,
        disabled_reason=None,
        diagnostics=(),
    )
    disabled = NullBindResult(
        status=NullBindStatus.DISABLED,
        bound=None,
        disabled_reason=NullDisabledReason.SHIFT_SPACE_EMPTY,
        diagnostics=("2*m > n",),
    )
    assert enabled.bound is bound
    assert disabled.disabled_reason is NullDisabledReason.SHIFT_SPACE_EMPTY
    with pytest.raises(TypeError):
        hash(enabled)


def test_enabled_bind_validates_without_executing_bound_methods() -> None:
    bound = SlottedBoundSpoof()
    result = NullBindResult(
        status=NullBindStatus.ENABLED,
        bound=bound,  # type: ignore[arg-type]
        disabled_reason=None,
        diagnostics=(),
    )
    assert result.bound is bound


@pytest.mark.parametrize(
    "parameters",
    [
        {"nested": {"min_shift": 1}, "labels": ["a", "b"]},
        ImmutableParameterMapping(
            (("min_shift", 1), ("weights", (0.25, 0.75)))
        ),
    ],
)
def test_enabled_bind_accepts_structural_mapping_snapshots(
    parameters: Mapping[str, JsonValue],
) -> None:
    bound = PlainStructuralBound(parameters)
    result = NullBindResult(
        status=NullBindStatus.ENABLED,
        bound=bound,
        disabled_reason=None,
        diagnostics=(),
    )
    assert result.bound is bound


@pytest.mark.parametrize(
    "bound",
    [
        SlottedBoundSpoof(name=1),
        SlottedBoundSpoof(name=""),
        SlottedBoundSpoof(parameters=object()),
        SlottedBoundSpoof(parameters={1: "invalid-key"}),
        SlottedBoundSpoof(parameters={"min_shift": np.int64(1)}),
        SlottedBoundSpoof(parameters={"weight": float("nan")}),
        SlottedBoundSpoof(parameters={"weight": float("inf")}),
        SlottedBoundSpoof(parameters={"nested": object()}),
        SlottedBoundSpoof(observed_length=True),
        SlottedBoundSpoof(observed_length=0),
        SlottedBoundSpoof(observed_length=np.int64(4)),
        SlottedBoundSpoof(total_state_count=True),
        SlottedBoundSpoof(total_state_count=0),
        SlottedBoundSpoof(total_state_count=4.0),
        SlottedBoundSpoof(identity_token=None),
        SlottedBoundSpoof(sample_token=None),
        SlottedBoundSpoof(snapshot_token=None),
        SlottedBoundSpoof(apply=None),
    ],
)
def test_enabled_bind_rejects_structural_spoofs(bound: object) -> None:
    with pytest.raises(V2IntegrityError):
        NullBindResult(
            status=NullBindStatus.ENABLED,
            bound=bound,  # type: ignore[arg-type]
            disabled_reason=None,
            diagnostics=(),
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"status": "enabled"},
        {"bound": None},
        {"disabled_reason": NullDisabledReason.SHIFT_SPACE_EMPTY},
        {"bound": object()},
        {"diagnostics": ["x"]},
        {"diagnostics": (1,)},
    ],
)
def test_enabled_bind_rejects_every_contradiction(changes: dict[str, object]) -> None:
    values: dict[str, object] = {
        "status": NullBindStatus.ENABLED,
        "bound": BoundCircular(),
        "disabled_reason": None,
        "diagnostics": (),
    }
    values.update(changes)
    with pytest.raises(V2IntegrityError):
        NullBindResult(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "changes",
    [
        {"status": "disabled"},
        {"bound": BoundCircular()},
        {"disabled_reason": None},
        {"disabled_reason": "shift_space_empty"},
        {"diagnostics": ["x"]},
    ],
)
def test_disabled_bind_rejects_every_contradiction(changes: dict[str, object]) -> None:
    values: dict[str, object] = {
        "status": NullBindStatus.DISABLED,
        "bound": None,
        "disabled_reason": NullDisabledReason.SHIFT_SPACE_EMPTY,
        "diagnostics": (),
    }
    values.update(changes)
    with pytest.raises(V2IntegrityError):
        NullBindResult(**values)  # type: ignore[arg-type]
