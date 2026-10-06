from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from selcal.calibration_v2 import calibrate_selected_family, verify_calibration_result
from selcal.canonical import semantic_input_sha256
from selcal.canonical_v2 import scientific_plan_v2_sha256
from selcal.contracts import ReplicateStatus, SeriesPair, StatisticResult, Validity
from selcal.contracts_v2 import (
    CircularShiftStateV2,
    NullBindStatus,
    NullDisabledReason,
    PlanRequestV2,
    V2IntegrityError,
)
from selcal.nulls.block_shuffle_v2 import BlockShuffleNullV2
from selcal.nulls.circular_shift_v2 import CircularShiftNullV2, circular_token_sha256
from selcal.randomness import ReplicateRandomSource
from selcal.registry import AdapterRegistry
from selcal.resolution_v2 import PlanResolutionV2, resolve_plan_v2
from selcal.selection_v2 import select_family_v2
from selcal.statistics.binned_nette import BinnedNetTEAdapter
from selcal.statistics.lagged_pearson import LaggedPearsonAdapter

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_ROOT = REPOSITORY_ROOT / "src" / "selcal"
PLAN_SHA = "dc7462598231451b36a94e2586037aa921a8159b2eabebb5fe5c81dcafa48931"
CONCRETE_ADAPTER_NAMES = {
    "equal_width_binned_nette_v1",
    "lagged_pearson_v1",
    "circular_shift_v1",
    "block_shuffle_v1",
    "circular_shift_v2",
    "block_shuffle_v2",
}


def _static_string_value(
    node: ast.AST,
    bindings: dict[str, str],
) -> str | None:
    if isinstance(node, ast.Constant) and type(node.value) is str:
        return node.value
    if isinstance(node, ast.Name):
        return bindings.get(node.id)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _static_string_value(node.left, bindings)
        right = _static_string_value(node.right, bindings)
        if left is not None and right is not None:
            return left + right
    if isinstance(node, ast.JoinedStr):
        pieces: list[str] = []
        for value in node.values:
            target = value.value if isinstance(value, ast.FormattedValue) else value
            piece = _static_string_value(target, bindings)
            if piece is None:
                return None
            pieces.append(piece)
        return "".join(pieces)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "join"
        and len(node.args) == 1
        and not node.keywords
        and isinstance(node.args[0], ast.Tuple | ast.List)
    ):
        separator = _static_string_value(node.func.value, bindings)
        pieces = tuple(
            _static_string_value(element, bindings) for element in node.args[0].elts
        )
        if separator is not None and all(piece is not None for piece in pieces):
            return separator.join(cast(tuple[str, ...], pieces))
    return None


def _assignment_names(node: ast.Assign | ast.AnnAssign) -> tuple[str, ...]:
    targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
    return tuple(target.id for target in targets if isinstance(target, ast.Name))


def _concrete_dispatch_findings(source: str) -> set[str]:
    """Bounded intra-module audit of static concrete or imported dispatch."""

    tree = ast.parse(source)
    static_strings: dict[str, str] = {}
    concrete_bindings: dict[str, set[str]] = {}
    imported_names: dict[str, tuple[bool, str, str]] = {}
    imported_modules: dict[str, str] = {}
    imported_containers: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                local_name = alias.asname or alias.name.split(".", maxsplit=1)[0]
                imported_modules[local_name] = alias.name
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        module_parts = (node.module or "").casefold().split(".")
        module_name = next(
            (name for name in CONCRETE_ADAPTER_NAMES if name in module_parts),
            None,
        )
        for alias in node.names:
            local_name = alias.asname or alias.name
            concrete_name = next(
                (
                    name
                    for name in CONCRETE_ADAPTER_NAMES
                    if alias.name.casefold() == name or module_name == name
                ),
                None,
            )
            if concrete_name is not None:
                concrete_bindings[local_name] = {concrete_name}
            else:
                imported_names[local_name] = (
                    alias.name.isupper(),
                    node.module or "",
                    alias.name,
                )

    def concrete_values(node: ast.AST) -> set[str]:
        static_value = _static_string_value(node, static_strings)
        findings = (
            {static_value} if static_value in CONCRETE_ADAPTER_NAMES else set()
        )
        if isinstance(node, ast.Name):
            findings.update(concrete_bindings.get(node.id, set()))
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id in imported_modules:
                normalized = node.attr.casefold()
                if normalized in CONCRETE_ADAPTER_NAMES:
                    findings.add(normalized)
        elif isinstance(node, ast.Tuple | ast.List | ast.Set):
            for element in node.elts:
                findings.update(concrete_values(element))
        elif isinstance(node, ast.Dict):
            for key in node.keys:
                if key is not None:
                    findings.update(concrete_values(key))
        return findings

    assignments = tuple(
        node for node in ast.walk(tree) if isinstance(node, ast.Assign | ast.AnnAssign)
    )

    def imported_container_values(node: ast.AST) -> set[str]:
        if isinstance(node, ast.Name):
            if node.id in imported_names:
                return {node.id}
            return set(imported_containers.get(node.id, set()))
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id in imported_modules:
                return {f"{node.value.id}.{node.attr}"}
        values: set[str] = set()
        if isinstance(node, ast.Dict):
            for value in node.values:
                values.update(imported_container_values(value))
        elif isinstance(node, ast.Tuple | ast.List | ast.Set):
            for element in node.elts:
                values.update(imported_container_values(element))
        return values

    for _ in range(len(assignments) + 1):
        changed = False
        for assignment in assignments:
            value = assignment.value
            if value is None:
                continue
            names = _assignment_names(assignment)
            static_value = _static_string_value(value, static_strings)
            values = concrete_values(value)
            opaque_value: tuple[bool, str, str] | None = None
            container_values = imported_container_values(value)
            if isinstance(value, ast.Name):
                opaque_value = imported_names.get(value.id)
            elif (
                isinstance(value, ast.Attribute)
                and isinstance(value.value, ast.Name)
                and value.value.id in imported_modules
            ):
                opaque_value = (
                    value.attr.isupper(),
                    imported_modules[value.value.id],
                    value.attr,
                )
            for name in names:
                if static_value is not None and static_strings.get(name) != static_value:
                    static_strings[name] = static_value
                    changed = True
                if values and concrete_bindings.get(name) != values:
                    concrete_bindings[name] = set(values)
                    changed = True
                if opaque_value is not None and imported_names.get(name) != opaque_value:
                    imported_names[name] = opaque_value
                    changed = True
                if container_values and imported_containers.get(name) != container_values:
                    imported_containers[name] = set(container_values)
                    changed = True
        if not changed:
            break

    findings: set[str] = set()

    def add_context(node: ast.AST, kind: str) -> None:
        findings.update(concrete_values(node))
        if isinstance(node, ast.Name) and node.id in imported_names:
            is_constant, source_module, origin_name = imported_names[node.id]
            dispatch_origin = any(
                marker in source_module.casefold()
                for marker in ("adapter", "catalog", "dispatch", "registry", "route")
            ) or origin_name.strip("_").casefold() in {
                "adapter",
                "choose_adapter",
                "dispatch",
                "factory",
                "resolver",
                "route",
                "router",
                "selector",
            }
            suspicious_call = node.id.casefold() in {
                "dispatch",
                "factory",
                "resolver",
                "route",
                "router",
            }
            if (
                (kind == "dispatch" and dispatch_origin)
                or (kind == "branch" and is_constant)
                or (kind == "call" and (suspicious_call or dispatch_origin))
            ):
                label = "dispatch" if kind in {"dispatch", "call"} else "branch"
                findings.add(f"<imported-{label}:{node.id}>")
        elif isinstance(node, ast.Name) and kind == "dispatch":
            findings.update(
                f"<imported-dispatch:{name}>"
                for name in imported_containers.get(node.id, set())
            )
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id in imported_modules:
                qualified_name = f"{node.value.id}.{node.attr}"
                dispatch_origin = any(
                    marker in imported_modules[node.value.id].casefold()
                    for marker in (
                        "adapter",
                        "catalog",
                        "dispatch",
                        "registry",
                        "resolver",
                        "route",
                    )
                )
                if (kind == "dispatch" and dispatch_origin) or (
                    kind == "branch" and node.attr.isupper()
                ):
                    label = "dispatch" if kind == "dispatch" else "branch"
                    findings.add(f"<imported-{label}:{qualified_name}>")

    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            add_context(node.left, "branch")
            for comparator in node.comparators:
                add_context(comparator, "branch")
        elif isinstance(node, ast.MatchValue):
            add_context(node.value, "branch")
        elif isinstance(node, ast.Subscript):
            add_context(node.value, "dispatch")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "get":
                add_context(node.func.value, "dispatch")
            elif isinstance(node.func, ast.Attribute) and node.func.attr in {
                "startswith",
                "endswith",
            }:
                for argument in node.args:
                    add_context(argument, "branch")
            else:
                add_context(node.func, "call")

    return findings


def _pair() -> SeriesPair:
    return SeriesPair(
        source=np.asarray((0.0, 3.0, 1.0, 2.0, 4.0, 5.0), dtype=np.float64),
        target=np.asarray((0.0, 1.0, 3.0, 2.0, 5.0, 4.0), dtype=np.float64),
    )


def _resolution(
    candidates: tuple[int, ...] = (1, 2, 3),
    *,
    replicates: int = 3,
) -> PlanResolutionV2:
    return resolve_plan_v2(
        PlanRequestV2(
            candidates=candidates,
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


def _stable_artifacts(
    candidates: tuple[int, ...],
    schedule: tuple[int, ...],
) -> dict[int, tuple[str, str, int, str]]:
    pair = _pair()
    resolution = _resolution(candidates, replicates=9)
    plan_hash = scientific_plan_v2_sha256(resolution.plan)
    semantic_hash = semantic_input_sha256(pair, resolution.plan.candidates)
    bind = resolution.adapters.null_model.bind(
        pair,
        semantic_input_sha256=semantic_hash,
        scientific_plan_sha256=plan_hash,
    )
    assert bind.status is NullBindStatus.ENABLED
    assert bind.bound is not None
    artifacts: dict[int, tuple[str, str, int, str]] = {}
    for replicate_id in schedule:
        index_source = ReplicateRandomSource(plan_hash, replicate_id, 9)
        token_source = ReplicateRandomSource(plan_hash, replicate_id, 9)
        token = bind.bound.sample_token(token_source)
        artifacts[replicate_id] = (
            plan_hash,
            index_source.seed_digest_sha256,
            index_source.randbelow(bind.bound.total_state_count),
            circular_token_sha256(token),
        )
    return artifacts


def test_order_and_determinism_matrix_is_narrowly_bound_to_stable_artifacts() -> None:
    candidate_order_left = _stable_artifacts((3, 1, 2), (0, 1, 2))
    candidate_order_right = _stable_artifacts((2, 3, 1), (0, 1, 2))
    worker_order_left = _stable_artifacts((1, 2, 3), (0, 1, 2))
    worker_order_right = _stable_artifacts((1, 2, 3), (2, 0, 1))
    retry = _stable_artifacts((1, 2, 3), (0,))

    statistic_forward = AdapterRegistry(
        (
            ("equal_width_binned_nette_v1", BinnedNetTEAdapter.from_parameters),
            ("lagged_pearson_v1", LaggedPearsonAdapter.from_parameters),
        )
    )
    statistic_reverse = AdapterRegistry(
        reversed(
            (
                ("equal_width_binned_nette_v1", BinnedNetTEAdapter.from_parameters),
                ("lagged_pearson_v1", LaggedPearsonAdapter.from_parameters),
            )
        )
    )
    null_forward = AdapterRegistry(
        (
            ("block_shuffle_v2", BlockShuffleNullV2.from_parameters),
            ("circular_shift_v2", CircularShiftNullV2.from_parameters),
        )
    )
    null_reverse = AdapterRegistry(
        reversed(
            (
                ("block_shuffle_v2", BlockShuffleNullV2.from_parameters),
                ("circular_shift_v2", CircularShiftNullV2.from_parameters),
            )
        )
    )

    assert candidate_order_left == candidate_order_right
    assert worker_order_left == worker_order_right
    assert retry[0] == worker_order_left[0]
    assert statistic_forward.names == statistic_reverse.names
    assert null_forward.names == null_reverse.names
    assert type(statistic_forward.build("lagged_pearson_v1", {})) is type(
        statistic_reverse.build("lagged_pearson_v1", {})
    )
    assert type(null_forward.build("circular_shift_v2", {"min_shift": 1})) is type(
        null_reverse.build("circular_shift_v2", {"min_shift": 1})
    )
    assert {row[0] for row in worker_order_left.values()} == {PLAN_SHA}


def test_independent_process_scheduling_preserves_only_frozen_identities() -> None:
    script = r"""
import json
import sys
import numpy as np
from selcal.canonical import semantic_input_sha256
from selcal.canonical_v2 import scientific_plan_v2_sha256
from selcal.contracts import SeriesPair
from selcal.contracts_v2 import PlanRequestV2
from selcal.nulls.circular_shift_v2 import circular_token_sha256
from selcal.randomness import ReplicateRandomSource
from selcal.resolution_v2 import resolve_plan_v2

candidates = tuple(int(value) for value in sys.argv[1].split(','))
schedule = tuple(int(value) for value in sys.argv[2].split(','))
pair = SeriesPair(
    source=np.asarray((0., 3., 1., 2., 4., 5.), dtype=np.float64),
    target=np.asarray((0., 1., 3., 2., 5., 4.), dtype=np.float64),
)
resolution = resolve_plan_v2(PlanRequestV2(
    candidates=candidates,
    statistic_name='lagged_pearson_v1', statistic_params={},
    selection_rule='max_upper',
    null_name='circular_shift_v2', null_params={'min_shift': 1},
    replicates=9, alpha=0.05, tie_tolerance=1e-12, root_seed=17,
))
plan_hash = scientific_plan_v2_sha256(resolution.plan)
bound = resolution.adapters.null_model.bind(
    pair,
    semantic_input_sha256=semantic_input_sha256(pair, resolution.plan.candidates),
    scientific_plan_sha256=plan_hash,
).bound
assert bound is not None
rows = {}
for rid in schedule:
    index_source = ReplicateRandomSource(plan_hash, rid, 9)
    token_source = ReplicateRandomSource(plan_hash, rid, 9)
    rows[str(rid)] = {
        'plan_hash': plan_hash,
        'seed_digest': index_source.seed_digest_sha256,
        'bounded_index': index_source.randbelow(bound.total_state_count),
        'transform_token': circular_token_sha256(bound.sample_token(token_source)),
    }
print(json.dumps(rows, sort_keys=True, separators=(',', ':')))
"""

    left_environment = os.environ.copy()
    left_environment["PYTHONPATH"] = str(REPOSITORY_ROOT / "src")
    left_environment["PYTHONHASHSEED"] = "1"
    right_environment = os.environ.copy()
    right_environment["PYTHONPATH"] = str(REPOSITORY_ROOT / "src")
    right_environment["PYTHONHASHSEED"] = "987654"

    left = subprocess.run(
        [sys.executable, "-c", script, "3,1,2", "0,1,2"],
        cwd=REPOSITORY_ROOT,
        env=left_environment,
        check=True,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=10,
    )
    right = subprocess.run(
        [sys.executable, "-c", script, "2,3,1", "2,0,1"],
        cwd=REPOSITORY_ROOT,
        env=right_environment,
        check=True,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=10,
    )

    assert json.loads(left.stdout) == json.loads(right.stdout)


@pytest.mark.parametrize("damage", ("duplicate", "missing", "out_of_range", "unordered"))
def test_public_verifier_rejects_noncanonical_replicate_ids(damage: str) -> None:
    resolution = _resolution(replicates=3)
    result = calibrate_selected_family(_pair(), resolution)
    assert all(outcome.status is ReplicateStatus.COMPLETE for outcome in result.replicates)

    if damage == "duplicate":
        object.__setattr__(result.replicates[1], "replicate_id", 0)
    elif damage == "missing":
        object.__setattr__(result, "replicates", result.replicates[:-1])
    elif damage == "out_of_range":
        object.__setattr__(result.replicates[-1], "replicate_id", 3)
    else:
        object.__setattr__(
            result,
            "replicates",
            (result.replicates[1], result.replicates[0], result.replicates[2]),
        )

    with pytest.raises(V2IntegrityError, match=r"replicate|planned|B outcomes"):
        verify_calibration_result(result, resolution)


def test_raw_score_injection_is_rejected_before_central_selection() -> None:
    raw = (
        StatisticResult(
            candidate_id=1,
            estimate=0.25,
            selection_score=0.25,
            support_n=4,
            validity=Validity.VALID,
            diagnostics=(),
            backend_identity="backend",
            preprocessing_identity="preprocessing",
        ),
        StatisticResult(
            candidate_id=2,
            estimate=0.5,
            selection_score=None,
            support_n=4,
            validity=Validity.VALID,
            diagnostics=(),
            backend_identity="backend",
            preprocessing_identity="preprocessing",
        ),
    )

    with pytest.raises(V2IntegrityError, match=r"raw|selection_score"):
        select_family_v2(raw, (1, 2), "max_upper", 1e-12)


@pytest.mark.parametrize("damage", ("token", "backend"))
def test_public_verifier_rejects_token_or_backend_drift(damage: str) -> None:
    resolution = _resolution(replicates=3)
    result = calibrate_selected_family(_pair(), resolution)
    if damage == "token":
        object.__setattr__(
            result.replicates[-1].transform_token,
            "scientific_plan_sha256",
            "f" * 64,
        )
    else:
        object.__setattr__(result.observed_results[0], "backend_identity", "drifted")

    with pytest.raises(V2IntegrityError, match=r"token|plan|backend|identity"):
        verify_calibration_result(result, resolution)


def test_null_transforms_complete_source_before_statistic_common_support() -> None:
    pair = _pair()
    resolution = _resolution(replicates=3)
    plan_hash = scientific_plan_v2_sha256(resolution.plan)
    bind = resolution.adapters.null_model.bind(
        pair,
        semantic_input_sha256=semantic_input_sha256(pair, resolution.plan.candidates),
        scientific_plan_sha256=plan_hash,
    )
    assert bind.bound is not None
    token = bind.bound.sample_token(ReplicateRandomSource(plan_hash, 0, 3))
    transformed = bind.bound.apply(token)
    raw = resolution.adapters.statistic.bind(pair, resolution.plan.candidates).evaluate_all(
        transformed.pair
    )

    assert transformed.pair.source.shape == pair.source.shape == (6,)
    shift = cast(CircularShiftStateV2, token.state).shift
    assert 0 <= shift < pair.source.size
    assert np.array_equal(transformed.pair.source, np.roll(pair.source, shift))
    assert {entry.support_n for entry in raw} == {3}


def test_resource_caps_fire_before_replicate_or_block_state_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import selcal.calibration_v2 as calibration_module

    executed = False

    def allocation_bomb(*args: object, **kwargs: object) -> object:
        nonlocal executed
        del args, kwargs
        executed = True
        raise AssertionError("allocation happened before the cap")

    monkeypatch.setattr(calibration_module, "_execute_replicates", allocation_bomb)
    with pytest.raises(ValueError, match="replicates"):
        PlanRequestV2(
            candidates=(1,),
            statistic_name="lagged_pearson_v1",
            statistic_params={},
            selection_rule="max_upper",
            null_name="circular_shift_v2",
            null_params={"min_shift": 1},
            replicates=1_000_001,
            alpha=0.05,
            tie_tolerance=1e-12,
            root_seed=17,
        )
    assert not executed

    import selcal.nulls.block_shuffle_v2 as block_module

    monkeypatch.setattr(block_module, "_factorial", allocation_bomb)
    monkeypatch.setattr(block_module, "_ascending_labels", allocation_bomb)
    large_pair = SeriesPair(
        source=np.arange(4097, dtype=np.float64),
        target=np.arange(4097, dtype=np.float64)[::-1],
    )
    disabled = BlockShuffleNullV2(block_length=1).bind(
        large_pair,
        semantic_input_sha256="8" * 64,
        scientific_plan_sha256="d" * 64,
    )

    assert disabled.status is NullBindStatus.DISABLED
    assert disabled.disabled_reason is NullDisabledReason.TOO_MANY_BLOCKS
    assert not executed


def test_generic_orchestration_has_no_concrete_adapter_name_branches() -> None:
    generic_modules = (
        "calibration_v2.py",
        "canonical.py",
        "canonical_v2.py",
        "randomness.py",
        "selection.py",
        "selection_v2.py",
        "support.py",
    )

    for module_name in generic_modules:
        source = (PRODUCTION_ROOT / module_name).read_text(encoding="utf-8")
        assert not _concrete_dispatch_findings(source), module_name


def test_concrete_name_guard_catches_match_startswith_and_dict_dispatch() -> None:
    synthetic = """
def route(name):
    match name:
        case 'circular_shift_v2':
            return 1
    if name.startswith('lagged_pearson_v1'):
        return 2
    return {'block_shuffle_v2': 3}.get(name)
"""

    assert _concrete_dispatch_findings(synthetic) == {
        "circular_shift_v2",
        "lagged_pearson_v1",
        "block_shuffle_v2",
    }


@pytest.mark.parametrize(
    ("synthetic", "expected"),
    (
        (
            """
NAME = 'circular_' + 'shift_v2'
def route(name):
    return name == NAME
""",
            {"circular_shift_v2"},
        ),
        (
            """
from adapter_catalog import BLOCK_SHUFFLE_V2
def route(name):
    return name == BLOCK_SHUFFLE_V2
""",
            {"block_shuffle_v2"},
        ),
        (
            """
from adapter_catalog import LAGGED_PEARSON_V1
ROUTES = {LAGGED_PEARSON_V1: object()}
def route(name):
    return ROUTES[name]
""",
            {"lagged_pearson_v1"},
        ),
        (
            """
from adapter_catalog import ACTIVE_NULL_NAME
def route(name):
    return name == ACTIVE_NULL_NAME
""",
            {"<imported-branch:ACTIVE_NULL_NAME>"},
        ),
        (
            """
from adapter_catalog import NULL_ROUTES
def route(name):
    return NULL_ROUTES[name]
""",
            {"<imported-dispatch:NULL_ROUTES>"},
        ),
        (
            """
from adapter_catalog import ACTIVE_NULL_NAME as alias
def route(name):
    return name == alias
""",
            {"<imported-branch:alias>"},
        ),
        (
            """
import adapter_catalog as catalog
def route(name):
    return catalog.NULL_ROUTES[name]
""",
            {"<imported-dispatch:catalog.NULL_ROUTES>"},
        ),
    ),
    ids=(
        "constant_concatenation",
        "imported_constant",
        "indirect_mapping",
        "aliased_imported_branch",
        "opaque_imported_mapping",
        "renamed_imported_constant",
        "module_qualified_imported_mapping",
    ),
)
def test_bounded_concrete_guard_catches_static_imported_and_indirect_dispatch(
    synthetic: str,
    expected: set[str],
) -> None:
    assert _concrete_dispatch_findings(synthetic) == expected


@pytest.mark.parametrize(
    ("synthetic", "expected"),
    (
        (
            """
from adapter_catalog import routes
def route(name):
    return routes[name]
""",
            {"<imported-dispatch:routes>"},
        ),
        (
            """
from adapter_catalog import routes
local_routes = routes
def route(name):
    return local_routes[name]
""",
            {"<imported-dispatch:local_routes>"},
        ),
        (
            """
from adapter_catalog import resolver
def route(name):
    return resolver(name)
""",
            {"<imported-dispatch:resolver>"},
        ),
        (
            """
NAME = f"{'circular'}_shift_v2"
def route(name):
    return name == NAME
""",
            {"circular_shift_v2"},
        ),
        (
            """
NAME = '_'.join(('block', 'shuffle', 'v2'))
def route(name):
    return name == NAME
""",
            {"block_shuffle_v2"},
        ),
        (
            """
NOTE = 'circular_shift_v2'
def describe():
    return NOTE
""",
            set(),
        ),
        (
            """
from adapter_catalog import routes
ROUTES = {'active': routes}
def route(name):
    return ROUTES['active'][name]
""",
            {"<imported-dispatch:routes>"},
        ),
        (
            """
from adapter_catalog import choose_adapter
def route(name):
    return choose_adapter(name)
""",
            {"<imported-dispatch:choose_adapter>"},
        ),
        (
            """
from helpers import resolver as helper
selector = helper
def route(name):
    return selector(name)
""",
            {"<imported-dispatch:selector>"},
        ),
        (
            """
import logging
logger = logging.getLogger(__name__)
def describe():
    logger.info('circular_shift_v2')
""",
            set(),
        ),
    ),
    ids=(
        "lowercase_imported_mapping",
        "local_import_alias",
        "ordinary_dispatch_call",
        "static_fstring",
        "static_join",
        "harmless_non_dispatch_literal",
        "imported_mapping_nested_in_container",
        "ordinary_imported_adapter_chooser",
        "resolver_helper_propagated_alias",
        "harmless_logger_literal",
    ),
)
def test_bounded_intra_module_dispatch_audit_handles_reviewed_static_cases(
    synthetic: str,
    expected: set[str],
) -> None:
    """Bounded intra-module audit; not arbitrary reflection or whole-program flow."""

    assert _concrete_dispatch_findings(synthetic) == expected


def test_concrete_dispatch_audit_states_its_bounded_intra_module_scope() -> None:
    assert "bounded intra-module" in (_concrete_dispatch_findings.__doc__ or "").casefold()
