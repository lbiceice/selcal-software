from __future__ import annotations

import importlib
import operator
import sys
from collections import UserDict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import partial
from types import FunctionType, MappingProxyType, MethodType, ModuleType
from typing import cast

import numpy as np
import pytest
from _random_behavior_graph_v2 import (
    behavior_capture_graph,
    random_behavior_reachability_findings,
)

import selcal.contracts_v2 as contracts_v2
import selcal.nulls.block_shuffle_v2 as block_shuffle_v2
import selcal.nulls.circular_shift_v2 as circular_shift_v2
import selcal.randomness as randomness
from selcal.contracts import SeriesPair
from selcal.contracts_v2 import PlanRequestV2, V2IntegrityError
from selcal.resolution_v2 import PlanResolutionV2, resolve_plan_v2
from selcal.statistics.lagged_pearson import LaggedPearsonAdapter, _BoundLaggedPearsonAdapter

_SAFE_NUMPY_CALLABLE_TERMINALS = (
    np.all,
    np.asarray,
    np.broadcast_to,
    np.concatenate,
    np.diff,
    np.empty_like,
    np.floor,
    np.isfinite,
    np.linspace,
    np.logical_and,
    np.max,
    np.maximum,
    np.min,
    np.minimum,
    np.moveaxis,
    np.ndim,
    np.not_equal,
    np.subtract,
)


def _oracle_module() -> ModuleType:
    try:
        return importlib.import_module("selcal.exact_oracle_v0")
    except ModuleNotFoundError as error:
        pytest.fail(f"Task 9 exact oracle module is absent: {error}")


def _oracle() -> Callable[[SeriesPair, PlanResolutionV2], object]:
    candidate = getattr(_oracle_module(), "exact_state_oracle_v0", None)
    assert callable(candidate)
    return cast(Callable[[SeriesPair, PlanResolutionV2], object], candidate)


@contextmanager
def _trace_canonical_capsule_operations() -> Iterator[list[str]]:
    capsule = randomness._RANDOM_CAPSULE_V2
    operation_codes = {
        tuple.__getitem__(capsule, index).__code__: name
        for index, name in enumerate(
            (
                "create_stream",
                "seed_digest",
                "randbelow",
                "seed_digest_for_replicate",
            )
        )
    }
    calls: list[str] = []

    def profile(frame: object, event: str, arg: object) -> None:
        del arg
        if event == "call" and frame.f_code in operation_codes:  # type: ignore[attr-defined]
            calls.append(operation_codes[frame.f_code])  # type: ignore[attr-defined]

    previous_profile = sys.getprofile()
    sys.setprofile(profile)
    try:
        yield calls
    finally:
        sys.setprofile(previous_profile)


def _resolution(
    *,
    null_name: str,
    null_params: dict[str, int],
) -> PlanResolutionV2:
    return resolve_plan_v2(
        PlanRequestV2(
            candidates=(1, 2),
            statistic_name="lagged_pearson_v1",
            statistic_params={},
            selection_rule="max_upper",
            null_name=null_name,
            null_params=null_params,
            replicates=1,
            alpha=0.05,
            tie_tolerance=1e-12,
            root_seed=17,
        )
    )


def _disable_every_production_transform_path(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _oracle_module()

    def explode(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("exact oracle touched a forbidden production transform path")

    for owner, name in (
        (randomness, "uniform_randbelow"),
        (randomness, "_uniform_randbelow_from_next"),
        (randomness.ReplicateRandomSource, "__init__"),
        (randomness.ReplicateRandomSource, "randbelow"),
        (randomness.ReplicateRandomSource, "_next_raw64"),
        (circular_shift_v2.CircularShiftNullV2, "bind"),
        (circular_shift_v2._BoundCircularShiftV2, "identity_token"),
        (circular_shift_v2._BoundCircularShiftV2, "sample_token"),
        (circular_shift_v2._BoundCircularShiftV2, "apply"),
        (circular_shift_v2, "_apply_state"),
        (block_shuffle_v2.BlockShuffleNullV2, "bind"),
        (block_shuffle_v2._BoundBlockShuffleV2, "identity_token"),
        (block_shuffle_v2._BoundBlockShuffleV2, "sample_token"),
        (block_shuffle_v2._BoundBlockShuffleV2, "apply"),
        (block_shuffle_v2, "_apply_state"),
        (block_shuffle_v2, "_ascending_labels"),
        (contracts_v2.CircularShiftStateV2, "__init__"),
        (contracts_v2.BlockShuffleStateV2, "__init__"),
    ):
        monkeypatch.setattr(owner, name, explode)
    monkeypatch.setattr(
        randomness.ReplicateRandomSource,
        "seed_digest_sha256",
        property(explode),
    )
    monkeypatch.setattr(np.random, "PCG64", explode)

    for forbidden_name in (
        "uniform_randbelow",
        "CircularShiftStateV2",
        "BlockShuffleStateV2",
        "_apply_state",
        "_ascending_labels",
    ):
        monkeypatch.setattr(module, forbidden_name, explode, raising=False)


def test_circular_oracle_survives_all_production_paths_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair = SeriesPair(
        source=np.asarray([0, 3, 1, 2, 4, 5], dtype=np.float64),
        target=np.asarray([0, 1, 3, 2, 5, 4], dtype=np.float64),
    )
    resolution = _resolution(
        null_name="circular_shift_v2",
        null_params={"min_shift": 1},
    )
    _disable_every_production_transform_path(monkeypatch)

    with _trace_canonical_capsule_operations() as random_calls:
        result = _oracle()(pair, resolution)

    assert random_calls == []
    assert result.total_state_count == 6
    assert result.nonidentity_exceedance_count == 2
    assert result.p_exact == 0.5


def test_block_oracle_survives_all_production_paths_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair = SeriesPair(
        source=np.asarray([0, 1, 0, 1, 2, 3], dtype=np.float64),
        target=np.asarray([0, 2, 1, 4, 3, 5], dtype=np.float64),
    )
    resolution = _resolution(
        null_name="block_shuffle_v2",
        null_params={"block_length": 2},
    )
    _disable_every_production_transform_path(monkeypatch)

    with _trace_canonical_capsule_operations() as random_calls:
        result = _oracle()(pair, resolution)

    assert random_calls == []
    assert result.total_state_count == 6
    assert 0 <= result.nonidentity_exceedance_count <= 5
    assert 0.0 < result.p_exact <= 1.0


def test_exact_oracle_entrypoint_has_no_reachable_random_behavior_surface() -> None:
    capsule = randomness._RANDOM_CAPSULE_V2

    assert random_behavior_reachability_findings(
        _oracle(),
        capsule_type=randomness._RandomCapsuleV2,
        canonical_capsule=capsule,
        public_random_source_type=randomness.ReplicateRandomSource,
        pcg64_type=np.random.PCG64,
        pcg64_raw_descriptor=np.random.PCG64.random_raw,
        numpy_random_modules=(np.random,),
        numpy_array_type=np.ndarray,
        safe_terminal_identities=_SAFE_NUMPY_CALLABLE_TERMINALS,
    ) == ()


def test_random_behavior_graph_rejects_an_aliased_operation_in_a_default_tuple() -> None:
    capsule = randomness._RANDOM_CAPSULE_V2
    hidden_default = (tuple.__getitem__(capsule, 2),)

    def disguised_entrypoint(
        payload: object = hidden_default,
    ) -> object:
        return payload

    findings = random_behavior_reachability_findings(
        disguised_entrypoint,
        capsule_type=randomness._RandomCapsuleV2,
        canonical_capsule=capsule,
        public_random_source_type=randomness.ReplicateRandomSource,
        pcg64_type=np.random.PCG64,
        pcg64_raw_descriptor=np.random.PCG64.random_raw,
        numpy_random_modules=(np.random,),
        numpy_array_type=np.ndarray,
        safe_terminal_identities=_SAFE_NUMPY_CALLABLE_TERMINALS,
    )

    assert any("forbidden-identity" in finding for finding in findings)
    assert any("random-capsule-operation:randbelow" in finding for finding in findings)


def _random_behavior_findings(root: Callable[..., object]) -> tuple[str, ...]:
    capsule = randomness._RANDOM_CAPSULE_V2
    return random_behavior_reachability_findings(
        root,
        capsule_type=randomness._RandomCapsuleV2,
        canonical_capsule=capsule,
        public_random_source_type=randomness.ReplicateRandomSource,
        pcg64_type=np.random.PCG64,
        pcg64_raw_descriptor=np.random.PCG64.random_raw,
        numpy_random_modules=(np.random,),
        numpy_array_type=np.ndarray,
        safe_terminal_identities=_SAFE_NUMPY_CALLABLE_TERMINALS,
    )


def test_random_behavior_graph_rejects_partial_with_canonical_operation_and_stream() -> None:
    capsule = randomness._RANDOM_CAPSULE_V2
    stream = tuple.__getitem__(capsule, 0)("0" * 64, 0, 1)
    hidden = partial(tuple.__getitem__(capsule, 2), stream)

    findings = _random_behavior_findings(hidden)

    assert any("partial.func" in finding for finding in findings)
    assert any("pcg64-instance" in finding for finding in findings)


def test_random_behavior_graph_rejects_callable_instance_state_holding_operation() -> None:
    operation = tuple.__getitem__(randomness._RANDOM_CAPSULE_V2, 2)

    class CallableHolder:
        def __init__(self, retained_operation: object) -> None:
            self.retained_operation = retained_operation

        def __call__(self, _bound: int, /) -> int:
            return 0

    findings = _random_behavior_findings(CallableHolder(operation))

    assert any("state[retained_operation]" in finding for finding in findings)
    assert any("random-capsule-operation" in finding for finding in findings)


def test_random_behavior_graph_rejects_bound_method_owner_state_holding_operation() -> None:
    operation = tuple.__getitem__(randomness._RANDOM_CAPSULE_V2, 2)

    class MethodOwner:
        def __init__(self, retained_operation: object) -> None:
            self.retained_operation = retained_operation

        def invoke(self, _bound: int, /) -> int:
            return 0

    owner = MethodOwner(operation)
    hidden = MethodType(MethodOwner.invoke, owner)

    findings = _random_behavior_findings(hidden)

    assert any("method.self.state[retained_operation]" in finding for finding in findings)
    assert any("random-capsule-operation" in finding for finding in findings)


@pytest.mark.parametrize("wrapped", (False, True))
def test_random_behavior_graph_rejects_bound_pcg64_raw_provider(wrapped: bool) -> None:
    bound_raw = np.random.PCG64(7).random_raw
    hidden: Callable[..., object] = partial(bound_raw) if wrapped else bound_raw

    findings = _random_behavior_findings(hidden)

    assert any("pcg64-instance" in finding for finding in findings)
    assert any("forbidden-identity" in finding for finding in findings)


def test_random_behavior_graph_rejects_cloned_same_factory_operation_code() -> None:
    capsule = randomness._RANDOM_CAPSULE_V2
    canonical_randbelow = tuple.__getitem__(capsule, 2)
    cloned_randbelow = FunctionType(
        canonical_randbelow.__code__,
        canonical_randbelow.__globals__,
        name="alternate_random_operation",
        argdefs=canonical_randbelow.__defaults__,
        closure=canonical_randbelow.__closure__,
    )
    alternate_capsule = randomness._RandomCapsuleV2(
        create_stream=tuple.__getitem__(capsule, 0),
        seed_digest=tuple.__getitem__(capsule, 1),
        randbelow=cloned_randbelow,
        seed_digest_for_replicate=tuple.__getitem__(capsule, 3),
        external_leaves=(),
    )
    hidden = partial(tuple.__getitem__(alternate_capsule, 2), object())

    findings = _random_behavior_findings(hidden)

    assert any("random-capsule-operation-code:randbelow" in finding for finding in findings)


@pytest.mark.parametrize(
    "hidden_container",
    (
        [tuple.__getitem__(randomness._RANDOM_CAPSULE_V2, 2)],
        {"operation": tuple.__getitem__(randomness._RANDOM_CAPSULE_V2, 2)},
        {tuple.__getitem__(randomness._RANDOM_CAPSULE_V2, 2): "operation-key"},
        {tuple.__getitem__(randomness._RANDOM_CAPSULE_V2, 2)},
        frozenset({tuple.__getitem__(randomness._RANDOM_CAPSULE_V2, 2)}),
        MappingProxyType(
            {"operation": tuple.__getitem__(randomness._RANDOM_CAPSULE_V2, 2)}
        ),
        {
            "nested": [
                {
                    "stream": tuple.__getitem__(randomness._RANDOM_CAPSULE_V2, 0)(
                        "0" * 64,
                        0,
                        1,
                    )
                }
            ]
        },
    ),
)
def test_random_behavior_graph_rejects_operations_or_streams_in_exact_containers(
    hidden_container: object,
) -> None:
    def disguised_entrypoint(payload: object = hidden_container) -> object:
        return payload

    findings = _random_behavior_findings(disguised_entrypoint)

    assert findings
    assert any(
        "random-capsule-operation" in finding or "pcg64-instance" in finding
        for finding in findings
    )


def test_random_behavior_graph_does_not_treat_harmless_symbol_text_as_provider() -> None:
    def harmless_entrypoint() -> str:
        return "randbelow"

    assert _random_behavior_findings(harmless_entrypoint) == ()


def test_random_behavior_graph_does_not_trust_a_spoofed_numpy_random_module_name() -> None:
    harmless_type = type(
        "HarmlessCarrier",
        (),
        {"__module__": "numpy.random.fake"},
    )
    harmless = harmless_type()

    def harmless_entrypoint(payload: object = harmless) -> str:
        del payload
        return "not a provider"

    assert _random_behavior_findings(harmless_entrypoint) == ()


def test_random_behavior_graph_rejects_a_spoofed_numpy_ufunc_holding_random_behavior() -> None:
    capsule = randomness._RANDOM_CAPSULE_V2
    stream = tuple.__getitem__(capsule, 0)("0" * 64, 0, 1)
    hidden_operation = partial(tuple.__getitem__(capsule, 2), stream)
    fake_ufunc_type = type(
        "ufunc",
        (),
        {
            "__module__": "numpy",
            "__call__": staticmethod(hidden_operation),
        },
    )
    hidden = fake_ufunc_type()

    assert hidden(2) in {0, 1}
    assert _random_behavior_findings(hidden)


def test_random_behavior_graph_rejects_a_real_function_local_numpy_random_import() -> None:
    def hidden_provider() -> object:
        import numpy.random as random_provider

        return random_provider.PCG64

    findings = _random_behavior_findings(hidden_provider)

    assert any("numpy-random-provider" in finding for finding in findings)


def test_random_behavior_graph_resolves_exact_getattr_of_the_canonical_capsule() -> None:
    def hidden_provider() -> object:
        return getattr(randomness, "_RANDOM_CAPSULE_V2")  # noqa: B009 - adversarial call

    findings = _random_behavior_findings(hidden_provider)

    assert any("random-capsule-instance" in finding for finding in findings)


def test_random_behavior_graph_rejects_custom_metaclass_call_behavior() -> None:
    capsule = randomness._RANDOM_CAPSULE_V2
    operation = tuple.__getitem__(capsule, 2)

    class RandomMeta(type):
        def __call__(cls) -> object:
            return cls.retained_operation

    class HiddenRandomProvider(metaclass=RandomMeta):
        retained_operation = staticmethod(operation)

    assert HiddenRandomProvider() is operation
    assert _random_behavior_findings(HiddenRandomProvider)


def test_random_behavior_graph_rejects_callable_instance_class_attribute_state() -> None:
    capsule = randomness._RANDOM_CAPSULE_V2
    operation = tuple.__getitem__(capsule, 2)

    class HiddenRandomProvider:
        retained_operation = staticmethod(operation)

        def __call__(self) -> object:
            return self.retained_operation

    hidden = HiddenRandomProvider()

    assert hidden() is operation
    assert _random_behavior_findings(hidden)


def test_random_behavior_graph_rejects_callable_property_provider() -> None:
    capsule = randomness._RANDOM_CAPSULE_V2
    operation = tuple.__getitem__(capsule, 2)

    class HiddenRandomProvider:
        @property
        def retained_operation(self) -> Callable[[object, int], int]:
            return operation

        def __call__(self) -> object:
            return self.retained_operation

    hidden = HiddenRandomProvider()

    assert hidden() is operation
    assert _random_behavior_findings(hidden)


def test_random_behavior_graph_rejects_user_dict_holding_bound_pcg64_raw() -> None:
    bound_raw = np.random.PCG64(7).random_raw
    carrier = UserDict({"raw": bound_raw})

    def hidden_provider(payload: UserDict[str, object] = carrier) -> object:
        return payload["raw"]

    findings = _random_behavior_findings(hidden_provider)

    assert any("pcg64-instance" in finding for finding in findings)


def test_random_behavior_graph_rejects_object_array_holding_bound_pcg64_raw() -> None:
    bound_raw = np.random.PCG64(7).random_raw
    carrier = np.empty(1, dtype=object)
    carrier[0] = bound_raw

    def hidden_provider(payload: np.ndarray[tuple[int], np.dtype[object]] = carrier) -> object:
        return payload[0]

    findings = _random_behavior_findings(hidden_provider)

    assert any("pcg64-instance" in finding for finding in findings)


def test_random_behavior_graph_rejects_frompyfunc_holding_bound_pcg64_raw() -> None:
    hidden_provider = np.frompyfunc(np.random.PCG64(7).random_raw, 0, 1)

    findings = _random_behavior_findings(hidden_provider)

    assert any("opaque" in finding for finding in findings)


def test_random_behavior_graph_rejects_builtin_import_resolution() -> None:
    def hidden_provider() -> object:
        provider = __import__("numpy.random._pcg64", fromlist=("PCG64",))
        return provider.PCG64(7).random_raw()

    findings = _random_behavior_findings(hidden_provider)

    assert any("dynamic-resolver" in finding for finding in findings)


@pytest.mark.parametrize("resolver_name", ("eval", "exec"))
def test_random_behavior_graph_fails_closed_for_eval_and_exec(resolver_name: str) -> None:
    if resolver_name == "eval":

        def hidden_provider() -> object:
            return eval("randomness._RANDOM_CAPSULE_V2")

    else:

        def hidden_provider() -> object:
            exec("retained = randomness._RANDOM_CAPSULE_V2")
            return None

    findings = _random_behavior_findings(hidden_provider)

    assert any("dynamic-resolver" in finding for finding in findings)


@pytest.mark.parametrize(
    ("root", "limits", "match"),
    (
        ([object(), object()], {"node_limit": 1}, "node bound"),
        ([object(), object()], {"edge_limit": 1}, "edge bound"),
        ([[[object()]]], {"depth_limit": 1}, "depth bound"),
    ),
)
def test_behavior_capture_graph_fails_closed_at_every_graph_bound(
    root: object,
    limits: dict[str, int],
    match: str,
) -> None:
    with pytest.raises(AssertionError, match=match):
        behavior_capture_graph(root, **limits)


def test_random_behavior_graph_resolves_an_actual_numpy_random_attribute_chain() -> None:
    def hidden_provider() -> object:
        return np.random.PCG64

    findings = _random_behavior_findings(hidden_provider)

    assert any("provider[np.random]" in finding for finding in findings)
    assert any("provider[np.random.PCG64]" in finding for finding in findings)
    assert any("numpy-random-provider" in finding for finding in findings)


@pytest.mark.parametrize(
    "public_surface",
    (
        randomness.ReplicateRandomSource,
        randomness.ReplicateRandomSource.__init__,
        randomness.ReplicateRandomSource._next_raw64,
        randomness.ReplicateRandomSource.randbelow,
        randomness.ReplicateRandomSource.seed_digest_sha256,
    ),
)
def test_random_behavior_graph_rejects_exact_public_random_source_surfaces(
    public_surface: object,
) -> None:
    def hidden_surface(payload: object = public_surface) -> object:
        return payload

    assert any(
        "forbidden-identity" in finding
        for finding in _random_behavior_findings(hidden_surface)
    )


def test_random_behavior_graph_fails_closed_for_unknown_opaque_callable() -> None:
    hidden = operator.attrgetter("retained_operation")

    findings = _random_behavior_findings(hidden)

    assert any("opaque-callable" in finding for finding in findings)


@pytest.mark.parametrize(
    "known_safe_callable",
    (len, tuple.__getitem__, int, str, dict),
)
def test_random_behavior_graph_accepts_known_stateless_builtins_and_types(
    known_safe_callable: Callable[..., object],
) -> None:
    assert _random_behavior_findings(known_safe_callable) == ()


def test_exact_oracle_wraps_statistic_evaluation_failure_without_random_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair = SeriesPair(
        source=np.asarray([0, 3, 1, 2, 4, 5], dtype=np.float64),
        target=np.asarray([0, 1, 3, 2, 5, 4], dtype=np.float64),
    )
    resolution = _resolution(
        null_name="circular_shift_v2",
        null_params={"min_shift": 1},
    )

    def fail_evaluation(*_args: object, **_kwargs: object) -> object:
        raise ValueError("scripted foreign evaluation failure")

    monkeypatch.setattr(_BoundLaggedPearsonAdapter, "evaluate_all", fail_evaluation)
    with _trace_canonical_capsule_operations() as random_calls:
        with pytest.raises(V2IntegrityError, match="statistic evaluation failed"):
            _oracle()(pair, resolution)

    assert random_calls == []


def test_exact_oracle_wraps_statistic_binding_failure_without_random_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pair = SeriesPair(
        source=np.asarray([0, 3, 1, 2, 4, 5], dtype=np.float64),
        target=np.asarray([0, 1, 3, 2, 5, 4], dtype=np.float64),
    )
    resolution = _resolution(
        null_name="circular_shift_v2",
        null_params={"min_shift": 1},
    )

    def fail_binding(*_args: object, **_kwargs: object) -> object:
        raise ValueError("scripted foreign binding failure")

    monkeypatch.setattr(LaggedPearsonAdapter, "bind", fail_binding)
    with _trace_canonical_capsule_operations() as random_calls:
        with pytest.raises(V2IntegrityError, match="statistic binding failed"):
            _oracle()(pair, resolution)

    assert random_calls == []
