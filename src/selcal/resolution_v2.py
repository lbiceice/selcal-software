"""Exact resolver-owned association of scientific-plan v2 adapters."""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Callable, Mapping
from dataclasses import InitVar, dataclass
from math import factorial
from types import MappingProxyType, MemberDescriptorType
from typing import Any, NamedTuple, Protocol, cast, runtime_checkable
from weakref import ReferenceType, ref

import numpy as np
from numpy.typing import NDArray

import selcal.contracts_v2 as _contracts_v2_module
from selcal.canonical import canonical_json_bytes
from selcal.canonical_v2 import (
    _RESULT_VERIFIER_PLAN_CANONICAL_BYTES_V2,
    _RESULT_VERIFIER_PLAN_SHA256_V2,
    scientific_plan_v2_payload,
    scientific_plan_v2_sha256,
)
from selcal.contracts import JsonValue, SelectionRule, SeriesPair, canonical_candidates
from selcal.contracts_v2 import (
    _V2_RESOLUTION_SEAL,
    BlockShuffleStateV2,
    CircularShiftStateV2,
    NullTransformToken,
    PlanRequestV2,
    ResolvedScientificPlanV2,
    V2IntegrityError,
)
from selcal.nulls.block_shuffle_v2 import (
    _MAX_BLOCK_COUNT,
    BlockShuffleNullV2,
    _BoundBlockShuffleV2,
    _snapshot_block_token,
    _validate_block_order,
)
from selcal.nulls.circular_shift_exact_v1 import (
    CircularShiftExactNullV1,
    _BoundCircularShiftExactV1,
    _snapshot_exact_token,
)
from selcal.nulls.circular_shift_v2 import (
    CircularShiftNullV2,
    _BoundCircularShiftV2,
    _snapshot_circular_token,
)
from selcal.nulls.executable_base import NullBindResult
from selcal.nulls.owned_transform_v2 import (
    _owned_pair_content_guard_sha256,
)
from selcal.nulls.owned_transform_v2 import owner_digest as _owner_digest
from selcal.nulls.owned_transform_v2 import (
    parameter_digest as _parameter_digest,
)
from selcal.nulls.owned_transform_v2 import require_sha256 as _require_sha256
from selcal.parameters import freeze_exact_json_mapping
from selcal.randomness import _RANDOM_CAPSULE_V2
from selcal.registry import AdapterRegistry
from selcal.statistics.base import StatisticAdapter
from selcal.statistics.binned_nette import (
    _PREPROCESSING_IDENTITY as _BINNED_PREPROCESSING_IDENTITY,
)
from selcal.statistics.binned_nette import (
    BinnedNetTEAdapter,
    _BoundBinnedNetTEAdapter,
    _observed_edges,
)
from selcal.statistics.lagged_pearson import (
    _PREPROCESSING_IDENTITY as _PEARSON_PREPROCESSING_IDENTITY,
)
from selcal.statistics.lagged_pearson import (
    LaggedPearsonAdapter,
    _BoundLaggedPearsonAdapter,
)


@runtime_checkable
class _ExecutableNullV2(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def parameters(self) -> Mapping[str, JsonValue]: ...

    def bind(
        self,
        observed_pair: SeriesPair,
        *,
        semantic_input_sha256: str,
        scientific_plan_sha256: str,
    ) -> NullBindResult: ...


@dataclass(frozen=True, slots=True, eq=False)
class ResolvedAdaptersV2:
    """Unbound runtime adapters verified against one resolved v2 plan."""

    statistic: StatisticAdapter
    null_model: _ExecutableNullV2


@dataclass(frozen=True, slots=True, eq=False, weakref_slot=True)
class PlanResolutionV2:
    """A resolver-sealed exact v2 plan/adapters association."""

    plan: ResolvedScientificPlanV2
    adapters: ResolvedAdaptersV2
    seal: InitVar[object] = None

    def __post_init__(self, seal: object) -> None:
        if seal is not _V2_RESOLUTION_SEAL:
            raise ValueError("PlanResolutionV2 must be created by v2 plan resolution")
        if type(self) is not PlanResolutionV2:
            raise ValueError("resolution result must be an exact PlanResolutionV2")
        _validate_resolution_fields(self)


@dataclass(frozen=True, slots=True)
class _StatisticIdentitySnapshotV2:
    name: str
    parameters: Mapping[str, JsonValue]
    parameters_bytes: bytes
    operational_snapshot: tuple[JsonValue, ...]


@dataclass(frozen=True, slots=True)
class _NullIdentitySnapshotV2:
    name: str
    parameters: Mapping[str, JsonValue]
    parameters_bytes: bytes
    operational_snapshot: tuple[JsonValue, ...]


@dataclass(frozen=True, slots=True, eq=False)
class _FrozenOperationV2:
    """Ordinary descriptor/callable identity snapshot, not an interpreter trust seal."""

    descriptor: object
    callable: Callable[..., object]


@dataclass(frozen=True, slots=True, eq=False)
class _BoundStatisticExecutionSnapshotV2:
    implementation_type: type[object]
    operational_snapshot: tuple[object, ...]
    evaluate_all: _FrozenOperationV2


@dataclass(frozen=True, slots=True, eq=False)
class _BoundNullExecutionSnapshotV2:
    implementation_type: type[object]
    observed_pair: SeriesPair
    retained_token_state_units: int
    operational_snapshot: tuple[object, ...]
    identity_token: _FrozenOperationV2
    sample_token: _FrozenOperationV2
    snapshot_token: _FrozenOperationV2
    apply: _FrozenOperationV2
    enumerate_token: _FrozenOperationV2 | None = None


_StatisticResultIdentityV2 = Callable[[object], tuple[str, str]]
_ResultTokenVerifierV2 = Callable[..., None]


@dataclass(frozen=True, slots=True)
class _AdapterRegistrationEntryV2:
    """Untrusted registry-facing entry retained as an explicit test seam."""

    name: str
    factory: Callable[[object], object]
    implementation_type: type[object]
    statistic_identity_snapshot: Callable[[object], _StatisticIdentitySnapshotV2] | None = None
    null_identity_snapshot: Callable[[object], _NullIdentitySnapshotV2] | None = None
    statistic_result_identity: _StatisticResultIdentityV2 | None = None
    result_token_verifier: _ResultTokenVerifierV2 | None = None
    bound_implementation_type: type[object] | None = None
    bound_statistic_snapshot: (
        Callable[
            [
                object,
                SeriesPair,
                Mapping[str, JsonValue],
                _BoundStatisticExecutionSnapshotV2 | None,
            ],
            _BoundStatisticExecutionSnapshotV2,
        ]
        | None
    ) = None
    bound_null_snapshot: (
        Callable[
            [
                object,
                SeriesPair,
                Mapping[str, JsonValue],
                _BoundNullExecutionSnapshotV2 | None,
            ],
            _BoundNullExecutionSnapshotV2,
        ]
        | None
    ) = None


class _AdapterRegistrationV2(NamedTuple):
    """Immutable module-initialization trust record for one registry entry."""

    entry: _AdapterRegistrationEntryV2
    name: str
    factory: Callable[[object], object]
    implementation_type: type[object]
    statistic_identity_snapshot: Callable[[object], _StatisticIdentitySnapshotV2] | None
    null_identity_snapshot: Callable[[object], _NullIdentitySnapshotV2] | None
    statistic_result_identity: _StatisticResultIdentityV2 | None
    result_token_verifier: _ResultTokenVerifierV2 | None
    bound_implementation_type: type[object] | None
    bound_statistic_snapshot: (
        Callable[
            [
                object,
                SeriesPair,
                Mapping[str, JsonValue],
                _BoundStatisticExecutionSnapshotV2 | None,
            ],
            _BoundStatisticExecutionSnapshotV2,
        ]
        | None
    )
    bound_null_snapshot: (
        Callable[
            [
                object,
                SeriesPair,
                Mapping[str, JsonValue],
                _BoundNullExecutionSnapshotV2 | None,
            ],
            _BoundNullExecutionSnapshotV2,
        ]
        | None
    )


class _AdapterRegistryIdentityV2(NamedTuple):
    registry: AdapterRegistry[object]
    factories_mapping: Mapping[str, Callable[[object], object]]
    factories: tuple[tuple[str, Callable[[object], object]], ...]


_STRUCTURAL_SNAPSHOT_ERRORS = (AttributeError, TypeError, ValueError, OverflowError)
_SNAPSHOT_LOOKUP_ERRORS = (*_STRUCTURAL_SNAPSHOT_ERRORS, KeyError)
_SERIES_PAIR_SOURCE_DESCRIPTOR = inspect.getattr_static(SeriesPair, "source")
_SERIES_PAIR_TARGET_DESCRIPTOR = inspect.getattr_static(SeriesPair, "target")


@dataclass(frozen=True, slots=True)
class _SeriesPairReadView:
    source: NDArray[np.float64]
    target: NDArray[np.float64]


def _freeze_bound_operation(
    source: object,
    *,
    method_name: str,
    subject: str,
    initial: _FrozenOperationV2 | None,
) -> _FrozenOperationV2:
    """Freeze ordinary operation-object identity without claiming ``__code__`` safety."""

    try:
        descriptor = inspect.getattr_static(type(source), method_name)
    except _STRUCTURAL_SNAPSHOT_ERRORS as error:
        raise V2IntegrityError(f"{subject} {method_name} operation snapshot is invalid") from error
    if isinstance(descriptor, property):
        raise V2IntegrityError(f"{subject} {method_name} property is forbidden")
    if initial is not None:
        if descriptor is not initial.descriptor:
            raise V2IntegrityError(f"{subject} {method_name} operation drifted")
        return initial
    try:
        descriptor_get = descriptor.__get__
    except AttributeError:
        candidate = descriptor
    except (TypeError, ValueError, OverflowError) as error:
        raise V2IntegrityError(f"{subject} {method_name} descriptor access is invalid") from error
    else:
        try:
            candidate = descriptor_get(source, type(source))
        except _STRUCTURAL_SNAPSHOT_ERRORS as error:
            raise V2IntegrityError(
                f"{subject} {method_name} descriptor could not be bound"
            ) from error
    if not callable(candidate):
        raise V2IntegrityError(f"{subject} {method_name} must be callable")
    return _FrozenOperationV2(
        descriptor=descriptor,
        callable=cast(Callable[..., object], candidate),
    )


def _snapshot_float_vector(value: object, *, name: str) -> tuple[object, ...]:
    if type(value) is not np.ndarray:
        raise V2IntegrityError(f"{name} must be an exact ndarray")
    vector = value
    shape = tuple(vector.shape)
    dtype = vector.dtype.str
    writable = vector.flags.writeable
    payload = vector.tobytes(order="C")
    return (dtype, shape, writable, payload)


def _exact_series_pair_read_view(value: object) -> _SeriesPairReadView:
    if type(value) is not SeriesPair:
        raise V2IntegrityError("bound null observed pair must be an exact SeriesPair")
    try:
        current_source = inspect.getattr_static(SeriesPair, "source")
        current_target = inspect.getattr_static(SeriesPair, "target")
    except _STRUCTURAL_SNAPSHOT_ERRORS as error:
        raise V2IntegrityError("SeriesPair slot descriptor is invalid") from error
    if (
        current_source is not _SERIES_PAIR_SOURCE_DESCRIPTOR
        or current_target is not _SERIES_PAIR_TARGET_DESCRIPTOR
        or type(current_source) is not MemberDescriptorType
        or type(current_target) is not MemberDescriptorType
    ):
        raise V2IntegrityError("SeriesPair slot descriptor was replaced")
    try:
        source = _SERIES_PAIR_SOURCE_DESCRIPTOR.__get__(value, SeriesPair)
        target = _SERIES_PAIR_TARGET_DESCRIPTOR.__get__(value, SeriesPair)
    except _STRUCTURAL_SNAPSHOT_ERRORS as error:
        raise V2IntegrityError("SeriesPair slot snapshot is invalid") from error
    _snapshot_float_vector(source, name="SeriesPair source")
    _snapshot_float_vector(target, name="SeriesPair target")
    return _SeriesPairReadView(
        source=cast(NDArray[np.float64], source),
        target=cast(NDArray[np.float64], target),
    )


def _snapshot_series_pair_content(value: object) -> tuple[object, ...]:
    pair = _exact_series_pair_read_view(value)
    source = _snapshot_float_vector(pair.source, name="bound null observed source")
    target = _snapshot_float_vector(pair.target, name="bound null observed target")
    return (source, target)


def _read_exact_slot(source: object, name: str, *, subject: str) -> object:
    """Read one original dataclass slot without invoking a replacement descriptor."""

    try:
        descriptor = inspect.getattr_static(type(source), name)
    except _STRUCTURAL_SNAPSHOT_ERRORS as error:
        raise V2IntegrityError(
            f"{subject} slot execution snapshot descriptor is invalid"
        ) from error
    if type(descriptor) is not MemberDescriptorType:
        raise V2IntegrityError(f"{subject} slot execution snapshot descriptor was replaced")
    try:
        return descriptor.__get__(source, type(source))
    except _STRUCTURAL_SNAPSHOT_ERRORS as error:
        raise V2IntegrityError(f"{subject} slot execution snapshot is invalid") from error


def _freeze_adapter_registration(
    entry: object,
    *,
    subject: str,
) -> _AdapterRegistrationV2:
    if type(entry) is not _AdapterRegistrationEntryV2:
        raise V2IntegrityError(f"{subject} must be an exact registry entry")
    exact_entry = entry
    fields = {
        name: _read_exact_slot(exact_entry, name, subject=f"{subject} {name}")
        for name in (
            "name",
            "factory",
            "implementation_type",
            "statistic_identity_snapshot",
            "null_identity_snapshot",
            "statistic_result_identity",
            "result_token_verifier",
            "bound_implementation_type",
            "bound_statistic_snapshot",
            "bound_null_snapshot",
        )
    }
    if (
        type(fields["name"]) is not str
        or not callable(fields["factory"])
        or type(fields["implementation_type"]) is not type
        or (
            fields["statistic_identity_snapshot"] is not None
            and not callable(fields["statistic_identity_snapshot"])
        )
        or (
            fields["null_identity_snapshot"] is not None
            and not callable(fields["null_identity_snapshot"])
        )
        or (
            fields["statistic_result_identity"] is not None
            and not callable(fields["statistic_result_identity"])
        )
        or (
            fields["result_token_verifier"] is not None
            and not callable(fields["result_token_verifier"])
        )
        or (
            fields["bound_implementation_type"] is not None
            and type(fields["bound_implementation_type"]) is not type
        )
        or (
            fields["bound_statistic_snapshot"] is not None
            and not callable(fields["bound_statistic_snapshot"])
        )
        or (
            fields["bound_null_snapshot"] is not None
            and not callable(fields["bound_null_snapshot"])
        )
    ):
        raise V2IntegrityError(f"{subject} metadata is invalid")
    return _AdapterRegistrationV2(
        entry=exact_entry,
        name=fields["name"],
        factory=cast(Callable[[object], object], fields["factory"]),
        implementation_type=cast(type[object], fields["implementation_type"]),
        statistic_identity_snapshot=cast(
            Callable[[object], _StatisticIdentitySnapshotV2] | None,
            fields["statistic_identity_snapshot"],
        ),
        null_identity_snapshot=cast(
            Callable[[object], _NullIdentitySnapshotV2] | None,
            fields["null_identity_snapshot"],
        ),
        statistic_result_identity=cast(
            _StatisticResultIdentityV2 | None,
            fields["statistic_result_identity"],
        ),
        result_token_verifier=cast(
            _ResultTokenVerifierV2 | None,
            fields["result_token_verifier"],
        ),
        bound_implementation_type=cast(
            type[object] | None,
            fields["bound_implementation_type"],
        ),
        bound_statistic_snapshot=cast(
            Callable[
                [
                    object,
                    SeriesPair,
                    Mapping[str, JsonValue],
                    _BoundStatisticExecutionSnapshotV2 | None,
                ],
                _BoundStatisticExecutionSnapshotV2,
            ]
            | None,
            fields["bound_statistic_snapshot"],
        ),
        bound_null_snapshot=cast(
            Callable[
                [
                    object,
                    SeriesPair,
                    Mapping[str, JsonValue],
                    _BoundNullExecutionSnapshotV2 | None,
                ],
                _BoundNullExecutionSnapshotV2,
            ]
            | None,
            fields["bound_null_snapshot"],
        ),
    )


def _freeze_adapter_registry(
    registry: object,
    *,
    subject: str,
) -> _AdapterRegistryIdentityV2:
    if type(registry) is not AdapterRegistry:
        raise V2IntegrityError(f"{subject} must be an exact AdapterRegistry")
    exact_registry = registry
    factories = _read_exact_slot(
        exact_registry,
        "_factories",
        subject=f"{subject} factories",
    )
    if type(factories) is not MappingProxyType:
        raise V2IntegrityError(f"{subject} factories must remain an exact mapping proxy")
    try:
        items = tuple(sorted(factories.items(), key=lambda item: item[0]))
    except _STRUCTURAL_SNAPSHOT_ERRORS as error:
        raise V2IntegrityError(f"{subject} factories snapshot is invalid") from error
    if any(type(name) is not str or not callable(factory) for name, factory in items):
        raise V2IntegrityError(f"{subject} factories snapshot is invalid")
    return _AdapterRegistryIdentityV2(
        registry=exact_registry,
        factories_mapping=cast(
            Mapping[str, Callable[[object], object]],
            factories,
        ),
        factories=cast(
            tuple[tuple[str, Callable[[object], object]], ...],
            items,
        ),
    )


def _exact_mapping_proxy_bytes(value: object, *, subject: str) -> bytes:
    if type(value) is not MappingProxyType:
        raise V2IntegrityError(f"{subject} must remain an exact frozen mapping")
    try:
        frozen = freeze_exact_json_mapping(value, name=subject)
        return canonical_json_bytes(frozen)
    except _STRUCTURAL_SNAPSHOT_ERRORS as error:
        raise V2IntegrityError(f"{subject} is invalid") from error


def _bound_private_identity_snapshot(
    bound: object,
    *,
    descriptor_names: tuple[str, ...],
    subject: str,
) -> tuple[object, ...]:
    """Read immutable implementation state without invoking public getters."""

    try:
        parameters = _read_exact_slot(
            bound,
            "_parameters",
            subject=f"{subject} parameters",
        )
        descriptors = tuple(inspect.getattr_static(type(bound), name) for name in descriptor_names)
    except _STRUCTURAL_SNAPSHOT_ERRORS as error:
        raise V2IntegrityError(f"{subject} private identity snapshot is invalid") from error
    parameters_bytes = _exact_mapping_proxy_bytes(
        parameters,
        subject=f"{subject} private parameters",
    )
    return (parameters_bytes, descriptors)


def _pearson_bound_snapshot(
    bound: object,
    expected_pair: SeriesPair,
    expected_parameters: Mapping[str, JsonValue],
    initial: _BoundStatisticExecutionSnapshotV2 | None,
) -> _BoundStatisticExecutionSnapshotV2:
    exact = cast(_BoundLaggedPearsonAdapter, bound)
    try:
        expected_pair_view = _exact_series_pair_read_view(expected_pair)
        observed_length = _read_exact_slot(
            exact,
            "observed_length",
            subject="bound Pearson observed_length",
        )
        candidates = _read_exact_slot(
            exact,
            "candidates",
            subject="bound Pearson candidates",
        )
        private_identity = _bound_private_identity_snapshot(
            exact,
            descriptor_names=(
                "name",
                "parameters",
                "backend_identity",
                "preprocessing_identity",
            ),
            subject="bound Pearson",
        )
    except _STRUCTURAL_SNAPSHOT_ERRORS as error:
        raise V2IntegrityError("bound Pearson execution snapshot is invalid") from error
    if (
        expected_parameters
        or type(observed_length) is not int
        or observed_length != int(expected_pair_view.source.size)
    ):
        raise V2IntegrityError("bound Pearson observed length does not match the binding input")
    return _BoundStatisticExecutionSnapshotV2(
        implementation_type=type(exact),
        operational_snapshot=(observed_length, candidates, private_identity),
        evaluate_all=_freeze_bound_operation(
            exact,
            method_name="evaluate_all",
            subject="bound statistic",
            initial=None if initial is None else initial.evaluate_all,
        ),
    )


def _binned_bound_snapshot(
    bound: object,
    expected_pair: SeriesPair,
    expected_parameters: Mapping[str, JsonValue],
    initial: _BoundStatisticExecutionSnapshotV2 | None,
) -> _BoundStatisticExecutionSnapshotV2:
    exact = cast(_BoundBinnedNetTEAdapter, bound)
    try:
        expected_pair_view = _exact_series_pair_read_view(expected_pair)
        bins = _read_exact_slot(exact, "bins", subject="bound binned bins")
        observed_length = _read_exact_slot(
            exact,
            "observed_length",
            subject="bound binned observed_length",
        )
        candidates = _read_exact_slot(
            exact,
            "candidates",
            subject="bound binned candidates",
        )
        source_edges = _snapshot_float_vector(
            _read_exact_slot(
                exact,
                "source_edges",
                subject="bound binned source_edges",
            ),
            name="bound statistic source_edges",
        )
        target_edges = _snapshot_float_vector(
            _read_exact_slot(
                exact,
                "target_edges",
                subject="bound binned target_edges",
            ),
            name="bound statistic target_edges",
        )
        diagnostics = _read_exact_slot(
            exact,
            "bind_diagnostics",
            subject="bound binned diagnostics",
        )
        private_identity = _bound_private_identity_snapshot(
            exact,
            descriptor_names=(
                "name",
                "parameters",
                "backend_identity",
                "preprocessing_identity",
            ),
            subject="bound binned statistic",
        )
        expected_bins = expected_parameters["bins"]
        if (
            frozenset(expected_parameters) != frozenset({"bins"})
            or type(expected_bins) is not int
            or expected_bins < 2
            or type(bins) is not int
            or bins != expected_bins
            or type(observed_length) is not int
            or type(candidates) is not tuple
            or any(type(candidate) is not int for candidate in candidates)
            or type(diagnostics) is not tuple
            or any(type(entry) is not str for entry in diagnostics)
        ):
            raise V2IntegrityError("bound binned statistic execution snapshot is invalid")
        expected_source_edges, source_diagnostics = _observed_edges(
            expected_pair_view.source,
            bins=bins,
            role="source",
        )
        expected_target_edges, target_diagnostics = _observed_edges(
            expected_pair_view.target,
            bins=bins,
            role="target",
        )
    except _SNAPSHOT_LOOKUP_ERRORS as error:
        raise V2IntegrityError("bound binned statistic execution snapshot is invalid") from error
    if (
        observed_length != int(expected_pair_view.source.size)
        or source_edges
        != _snapshot_float_vector(
            expected_source_edges,
            name="expected statistic source_edges",
        )
        or target_edges
        != _snapshot_float_vector(
            expected_target_edges,
            name="expected statistic target_edges",
        )
        or diagnostics != source_diagnostics + target_diagnostics
    ):
        raise V2IntegrityError("bound binned statistic does not match the observed binding input")
    return _BoundStatisticExecutionSnapshotV2(
        implementation_type=type(exact),
        operational_snapshot=(
            bins,
            observed_length,
            candidates,
            source_edges,
            target_edges,
            diagnostics,
            private_identity,
        ),
        evaluate_all=_freeze_bound_operation(
            exact,
            method_name="evaluate_all",
            subject="bound statistic",
            initial=None if initial is None else initial.evaluate_all,
        ),
    )


def _null_bound_snapshot(
    bound: _BoundCircularShiftV2 | _BoundCircularShiftExactV1 | _BoundBlockShuffleV2,
    *,
    expected_pair: SeriesPair,
    retained_token_state_units: int,
    scalar_fields: tuple[object, ...],
    initial: _BoundNullExecutionSnapshotV2 | None,
) -> _BoundNullExecutionSnapshotV2:
    if type(retained_token_state_units) is not int or retained_token_state_units < 1:
        raise V2IntegrityError("bound null retained token state units are invalid")
    try:
        observed_pair = _read_exact_slot(
            bound,
            "_observed_pair",
            subject="bound null observed_pair",
        )
        pair_content = _snapshot_series_pair_content(observed_pair)
        exact_observed_pair = cast(SeriesPair, observed_pair)
        observed_guard = _read_exact_slot(
            bound,
            "_observed_content_guard_sha256",
            subject="bound null observed guard",
        )
        semantic_digest = _read_exact_slot(
            bound,
            "_semantic_input_sha256",
            subject="bound null semantic digest",
        )
        plan_digest = _read_exact_slot(
            bound,
            "_scientific_plan_sha256",
            subject="bound null plan digest",
        )
        parameter_digest = _read_exact_slot(
            bound,
            "_null_parameter_sha256",
            subject="bound null parameter digest",
        )
        owner = _read_exact_slot(
            bound,
            "_bound_null_owner_sha256",
            subject="bound null owner digest",
        )
        observed_length = _read_exact_slot(
            bound,
            "_observed_length",
            subject="bound null observed_length",
        )
        state_count = _read_exact_slot(
            bound,
            "_total_state_count",
            subject="bound null state_count",
        )
        observed_pair_view = _exact_series_pair_read_view(exact_observed_pair)
        exact_observed_pair_view = SeriesPair(
            source=observed_pair_view.source,
            target=observed_pair_view.target,
        )
        expected_guard = _owned_pair_content_guard_sha256(exact_observed_pair_view)
        private_identity = _bound_private_identity_snapshot(
            bound,
            descriptor_names=(
                "name",
                "parameters",
                "observed_length",
                "total_state_count",
                "null_parameter_sha256",
                "bound_null_owner_sha256",
            ),
            subject="bound null",
        )
    except _STRUCTURAL_SNAPSHOT_ERRORS as error:
        raise V2IntegrityError("bound null execution snapshot is invalid") from error
    if type(observed_guard) is not str or observed_guard != expected_guard:
        raise V2IntegrityError("bound null observed content guard is invalid")
    if pair_content != _snapshot_series_pair_content(expected_pair):
        raise V2IntegrityError("bound null observed pair does not match the binding input")
    return _BoundNullExecutionSnapshotV2(
        implementation_type=type(bound),
        observed_pair=exact_observed_pair,
        retained_token_state_units=retained_token_state_units,
        operational_snapshot=(
            *scalar_fields,
            retained_token_state_units,
            observed_length,
            state_count,
            semantic_digest,
            plan_digest,
            parameter_digest,
            owner,
            observed_guard,
            pair_content,
            private_identity,
        ),
        identity_token=_freeze_bound_operation(
            bound,
            method_name="identity_token",
            subject="bound null",
            initial=None if initial is None else initial.identity_token,
        ),
        sample_token=_freeze_bound_operation(
            bound,
            method_name="sample_token",
            subject="bound null",
            initial=None if initial is None else initial.sample_token,
        ),
        snapshot_token=_freeze_bound_operation(
            bound,
            method_name="snapshot_token",
            subject="bound null",
            initial=None if initial is None else initial.snapshot_token,
        ),
        apply=_freeze_bound_operation(
            bound,
            method_name="apply",
            subject="bound null",
            initial=None if initial is None else initial.apply,
        ),
        enumerate_token=(
            _freeze_bound_operation(
                bound,
                method_name="enumerate_token",
                subject="bound null",
                initial=None if initial is None else initial.enumerate_token,
            )
            if hasattr(type(bound), "enumerate_token")
            else None
        ),
    )


def _circular_bound_snapshot(
    bound: object,
    expected_pair: SeriesPair,
    expected_parameters: Mapping[str, JsonValue],
    initial: _BoundNullExecutionSnapshotV2 | None,
) -> _BoundNullExecutionSnapshotV2:
    exact = cast(_BoundCircularShiftV2, bound)
    try:
        expected_pair_view = _exact_series_pair_read_view(expected_pair)
        min_shift = _read_exact_slot(
            exact,
            "_min_shift",
            subject="bound circular min_shift",
        )
        observed_length = _read_exact_slot(
            exact,
            "_observed_length",
            subject="bound circular observed_length",
        )
        state_count = _read_exact_slot(
            exact,
            "_total_state_count",
            subject="bound circular state_count",
        )
        expected_min_shift = expected_parameters["min_shift"]
    except _SNAPSHOT_LOOKUP_ERRORS as error:
        raise V2IntegrityError("bound circular null execution snapshot is invalid") from error
    if (
        frozenset(expected_parameters) != frozenset({"min_shift"})
        or type(expected_min_shift) is not int
        or expected_min_shift < 1
        or type(min_shift) is not int
        or min_shift != expected_min_shift
        or type(observed_length) is not int
        or observed_length != int(expected_pair_view.source.size)
        or type(state_count) is not int
        or state_count != observed_length - 2 * min_shift + 2
        or state_count < 1
    ):
        raise V2IntegrityError("bound circular null execution snapshot is invalid")
    return _null_bound_snapshot(
        exact,
        expected_pair=expected_pair,
        retained_token_state_units=1,
        scalar_fields=(min_shift,),
        initial=initial,
    )


def _exact_circular_bound_snapshot(
    bound: object,
    expected_pair: SeriesPair,
    expected_parameters: Mapping[str, JsonValue],
    initial: _BoundNullExecutionSnapshotV2 | None,
) -> _BoundNullExecutionSnapshotV2:
    exact = cast(_BoundCircularShiftExactV1, bound)
    try:
        expected_pair_view = _exact_series_pair_read_view(expected_pair)
        min_shift = _read_exact_slot(
            exact, "_min_shift", subject="bound exact circular min_shift"
        )
        observed_length = _read_exact_slot(
            exact, "_observed_length", subject="bound exact circular observed_length"
        )
        state_count = _read_exact_slot(
            exact, "_total_state_count", subject="bound exact circular state_count"
        )
        expected_min_shift = expected_parameters["min_shift"]
    except _SNAPSHOT_LOOKUP_ERRORS as error:
        raise V2IntegrityError(
            "bound exact circular null execution snapshot is invalid"
        ) from error
    if (
        frozenset(expected_parameters) != frozenset({"min_shift"})
        or expected_min_shift != 1
        or type(min_shift) is not int
        or min_shift != 1
        or type(observed_length) is not int
        or observed_length != int(expected_pair_view.source.size)
        or type(state_count) is not int
        or state_count != observed_length
        or state_count < 1
    ):
        raise V2IntegrityError("bound exact circular null execution snapshot is invalid")
    return _null_bound_snapshot(
        exact,
        expected_pair=expected_pair,
        retained_token_state_units=1,
        scalar_fields=(min_shift,),
        initial=initial,
    )


def _block_bound_snapshot(
    bound: object,
    expected_pair: SeriesPair,
    expected_parameters: Mapping[str, JsonValue],
    initial: _BoundNullExecutionSnapshotV2 | None,
) -> _BoundNullExecutionSnapshotV2:
    exact = cast(_BoundBlockShuffleV2, bound)
    try:
        expected_pair_view = _exact_series_pair_read_view(expected_pair)
        block_length = _read_exact_slot(
            exact,
            "_block_length",
            subject="bound block block_length",
        )
        block_count = _read_exact_slot(
            exact,
            "_block_count",
            subject="bound block block_count",
        )
        observed_length = _read_exact_slot(
            exact,
            "_observed_length",
            subject="bound block observed_length",
        )
        state_count = _read_exact_slot(
            exact,
            "_total_state_count",
            subject="bound block state_count",
        )
        expected_block_length = expected_parameters["block_length"]
    except _SNAPSHOT_LOOKUP_ERRORS as error:
        raise V2IntegrityError("bound block null execution snapshot is invalid") from error
    if (
        frozenset(expected_parameters) != frozenset({"block_length"})
        or type(expected_block_length) is not int
        or expected_block_length < 1
        or type(block_length) is not int
        or block_length != expected_block_length
        or type(block_count) is not int
        or block_count < 2
        or block_count > _MAX_BLOCK_COUNT
        or type(observed_length) is not int
        or observed_length != int(expected_pair_view.source.size)
        or observed_length % block_length != 0
        or block_count != observed_length // block_length
        or type(state_count) is not int
        or state_count != factorial(block_count)
    ):
        raise V2IntegrityError("bound block null execution snapshot is invalid")
    return _null_bound_snapshot(
        exact,
        expected_pair=expected_pair,
        retained_token_state_units=block_count,
        scalar_fields=(block_length, block_count),
        initial=initial,
    )


def _statistic_identity_snapshot(
    adapter: BinnedNetTEAdapter | LaggedPearsonAdapter,
    *,
    expected_name: str,
    scalar_name: str | None,
    parameter_name: str | None,
    subject: str,
) -> _StatisticIdentitySnapshotV2:
    try:
        reported_name = adapter.name
        reported_parameters = adapter.parameters
        scalar = None if scalar_name is None else getattr(adapter, scalar_name)
    except V2IntegrityError:
        raise
    except (AttributeError, TypeError, ValueError) as error:
        raise V2IntegrityError(f"{subject} identity snapshot is invalid") from error
    try:
        frozen_parameters = freeze_exact_json_mapping(
            reported_parameters,
            name=f"{expected_name} operational parameters",
        )
    except ValueError as error:
        raise V2IntegrityError(str(error)) from error
    if type(reported_name) is not str or reported_name != expected_name:
        raise V2IntegrityError(f"{subject} operational name is invalid")
    if scalar_name is None:
        if parameter_name is not None or frozen_parameters:
            raise V2IntegrityError(f"{subject} operational parameters are invalid")
        operational_snapshot: tuple[JsonValue, ...] = ()
    else:
        if (
            parameter_name is None
            or type(scalar) is not int
            or scalar < 2
            or frozenset(frozen_parameters) != frozenset({parameter_name})
            or type(frozen_parameters[parameter_name]) is not int
            or frozen_parameters[parameter_name] != scalar
        ):
            raise V2IntegrityError(f"{subject} operational scalar is invalid")
        operational_snapshot = (scalar,)
    return _StatisticIdentitySnapshotV2(
        name=reported_name,
        parameters=frozen_parameters,
        parameters_bytes=canonical_json_bytes(frozen_parameters),
        operational_snapshot=operational_snapshot,
    )


def _pearson_identity_snapshot(adapter: object) -> _StatisticIdentitySnapshotV2:
    if type(adapter) is not LaggedPearsonAdapter:
        raise V2IntegrityError("lagged_pearson_v1 requires its exact registered implementation")
    return _statistic_identity_snapshot(
        adapter,
        expected_name="lagged_pearson_v1",
        scalar_name=None,
        parameter_name=None,
        subject="lagged Pearson statistic",
    )


def _null_operational_snapshot(
    adapter: CircularShiftNullV2 | CircularShiftExactNullV1 | BlockShuffleNullV2,
    *,
    expected_name: str,
    scalar_name: str,
    parameter_name: str,
    subject: str,
) -> _NullIdentitySnapshotV2:
    try:
        reported_name = adapter.name
        reported_parameters = adapter.parameters
        internal_digest = adapter._null_parameter_sha256
        scalar = getattr(adapter, scalar_name)
        frozen_parameters = freeze_exact_json_mapping(
            reported_parameters,
            name=f"{expected_name} operational parameters",
        )
    except V2IntegrityError:
        raise
    except (AttributeError, TypeError, ValueError) as error:
        raise V2IntegrityError(f"{subject} operational identity snapshot is invalid") from error
    if type(reported_name) is not str or reported_name != expected_name:
        raise V2IntegrityError(f"{subject} operational name is invalid")
    if (
        type(scalar) is not int
        or scalar < 1
        or frozenset(frozen_parameters) != frozenset({parameter_name})
        or type(frozen_parameters[parameter_name]) is not int
        or frozen_parameters[parameter_name] != scalar
    ):
        raise V2IntegrityError(f"{subject} operational scalar is invalid")
    exact_digest = _require_sha256(
        internal_digest,
        name="null operational parameter digest",
    )
    expected_digest = _parameter_digest(
        null_name=reported_name,
        params=frozen_parameters,
    )
    if exact_digest != expected_digest:
        raise V2IntegrityError(f"{subject} operational parameter digest is invalid")
    return _NullIdentitySnapshotV2(
        name=reported_name,
        parameters=frozen_parameters,
        parameters_bytes=canonical_json_bytes(frozen_parameters),
        operational_snapshot=(scalar, exact_digest),
    )


def _binned_identity_snapshot(adapter: object) -> _StatisticIdentitySnapshotV2:
    if type(adapter) is not BinnedNetTEAdapter:
        raise V2IntegrityError(
            "equal_width_binned_nette_v1 requires its exact registered implementation"
        )
    return _statistic_identity_snapshot(
        adapter,
        expected_name="equal_width_binned_nette_v1",
        scalar_name="bins",
        parameter_name="bins",
        subject="binned statistic",
    )


def _circular_identity_snapshot(adapter: object) -> _NullIdentitySnapshotV2:
    if type(adapter) is not CircularShiftNullV2:
        raise V2IntegrityError("circular_shift_v2 requires its exact registered implementation")
    return _null_operational_snapshot(
        adapter,
        expected_name="circular_shift_v2",
        scalar_name="min_shift",
        parameter_name="min_shift",
        subject="circular null",
    )


def _exact_circular_identity_snapshot(adapter: object) -> _NullIdentitySnapshotV2:
    if type(adapter) is not CircularShiftExactNullV1:
        raise V2IntegrityError(
            "circular_shift_exact_v1 requires its exact registered implementation"
        )
    return _null_operational_snapshot(
        adapter,
        expected_name="circular_shift_exact_v1",
        scalar_name="min_shift",
        parameter_name="min_shift",
        subject="exact circular null",
    )


def _block_identity_snapshot(adapter: object) -> _NullIdentitySnapshotV2:
    if type(adapter) is not BlockShuffleNullV2:
        raise V2IntegrityError("block_shuffle_v2 requires its exact registered implementation")
    return _null_operational_snapshot(
        adapter,
        expected_name="block_shuffle_v2",
        scalar_name="block_length",
        parameter_name="block_length",
        subject="block null",
    )


def _pearson_result_identity(adapter: object) -> tuple[str, str]:
    """Return the result identities owned by the registered Pearson adapter."""

    _pearson_identity_snapshot(adapter)
    return (
        f"selcal.lagged_pearson.v1|numpy={np.__version__}|units=pearson_correlation",
        _PEARSON_PREPROCESSING_IDENTITY,
    )


def _binned_result_identity(adapter: object) -> tuple[str, str]:
    """Return the result identities owned by the registered binned adapter."""

    snapshot = _binned_identity_snapshot(adapter)
    bins = snapshot.operational_snapshot[0]
    if type(bins) is not int:
        raise V2IntegrityError("registered binned result identity is invalid")
    return (
        f"selcal.equal_width_binned_nette.v1|numpy={np.__version__}|bins={bins}|units=nats",
        _BINNED_PREPROCESSING_IDENTITY,
    )


def _verify_result_token_common(
    token: NullTransformToken,
    *,
    expected_name: str,
    expected_parameters: Mapping[str, JsonValue],
    semantic_input_sha256: str,
    scientific_plan_sha256: str,
    observed_length: int,
) -> None:
    if type(observed_length) is not int or observed_length < 1:
        raise V2IntegrityError("result token observed length is invalid")
    expected_parameter_digest = _parameter_digest(
        null_name=expected_name,
        params=expected_parameters,
    )
    expected_owner_digest = _owner_digest(
        semantic_input_sha256=semantic_input_sha256,
        scientific_plan_sha256=scientific_plan_sha256,
        null_parameter_sha256=expected_parameter_digest,
        observed_length=observed_length,
    )
    if (
        token.null_name != expected_name
        or token.null_parameter_sha256 != expected_parameter_digest
        or token.semantic_input_sha256 != semantic_input_sha256
        or token.scientific_plan_sha256 != scientific_plan_sha256
        or token.bound_null_owner_sha256 != expected_owner_digest
    ):
        raise V2IntegrityError("result token ownership contradicts the sealed resolution")


def _verify_circular_result_token(
    adapter: object,
    token: object,
    semantic_input_sha256: str,
    scientific_plan_sha256: str,
    observed_length: int,
    *,
    replicate_id: int | None = None,
    planned_replicates: int | None = None,
) -> None:
    identity = _circular_identity_snapshot(adapter)
    min_shift = identity.operational_snapshot[0]
    if type(min_shift) is not int or 2 * min_shift > observed_length:
        raise V2IntegrityError("result circular state space is invalid")
    snapshot = _snapshot_circular_token(token)
    _verify_result_token_common(
        snapshot,
        expected_name=identity.name,
        expected_parameters=identity.parameters,
        semantic_input_sha256=semantic_input_sha256,
        scientific_plan_sha256=scientific_plan_sha256,
        observed_length=observed_length,
    )
    state = cast(CircularShiftStateV2, snapshot.state)
    if state.shift != 0 and not (min_shift <= state.shift <= observed_length - min_shift):
        raise V2IntegrityError("result token shift is outside its bound state universe")


def _verify_exact_circular_result_token(
    adapter: object,
    token: object,
    semantic_input_sha256: str,
    scientific_plan_sha256: str,
    observed_length: int,
    *,
    replicate_id: int | None = None,
    planned_replicates: int | None = None,
) -> None:
    identity = _exact_circular_identity_snapshot(adapter)
    if observed_length < 2:
        raise V2IntegrityError("result exact circular state space is invalid")
    snapshot = _snapshot_exact_token(token)
    _verify_result_token_common(
        snapshot,
        expected_name=identity.name,
        expected_parameters=identity.parameters,
        semantic_input_sha256=semantic_input_sha256,
        scientific_plan_sha256=scientific_plan_sha256,
        observed_length=observed_length,
    )
    state = cast(CircularShiftStateV2, snapshot.state)
    if not 0 <= state.shift < observed_length:
        raise V2IntegrityError("result token shift is outside its enumerated state universe")
    # Exact enumeration is a structural claim checkable from tokens alone: B = n - 1 and
    # replicate i carries shift i + 1 (no identity, no repeat, no gap, fixed order).
    if (
        type(replicate_id) is not int
        or type(planned_replicates) is not int
        or planned_replicates != observed_length - 1
        or state.shift != replicate_id + 1
    ):
        raise V2IntegrityError("result replicates do not enumerate the complete circular group")


def _verify_block_result_token(
    adapter: object,
    token: object,
    semantic_input_sha256: str,
    scientific_plan_sha256: str,
    observed_length: int,
    *,
    replicate_id: int | None = None,
    planned_replicates: int | None = None,
) -> None:
    identity = _block_identity_snapshot(adapter)
    block_length = identity.operational_snapshot[0]
    if (
        type(block_length) is not int
        or observed_length % block_length != 0
        or not 2 <= observed_length // block_length <= _MAX_BLOCK_COUNT
    ):
        raise V2IntegrityError("result block state space is invalid")
    snapshot = _snapshot_block_token(token)
    _verify_result_token_common(
        snapshot,
        expected_name=identity.name,
        expected_parameters=identity.parameters,
        semantic_input_sha256=semantic_input_sha256,
        scientific_plan_sha256=scientific_plan_sha256,
        observed_length=observed_length,
    )
    state = cast(BlockShuffleStateV2, snapshot.state)
    _validate_block_order(
        state.block_order,
        expected_count=observed_length // block_length,
    )


def _freeze_result_verifier_science_leaves_v2() -> (
    tuple[
        Mapping[type[object], Callable[[object], tuple[object, ...]]],
        Mapping[type[object], _StatisticResultIdentityV2],
        Mapping[type[object], Callable[[object], tuple[object, ...]]],
        Mapping[type[object], _ResultTokenVerifierV2],
    ]
):
    """Freeze adapter identity and token operations for result verification."""

    exact_type = type
    tuple_type = tuple
    integer_type = int
    string_type = str
    boolean_type = bool
    length = len
    any_value = any
    set_type = set
    range_value = range
    str_value = str
    member_descriptor_get = MemberDescriptorType.__get__
    mapping_proxy_type = MappingProxyType
    integrity_error_type = V2IntegrityError
    structural_errors = (AttributeError, TypeError, ValueError, OverflowError)
    sha256 = hashlib.sha256
    json_dumps = json.dumps
    numpy_version = str(np.__version__)
    pearson_preprocessing = _PEARSON_PREPROCESSING_IDENTITY
    binned_preprocessing = _BINNED_PREPROCESSING_IDENTITY
    max_block_count = _MAX_BLOCK_COUNT
    pearson_type = LaggedPearsonAdapter
    binned_type = BinnedNetTEAdapter
    circular_type = CircularShiftNullV2
    exact_circular_type = CircularShiftExactNullV1
    block_type = BlockShuffleNullV2
    token_type = NullTransformToken
    circular_state_type = CircularShiftStateV2
    block_state_type = BlockShuffleStateV2

    def freeze_member_slots(
        owner: type[object],
        names: tuple[str, ...],
    ) -> tuple[tuple[str, MemberDescriptorType], ...]:
        frozen: list[tuple[str, MemberDescriptorType]] = []
        for name in names:
            descriptor = owner.__dict__.get(name)
            if exact_type(descriptor) is not MemberDescriptorType:
                raise RuntimeError(
                    f"result verifier member slot is invalid: {owner.__name__}.{name}"
                )
            frozen.append((name, cast(MemberDescriptorType, descriptor)))
        return tuple_type(frozen)

    def freeze_descriptors(
        owner: type[object],
        names: tuple[str, ...],
    ) -> tuple[tuple[str, object], ...]:
        frozen: list[tuple[str, object]] = []
        for name in names:
            descriptor = owner.__dict__.get(name)
            if descriptor is None:
                raise RuntimeError(
                    f"result verifier descriptor is missing: {owner.__name__}.{name}"
                )
            frozen.append((name, descriptor))
        return tuple_type(frozen)

    pearson_slots = freeze_member_slots(LaggedPearsonAdapter, ("_parameters",))
    binned_slots = freeze_member_slots(BinnedNetTEAdapter, ("bins", "_parameters"))
    circular_slots = freeze_member_slots(
        CircularShiftNullV2,
        ("min_shift", "_parameters", "_null_parameter_sha256"),
    )
    exact_circular_slots = freeze_member_slots(
        CircularShiftExactNullV1,
        ("min_shift", "_parameters", "_null_parameter_sha256"),
    )
    block_slots = freeze_member_slots(
        BlockShuffleNullV2,
        ("block_length", "_parameters", "_null_parameter_sha256"),
    )
    pearson_descriptors = freeze_descriptors(
        LaggedPearsonAdapter,
        ("name", "parameters"),
    )
    binned_descriptors = freeze_descriptors(
        BinnedNetTEAdapter,
        ("name", "parameters"),
    )
    circular_descriptors = freeze_descriptors(
        CircularShiftNullV2,
        ("name", "parameters", "null_parameter_sha256"),
    )
    exact_circular_descriptors = freeze_descriptors(
        CircularShiftExactNullV1,
        ("name", "parameters", "null_parameter_sha256"),
    )
    block_descriptors = freeze_descriptors(
        BlockShuffleNullV2,
        ("name", "parameters", "null_parameter_sha256"),
    )
    token_slots = freeze_member_slots(
        token_type,
        (
            "schema",
            "null_name",
            "null_parameter_sha256",
            "semantic_input_sha256",
            "scientific_plan_sha256",
            "bound_null_owner_sha256",
            "is_identity",
            "state",
        ),
    )
    circular_state_slots = freeze_member_slots(
        circular_state_type,
        ("schema", "shift"),
    )
    block_state_slots = freeze_member_slots(
        block_state_type,
        ("schema", "block_order"),
    )

    def read_slots(
        value: Any,
        owner: type[object],
        frozen: tuple[tuple[str, MemberDescriptorType], ...],
        subject: str,
    ) -> tuple[Any, ...]:
        if exact_type(value) is not owner:
            raise integrity_error_type(f"{subject} has the wrong exact type")
        values: list[Any] = []
        for name, descriptor in frozen:
            if owner.__dict__.get(name) is not descriptor:
                raise integrity_error_type(f"{subject} descriptor drifted: {name}")
            try:
                values.append(member_descriptor_get(descriptor, value, owner))
            except structural_errors as error:
                raise integrity_error_type(f"{subject} slot snapshot is invalid") from error
        return tuple_type(values)

    def descriptor_snapshot(
        owner: type[object],
        frozen: tuple[tuple[str, object], ...],
        subject: str,
    ) -> tuple[object, ...]:
        for name, descriptor in frozen:
            if owner.__dict__.get(name) is not descriptor:
                raise integrity_error_type(f"{subject} descriptor drifted: {name}")
        return tuple_type(descriptor for _name, descriptor in frozen)

    def require_empty_parameters(value: Any, subject: str) -> bytes:
        if exact_type(value) is not mapping_proxy_type or length(value) != 0:
            raise integrity_error_type(f"{subject} parameters drifted")
        return b"{}"

    def require_scalar_parameters(
        value: Any,
        parameter_name: str,
        scalar: Any,
        minimum: int,
        subject: str,
    ) -> bytes:
        if (
            exact_type(scalar) is not integer_type
            or scalar < minimum
            or exact_type(value) is not mapping_proxy_type
            or length(value) != 1
            or value.get(parameter_name) != scalar
            or tuple_type(value) != (parameter_name,)
        ):
            raise integrity_error_type(f"{subject} parameters drifted")
        return ('{"' + parameter_name + '":' + str_value(scalar) + "}").encode("utf-8")

    def require_sha256(value: Any, subject: str) -> str:
        if (
            exact_type(value) is not string_type
            or length(value) != 64
            or any_value(character not in "0123456789abcdef" for character in value)
        ):
            raise integrity_error_type(f"{subject} must be a lowercase SHA-256 digest")
        return str_value(value)

    def canonical_digest(payload: object) -> str:
        canonical = json_dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return sha256(canonical).hexdigest()

    def parameter_digest(null_name: str, parameter_name: str, scalar: int) -> str:
        return canonical_digest(
            {
                "schema": "selcal.null-parameters.v2",
                "name": null_name,
                "params": {parameter_name: scalar},
            }
        )

    def owner_digest(
        semantic_input_sha256: str,
        scientific_plan_sha256: str,
        null_parameter_sha256: str,
        observed_length: int,
    ) -> str:
        return canonical_digest(
            {
                "schema": "selcal.bound-null-owner.v2",
                "semantic_input_sha256": semantic_input_sha256,
                "scientific_plan_sha256": scientific_plan_sha256,
                "null_parameter_sha256": null_parameter_sha256,
                "observed_length": observed_length,
                "transformed_role": "source",
            }
        )

    def _pearson_adapter_snapshot(adapter: object) -> tuple[Any, ...]:
        (parameters,) = read_slots(
            adapter,
            pearson_type,
            pearson_slots,
            "lagged Pearson statistic",
        )
        parameters_bytes = require_empty_parameters(
            parameters,
            "lagged Pearson statistic",
        )
        descriptors = descriptor_snapshot(
            pearson_type,
            pearson_descriptors,
            "lagged Pearson statistic",
        )
        return (
            parameters_bytes,
            (),
            descriptors,
            f"selcal.lagged_pearson.v1|numpy={numpy_version}|units=pearson_correlation",
            pearson_preprocessing,
        )

    def _binned_adapter_snapshot(adapter: object) -> tuple[Any, ...]:
        bins, parameters = read_slots(
            adapter,
            binned_type,
            binned_slots,
            "binned statistic",
        )
        parameters_bytes = require_scalar_parameters(
            parameters,
            "bins",
            bins,
            2,
            "binned statistic",
        )
        descriptors = descriptor_snapshot(
            binned_type,
            binned_descriptors,
            "binned statistic",
        )
        return (
            parameters_bytes,
            (bins,),
            descriptors,
            f"selcal.equal_width_binned_nette.v1|numpy={numpy_version}|bins={bins}|units=nats",
            binned_preprocessing,
        )

    def _pearson_result_identity(adapter: object) -> tuple[str, str]:
        snapshot = _pearson_adapter_snapshot(adapter)
        return str_value(snapshot[3]), str_value(snapshot[4])

    def _binned_result_identity(adapter: object) -> tuple[str, str]:
        snapshot = _binned_adapter_snapshot(adapter)
        return str_value(snapshot[3]), str_value(snapshot[4])

    def _circular_adapter_snapshot(adapter: object) -> tuple[Any, ...]:
        min_shift, parameters, stored_digest = read_slots(
            adapter,
            circular_type,
            circular_slots,
            "circular null",
        )
        parameters_bytes = require_scalar_parameters(
            parameters,
            "min_shift",
            min_shift,
            1,
            "circular null",
        )
        exact_min_shift = min_shift
        expected_digest = parameter_digest(
            "circular_shift_v2",
            "min_shift",
            exact_min_shift,
        )
        if require_sha256(stored_digest, "circular null parameter digest") != expected_digest:
            raise integrity_error_type("circular null parameter digest drifted")
        descriptors = descriptor_snapshot(
            circular_type,
            circular_descriptors,
            "circular null",
        )
        return parameters_bytes, (exact_min_shift, expected_digest), descriptors

    def _exact_circular_adapter_snapshot(adapter: object) -> tuple[Any, ...]:
        min_shift, parameters, stored_digest = read_slots(
            adapter,
            exact_circular_type,
            exact_circular_slots,
            "exact circular null",
        )
        parameters_bytes = require_scalar_parameters(
            parameters,
            "min_shift",
            min_shift,
            1,
            "exact circular null",
        )
        if min_shift != 1:
            raise integrity_error_type("exact circular null requires the complete group")
        expected_digest = parameter_digest(
            "circular_shift_exact_v1",
            "min_shift",
            min_shift,
        )
        if require_sha256(stored_digest, "exact circular null parameter digest") != expected_digest:
            raise integrity_error_type("exact circular null parameter digest drifted")
        descriptors = descriptor_snapshot(
            exact_circular_type,
            exact_circular_descriptors,
            "exact circular null",
        )
        return parameters_bytes, (min_shift, expected_digest), descriptors

    def _block_adapter_snapshot(adapter: object) -> tuple[Any, ...]:
        block_length, parameters, stored_digest = read_slots(
            adapter,
            block_type,
            block_slots,
            "block null",
        )
        parameters_bytes = require_scalar_parameters(
            parameters,
            "block_length",
            block_length,
            1,
            "block null",
        )
        exact_block_length = block_length
        expected_digest = parameter_digest(
            "block_shuffle_v2",
            "block_length",
            exact_block_length,
        )
        if require_sha256(stored_digest, "block null parameter digest") != expected_digest:
            raise integrity_error_type("block null parameter digest drifted")
        descriptors = descriptor_snapshot(
            block_type,
            block_descriptors,
            "block null",
        )
        return parameters_bytes, (exact_block_length, expected_digest), descriptors

    def verify_common_token(
        adapter_snapshot: tuple[Any, ...],
        token: object,
        expected_name: str,
        semantic_input_sha256: str,
        scientific_plan_sha256: str,
        observed_length: int,
    ) -> tuple[Any, Any]:
        if exact_type(observed_length) is not integer_type or observed_length < 1:
            raise integrity_error_type("result token observed length is invalid")
        operational = adapter_snapshot[1]
        parameter_digest_value = operational[1]
        (
            schema,
            null_name,
            token_parameter_digest,
            token_semantic_digest,
            token_plan_digest,
            token_owner_digest,
            is_identity,
            state,
        ) = read_slots(token, token_type, token_slots, "result token")
        exact_semantic_digest = require_sha256(
            semantic_input_sha256,
            "semantic_input_sha256",
        )
        exact_plan_digest = require_sha256(
            scientific_plan_sha256,
            "scientific_plan_sha256",
        )
        expected_owner_digest = owner_digest(
            exact_semantic_digest,
            exact_plan_digest,
            parameter_digest_value,
            observed_length,
        )
        if (
            schema != "selcal.null-transform-token.v2"
            or exact_type(null_name) is not string_type
            or null_name != expected_name
            or require_sha256(token_parameter_digest, "null_parameter_sha256")
            != parameter_digest_value
            or require_sha256(token_semantic_digest, "semantic_input_sha256")
            != exact_semantic_digest
            or require_sha256(token_plan_digest, "scientific_plan_sha256") != exact_plan_digest
            or require_sha256(token_owner_digest, "bound_null_owner_sha256")
            != expected_owner_digest
            or exact_type(is_identity) is not boolean_type
        ):
            raise integrity_error_type("result token ownership contradicts the sealed resolution")
        return state, is_identity

    def _verify_circular_result_token(
        adapter: object,
        token: object,
        semantic_input_sha256: str,
        scientific_plan_sha256: str,
        observed_length: int,
        *,
        replicate_id: int | None = None,
        planned_replicates: int | None = None,
    ) -> None:
        adapter_snapshot = _circular_adapter_snapshot(adapter)
        operational = adapter_snapshot[1]
        min_shift = operational[0]
        if 2 * min_shift > observed_length:
            raise integrity_error_type("result circular state space is invalid")
        state, is_identity = verify_common_token(
            adapter_snapshot,
            token,
            "circular_shift_v2",
            semantic_input_sha256,
            scientific_plan_sha256,
            observed_length,
        )
        state_schema, shift = read_slots(
            state,
            circular_state_type,
            circular_state_slots,
            "circular token state",
        )
        if (
            state_schema != "selcal.circular-shift-state.v2"
            or exact_type(shift) is not integer_type
            or is_identity is not (shift == 0)
            or (shift != 0 and not (min_shift <= shift <= observed_length - min_shift))
        ):
            raise integrity_error_type("result token shift is outside its bound state universe")

    def _verify_exact_circular_result_token(
        adapter: object,
        token: object,
        semantic_input_sha256: str,
        scientific_plan_sha256: str,
        observed_length: int,
        *,
        replicate_id: int | None = None,
        planned_replicates: int | None = None,
    ) -> None:
        adapter_snapshot = _exact_circular_adapter_snapshot(adapter)
        if observed_length < 2:
            raise integrity_error_type("result exact circular state space is invalid")
        state, is_identity = verify_common_token(
            adapter_snapshot,
            token,
            "circular_shift_exact_v1",
            semantic_input_sha256,
            scientific_plan_sha256,
            observed_length,
        )
        state_schema, shift = read_slots(
            state,
            circular_state_type,
            circular_state_slots,
            "exact circular token state",
        )
        if (
            state_schema != "selcal.circular-shift-state.v2"
            or exact_type(shift) is not integer_type
            or is_identity is not (shift == 0)
            or not 0 <= shift < observed_length
        ):
            raise integrity_error_type("result token shift is outside its bound state universe")
        if (
            exact_type(replicate_id) is not integer_type
            or exact_type(planned_replicates) is not integer_type
            or planned_replicates != observed_length - 1
            or replicate_id is None
            or shift != replicate_id + 1
        ):
            raise integrity_error_type(
                "result replicates do not enumerate the complete circular group"
            )

    def _verify_block_result_token(
        adapter: object,
        token: object,
        semantic_input_sha256: str,
        scientific_plan_sha256: str,
        observed_length: int,
        *,
        replicate_id: int | None = None,
        planned_replicates: int | None = None,
    ) -> None:
        adapter_snapshot = _block_adapter_snapshot(adapter)
        operational = adapter_snapshot[1]
        block_length = operational[0]
        if observed_length % block_length != 0:
            raise integrity_error_type("result block state space is invalid")
        block_count = observed_length // block_length
        if not 2 <= block_count <= max_block_count:
            raise integrity_error_type("result block state space is invalid")
        state, is_identity = verify_common_token(
            adapter_snapshot,
            token,
            "block_shuffle_v2",
            semantic_input_sha256,
            scientific_plan_sha256,
            observed_length,
        )
        state_schema, block_order = read_slots(
            state,
            block_state_type,
            block_state_slots,
            "block token state",
        )
        if (
            state_schema != "selcal.block-shuffle-state.v2"
            or exact_type(block_order) is not tuple_type
            or length(block_order) != block_count
            or any_value(exact_type(label) is not integer_type for label in block_order)
            or set_type(block_order) != set_type(range_value(block_count))
            or is_identity is not (block_order == tuple_type(range_value(block_count)))
        ):
            raise integrity_error_type("result token block order is invalid")

    statistic_snapshots: Mapping[type[object], Callable[[object], tuple[object, ...]]] = (
        MappingProxyType(
            {
                LaggedPearsonAdapter: _pearson_adapter_snapshot,
                BinnedNetTEAdapter: _binned_adapter_snapshot,
            }
        )
    )
    statistic_identities: Mapping[type[object], _StatisticResultIdentityV2] = MappingProxyType(
        {
            LaggedPearsonAdapter: _pearson_result_identity,
            BinnedNetTEAdapter: _binned_result_identity,
        }
    )
    null_snapshots: Mapping[type[object], Callable[[object], tuple[object, ...]]] = (
        MappingProxyType(
            {
                CircularShiftNullV2: _circular_adapter_snapshot,
                CircularShiftExactNullV1: _exact_circular_adapter_snapshot,
                BlockShuffleNullV2: _block_adapter_snapshot,
            }
        )
    )
    token_verifiers: Mapping[type[object], _ResultTokenVerifierV2] = MappingProxyType(
        {
            CircularShiftNullV2: _verify_circular_result_token,
            CircularShiftExactNullV1: _verify_exact_circular_result_token,
            BlockShuffleNullV2: _verify_block_result_token,
        }
    )
    return (
        statistic_snapshots,
        statistic_identities,
        null_snapshots,
        token_verifiers,
    )


(
    _SEALED_STATISTIC_SNAPSHOTS_V2,
    _SEALED_STATISTIC_RESULT_IDENTITIES_V2,
    _SEALED_NULL_SNAPSHOTS_V2,
    _SEALED_RESULT_TOKEN_VERIFIERS_V2,
) = _freeze_result_verifier_science_leaves_v2()
del _freeze_result_verifier_science_leaves_v2


_STATISTIC_REGISTRATIONS_V2 = MappingProxyType(
    {
        registration.name: registration
        for registration in (
            _AdapterRegistrationEntryV2(
                name="equal_width_binned_nette_v1",
                factory=BinnedNetTEAdapter.from_parameters,
                implementation_type=BinnedNetTEAdapter,
                statistic_identity_snapshot=_binned_identity_snapshot,
                statistic_result_identity=_binned_result_identity,
                bound_implementation_type=_BoundBinnedNetTEAdapter,
                bound_statistic_snapshot=_binned_bound_snapshot,
            ),
            _AdapterRegistrationEntryV2(
                name="lagged_pearson_v1",
                factory=LaggedPearsonAdapter.from_parameters,
                implementation_type=LaggedPearsonAdapter,
                statistic_identity_snapshot=_pearson_identity_snapshot,
                statistic_result_identity=_pearson_result_identity,
                bound_implementation_type=_BoundLaggedPearsonAdapter,
                bound_statistic_snapshot=_pearson_bound_snapshot,
            ),
        )
    }
)
_NULL_REGISTRATIONS_V2 = MappingProxyType(
    {
        registration.name: registration
        for registration in (
            _AdapterRegistrationEntryV2(
                name="block_shuffle_v2",
                factory=BlockShuffleNullV2.from_parameters,
                implementation_type=BlockShuffleNullV2,
                null_identity_snapshot=_block_identity_snapshot,
                result_token_verifier=_verify_block_result_token,
                bound_implementation_type=_BoundBlockShuffleV2,
                bound_null_snapshot=_block_bound_snapshot,
            ),
            _AdapterRegistrationEntryV2(
                name="circular_shift_exact_v1",
                factory=CircularShiftExactNullV1.from_parameters,
                implementation_type=CircularShiftExactNullV1,
                null_identity_snapshot=_exact_circular_identity_snapshot,
                result_token_verifier=_verify_exact_circular_result_token,
                bound_implementation_type=_BoundCircularShiftExactV1,
                bound_null_snapshot=_exact_circular_bound_snapshot,
            ),
            _AdapterRegistrationEntryV2(
                name="circular_shift_v2",
                factory=CircularShiftNullV2.from_parameters,
                implementation_type=CircularShiftNullV2,
                null_identity_snapshot=_circular_identity_snapshot,
                result_token_verifier=_verify_circular_result_token,
                bound_implementation_type=_BoundCircularShiftV2,
                bound_null_snapshot=_circular_bound_snapshot,
            ),
        )
    }
)
_CANONICAL_STATISTIC_REGISTRATIONS_V2 = MappingProxyType(
    {
        name: _freeze_adapter_registration(
            registration,
            subject=f"statistic registration {name}",
        )
        for name, registration in _STATISTIC_REGISTRATIONS_V2.items()
    }
)
_CANONICAL_NULL_REGISTRATIONS_V2 = MappingProxyType(
    {
        name: _freeze_adapter_registration(
            registration,
            subject=f"null registration {name}",
        )
        for name, registration in _NULL_REGISTRATIONS_V2.items()
    }
)
_STATISTIC_REGISTRY_V2: AdapterRegistry[object] = AdapterRegistry(
    tuple(
        (registration.name, registration.factory)
        for registration in _STATISTIC_REGISTRATIONS_V2.values()
    )
)
_NULL_REGISTRY_V2: AdapterRegistry[object] = AdapterRegistry(
    tuple(
        (registration.name, registration.factory)
        for registration in _NULL_REGISTRATIONS_V2.values()
    )
)
_CANONICAL_STATISTIC_ADAPTER_REGISTRY_V2 = _freeze_adapter_registry(
    _STATISTIC_REGISTRY_V2,
    subject="statistic adapter registry",
)
_CANONICAL_NULL_ADAPTER_REGISTRY_V2 = _freeze_adapter_registry(
    _NULL_REGISTRY_V2,
    subject="null adapter registry",
)


def _registration_snapshot_matches(
    current: _AdapterRegistrationV2,
    canonical: _AdapterRegistrationV2,
) -> bool:
    return (
        current.entry is canonical.entry
        and type(current.name) is str
        and current.name == canonical.name
        and current.factory is canonical.factory
        and current.implementation_type is canonical.implementation_type
        and current.statistic_identity_snapshot is canonical.statistic_identity_snapshot
        and current.null_identity_snapshot is canonical.null_identity_snapshot
        and current.statistic_result_identity is canonical.statistic_result_identity
        and current.result_token_verifier is canonical.result_token_verifier
        and current.bound_implementation_type is canonical.bound_implementation_type
        and current.bound_statistic_snapshot is canonical.bound_statistic_snapshot
        and current.bound_null_snapshot is canonical.bound_null_snapshot
    )


def _require_canonical_adapter_registration(
    name: str,
    entry: object,
    *,
    role: str,
) -> _AdapterRegistrationV2:
    canonical_registrations = (
        _CANONICAL_STATISTIC_REGISTRATIONS_V2
        if role == "statistic"
        else _CANONICAL_NULL_REGISTRATIONS_V2
    )
    try:
        canonical = canonical_registrations[name]
    except KeyError as error:
        raise V2IntegrityError(f"{role} has no canonical v2 registration") from error
    current = _freeze_adapter_registration(entry, subject=f"{role} registration")
    if not _registration_snapshot_matches(current, canonical):
        raise V2IntegrityError(f"{role} registration metadata drifted")
    return canonical


def _adapter_registry_snapshot_matches(
    current: _AdapterRegistryIdentityV2,
    canonical: _AdapterRegistryIdentityV2,
) -> bool:
    return (
        current.registry is canonical.registry
        and current.factories_mapping is canonical.factories_mapping
        and len(current.factories) == len(canonical.factories)
        and all(
            type(current_name) is str
            and current_name == canonical_name
            and current_factory is canonical_factory
            for (current_name, current_factory), (
                canonical_name,
                canonical_factory,
            ) in zip(current.factories, canonical.factories, strict=True)
        )
    )


def _require_canonical_adapter_registry(
    *,
    role: str,
) -> _AdapterRegistryIdentityV2:
    if role == "statistic":
        registry = _STATISTIC_REGISTRY_V2
        canonical = _CANONICAL_STATISTIC_ADAPTER_REGISTRY_V2
    else:
        registry = _NULL_REGISTRY_V2
        canonical = _CANONICAL_NULL_ADAPTER_REGISTRY_V2
    current = _freeze_adapter_registry(
        registry,
        subject=f"{role} adapter registry",
    )
    if not _adapter_registry_snapshot_matches(current, canonical):
        raise V2IntegrityError(
            f"{role} adapter registry factory identity drifted before statistic "
            "name and parameters, exact registered StatisticAdapter implementation, "
            "or null name, parameters, identity, snapshot, and exact executable v2 "
            "null validation, including the public null infrastructure failure boundary"
        )
    return current


def _exact_snapshot_value_matches(current: object, initial: object) -> bool:
    if type(current) is not type(initial):
        return False
    if type(current) is tuple:
        current_tuple = cast(tuple[object, ...], current)
        initial_tuple = cast(tuple[object, ...], initial)
        return len(current_tuple) == len(initial_tuple) and all(
            _exact_snapshot_value_matches(current_value, initial_value)
            for current_value, initial_value in zip(
                current_tuple,
                initial_tuple,
                strict=True,
            )
        )
    return bool(current == initial)


def _bound_statistic_execution_snapshot_matches(
    current: _BoundStatisticExecutionSnapshotV2,
    initial: _BoundStatisticExecutionSnapshotV2,
) -> bool:
    return (
        current.implementation_type is initial.implementation_type
        and _exact_snapshot_value_matches(
            current.operational_snapshot,
            initial.operational_snapshot,
        )
        and current.evaluate_all.descriptor is initial.evaluate_all.descriptor
        and current.evaluate_all.callable is initial.evaluate_all.callable
    )


def _bound_null_execution_snapshot_matches(
    current: _BoundNullExecutionSnapshotV2,
    initial: _BoundNullExecutionSnapshotV2,
) -> bool:
    return (
        current.implementation_type is initial.implementation_type
        and current.observed_pair is initial.observed_pair
        and current.retained_token_state_units == initial.retained_token_state_units
        and _exact_snapshot_value_matches(
            current.operational_snapshot,
            initial.operational_snapshot,
        )
        and current.identity_token.descriptor is initial.identity_token.descriptor
        and current.identity_token.callable is initial.identity_token.callable
        and current.sample_token.descriptor is initial.sample_token.descriptor
        and current.sample_token.callable is initial.sample_token.callable
        and (current.enumerate_token is None) is (initial.enumerate_token is None)
        and (
            current.enumerate_token is None
            or (
                initial.enumerate_token is not None
                and current.enumerate_token.descriptor is initial.enumerate_token.descriptor
                and current.enumerate_token.callable is initial.enumerate_token.callable
            )
        )
        and current.snapshot_token.descriptor is initial.snapshot_token.descriptor
        and current.snapshot_token.callable is initial.snapshot_token.callable
        and current.apply.descriptor is initial.apply.descriptor
        and current.apply.callable is initial.apply.callable
    )


class _ResolutionOwnershipSnapshotV2(NamedTuple):
    plan: ResolvedScientificPlanV2
    adapters: ResolvedAdaptersV2
    statistic: StatisticAdapter
    null_model: _ExecutableNullV2
    plan_canonical_bytes: bytes
    plan_sha256: str
    statistic_implementation_type: type[object]
    null_implementation_type: type[object]
    statistic_parameters_bytes: bytes
    null_parameters_bytes: bytes
    statistic_operational_snapshot: tuple[JsonValue, ...]
    null_operational_snapshot: tuple[JsonValue, ...]
    statistic_descriptor_snapshot: tuple[object, ...]
    null_descriptor_snapshot: tuple[object, ...]
    statistic_registry_identity: _AdapterRegistryIdentityV2
    null_registry_identity: _AdapterRegistryIdentityV2
    statistic_registration: _AdapterRegistrationEntryV2
    null_registration: _AdapterRegistrationEntryV2
    statistic_result_identity: _StatisticResultIdentityV2
    result_token_verifier: _ResultTokenVerifierV2
    statistic_bound_implementation_type: type[object]
    null_bound_implementation_type: type[object]
    statistic_bound_snapshotter: Callable[
        [
            object,
            SeriesPair,
            Mapping[str, JsonValue],
            _BoundStatisticExecutionSnapshotV2 | None,
        ],
        _BoundStatisticExecutionSnapshotV2,
    ]
    null_bound_snapshotter: Callable[
        [
            object,
            SeriesPair,
            Mapping[str, JsonValue],
            _BoundNullExecutionSnapshotV2 | None,
        ],
        _BoundNullExecutionSnapshotV2,
    ]


class _ResolutionIdentityRecordV2(NamedTuple):
    reference: ReferenceType[PlanResolutionV2]
    snapshot: _ResolutionOwnershipSnapshotV2


_resolver_owned_by_id_bootstrap: dict[int, _ResolutionIdentityRecordV2] = {}


def _static_descriptor_snapshot(
    source: object,
    names: tuple[str, ...],
    *,
    subject: str,
) -> tuple[object, ...]:
    try:
        return tuple(inspect.getattr_static(type(source), name) for name in names)
    except _STRUCTURAL_SNAPSHOT_ERRORS as error:
        raise V2IntegrityError(f"{subject} descriptor snapshot is invalid") from error


def _pure_statistic_adapter_snapshot(
    adapter: object,
) -> tuple[type[object], bytes, tuple[JsonValue, ...], tuple[object, ...]]:
    implementation_type = type(adapter)
    if implementation_type is LaggedPearsonAdapter:
        parameters = _read_exact_slot(
            adapter,
            "_parameters",
            subject="Pearson adapter parameters",
        )
        operational: tuple[JsonValue, ...] = ()
        expected_parameters: Mapping[str, JsonValue] = MappingProxyType({})
    elif implementation_type is BinnedNetTEAdapter:
        bins = _read_exact_slot(adapter, "bins", subject="binned adapter bins")
        parameters = _read_exact_slot(
            adapter,
            "_parameters",
            subject="binned adapter parameters",
        )
        if type(bins) is not int or bins < 2:
            raise V2IntegrityError("binned adapter scalar snapshot is invalid")
        operational = (bins,)
        expected_parameters = MappingProxyType({"bins": bins})
    else:
        raise V2IntegrityError("statistic adapter has no pure registered snapshot")
    parameters_bytes = _exact_mapping_proxy_bytes(
        parameters,
        subject="statistic adapter parameters",
    )
    if parameters_bytes != canonical_json_bytes(expected_parameters):
        raise V2IntegrityError("statistic adapter parameters drifted")
    descriptors = _static_descriptor_snapshot(
        adapter,
        ("name", "parameters"),
        subject="statistic adapter",
    )
    return implementation_type, parameters_bytes, operational, descriptors


def _pure_null_adapter_snapshot(
    adapter: object,
) -> tuple[type[object], bytes, tuple[JsonValue, ...], tuple[object, ...]]:
    implementation_type = type(adapter)
    if implementation_type is CircularShiftNullV2:
        scalar_name = "min_shift"
        parameter_name = "min_shift"
        null_name = "circular_shift_v2"
    elif implementation_type is CircularShiftExactNullV1:
        scalar_name = "min_shift"
        parameter_name = "min_shift"
        null_name = "circular_shift_exact_v1"
    elif implementation_type is BlockShuffleNullV2:
        scalar_name = "block_length"
        parameter_name = "block_length"
        null_name = "block_shuffle_v2"
    else:
        raise V2IntegrityError("null adapter has no pure registered snapshot")
    scalar = _read_exact_slot(adapter, scalar_name, subject=f"null adapter {scalar_name}")
    parameters = _read_exact_slot(
        adapter,
        "_parameters",
        subject="null adapter parameters",
    )
    digest = _read_exact_slot(
        adapter,
        "_null_parameter_sha256",
        subject="null adapter parameter digest",
    )
    if type(scalar) is not int or scalar < 1:
        raise V2IntegrityError("null adapter scalar snapshot is invalid")
    parameters_bytes = _exact_mapping_proxy_bytes(
        parameters,
        subject="null adapter parameters",
    )
    expected_parameters: Mapping[str, JsonValue] = MappingProxyType({parameter_name: scalar})
    if parameters_bytes != canonical_json_bytes(expected_parameters):
        raise V2IntegrityError("null adapter parameters drifted")
    exact_digest = _require_sha256(digest, name="null adapter parameter digest")
    expected_digest = _parameter_digest(
        null_name=null_name,
        params=expected_parameters,
    )
    if exact_digest != expected_digest:
        raise V2IntegrityError("null adapter parameter digest drifted")
    descriptors = _static_descriptor_snapshot(
        adapter,
        ("name", "parameters", "null_parameter_sha256"),
        subject="null adapter",
    )
    return implementation_type, parameters_bytes, (scalar, exact_digest), descriptors


def _verify_identity(
    reported_name: str,
    reported_params: Mapping[str, JsonValue],
    *,
    expected_name: str,
    expected_params: Mapping[str, JsonValue],
    role: str,
) -> None:
    if type(reported_name) is not str or reported_name != expected_name:
        raise V2IntegrityError(f"{role} adapter name does not match the v2 plan")
    if canonical_json_bytes(reported_params) != canonical_json_bytes(expected_params):
        raise V2IntegrityError(f"{role} adapter parameters do not match the v2 plan")


def _verified_statistic(
    adapter: object,
    *,
    expected_name: str,
    expected_params: Mapping[str, JsonValue],
) -> tuple[StatisticAdapter, _StatisticIdentitySnapshotV2]:
    try:
        registration = _STATISTIC_REGISTRATIONS_V2[expected_name]
    except KeyError as error:
        raise V2IntegrityError("statistic has no registered v2 implementation metadata") from error
    current_registration = _freeze_adapter_registration(
        registration,
        subject="statistic registration",
    )
    if current_registration.statistic_identity_snapshot is None:
        raise V2IntegrityError("statistic registration has no identity snapshot validator")
    canonical_registration = _require_canonical_adapter_registration(
        expected_name,
        registration,
        role="statistic",
    )
    if type(adapter) is not canonical_registration.implementation_type:
        raise V2IntegrityError("statistic adapter is not the exact registered implementation")
    snapshotter = cast(
        Callable[[object], _StatisticIdentitySnapshotV2],
        canonical_registration.statistic_identity_snapshot,
    )
    exact_adapter = cast(StatisticAdapter, adapter)
    snapshot = snapshotter(exact_adapter)
    _verify_identity(
        snapshot.name,
        snapshot.parameters,
        expected_name=expected_name,
        expected_params=expected_params,
        role="statistic",
    )
    try:
        current_registration_entry = _STATISTIC_REGISTRATIONS_V2[expected_name]
    except KeyError as error:
        raise V2IntegrityError("statistic has no registered v2 implementation metadata") from error
    _require_canonical_adapter_registration(
        expected_name,
        current_registration_entry,
        role="statistic",
    )
    return exact_adapter, snapshot


def _verified_null(
    adapter: object,
    *,
    expected_name: str,
    expected_params: Mapping[str, JsonValue],
) -> tuple[_ExecutableNullV2, _NullIdentitySnapshotV2]:
    try:
        registration = _NULL_REGISTRATIONS_V2[expected_name]
    except KeyError as error:
        raise V2IntegrityError("null has no registered v2 implementation metadata") from error
    current_registration = _freeze_adapter_registration(
        registration,
        subject="null registration",
    )
    if current_registration.null_identity_snapshot is None:
        raise V2IntegrityError("null registration has no identity snapshot validator")
    canonical_registration = _require_canonical_adapter_registration(
        expected_name,
        registration,
        role="null",
    )
    if type(adapter) is not canonical_registration.implementation_type:
        raise V2IntegrityError("null adapter is not the exact registered implementation")
    snapshotter = cast(
        Callable[[object], _NullIdentitySnapshotV2],
        canonical_registration.null_identity_snapshot,
    )
    exact_adapter = cast(_ExecutableNullV2, adapter)
    snapshot = snapshotter(exact_adapter)
    _verify_identity(
        snapshot.name,
        snapshot.parameters,
        expected_name=expected_name,
        expected_params=expected_params,
        role="null",
    )
    try:
        current_registration_entry = _NULL_REGISTRATIONS_V2[expected_name]
    except KeyError as error:
        raise V2IntegrityError("null has no registered v2 implementation metadata") from error
    _require_canonical_adapter_registration(
        expected_name,
        current_registration_entry,
        role="null",
    )
    return exact_adapter, snapshot


def _plan_request_snapshot(plan: ResolvedScientificPlanV2) -> PlanRequestV2:
    return PlanRequestV2(
        candidates=plan.candidates,
        statistic_name=plan.statistic_name,
        statistic_params=plan.statistic_params,
        selection_rule=plan.selection_rule,
        null_name=plan.null_name,
        null_params=plan.null_params,
        replicates=plan.replicates,
        alpha=plan.alpha,
        tie_tolerance=plan.tie_tolerance,
        root_seed=plan.root_seed,
    )


def _validate_resolution_fields(
    resolution: PlanResolutionV2,
) -> tuple[_StatisticIdentitySnapshotV2, _NullIdentitySnapshotV2]:
    if type(resolution.plan) is not ResolvedScientificPlanV2:
        raise V2IntegrityError("resolution plan must be an exact ResolvedScientificPlanV2")
    if type(resolution.adapters) is not ResolvedAdaptersV2:
        raise V2IntegrityError("resolution adapters must be an exact ResolvedAdaptersV2")
    _plan_request_snapshot(resolution.plan)
    if canonical_candidates(resolution.plan.candidates) != resolution.plan.candidates:
        raise V2IntegrityError("resolution plan candidates must remain canonical")
    _, statistic_snapshot = _verified_statistic(
        resolution.adapters.statistic,
        expected_name=resolution.plan.statistic_name,
        expected_params=resolution.plan.statistic_params,
    )
    _, null_snapshot = _verified_null(
        resolution.adapters.null_model,
        expected_name=resolution.plan.null_name,
        expected_params=resolution.plan.null_params,
    )
    return statistic_snapshot, null_snapshot


def _capture_resolution_snapshot(
    resolution: PlanResolutionV2,
) -> _ResolutionOwnershipSnapshotV2:
    _require_canonical_adapter_registry(role="statistic")
    _require_canonical_adapter_registry(role="null")
    statistic_snapshot, null_snapshot = _validate_resolution_fields(resolution)
    _require_canonical_adapter_registry(role="statistic")
    _require_canonical_adapter_registry(role="null")
    try:
        statistic_registration = _STATISTIC_REGISTRATIONS_V2[resolution.plan.statistic_name]
        null_registration = _NULL_REGISTRATIONS_V2[resolution.plan.null_name]
    except KeyError as error:
        raise V2IntegrityError("resolution registration snapshot is invalid") from error
    canonical_statistic_registration = _require_canonical_adapter_registration(
        resolution.plan.statistic_name,
        statistic_registration,
        role="statistic",
    )
    canonical_null_registration = _require_canonical_adapter_registration(
        resolution.plan.null_name,
        null_registration,
        role="null",
    )
    statistic_bound_type = cast(
        type[object],
        canonical_statistic_registration.bound_implementation_type,
    )
    null_bound_type = cast(
        type[object],
        canonical_null_registration.bound_implementation_type,
    )
    statistic_bound_snapshotter = cast(
        Callable[
            [
                object,
                SeriesPair,
                Mapping[str, JsonValue],
                _BoundStatisticExecutionSnapshotV2 | None,
            ],
            _BoundStatisticExecutionSnapshotV2,
        ],
        canonical_statistic_registration.bound_statistic_snapshot,
    )
    null_bound_snapshotter = cast(
        Callable[
            [
                object,
                SeriesPair,
                Mapping[str, JsonValue],
                _BoundNullExecutionSnapshotV2 | None,
            ],
            _BoundNullExecutionSnapshotV2,
        ],
        canonical_null_registration.bound_null_snapshot,
    )
    statistic = resolution.adapters.statistic
    null_model = resolution.adapters.null_model
    statistic_result_identity = _SEALED_STATISTIC_RESULT_IDENTITIES_V2.get(type(statistic))
    result_token_verifier = _SEALED_RESULT_TOKEN_VERIFIERS_V2.get(type(null_model))
    if statistic_result_identity is None or result_token_verifier is None:
        raise V2IntegrityError("resolution has no sealed result verification leaves")
    (
        pure_statistic_type,
        pure_statistic_parameters,
        pure_statistic_operational,
        statistic_descriptors,
    ) = _pure_statistic_adapter_snapshot(statistic)
    (
        pure_null_type,
        pure_null_parameters,
        pure_null_operational,
        null_descriptors,
    ) = _pure_null_adapter_snapshot(null_model)
    if (
        pure_statistic_type is not type(statistic)
        or pure_null_type is not type(null_model)
        or pure_statistic_parameters != statistic_snapshot.parameters_bytes
        or pure_null_parameters != null_snapshot.parameters_bytes
        or not _exact_snapshot_value_matches(
            pure_statistic_operational,
            statistic_snapshot.operational_snapshot,
        )
        or not _exact_snapshot_value_matches(
            pure_null_operational,
            null_snapshot.operational_snapshot,
        )
    ):
        raise V2IntegrityError("pure adapter identity does not match live resolution")
    plan_payload = scientific_plan_v2_payload(resolution.plan)
    statistic_registry_identity = _require_canonical_adapter_registry(role="statistic")
    null_registry_identity = _require_canonical_adapter_registry(role="null")
    return _ResolutionOwnershipSnapshotV2(
        plan=resolution.plan,
        adapters=resolution.adapters,
        statistic=statistic,
        null_model=null_model,
        plan_canonical_bytes=canonical_json_bytes(plan_payload),
        plan_sha256=scientific_plan_v2_sha256(resolution.plan),
        statistic_implementation_type=type(statistic),
        null_implementation_type=type(null_model),
        statistic_parameters_bytes=statistic_snapshot.parameters_bytes,
        null_parameters_bytes=null_snapshot.parameters_bytes,
        statistic_operational_snapshot=statistic_snapshot.operational_snapshot,
        null_operational_snapshot=null_snapshot.operational_snapshot,
        statistic_descriptor_snapshot=statistic_descriptors,
        null_descriptor_snapshot=null_descriptors,
        statistic_registry_identity=statistic_registry_identity,
        null_registry_identity=null_registry_identity,
        statistic_registration=statistic_registration,
        null_registration=null_registration,
        statistic_result_identity=statistic_result_identity,
        result_token_verifier=result_token_verifier,
        statistic_bound_implementation_type=statistic_bound_type,
        null_bound_implementation_type=null_bound_type,
        statistic_bound_snapshotter=statistic_bound_snapshotter,
        null_bound_snapshotter=null_bound_snapshotter,
    )


def _freeze_resolution_snapshot_operation_v2(
    snapshot_core: Callable[[PlanResolutionV2], _ResolutionOwnershipSnapshotV2],
    integrity_error_type: type[V2IntegrityError],
    snapshot_error_types: tuple[type[Exception], ...],
    /,
) -> Callable[[PlanResolutionV2], _ResolutionOwnershipSnapshotV2]:
    def _capture_resolution_snapshot_fail_closed(
        resolution: PlanResolutionV2,
    ) -> _ResolutionOwnershipSnapshotV2:
        try:
            return snapshot_core(resolution)
        except integrity_error_type:
            raise
        except snapshot_error_types as error:
            raise integrity_error_type("resolution identity snapshot is invalid") from error

    return _capture_resolution_snapshot_fail_closed


_capture_resolution_snapshot_fail_closed = _freeze_resolution_snapshot_operation_v2(
    _capture_resolution_snapshot,
    V2IntegrityError,
    (AttributeError, RuntimeError, TypeError, ValueError, OverflowError),
)
del _freeze_resolution_snapshot_operation_v2


_ResolutionOwnershipLookupV2 = Callable[[object], _ResolutionOwnershipSnapshotV2]


def _freeze_resolution_identity_lookup_v2(
    records: dict[int, _ResolutionIdentityRecordV2],
    /,
) -> _ResolutionOwnershipLookupV2:
    """Freeze the only production lookup path for resolver-owned resolutions."""

    exact_type = type
    identity = id
    tuple_len = tuple.__len__
    tuple_getitem = tuple.__getitem__
    records_get = records.get
    resolution_type = PlanResolutionV2
    record_type = _ResolutionIdentityRecordV2
    snapshot_type = _ResolutionOwnershipSnapshotV2
    reference_type = ReferenceType
    reference_call = ReferenceType.__call__
    integrity_error_type = V2IntegrityError

    def lookup(resolution: object, /) -> _ResolutionOwnershipSnapshotV2:
        if exact_type(resolution) is not resolution_type:
            raise integrity_error_type(
                "resolution must be an exact resolver-owned PlanResolutionV2"
            )
        record = records_get(identity(resolution))
        if record is None:
            raise integrity_error_type("resolution is not resolver-owned")
        if exact_type(record) is not record_type or tuple_len(record) != 2:
            raise integrity_error_type("resolution ownership record is invalid")
        reference = tuple_getitem(record, 0)
        snapshot = tuple_getitem(record, 1)
        if exact_type(reference) is not reference_type or exact_type(snapshot) is not snapshot_type:
            raise integrity_error_type("resolution ownership record fields are invalid")
        exact_reference: Any = reference
        exact_snapshot: Any = snapshot
        if reference_call(exact_reference) is not resolution:
            raise integrity_error_type("resolution identity drifted")
        return exact_snapshot  # type: ignore[no-any-return]

    return lookup


_LOOKUP_RESOLUTION_OWNERSHIP_V2 = _freeze_resolution_identity_lookup_v2(
    _resolver_owned_by_id_bootstrap,
)
del _freeze_resolution_identity_lookup_v2


@dataclass(frozen=True, slots=True)
class _ResultVerificationContextV2:
    """Frozen registry-owned operations needed to verify one stored result."""

    plan: ResolvedScientificPlanV2
    plan_sha256: str
    backend_identity: str
    preprocessing_identity: str
    null_adapter: object
    verify_token: _ResultTokenVerifierV2


def _freeze_result_verification_context_reader_v2(
    plan_canonical_bytes: Callable[[object], bytes],
    plan_sha256: Callable[[object], str],
    statistic_snapshots: Mapping[type[object], Callable[[object], tuple[object, ...]]],
    statistic_identities: Mapping[type[object], _StatisticResultIdentityV2],
    null_snapshots: Mapping[type[object], Callable[[object], tuple[object, ...]]],
    token_verifiers: Mapping[type[object], _ResultTokenVerifierV2],
    /,
) -> Callable[[object], object]:
    """Freeze a full pure-ownership read that returns only an exact tuple."""

    exact_type = type
    tuple_type = tuple
    list_type = list
    string_type = str
    integer_type = int
    float_type = float
    length = len
    all_values = all
    any_values = any
    zip_values = zip
    range_value = range
    boolean_value = bool
    member_descriptor_get = MemberDescriptorType.__get__
    tuple_getitem = tuple.__getitem__
    mapping_proxy_type = MappingProxyType
    resolution_type = PlanResolutionV2
    adapters_type = ResolvedAdaptersV2
    plan_type = ResolvedScientificPlanV2
    selection_rule_type = SelectionRule
    registration_entry_type = _AdapterRegistrationEntryV2
    registry_type = AdapterRegistry
    registry_identity_type = _AdapterRegistryIdentityV2
    registration_identity_type = _AdapterRegistrationV2
    ownership_type = _ResolutionOwnershipSnapshotV2
    integrity_error_type = V2IntegrityError
    structural_errors = (
        AttributeError,
        TypeError,
        ValueError,
        OverflowError,
        IndexError,
    )
    ownership_lookup = _LOOKUP_RESOLUTION_OWNERSHIP_V2
    statistic_registration_get = _STATISTIC_REGISTRATIONS_V2.get
    null_registration_get = _NULL_REGISTRATIONS_V2.get
    source_canonical_statistic_registrations = _CANONICAL_STATISTIC_REGISTRATIONS_V2
    source_canonical_null_registrations = _CANONICAL_NULL_REGISTRATIONS_V2
    canonical_statistic_registry = _CANONICAL_STATISTIC_ADAPTER_REGISTRY_V2
    canonical_null_registry = _CANONICAL_NULL_ADAPTER_REGISTRY_V2

    def freeze_member_slots(
        owner: type[object],
        names: tuple[str, ...],
    ) -> tuple[tuple[str, MemberDescriptorType], ...]:
        frozen: list[tuple[str, MemberDescriptorType]] = []
        for name in names:
            descriptor = owner.__dict__.get(name)
            if exact_type(descriptor) is not MemberDescriptorType:
                raise RuntimeError(
                    f"result context member slot is invalid: {owner.__name__}.{name}"
                )
            frozen.append((name, cast(MemberDescriptorType, descriptor)))
        return tuple_type(frozen)

    def freeze_tuple_fields(
        owner: type[object],
        names: tuple[str, ...],
    ) -> tuple[tuple[str, object], ...]:
        frozen: list[tuple[str, object]] = []
        for name in names:
            descriptor = owner.__dict__.get(name)
            if descriptor is None:
                raise RuntimeError(
                    f"result context tuple field is missing: {owner.__name__}.{name}"
                )
            frozen.append((name, descriptor))
        return tuple_type(frozen)

    resolution_slots = freeze_member_slots(resolution_type, ("plan", "adapters"))
    adapter_slots = freeze_member_slots(adapters_type, ("statistic", "null_model"))
    plan_slots = freeze_member_slots(
        plan_type,
        (
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
        ),
    )
    registration_entry_slots = freeze_member_slots(
        registration_entry_type,
        (
            "name",
            "factory",
            "implementation_type",
            "statistic_identity_snapshot",
            "null_identity_snapshot",
            "statistic_result_identity",
            "result_token_verifier",
            "bound_implementation_type",
            "bound_statistic_snapshot",
            "bound_null_snapshot",
        ),
    )
    registry_slots = freeze_member_slots(registry_type, ("_factories",))
    registry_identity_fields = freeze_tuple_fields(
        registry_identity_type,
        tuple_type(registry_identity_type._fields),
    )
    registration_identity_fields = freeze_tuple_fields(
        registration_identity_type,
        tuple_type(registration_identity_type._fields),
    )
    ownership_fields = freeze_tuple_fields(
        ownership_type,
        tuple_type(ownership_type._fields),
    )

    identity = id

    def frozen_callable_identity(value: object) -> object:
        return None if value is None else identity(value)

    def sanitize_registration(
        canonical: _AdapterRegistrationV2,
    ) -> tuple[Any, ...]:
        values: tuple[Any, ...] = tuple_type(
            tuple_getitem(canonical, index)
            for index in range_value(length(registration_identity_fields))
        )
        (
            entry,
            name,
            factory,
            implementation_type,
            statistic_identity_snapshot,
            null_identity_snapshot,
            statistic_result_identity,
            result_token_verifier,
            bound_implementation_type,
            bound_statistic_snapshot,
            bound_null_snapshot,
        ) = values
        return (
            entry,
            name,
            frozen_callable_identity(factory),
            implementation_type,
            frozen_callable_identity(statistic_identity_snapshot),
            frozen_callable_identity(null_identity_snapshot),
            frozen_callable_identity(statistic_result_identity),
            frozen_callable_identity(result_token_verifier),
            bound_implementation_type,
            frozen_callable_identity(bound_statistic_snapshot),
            frozen_callable_identity(bound_null_snapshot),
        )

    canonical_statistic_registrations = mapping_proxy_type(
        {
            name: sanitize_registration(canonical)
            for name, canonical in source_canonical_statistic_registrations.items()
        }
    )
    canonical_null_registrations = mapping_proxy_type(
        {
            name: sanitize_registration(canonical)
            for name, canonical in source_canonical_null_registrations.items()
        }
    )
    canonical_statistic_registration_get = canonical_statistic_registrations.get
    canonical_null_registration_get = canonical_null_registrations.get

    def read_slots(
        value: Any,
        owner: type[object],
        frozen: tuple[tuple[str, MemberDescriptorType], ...],
        subject: str,
    ) -> tuple[Any, ...]:
        if exact_type(value) is not owner:
            raise integrity_error_type(f"{subject} must have its exact canonical type")
        values: list[Any] = list_type()
        for name, descriptor in frozen:
            if owner.__dict__.get(name) is not descriptor:
                raise integrity_error_type(f"{subject} slot descriptor drifted: {name}")
            try:
                values.append(member_descriptor_get(descriptor, value, owner))
            except structural_errors as error:
                raise integrity_error_type(f"{subject} slot snapshot is invalid") from error
        return tuple_type(values)

    def read_tuple_fields(
        value: Any,
        owner: type[object],
        frozen: tuple[tuple[str, object], ...],
        subject: str,
    ) -> tuple[Any, ...]:
        if exact_type(value) is not owner:
            raise integrity_error_type(f"{subject} must have its exact canonical type")
        for name, descriptor in frozen:
            if owner.__dict__.get(name) is not descriptor:
                raise integrity_error_type(f"{subject} field descriptor drifted: {name}")
        try:
            return tuple_type(tuple_getitem(value, index) for index in range_value(length(frozen)))
        except structural_errors as error:
            raise integrity_error_type(f"{subject} tuple snapshot is invalid") from error

    def exact_snapshot_matches(current: Any, initial: Any) -> bool:
        if exact_type(current) is not exact_type(initial):
            return False
        if exact_type(current) is tuple_type:
            if length(current) != length(initial):
                return False
            return all_values(
                exact_snapshot_matches(current_value, initial_value)
                for current_value, initial_value in zip_values(
                    current,
                    initial,
                    strict=True,
                )
            )
        if exact_type(current) in {string_type, integer_type, float_type}:
            return boolean_value(current == initial)
        return current is initial

    def validate_registry_identity(
        current: Any,
        canonical: Any,
        role: str,
    ) -> None:
        current_values = read_tuple_fields(
            current,
            registry_identity_type,
            registry_identity_fields,
            f"{role} registry identity",
        )
        canonical_values = read_tuple_fields(
            canonical,
            registry_identity_type,
            registry_identity_fields,
            f"canonical {role} registry identity",
        )
        current_registry, current_mapping, current_factories = current_values
        canonical_registry_value, canonical_mapping, canonical_factories = canonical_values
        if (
            current_registry is not canonical_registry_value
            or current_mapping is not canonical_mapping
            or exact_type(current_mapping) is not mapping_proxy_type
            or exact_type(current_factories) is not tuple_type
            or not exact_snapshot_matches(current_factories, canonical_factories)
        ):
            raise integrity_error_type(f"{role} adapter registry identity drifted")
        (live_mapping,) = read_slots(
            current_registry,
            registry_type,
            registry_slots,
            f"{role} adapter registry",
        )
        if live_mapping is not current_mapping or length(live_mapping) != length(current_factories):
            raise integrity_error_type(f"{role} adapter registry mapping drifted")
        for item in current_factories:
            if exact_type(item) is not tuple_type or length(item) != 2:
                raise integrity_error_type(f"{role} adapter registry entry is invalid")
            name, factory = item
            if exact_type(name) is not string_type or live_mapping.get(name) is not factory:
                raise integrity_error_type(f"{role} adapter registry factory drifted")

    def validate_registration(
        entry: Any,
        canonical: Any,
        role: str,
    ) -> tuple[Any, ...]:
        entry_values = read_slots(
            entry,
            registration_entry_type,
            registration_entry_slots,
            f"{role} registration",
        )
        for name, descriptor in registration_identity_fields:
            if registration_identity_type.__dict__.get(name) is not descriptor:
                raise integrity_error_type(
                    f"canonical {role} registration field descriptor drifted: {name}"
                )
        if exact_type(canonical) is not tuple_type or length(canonical) != length(
            registration_identity_fields
        ):
            raise integrity_error_type(f"canonical {role} registration is invalid")
        canonical_values: tuple[Any, ...] = tuple_type(canonical)
        (
            entry_name,
            entry_factory,
            entry_implementation_type,
            entry_statistic_identity_snapshot,
            entry_null_identity_snapshot,
            entry_statistic_result_identity,
            entry_result_token_verifier,
            entry_bound_implementation_type,
            entry_bound_statistic_snapshot,
            entry_bound_null_snapshot,
        ) = entry_values
        (
            canonical_entry,
            canonical_name,
            canonical_factory_identity,
            canonical_implementation_type,
            canonical_statistic_identity_snapshot,
            canonical_null_identity_snapshot,
            canonical_statistic_result_identity,
            canonical_result_token_verifier,
            canonical_bound_implementation_type,
            canonical_bound_statistic_snapshot,
            canonical_bound_null_snapshot,
        ) = canonical_values
        if entry is not canonical_entry or length(entry_values) != length(canonical_values) - 1:
            raise integrity_error_type(f"{role} registration association drifted")
        if (
            entry_name != canonical_name
            or identity(entry_factory) != canonical_factory_identity
            or entry_implementation_type is not canonical_implementation_type
            or frozen_callable_identity(entry_statistic_identity_snapshot)
            != canonical_statistic_identity_snapshot
            or frozen_callable_identity(entry_null_identity_snapshot)
            != canonical_null_identity_snapshot
            or frozen_callable_identity(entry_statistic_result_identity)
            != canonical_statistic_result_identity
            or frozen_callable_identity(entry_result_token_verifier)
            != canonical_result_token_verifier
            or entry_bound_implementation_type is not canonical_bound_implementation_type
            or frozen_callable_identity(entry_bound_statistic_snapshot)
            != canonical_bound_statistic_snapshot
            or frozen_callable_identity(entry_bound_null_snapshot) != canonical_bound_null_snapshot
        ):
            raise integrity_error_type(f"{role} registration metadata drifted")
        return canonical_values

    def resolution_context(resolution: object, /) -> object:
        try:
            plan, adapters = read_slots(
                resolution,
                resolution_type,
                resolution_slots,
                "result verifier resolution",
            )
            statistic, null_adapter = read_slots(
                adapters,
                adapters_type,
                adapter_slots,
                "result verifier adapters",
            )
            (
                candidates,
                statistic_name,
                statistic_params,
                selection_rule,
                null_name,
                null_params,
                replicates,
                alpha,
                tie_tolerance,
                root_seed,
            ) = read_slots(
                plan,
                plan_type,
                plan_slots,
                "result verifier plan",
            )
            if (
                exact_type(candidates) is not tuple_type
                or not candidates
                or any_values(exact_type(candidate) is not integer_type for candidate in candidates)
                or exact_type(statistic_name) is not string_type
                or exact_type(statistic_params) is not mapping_proxy_type
                or exact_type(selection_rule) is not selection_rule_type
                or exact_type(null_name) is not string_type
                or exact_type(null_params) is not mapping_proxy_type
                or exact_type(replicates) is not integer_type
                or exact_type(alpha) is not float_type
                or exact_type(tie_tolerance) is not float_type
                or exact_type(root_seed) is not integer_type
            ):
                raise integrity_error_type("result verifier plan fields are invalid")
            registered = ownership_lookup(resolution)
            registered_values = read_tuple_fields(
                registered,
                ownership_type,
                ownership_fields,
                "resolver ownership snapshot",
            )
            (
                registered_plan,
                registered_adapters,
                registered_statistic,
                registered_null_adapter,
                registered_plan_bytes,
                registered_plan_sha256,
                registered_statistic_type,
                registered_null_type,
                registered_statistic_parameters,
                registered_null_parameters,
                registered_statistic_operational,
                registered_null_operational,
                registered_statistic_descriptors,
                registered_null_descriptors,
                registered_statistic_registry,
                registered_null_registry,
                registered_statistic_entry,
                registered_null_entry,
                registered_statistic_identity,
                registered_token_verifier,
                registered_statistic_bound_type,
                registered_null_bound_type,
                registered_statistic_bound_snapshotter,
                registered_null_bound_snapshotter,
            ) = registered_values
            if (
                plan is not registered_plan
                or adapters is not registered_adapters
                or statistic is not registered_statistic
                or null_adapter is not registered_null_adapter
                or exact_type(statistic) is not registered_statistic_type
                or exact_type(null_adapter) is not registered_null_type
            ):
                raise integrity_error_type("resolver-owned result verification association drifted")
            current_plan_bytes = plan_canonical_bytes(plan)
            current_plan_sha256 = plan_sha256(plan)
            if (
                current_plan_bytes != registered_plan_bytes
                or current_plan_sha256 != registered_plan_sha256
            ):
                raise integrity_error_type("resolver-owned result verification plan drifted")

            validate_registry_identity(
                registered_statistic_registry,
                canonical_statistic_registry,
                "statistic",
            )
            validate_registry_identity(
                registered_null_registry,
                canonical_null_registry,
                "null",
            )
            statistic_entry = statistic_registration_get(statistic_name)
            null_entry = null_registration_get(null_name)
            canonical_statistic = canonical_statistic_registration_get(statistic_name)
            canonical_null = canonical_null_registration_get(null_name)
            statistic_registration_values = validate_registration(
                statistic_entry,
                canonical_statistic,
                "statistic",
            )
            null_registration_values = validate_registration(
                null_entry,
                canonical_null,
                "null",
            )
            (
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                statistic_bound_type,
                statistic_bound_snapshotter,
                _,
            ) = statistic_registration_values
            (
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                null_bound_type,
                _,
                null_bound_snapshotter,
            ) = null_registration_values
            if (
                statistic_entry is not registered_statistic_entry
                or null_entry is not registered_null_entry
                or statistic_bound_type is not registered_statistic_bound_type
                or null_bound_type is not registered_null_bound_type
                or statistic_bound_snapshotter
                != frozen_callable_identity(registered_statistic_bound_snapshotter)
                or null_bound_snapshotter
                != frozen_callable_identity(registered_null_bound_snapshotter)
            ):
                raise integrity_error_type("resolver-owned bound registration snapshot drifted")

            statistic_snapshotter = statistic_snapshots.get(exact_type(statistic))
            statistic_identity = statistic_identities.get(exact_type(statistic))
            null_snapshotter = null_snapshots.get(exact_type(null_adapter))
            token_verifier = token_verifiers.get(exact_type(null_adapter))
            if (
                statistic_snapshotter is None
                or statistic_identity is None
                or null_snapshotter is None
                or token_verifier is None
                or statistic_identity is not registered_statistic_identity
                or token_verifier is not registered_token_verifier
            ):
                raise integrity_error_type("resolver-owned result verification leaves drifted")
            statistic_snapshot = statistic_snapshotter(statistic)
            null_snapshot = null_snapshotter(null_adapter)
            (
                statistic_parameters,
                statistic_operational,
                statistic_descriptors,
                backend_identity,
                preprocessing_identity,
            ) = statistic_snapshot
            (
                null_parameters,
                null_operational,
                null_descriptors,
            ) = null_snapshot
            if (
                statistic_parameters != registered_statistic_parameters
                or null_parameters != registered_null_parameters
                or not exact_snapshot_matches(
                    statistic_operational,
                    registered_statistic_operational,
                )
                or not exact_snapshot_matches(
                    null_operational,
                    registered_null_operational,
                )
                or not exact_snapshot_matches(
                    statistic_descriptors,
                    registered_statistic_descriptors,
                )
                or not exact_snapshot_matches(
                    null_descriptors,
                    registered_null_descriptors,
                )
            ):
                raise integrity_error_type("resolver-owned pure identity snapshot drifted")
            if (
                exact_type(backend_identity) is not string_type
                or not backend_identity
                or exact_type(preprocessing_identity) is not string_type
                or not preprocessing_identity
            ):
                raise integrity_error_type("registered statistic result identity is invalid")
        except integrity_error_type:
            raise
        except structural_errors as error:
            raise integrity_error_type("result verifier resolution snapshot is invalid") from error
        return (
            plan,
            current_plan_sha256,
            candidates,
            selection_rule,
            replicates,
            alpha,
            tie_tolerance,
            backend_identity,
            preprocessing_identity,
            null_adapter,
            token_verifier,
        )

    return resolution_context


class _BoundStatisticSnapshotEntrypointV2(Protocol):
    def __call__(
        self,
        resolution: PlanResolutionV2,
        bound: object,
        *,
        expected_pair: SeriesPair,
        initial: _BoundStatisticExecutionSnapshotV2 | None = None,
    ) -> _BoundStatisticExecutionSnapshotV2: ...


class _BoundNullSnapshotEntrypointV2(Protocol):
    def __call__(
        self,
        resolution: PlanResolutionV2,
        bound: object,
        *,
        expected_pair: SeriesPair,
        initial: _BoundNullExecutionSnapshotV2 | None = None,
    ) -> _BoundNullExecutionSnapshotV2: ...


class _ResolvePlanEntrypointV2(Protocol):
    def __call__(self, request: PlanRequestV2, /) -> PlanResolutionV2: ...


class _ResolutionEntrypointsV2(NamedTuple):
    require_owned: Callable[[object], PlanResolutionV2]
    require_owned_pure: Callable[[object], PlanResolutionV2]
    snapshot_context: Callable[[object], _ResultVerificationContextV2]
    snapshot_bound_statistic: _BoundStatisticSnapshotEntrypointV2
    snapshot_bound_null: _BoundNullSnapshotEntrypointV2
    resolve: _ResolvePlanEntrypointV2


def _freeze_require_owned_v2(
    ownership_lookup: _ResolutionOwnershipLookupV2,
    snapshotter: Callable[[PlanResolutionV2], _ResolutionOwnershipSnapshotV2],
    /,
) -> Callable[[object], PlanResolutionV2]:
    def _require_resolver_owned_resolution_v2(
        resolution: object,
    ) -> PlanResolutionV2:
        if type(resolution) is not PlanResolutionV2:
            raise V2IntegrityError("resolution must be an exact resolver-owned PlanResolutionV2")
        exact = resolution
        registered = ownership_lookup(exact)
        if exact.plan is not registered.plan or exact.adapters is not registered.adapters:
            raise V2IntegrityError("resolver-owned resolution association was replaced")
        if (
            exact.adapters.statistic is not registered.statistic
            or exact.adapters.null_model is not registered.null_model
        ):
            raise V2IntegrityError("resolver-owned adapter identity was replaced")
        current = snapshotter(exact)
        if (
            current.plan_canonical_bytes != registered.plan_canonical_bytes
            or current.plan_sha256 != registered.plan_sha256
            or current.statistic_implementation_type is not registered.statistic_implementation_type
            or current.null_implementation_type is not registered.null_implementation_type
            or current.statistic_parameters_bytes != registered.statistic_parameters_bytes
            or current.null_parameters_bytes != registered.null_parameters_bytes
            or current.statistic_operational_snapshot != registered.statistic_operational_snapshot
            or current.null_operational_snapshot != registered.null_operational_snapshot
            or not _adapter_registry_snapshot_matches(
                current.statistic_registry_identity,
                registered.statistic_registry_identity,
            )
            or not _adapter_registry_snapshot_matches(
                current.null_registry_identity,
                registered.null_registry_identity,
            )
            or current.statistic_registration is not registered.statistic_registration
            or current.null_registration is not registered.null_registration
            or current.statistic_result_identity is not registered.statistic_result_identity
            or current.result_token_verifier is not registered.result_token_verifier
            or current.statistic_bound_implementation_type
            is not registered.statistic_bound_implementation_type
            or current.null_bound_implementation_type
            is not registered.null_bound_implementation_type
            or current.statistic_bound_snapshotter is not registered.statistic_bound_snapshotter
            or current.null_bound_snapshotter is not registered.null_bound_snapshotter
        ):
            raise V2IntegrityError("resolver-owned resolution identity snapshot drifted")
        return exact

    return _require_resolver_owned_resolution_v2


def _freeze_require_owned_pure_v2(
    ownership_lookup: _ResolutionOwnershipLookupV2,
    /,
) -> Callable[[object], PlanResolutionV2]:
    def _require_resolver_owned_resolution_v2_pure(
        resolution: object,
    ) -> PlanResolutionV2:
        if type(resolution) is not PlanResolutionV2:
            raise V2IntegrityError("resolution must be an exact resolver-owned PlanResolutionV2")
        exact = resolution
        try:
            statistic_registry_identity = _require_canonical_adapter_registry(role="statistic")
            null_registry_identity = _require_canonical_adapter_registry(role="null")
            registered = ownership_lookup(exact)
            plan = _read_exact_slot(exact, "plan", subject="resolution plan")
            adapters = _read_exact_slot(exact, "adapters", subject="resolution adapters")
            if type(plan) is not ResolvedScientificPlanV2:
                raise V2IntegrityError("pure resolution plan must be exact")
            if type(adapters) is not ResolvedAdaptersV2:
                raise V2IntegrityError("pure resolution adapters must be exact")
            statistic = _read_exact_slot(
                adapters,
                "statistic",
                subject="resolution statistic adapter",
            )
            null_model = _read_exact_slot(
                adapters,
                "null_model",
                subject="resolution null adapter",
            )
            plan_fields = {
                name: _read_exact_slot(plan, name, subject=f"scientific plan {name}")
                for name in (
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
            }
            candidates = plan_fields["candidates"]
            if type(candidates) is not tuple or any(
                type(candidate) is not int for candidate in cast(tuple[object, ...], candidates)
            ):
                raise V2IntegrityError("pure scientific plan candidates must be exact")
            if (
                type(plan_fields["statistic_name"]) is not str
                or type(plan_fields["null_name"]) is not str
                or type(plan_fields["selection_rule"]) is not SelectionRule
                or type(plan_fields["replicates"]) is not int
                or type(plan_fields["alpha"]) is not float
                or type(plan_fields["tie_tolerance"]) is not float
                or type(plan_fields["root_seed"]) is not int
            ):
                raise V2IntegrityError("pure scientific plan scalar types drifted")
            statistic_registration = _STATISTIC_REGISTRATIONS_V2[plan_fields["statistic_name"]]
            null_registration = _NULL_REGISTRATIONS_V2[plan_fields["null_name"]]
            canonical_statistic_registration = _require_canonical_adapter_registration(
                plan_fields["statistic_name"],
                statistic_registration,
                role="statistic",
            )
            canonical_null_registration = _require_canonical_adapter_registration(
                plan_fields["null_name"],
                null_registration,
                role="null",
            )
            _exact_mapping_proxy_bytes(
                plan_fields["statistic_params"],
                subject="scientific plan statistic_params",
            )
            _exact_mapping_proxy_bytes(
                plan_fields["null_params"],
                subject="scientific plan null_params",
            )
            (
                statistic_type,
                statistic_parameters,
                statistic_operational,
                statistic_descriptors,
            ) = _pure_statistic_adapter_snapshot(statistic)
            (
                null_type,
                null_parameters,
                null_operational,
                null_descriptors,
            ) = _pure_null_adapter_snapshot(null_model)
            plan_payload = scientific_plan_v2_payload(plan)
            plan_canonical_bytes = canonical_json_bytes(plan_payload)
            plan_sha256 = scientific_plan_v2_sha256(plan)
            statistic_registry_identity = _require_canonical_adapter_registry(role="statistic")
            null_registry_identity = _require_canonical_adapter_registry(role="null")
        except KeyError as error:
            raise V2IntegrityError("resolution is not resolver-owned") from error
        except _STRUCTURAL_SNAPSHOT_ERRORS as error:
            raise V2IntegrityError("pure resolution identity snapshot is invalid") from error
        if plan is not registered.plan or adapters is not registered.adapters:
            raise V2IntegrityError("resolver-owned resolution association was replaced")
        if statistic is not registered.statistic or null_model is not registered.null_model:
            raise V2IntegrityError("resolver-owned adapter identity was replaced")
        if not _adapter_registry_snapshot_matches(
            statistic_registry_identity,
            registered.statistic_registry_identity,
        ) or not _adapter_registry_snapshot_matches(
            null_registry_identity,
            registered.null_registry_identity,
        ):
            raise V2IntegrityError("resolver-owned adapter registry snapshot drifted")
        if (
            statistic_registration is not registered.statistic_registration
            or null_registration is not registered.null_registration
            or _SEALED_STATISTIC_RESULT_IDENTITIES_V2.get(type(statistic))
            is not registered.statistic_result_identity
            or _SEALED_RESULT_TOKEN_VERIFIERS_V2.get(type(null_model))
            is not registered.result_token_verifier
            or canonical_statistic_registration.bound_implementation_type
            is not registered.statistic_bound_implementation_type
            or canonical_null_registration.bound_implementation_type
            is not registered.null_bound_implementation_type
            or canonical_statistic_registration.bound_statistic_snapshot
            is not registered.statistic_bound_snapshotter
            or canonical_null_registration.bound_null_snapshot
            is not registered.null_bound_snapshotter
        ):
            raise V2IntegrityError("resolver-owned bound registration snapshot drifted")
        if (
            plan_canonical_bytes != registered.plan_canonical_bytes
            or plan_sha256 != registered.plan_sha256
            or statistic_type is not registered.statistic_implementation_type
            or null_type is not registered.null_implementation_type
            or statistic_parameters != registered.statistic_parameters_bytes
            or null_parameters != registered.null_parameters_bytes
            or not _exact_snapshot_value_matches(
                statistic_operational,
                registered.statistic_operational_snapshot,
            )
            or not _exact_snapshot_value_matches(
                null_operational,
                registered.null_operational_snapshot,
            )
            or not _exact_snapshot_value_matches(
                statistic_descriptors,
                registered.statistic_descriptor_snapshot,
            )
            or not _exact_snapshot_value_matches(
                null_descriptors,
                registered.null_descriptor_snapshot,
            )
        ):
            raise V2IntegrityError("resolver-owned pure identity snapshot drifted")
        return exact

    return _require_resolver_owned_resolution_v2_pure


def _freeze_context_snapshot_v2(
    _require_resolver_owned_resolution_v2_pure: Callable[[object], PlanResolutionV2],
    ownership_lookup: _ResolutionOwnershipLookupV2,
    /,
) -> Callable[[object], _ResultVerificationContextV2]:
    def _snapshot_result_verification_context_v2(
        resolution: object,
        /,
    ) -> _ResultVerificationContextV2:
        exact = _require_resolver_owned_resolution_v2_pure(resolution)
        plan = exact.plan
        registered = ownership_lookup(exact)
        identity_operation = registered.statistic_result_identity
        token_operation = registered.result_token_verifier
        backend_identity, preprocessing_identity = identity_operation(exact.adapters.statistic)
        if (
            type(backend_identity) is not str
            or not backend_identity
            or type(preprocessing_identity) is not str
            or not preprocessing_identity
        ):
            raise V2IntegrityError("registered statistic result identity is invalid")
        plan_sha256 = scientific_plan_v2_sha256(plan)
        return _ResultVerificationContextV2(
            plan=plan,
            plan_sha256=plan_sha256,
            backend_identity=backend_identity,
            preprocessing_identity=preprocessing_identity,
            null_adapter=exact.adapters.null_model,
            verify_token=token_operation,
        )

    return _snapshot_result_verification_context_v2


def _freeze_bound_statistic_snapshot_v2(
    _require_resolver_owned_resolution_v2_pure: Callable[[object], PlanResolutionV2],
    ownership_lookup: _ResolutionOwnershipLookupV2,
    /,
) -> _BoundStatisticSnapshotEntrypointV2:
    def _snapshot_registered_bound_statistic_v2(
        resolution: PlanResolutionV2,
        bound: object,
        *,
        expected_pair: SeriesPair,
        initial: _BoundStatisticExecutionSnapshotV2 | None = None,
    ) -> _BoundStatisticExecutionSnapshotV2:
        try:
            exact_resolution = _require_resolver_owned_resolution_v2_pure(resolution)
            registered = ownership_lookup(exact_resolution)
            expected_parameters = exact_resolution.plan.statistic_params
        except V2IntegrityError as error:
            raise V2IntegrityError(
                "bound statistic registry snapshot or execution snapshot is invalid or drifted"
            ) from error
        if type(bound) is not registered.statistic_bound_implementation_type:
            raise V2IntegrityError("bound statistic is not the exact registered implementation")
        snapshotter = registered.statistic_bound_snapshotter
        return snapshotter(bound, expected_pair, expected_parameters, initial)

    return _snapshot_registered_bound_statistic_v2


def _freeze_bound_null_snapshot_v2(
    _require_resolver_owned_resolution_v2_pure: Callable[[object], PlanResolutionV2],
    ownership_lookup: _ResolutionOwnershipLookupV2,
    /,
) -> _BoundNullSnapshotEntrypointV2:
    def _snapshot_registered_bound_null_v2(
        resolution: PlanResolutionV2,
        bound: object,
        *,
        expected_pair: SeriesPair,
        initial: _BoundNullExecutionSnapshotV2 | None = None,
    ) -> _BoundNullExecutionSnapshotV2:
        try:
            exact_resolution = _require_resolver_owned_resolution_v2_pure(resolution)
            registered = ownership_lookup(exact_resolution)
            expected_parameters = exact_resolution.plan.null_params
        except V2IntegrityError as error:
            raise V2IntegrityError(
                "bound null registry snapshot or execution snapshot is invalid or drifted"
            ) from error
        if type(bound) is not registered.null_bound_implementation_type:
            raise V2IntegrityError("bound null is not the exact registered implementation")
        snapshotter = registered.null_bound_snapshotter
        return snapshotter(bound, expected_pair, expected_parameters, initial)

    return _snapshot_registered_bound_null_v2


def _freeze_resolve_plan_v2(
    records: dict[int, _ResolutionIdentityRecordV2],
    snapshotter: Callable[[PlanResolutionV2], _ResolutionOwnershipSnapshotV2],
    /,
) -> _ResolvePlanEntrypointV2:
    exact_type = type
    identity = id
    tuple_type = tuple
    tuple_new = tuple.__new__
    tuple_len = tuple.__len__
    tuple_getitem: Any = tuple.__getitem__
    (
        ownership_plan,
        ownership_adapters,
        ownership_statistic,
        ownership_null,
        ownership_plan_bytes,
        ownership_plan_sha256,
        ownership_statistic_type,
        ownership_null_type,
        ownership_statistic_parameters,
        ownership_null_parameters,
        ownership_statistic_operational,
        ownership_null_operational,
        ownership_statistic_descriptors,
        ownership_null_descriptors,
        ownership_statistic_registry,
        ownership_null_registry,
        ownership_statistic_registration,
        ownership_null_registration,
        ownership_statistic_result_identity,
        ownership_token_verifier,
        ownership_statistic_bound_type,
        ownership_null_bound_type,
        ownership_statistic_bound_snapshotter,
        ownership_null_bound_snapshotter,
    ) = range(24)
    record_type = _ResolutionIdentityRecordV2
    snapshot_type = _ResolutionOwnershipSnapshotV2
    reference_type = ReferenceType
    reference_factory = ref
    integrity_error_type = V2IntegrityError
    exact_snapshot_value_matches = _exact_snapshot_value_matches
    registry_snapshot_matches = _adapter_registry_snapshot_matches

    def resolve_plan_v2(request: PlanRequestV2, /) -> PlanResolutionV2:
        """Resolve one exact v2 request through only the built-in v2 registries."""

        if type(request) is not PlanRequestV2:
            raise TypeError("request must be an exact PlanRequestV2")
        _require_canonical_adapter_registry(role="statistic")
        statistic, _ = _verified_statistic(
            _STATISTIC_REGISTRY_V2.build(request.statistic_name, request.statistic_params),
            expected_name=request.statistic_name,
            expected_params=request.statistic_params,
        )
        _require_canonical_adapter_registry(role="null")
        null_model, _ = _verified_null(
            _NULL_REGISTRY_V2.build(request.null_name, request.null_params),
            expected_name=request.null_name,
            expected_params=request.null_params,
        )
        _require_canonical_adapter_registry(role="statistic")
        _require_canonical_adapter_registry(role="null")
        plan = ResolvedScientificPlanV2(
            candidates=request.candidates,
            statistic_name=request.statistic_name,
            statistic_params=request.statistic_params,
            selection_rule=request.selection_rule,
            null_name=request.null_name,
            null_params=request.null_params,
            replicates=request.replicates,
            alpha=request.alpha,
            tie_tolerance=request.tie_tolerance,
            root_seed=request.root_seed,
            seal=_V2_RESOLUTION_SEAL,
        )
        adapters = ResolvedAdaptersV2(statistic=statistic, null_model=null_model)
        resolution = PlanResolutionV2(
            plan=plan,
            adapters=adapters,
            seal=_V2_RESOLUTION_SEAL,
        )
        snapshot = snapshotter(resolution)
        confirmation = snapshotter(resolution)
        if (
            exact_type(snapshot) is not snapshot_type
            or exact_type(confirmation) is not snapshot_type
        ):
            raise integrity_error_type("resolution ownership snapshot is invalid")
        if (
            tuple_getitem(confirmation, ownership_plan)
            is not tuple_getitem(snapshot, ownership_plan)
            or tuple_getitem(confirmation, ownership_adapters)
            is not tuple_getitem(snapshot, ownership_adapters)
            or tuple_getitem(confirmation, ownership_statistic)
            is not tuple_getitem(snapshot, ownership_statistic)
            or tuple_getitem(confirmation, ownership_null)
            is not tuple_getitem(snapshot, ownership_null)
            or tuple_getitem(confirmation, ownership_plan_bytes)
            != tuple_getitem(snapshot, ownership_plan_bytes)
            or tuple_getitem(confirmation, ownership_plan_sha256)
            != tuple_getitem(snapshot, ownership_plan_sha256)
            or tuple_getitem(confirmation, ownership_statistic_type)
            is not tuple_getitem(snapshot, ownership_statistic_type)
            or tuple_getitem(confirmation, ownership_null_type)
            is not tuple_getitem(snapshot, ownership_null_type)
            or tuple_getitem(confirmation, ownership_statistic_parameters)
            != tuple_getitem(snapshot, ownership_statistic_parameters)
            or tuple_getitem(confirmation, ownership_null_parameters)
            != tuple_getitem(snapshot, ownership_null_parameters)
            or not exact_snapshot_value_matches(
                tuple_getitem(confirmation, ownership_statistic_operational),
                tuple_getitem(snapshot, ownership_statistic_operational),
            )
            or not exact_snapshot_value_matches(
                tuple_getitem(confirmation, ownership_null_operational),
                tuple_getitem(snapshot, ownership_null_operational),
            )
            or not exact_snapshot_value_matches(
                tuple_getitem(confirmation, ownership_statistic_descriptors),
                tuple_getitem(snapshot, ownership_statistic_descriptors),
            )
            or not exact_snapshot_value_matches(
                tuple_getitem(confirmation, ownership_null_descriptors),
                tuple_getitem(snapshot, ownership_null_descriptors),
            )
            or not registry_snapshot_matches(
                tuple_getitem(confirmation, ownership_statistic_registry),
                tuple_getitem(snapshot, ownership_statistic_registry),
            )
            or not registry_snapshot_matches(
                tuple_getitem(confirmation, ownership_null_registry),
                tuple_getitem(snapshot, ownership_null_registry),
            )
            or tuple_getitem(confirmation, ownership_statistic_registration)
            is not tuple_getitem(snapshot, ownership_statistic_registration)
            or tuple_getitem(confirmation, ownership_null_registration)
            is not tuple_getitem(snapshot, ownership_null_registration)
            or tuple_getitem(confirmation, ownership_statistic_result_identity)
            is not tuple_getitem(snapshot, ownership_statistic_result_identity)
            or tuple_getitem(confirmation, ownership_token_verifier)
            is not tuple_getitem(snapshot, ownership_token_verifier)
            or tuple_getitem(confirmation, ownership_statistic_bound_type)
            is not tuple_getitem(snapshot, ownership_statistic_bound_type)
            or tuple_getitem(confirmation, ownership_null_bound_type)
            is not tuple_getitem(snapshot, ownership_null_bound_type)
            or tuple_getitem(confirmation, ownership_statistic_bound_snapshotter)
            is not tuple_getitem(snapshot, ownership_statistic_bound_snapshotter)
            or tuple_getitem(confirmation, ownership_null_bound_snapshotter)
            is not tuple_getitem(snapshot, ownership_null_bound_snapshotter)
        ):
            raise integrity_error_type("resolution ownership snapshot drifted before registration")
        resolution_identity = identity(resolution)

        def discard_identity_record(dead_reference: object, /) -> None:
            current: Any = records.get(resolution_identity)
            current_reference: Any = None
            if exact_type(current) is record_type:
                if tuple_len(current) >= 1:
                    current_reference = tuple_getitem(current, 0)
            elif exact_type(current) is tuple_type and tuple_len(current) >= 1:
                current_reference = tuple_getitem(current, 0)
            if current_reference is dead_reference:
                records.pop(resolution_identity, None)

        reference = reference_factory(resolution, discard_identity_record)
        record = tuple_new(record_type, (reference, snapshot))
        if (
            exact_type(record) is not record_type
            or tuple_len(record) != 2
            or exact_type(tuple_getitem(record, 0)) is not reference_type
            or exact_type(tuple_getitem(record, 1)) is not snapshot_type
        ):
            raise integrity_error_type("resolution ownership record construction failed")
        exact_record: Any = record
        records[resolution_identity] = exact_record
        return resolution

    return resolve_plan_v2


def _freeze_resolution_entrypoints_v2(
    ownership_lookup: _ResolutionOwnershipLookupV2,
    ownership_records: dict[int, _ResolutionIdentityRecordV2],
    snapshotter: Callable[[PlanResolutionV2], _ResolutionOwnershipSnapshotV2],
    require_owned_builder: Callable[
        [
            _ResolutionOwnershipLookupV2,
            Callable[[PlanResolutionV2], _ResolutionOwnershipSnapshotV2],
        ],
        Callable[[object], PlanResolutionV2],
    ],
    require_owned_pure_builder: Callable[
        [_ResolutionOwnershipLookupV2],
        Callable[[object], PlanResolutionV2],
    ],
    context_builder: Callable[
        [Callable[[object], PlanResolutionV2], _ResolutionOwnershipLookupV2],
        Callable[[object], _ResultVerificationContextV2],
    ],
    bound_statistic_builder: Callable[
        [Callable[[object], PlanResolutionV2], _ResolutionOwnershipLookupV2],
        _BoundStatisticSnapshotEntrypointV2,
    ],
    bound_null_builder: Callable[
        [Callable[[object], PlanResolutionV2], _ResolutionOwnershipLookupV2],
        _BoundNullSnapshotEntrypointV2,
    ],
    resolve_builder: Callable[
        [
            dict[int, _ResolutionIdentityRecordV2],
            Callable[[PlanResolutionV2], _ResolutionOwnershipSnapshotV2],
        ],
        _ResolvePlanEntrypointV2,
    ],
    /,
) -> _ResolutionEntrypointsV2:
    """Publish the six ownership entrypoints over one frozen lookup."""

    require_owned = require_owned_builder(ownership_lookup, snapshotter)
    require_owned_pure = require_owned_pure_builder(ownership_lookup)
    snapshot_context = context_builder(
        require_owned_pure,
        ownership_lookup,
    )
    snapshot_bound_statistic = bound_statistic_builder(
        require_owned_pure,
        ownership_lookup,
    )
    snapshot_bound_null = bound_null_builder(
        require_owned_pure,
        ownership_lookup,
    )
    resolve = resolve_builder(
        ownership_records,
        snapshotter,
    )
    return _ResolutionEntrypointsV2(
        require_owned=require_owned,
        require_owned_pure=require_owned_pure,
        snapshot_context=snapshot_context,
        snapshot_bound_statistic=snapshot_bound_statistic,
        snapshot_bound_null=snapshot_bound_null,
        resolve=resolve,
    )


_resolution_entrypoints_v2 = _freeze_resolution_entrypoints_v2(
    _LOOKUP_RESOLUTION_OWNERSHIP_V2,
    _resolver_owned_by_id_bootstrap,
    _capture_resolution_snapshot_fail_closed,
    _freeze_require_owned_v2,
    _freeze_require_owned_pure_v2,
    _freeze_context_snapshot_v2,
    _freeze_bound_statistic_snapshot_v2,
    _freeze_bound_null_snapshot_v2,
    _freeze_resolve_plan_v2,
)
del _resolver_owned_by_id_bootstrap
_require_resolver_owned_resolution_v2 = _resolution_entrypoints_v2.require_owned
_require_resolver_owned_resolution_v2_pure = _resolution_entrypoints_v2.require_owned_pure
_snapshot_result_verification_context_v2 = _resolution_entrypoints_v2.snapshot_context
_snapshot_registered_bound_statistic_v2 = _resolution_entrypoints_v2.snapshot_bound_statistic
_snapshot_registered_bound_null_v2 = _resolution_entrypoints_v2.snapshot_bound_null
resolve_plan_v2 = _resolution_entrypoints_v2.resolve
for _entrypoint_name_v2, _entrypoint_v2 in (
    ("_require_resolver_owned_resolution_v2", _require_resolver_owned_resolution_v2),
    ("_require_resolver_owned_resolution_v2_pure", _require_resolver_owned_resolution_v2_pure),
    ("_snapshot_result_verification_context_v2", _snapshot_result_verification_context_v2),
    ("_snapshot_registered_bound_statistic_v2", _snapshot_registered_bound_statistic_v2),
    ("_snapshot_registered_bound_null_v2", _snapshot_registered_bound_null_v2),
    ("resolve_plan_v2", resolve_plan_v2),
):
    _exact_entrypoint_v2: Any = _entrypoint_v2
    _exact_entrypoint_v2.__module__ = __name__
    _exact_entrypoint_v2.__name__ = _entrypoint_name_v2
    _exact_entrypoint_v2.__qualname__ = _entrypoint_name_v2
del _entrypoint_name_v2, _entrypoint_v2, _exact_entrypoint_v2
del (
    _freeze_require_owned_v2,
    _freeze_require_owned_pure_v2,
    _freeze_context_snapshot_v2,
    _freeze_bound_statistic_snapshot_v2,
    _freeze_bound_null_snapshot_v2,
    _freeze_resolve_plan_v2,
    _freeze_resolution_entrypoints_v2,
    _resolution_entrypoints_v2,
)


_result_verification_context_v2 = _freeze_result_verification_context_reader_v2(
    _RESULT_VERIFIER_PLAN_CANONICAL_BYTES_V2,
    _RESULT_VERIFIER_PLAN_SHA256_V2,
    _SEALED_STATISTIC_SNAPSHOTS_V2,
    _SEALED_STATISTIC_RESULT_IDENTITIES_V2,
    _SEALED_NULL_SNAPSHOTS_V2,
    _SEALED_RESULT_TOKEN_VERIFIERS_V2,
)
_random_capsule_getitem_v2 = cast(
    Callable[[tuple[Any, ...], int], Any],
    tuple.__getitem__,
)
_contracts_v2_module._bootstrap_result_verifier_v2(
    _RESULT_VERIFIER_PLAN_SHA256_V2,
    _result_verification_context_v2,
    _random_capsule_getitem_v2(_RANDOM_CAPSULE_V2, 3),
)
_OWNERSHIP_OPERATION_SENTINEL_V2: Any = object()
_REGISTER_RESOLUTION_OWNERSHIP_V2 = _OWNERSHIP_OPERATION_SENTINEL_V2
_LOOKUP_RESOLUTION_OWNERSHIP_V2 = _OWNERSHIP_OPERATION_SENTINEL_V2
delattr(_contracts_v2_module, "_bootstrap_result_verifier_v2")
delattr(_contracts_v2_module, "_freeze_result_verifier_capsule_v2")
delattr(_contracts_v2_module, "_RESULT_VERIFIER_EXTERNAL_LEAVES_V2")
delattr(_contracts_v2_module, "_RESULT_VERIFIER_CONTRACT_LEAVES_V2")
del _freeze_result_verification_context_reader_v2
del _result_verification_context_v2, _random_capsule_getitem_v2, _RANDOM_CAPSULE_V2
