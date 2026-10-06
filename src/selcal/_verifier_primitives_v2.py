"""Internal, sealed verification leaves; never used by result construction."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sized
from types import MemberDescriptorType
from typing import Any, NamedTuple, Protocol, cast


class ExactSlotSchemaV2(NamedTuple):
    owner_type: type[object]
    slots: tuple[tuple[str, MemberDescriptorType], ...]


class VerifierTypesV2(NamedTuple):
    calibration_result_type: type[object]
    replicate_outcome_type: type[object]
    statistic_result_type: type[object]
    selection_result_type: type[object]
    selection_rule_type: type[object]
    max_upper_member: object
    validity_type: type[object]
    valid_member: object
    analytic_failure_member: object
    run_status_type: type[object]
    run_complete_member: object
    run_not_evaluable_member: object
    run_failure_stage_type: type[object]
    null_bind_member: object
    observed_scan_member: object
    replicate_execution_member: object
    replicate_status_type: type[object]
    replicate_complete_member: object
    replicate_failure_member: object
    replicate_failure_stage_type: type[object]
    statistic_scan_member: object


class VerifierSchemasV2(NamedTuple):
    calibration_result: ExactSlotSchemaV2
    replicate_outcome: ExactSlotSchemaV2
    statistic_result: ExactSlotSchemaV2
    selection_result: ExactSlotSchemaV2


class SeedDigestV2(Protocol):
    def __call__(self, plan_digest: str, replicate_id: int, replicates: int, /) -> str: ...


class IsFiniteV2(Protocol):
    def __call__(self, value: float, /) -> bool: ...


class ZipValuesV2(Protocol):
    def __call__(
        self, left: tuple[object, ...], right: tuple[object, ...], *, strict: bool
    ) -> Iterable[tuple[object, object]]: ...


class VerifierExternalOpsV2(NamedTuple):
    integrity_error_type: type[Exception]
    exact_type: Callable[[object], type[object]]
    tuple_factory: Callable[[Iterable[object]], tuple[object, ...]]
    list_factory: Callable[[], list[object]]
    any_value: Callable[[Iterable[object]], bool]
    length: Callable[[Sized], int]
    maximum: Callable[[tuple[float, ...]], float]
    enumerate_values: Callable[[tuple[object, ...]], Iterable[tuple[int, object]]]
    zip_values: ZipValuesV2
    absolute: Callable[[float], float]
    isfinite: IsFiniteV2
    structural_errors: tuple[type[Exception], ...]
    seed_digest: SeedDigestV2


class VectorVerificationPolicyV2(NamedTuple):
    candidates: tuple[int, ...]
    selection_rule: object
    max_upper_member: object
    tie_tolerance: float
    backend_identity: str
    preprocessing_identity: str


class ReplicateExpectationV2(NamedTuple):
    replicate_id: int
    plan_digest: str
    planned_replicates: int
    observed_support_n: int


class TerminalEvidenceV2(NamedTuple):
    status: object
    failure_stage: object
    observed_results: object
    observed_selection: object
    replicates: tuple[object, ...]
    exceedance_count: object
    failure_count: object
    p_value: object
    bound_low: object
    bound_high: object
    reject_null: object
    planned_replicates: int
    alpha: float


class DerivedCountsV2(NamedTuple):
    exceedances: int
    failures: int
    retained_replicates: int


class VerifiedVectorEvidenceV2(NamedTuple):
    support_n: int
    decision_statistic: float | None
    analytic_failure_count: int


class VerifiedReplicateEvidenceV2(NamedTuple):
    support_n: int
    decision_statistic: float | None
    is_failure: bool


class PendingReplicateEvidenceV2(NamedTuple):
    support_n: int
    pre_token_decision_statistic: float | None
    is_failure: bool
    selection: object
    transform_token: object
    replicate_status: object
    replicate_failure_stage: object


class GrandfatheredExternalFrontierV2(NamedTuple):
    role: str
    module: str
    qualname: str
    source_path: str
    source_sha256: str
    first_line: int
    source_segment_sha256: str


class RequireSha256V2(Protocol):
    def __call__(self, value: object, *, name: str) -> str: ...


class RequireDiagnosticsV2(Protocol):
    def __call__(self, value: object, *, subject: str) -> None: ...


class ReadExactSlotsV2(Protocol):
    def __call__(
        self, value: object, schema: ExactSlotSchemaV2, *, subject: str
    ) -> tuple[object, ...]: ...


class VerifySelectionV2(Protocol):
    def __call__(
        self,
        selection: object,
        policy: VectorVerificationPolicyV2,
        scores: tuple[float, ...],
        *,
        subject: str,
    ) -> float: ...


class VerifyCompleteVectorV2(Protocol):
    def __call__(
        self,
        vector: object,
        selection: object,
        policy: VectorVerificationPolicyV2,
        *,
        subject: str,
    ) -> VerifiedVectorEvidenceV2: ...


class VerifyFailureVectorV2(Protocol):
    def __call__(
        self,
        vector: object,
        selection: object,
        policy: VectorVerificationPolicyV2,
        *,
        subject: str,
    ) -> VerifiedVectorEvidenceV2: ...


class VerifyReplicatePrefixV2(Protocol):
    def __call__(
        self,
        outcome: object,
        expected: ReplicateExpectationV2,
        policy: VectorVerificationPolicyV2,
        *,
        subject: str,
    ) -> PendingReplicateEvidenceV2: ...


class VerifyReplicateSuffixV2(Protocol):
    def __call__(
        self,
        pending: PendingReplicateEvidenceV2,
        *,
        subject: str,
    ) -> VerifiedReplicateEvidenceV2: ...


class VerifyTerminalV2(Protocol):
    def __call__(
        self,
        terminal: TerminalEvidenceV2,
        derived: DerivedCountsV2,
        *,
        subject: str,
    ) -> None: ...


class VerifierPrimitiveOpsV2(NamedTuple):
    require_sha256: RequireSha256V2
    require_diagnostics: RequireDiagnosticsV2
    read_exact_slots: ReadExactSlotsV2
    verify_selection: VerifySelectionV2
    verify_complete_vector: VerifyCompleteVectorV2
    verify_failure_vector: VerifyFailureVectorV2
    verify_replicate_prefix: VerifyReplicatePrefixV2
    verify_replicate_suffix: VerifyReplicateSuffixV2
    verify_terminal: VerifyTerminalV2


def _freeze_sha256(
    externals: VerifierExternalOpsV2, /, str_type: type[str] = str
) -> RequireSha256V2:
    cast_value = cast

    tuple_getitem: Callable[[tuple[object, ...], int], Any] = tuple.__getitem__

    def require_sha256(value: object, *, name: str) -> str:
        if (
            tuple_getitem(externals, 1)(value) is not str_type
            or tuple_getitem(externals, 5)(cast_value("str", value)) != 64
            or tuple_getitem(externals, 4)(
                character not in "0123456789abcdef" for character in cast_value("str", value)
            )
        ):
            raise tuple_getitem(externals, 0)(f"{name} must be a lowercase SHA-256 hex digest")
        exact_value: str = cast_value("str", value)
        return exact_value

    return require_sha256


def _freeze_diagnostics(
    externals: VerifierExternalOpsV2,
    /,
    tuple_type: type[tuple[object, ...]] = tuple,
    str_type: type[str] = str,
) -> RequireDiagnosticsV2:
    cast_value = cast

    tuple_getitem: Callable[[tuple[object, ...], int], Any] = tuple.__getitem__

    def require_diagnostics(value: object, *, subject: str) -> None:
        if tuple_getitem(externals, 1)(value) is not tuple_type:
            raise tuple_getitem(externals, 0)(
                f"{subject} diagnostics must be an exact string tuple"
            )
        if tuple_getitem(externals, 4)(
            tuple_getitem(externals, 1)(item) is not str_type
            for item in cast_value("tuple[object, ...]", value)
        ):
            raise tuple_getitem(externals, 0)(
                f"{subject} diagnostics must be an exact string tuple"
            )

    return require_diagnostics


def _freeze_slot_reader(externals: VerifierExternalOpsV2, /) -> ReadExactSlotsV2:
    tuple_getitem: Callable[[tuple[object, ...], int], Any] = tuple.__getitem__

    def read_exact_slots(
        value: object,
        schema: ExactSlotSchemaV2,
        *,
        subject: str,
    ) -> tuple[object, ...]:
        if tuple_getitem(externals, 1)(value) is not tuple_getitem(schema, 0):
            raise tuple_getitem(externals, 0)(f"{subject} must have its exact canonical type")
        # BUILD_LIST reuses CPython's bounded freelist; list() does not consume it.
        values: list[object] = []
        for name, descriptor in tuple_getitem(schema, 1):
            if tuple_getitem(schema, 0).__dict__.get(name) is not descriptor:
                raise tuple_getitem(externals, 0)(f"{subject} slot descriptor drifted: {name}")
            try:
                values.append(descriptor.__get__(value, tuple_getitem(schema, 0)))
            except tuple_getitem(externals, 11) as error:
                raise tuple_getitem(externals, 0)(f"{subject} slot snapshot is invalid") from error
        snapshot: tuple[object, ...] = tuple_getitem(externals, 2)(values)
        return snapshot

    return read_exact_slots


def _freeze_selection(
    schemas: VerifierSchemasV2,
    externals: VerifierExternalOpsV2,
    read: ReadExactSlotsV2,
    /,
    scalar_types: tuple[type[int], type[float], type[tuple[object, ...]]] = (int, float, tuple),
) -> VerifySelectionV2:
    int_type, float_type, tuple_type = scalar_types
    cast_value = cast

    tuple_getitem: Callable[[tuple[object, ...], int], Any] = tuple.__getitem__

    def verify_selection(
        selection: object,
        policy: VectorVerificationPolicyV2,
        scores: tuple[float, ...],
        *,
        subject: str,
    ) -> float:
        candidate, index, statistic, ties = read(
            selection,
            tuple_getitem(schemas, 3),
            subject=f"{subject} selection",
        )
        decision = tuple_getitem(externals, 6)(scores)
        tied_indices = cast_value(
            "tuple[int, ...]",
            tuple_getitem(externals, 2)(
                index
                for index, score in tuple_getitem(externals, 7)(scores)
                if decision - cast_value("float", score) <= tuple_getitem(policy, 3)
            ),
        )
        expected_ties = tuple_getitem(externals, 2)(
            tuple_getitem(policy, 0)[index] for index in tied_indices
        )
        expected_index = tied_indices[0]
        if (
            tuple_getitem(externals, 1)(candidate) is not int_type
            or candidate != tuple_getitem(policy, 0)[expected_index]
            or tuple_getitem(externals, 1)(index) is not int_type
            or index != expected_index
            or tuple_getitem(externals, 1)(statistic) is not float_type
            or statistic != decision
            or tuple_getitem(externals, 1)(ties) is not tuple_type
            or ties != expected_ties
        ):
            raise tuple_getitem(externals, 0)(
                f"{subject} selection contradicts central reselection"
            )
        exact_decision: float = decision
        return exact_decision

    return verify_selection


def _freeze_vector_reader(
    types: VerifierTypesV2,
    schemas: VerifierSchemasV2,
    externals: VerifierExternalOpsV2,
    read: ReadExactSlotsV2,
    diagnostics: RequireDiagnosticsV2,
    /,
    scalar_types: tuple[type[int], type[str]] = (int, str),
) -> Callable[[object, int, int | None, VectorVerificationPolicyV2, str], tuple[object, ...]]:
    int_type, str_type = scalar_types
    cast_value = cast

    tuple_getitem: Callable[[tuple[object, ...], int], Any] = tuple.__getitem__

    def read_statistic(
        item: object,
        expected_candidate: int,
        support_n: int | None,
        policy: VectorVerificationPolicyV2,
        subject: str,
    ) -> tuple[object, ...]:
        values = read(item, tuple_getitem(schemas, 2), subject=f"{subject} statistic")
        candidate, _estimate, _score, support, validity, diagnostic, backend, preprocessing = values
        diagnostics(diagnostic, subject=f"{subject} statistic")
        if (
            tuple_getitem(externals, 1)(candidate) is not int_type
            or candidate != expected_candidate
        ):
            raise tuple_getitem(externals, 0)(f"{subject} candidate IDs contradict the plan")
        if tuple_getitem(externals, 1)(support) is not int_type or cast_value("int", support) < 1:
            raise tuple_getitem(externals, 0)(f"{subject} support_n is invalid")
        if support_n is not None and support != support_n:
            raise tuple_getitem(externals, 0)(f"{subject} violates common support")
        if (
            tuple_getitem(externals, 1)(backend) is not str_type
            or backend != tuple_getitem(policy, 4)
            or tuple_getitem(externals, 1)(preprocessing) is not str_type
            or preprocessing != tuple_getitem(policy, 5)
        ):
            raise tuple_getitem(externals, 0)(
                f"{subject} statistic implementation identity drifted"
            )
        if tuple_getitem(externals, 1)(validity) is not tuple_getitem(types, 6):
            raise tuple_getitem(externals, 0)(f"{subject} validity is not canonical")
        return values

    return read_statistic


def _freeze_valid_score(
    externals: VerifierExternalOpsV2,
    /,
    float_type: type[float] = float,
) -> Callable[[object, VectorVerificationPolicyV2, str], float]:
    cast_value = cast

    tuple_getitem: Callable[[tuple[object, ...], int], Any] = tuple.__getitem__

    def valid_score(estimate: object, policy: VectorVerificationPolicyV2, subject: str) -> float:
        if tuple_getitem(externals, 1)(estimate) is not float_type or not tuple_getitem(
            externals, 10
        )(cast_value("float", estimate)):
            raise tuple_getitem(externals, 0)(f"{subject} valid estimate is invalid")
        exact_estimate = cast_value("float", estimate)
        score = (
            exact_estimate
            if tuple_getitem(policy, 1) is tuple_getitem(policy, 2)
            else tuple_getitem(externals, 9)(exact_estimate)
        )
        return 0.0 if score == 0.0 else score

    return valid_score


def _freeze_vector_shape(
    externals: VerifierExternalOpsV2,
    /,
    tuple_type: type[tuple[object, ...]] = tuple,
) -> Callable[[object, VectorVerificationPolicyV2, str], tuple[object, ...]]:
    cast_value = cast

    tuple_getitem: Callable[[tuple[object, ...], int], Any] = tuple.__getitem__

    def vector_shape(
        vector: object,
        policy: VectorVerificationPolicyV2,
        subject: str,
    ) -> tuple[object, ...]:
        if tuple_getitem(externals, 1)(vector) is not tuple_type:
            raise tuple_getitem(externals, 0)(f"{subject} must be an exact tuple")
        exact_vector: tuple[object, ...] = cast_value("tuple[object, ...]", vector)
        if tuple_getitem(externals, 5)(exact_vector) != tuple_getitem(externals, 5)(
            tuple_getitem(policy, 0)
        ):
            raise tuple_getitem(externals, 0)(
                f"{subject} must retain the complete candidate family"
            )
        return exact_vector

    return vector_shape


def _freeze_complete_vector(
    types: VerifierTypesV2,
    externals: VerifierExternalOpsV2,
    read_item: Callable[
        [object, int, int | None, VectorVerificationPolicyV2, str], tuple[object, ...]
    ],
    shape: Callable[[object, VectorVerificationPolicyV2, str], tuple[object, ...]],
    score_of: Callable[[object, VectorVerificationPolicyV2, str], float],
    select: VerifySelectionV2,
    /,
    evidence_type: type[VerifiedVectorEvidenceV2] = VerifiedVectorEvidenceV2,
    float_type: type[float] = float,
) -> VerifyCompleteVectorV2:
    cast_value = cast
    construct_evidence = evidence_type.__new__

    tuple_getitem: Callable[[tuple[object, ...], int], Any] = tuple.__getitem__

    def verify_complete_vector(
        vector: object,
        selection: object,
        policy: VectorVerificationPolicyV2,
        *,
        subject: str,
    ) -> VerifiedVectorEvidenceV2:
        exact_vector = shape(vector, policy, subject)
        support: int | None = None
        scores: list[object] = []
        failure_count = 0
        for candidate, item in tuple_getitem(externals, 8)(
            tuple_getitem(policy, 0), exact_vector, strict=True
        ):
            values = read_item(item, cast_value("int", candidate), support, policy, subject)
            (
                _candidate,
                estimate,
                selection_score,
                item_support,
                validity,
                _diagnostic,
                _backend,
                _pre,
            ) = values
            support = cast_value("int", item_support)
            if validity is tuple_getitem(types, 8):
                failure_count += 1
                if estimate is not None or selection_score is not None:
                    raise tuple_getitem(externals, 0)(
                        f"{subject} analytical failure must remain raw"
                    )
                continue
            score = score_of(estimate, policy, subject)
            if (
                tuple_getitem(externals, 1)(selection_score) is not float_type
                or selection_score != score
            ):
                raise tuple_getitem(externals, 0)(
                    f"{subject} selection score contradicts the plan rule"
                )
            scores.append(score)
        assert support is not None
        if (
            failure_count
            or tuple_getitem(externals, 5)(scores)
            != tuple_getitem(externals, 5)(tuple_getitem(policy, 0))
            or selection is None
        ):
            raise tuple_getitem(externals, 0)(f"{subject} complete vector is invalid")
        decision = select(
            selection,
            policy,
            cast_value("tuple[float, ...]", tuple_getitem(externals, 2)(scores)),
            subject=subject,
        )
        return construct_evidence(evidence_type, support, decision, 0)

    return verify_complete_vector


def _freeze_failure_vector(
    types: VerifierTypesV2,
    externals: VerifierExternalOpsV2,
    read_item: Callable[
        [object, int, int | None, VectorVerificationPolicyV2, str], tuple[object, ...]
    ],
    shape: Callable[[object, VectorVerificationPolicyV2, str], tuple[object, ...]],
    score_of: Callable[[object, VectorVerificationPolicyV2, str], float],
    /,
    evidence_type: type[VerifiedVectorEvidenceV2] = VerifiedVectorEvidenceV2,
) -> VerifyFailureVectorV2:
    cast_value = cast
    construct_evidence = evidence_type.__new__

    tuple_getitem: Callable[[tuple[object, ...], int], Any] = tuple.__getitem__

    def verify_failure_vector(
        vector: object,
        selection: object,
        policy: VectorVerificationPolicyV2,
        *,
        subject: str,
    ) -> VerifiedVectorEvidenceV2:
        exact_vector = shape(vector, policy, subject)
        support: int | None = None
        failure_count = 0
        for candidate, item in tuple_getitem(externals, 8)(
            tuple_getitem(policy, 0), exact_vector, strict=True
        ):
            values = read_item(item, cast_value("int", candidate), support, policy, subject)
            (
                _candidate,
                estimate,
                selection_score,
                item_support,
                validity,
                _diagnostic,
                _backend,
                _pre,
            ) = values
            support = cast_value("int", item_support)
            if validity is tuple_getitem(types, 8):
                failure_count += 1
                if estimate is not None or selection_score is not None:
                    raise tuple_getitem(externals, 0)(
                        f"{subject} analytical failure must remain raw"
                    )
                continue
            score_of(estimate, policy, subject)
            if selection_score is not None:
                raise tuple_getitem(externals, 0)(f"{subject} failure vector must remain unscored")
        assert support is not None
        if failure_count < 1 or selection is not None:
            raise tuple_getitem(externals, 0)(f"{subject} must retain an analytical failure")
        return construct_evidence(evidence_type, support, None, failure_count)

    return verify_failure_vector


def _freeze_replicate_identity(
    types: VerifierTypesV2,
    externals: VerifierExternalOpsV2,
    /,
    scalar_types: tuple[type[int], type[str]] = (int, str),
) -> Callable[[tuple[object, ...], ReplicateExpectationV2, str], None]:
    int_type, str_type = scalar_types

    tuple_getitem: Callable[[tuple[object, ...], int], Any] = tuple.__getitem__

    def verify_identity(
        values: tuple[object, ...], expected: ReplicateExpectationV2, subject: str
    ) -> None:
        replicate_id, seed_digest, status, stage, _token, _vector, _selection, _diagnostics = values
        expected_seed = tuple_getitem(externals, 12)(
            tuple_getitem(expected, 1), tuple_getitem(expected, 0), tuple_getitem(expected, 2)
        )
        if (
            tuple_getitem(externals, 1)(replicate_id) is not int_type
            or replicate_id != tuple_getitem(expected, 0)
            or tuple_getitem(externals, 1)(seed_digest) is not str_type
            or seed_digest != expected_seed
            or tuple_getitem(externals, 1)(status) is not tuple_getitem(types, 16)
            or (
                stage is not None
                and tuple_getitem(externals, 1)(stage) is not tuple_getitem(types, 19)
            )
        ):
            raise tuple_getitem(externals, 0)(f"{subject} identity drifted")

    return verify_identity


def _freeze_replicate_prefix(
    types: VerifierTypesV2,
    schemas: VerifierSchemasV2,
    externals: VerifierExternalOpsV2,
    read: ReadExactSlotsV2,
    diagnostics: RequireDiagnosticsV2,
    complete: VerifyCompleteVectorV2,
    failure: VerifyFailureVectorV2,
    /,
    evidence_type: type[PendingReplicateEvidenceV2] = PendingReplicateEvidenceV2,
) -> VerifyReplicatePrefixV2:
    identity = _freeze_replicate_identity(types, externals)
    construct_evidence = evidence_type.__new__

    tuple_getitem: Callable[[tuple[object, ...], int], Any] = tuple.__getitem__

    def verify_replicate_prefix(
        outcome: object,
        expected: ReplicateExpectationV2,
        policy: VectorVerificationPolicyV2,
        *,
        subject: str,
    ) -> PendingReplicateEvidenceV2:
        values = read(outcome, tuple_getitem(schemas, 1), subject=subject)
        _id, _seed, status, stage, token, vector, selection, diagnostic = values
        diagnostics(diagnostic, subject=subject)
        identity(values, expected, subject)
        is_failure = status is tuple_getitem(types, 18)
        if is_failure:
            evidence = failure(vector, selection, policy, subject=f"{subject} result vector")
        else:
            evidence = complete(vector, selection, policy, subject=f"{subject} result vector")
        if tuple_getitem(evidence, 0) != tuple_getitem(expected, 3):
            raise tuple_getitem(externals, 0)("all result vectors must use one common support")
        return construct_evidence(
            evidence_type,
            tuple_getitem(evidence, 0),
            tuple_getitem(evidence, 1),
            is_failure,
            selection,
            token,
            status,
            stage,
        )

    return verify_replicate_prefix


def _freeze_replicate_suffix(
    types: VerifierTypesV2,
    schemas: VerifierSchemasV2,
    externals: VerifierExternalOpsV2,
    read: ReadExactSlotsV2,
    /,
    evidence_type: type[VerifiedReplicateEvidenceV2] = VerifiedReplicateEvidenceV2,
) -> VerifyReplicateSuffixV2:
    cast_value = cast
    construct_evidence = evidence_type.__new__

    tuple_getitem: Callable[[tuple[object, ...], int], Any] = tuple.__getitem__

    def verify_replicate_suffix(
        pending: PendingReplicateEvidenceV2,
        *,
        subject: str,
    ) -> VerifiedReplicateEvidenceV2:
        if tuple_getitem(pending, 2):
            if tuple_getitem(pending, 6) is not tuple_getitem(types, 20):
                raise tuple_getitem(externals, 0)("failed replicate has the wrong failure stage")
            return construct_evidence(evidence_type, tuple_getitem(pending, 0), None, True)
        if tuple_getitem(pending, 5) is not tuple_getitem(types, 17):
            raise tuple_getitem(externals, 0)("replicate status is unsupported")
        if tuple_getitem(pending, 6) is not None:
            raise tuple_getitem(externals, 0)("complete replicate has a failure stage")
        assert tuple_getitem(pending, 3) is not None
        decision = read(
            tuple_getitem(pending, 3), tuple_getitem(schemas, 3), subject=f"{subject} selection"
        )[2]
        # A typing-only cast preserves the required post-token operand consumption.
        return construct_evidence(
            evidence_type, tuple_getitem(pending, 0), cast_value("float", decision), False
        )

    return verify_replicate_suffix


def _freeze_complete_terminal(
    externals: VerifierExternalOpsV2,
    /,
    float_type: type[float] = float,
    bool_type: type[bool] = bool,
) -> Callable[[TerminalEvidenceV2, DerivedCountsV2], None]:
    tuple_getitem: Callable[[tuple[object, ...], int], Any] = tuple.__getitem__

    def verify_complete(terminal: TerminalEvidenceV2, derived: DerivedCountsV2) -> None:
        if (
            tuple_getitem(terminal, 1) is not None
            or tuple_getitem(derived, 1)
            or tuple_getitem(terminal, 6) != 0
        ):
            raise tuple_getitem(externals, 0)("COMPLETE result retains a failure")
        if tuple_getitem(derived, 2) != tuple_getitem(terminal, 11):
            raise tuple_getitem(externals, 0)("COMPLETE result does not retain exact B outcomes")
        expected_p = (1 + tuple_getitem(derived, 0)) / (tuple_getitem(terminal, 11) + 1)
        if (
            tuple_getitem(terminal, 5) != tuple_getitem(derived, 0)
            or tuple_getitem(externals, 1)(tuple_getitem(terminal, 7)) is not float_type
            or tuple_getitem(terminal, 7) != expected_p
            or tuple_getitem(terminal, 8) is not None
            or tuple_getitem(terminal, 9) is not None
            or tuple_getitem(externals, 1)(tuple_getitem(terminal, 10)) is not bool_type
            or tuple_getitem(terminal, 10) is not (expected_p <= tuple_getitem(terminal, 12))
        ):
            raise tuple_getitem(externals, 0)("COMPLETE result decision arithmetic is invalid")

    return verify_complete


def _freeze_execution_terminal(
    types: VerifierTypesV2,
    externals: VerifierExternalOpsV2,
    /,
    float_type: type[float] = float,
) -> Callable[[TerminalEvidenceV2, DerivedCountsV2], None]:
    tuple_getitem: Callable[[tuple[object, ...], int], Any] = tuple.__getitem__

    def verify_execution(terminal: TerminalEvidenceV2, derived: DerivedCountsV2) -> None:
        if (
            tuple_getitem(terminal, 0) is not tuple_getitem(types, 11)
            or tuple_getitem(derived, 1) < 1
        ):
            raise tuple_getitem(externals, 0)("replicate-execution result status is invalid")
        if tuple_getitem(derived, 2) != tuple_getitem(terminal, 11):
            raise tuple_getitem(externals, 0)("replicate-execution result does not retain exact B")
        low = (1 + tuple_getitem(derived, 0)) / (tuple_getitem(terminal, 11) + 1)
        high = (1 + tuple_getitem(derived, 0) + tuple_getitem(derived, 1)) / (
            tuple_getitem(terminal, 11) + 1
        )
        if (
            tuple_getitem(terminal, 5) != tuple_getitem(derived, 0)
            or tuple_getitem(terminal, 6) != tuple_getitem(derived, 1)
            or tuple_getitem(terminal, 7) is not None
            or tuple_getitem(terminal, 10) is not None
            or tuple_getitem(externals, 1)(tuple_getitem(terminal, 8)) is not float_type
            or tuple_getitem(terminal, 8) != low
            or tuple_getitem(externals, 1)(tuple_getitem(terminal, 9)) is not float_type
            or tuple_getitem(terminal, 9) != high
        ):
            raise tuple_getitem(externals, 0)("replicate-execution bounds contradict E, F, and B")

    return verify_execution


def _freeze_null_terminal(
    types: VerifierTypesV2,
    externals: VerifierExternalOpsV2,
    /,
) -> Callable[[TerminalEvidenceV2], None]:
    tuple_getitem: Callable[[tuple[object, ...], int], Any] = tuple.__getitem__

    def verify_null(terminal: TerminalEvidenceV2) -> None:
        if (
            tuple_getitem(terminal, 0) is not tuple_getitem(types, 11)
            or tuple_getitem(terminal, 2) != ()
            or tuple_getitem(terminal, 3) is not None
            or tuple_getitem(terminal, 4)
            or tuple_getitem(terminal, 5) != 0
            or tuple_getitem(terminal, 6) != 0
            or tuple_getitem(terminal, 7) is not None
            or tuple_getitem(terminal, 8) is not None
            or tuple_getitem(terminal, 9) is not None
            or tuple_getitem(terminal, 10) is not None
        ):
            raise tuple_getitem(externals, 0)("NULL_BIND terminal result is invalid")

    return verify_null


def _freeze_observed_terminal(
    types: VerifierTypesV2,
    externals: VerifierExternalOpsV2,
    /,
) -> Callable[[TerminalEvidenceV2], None]:
    tuple_getitem: Callable[[tuple[object, ...], int], Any] = tuple.__getitem__

    def verify_observed(terminal: TerminalEvidenceV2) -> None:
        if (
            tuple_getitem(terminal, 0) is not tuple_getitem(types, 11)
            or tuple_getitem(terminal, 3) is not None
            or tuple_getitem(terminal, 4)
            or tuple_getitem(terminal, 5) != 0
            or tuple_getitem(terminal, 6) != 0
            or tuple_getitem(terminal, 7) is not None
            or tuple_getitem(terminal, 8) is not None
            or tuple_getitem(terminal, 9) is not None
            or tuple_getitem(terminal, 10) is not None
        ):
            raise tuple_getitem(externals, 0)("OBSERVED_STATISTIC_SCAN terminal result is invalid")

    return verify_observed


def _freeze_terminal(
    types: VerifierTypesV2,
    externals: VerifierExternalOpsV2,
    complete: Callable[[TerminalEvidenceV2, DerivedCountsV2], None],
    execution: Callable[[TerminalEvidenceV2, DerivedCountsV2], None],
    null: Callable[[TerminalEvidenceV2], None],
    observed: Callable[[TerminalEvidenceV2], None],
    /,
    int_type: type[int] = int,
) -> VerifyTerminalV2:
    tuple_getitem: Callable[[tuple[object, ...], int], Any] = tuple.__getitem__

    def verify_terminal(
        terminal: TerminalEvidenceV2,
        derived: DerivedCountsV2,
        *,
        subject: str,
    ) -> None:
        if (
            tuple_getitem(externals, 1)(tuple_getitem(terminal, 5)) is not int_type
            or tuple_getitem(externals, 1)(tuple_getitem(terminal, 6)) is not int_type
        ):
            raise tuple_getitem(externals, 0)("calibration counts must be exact built-in integers")
        if tuple_getitem(terminal, 0) is tuple_getitem(types, 10):
            complete(terminal, derived)
        elif tuple_getitem(terminal, 1) is tuple_getitem(types, 15):
            execution(terminal, derived)
        elif tuple_getitem(terminal, 1) is tuple_getitem(types, 13):
            null(terminal)
        elif tuple_getitem(terminal, 1) is tuple_getitem(types, 14):
            observed(terminal)
        else:
            raise tuple_getitem(externals, 0)(
                "calibration result has an unsupported terminal state"
            )

    return verify_terminal


def freeze_verifier_primitives_v2(
    types: VerifierTypesV2,
    schemas: VerifierSchemasV2,
    externals: VerifierExternalOpsV2,
    /,
) -> VerifierPrimitiveOpsV2:
    """Freeze focused leaves once; all runtime behavior is closure-bound."""
    read = _freeze_slot_reader(externals)
    diagnostics = _freeze_diagnostics(externals)
    selection = _freeze_selection(schemas, externals, read)
    item = _freeze_vector_reader(types, schemas, externals, read, diagnostics)
    shape = _freeze_vector_shape(externals)
    score = _freeze_valid_score(externals)
    complete = _freeze_complete_vector(types, externals, item, shape, score, selection)
    failure = _freeze_failure_vector(types, externals, item, shape, score)
    return VerifierPrimitiveOpsV2(
        _freeze_sha256(externals),
        diagnostics,
        read,
        selection,
        complete,
        failure,
        _freeze_replicate_prefix(types, schemas, externals, read, diagnostics, complete, failure),
        _freeze_replicate_suffix(types, schemas, externals, read),
        _freeze_terminal(
            types,
            externals,
            _freeze_complete_terminal(externals),
            _freeze_execution_terminal(types, externals),
            _freeze_null_terminal(types, externals),
            _freeze_observed_terminal(types, externals),
        ),
    )
