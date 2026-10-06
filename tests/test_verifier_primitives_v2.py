"""Literal checks for the bounded verifier extraction, independent of the builder."""

from __future__ import annotations

import importlib
import inspect
import sys
from collections.abc import Callable
from types import FrameType, FunctionType

import pytest
from test_verifier_integration_v2 import (
    _adjacent_fault_complete_case,
    _replicate_execution_terminal_literal_case,
)

import selcal._verifier_primitives_v2 as primitives_v2
import selcal.contracts_v2 as contracts_v2


def test_typed_primitives_are_on_the_public_verification_path() -> None:
    try:
        primitives = importlib.import_module("selcal._verifier_primitives_v2")
    except ModuleNotFoundError:
        pytest.fail("typed verifier primitives have not been extracted")
    captured = inspect.getclosurevars(contracts_v2.verify_calibration_result).nonlocals
    assert type(captured["primitive_ops"]) is primitives.VerifierPrimitiveOpsV2
    assert "_verify_result_vector_against_plan" not in captured
    assert primitives.VerifierPrimitiveOpsV2._fields == (
        "require_sha256",
        "require_diagnostics",
        "read_exact_slots",
        "verify_selection",
        "verify_complete_vector",
        "verify_failure_vector",
        "verify_replicate_prefix",
        "verify_replicate_suffix",
        "verify_terminal",
    )


def test_public_verification_executes_every_captured_primitive() -> None:
    captured = inspect.getclosurevars(contracts_v2.verify_calibration_result).nonlocals
    ops = captured["primitive_ops"]
    operations = {name: getattr(ops, name) for name in ops._fields}
    assert all(type(operation) is FunctionType for operation in operations.values())
    names_by_code = {operation.__code__: name for name, operation in operations.items()}
    assert len(names_by_code) == len(operations) == 9
    calls: set[str] = set()

    def profile(frame: FrameType, event: str, _arg: object) -> None:
        if event == "call" and frame.f_code in names_by_code:
            calls.add(names_by_code[frame.f_code])

    # Literal input graphs are built without the production result builder.
    cases = (_adjacent_fault_complete_case(), _replicate_execution_terminal_literal_case())
    previous = sys.getprofile()
    sys.setprofile(profile)
    runtime_verifier: Callable[[object, object], object] = contracts_v2.verify_calibration_result
    try:
        for result, resolution in cases:
            assert runtime_verifier(result, resolution) is None
    finally:
        sys.setprofile(previous)
    assert sys.getprofile() is previous
    assert calls == set(operations)


@pytest.mark.parametrize("target", ["operation", "terminal_value"])
def test_internal_descriptor_replacement_cannot_accept_invalid_p_value(
    monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    result, resolution = _adjacent_fault_complete_case()
    correct_p = result.p_value
    object.__setattr__(result, "p_value", 0.123)
    with pytest.raises(contracts_v2.V2IntegrityError, match="decision arithmetic is invalid"):
        contracts_v2.verify_calibration_result(result, resolution)

    if target == "operation":
        monkeypatch.setattr(
            primitives_v2.VerifierPrimitiveOpsV2,
            "verify_terminal",
            property(lambda _self: lambda *_args, **_kwargs: None),
        )
    else:
        monkeypatch.setattr(
            primitives_v2.TerminalEvidenceV2, "p_value", property(lambda _self: correct_p)
        )
    with pytest.raises(contracts_v2.V2IntegrityError, match="decision arithmetic is invalid"):
        contracts_v2.verify_calibration_result(result, resolution)


@pytest.mark.parametrize(
    "record_type",
    [
        primitives_v2.VerifierPrimitiveOpsV2,
        primitives_v2.VerifierExternalOpsV2,
        primitives_v2.VerifierTypesV2,
        primitives_v2.VerifierSchemasV2,
        primitives_v2.ExactSlotSchemaV2,
        primitives_v2.VectorVerificationPolicyV2,
        primitives_v2.ReplicateExpectationV2,
        primitives_v2.TerminalEvidenceV2,
        primitives_v2.DerivedCountsV2,
        primitives_v2.VerifiedVectorEvidenceV2,
        primitives_v2.PendingReplicateEvidenceV2,
        primitives_v2.VerifiedReplicateEvidenceV2,
    ],
)
def test_published_verifier_does_not_resolve_internal_record_descriptors(
    monkeypatch: pytest.MonkeyPatch, record_type: type[tuple[object, ...]]
) -> None:
    cases = (_adjacent_fault_complete_case(), _replicate_execution_terminal_literal_case())

    def bomb(_self: object) -> object:
        raise AssertionError("replacement internal record descriptor executed")

    for name in vars(record_type)["_fields"]:
        monkeypatch.setattr(record_type, name, property(bomb))
    for result, resolution in cases:
        contracts_v2.verify_calibration_result(result, resolution)


def test_internal_constructor_replacement_cannot_accept_invalid_p_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, resolution = _adjacent_fault_complete_case()
    correct_p = result.p_value
    object.__setattr__(result, "p_value", 0.123)
    original: Callable[..., object] = primitives_v2.TerminalEvidenceV2.__new__

    def rewrite_p_value(cls: type[object], *args: object) -> object:
        values = list(args)
        values[7] = correct_p
        return original(cls, *values)

    with pytest.raises(contracts_v2.V2IntegrityError, match="decision arithmetic is invalid"):
        contracts_v2.verify_calibration_result(result, resolution)
    monkeypatch.setattr(primitives_v2.TerminalEvidenceV2, "__new__", rewrite_p_value)
    with pytest.raises(contracts_v2.V2IntegrityError, match="decision arithmetic is invalid"):
        contracts_v2.verify_calibration_result(result, resolution)


@pytest.mark.parametrize(
    "record_type",
    [
        primitives_v2.VectorVerificationPolicyV2,
        primitives_v2.ReplicateExpectationV2,
        primitives_v2.TerminalEvidenceV2,
        primitives_v2.DerivedCountsV2,
        primitives_v2.VerifiedVectorEvidenceV2,
        primitives_v2.PendingReplicateEvidenceV2,
        primitives_v2.VerifiedReplicateEvidenceV2,
    ],
)
def test_published_verifier_does_not_resolve_replaced_internal_constructors(
    monkeypatch: pytest.MonkeyPatch, record_type: type[tuple[object, ...]]
) -> None:
    cases = (_adjacent_fault_complete_case(), _replicate_execution_terminal_literal_case())

    def bomb(_cls: object, *_args: object) -> object:
        raise AssertionError("replacement internal record constructor executed")

    monkeypatch.setattr(record_type, "__new__", bomb)
    for result, resolution in cases:
        contracts_v2.verify_calibration_result(result, resolution)
