"""Same-environment replay before continuing a local checkpoint.

Every call runs the complete scientific kernel once. Retained outcomes are exact
comparison targets, never authority to skip computation. The writer lock spans
snapshot, replay, append, finalization and exclusive terminal-record export.
"""

from __future__ import annotations

import errno
import hashlib
import json
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from . import verify_calibration_result, workflow
from .calibration_v2 import _calibrate_selected_family_observed
from .canonical_v2 import scientific_plan_v2_sha256
from .checkpoint_store import (
    CheckpointSession,
    CheckpointSnapshot,
    CheckpointStoreError,
    create_checkpoint,
    open_checkpoint,
)
from .contracts_v2 import CalibrationResult, SelCalV2Error
from .inputs import LoadedInput
from .resolution_v2 import PlanResolutionV2, resolve_plan_v2
from .result_wire import _encode_replicate_outcome, encode_calibration_result
from .workflow_config import (
    WorkflowConfig,
    WorkflowConfigError,
    decode_workflow_config,
    encode_workflow_config,
)
from .workflow_store import write_record

_RESERVED_CHECKPOINT_NAMES = frozenset(
    {
        "state.sqlite",
        "writer.lock",
        "state.sqlite-journal",
        "state.sqlite-wal",
        "state.sqlite-shm",
    }
)


@dataclass(frozen=True, slots=True)
class RecoveryRun:
    """This call's actual result and operational replay/append counts."""

    result: CalibrationResult
    execution_id: str
    replayed_replicates: int
    appended_replicates: int


@dataclass(frozen=True, slots=True)
class RecoveryProgress:
    """Committed/replayed operational counts, without scientific result content."""

    execution_id: str
    phase: Literal["replay", "commit", "finalized"]
    planned_replicates: int
    retained_replicates: int
    replayed_replicates: int


def _output(output_path: str | Path, max_bytes: int) -> None:
    workflow._limit(max_bytes)
    output = Path(output_path)
    if output.exists() or output.is_symlink():
        raise FileExistsError(errno.EEXIST, "output already exists", str(output))
    if not output.parent.exists():
        raise FileNotFoundError(
            errno.ENOENT, "output parent directory does not exist", str(output.parent)
        )
    if not output.parent.is_dir():
        raise NotADirectoryError(
            errno.ENOTDIR, "output parent is not a directory", str(output.parent)
        )


def _checkpoint_output(output_path: str | Path, checkpoint_directory: str | Path) -> None:
    """Keep terminal exports away from this checkpoint's database and sidecars.

    Directory identity handles symlink and case aliases. Reserve ASCII names
    case-insensitively even when an absent sidecar cannot be compared by inode.
    Ordinary output existence/parent checks run first and retain their errors.
    """
    output = Path(output_path)
    if output.name.casefold() not in _RESERVED_CHECKPOINT_NAMES:
        return
    try:
        same_directory = output.parent.samefile(checkpoint_directory)
    except FileNotFoundError:
        # Initial execution has not created its new checkpoint directory yet.
        return
    if same_directory:
        raise CheckpointStoreError(
            "RESERVED_OUTPUT",
            "output conflicts with a reserved checkpoint file",
        )


def _admit(config: WorkflowConfig, loaded: LoadedInput, allow_unattainable: bool) -> None:
    plan = workflow.preflight(config.request, int(loaded.pair.source.size))["plan"]
    if "resource_budget_exceeded" in plan["reasons"]:
        raise WorkflowConfigError(
            "resource_budget_exceeded",
            f"plan exceeds the in-memory execution budget: {plan['resource_budget']}",
        )
    if workflow.NON_GROUP_NULL in plan["reasons"]:
        raise WorkflowConfigError(
            "invalid_null_for_inference",
            "min_shift > 1 is not supported for inference; use min_shift = 1 (the full group)",
        )
    refusals = [reason for reason in plan["reasons"] if reason.startswith("REFUSE")]
    if refusals and not allow_unattainable:
        raise WorkflowConfigError(
            "unattainable_plan", f"plan cannot reach alpha on this input: {refusals[0]}"
        )


def _require(condition: bool, message: str, code: str = "REPLAY_MISMATCH") -> None:
    if not condition:
        raise CheckpointStoreError(code, message)


def _execute(
    store: CheckpointSession,
    snapshot: CheckpointSnapshot,
    header: dict[str, Any],
    loaded: LoadedInput,
    resolution: PlanResolutionV2,
    output_path: str | Path,
    progress: Callable[[RecoveryProgress], None] | None,
) -> RecoveryRun:
    saved = snapshot.replicates
    matched = seen = appended = 0

    def emit(phase: Literal["replay", "commit", "finalized"]) -> None:
        if progress is not None:
            progress(
                RecoveryProgress(
                    header["execution_id"],
                    phase,
                    header["planned_replicates"],
                    len(saved) + appended,
                    matched,
                )
            )

    def observe(index: int, payload: bytes) -> None:
        nonlocal matched, seen, appended
        _require(type(index) is int and index == seen, "kernel outcome sequence differs")
        if index < len(saved):
            _require(saved[index] == (index, payload), "retained outcome differs from replay")
            matched += 1
            emit("replay")
        else:
            _require(matched == len(saved), "retained prefix has not been completely replayed")
            _require(snapshot.phase == "OPEN", "finalized checkpoint has additional outcomes")
            store.append(index, payload)
            appended += 1
            emit("commit")
        seen += 1

    actual = _calibrate_selected_family_observed(loaded.pair, resolution, observe)
    _require(matched == len(saved), "kernel returned before replaying the retained prefix")
    _require(seen == len(actual.replicates), "kernel outcome count differs from actual result")
    verify_calibration_result(actual, resolution)
    current = store.snapshot()
    _require(
        current.replicates
        == tuple((row.replicate_id, _encode_replicate_outcome(row)) for row in actual.replicates),
        "kernel outcome stream differs from actual result",
    )
    result_bytes = encode_calibration_result(actual, max_bytes=header["max_bytes"])
    metadata_bytes = workflow._json(
        {
            "schema": "selcal.workflow-record.v2",
            "raw_input_sha256": loaded.raw_input_sha256,
            "semantic_input_sha256": actual.semantic_input_sha256,
            "scientific_plan_sha256": actual.scientific_plan_sha256,
            "software": header["software"],
        }
    )
    _require(
        workflow._software_identity() == header["software"],
        "software identity changed during execution",
        "IDENTITY_MISMATCH",
    )
    if snapshot.phase == "OPEN":
        store.finalize(result_bytes, metadata_bytes)
    else:
        _require(
            (result_bytes, metadata_bytes) == (snapshot.result_bytes, snapshot.metadata_bytes),
            "finalized content differs from this execution",
        )
    emit("finalized")
    write_record(
        output_path,
        {
            "input": snapshot.input_bytes,
            "request": snapshot.request_bytes,
            "result": result_bytes,
            "metadata": metadata_bytes,
        },
        max_bytes=header["max_bytes"],
    )
    return RecoveryRun(actual, header["execution_id"], matched, appended)


def run_checkpointed_files(
    input_path: str | Path,
    config_path: str | Path,
    output_path: str | Path,
    checkpoint_directory: str | Path,
    *,
    max_bytes: int,
    allow_unattainable: bool = False,
    progress: Callable[[RecoveryProgress], None] | None = None,
) -> RecoveryRun:
    """Admit an ordinary file run, then retain committed outcomes under a new UUID.

    A failed initialization may leave an incomplete directory; specifying a
    checkpoint does not guarantee that it is recoverable. Callback exceptions
    propagate unchanged and do not become scientific not-evaluable results.
    """
    workflow._limit(max_bytes)
    if type(allow_unattainable) is not bool:
        raise workflow.WorkflowError("invalid_attainability_option")
    _output(output_path, max_bytes)
    _checkpoint_output(output_path, checkpoint_directory)
    config, resolution, loaded, raw = workflow._files(input_path, config_path)
    _admit(config, loaded, allow_unattainable)
    request = encode_workflow_config(config)
    header = {
        "schema": "selcal.checkpoint.v1",
        "execution_id": str(uuid4()),
        "planned_replicates": config.request.replicates,
        "input_format": config.source_format,
        "max_bytes": max_bytes,
        "allow_unattainable": allow_unattainable,
        "raw_input_sha256": loaded.raw_input_sha256,
        "semantic_input_sha256": loaded.semantic_input_sha256,
        "scientific_plan_sha256": scientific_plan_v2_sha256(resolution.plan),
        "request_sha256": hashlib.sha256(request).hexdigest(),
        "software": workflow._software_identity(),
    }
    with create_checkpoint(
        checkpoint_directory,
        header_bytes=workflow._json(header),
        input_bytes=raw,
        request_bytes=request,
        max_bytes=max_bytes,
    ) as store:
        return _execute(store, store.snapshot(), header, loaded, resolution, output_path, progress)


def resume_checkpoint(
    checkpoint_directory: str | Path,
    output_path: str | Path,
    *,
    max_bytes: int,
    progress: Callable[[RecoveryProgress], None] | None = None,
) -> RecoveryRun:
    """Replay the full original plan and strictly check its prefix before appending.

    This also recomputes a finalized checkpoint before exporting it again. There
    is no seed, plan, input, override, code or environment migration facility.
    """
    _output(output_path, max_bytes)
    _checkpoint_output(output_path, checkpoint_directory)
    software = workflow._software_identity()
    with open_checkpoint(
        checkpoint_directory,
        expected_software=software,
        max_bytes=max_bytes,
    ) as store:
        snapshot = store.snapshot()
        header = json.loads(snapshot.header_bytes)
        try:
            config = decode_workflow_config(snapshot.request_bytes)
            resolution = resolve_plan_v2(config.request)
            with tempfile.TemporaryDirectory(prefix="selcal-resume-") as folder:
                retained = Path(folder) / ("input." + config.source_format)
                retained.write_bytes(snapshot.input_bytes)
                loaded = workflow._load(retained, config)
        except (ValueError, TypeError, SelCalV2Error) as error:
            raise CheckpointStoreError(
                "INVALID_CHECKPOINT", "retained input/plan is invalid"
            ) from error
        _admit(config, loaded, header["allow_unattainable"])
        _require(
            loaded.raw_input_sha256 == header["raw_input_sha256"]
            and loaded.semantic_input_sha256 == header["semantic_input_sha256"]
            and scientific_plan_v2_sha256(resolution.plan) == header["scientific_plan_sha256"]
            and config.request.replicates == header["planned_replicates"]
            and config.source_format == header["input_format"],
            "resolved retained input/plan differs from checkpoint",
            "IDENTITY_MISMATCH",
        )
        return _execute(store, snapshot, header, loaded, resolution, output_path, progress)
