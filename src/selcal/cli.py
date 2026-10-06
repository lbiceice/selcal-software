"""Thin command-line consumer of the public file workflow."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any

from selcal import __version__
from selcal.checkpoint_lock import CheckpointLockError
from selcal.checkpoint_store import CheckpointStoreError
from selcal.contracts_v2 import ResourceLimitError, SelCalV2Error
from selcal.result_wire import ResultWireError
from selcal.workflow import (
    WorkflowError,
    doctor,
    report_record,
    result_summary,
    run_files,
    validate_files,
    verify_record,
)
from selcal.workflow_config import WorkflowConfigError
from selcal.workflow_export import ExportError, export_record, verify_export
from selcal.workflow_recovery import (
    RecoveryProgress,
    RecoveryRun,
    resume_checkpoint,
    run_checkpointed_files,
)
from selcal.workflow_store import RecordStoreError


def _positive(value: str) -> int:
    try:
        limit = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("max-bytes must be a positive integer") from error
    if limit < 1:
        raise argparse.ArgumentTypeError("max-bytes must be a positive integer")
    return limit


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="selcal", description="Selection-aware calibration")
    parser.add_argument("--version", action="version", version=f"SelCal {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "run"):
        command = commands.add_parser(name)
        command.add_argument("input")
        command.add_argument("config")
        if name == "run":
            command.add_argument("output")
            command.add_argument("--max-bytes", type=_positive, required=True)
            command.add_argument(
                "--checkpoint",
                metavar="DIRECTORY",
                help="retain outcomes for same-environment replay before continuing",
            )
            command.add_argument(
                "--allow-unattainable-plan",
                action="store_true",
                help="run a plan whose smallest attainable p-value exceeds alpha",
            )
    resume = commands.add_parser("resume", help="replay the retained prefix, then continue")
    resume.add_argument("checkpoint", metavar="DIRECTORY")
    resume.add_argument("output")
    resume.add_argument("--max-bytes", type=_positive, required=True)
    for name in ("verify", "report"):
        command = commands.add_parser(name)
        command.add_argument("record")
        command.add_argument("--max-bytes", type=_positive, required=True)
        if name == "verify":
            modes = command.add_mutually_exclusive_group()
            modes.add_argument(
                "--replay",
                action="store_true",
                help="recompute and compare the complete saved result byte for byte "
                "(same code, Python, NumPy and platform)",
            )
            modes.add_argument(
                "--replay-decision",
                action="store_true",
                help="recompute with the same code on any platform; decisions must match "
                "exactly, statistic values within a stated number of units in the last place",
            )
        else:
            command.add_argument("output")
    doctor_command = commands.add_parser("doctor")
    doctor_command.add_argument(
        "--folder",
        help="check saving and reading records in this folder (default: the temporary folder)",
    )
    export = commands.add_parser("export", help="export a checked portable evidence directory")
    export.add_argument("record")
    export.add_argument("output", metavar="DIRECTORY")
    export.add_argument("--max-bytes", type=_positive, required=True)
    check_export = commands.add_parser("verify-export", help="check bundle content without replay")
    check_export.add_argument("directory", metavar="DIRECTORY")
    check_export.add_argument("--max-bytes", type=_positive, required=True)
    ui = commands.add_parser("ui", help="open the basic offline local interface")
    ui.add_argument("--workspace", required=True, help="new or owned local workspace directory")
    ui.add_argument(
        "--port", type=int, default=0, help="loopback port; default chooses a free port"
    )
    ui.add_argument(
        "--no-browser", action="store_true", help="print the local URL without opening it"
    )
    return parser


def _recovery_progress(event: RecoveryProgress) -> None:
    count = event.replayed_replicates if event.phase == "replay" else event.retained_replicates
    interval = max(1, (event.planned_replicates + 19) // 20)
    if event.phase == "finalized" or count == 1 or count % interval == 0:
        sys.stderr.write(
            f"SelCal checkpoint {event.execution_id}: {event.phase}; "
            f"retained={event.retained_replicates}/{event.planned_replicates}, "
            f"replayed={event.replayed_replicates}\n"
        )
        sys.stderr.flush()


def _recovery_summary(run: RecoveryRun) -> dict[str, Any]:
    summary = result_summary(run.result)
    summary["checkpoint"] = {
        "execution_id": run.execution_id,
        "replayed_replicates": run.replayed_replicates,
        "appended_replicates": run.appended_replicates,
        "mode": "replay_before_continue",
    }
    return summary


def _execute(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "validate":
        return validate_files(args.input, args.config)
    if args.command == "run":
        if args.checkpoint is not None:
            return _recovery_summary(
                run_checkpointed_files(
                    args.input,
                    args.config,
                    args.output,
                    args.checkpoint,
                    max_bytes=args.max_bytes,
                    allow_unattainable=args.allow_unattainable_plan,
                    progress=_recovery_progress,
                )
            )
        return result_summary(
            run_files(
                args.input,
                args.config,
                args.output,
                max_bytes=args.max_bytes,
                allow_unattainable=args.allow_unattainable_plan,
            )
        )
    if args.command == "resume":
        return _recovery_summary(
            resume_checkpoint(
                args.checkpoint,
                args.output,
                max_bytes=args.max_bytes,
                progress=_recovery_progress,
            )
        )
    if args.command == "verify":
        return verify_record(
            args.record,
            max_bytes=args.max_bytes,
            replay=args.replay,
            replay_decision=args.replay_decision,
        )
    if args.command == "report":
        return report_record(args.record, args.output, max_bytes=args.max_bytes)
    if args.command == "export":
        return export_record(args.record, args.output, max_bytes=args.max_bytes)
    if args.command == "verify-export":
        return verify_export(args.directory, max_bytes=args.max_bytes)
    return doctor(args.folder)


def _has_non_directory_parent(path: Path) -> bool:
    """Use the nearest stat-able ancestor, without inferring from a missing path."""
    for parent in path.parents:
        try:
            metadata = parent.stat()
        except (FileNotFoundError, NotADirectoryError):
            continue
        except OSError:
            return False
        return not stat.S_ISDIR(metadata.st_mode)
    return False


def _io_detail(error: OSError, args: argparse.Namespace) -> str:
    """Give recovery steps without exposing an absolute local path or an OS traceback."""
    output = getattr(args, "output", None)
    filename = os.fsdecode(error.filename) if error.filename is not None else None
    if isinstance(error, FileExistsError):
        name = Path(filename or output or "output").name
        return (
            f"Output already exists ({name!r}). "
            "Choose a new filename; existing files are not replaced."
        )
    if isinstance(error, FileNotFoundError):
        failed_path = Path(filename) if filename is not None else None
        # run may report the output parent itself; other operations report the file.
        ancestor_target = (
            Path(output) if output and failed_path == Path(output).parent else failed_path
        )
        if ancestor_target is not None and _has_non_directory_parent(ancestor_target):
            return (
                "A parent path is not a directory. "
                "Check the supplied path and its parent directory."
            )
        for attribute, label in (
            ("config", "Configuration file"),
            ("input", "Input file"),
            ("record", "Record file"),
        ):
            path = getattr(args, attribute, None)
            if path and failed_path == Path(path):
                return f"{label} {Path(path).name!r} was not found. Check the path and filename."
        if output and failed_path in {Path(output), Path(output).parent}:
            return (
                "Output parent directory does not exist. Create it or choose an existing directory."
            )
        return "A required file or directory was not found. Check the supplied paths."
    if isinstance(error, PermissionError):
        return (
            "Permission denied. Check file access permissions "
            "or choose a writable output directory."
        )
    if isinstance(error, NotADirectoryError):
        return "A parent path is not a directory. Check the supplied path and its parent directory."
    return (
        "The file operation failed. Check the supplied paths, storage availability and file access."
    )


def main(argv: list[str] | None = None) -> int:
    """Emit one JSON operation result; scientific NE uses exit 7, not an I/O failure."""
    args = _parser().parse_args(argv)
    if args.command == "ui":
        from selcal.ui import serve
        from selcal.ui_jobs import UIError

        try:
            return serve(args.workspace, port=args.port, no_browser=args.no_browser)
        except (UIError, OSError) as error:
            sys.stderr.write(f"SelCal UI could not start: {error}\n")
            return 4
    code, outcome, error_code = 0, "PASS", None
    detail: str | None = None
    data: dict[str, Any] | None = None
    try:
        data = _execute(args)
        plan = data.get("preflight", {}).get("plan") if args.command == "validate" else None
        if plan is not None and plan["status"] != "EXECUTABLE":
            code, outcome, error_code = 2, "PLAN_NOT_EXECUTABLE", plan["reasons"][0]
        elif data.get("status") == "not_evaluable":
            code, outcome = 7, "NOT_EVALUABLE"
            if args.command == "export":
                detail = (
                    "Evidence bundle written successfully; scientific status remains NOT_EVALUABLE."
                )
            elif args.command == "verify-export":
                detail = "Bundle content verified; scientific status remains NOT_EVALUABLE."
        elif data.get("status") == "complete":
            outcome = "COMPLETE"
        elif args.command == "doctor" and data.get("record_round_trip") != "PASS":
            code, outcome, error_code = 4, "FAIL", "record_round_trip_failed"
            detail = f"records cannot be saved and read back here: {data['record_round_trip']}"
    except (CheckpointLockError, CheckpointStoreError) as error:
        code, outcome, error_code = 4, "FAIL", error.code
        detail = str(error)
    except (WorkflowError, RecordStoreError, ResultWireError) as error:
        code, outcome, error_code = 4, "FAIL", error.code
        detail = str(error) if isinstance(error, (RecordStoreError, ExportError)) else None
    except (WorkflowConfigError, ResourceLimitError, TypeError, ValueError) as error:
        code, outcome = 2, "INVALID_REQUEST"
        error_code = error.code if isinstance(error, WorkflowConfigError) else "invalid_request"
        detail = str(error) or None
    except OSError as error:
        code, outcome, error_code = 4, "FAIL", "io_or_integrity_failure"
        detail = _io_detail(error, args)
        if args.command == "export":
            detail += (
                " If a directory was created, retain it and inspect it with verify-export, "
                "or use a fresh destination."
            )
    except SelCalV2Error:
        code, outcome, error_code = 4, "FAIL", "io_or_integrity_failure"
    except KeyboardInterrupt:
        code, outcome, error_code = 130, "CANCELLED", "interrupted_no_resume_claim"
        if args.command == "resume" or getattr(args, "checkpoint", None) is not None:
            error_code = "interrupted_checkpoint_requires_validation"
            detail = (
                "Checkpoint initialization or execution was interrupted; recoverability is "
                "not guaranteed. Check the retained directory, then use resume with the same "
                "environment, original max-bytes and a new output filename."
            )
    terminal = {
        "schema": "selcal.cli.v1",
        "command": args.command,
        "outcome": outcome,
        "exit_code": code,
        "error": error_code,
        "detail": detail,
        "data": data,
    }
    try:
        if error_code is not None:
            suffix = f": {detail}" if detail else ""
            sys.stderr.write(f"SelCal: {error_code}{suffix}\n")
        elif code == 7 and detail is not None:
            sys.stderr.write(f"SelCal: {detail}\n")
        sys.stdout.write(json.dumps(terminal, allow_nan=False, sort_keys=True) + "\n")
        sys.stdout.flush()
    except OSError:
        return 4
    return code
