"""Local UI bookkeeping and bounded ownership of the existing CLI process.

Saved observations describe what this helper saw; they are never scientific
verification. The immutable request and raw input are passed to the normal CLI.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
import re
import signal
import stat
import subprocess
import sys
import threading
import uuid
import zipfile
from contextlib import ExitStack
from ctypes import wintypes
from datetime import UTC, datetime
from io import BufferedIOBase, BytesIO
from pathlib import Path
from typing import Any, BinaryIO

from selcal.checkpoint_lock import CheckpointLockError, checkpoint_writer_lock
from selcal.checkpoint_store import open_checkpoint
from selcal.contracts_v2 import ResourceLimitError
from selcal.input_resources import INPUT_LIMITS_V1, read_regular_file_snapshot
from selcal.workflow import _software_identity, read_workflow
from selcal.workflow_config import (
    WorkflowConfigError,
    decode_workflow_config,
    encode_workflow_config,
)
from selcal.workflow_export import verify_export
from selcal.workflow_store import _is_link, read_record, refuse_existing_path

_ID = re.compile(r"[0-9a-f]{32}\Z")
# Files that macOS Finder and Windows Explorer drop into any folder they display. Only these exact
# names, and only as regular non-link files, are ignored; every other unexpected member still
# refuses the workspace or bundle (R19 v3 review: a viewed folder locked out every saved job).
_OS_METADATA = frozenset({".DS_Store", "desktop.ini", "Thumbs.db"})


def _is_os_metadata(path: Path) -> bool:
    if path.name not in _OS_METADATA:
        return False
    try:
        info = path.lstat()
    except OSError:
        return False
    return not _is_link(info) and stat.S_ISREG(info.st_mode)
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_MARKER = {"schema": "selcal.ui-workspace.v1"}
_META_KEYS = {
    "schema",
    "id",
    "created_at",
    "format",
    "max_bytes",
    "allow_unattainable",
    "config_sha256",
}
_OBS_KEYS = {
    "schema",
    "state",
    "operation",
    "input_status",
    "input_sha256",
    "progress",
    "message",
    "result",
}
STDOUT_LIMIT = 524288
STDERR_LIMIT = 2097152
PROGRESS_LIMIT = 8192
ZIP_OVERHEAD = 65536
_ARTIFACTS = {
    "record": ("result.sqlite", "application/octet-stream"),
    "report": ("report.html", "text/html"),
    "bundle": ("evidence.zip", "application/zip"),
}
_BUNDLE_NAMES = frozenset(
    {
        "request.json",
        "result.json",
        "metadata.json",
        "plan.json",
        "software.json",
        "summary.csv",
        "candidates.csv",
        "replicates.csv",
        "report.html",
        "manifest.json",
    }
)


class UIError(ValueError):
    """A bounded user-facing local workflow refusal."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message[:1000])
        self.status = status


def _windows_image() -> str:
    """Read the current native executable, which may differ from sys.executable."""
    if sys.platform != "win32":
        raise OSError("The native Windows image query requires Windows.")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    query = kernel32.GetModuleFileNameW
    query.argtypes = [wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]
    query.restype = wintypes.DWORD
    buffer = ctypes.create_unicode_buffer(32768)
    size = query(None, buffer, len(buffer))
    if size == 0 or size >= len(buffer):
        raise OSError("The current Windows executable image could not be read in full.")
    return buffer.value


def _child_runtime() -> tuple[str | None, dict[str, str] | None]:
    if sys.platform != "win32":
        return None, None
    if sys.implementation.name != "cpython" or getattr(sys, "frozen", False):
        raise OSError("Windows UI workers require an unfrozen CPython interpreter.")
    executable = getattr(sys, "executable", None)
    base = getattr(sys, "_base_executable", None)
    if (
        not isinstance(executable, str)
        or not executable
        or not os.path.isabs(executable)
        or not isinstance(base, str)
        or not base
        or not os.path.isabs(base)
    ):
        raise OSError("Windows UI workers require absolute current and base interpreter paths.")
    image = _windows_image()
    if not os.path.samefile(image, base):
        raise OSError("The Windows native image does not identify the CPython base interpreter.")
    # CPython multiprocessing uses this internal getpath convention to bypass the
    # venv redirector while retaining its environment. It is not a public Python API.
    environment = os.environ.copy()
    environment["__PYVENV_LAUNCHER__"] = executable
    return image, environment


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise UIError("Duplicate JSON keys are not allowed.")
        result[key] = value
    return result


def _constant(_: str) -> None:
    raise UIError("JSON numbers must be finite.")


def _float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise UIError("JSON numbers must be finite.")
    return value


def strict_json(data: bytes) -> Any:
    """Read one UTF-8 JSON value, rejecting duplicate keys and nonfinite tokens."""
    try:
        return json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=_constant,
            parse_float=_float,
        )
    except (ValueError, UnicodeError, RecursionError) as error:
        raise UIError("Invalid UTF-8 JSON; check syntax, nesting and duplicate keys.") from error


def _real_directory(path: Path) -> None:
    # Check ancestors too: resolving first would silently accept a linked workspace.
    # _is_link also catches Windows junctions, which lstat reports as directories (R10v3).
    for part in (*reversed(path.parents), path):
        try:
            info = part.lstat()
        except OSError as error:
            raise UIError("A required workspace directory is unavailable.") from error
        if _is_link(info) or not stat.S_ISDIR(info.st_mode):
            raise UIError("Workspace directories must be real directories, not links.")


def _regular(path: Path) -> None:
    _real_directory(path.parent)
    try:
        info = path.lstat()
    except OSError as error:
        raise UIError(
            "A saved job file is unavailable; retain the workspace for inspection."
        ) from error
    if _is_link(info) or not stat.S_ISREG(info.st_mode):
        raise UIError("Saved job members must be regular files, not links or special files.")


def _read(path: Path, limit: int) -> bytes:
    _regular(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    with os.fdopen(os.open(path, flags), "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise UIError("Saved job member is not a regular file.")
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise UIError("Saved job metadata exceeds its size limit.")
    return data


def _digest(path: Path) -> str:
    _regular(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    digest = hashlib.sha256()
    with os.fdopen(os.open(path, flags), "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise UIError("Input is not a regular file.")
        while chunk := stream.read(65536):
            digest.update(chunk)
    return digest.hexdigest()


def _record_identity(path: Path, limit: int) -> tuple[int, str]:
    """Count and hash actual record bytes without allocating from the caller's cap."""
    _regular(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    digest = hashlib.sha256()
    size = 0
    with os.fdopen(os.open(path, flags), "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise UIError("Saved record is not a regular file.")
        while chunk := stream.read(min(65536, limit - size + 1)):
            size += len(chunk)
            if size > limit:
                raise UIError("Saved record exceeds its size limit.")
            digest.update(chunk)
    return size, digest.hexdigest()


def _artifact_bytes(path: Path, limit: int) -> bytes:
    """Capture actual bounded bytes, never allocate based on a caller's ceiling."""
    _regular(path)
    try:
        raw = read_regular_file_snapshot(path, raw_limit=limit, reason="UI_ARTIFACT_BYTES")
    except ResourceLimitError as error:
        raise UIError(str(error)) from error
    if not raw:
        raise UIError("Saved artifacts must not be empty.")
    return raw


def _transport_zip(members: dict[str, bytes], limit: int) -> bytes:
    """The sole transport encoding of the eleven already checked flat members."""
    stream = BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, raw in sorted(members.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o600) << 16
            info.compress_type = zipfile.ZIP_STORED
            archive.writestr(info, raw)
    payload = stream.getvalue()
    if len(payload) > limit + ZIP_OVERHEAD:
        raise UIError("ZIP transport exceeds its bounded header allowance.")
    return payload


def _write_new(path: Path, data: bytes) -> None:
    _real_directory(path.parent)
    refuse_existing_path(path)
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _encoded(value: object) -> bytes:
    return (json.dumps(value, allow_nan=False, ensure_ascii=True) + "\n").encode("utf-8")


def _budget(value: object) -> int:
    if type(value) is not str or not re.fullmatch(r"[0-9]+", value):
        raise UIError("Byte budget must be a positive decimal integer string.")
    try:
        number = int(value)
    except ValueError as error:
        raise UIError("Byte budget exceeds Python's integer parsing limit.") from error
    if number < 1:
        raise UIError("Byte budget must be positive.")
    return number


def _terminal(data: bytes, command: str, exit_code: int) -> dict[str, Any]:
    value = strict_json(data)
    keys = {"schema", "command", "outcome", "exit_code", "error", "detail", "data"}
    if (
        type(value) is not dict
        or value.keys() != keys
        or value["schema"] != "selcal.cli.v1"
        or value["command"] != command
        or type(value["exit_code"]) is not int
        or value["exit_code"] != exit_code
        or (value["detail"] is not None and type(value["detail"]) is not str)
        or (value["error"] is not None and type(value["error"]) is not str)
    ):
        raise UIError("The child did not return a matching SelCal terminal record.")
    payload, outcome = value["data"], value["outcome"]
    compatible = False
    if exit_code == 0 and value["error"] is None and type(payload) is dict:
        compatible = (
            command == "validate"
            and outcome == "PASS"
            and payload.get("preflight", {}).get("plan", {}).get("status") == "EXECUTABLE"
        ) or (
            command in {"run", "resume", "verify", "report", "export"}
            and outcome == "COMPLETE"
            and payload.get("status") == "complete"
            and type(payload.get("p_value")) in {int, float}
            and type(payload.get("reject_null")) is bool
        )
    elif (
        exit_code == 7
        and command in {"run", "resume", "verify", "report", "export"}
        and outcome == "NOT_EVALUABLE"
    ):
        compatible = (
            value["error"] is None
            and type(payload) is dict
            and payload.get("status") == "not_evaluable"
            and payload.get("failure_stage")
            in {"null_bind", "observed_statistic_scan", "replicate_execution"}
            and payload.get("p_value") is None
            and payload.get("reject_null") is None
        )
    elif exit_code == 2 and type(value["error"]) is str and value["error"]:
        compatible = outcome == "INVALID_REQUEST" and payload is None
        if command == "validate" and outcome == "PLAN_NOT_EXECUTABLE" and type(payload) is dict:
            compatible = payload.get("preflight", {}).get("plan", {}).get("status") != "EXECUTABLE"
    elif exit_code in {4, 130} and type(value["error"]) is str and value["error"]:
        compatible = payload is None and outcome == ("FAIL" if exit_code == 4 else "CANCELLED")
    if command == "verify" and exit_code in {0, 7}:
        compatible = compatible and (
            payload.get("verification_scope") == "input_plan_result_consistency"
            and payload.get("replay") == "NOT_PERFORMED"
            and payload.get("historical_execution_authenticated") is False
        )
    if command in {"report", "export"} and exit_code in {0, 7}:
        compatible = compatible and payload.get("replay") == "NOT_PERFORMED"
        if command == "export":
            compatible = compatible and (
                payload.get("bundle_schema") == "selcal.evidence-bundle.v1"
                and type(payload.get("member_count")) is int
                and payload["member_count"] == 11
                and payload.get("verification_scope")
                == "input_plan_result_and_projection_consistency"
                and payload.get("historical_execution_authenticated") is False
            )
    if not compatible:
        raise UIError("The child terminal outcome conflicts with its exit or scientific state.")
    return value


class JobManager:
    """One owned CLI child, immutable requests, and retained local observations."""

    def __init__(self, workspace: str | Path) -> None:
        self.workspace = Path(os.path.abspath(workspace))
        self._lock = threading.RLock()
        self._active: tuple[str, subprocess.Popen[bytes], threading.Thread] | None = None
        self._cancelled = False
        self._closed = False
        self._observations: dict[str, dict[str, Any]] = {}
        self.unloadable_jobs: list[str] = []
        if not self.workspace.exists() and not self.workspace.is_symlink():
            _real_directory(self.workspace.parent)
            self.workspace.mkdir(mode=0o700)
        _real_directory(self.workspace)
        marker = self.workspace / "workspace.json"
        if not any(not _is_os_metadata(p) for p in self.workspace.iterdir()):
            _write_new(marker, _encoded(_MARKER))
            (self.workspace / "jobs").mkdir(mode=0o700)
        if strict_json(_read(marker, 1024)) != _MARKER:
            raise UIError("This directory is not an owned SelCal UI workspace.")
        # One helper per workspace, held until close(): a second helper would otherwise mark
        # the first one's running jobs as interrupted and overwrite their observations.
        self._workspace_lock = ExitStack()
        try:
            self._workspace_lock.enter_context(checkpoint_writer_lock(self.workspace))
        except CheckpointLockError as error:
            if error.code == "CHECKPOINT_BUSY":
                raise UIError(
                    "This workspace is already open in another SelCal helper. Use that helper, "
                    "or stop it with Ctrl-C in its terminal before opening the workspace again."
                ) from error
            raise UIError(
                "The workspace lock file is unsafe; keep the directory for inspection."
            ) from error
        try:
            self._load_jobs()
        except BaseException:
            self._workspace_lock.close()
            raise

    def _load_jobs(self) -> None:
        _real_directory(self.workspace / "jobs")
        for path in sorted((self.workspace / "jobs").iterdir()):
            if _is_os_metadata(path):
                continue
            if not _ID.fullmatch(path.name):
                raise UIError("Unexpected workspace member; keep the directory for inspection.")
            try:
                self._metadata(path.name)
                observation = strict_json(
                    _read(path / "observation.json", STDOUT_LIMIT + 65536)
                )
                self._check_observation(observation)
            except (UIError, OSError):
                # One unreadable job (for example an admission interrupted before its metadata
                # was written) must not lock every healthy job out of the workspace. It is left
                # untouched for inspection and is not offered for any operation.
                self.unloadable_jobs.append(path.name)
                continue
            if observation["state"] == "running":
                observation.update(
                    state="interrupted",
                    result=None,
                    message=(
                        "Previous operation stopped or became unknown. Saved results are "
                        "not verified. Resume replays and checks retained replicates before "
                        "continuing. Interrupted initialization may lack a usable checkpoint."
                    ),
                )
            self._observations[path.name] = observation
            self._save(path.name)

    def _path(self, job_id: str) -> Path:
        if type(job_id) is not str or not _ID.fullmatch(job_id):
            raise UIError("Unknown job identifier.", 404)
        path = self.workspace / "jobs" / job_id
        _real_directory(path)
        return path

    def _metadata(self, job_id: str) -> dict[str, Any]:
        path = self._path(job_id)
        value = strict_json(_read(path / "metadata.json", 8192))
        if (
            type(value) is not dict
            or value.keys() != _META_KEYS
            or value["schema"] != "selcal.ui-job.v1"
            or value["id"] != job_id
            or type(value["created_at"]) is not str
            or len(value["created_at"]) > 64
            or type(value["format"]) is not str
            or value["format"] not in {"csv", "npz"}
            or type(value["max_bytes"]) is not str
            or type(value["allow_unattainable"]) is not bool
            or type(value["config_sha256"]) is not str
            or not _HASH.fullmatch(value["config_sha256"])
        ):
            raise UIError("Saved job metadata is invalid.")
        _budget(value["max_bytes"])
        raw = _read(path / "request.json", 65536)
        if hashlib.sha256(raw).hexdigest() != value["config_sha256"]:
            raise UIError("Saved configuration changed; create a new job.")
        try:
            if decode_workflow_config(raw).source_format != value["format"]:
                raise UIError("Saved input format conflicts with the configuration.")
        except WorkflowConfigError as error:
            raise UIError("Saved configuration is invalid.") from error
        return value

    @staticmethod
    def _check_observation(value: Any) -> None:
        if (
            type(value) is not dict
            or value.keys() != _OBS_KEYS
            or value["schema"] != "selcal.ui-observation.v1"
            or type(value["state"]) is not str
            or value["state"]
            not in {
                "created",
                "ready",
                "running",
                "validated",
                "complete",
                "not_evaluable",
                "failed",
                "interrupted",
            }
            or (value["operation"] is not None and type(value["operation"]) is not str)
            or value["operation"]
            not in {None, "validate", "run", "resume", "verify", "report", "export"}
            or type(value["input_status"]) is not str
            or value["input_status"] not in {"missing", "incomplete", "complete"}
            or (
                value["input_sha256"] is not None
                and (
                    type(value["input_sha256"]) is not str
                    or not _HASH.fullmatch(value["input_sha256"])
                )
            )
            or type(value["progress"]) is not str
            or len(value["progress"]) > PROGRESS_LIMIT
            or type(value["message"]) is not str
            or len(value["message"]) > 2000
            or (value["result"] is not None and type(value["result"]) is not dict)
        ):
            raise UIError("Saved operation observation is invalid; retain the workspace.")

    def _save(self, job_id: str) -> None:
        path = self._path(job_id)
        target = path / "observation.json"
        if target.exists() or target.is_symlink():
            _regular(target)
        temporary = path / ("observation-" + uuid.uuid4().hex + ".tmp")
        _write_new(temporary, _encoded(self._observations[job_id]))
        os.replace(temporary, target)

    def admit(self, request: object) -> dict[str, Any]:
        if type(request) is not dict or request.keys() != {
            "config_text",
            "max_bytes",
            "allow_unattainable",
        }:
            raise UIError(
                "Admission requires exactly config_text, max_bytes and allow_unattainable."
            )
        if type(request["config_text"]) is not str:
            raise UIError("Configuration must be its original JSON text.")
        limit = request["max_bytes"]
        _budget(limit)
        if type(request["allow_unattainable"]) is not bool:
            raise UIError("The unattainable-plan override must be true or false.")
        try:
            raw = request["config_text"].encode("utf-8")
            config = decode_workflow_config(raw)
        except (WorkflowConfigError, UnicodeError) as error:
            raise UIError(f"Configuration refused: {error}") from error
        with self._lock:
            if self._closed:
                raise UIError("The helper is shutting down.", 409)
            job_id = uuid.uuid4().hex
            path = self.workspace / "jobs" / job_id
            _real_directory(path.parent)
            path.mkdir(mode=0o700)
            metadata = dict(
                schema="selcal.ui-job.v1",
                id=job_id,
                created_at=datetime.now(UTC).isoformat(),
                format=config.source_format,
                max_bytes=limit,
                allow_unattainable=request["allow_unattainable"],
                config_sha256=hashlib.sha256(raw).hexdigest(),
            )
            _write_new(path / "request.json", raw)
            _write_new(path / "metadata.json", _encoded(metadata))
            (path / "operations").mkdir(mode=0o700)
            self._observations[job_id] = dict(
                schema="selcal.ui-observation.v1",
                state="created",
                operation=None,
                input_status="missing",
                input_sha256=None,
                progress="",
                message="Upload the input once, then validate the plan.",
                result=None,
            )
            self._save(job_id)
            return self.get(job_id)

    def get(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            if job_id not in self._observations:
                raise UIError("Unknown job identifier.", 404)
            metadata = self._metadata(job_id)
            try:
                _, reference = self._record_target(job_id, _budget(metadata["max_bytes"]))
                target = {"available": True, "reference": reference, "reason": None}
            except (UIError, OSError) as error:
                target = {"available": False, "reference": None, "reason": str(error)[:1000]}
            artifacts = {}
            for kind in ("report", "bundle"):
                try:
                    _, ref = self._artifact_target(
                        job_id, kind, target["reference"], _budget(metadata["max_bytes"])
                    )
                    artifacts[kind] = {"available": True, "reference": ref, "reason": None}
                except (UIError, OSError) as error:
                    artifacts[kind] = {
                        "available": False,
                        "reference": None,
                        "reason": str(error)[:1000],
                    }
            return {
                **metadata,
                **self._observations[job_id],
                "config_text": _read(self._path(job_id) / "request.json", 65536).decode("utf-8"),
                "saved_observation_only": True,
                "record_target": target,
                "artifacts": artifacts,
            }

    def _record_target(self, job_id: str, limit: int) -> tuple[Path, dict[str, Any]]:
        path = self._path(job_id)
        reference_path = path / "record.json"
        if not reference_path.exists() and not reference_path.is_symlink():
            raise UIError(
                "No UI verification target reference. Earlier records remain available "
                "through CLI verify with their original byte budget."
            )
        reference = strict_json(_read(reference_path, 8192))
        if (
            type(reference) is not dict
            or reference.keys() != {"schema", "operation_id", "bytes", "sha256"}
            or reference["schema"] != "selcal.ui-record-reference.v1"
            or type(reference["operation_id"]) is not str
            or not _ID.fullmatch(reference["operation_id"])
            or type(reference["bytes"]) is not int
            or not 0 < reference["bytes"] <= limit
            or type(reference["sha256"]) is not str
            or not _HASH.fullmatch(reference["sha256"])
        ):
            raise UIError("The saved record reference is invalid; retain the workspace.")
        target = path / "operations" / reference["operation_id"] / "result.sqlite"
        size, digest = _record_identity(target, limit)
        if size != reference["bytes"] or digest != reference["sha256"]:
            raise UIError("The referenced saved record changed; content checking was refused.")
        return target, reference

    def _bind_record(self, job: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
        limit = _budget(job["max_bytes"])
        try:
            target, reference = self._record_target(job["id"], limit)
            record = read_workflow(target, max_bytes=limit)
            expected = encode_workflow_config(decode_workflow_config(job["config_text"].encode()))
            if (
                record.metadata["raw_input_sha256"] != job["input_sha256"]
                or encode_workflow_config(record.config) != expected
            ):
                raise UIError("The saved record does not belong to this job's input and plan.")
            if self._record_target(job["id"], limit) != (target, reference):
                raise UIError("The saved record reference changed during its content check.")
            return target, reference
        except (ValueError, OSError) as error:
            raise UIError(f"Saved record verification refused: {error}") from error

    def _save_record_reference(self, job_id: str, operation: Path, limit: int) -> None:
        size, digest = _record_identity(operation / "result.sqlite", limit)
        if not size:
            raise UIError("The child did not save a nonempty record.")
        reference = {
            "schema": "selcal.ui-record-reference.v1",
            "operation_id": operation.name,
            "bytes": size,
            "sha256": digest,
        }
        path = self._path(job_id)
        target = path / "record.json"
        if target.exists() or target.is_symlink():
            _regular(target)
        temporary = path / ("record-" + uuid.uuid4().hex + ".tmp")
        _write_new(temporary, _encoded(reference))
        os.replace(temporary, target)

    def _checked_subject(self, job_id: str) -> tuple[Path, dict[str, Any]]:
        job = self.get(job_id)
        source = self._path(job_id) / ("input." + job["format"])
        if job["input_status"] != "complete" or _digest(source) != job["input_sha256"]:
            raise UIError("Saved input changed or is incomplete; artifact access refused.")
        return self._bind_record(job)

    def _artifact_target(
        self, job_id: str, kind: str, record: object, limit: int
    ) -> tuple[Path, dict[str, Any]]:
        path = self._path(job_id)
        reference = strict_json(_read(path / (kind + ".json"), 8192))
        bound = limit + (ZIP_OVERHEAD if kind == "bundle" else 0)
        if (
            type(reference) is not dict
            or reference.keys() != {"schema", "kind", "operation_id", "record", "bytes", "sha256"}
            or reference["schema"] != "selcal.ui-artifact-reference.v1"
            or reference["kind"] != kind
            or record is None
            or type(reference["record"]) is not dict
            or type(reference["record"].get("bytes")) is not int
            or reference["record"] != record
            or type(reference["operation_id"]) is not str
            or not _ID.fullmatch(reference["operation_id"])
            or type(reference["bytes"]) is not int
            or not 0 < reference["bytes"] <= bound
            or type(reference["sha256"]) is not str
            or not _HASH.fullmatch(reference["sha256"])
        ):
            raise UIError("Artifact reference is invalid or belongs to a different saved record.")
        target = path / "operations" / reference["operation_id"] / _ARTIFACTS[kind][0]
        _regular(target)
        return target, reference

    def _bundle_snapshot(
        self, operation: Path, subject: tuple[Path, dict[str, Any]], limit: int, form: str
    ) -> tuple[dict[str, bytes], dict[str, Any]]:
        directory = operation / "bundle"
        _real_directory(directory)
        summary = verify_export(directory, max_bytes=limit)
        names = _BUNDLE_NAMES | {"input." + form}
        if {p.name for p in directory.iterdir() if not _is_os_metadata(p)} != names:
            raise UIError("An evidence bundle must contain exactly its eleven fixed members.")
        members: dict[str, bytes] = {}
        remaining = limit
        for name in sorted(names):
            members[name] = _artifact_bytes(directory / name, remaining)
            remaining -= len(members[name])
        source = read_record(subject[0], max_bytes=limit)
        if any(
            members["input." + form if key == "input" else key + ".json"] != raw
            for key, raw in source.items()
        ):
            raise UIError("The evidence bundle does not contain this saved record's exact members.")
        # The captured transport bytes must still be the snapshot checked by the shared verifier.
        if verify_export(directory, max_bytes=limit) != summary or any(
            _record_identity(directory / name, limit) != (len(raw), hashlib.sha256(raw).hexdigest())
            for name, raw in members.items()
        ):
            raise UIError("Evidence bundle changed during content checking.")
        return members, summary

    def _publish_artifact(
        self,
        job_id: str,
        action: str,
        operation: Path,
        subject: tuple[Path, dict[str, Any]],
        data: dict[str, Any],
        job: dict[str, Any],
    ) -> None:
        if self._metadata(job_id) != job:
            raise UIError("Saved job binding changed before artifact publication.")
        limit = _budget(job["max_bytes"])
        kind = "report" if action == "report" else "bundle"
        if kind == "bundle":
            members, summary = self._bundle_snapshot(operation, subject, limit, job["format"])
            if summary != data:
                raise UIError("Export terminal differs from the checked evidence bundle.")
            raw = _transport_zip(members, limit)
            _write_new(operation / "evidence.zip", raw)
        else:
            raw = _artifact_bytes(operation / "report.html", limit)
        reference = {
            "schema": "selcal.ui-artifact-reference.v1",
            "kind": kind,
            "operation_id": operation.name,
            "record": subject[1],
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        path = self._path(job_id)
        target = path / (kind + ".json")
        if target.exists() or target.is_symlink():
            _regular(target)
        temporary = path / (kind + "-" + uuid.uuid4().hex + ".tmp")
        _write_new(temporary, _encoded(reference))
        if (
            self._cancelled
            or self._checked_subject(job_id) != subject
            or self._metadata(job_id) != job
        ):
            raise UIError("Saved subject changed or operation was cancelled before publication.")
        os.replace(temporary, target)

    def download(self, job_id: str, kind: str) -> tuple[bytes, str, str, str]:
        """Return only a fully checked snapshot; never authorize from saved observations.

        The last item is the checked original's workspace-relative path, so the file can be
        copied from the workspace when no browser or download manager saved it.
        """
        if kind not in _ARTIFACTS:
            raise UIError("Unknown artifact kind.", 404)
        with self._lock:
            if self._closed or self._active is not None:
                raise UIError("One operation is active, or the helper is shutting down.", 409)
            try:
                subject = self._checked_subject(job_id)
                job = self._metadata(job_id)
                limit = _budget(job["max_bytes"])
                target, reference = (
                    subject
                    if kind == "record"
                    else self._artifact_target(job_id, kind, subject[1], limit)
                )
                raw = _artifact_bytes(target, limit + (ZIP_OVERHEAD if kind == "bundle" else 0))
                if (
                    len(raw) != reference["bytes"]
                    or hashlib.sha256(raw).hexdigest() != reference["sha256"]
                ):
                    raise UIError("Saved artifact bytes changed; download refused.")
                if kind == "bundle":
                    members, _ = self._bundle_snapshot(target.parent, subject, limit, job["format"])
                    if raw != _transport_zip(members, limit):
                        raise UIError("ZIP is not the exact checked eleven-member transport.")
                if self._checked_subject(job_id) != subject or self._metadata(job_id) != job:
                    raise UIError("Saved subject changed during download checking.")
                base, media = _ARTIFACTS[kind]
                # R12 download retest: name the job and operation so a download is not confused
                # with an earlier "evidence (3).zip" in the same folder.
                filename = f"selcal-{job_id[:8]}-{reference['operation_id'][:8]}-{base}"
                location = target.relative_to(self.workspace).as_posix()
                if not location.isascii():
                    raise UIError("Saved artifact location is not a plain workspace path.")
                return raw, media, filename, location
            except UIError:
                raise
            except (ValueError, OSError, TypeError, OverflowError, MemoryError) as error:
                raise UIError(f"Artifact download refused: {error}") from error

    def list_jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            return [self.get(job_id) for job_id in reversed(self._observations)]

    def input_limit(self, job_id: str) -> int:
        metadata = self.get(job_id)
        return (
            INPUT_LIMITS_V1.csv_raw_bytes
            if metadata["format"] == "csv"
            else INPUT_LIMITS_V1.npz_raw_bytes
        )

    def upload(self, job_id: str, stream: BinaryIO | BufferedIOBase, length: int) -> dict[str, Any]:
        with self._lock:
            job = self.get(job_id)
            if self._closed or job["input_status"] != "missing":
                raise UIError("Input is immutable. Create a new job to change or retry it.", 409)
            if type(length) is not int or length < 1 or length > self.input_limit(job_id):
                raise UIError(
                    "Input length is empty or exceeds the existing CSV/NPZ raw byte limit.", 413
                )
            path = self._path(job_id) / ("input." + job["format"])
            observation = self._observations[job_id]
            observation.update(
                input_status="incomplete",
                message="Input upload is incomplete; it cannot run. Create a new job to retry.",
            )
            self._save(job_id)
            digest = hashlib.sha256()
            try:
                with path.open("xb") as output:
                    remaining = length
                    while remaining:
                        chunk = stream.read(min(65536, remaining))
                        if not chunk:
                            raise UIError("Input upload ended early; partial bytes were retained.")
                        output.write(chunk)
                        digest.update(chunk)
                        remaining -= len(chunk)
                    output.flush()
                    os.fsync(output.fileno())
            except OSError as error:
                raise UIError(
                    "Input upload could not finish; partial bytes were retained."
                ) from error
            observation.update(
                state="ready",
                input_status="complete",
                input_sha256=digest.hexdigest(),
                message="Input saved. Validate before running.",
            )
            self._save(job_id)
            return self.get(job_id)

    def start(self, job_id: str, action: str) -> dict[str, Any]:
        if action not in {"validate", "run", "resume", "verify", "report", "export"}:
            raise UIError("This action is not available in the basic UI.", 404)
        with self._lock:
            job = self.get(job_id)
            binding = self._metadata(job_id)
            if self._closed or self._active is not None:
                raise UIError(
                    "One operation is already active, or the helper is shutting down.", 409
                )
            if job["input_status"] != "complete":
                raise UIError("A complete input upload is required.", 409)
            path = self._path(job_id)
            source = path / ("input." + job["format"])
            if _digest(source) != job["input_sha256"]:
                raise UIError("Saved input changed; create a new job.")
            subject = self._bind_record(job) if action in {"verify", "report", "export"} else None
            checkpoint = path / "checkpoint"
            if action == "run" and (checkpoint.exists() or checkpoint.is_symlink()):
                raise UIError(
                    "A checkpoint already exists. Select this saved job and use Resume; "
                    "retained replicates are replayed and checked before continuing.",
                    409,
                )
            if action == "resume":
                # Bind the selected immutable job using the existing store's validation.
                # This does not grant replay authority: the CLI reopens and checks it.
                try:
                    with open_checkpoint(
                        checkpoint,
                        expected_software=_software_identity(),
                        max_bytes=_budget(job["max_bytes"]),
                    ) as store:
                        snapshot = store.snapshot()
                        header = strict_json(snapshot.header_bytes)
                        expected_request = encode_workflow_config(
                            decode_workflow_config(job["config_text"].encode("utf-8"))
                        )
                        if (
                            header["raw_input_sha256"] != job["input_sha256"]
                            or snapshot.request_bytes != expected_request
                            or header["allow_unattainable"] != job["allow_unattainable"]
                            or header["max_bytes"] != _budget(job["max_bytes"])
                        ):
                            raise UIError(
                                "The checkpoint does not belong to this saved input and plan."
                            )
                except (ValueError, OSError) as error:
                    raise UIError(
                        f"Checkpoint resume refused: {error}. Files retained; interrupted "
                        "initialization may lack a usable checkpoint."
                    ) from error
            operations = path / "operations"
            _real_directory(operations)
            operation = operations / uuid.uuid4().hex
            operation.mkdir(mode=0o700)
            argv = [getattr(sys, "executable", "")]
            if sys.flags.isolated:
                argv.append("-I")
            else:
                if sys.flags.ignore_environment:
                    argv.append("-E")
                # "-m selcal" would otherwise put the working directory first on sys.path, where
                # a stray selcal.py could replace the installed package; -P (Python 3.11+) leaves
                # it out, so the child keeps the helper's working directory (R17 item 2: a deep
                # operation folder as working directory cannot start a process on Windows).
                argv.append("-P")
            argv += ["-m", "selcal", action]
            if action in {"verify", "report", "export"}:
                assert subject is not None
                argv.append(str(subject[0]))
                if action != "verify":
                    argv.append(
                        str(operation / ("report.html" if action == "report" else "bundle"))
                    )
                argv += ["--max-bytes", job["max_bytes"]]
            elif action == "resume":
                argv += [
                    str(checkpoint),
                    str(operation / "result.sqlite"),
                    "--max-bytes",
                    job["max_bytes"],
                ]
            else:
                argv += [str(source), str(path / "request.json")]
            if action == "run":
                argv += [
                    str(operation / "result.sqlite"),
                    "--max-bytes",
                    job["max_bytes"],
                    "--checkpoint",
                    str(checkpoint),
                ]
                if job["allow_unattainable"]:
                    argv.append("--allow-unattainable-plan")
            observation = self._observations[job_id]
            observation.update(
                state="running",
                operation=action,
                result=None,
                progress="",
                message=(
                    "Resuming through SelCal: retained replicates are replayed and checked "
                    "before continuing; final progress is not a saved result."
                    if action == "resume"
                    else (
                        "Checking saved record content with SelCal; no replay or "
                        "historical execution authentication."
                        if action in {"verify", "report", "export"}
                        else "Running the existing SelCal CLI; progress is not a saved result."
                    )
                ),
            )
            self._save(job_id)
            try:
                executable, environment = _child_runtime()
                if executable is not None:
                    # argv[0] must also name the base image for CPython getpath.
                    argv[0] = executable
                child = subprocess.Popen(
                    argv,
                    executable=executable,
                    env=environment,
                    shell=False,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            except OSError as error:
                observation.update(state="failed", message="The SelCal process could not start.")
                self._save(job_id)
                raise UIError("The SelCal process could not start.") from error
            self._cancelled = False
            worker = threading.Thread(
                target=self._watch,
                args=(job_id, action, child, operation, subject, binding),
                daemon=True,
            )
            self._active = (job_id, child, worker)
            worker.start()
            return self.get(job_id)

    def _watch(
        self,
        job_id: str,
        action: str,
        child: subprocess.Popen[bytes],
        operation: Path,
        subject: tuple[Path, dict[str, Any]] | None = None,
        binding: dict[str, Any] | None = None,
    ) -> None:
        stdout = bytearray()
        overflow = threading.Event()

        def drain(pipe: BinaryIO, is_stdout: bool) -> None:
            count = 0
            try:
                while chunk := os.read(pipe.fileno(), 4096):
                    count += len(chunk)
                    if count > (STDOUT_LIMIT if is_stdout else STDERR_LIMIT):
                        overflow.set()
                        child.kill()
                        break
                    if is_stdout:
                        stdout.extend(chunk)
                    else:
                        with self._lock:
                            observation = self._observations[job_id]
                            observation["progress"] = (
                                observation["progress"] + chunk.decode("utf-8", errors="replace")
                            )[-PROGRESS_LIMIT:]
                            self._save(job_id)
            except (OSError, UIError):
                overflow.set()
                try:
                    child.kill()
                except OSError:
                    pass
            finally:
                pipe.close()

        assert child.stdout is not None and child.stderr is not None
        readers = [
            threading.Thread(target=drain, args=(child.stdout, True)),
            threading.Thread(target=drain, args=(child.stderr, False)),
        ]
        for reader in readers:
            reader.start()
        code = child.wait()
        for reader in readers:
            reader.join()
        with self._lock:
            observation = self._observations[job_id]
            try:
                if overflow.is_set():
                    raise UIError(
                        "Child output exceeded its bound or could not be retained; process stopped."
                    )
                if not stdout.strip():
                    raise UIError(
                        f"The SelCal process exited with code {code} without returning "
                        "a terminal result. See Progress and operation details for retained "
                        "diagnostics."
                    )
                try:
                    result = _terminal(bytes(stdout), action, code)
                except (ValueError, TypeError, AttributeError, OverflowError) as error:
                    raise UIError(
                        f"The SelCal process exited with code {code} and returned an invalid "
                        "terminal result. See Progress and operation details for retained "
                        "diagnostics."
                    ) from error
                _write_new(operation / "terminal.json", bytes(stdout))
                if self._cancelled or code == 130:
                    state, message = (
                        "interrupted",
                        "Operation interrupted. Files retained. Resume replays and checks "
                        "retained replicates; interrupted initialization may lack "
                        "a usable checkpoint.",
                    )
                elif code in {0, 7}:
                    job = self._metadata(job_id)
                    limit = _budget(job["max_bytes"])
                    if action == "verify" and self._record_target(job_id, limit) != subject:
                        raise UIError("The saved record reference changed during verification.")
                    if action in {"run", "resume"}:
                        self._save_record_reference(job_id, operation, limit)
                    if action in {"report", "export"}:
                        assert subject is not None
                        assert binding is not None
                        self._publish_artifact(
                            job_id, action, operation, subject, result["data"], binding
                        )
                    state = (
                        "validated"
                        if action == "validate"
                        else ("complete" if code == 0 else "not_evaluable")
                    )
                    message = (
                        "Validation completed."
                        if action == "validate"
                        else (
                            "Run and record saving completed; scientific assumptions still apply."
                            if code == 0
                            else "Scientific NOT_EVALUABLE. Inspect the failure stage; "
                            "this is not p=1 or a negative finding."
                        )
                    )
                    if action == "verify":
                        message = (
                            "Saved record content check completed: input, plan and result "
                            "consistency. No replay; historical execution is not authenticated."
                            + (
                                " Scientific NOT_EVALUABLE retains null p/decision; "
                                "it is not p=1 or a negative finding."
                                if code == 7
                                else ""
                            )
                        )
                    if action in {"report", "export"}:
                        message = (
                            (
                                "HTML report saved"
                                if action == "report"
                                else "Evidence bundle and transport ZIP saved"
                            )
                            + "; checked content only. No replay; "
                            "historical execution is not authenticated."
                            + (
                                " Scientific NOT_EVALUABLE retains null p/decision."
                                if code == 7
                                else ""
                            )
                        )
                else:
                    state, message = (
                        "failed",
                        result["detail"]
                        or result["error"]
                        or "Operation refused. Inspect the result details.",
                    )
                observation.update(state=state, message=message[:2000], result=result)
            except (
                ValueError,
                OSError,
                TypeError,
                AttributeError,
                OverflowError,
                MemoryError,
            ) as error:
                state = "interrupted" if self._cancelled and not overflow.is_set() else "failed"
                message = (
                    "Operation interrupted; saved work is retained. Resume replays and checks "
                    "retained replicates; interrupted initialization may lack a usable checkpoint."
                    if state == "interrupted"
                    else str(error)[:1000]
                )
                observation.update(state=state, message=message, result=None)
            finally:
                try:
                    self._save(job_id)
                finally:
                    self._active = None

    def cancel(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            active = self._active
            if active is None or active[0] != job_id:
                return self.get(job_id)
            child, worker = active[1:]
            # An exited child can still have an unsettled operation to publish.
            self._cancelled = True
            if child.poll() is None:
                try:
                    child.send_signal(signal.SIGINT) if os.name != "nt" else child.terminate()
                except ProcessLookupError:
                    pass
        try:
            child.wait(timeout=2)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
        worker.join(timeout=5)
        return self.get(job_id)

    def close(self) -> None:
        """Stop and reap only this helper's child. Retain all workspace files."""
        with self._lock:
            self._closed = True
            active = self._active
            if active is not None:
                # Accept shutdown before releasing the publication lock.
                self._cancelled = True
        if active is not None:
            try:
                self.cancel(active[0])
            except (UIError, OSError):
                # Shutdown must not depend on the readability of a saved observation.
                # cancel has already reaped the process before reading its final display.
                pass
        # Release the workspace only after the child is reaped and observations are saved.
        self._workspace_lock.close()
