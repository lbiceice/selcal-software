"""Portable projections of one checked terminal snapshot; never scientific replay.

The byte limit bounds serialized source and aggregate bundle bytes, including
the manifest, not peak process memory. Verification establishes content and
projection consistency, not historical execution or durable storage. Hostile
same-user mutation racing these operations is outside this contract.
"""

from __future__ import annotations

import csv
import errno
import hashlib
import io
import json
import re
import stat
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from selcal import workflow
from selcal.canonical import canonical_json_bytes
from selcal.canonical_v2 import scientific_plan_v2_payload
from selcal.contracts import StatisticResult
from selcal.contracts_v2 import ResourceLimitError
from selcal.input_resources import read_regular_file_snapshot
from selcal.workflow import WorkflowError, WorkflowRecord
from selcal.workflow_store import RecordStoreError, _is_link, read_record

_SCHEMA = "selcal.evidence-bundle.v1"
_IDENTITIES = ("raw_input_sha256", "semantic_input_sha256", "scientific_plan_sha256")
_FIXED_NAMES = frozenset(
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
_SUMMARY_COLUMNS = (
    "status",
    "failure_stage",
    "alpha",
    "planned_replicates",
    "retained_replicates",
    "exceedance_count",
    "failure_count",
    "p_value",
    "selected_candidate",
    "decision_statistic",
    "tied_candidates",
    "reject_null",
    "exceedance_bound_low",
    "exceedance_bound_high",
    "semantic_input_sha256",
    "scientific_plan_sha256",
    "diagnostics",
)
_CANDIDATE_COLUMNS = (
    "scope",
    "replicate_id",
    "candidate_id",
    "estimate",
    "selection_score",
    "support_n",
    "validity",
    "diagnostics",
    "backend_identity",
    "preprocessing_identity",
)
_REPLICATE_COLUMNS = (
    "replicate_id",
    "seed_digest_sha256",
    "status",
    "failure_stage",
    "selected_candidate",
    "selected_index",
    "decision_statistic",
    "tied_candidates",
    "transform_token",
    "diagnostics",
)
_DETAILS = {
    "invalid_limit": "max_bytes must be a positive integer.",
    "bundle_size_limit": "The complete bundle exceeds max_bytes; choose a larger limit.",
    "invalid_source": "The source record is invalid or exceeds max_bytes.",
    "invalid_bundle": "The bundle must contain exactly its eleven regular, nonlinked files.",
    "invalid_manifest": "The manifest schema, fields, sizes or digests are invalid.",
    "digest_mismatch": "A bundle member differs from its declared size or SHA-256.",
    "invalid_content": "The retained input, request, result or metadata is invalid.",
    "projection_mismatch": "A derived file or identity differs from the retained content.",
    "unsafe_parent": "The output parent must be an existing real directory, not a link.",
}


class ExportError(WorkflowError):
    """A bounded, path-free export integrity error; the CLI reports exit 4."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.args = (_DETAILS[code],)


def _limit(max_bytes: int) -> None:
    if type(max_bytes) is not int or max_bytes < 1:
        raise ExportError("invalid_limit")


def _compact(value: object) -> str:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    )


def _csv_cell(value: object) -> str:
    if value is None:
        return ""
    if type(value) is bool:
        return "true" if value else "false"
    if type(value) is str:
        return "'" + value if value.startswith(("=", "+", "-", "@", "\t", "\r")) else value
    if type(value) is float:
        return repr(value)
    if type(value) in (tuple, list, dict):
        return _compact(value)
    return str(value)


def _csv(columns: tuple[str, ...], rows: Iterable[dict[str, Any]], remaining: int) -> bytes:
    """Accumulate at most the remaining bundle allowance, checking every row."""
    body = bytearray()
    buffer = io.StringIO(newline="")
    # Both reader newline characters must trigger quoting, including on older Python.
    writer = csv.writer(buffer, lineterminator="\r\n")

    def append(values: Iterable[object]) -> None:
        buffer.seek(0)
        buffer.truncate(0)
        writer.writerow([_csv_cell(value) for value in values])
        # Replace only this row's delimiter; embedded CR/LF bytes remain untouched.
        payload = (buffer.getvalue()[:-2] + "\n").encode("utf-8")
        if len(body) + len(payload) > remaining:
            raise ExportError("bundle_size_limit")
        body.extend(payload)

    append(columns)
    for row in rows:
        append(row[column] for column in columns)
    return bytes(body)


def _candidate_rows(record: WorkflowRecord) -> Iterator[dict[str, Any]]:
    for item in record.result.observed_results:
        yield _candidate_row("observed", None, item)
    for outcome in record.result.replicates:
        for item in outcome.statistic_results:
            yield _candidate_row("replicate", outcome.replicate_id, item)


def _candidate_row(scope: str, rid: int | None, item: StatisticResult) -> dict[str, Any]:
    return {
        "scope": scope,
        "replicate_id": rid,
        "candidate_id": item.candidate_id,
        "estimate": item.estimate,
        "selection_score": item.selection_score,
        "support_n": item.support_n,
        "validity": item.validity.value,
        "diagnostics": item.diagnostics,
        "backend_identity": item.backend_identity,
        "preprocessing_identity": item.preprocessing_identity,
    }


def _replicate_rows(record: WorkflowRecord, wire: dict[str, Any]) -> Iterator[dict[str, Any]]:
    for outcome, raw in zip(record.result.replicates, wire["replicates"], strict=True):
        selection = outcome.selection
        yield {
            "replicate_id": outcome.replicate_id,
            "seed_digest_sha256": outcome.seed_digest_sha256,
            "status": outcome.status.value,
            "failure_stage": None if outcome.failure_stage is None else outcome.failure_stage.value,
            "selected_candidate": None if selection is None else selection.selected_candidate,
            "selected_index": None if selection is None else selection.selected_index,
            "decision_statistic": None if selection is None else selection.decision_statistic,
            "tied_candidates": None if selection is None else selection.tied_candidates,
            # The strict shared decoder already validated these exact wire token fields.
            "transform_token": raw["transform_token"],
            "diagnostics": outcome.diagnostics,
        }


def _projections(
    members: dict[str, bytes], max_bytes: int
) -> tuple[dict[str, bytes], dict[str, Any]]:
    try:
        record, _, resolution, saved = workflow._context_from_members(members, max_bytes)
        summary = {
            **workflow.result_summary(record.result),
            "alpha": record.result.alpha,
            "diagnostics": list(record.result.diagnostics),
            "bundle_schema": _SCHEMA,
            "member_count": 11,
            "raw_input_sha256": record.metadata["raw_input_sha256"],
            "replay": "NOT_PERFORMED",
            "verification_scope": "input_plan_result_and_projection_consistency",
            "historical_execution_authenticated": False,
        }
        payloads = {
            "input." + record.config.source_format: members["input"],
            **{name + ".json": members[name] for name in ("request", "result", "metadata")},
        }
        total = sum(map(len, payloads.values()))

        def add(name: str, payload: bytes) -> None:
            nonlocal total
            total += len(payload)
            if total > max_bytes:
                raise ExportError("bundle_size_limit")
            payloads[name] = payload

        add("plan.json", canonical_json_bytes(scientific_plan_v2_payload(resolution.plan)))
        add("software.json", canonical_json_bytes(record.metadata["software"]))
        add("summary.csv", _csv(_SUMMARY_COLUMNS, [summary], max_bytes - total))
        add("candidates.csv", _csv(_CANDIDATE_COLUMNS, _candidate_rows(record), max_bytes - total))
        # This is a projection of the already decoded and checked wire, not a second result parser.
        wire = json.loads(saved)
        add(
            "replicates.csv",
            _csv(_REPLICATE_COLUMNS, _replicate_rows(record, wire), max_bytes - total),
        )
        if total >= max_bytes:
            raise ExportError("bundle_size_limit")
        try:
            report = workflow._render_report(record, summary, max_bytes - total)
        except WorkflowError as error:
            if error.code == "report_size_limit":
                raise ExportError("bundle_size_limit") from error
            raise
        add("report.html", report)
        return payloads, summary
    except ExportError:
        raise
    except (ValueError, TypeError, RecursionError) as error:
        raise ExportError("invalid_content") from error


def export_record(
    record_path: str | Path, output_directory: str | Path, *, max_bytes: int
) -> dict[str, Any]:
    """Export all retained outcomes to a new directory; do not recompute science.

    max_bytes bounds the source and aggregate bundle including manifest. On a
    write/close failure the partial directory is retained. Use verify-export to
    inspect its content, or choose a fresh destination; never assume durability.
    A returned NOT_EVALUABLE scientific status still means the export completed.
    """
    _limit(max_bytes)
    destination = Path(output_directory)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(errno.EEXIST, "output already exists", str(destination))
    parent = destination.parent.lstat()
    # _is_link also catches Windows junctions, which lstat reports as directories (R10v3).
    if _is_link(parent):
        raise ExportError("unsafe_parent")
    if not stat.S_ISDIR(parent.st_mode):
        raise NotADirectoryError(
            errno.ENOTDIR, "output parent is not a directory", str(destination.parent)
        )
    try:
        members = read_record(record_path, max_bytes=max_bytes)
    except RecordStoreError as error:
        raise ExportError("invalid_source") from error
    payloads, summary = _projections(members, max_bytes)
    manifest = {
        "schema": _SCHEMA,
        "members": {
            name: {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
            for name, payload in payloads.items()
        },
        **{key: summary[key] for key in _IDENTITIES},
        "replay": "NOT_PERFORMED",
        "historical_execution_authenticated": False,
    }
    payloads["manifest.json"] = (_compact(manifest) + "\n").encode("utf-8")
    if sum(map(len, payloads.values())) > max_bytes:
        raise ExportError("bundle_size_limit")
    destination.mkdir()  # Exclusive even when another export reaches this point concurrently.
    for name, payload in payloads.items():  # manifest was inserted last, after all projections.
        with (destination / name).open("xb") as stream:
            if stream.write(payload) != len(payload):
                raise OSError("incomplete bundle write")
    return summary


def _manifest(raw: bytes, names: set[str], max_bytes: int) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ExportError("invalid_manifest")
            result[key] = value
        return result

    def forbidden(value: str) -> None:
        raise ExportError("invalid_manifest")

    try:
        data = json.loads(raw.decode("utf-8"), object_pairs_hook=unique, parse_constant=forbidden)
        keys = {"schema", "members", *_IDENTITIES, "replay", "historical_execution_authenticated"}
        if (
            type(data) is not dict
            or data.keys() != keys
            or data["schema"] != _SCHEMA
            or data["replay"] != "NOT_PERFORMED"
            or data["historical_execution_authenticated"] is not False
            or type(data["members"]) is not dict
            or data["members"].keys() != names - {"manifest.json"}
        ):
            raise ExportError("invalid_manifest")
        hashes = [data[key] for key in _IDENTITIES]
        total = len(raw)
        for descriptor in data["members"].values():
            if (
                type(descriptor) is not dict
                or descriptor.keys() != {"bytes", "sha256"}
                or type(descriptor["bytes"]) is not int
                or descriptor["bytes"] < 1
            ):
                raise ExportError("invalid_manifest")
            hashes.append(descriptor["sha256"])
            total += descriptor["bytes"]
        if any(
            type(value) is not str or re.fullmatch("[0-9a-f]{64}", value) is None
            for value in hashes
        ):
            raise ExportError("invalid_manifest")
        if total > max_bytes:
            raise ExportError("bundle_size_limit")
        return data
    except ExportError:
        raise
    except (ValueError, TypeError, RecursionError) as error:
        raise ExportError("invalid_manifest") from error


def _read_member(path: Path, remaining: int) -> bytes:
    info = path.lstat()
    if _is_link(info) or not stat.S_ISREG(info.st_mode) or info.st_size < 1:
        raise ExportError("invalid_bundle")
    if info.st_size > remaining:
        raise ExportError("bundle_size_limit")
    try:
        raw = read_regular_file_snapshot(path, raw_limit=remaining, reason="EVIDENCE_BUNDLE_BYTES")
    except ResourceLimitError as error:
        raise ExportError("bundle_size_limit") from error
    except ValueError as error:
        raise ExportError("invalid_bundle") from error
    if not raw:
        raise ExportError("invalid_bundle")
    return raw


def verify_export(bundle_directory: str | Path, *, max_bytes: int) -> dict[str, Any]:
    """Read one bounded portable snapshot and check every raw and derived member.

    Does not open a checkpoint, recreate a database, replay calibration, or
    authenticate historical execution. Successful verification is content-only,
    even if the original export operation reported a final close failure.
    """
    _limit(max_bytes)
    directory = Path(bundle_directory)
    root = directory.lstat()
    if _is_link(root) or not stat.S_ISDIR(root.st_mode):
        raise ExportError("invalid_bundle")
    names: set[str] = set()
    for path in directory.iterdir():
        names.add(path.name)
        if len(names) > 11:
            raise ExportError("invalid_bundle")
    if names not in (_FIXED_NAMES | {"input.csv"}, _FIXED_NAMES | {"input.npz"}):
        raise ExportError("invalid_bundle")
    captured = {"manifest.json": _read_member(directory / "manifest.json", max_bytes)}
    manifest = _manifest(captured["manifest.json"], names, max_bytes)
    total = len(captured["manifest.json"])
    for name in sorted(names - {"manifest.json"}):
        raw = _read_member(directory / name, max_bytes - total)
        total += len(raw)
        descriptor = manifest["members"][name]
        if (
            len(raw) != descriptor["bytes"]
            or hashlib.sha256(raw).hexdigest() != descriptor["sha256"]
        ):
            raise ExportError("digest_mismatch")
        captured[name] = raw
    input_name = "input.csv" if "input.csv" in names else "input.npz"
    members = {
        "input": captured[input_name],
        **{key: captured[key + ".json"] for key in ("request", "result", "metadata")},
    }
    projected, summary = _projections(members, max_bytes - len(captured["manifest.json"]))
    if any(manifest[key] != summary[key] for key in _IDENTITIES):
        raise ExportError("projection_mismatch")
    # Manifest whitespace/order is not substantive, but every other byte is deterministic.
    if projected.keys() != names - {"manifest.json"} or any(
        projected[name] != captured[name] for name in names - {"manifest.json"}
    ):
        raise ExportError("projection_mismatch")
    return summary
