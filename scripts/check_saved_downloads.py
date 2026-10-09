"""Check files a person saved from the SelCal page against the UI workspace that produced them.

Usage::

    python -I scripts/check_saved_downloads.py --python PYTHON --workspace UI_WORKSPACE \\
        --saved SAVED_DIR --out NEW_DIR [--expect EXPECT.json] [--allow-history]

Each saved ``selcal-<job8>-<op8>-{report.html,result.sqlite,evidence.zip}`` file is matched to one
complete job and operation. Stop the source UI helper normally before running this tool; a live
workspace is refused, not copied around its lock. Its retained ``writer.lock`` is not deleted or
changed. A read-only native availability probe precedes copying and follows checking; this is a
cooperative offline-use contract, not hostile-writer or atomic-snapshot isolation. The workspace
is never opened for writing: it is copied to a snapshot
inside the output folder (removed afterwards), and the product's own download check (job metadata,
request, saved input, record reference, artifact reference, record and bundle content) runs on the
snapshot. Separate results are kept for each kind of evidence:

* ``file_equality``: saved bytes equal the checked artifact bytes;
* ``job_binding``: the product download check passed for the job's current reference;
* ``reference_scope``: ``current``, or ``history`` for an older operation of the same job;
* ``record_replay`` (records): ``selcal verify --replay`` on the saved file is MATCH;
* ``export_content`` (ZIP): safe unpack and ``selcal verify-export`` pass;
* ``html_content`` (reports): strict UTF-8 HTML that equals a report regenerated from the job's
  checked record;
* ``browser_save_observed``: always ``NOT_OBSERVED_BY_THIS_TOOL``; only the person saving can
  record how the browser or a download manager saved the file.

A history file is not a current result: without ``--allow-history`` it fails, with it the
receipt says ``history`` and ``job_binding`` is ``NOT_APPLICABLE_HISTORY``. Exit 0 only if every
file passes every check that applies to it, inputs are re-identified unchanged, and snapshot
cleanup succeeds. Unestablished or unreadable baselines are ``NOT_CHECKED``, never evidence that
inputs stayed unchanged. The snapshot manager is closed before removal; cleanup errors remain
in the receipt. This tool never stops the source helper or bypasses a locked/unreadable member.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import lzma
import os
import re
import shutil
import stat
import subprocess
import sys
import zipfile
import zlib
from pathlib import Path

SCHEMA = "selcal.saved-downloads-check.v2"
NAME = re.compile(r"selcal-([0-9a-f]{8})-([0-9a-f]{8})-(report\.html|result\.sqlite|evidence\.zip)")
# Browsers add " (1)" when a name already exists in the folder.
RENAMED = re.compile(r"(selcal-[0-9a-f]{8}-[0-9a-f]{8}-(?:report|result|evidence)) \(\d+\)(\.\w+)")
KIND = {"report.html": "report", "result.sqlite": "record", "evidence.zip": "bundle"}
FULL_ID = re.compile(r"[0-9a-f]{32}")
ZIP_MEMBERS, ZIP_TOTAL = 11, 32 * 1024 * 1024


class CheckFailed(Exception):
    def __init__(self, message: str, *, path: Path | None = None) -> None:
        self.path = path
        super().__init__(message)


def _identity(path: Path) -> dict:
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
        raise CheckFailed(f"linked identity file refused: {path}", path=path)
    try:
        raw = path.read_bytes()
    except OSError as error:
        # Windows read-denying byte locks may raise after open, without filename attached.
        # Retain the native error type/errno while making the exact failed member observable.
        if error.filename is None:
            error.filename = str(path)
        raise
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _regular(path: Path) -> None:
    for node in (path, *path.parents):
        info = node.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise CheckFailed(f"linked path refused: {node}", path=node)
    if not path.is_file() or path.stat().st_size == 0:
        raise CheckFailed(f"empty or nonregular file: {path.name}")


def _plain_directory(folder: Path, *, allow_missing: bool = False) -> None:
    # Inspect ancestors first so a junction is refused before examining its target.
    for node in (*reversed(folder.parents), folder):
        try:
            info = node.lstat()
        except FileNotFoundError:
            if allow_missing:
                return  # Only a suffix may be missing, after every existing ancestor passed.
            raise
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise CheckFailed(f"linked folder path refused: {node}", path=node)
        if not stat.S_ISDIR(info.st_mode):
            raise CheckFailed(f"not a plain folder: {node}", path=node)


def _saved_tree(folder: Path) -> dict:
    """Saved inputs are a flat folder; never filter away unreadable or linked entries."""
    _plain_directory(folder)
    result = {}
    for path in sorted(folder.iterdir()):
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise CheckFailed(f"linked saved member refused: {path}", path=path)
        if not stat.S_ISREG(info.st_mode):
            raise CheckFailed(f"nonregular saved member refused: {path}", path=path)
        result[path.name] = _identity(path)
    return result


def _tree(folder: Path) -> dict:
    # rglob/is_symlink can follow a Windows directory junction before recognizing it.
    # Inspect each directory entry without following it, and refuse all reparse points
    # before any recursion or content read. Root and ancestor links are refused too.
    _plain_directory(folder)
    result = {}

    def visit(directory: Path) -> None:
        for path in sorted(directory.iterdir()):
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                raise CheckFailed(f"linked workspace member refused: {path}", path=path)
            if stat.S_ISDIR(info.st_mode):
                visit(path)
            elif stat.S_ISREG(info.st_mode):
                result[path.relative_to(folder).as_posix()] = _identity(path)
            else:
                raise CheckFailed(f"nonregular workspace member refused: {path}", path=path)

    visit(folder)
    return result


def _require_offline_workspace(workspace: Path) -> None:
    """Probe the existing control file without creating, writing or removing it.

    Opening the product's writer context here would create/open a source file for writing.
    Instead reuse its native contention and identity checks with a read-only descriptor.
    Release this short availability probe before hashing: on Windows a separate handle cannot
    read even our own locked byte. Unchanged full-tree identities remain a separate requirement.
    """
    from selcal.checkpoint_lock import (
        CheckpointLockError,
        _operations,
        _safe_identity,
        _verify_identity,
    )

    _plain_directory(workspace)
    marker = workspace / "writer.lock"
    try:
        expected = _safe_identity(marker.lstat())
        flags = os.O_RDONLY
        for optional in ("O_CLOEXEC", "O_BINARY", "O_NOFOLLOW"):
            flags |= getattr(os, optional, 0)
        descriptor = os.open(marker, flags)
        try:
            os.set_inheritable(descriptor, False)
            _verify_identity(descriptor, marker, expected)
            acquire, release = _operations()
            acquire(descriptor)
            try:
                _verify_identity(descriptor, marker, expected)
            finally:
                release(descriptor)
        finally:
            os.close(descriptor)
    except CheckpointLockError as error:
        if error.code == "CHECKPOINT_BUSY":
            raise CheckFailed(
                "source workspace has a live writer; content checks NOT_CHECKED. "
                "Stop its UI helper normally, then retry. Do not delete writer.lock.",
                path=marker,
            ) from error
        raise CheckFailed(
            f"offline source check refused ({error.code}): {error}", path=marker
        ) from error


def _record_error(receipt: dict, error: Exception, phase: str, path: Path | None) -> None:
    failed_path = getattr(error, "path", None) or getattr(error, "filename", None) or path
    details = {
        "phase": phase,
        "path": str(failed_path) if failed_path is not None else None,
        "error": f"{type(error).__name__}: {error}",
    }
    receipt["errors"].append(details)
    receipt["status"] = "FAIL"
    if receipt["first_error"] is None:
        receipt["first_error"] = details["error"]
        receipt["failure_phase"] = phase
        receipt["failure_path"] = details["path"]


def _reidentify(receipt: dict, args: argparse.Namespace, workspace_before, saved_before) -> None:
    """Unknown is not unchanged, and one unreadable input does not hide the other's result."""
    unchanged = []
    for name, before, folder in (
        ("workspace", workspace_before, args.workspace),
        ("saved", saved_before, args.saved),
    ):
        row = receipt["input_checks"][name]
        if before is None:
            unchanged.append(None)
            continue
        try:
            if name == "workspace":
                _require_offline_workspace(folder)
                after = _tree(folder)
            else:
                after = _saved_tree(folder)
        except (OSError, CheckFailed, ValueError) as error:
            row["status"] = "NOT_CHECKED"
            _record_error(receipt, error, f"{name}_reidentification", folder)
            unchanged.append(None)
        else:
            same = before == after
            row["status"] = "PASS" if same else "FAIL"
            unchanged.append(same)
            if not same:
                _record_error(
                    receipt,
                    CheckFailed(f"{name} files changed", path=folder),
                    f"{name}_reidentification",
                    folder,
                )
    receipt["inputs_unchanged"] = (
        False if False in unchanged else None if None in unchanged else True
    )
    if receipt["inputs_unchanged"] is not True:
        receipt["status"] = "FAIL"


def _cli(python: str, out: Path, label: str, *arguments: object) -> dict:
    command = [python, "-I", "-B", "-m", "selcal", *map(str, arguments)]
    try:
        run = subprocess.run(command, capture_output=True, timeout=300)
    except subprocess.TimeoutExpired as error:
        (out / f"{label}.stdout").write_bytes(error.output or b"")
        (out / f"{label}.stderr").write_bytes(error.stderr or b"")
        raise CheckFailed(f"{label}: timed out; partial output retained") from error
    (out / f"{label}.stdout").write_bytes(run.stdout)
    (out / f"{label}.stderr").write_bytes(run.stderr)
    if run.returncode != 0:
        raise CheckFailed(f"{label}: exit {run.returncode}")
    return json.loads(run.stdout.decode("utf-8"))


def _unpack(archive: Path, target: Path) -> None:
    try:
        try:
            zipped = zipfile.ZipFile(archive)
        except NotImplementedError as error:
            # ZipFile rejects unsupported required-extraction versions at construction.
            # Keep this conversion local: unrelated errors from later code must propagate.
            raise CheckFailed(f"unsupported ZIP archive: {error}") from error
        with zipped:
            members = zipped.infolist()
            names = [member.filename for member in members]
            if (
                len(members) != ZIP_MEMBERS
                or len(set(names)) != ZIP_MEMBERS
                or sum(member.file_size for member in members) > ZIP_TOTAL
                or any(
                    member.is_dir()
                    or not name
                    or name in {".", ".."}
                    or any(c in name for c in "/\\:\x00")
                    or stat.S_IFMT(member.external_attr >> 16) not in {0, stat.S_IFREG}
                    or member.flag_bits & (1 | 32 | 64)
                    or member.compress_type not in {
                        zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED,
                        zipfile.ZIP_BZIP2, zipfile.ZIP_LZMA,
                    }
                    for member, name in zip(members, names, strict=True)
                )
            ):
                raise CheckFailed("ZIP member count, names, types, sizes or encoding are unsafe")
            # Validate all bounded bytes, including each CRC, before creating an output folder.
            # ZIP_TOTAL caps this capture at 32 MiB; failed archives leave no partial extraction.
            captured = [(member.filename, zipped.read(member)) for member in members]
    except (
        zipfile.BadZipFile, zipfile.LargeZipFile, EOFError, OSError,
        zlib.error, lzma.LZMAError,
    ) as error:
        # Bad central directories AND malformed member streams are failed files, not crashes.
        raise CheckFailed(f"not a readable ZIP archive: {error}") from error
    target.mkdir()
    for name, raw in captured:
        (target / name).write_bytes(raw)


def _locate(snapshot: Path, job8: str, op8: str, base: str) -> tuple[str, str]:
    jobs = [
        p
        for p in (snapshot / "jobs").iterdir()
        if FULL_ID.fullmatch(p.name) and p.name.startswith(job8)
    ]
    if len(jobs) != 1:
        raise CheckFailed(f"{len(jobs)} jobs match {job8}")
    operations = [
        p
        for p in (jobs[0] / "operations").iterdir()
        if FULL_ID.fullmatch(p.name) and p.name.startswith(op8) and (p / base).is_file()
    ]
    if len(operations) != 1:
        raise CheckFailed(f"{len(operations)} operations with {base} match {op8}")
    return jobs[0].name, operations[0].name


def check(args: argparse.Namespace) -> dict:
    from selcal.ui_jobs import JobManager, UIError

    out: Path = args.out
    receipt: dict = {
        "schema": SCHEMA,
        "status": "FAIL",
        "files": [],
        "first_error": None,
        "failure_phase": None,
        "failure_path": None,
        "errors": [],
        "source_access": "NOT_CHECKED",
        "content_checks": "NOT_CHECKED",
        "input_checks": {
            "workspace": {"baseline": "NOT_ESTABLISHED", "status": "NOT_CHECKED"},
            "saved": {"baseline": "NOT_ESTABLISHED", "status": "NOT_CHECKED"},
        },
        "inputs_unchanged": None,
        "snapshot_cleanup": {
            "status": "NOT_CREATED", "path": None, "manager_close": "NOT_CREATED"
        },
        "started_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "workspace": str(args.workspace),
        "saved": str(args.saved),
        "allow_history": args.allow_history,
        "browser_save_observed": "NOT_OBSERVED_BY_THIS_TOOL",
    }
    workspace_before = saved_before = snapshot_root = None
    manager = None
    phase, failed_path = "input_folders", None
    try:
        for folder in (args.workspace, args.saved):
            failed_path = folder
            _plain_directory(folder)
        expect = {}
        if args.expect:
            phase, failed_path = "expectation", args.expect
            spec = json.loads(args.expect.read_text(encoding="utf-8"))
            expect = {case["raw_input_sha256"]: case for case in spec["cases"]}
            receipt["expectation"] = _identity(args.expect)
        phase, failed_path = "source_offline_check", args.workspace / "writer.lock"
        _require_offline_workspace(args.workspace)
        receipt["source_access"] = "OFFLINE_AVAILABLE"
        phase, failed_path = "workspace_baseline", args.workspace
        workspace_before = _tree(args.workspace)
        receipt["input_checks"]["workspace"]["baseline"] = "COMPLETE"
        phase, failed_path = "saved_baseline", args.saved
        saved_before = _saved_tree(args.saved)
        saved_files = [args.saved / name for name in saved_before]
        receipt["input_checks"]["saved"]["baseline"] = "COMPLETE"
        if not saved_files:
            raise CheckFailed("no saved files")
        if args.count is not None and len(saved_files) != args.count:
            raise CheckFailed(f"expected {args.count} saved files, found {len(saved_files)}")
        # The snapshot lives in the new output folder (the drive the tester chose), resolved
        # because the product refuses workspaces reached through a link; removed afterwards.
        phase, failed_path = "snapshot_create", out / "workspace-snapshot"
        candidate_root = failed_path.resolve()
        candidate_root.mkdir()
        # Only remove a directory this invocation successfully created, never a collision.
        snapshot_root = candidate_root
        receipt["snapshot_cleanup"].update(status="PENDING", path=str(snapshot_root))
        snapshot = snapshot_root / "workspace"
        phase, failed_path = "snapshot_copy", args.workspace
        shutil.copytree(args.workspace, snapshot, symlinks=True)
        phase, failed_path = "snapshot_open", snapshot
        manager = JobManager(snapshot)
        receipt["snapshot_cleanup"]["manager_close"] = "PENDING"
        receipt["content_checks"] = "FAIL"
        for saved in saved_files:
            phase, failed_path = "saved_file_check", saved
            row: dict = {"saved": saved.name, "status": "FAIL"}
            receipt["files"].append(row)
            _regular(saved)
            row.update(_identity(saved))
            renamed = RENAMED.fullmatch(saved.name)
            name = renamed.group(1) + renamed.group(2) if renamed else saved.name
            row["renamed_by_browser"] = bool(renamed)
            match = NAME.fullmatch(name)
            if not match:
                raise CheckFailed(f"not a SelCal download name: {saved.name}")
            job8, op8, base = match.groups()
            kind = KIND[base]
            job_id, operation_id = _locate(snapshot, job8, op8, base)
            row.update(job_id=job_id, operation_id=operation_id, kind=kind)
            original = snapshot / "jobs" / job_id / "operations" / operation_id / base
            row["file_equality"] = "PASS" if original.read_bytes() == saved.read_bytes() else "FAIL"
            if row["file_equality"] != "PASS":
                raise CheckFailed(f"{saved.name}: bytes differ from the workspace original")
            current_ref = json.loads(
                (snapshot / "jobs" / job_id / f"{kind}.json").read_text(encoding="utf-8")
            )["operation_id"]
            if current_ref == operation_id:
                row["reference_scope"] = "current"
                try:
                    raw, _, served_name, _ = manager.download(job_id, kind)
                except UIError as error:
                    row["job_binding"] = "FAIL"
                    raise CheckFailed(f"{saved.name}: product check refused: {error}") from error
                row["job_binding"] = (
                    "PASS" if raw == saved.read_bytes() and served_name == name else "FAIL"
                )
                if row["job_binding"] != "PASS":
                    raise CheckFailed(f"{saved.name}: differs from the product-checked download")
            else:
                row["reference_scope"] = "history"
                row["job_binding"] = "NOT_APPLICABLE_HISTORY"
                if not args.allow_history:
                    raise CheckFailed(f"{saved.name}: not the job's current {kind}")
            label = saved.stem
            record_path = (
                snapshot
                / "jobs"
                / job_id
                / "operations"
                / json.loads(
                    (snapshot / "jobs" / job_id / "record.json").read_text(encoding="utf-8")
                )["operation_id"]
                / "result.sqlite"
            )
            if kind == "record":
                phase = "record_replay"
                data = _cli(
                    args.python,
                    out,
                    label + "-replay",
                    "verify",
                    saved,
                    "--max-bytes",
                    args.max_bytes,
                    "--replay",
                )["data"]
                row["record_replay"] = "PASS" if data["replay"] == "MATCH" else "FAIL"
                case = expect.get(manager.get(job_id)["input_sha256"])
                if case:
                    observed = (
                        data["selected_candidate"],
                        data["exceedance_count"],
                        data["retained_replicates"],
                        data["p_value"],
                        data["reject_null"],
                    )
                    wanted = (
                        case["selected_candidate"],
                        case["exceedance_count"],
                        case["replicates"],
                        (case["exceedance_count"] + 1) / (case["replicates"] + 1),
                        case["reject_null"],
                    )
                    row["expected_result"] = "PASS" if observed == wanted else "FAIL"
                    row["case"] = case["name"]
                if row["record_replay"] != "PASS" or row.get("expected_result") == "FAIL":
                    raise CheckFailed(f"{saved.name}: replay or expected result failed")
            elif kind == "bundle":
                phase = "export_content"
                target = out / (label + "-unpacked")
                _unpack(saved, target)
                _cli(
                    args.python,
                    out,
                    label + "-verify-export",
                    "verify-export",
                    target,
                    "--max-bytes",
                    args.max_bytes,
                )
                row["export_content"] = "PASS"
            else:
                phase = "html_content"
                text = saved.read_bytes().decode("utf-8")  # strict: a non-UTF-8 file fails
                if not (
                    text.lstrip().lower().startswith("<!doctype html")
                    and text.rstrip().lower().endswith("</html>")
                ):
                    row["html_content"] = "FAIL"
                    raise CheckFailed(f"{saved.name}: not a complete HTML document")
                regenerated = out / (label + "-regenerated.html")
                _cli(
                    args.python,
                    out,
                    label + "-regenerate",
                    "report",
                    record_path,
                    regenerated,
                    "--max-bytes",
                    args.max_bytes,
                )
                row["html_content"] = (
                    "PASS" if regenerated.read_bytes() == saved.read_bytes() else "FAIL"
                )
                if row["html_content"] != "PASS":
                    raise CheckFailed(f"{saved.name}: not the report of the job's checked record")
            row["status"] = "PASS"
        receipt["content_checks"] = "PASS"
        receipt["status"] = "PASS"
    except (CheckFailed, OSError, ValueError, KeyError, UnicodeDecodeError) as error:
        _record_error(receipt, error, phase, failed_path)
    finally:
        cleanup = receipt["snapshot_cleanup"]
        if manager is not None:
            try:
                manager.close()
            except Exception as error:
                # Keep the snapshot rather than unlinking a manager that did not close.
                cleanup.update(status="FAIL", manager_close="FAIL")
                _record_error(receipt, error, "snapshot_manager_close", snapshot_root)
            else:
                cleanup["manager_close"] = "PASS"
        if snapshot_root is not None and cleanup["manager_close"] != "FAIL":
            try:
                shutil.rmtree(snapshot_root)
            except OSError as error:
                cleanup["status"] = "FAIL"
                _record_error(receipt, error, "snapshot_cleanup", snapshot_root)
            else:
                cleanup["status"] = "PASS"
        _reidentify(receipt, args, workspace_before, saved_before)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog=(
            "OFFLINE WORKSPACE ONLY: finish downloads and stop the source UI helper normally "
            "before checking. Never delete writer.lock. A live or unreadable workspace is "
            "NOT_CHECKED, not proof of corrupt downloads or unchanged inputs. Snapshot cleanup "
            "errors cause failure and are retained in receipt.json."
        ),
    )
    parser.add_argument("--python", required=True)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--saved", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--expect", type=Path)
    parser.add_argument("--count", type=int)
    parser.add_argument("--max-bytes", default="8388608")
    parser.add_argument("--allow-history", action="store_true")
    args = parser.parse_args(argv)
    args.workspace, args.saved = args.workspace.absolute(), args.saved.absolute()
    args.out = args.out.absolute()
    if any(args.out.is_relative_to(folder) for folder in (args.workspace, args.saved)):
        print("The output folder must be outside the workspace and saved folders.", file=sys.stderr)
        return 2
    try:
        _plain_directory(args.out.parent, allow_missing=True)
        args.out.mkdir(parents=True, exist_ok=False)
    except (OSError, CheckFailed) as error:
        print(f"Output folder must be new and writable: {error}", file=sys.stderr)
        return 2
    receipt = check(args)
    receipt["finished_at_utc"] = datetime.datetime.now(datetime.UTC).isoformat()
    (args.out / "receipt.json").write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "first_error": receipt["first_error"],
                "receipt": str(args.out / "receipt.json"),
            },
            # Keep redirected Windows console output independent of its code page.
            # The full human-readable receipt above remains strict UTF-8.
            ensure_ascii=True,
        )
    )
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
