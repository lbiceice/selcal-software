"""Check files a person saved from the SelCal page against the UI workspace that produced them.

Usage::

    python -I scripts/check_saved_downloads.py --python PYTHON --workspace UI_WORKSPACE \\
        --saved SAVED_DIR --out NEW_DIR [--expect EXPECT.json] [--allow-history]

Each saved ``selcal-<job8>-<op8>-{report.html,result.sqlite,evidence.zip}`` file is matched to one
complete job and operation. The workspace is never opened for writing: it is copied to a snapshot
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
file passes every check that applies to it. Inputs are re-identified afterwards.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import re
import shutil
import stat
import subprocess
import sys
import zipfile
from pathlib import Path

SCHEMA = "selcal.saved-downloads-check.v2"
NAME = re.compile(r"selcal-([0-9a-f]{8})-([0-9a-f]{8})-(report\.html|result\.sqlite|evidence\.zip)")
# Browsers add " (1)" when a name already exists in the folder.
RENAMED = re.compile(r"(selcal-[0-9a-f]{8}-[0-9a-f]{8}-(?:report|result|evidence)) \(\d+\)(\.\w+)")
KIND = {"report.html": "report", "result.sqlite": "record", "evidence.zip": "bundle"}
FULL_ID = re.compile(r"[0-9a-f]{32}")
ZIP_MEMBERS, ZIP_TOTAL = 11, 32 * 1024 * 1024


class CheckFailed(Exception):
    pass


def _identity(path: Path) -> dict:
    raw = path.read_bytes()
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _regular(path: Path) -> None:
    for node in (path, *path.parents):
        info = node.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise CheckFailed(f"linked path refused: {node}")
    if not path.is_file() or path.stat().st_size == 0:
        raise CheckFailed(f"empty or nonregular file: {path.name}")


def _tree(folder: Path) -> dict:
    result = {}
    for path in sorted(folder.rglob("*")):
        if path.is_symlink():
            raise CheckFailed(f"linked workspace member refused: {path}")
        if path.is_file():
            result[path.relative_to(folder).as_posix()] = _identity(path)
    return result


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
    with zipfile.ZipFile(archive) as zipped:
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
                for member, name in zip(members, names, strict=True)
            )
        ):
            raise CheckFailed("ZIP member count, names, types or sizes are unsafe")
        target.mkdir()
        for member in members:
            (target / member.filename).write_bytes(zipped.read(member))


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
        "started_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "workspace": str(args.workspace),
        "saved": str(args.saved),
        "allow_history": args.allow_history,
        "browser_save_observed": "NOT_OBSERVED_BY_THIS_TOOL",
    }
    workspace_before = saved_before = snapshot_root = None
    try:
        for folder in (args.workspace, args.saved):
            if folder.is_symlink() or not folder.is_dir():
                raise CheckFailed(f"not a plain folder: {folder}")
        expect = {}
        if args.expect:
            spec = json.loads(args.expect.read_text(encoding="utf-8"))
            expect = {case["raw_input_sha256"]: case for case in spec["cases"]}
            receipt["expectation"] = _identity(args.expect)
        workspace_before = _tree(args.workspace)
        saved_files = sorted(p for p in args.saved.iterdir())
        saved_before = {p.name: _identity(p) for p in saved_files if p.is_file()}
        if not saved_files:
            raise CheckFailed("no saved files")
        if args.count is not None and len(saved_files) != args.count:
            raise CheckFailed(f"expected {args.count} saved files, found {len(saved_files)}")
        # The snapshot lives in the new output folder (the drive the tester chose), resolved
        # because the product refuses workspaces reached through a link; removed afterwards.
        snapshot_root = (out / "workspace-snapshot").resolve()
        snapshot_root.mkdir()
        snapshot = snapshot_root / "workspace"
        shutil.copytree(args.workspace, snapshot, symlinks=True)
        manager = JobManager(snapshot)
        for saved in saved_files:
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
        receipt["status"] = "PASS"
    except (CheckFailed, OSError, ValueError, KeyError, UnicodeDecodeError) as error:
        receipt["first_error"] = f"{type(error).__name__}: {error}"
    finally:
        if snapshot_root is not None:
            shutil.rmtree(snapshot_root, ignore_errors=True)
        try:
            same_workspace = workspace_before is None or _tree(args.workspace) == workspace_before
            same_saved = saved_before is None or saved_before == {
                p.name: _identity(p) for p in args.saved.iterdir() if p.is_file()
            }
            unchanged = same_workspace and same_saved
        except (OSError, CheckFailed) as error:
            unchanged = False
            receipt["first_error"] = receipt["first_error"] or f"re-identification failed: {error}"
        receipt["inputs_unchanged"] = unchanged
        if not unchanged:
            receipt["status"] = "FAIL"
            receipt["first_error"] = receipt["first_error"] or "workspace or saved files changed"
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
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
        args.out.mkdir(parents=True, exist_ok=False)
    except OSError as error:
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
            ensure_ascii=False,
        )
    )
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
