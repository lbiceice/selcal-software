"""Check that an installed SelCal is exactly the files of the intended wheel.

Usage: python -I scripts/verify_installed_identity.py [--wheel SELCAL.whl] [--distribution NAME]

Every installed file listed with a hash in the distribution's RECORD must still have that hash,
so an edited or replaced file is found. With --wheel, the installed RECORD must also list the same
files and hashes as the wheel's own RECORD, so an installation from another build is found.
Prints one JSON document; exit 0 only if every check passed, 1 on a mismatch, 2 on a usage or
read error. An acceptance runner that reuses an existing installation should run this first and
install into a new folder if it fails, never trust the installation because a marker file exists.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib.metadata
import io
import json
import sys
import zipfile
from pathlib import Path


def _record_rows(text: str) -> dict[str, str]:
    """RECORD path -> "sha256=..." for rows that carry a hash."""
    rows: dict[str, str] = {}
    for row in csv.reader(io.StringIO(text)):
        if len(row) >= 2 and row[1]:
            rows[row[0]] = row[1]
    return rows


def _digest(data: bytes) -> str:
    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")


def check(distribution: str, wheel: Path | None) -> dict[str, object]:
    dist = importlib.metadata.distribution(distribution)
    record = dist.read_text("RECORD")
    if record is None:
        raise ValueError("the installed distribution has no RECORD")
    installed = _record_rows(record)
    mismatched, missing = [], []
    for name, expected in sorted(installed.items()):
        if not expected.startswith("sha256="):
            raise ValueError(f"unsupported RECORD hash for {name}")
        path = Path(str(dist.locate_file(name)))
        try:
            actual = _digest(path.read_bytes())
        except FileNotFoundError:
            missing.append(name)
            continue
        if actual != expected:
            mismatched.append(name)
    report: dict[str, object] = {
        "distribution": distribution,
        "version": dist.version,
        "files_checked": len(installed),
        "missing": missing,
        "changed": mismatched,
    }
    if wheel is not None:
        with zipfile.ZipFile(wheel) as archive:
            names = [n for n in archive.namelist() if n.endswith(".dist-info/RECORD")]
            if len(names) != 1:
                raise ValueError("the wheel must hold exactly one RECORD")
            wheel_rows = _record_rows(archive.read(names[0]).decode("utf-8"))
        report["wheel"] = wheel.name
        report["not_from_wheel"] = sorted(
            set(installed.items()).symmetric_difference(wheel_rows.items())
        )
    report["status"] = (
        "PASS" if not missing and not mismatched and not report.get("not_from_wheel") else "FAIL"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--distribution", default="selcal")
    args = parser.parse_args(argv)
    try:
        report = check(args.distribution, args.wheel)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile,
            importlib.metadata.PackageNotFoundError) as error:
        print(json.dumps({"status": "ERROR", "detail": str(error)}))
        return 2
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
