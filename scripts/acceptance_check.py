"""End-to-end acceptance check of an installed SelCal, on macOS, Linux or Windows.

Run from the repository root with the Python that has SelCal installed:

    python scripts/acceptance_check.py [--records DIR] [--out DIR]

For both shipped example plans (sampled circular null and exact enumeration) it runs
doctor, validate, run, verify, verify --replay, verify --replay-decision and report, and checks
the results, not only exit codes: selected lag 2, p = 0.015 / 0.03, null rejected, byte replay
MATCH, decision replay DECISION_MATCH, and the HTML report showing the same p-value and lag.
With --legacy-records it checks records of an earlier version: still readable, but byte and
decision replay are refused as environment_mismatch because the code differs.
With --records it also checks records made elsewhere (e.g. docs/status/evidence/windows_check/
records): plain verify must pass; byte replay must fail with exactly environment_mismatch when
the platform differs; decision replay must give DECISION_MATCH within the stated tolerance.
Use the same NumPy version as the records: cross-NumPy record verification is not supported.
An explicitly requested record directory must contain records; missing, empty or unreadable
inputs are failures, not skipped checks.

--precision adds the checks that the results stay exact in use: the same input and plan give a
byte-identical record in five separate processes, a record keeps its meaning when it is moved to
another folder, and a record with a single altered byte is refused.

Every command's argv, working folder, exit code, stdout and stderr are kept in a new folder
(nothing is overwritten). The script exits 0 only if every check passed, 1 otherwise.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "workflow"
CAP = "1048576"
EXPECTED = {"pearson": {"lag": 2, "p": 0.015}, "exact": {"lag": 2, "p": 0.03}}


class Checker:
    def __init__(self, out: Path) -> None:
        self.out = out
        self.failures: list[str] = []
        self.count = 0

    def cli(self, name: str, *args: str, cwd: Path) -> tuple[int, dict[str, Any] | None]:
        for suffix in ("stdout", "stderr", "receipt.json"):
            if (self.out / f"{name}.{suffix}").exists():
                raise FileExistsError(f"command evidence already exists: {name}.{suffix}")
        argv = [sys.executable, "-m", "selcal", *args]
        done = subprocess.run(argv, cwd=cwd, capture_output=True, check=False)
        (self.out / f"{name}.stdout").write_bytes(done.stdout)
        (self.out / f"{name}.stderr").write_bytes(done.stderr)
        receipt = {"argv": argv, "cwd": str(cwd), "exit_code": done.returncode}
        (self.out / f"{name}.receipt.json").write_text(
            json.dumps(receipt, indent=1) + "\n", encoding="utf-8", newline="\n"
        )
        try:
            payload = json.loads(done.stdout.decode("utf-8").strip().splitlines()[-1])
        except (IndexError, UnicodeDecodeError, json.JSONDecodeError):
            payload = None
        return done.returncode, payload

    def check(self, label: str, condition: bool, detail: str = "") -> None:
        self.count += 1
        status = "PASS" if condition else "FAIL"
        line = f"{status}  {label}" + (f"  ({detail})" if detail and not condition else "")
        print(line, flush=True)
        with (self.out / "summary.txt").open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(line + "\n")
        if not condition:
            self.failures.append(label)


def _plans(work: Path) -> dict[str, Path]:
    (work / "series.csv").write_bytes((EXAMPLE / "series.csv").read_bytes())
    pearson = json.loads((EXAMPLE / "pearson.json").read_text(encoding="utf-8"))
    exact = json.loads(json.dumps(pearson))
    exact["plan"].update(null_name="circular_shift_exact_v1", replicates=99)
    paths = {}
    for name, config in (("pearson", pearson), ("exact", exact)):
        paths[name] = work / f"{name}.json"
        paths[name].write_text(json.dumps(config, indent=1), encoding="utf-8", newline="\n")
    return paths


def _case(c: Checker, name: str, plan: Path, work: Path) -> None:
    expected = EXPECTED[name]
    record, report = f"{name}.sqlite", f"{name}.html"
    code, out = c.cli(f"{name}-validate", "validate", "series.csv", plan.name, cwd=work)
    c.check(f"{name}: validate exit 0", code == 0, f"exit {code}")
    preflight = ((out or {}).get("data") or {}).get("preflight") or {}
    c.check(
        f"{name}: validate reads 100 samples, plan EXECUTABLE",
        (preflight.get("input") or {}).get("sample_count") == 100
        and (preflight.get("plan") or {}).get("status") == "EXECUTABLE",
        f"error {(out or {}).get('error')}: {(out or {}).get('detail')}",
    )
    code, out = c.cli(
        f"{name}-run", "run", "series.csv", plan.name, record, "--max-bytes", CAP, cwd=work
    )
    data = (out or {}).get("data") or {}
    c.check(
        f"{name}: run exit 0 and record saved",
        code == 0 and (work / record).is_file(),
        f"exit {code}, error {(out or {}).get('error')}",
    )
    c.check(
        f"{name}: lag {expected['lag']}, p = {expected['p']}, null rejected",
        data.get("selected_candidate") == expected["lag"]
        and data.get("p_value") == expected["p"]
        and data.get("reject_null") is True,
        f"lag {data.get('selected_candidate')}, p {data.get('p_value')}",
    )
    for mode, flag, verdict in (
        ("verify", None, "NOT_PERFORMED"),
        ("replay", "--replay", "MATCH"),
        ("decision", "--replay-decision", "DECISION_MATCH"),
    ):
        args = ["verify", record, "--max-bytes", CAP] + ([flag] if flag else [])
        code, out = c.cli(f"{name}-{mode}", *args, cwd=work)
        got = ((out or {}).get("data") or {}).get("replay")
        c.check(
            f"{name}: verify {flag or ''} -> {verdict}".replace("  ", " "),
            code == 0 and got == verdict,
            f"exit {code}, {got}, error {(out or {}).get('error')}",
        )
    code, out = c.cli(f"{name}-report", "report", record, report, "--max-bytes", CAP, cwd=work)
    text = (work / report).read_text(encoding="utf-8") if (work / report).is_file() else ""
    c.check(f"{name}: report exit 0 and HTML written", code == 0 and bool(text), f"exit {code}")
    shown = html.unescape(text)
    c.check(
        f"{name}: report shows p-value {expected['p']} and lag {expected['lag']}",
        f'"p_value": {expected["p"]}' in shown
        and f'"selected_candidate": {expected["lag"]}' in shown,
    )


def _recorded_software(path: Path) -> dict[str, Any] | str:
    """The software identity stored in a record, or an error text if it cannot be read."""
    from selcal import workflow_store

    try:
        members = workflow_store.read_record(path, max_bytes=int(CAP))
        software = json.loads(members["metadata"])["software"]
        if not isinstance(software, dict):
            return "unreadable: software identity is not an object"
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        return f"unreadable: {type(error).__name__}: {error}"
    return software


def _record_sources(c: Checker, records: Path, label: str) -> list[Path]:
    """An explicitly requested record check must examine at least one input."""
    try:
        if not records.is_dir():
            raise ValueError("record directory is missing or is not a directory")
        sources = sorted(records.glob("*.sqlite"))
        if not sources:
            raise ValueError("record directory contains no .sqlite records")
        invalid = [source.name for source in sources if not source.is_file()]
        if invalid:
            raise ValueError(f"record inputs are not regular files: {', '.join(invalid)}")
    except (OSError, ValueError) as error:
        c.check(f"{label}: record inputs available", False, f"{records}: {error}")
        return []
    c.check(f"{label}: record inputs available", True)
    return sources


def _copy_record(c: Checker, source: Path, local: Path) -> bool:
    """Keep unreadable inputs in the failure report instead of aborting the checker."""
    try:
        local.write_bytes(source.read_bytes())
    except OSError as error:
        c.check(f"{source.name}: record copied for checking", False, str(error))
        return False
    return True


def _foreign_records(c: Checker, records: Path, work: Path) -> None:
    """Records made by the same SelCal code, possibly elsewhere (other Python, NumPy, platform)."""
    from selcal import workflow

    current = workflow._software_identity()
    for source in _record_sources(c, records, "cross-platform"):
        name = f"foreign-{source.stem}"
        local = work / f"foreign-{source.name}"
        if not _copy_record(c, source, local):
            continue
        recorded = _recorded_software(local)
        if isinstance(recorded, str):
            c.check(f"{source.name}: record readable", False, recorded)
            continue
        same_code = all(recorded.get(k) == current[k] for k in ("selcal_version", "source_files"))
        c.check(
            f"{source.name}: made by the installed SelCal code",
            same_code,
            "made by different code (version or source files differ); regenerate these records "
            "with the code under test, or check them with --legacy-records",
        )
        code, out = c.cli(f"{name}-verify", "verify", local.name, "--max-bytes", CAP, cwd=work)
        c.check(
            f"{source.name}: verify exit 0", code == 0, f"exit {code}, {(out or {}).get('error')}"
        )
        code, out = c.cli(
            f"{name}-replay", "verify", local.name, "--max-bytes", CAP, "--replay", cwd=work
        )
        if recorded == current:
            c.check(
                f"{source.name}: same code, Python, NumPy and platform, byte replay MATCH",
                code == 0 and ((out or {}).get("data") or {}).get("replay") == "MATCH",
                f"exit {code}, {(out or {}).get('error')}",
            )
        else:
            differs = [
                key
                for key in ("python_version", "numpy_version", "platform")
                if recorded.get(key) != current[key]
            ]
            c.check(
                f"{source.name}: byte replay refused as environment_mismatch "
                f"(recorded {', '.join(differs) or 'code'} differs)",
                code == 4 and (out or {}).get("error") == "environment_mismatch",
                f"exit {code}, {(out or {}).get('error')}",
            )
        code, out = c.cli(
            f"{name}-decision",
            "verify",
            local.name,
            "--max-bytes",
            CAP,
            "--replay-decision",
            cwd=work,
        )
        data = (out or {}).get("data") or {}
        detail = data.get("decision_replay") or {}
        c.check(
            f"{source.name}: decision replay DECISION_MATCH within tolerance",
            code == 0
            and data.get("replay") == "DECISION_MATCH"
            and detail.get("max_statistic_ulp", 10**9) <= detail.get("tolerance_ulp", -1),
            f"exit {code}, {(out or {}).get('error')}, gap {detail.get('max_statistic_ulp')}",
        )


def _legacy_records(c: Checker, records: Path, work: Path) -> None:
    """Records made by an earlier SelCal version: readable, but every replay is refused."""
    for source in _record_sources(c, records, "legacy"):
        name = f"legacy-{source.stem}"
        local = work / f"legacy-{source.name}"
        if not _copy_record(c, source, local):
            continue
        code, out = c.cli(f"{name}-verify", "verify", local.name, "--max-bytes", CAP, cwd=work)
        c.check(
            f"legacy {source.name}: still readable, verify exit 0",
            code == 0,
            f"exit {code}, {(out or {}).get('error')}",
        )
        for mode, flag in (("replay", "--replay"), ("decision", "--replay-decision")):
            code, out = c.cli(
                f"{name}-{mode}", "verify", local.name, "--max-bytes", CAP, flag, cwd=work
            )
            c.check(
                f"legacy {source.name}: {flag} refused as environment_mismatch (other code)",
                code == 4 and (out or {}).get("error") == "environment_mismatch",
                f"exit {code}, {(out or {}).get('error')}",
            )


def _precision(c: Checker, work: Path) -> None:
    """The results must stay exact in use: repeatable, movable, and tamper-evident."""
    plans = _plans(work)
    for name, plan in plans.items():
        expected = EXPECTED[name]
        digests, values = set(), set()
        for attempt in range(5):
            record = f"repeat{attempt}-{name}.sqlite"
            code, out = c.cli(
                f"{name}-repeat{attempt}",
                "run",
                "series.csv",
                plan.name,
                record,
                "--max-bytes",
                CAP,
                cwd=work,
            )
            data = (out or {}).get("data") or {}
            digests.add(hashlib.sha256((work / record).read_bytes()).hexdigest())
            values.add(
                (
                    code,
                    data.get("selected_candidate"),
                    data.get("p_value"),
                    data.get("decision_statistic"),
                    data.get("reject_null"),
                )
            )
        c.check(
            f"{name}: five separate runs give one identical record (byte for byte)",
            len(digests) == 1,
            f"{len(digests)} different records",
        )
        c.check(
            f"{name}: five separate runs give the same lag, statistic, p and decision",
            values == {(0, expected["lag"], expected["p"], values and next(iter(values))[3], True)},
            f"{sorted(values)}",
        )

        # A record keeps its meaning when the user moves it somewhere else.
        moved = work / "moved" / f"{name}.sqlite"
        moved.parent.mkdir(exist_ok=True)
        shutil.copy2(work / f"repeat0-{name}.sqlite", moved)
        code, out = c.cli(
            f"{name}-moved", "verify", str(moved), "--max-bytes", CAP, "--replay", cwd=work
        )
        summary = (out or {}).get("data") or {}
        c.check(
            f"{name}: a record moved to another folder still replays (MATCH) with the same p",
            code == 0
            and summary.get("replay") == "MATCH"
            and summary.get("p_value") == expected["p"],
            f"exit {code}, {(out or {}).get('error')}, p {summary.get('p_value')}",
        )

        # One altered byte inside the stored result must be refused, not silently accepted.
        tampered = work / f"tampered-{name}.sqlite"
        raw = bytearray((work / f"repeat0-{name}.sqlite").read_bytes())
        marker = raw.find(b'"p_value"')
        raw[marker] = ord("q") if marker >= 0 else raw[marker]
        tampered.write_bytes(bytes(raw))
        code, out = c.cli(f"{name}-tampered", "verify", str(tampered), "--max-bytes", CAP, cwd=work)
        c.check(
            f"{name}: a record with one altered byte is refused",
            marker >= 0 and code == 4,
            f"marker at {marker}, exit {code}, {(out or {}).get('error')}",
        )


def _installation(c: Checker, work: Path) -> None:
    """Report where SelCal is imported from: an installed package, not the source folder."""
    code, out = c.cli("installation", "doctor", cwd=work)
    data = (out or {}).get("data") or {}
    done = subprocess.run(
        [sys.executable, "-c", "import selcal, sys; print(selcal.__file__); print(sys.version)"],
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
        cwd=work,
    )
    location = done.stdout.strip().splitlines()[0] if done.stdout.strip() else "unknown"
    (c.out / "installation.txt").write_text(
        f"selcal.__file__: {location}\n{done.stdout}\n", encoding="utf-8", newline="\n"
    )
    c.check(
        "SelCal is imported from an installed package, not from a source checkout",
        "site-packages" in location,
        location,
    )
    c.check(
        f"doctor reports version {data.get('selcal_version')} and a working record round trip",
        code == 0 and data.get("record_round_trip") == "PASS",
        f"exit {code}, {data.get('record_round_trip')}",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--records", type=Path, help="folder with records made elsewhere")
    parser.add_argument(
        "--legacy-records", type=Path, help="folder with records made by an earlier version"
    )
    parser.add_argument(
        "--precision",
        action="store_true",
        help="also check repeatability, relocation and tamper rejection",
    )
    parser.add_argument(
        "--installed",
        action="store_true",
        help="also check that SelCal is imported from an installed package",
    )
    parser.add_argument("--out", type=Path, help="parent folder for results (default: here)")
    args = parser.parse_args()
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    # Child commands run inside work; output paths must not be reinterpreted from there.
    out = (args.out or Path.cwd()).resolve() / f"selcal-acceptance-{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    work = out / "work"
    work.mkdir()
    c = Checker(out)
    print(f"SelCal acceptance check; results in {out}", flush=True)
    code, report = c.cli("doctor", "doctor", cwd=work)
    data = (report or {}).get("data") or {}
    c.check(
        "doctor exit 0, records can be saved and read back",
        code == 0 and data.get("record_round_trip") == "PASS",
        f"exit {code}, {data.get('record_round_trip')}",
    )
    for name, plan in _plans(work).items():
        _case(c, name, plan, work)
    if args.records:
        _foreign_records(c, args.records, work)
    if args.legacy_records:
        _legacy_records(c, args.legacy_records, work)
    if args.precision:
        _precision(c, work)
    if args.installed:
        _installation(c, work)
    verdict = "ALL PASSED" if not c.failures else f"{len(c.failures)} FAILED"
    print(f"{verdict} ({c.count} checks)", flush=True)
    with (out / "summary.txt").open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(f"{verdict} ({c.count} checks)\n")
    return 0 if not c.failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
