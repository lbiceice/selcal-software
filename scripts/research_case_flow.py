"""Run named research cases through every SelCal CLI step and keep evidence for any failure.

Usage::

    python -I scripts/research_case_flow.py --python PYTHON --data DIR \\
        --expect EXPECT.json --out NEW_DIR

EXPECT.json lists the cases: ``{"schema": "selcal.research-cases.v1", "max_bytes": 8388608,
"cases": [{"name": ..., "input": ..., "config": ..., "selected_candidate": ...,
"exceedance_count": ..., "replicates": ..., "reject_null": ...}]}``. The p-value is checked as
exactly ``(E + 1) / (B + 1)``.

Every step is written to the receipt before it starts. A timeout keeps the partial stdout and
stderr as raw bytes, the first failure stays the reported error, and input identities are
re-read in ``finally`` (unchanged, changed or unreadable are recorded separately). The receipt is
UTF-8 JSON and the exit code is nonzero unless every step and expectation passed. This checks
software execution and arithmetic on the given cases, not the cases' scientific assumptions.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

SCHEMA = "selcal.research-case-flow.v2"
STEP_TIMEOUT_SECONDS = 180


def _now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


def _identity(path: Path) -> dict:
    raw = path.read_bytes()
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


class StepFailed(Exception):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--python", required=True)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--expect", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=STEP_TIMEOUT_SECONDS)
    args = parser.parse_args(argv)
    out = args.out.absolute()
    try:
        out.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        print(f"Output folder must be new and writable: {error}", file=sys.stderr)
        return 2
    receipt: dict = {
        "schema": SCHEMA,
        "status": "FAIL",
        "started_at_utc": _now(),
        "python": args.python,
        "steps": [],
        "cases": {},
        "first_error": None,
        "scientific_assumptions": "NOT_VERIFIED",
        "historical_execution_authenticated": False,
    }

    def save() -> None:
        (out / "receipt.json").write_text(
            json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    def step(name: str, *arguments: object) -> dict:
        stem = f"{len(receipt['steps']):02d}-{name}"
        command = [args.python, "-I", "-m", "selcal", *map(str, arguments)]
        entry = {
            "name": name,
            "command": command,
            "status": "RUNNING",
            "started_at_utc": _now(),
            "timeout_seconds": args.timeout,
            "exit_code": None,
            "stdout": stem + ".stdout",
            "stderr": stem + ".stderr",
        }
        receipt["steps"].append(entry)
        save()
        # Output streams straight to the step's files, so a timeout or crash keeps every byte
        # the child wrote before it stopped.
        try:
            with (
                (out / entry["stdout"]).open("wb") as stdout,
                (out / entry["stderr"]).open("wb") as stderr,
            ):
                child = subprocess.Popen(command, stdout=stdout, stderr=stderr, cwd=out)
                try:
                    code = child.wait(timeout=args.timeout)
                except subprocess.TimeoutExpired as error:
                    child.kill()
                    child.wait()
                    entry.update(
                        status="TIMEOUT", finished_at_utc=_now(), child="killed_and_reaped"
                    )
                    save()
                    raise StepFailed(
                        f"{name}: timed out after {args.timeout} s; partial output retained"
                    ) from error
        except OSError as error:
            entry.update(status="CANNOT_START", finished_at_utc=_now(), error=str(error))
            save()
            raise StepFailed(f"{name}: cannot start: {error}") from error
        entry.update(exit_code=code, finished_at_utc=_now(), status="PASS" if code == 0 else "FAIL")
        save()
        if code != 0:
            raise StepFailed(f"{name}: exit {code}; raw output retained")
        try:
            value = json.loads((out / entry["stdout"]).read_bytes().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            entry["status"] = "MALFORMED_OUTPUT"
            save()
            raise StepFailed(f"{name}: output is not CLI JSON: {error}") from error
        if value.get("exit_code") != 0:
            entry["status"] = "INCONSISTENT_EXIT"
            save()
            raise StepFailed(f"{name}: CLI JSON reports exit {value.get('exit_code')}")
        return value

    inputs: dict[str, dict] = {}
    try:
        spec = json.loads(args.expect.read_text(encoding="utf-8"))
        if spec.get("schema") != "selcal.research-cases.v1" or not spec.get("cases"):
            raise StepFailed("expectation file has an unknown schema or no cases")
        cap = str(spec.get("max_bytes", 8388608))
        for case in spec["cases"]:
            for key in ("input", "config"):
                inputs[case[key]] = _identity(args.data / case[key])
        receipt["expectation"] = _identity(args.expect)
        receipt["inputs_before"] = inputs
        save()
        step("doctor", "doctor")
        for case in spec["cases"]:
            name = case["name"]
            inp, cfg = args.data / case["input"], args.data / case["config"]
            record, report = out / f"{name}.sqlite", out / f"{name}.html"
            export = out / f"{name}_export"
            step(f"{name}-validate", "validate", inp, cfg)
            step(f"{name}-run", "run", inp, cfg, record, "--max-bytes", cap)
            verified = step(f"{name}-verify", "verify", record, "--max-bytes", cap)["data"]
            replay = step(f"{name}-replay", "verify", record, "--max-bytes", cap, "--replay")
            step(f"{name}-report", "report", record, report, "--max-bytes", cap)
            step(f"{name}-export", "export", record, export, "--max-bytes", cap)
            step(f"{name}-verify-export", "verify-export", export, "--max-bytes", cap)
            exceedances, replicates = case["exceedance_count"], case["replicates"]
            observed = {
                "selected_candidate": verified["selected_candidate"],
                "exceedance_count": verified["exceedance_count"],
                "replicates": verified["retained_replicates"],
                "planned_replicates": verified["planned_replicates"],
                "p_value": verified["p_value"],
                "reject_null": verified["reject_null"],
                "failure_count": verified["failure_count"],
                "replay": replay["data"]["replay"],
            }
            expected = {
                "selected_candidate": case["selected_candidate"],
                "exceedance_count": exceedances,
                "replicates": replicates,
                "planned_replicates": replicates,
                "p_value": (exceedances + 1) / (replicates + 1),
                "reject_null": case["reject_null"],
                "failure_count": 0,
                "replay": "MATCH",
            }
            receipt["cases"][name] = {
                "observed": observed,
                "expected": expected,
                "record": _identity(record),
                "report": _identity(report),
            }
            save()
            if observed != expected:
                raise StepFailed(f"{name}: result differs from the expectation")
        receipt["status"] = "PASS"
    except Exception as error:  # every failure is recorded, then the exit code is nonzero
        receipt["status"] = "FAIL"
        receipt["first_error"] = f"{type(error).__name__}: {error}"
    finally:
        after: dict[str, dict] = {}
        for label in inputs:
            try:
                now = _identity(args.data / label)
                after[label] = {**now, "state": "unchanged" if now == inputs[label] else "changed"}
            except OSError as error:
                after[label] = {"state": "unreadable", "error": str(error)}
        receipt["inputs_after"] = after
        if any(item["state"] != "unchanged" for item in after.values()):
            receipt["status"] = "FAIL"
            receipt["first_error"] = receipt["first_error"] or "inputs changed or became unreadable"
        receipt["finished_at_utc"] = _now()
        save()
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "first_error": receipt["first_error"],
                "receipt": str(out / "receipt.json"),
            },
            ensure_ascii=False,
        )
    )
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
