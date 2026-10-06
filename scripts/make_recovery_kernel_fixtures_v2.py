"""Write tests/fixtures/recovery_kernel_v2 from the current kernel; never overwrite.

v1 holds the kernel references before the Pearson numeric-method change (R10v3 Windows core
audit, ALG-01, 2026-10-03) and stays unchanged as history. v2 is generated with the same four
cases, requests and inputs; tests bind v1 -> v2 differences to at most one scaled ULP with
identical p, E and decisions.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "src"))

import test_calibration_v2_progress as references  # noqa: E402

from selcal import calibration_v2 as kernel  # noqa: E402
from selcal import resolve_plan_v2  # noqa: E402
from selcal import result_wire as wire  # noqa: E402

OLD = ROOT / "tests/fixtures/recovery_kernel_v1"
NEW = ROOT / "tests/fixtures/recovery_kernel_v2"
SOURCES = ("src/selcal/calibration_v2.py", "src/selcal/result_wire.py",
           "src/selcal/statistics/lagged_pearson.py")


def main() -> int:
    NEW.mkdir(exist_ok=False)
    manifest = json.loads((OLD / "manifest.json").read_text(encoding="utf-8"))
    manifest["scope"] = (
        "Kernel regression references after the ALG-01 Pearson numeric-method change; "
        "not independent scientific validation"
    )
    verified = json.loads((OLD / "verified_environment.json").read_text(encoding="utf-8"))
    verified["environment"] = references._runtime_reference_environment()
    verified["source_identity"] = [
        {"path": name, "sha256": hashlib.sha256((ROOT / name).read_bytes()).hexdigest()}
        for name in SOURCES
    ]
    for name, case in manifest["cases"].items():
        pair, request = references._case(name, case["request"]["selection_rule"])
        payload = wire.encode_calibration_result(
            kernel.calibrate_selected_family(pair, resolve_plan_v2(request)), max_bytes=1_000_000
        )
        with (NEW / case["file"]).open("xb") as handle:
            handle.write(payload)
        case.update(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
        row = next(row for row in verified["cases"] if row["case"] == name)
        row["sha256"] = case["sha256"]
    for name, document in (("manifest.json", manifest), ("verified_environment.json", verified)):
        with (NEW / name).open("x", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2)
            handle.write("\n")
    print(json.dumps({name: case["sha256"] for name, case in manifest["cases"].items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
