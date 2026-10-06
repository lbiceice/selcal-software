"""Save a complete synthetic result, then read it in another process.

This is a result-content example, not a checkpoint/resume implementation or a
scientific experiment. The read operation never performs a calibration replay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from selcal import (
    PlanRequestV2,
    calibrate_selected_family,
    resolve_plan_v2,
    verify_calibration_result,
)
from selcal.contracts import SeriesPair
from selcal.contracts_v2 import CalibrationResult
from selcal.result_wire import (
    ResultWireError,
    decode_calibration_result,
    encode_calibration_result,
)


def calculate_example() -> CalibrationResult:
    """Run the same small synthetic case as the existing basic example."""
    pair = SeriesPair(
        source=np.asarray((0.0, 3.0, 1.0, 2.0, 4.0, 5.0), dtype=np.float64),
        target=np.asarray((0.0, 1.0, 3.0, 2.0, 5.0, 4.0), dtype=np.float64),
    )
    resolution = resolve_plan_v2(
        PlanRequestV2(
            candidates=(1, 2, 3),
            statistic_name="lagged_pearson_v1",
            statistic_params={},
            selection_rule="max_absolute",
            null_name="circular_shift_v2",
            null_params={"min_shift": 1},
            replicates=9,
            alpha=0.05,
            tie_tolerance=1e-12,
            root_seed=17,
        )
    )
    result = calibrate_selected_family(pair, resolution)
    verify_calibration_result(result, resolution)
    return result


def positive_limit(value: str) -> int:
    """Accept an explicit positive per-record byte budget."""
    try:
        limit = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("max-bytes must be a positive integer") from error
    if limit <= 0:
        raise argparse.ArgumentTypeError("max-bytes must be a positive integer")
    return limit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("save", "read"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--max-bytes", required=True, type=positive_limit)
    args = parser.parse_args()
    try:
        if args.action == "save":
            result = calculate_example()
            data = encode_calibration_result(result, max_bytes=args.max_bytes)
            # Admit and encode before touching the output; never overwrite user data.
            with args.path.open("xb") as stream:
                stream.write(data)
        else:
            with args.path.open("rb") as stream:
                # Read one extra byte to distinguish an exact-size record from truncation.
                # Chunk size is an I/O choice, not a record or memory-budget guarantee.
                remaining = args.max_bytes + 1
                chunks = []
                while remaining:
                    chunk = stream.read(min(65_536, remaining))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    remaining -= len(chunk)
                data = b"".join(chunks)
            result = decode_calibration_result(data, max_bytes=args.max_bytes)
    except ResultWireError as error:
        print(f"result wire rejected: {error.code}", file=sys.stderr)
        return 2
    except OSError:
        print("file operation failed; no successful result is reported", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "action": args.action,
                "status": result.status.value,
                "planned_replicates": result.planned_replicates,
                "retained_replicates": len(result.replicates),
                "failure_count": result.failure_count,
                "p_value": result.p_value,
                "byte_count": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "verification_scope": "content_only_not_replay",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
