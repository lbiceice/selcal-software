"""Characterize SelCal's current single-process in-memory public API."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import statistics
import time
import tracemalloc
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import numpy as np

import selcal
from selcal import (
    PlanRequestV2,
    calibrate_selected_family,
    resolve_plan_v2,
    verify_calibration_result,
)
from selcal.contracts import SeriesPair

SCHEMA: Final = "selcal.benchmark.in-memory.v1"
EXECUTION: Final = {
    "gpu_used": False,
    "mode": "single_process",
    "workers": 1,
}
MEASUREMENT: Final = {
    "memory": "tracemalloc_python_allocations",
    "scope": "resolve_calibrate_verify",
    "timer": "time.perf_counter_ns",
}


@dataclass(frozen=True)
class BenchmarkCase:
    """One bounded benchmark configuration."""

    planned_replicates: int
    candidate_count: int
    series_length: int
    repeat_count: int

    @property
    def case_id(self) -> str:
        return (
            f"B{self.planned_replicates:04d}-"
            f"C{self.candidate_count:02d}-"
            f"N{self.series_length:04d}"
        )


SMOKE_CASES: Final = (
    BenchmarkCase(5, 2, 32, 2),
    BenchmarkCase(10, 2, 32, 2),
    BenchmarkCase(10, 4, 32, 2),
)
STANDARD_CASES: Final = (
    BenchmarkCase(25, 4, 128, 3),
    BenchmarkCase(100, 4, 128, 3),
    BenchmarkCase(250, 4, 128, 3),
    BenchmarkCase(100, 2, 128, 3),
    BenchmarkCase(100, 8, 128, 3),
    BenchmarkCase(100, 4, 64, 3),
    BenchmarkCase(100, 4, 256, 3),
)


def _series_pair(length: int) -> SeriesPair:
    index = np.arange(length, dtype=np.float64)
    source = (
        np.sin(index / 3.0)
        + 0.2 * np.cos(index / 7.0)
        + 0.01 * np.remainder(index, 5.0)
    )
    target = np.roll(source, 1) + 0.05 * np.sin(index / 5.0)
    return SeriesPair(source=source, target=target)


def _request(case: BenchmarkCase) -> PlanRequestV2:
    return PlanRequestV2(
        candidates=tuple(range(1, case.candidate_count + 1)),
        statistic_name="lagged_pearson_v1",
        statistic_params={},
        selection_rule="max_absolute",
        null_name="circular_shift_v2",
        null_params={"min_shift": 1},
        replicates=case.planned_replicates,
        alpha=0.05,
        tie_tolerance=1e-12,
        root_seed=20260831,
    )


def _execute(pair: SeriesPair, request: PlanRequestV2) -> tuple[str, str]:
    resolution = resolve_plan_v2(request)
    result = calibrate_selected_family(pair, resolution)
    verify_calibration_result(result, resolution)
    return result.status.value, result.scientific_plan_sha256


def _measure_once(
    pair: SeriesPair,
    request: PlanRequestV2,
) -> tuple[float, int, str, str]:
    gc.collect()
    tracemalloc.start()
    started_ns = time.perf_counter_ns()
    try:
        status, plan_sha256 = _execute(pair, request)
        elapsed_ns = time.perf_counter_ns() - started_ns
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return elapsed_ns / 1_000_000_000, peak_bytes, status, plan_sha256


def _benchmark_case(case: BenchmarkCase) -> dict[str, object]:
    pair = _series_pair(case.series_length)
    request = _request(case)
    _execute(pair, request)

    wall_seconds: list[float] = []
    peak_bytes: list[int] = []
    statuses: Counter[str] = Counter()
    plan_hashes: set[str] = set()
    for _ in range(case.repeat_count):
        elapsed, peak, status, plan_sha256 = _measure_once(pair, request)
        wall_seconds.append(elapsed)
        peak_bytes.append(peak)
        statuses[status] += 1
        plan_hashes.add(plan_sha256)

    if len(plan_hashes) != 1:
        raise RuntimeError("benchmark repeats resolved different scientific plans")
    return {
        "case_id": case.case_id,
        "planned_replicates": case.planned_replicates,
        "candidate_count": case.candidate_count,
        "series_length": case.series_length,
        "repeat_count": case.repeat_count,
        "scientific_plan_sha256": next(iter(plan_hashes)),
        "result_status_counts": dict(sorted(statuses.items())),
        "wall_seconds": wall_seconds,
        "wall_seconds_min": min(wall_seconds),
        "wall_seconds_median": statistics.median(wall_seconds),
        "wall_seconds_max": max(wall_seconds),
        "peak_traced_bytes_max": max(peak_bytes),
    }


def _contract_sha256(preset: str, cases: tuple[BenchmarkCase, ...]) -> str:
    contract = {
        "schema": SCHEMA,
        "preset": preset,
        "execution": EXECUTION,
        "measurement": MEASUREMENT,
        "cases": [asdict(case) for case in cases],
    }
    canonical = json.dumps(contract, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _installed_source_identity() -> dict[str, object]:
    package_file = selcal.__file__
    if package_file is None:
        raise RuntimeError("installed SelCal package has no source location")
    package_root = Path(package_file).resolve().parent
    source_files = tuple(sorted(package_root.rglob("*.py")))
    digest = hashlib.sha256()
    for source_file in source_files:
        relative = source_file.relative_to(package_root).as_posix().encode("utf-8")
        content = source_file.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return {
        "scheme": "installed_python_sources_v1",
        "file_count": len(source_files),
        "sha256": digest.hexdigest(),
    }


def _harness_identity() -> dict[str, str]:
    return {
        "scheme": "script_sha256_v1",
        "sha256": hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest(),
    }


def run_benchmark(preset: str) -> dict[str, object]:
    """Run a named benchmark preset and return a JSON-compatible result."""

    cases = SMOKE_CASES if preset == "smoke" else STANDARD_CASES
    return {
        "schema": SCHEMA,
        "preset": preset,
        "benchmark_contract_sha256": _contract_sha256(preset, cases),
        "execution": EXECUTION,
        "measurement": MEASUREMENT,
        "source_identity": _installed_source_identity(),
        "harness_identity": _harness_identity(),
        "environment": {
            "machine": platform.machine(),
            "numpy": np.__version__,
            "platform": platform.system(),
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "selcal": selcal.__version__,
        },
        "cases": [_benchmark_case(case) for case in cases],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Characterize SelCal's single-process in-memory public API."
    )
    parser.add_argument(
        "--preset",
        choices=("smoke", "standard"),
        default="standard",
        help="bounded case matrix to execute (default: standard)",
    )
    arguments = parser.parse_args(argv)
    print(json.dumps(run_benchmark(arguments.preset), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
