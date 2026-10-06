from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from _documentation import public_documentation_text

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = REPOSITORY_ROOT / "scripts" / "benchmark_in_memory.py"
STANDARD_RECEIPT = (
    REPOSITORY_ROOT / "docs" / "benchmarks" / "in_memory_standard_20261006_r16.json"
)


def test_smoke_benchmark_exercises_public_api_and_emits_bounded_schema() -> None:
    completed = subprocess.run(
        [sys.executable, str(BENCHMARK), "--preset", "smoke"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stderr == ""
    payload = json.loads(completed.stdout)

    assert payload["schema"] == "selcal.benchmark.in-memory.v1"
    assert payload["preset"] == "smoke"
    assert len(payload["benchmark_contract_sha256"]) == 64
    assert payload["execution"] == {
        "gpu_used": False,
        "mode": "single_process",
        "workers": 1,
    }
    assert payload["measurement"] == {
        "memory": "tracemalloc_python_allocations",
        "scope": "resolve_calibrate_verify",
        "timer": "time.perf_counter_ns",
    }
    assert payload["source_identity"]["scheme"] == "installed_python_sources_v1"
    assert payload["source_identity"]["file_count"] >= 30
    assert len(payload["source_identity"]["sha256"]) == 64
    assert payload["harness_identity"]["scheme"] == "script_sha256_v1"
    assert len(payload["harness_identity"]["sha256"]) == 64
    assert set(payload["environment"]) == {
        "machine",
        "numpy",
        "platform",
        "python",
        "python_implementation",
        "selcal",
    }

    assert [case["case_id"] for case in payload["cases"]] == [
        "B0005-C02-N0032",
        "B0010-C02-N0032",
        "B0010-C04-N0032",
    ]
    for case in payload["cases"]:
        assert case["repeat_count"] == 2
        assert case["result_status_counts"] == {"complete": 2}
        assert len(case["scientific_plan_sha256"]) == 64
        assert len(case["wall_seconds"]) == 2
        assert all(sample > 0 for sample in case["wall_seconds"])
        assert case["wall_seconds_min"] <= case["wall_seconds_median"]
        assert case["wall_seconds_median"] <= case["wall_seconds_max"]
        assert case["peak_traced_bytes_max"] > 0


def test_readme_documents_benchmark_command_and_claim_boundary() -> None:
    readme = public_documentation_text(REPOSITORY_ROOT)

    assert "## Characterize in-memory performance" in readme
    assert (
        "python scripts/benchmark_in_memory.py --preset standard > benchmark.json"
        in readme
    )
    assert (
        "This is single-process characterization, not a cross-platform performance "
        "guarantee or M6 scientific evidence."
        in readme
    )


def test_recorded_standard_receipt_is_bound_to_current_source_harness_and_contract() -> None:
    assert STANDARD_RECEIPT.is_file(), "missing current measured standard receipt"
    receipt = json.loads(STANDARD_RECEIPT.read_text(encoding="utf-8"))
    smoke = subprocess.run(
        [sys.executable, str(BENCHMARK), "--preset", "smoke"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=60,
        check=True,
    )
    current = json.loads(smoke.stdout)

    assert receipt["schema"] == "selcal.benchmark.in-memory.v1"
    assert receipt["preset"] == "standard"
    assert receipt["benchmark_contract_sha256"] == (
        "c3786267f82c0313ecd433fef97e70ff845f620474a2a5e5af314d57da42c17f"
    )
    assert receipt["source_identity"] == current["source_identity"]
    assert receipt["harness_identity"] == current["harness_identity"]
    assert len(receipt["cases"]) == 7
    assert all(case["result_status_counts"] == {"complete": 3} for case in receipt["cases"])
