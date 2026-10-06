"""Portable checks of public input-quality commands and their documented limits."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
from _documentation import public_documentation_text

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PRESSURE_HARNESS = REPOSITORY_ROOT / "scripts/verify_input_resource_pressure.py"
EXPECTED_LIMITS = {
    "CSV_RAW_BYTES": 67_108_864,
    "CSV_RECORD_CHARACTERS": 1_024,
    "CSV_FIELD_CHARACTERS": 256,
    "CSV_DATA_ROWS": 1_000_000,
    "NPZ_RAW_BYTES": 33_554_432,
    "NPZ_CENTRAL_DIRECTORY_BYTES": 16_384,
    "NPY_HEADER_BYTES": 4_096,
    "NPY_ELEMENTS": 1_000_000,
    "NPZ_MEMBER_UNCOMPRESSED_BYTES": 8_004_108,
    "NPZ_TOTAL_UNCOMPRESSED_BYTES": 16_008_216,
}


def test_pressure_harness_runs_read_only_with_fresh_workers(tmp_path: Path) -> None:
    assert PRESSURE_HARNESS.is_file()
    spec = importlib.util.spec_from_file_location("pressure_harness", PRESSURE_HARNESS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.FORMATS == ("CSV", "NPZ")
    assert module.SIZES == (10_000, 100_000, 250_000, 1_000_000)
    assert module.DTYPE == "float64"
    assert module.NPZ_STORAGE == "NPZ_STORED"
    assert module.REPEAT_COUNT == 1
    assert module.RSS_SAMPLE_INTERVAL_SECONDS == 0.001

    output = tmp_path / "pressure.json"
    completed = subprocess.run(
        [
            "uv",
            "run",
            "--no-project",
            "--python",
            sys.executable,
            "--with",
            "numpy==2.4.6",
            "--with",
            "psutil==7.0.0",
            "python",
            "-B",
            str(PRESSURE_HARNESS),
            "--formats",
            "CSV",
            "NPZ",
            "--sizes",
            "10",
            "--output",
            str(output),
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        errors="replace",
    )
    assert completed.returncode == 0, completed.stderr
    observed = json.loads(output.read_text("utf-8"))
    method = observed["method"]
    assert "command" not in method
    assert "<temporary-receipt-path>" in method["example_command"]
    invocation = method["invocation"]
    assert invocation["scope"] == "python_process_not_parent_launcher"
    assert invocation["argv"][1:] == [
        "-B",
        str(PRESSURE_HARNESS),
        "--formats",
        "CSV",
        "NPZ",
        "--sizes",
        "10",
        "--output",
        str(output),
    ]
    assert invocation["cwd"] == str(REPOSITORY_ROOT)
    assert Path(invocation["executable"]).is_absolute()
    # uv may remove its temporary interpreter as soon as the child exits.
    # Its identity must be observed inside that process, not checked afterward.
    assert invocation["executable_exists_during_run"] is True
    executable_sha = invocation["executable_sha256"]
    assert len(executable_sha) == 64
    assert all(character in "0123456789abcdef" for character in executable_sha)
    assert method["harnessSha256"] == hashlib.sha256(PRESSURE_HARNESS.read_bytes()).hexdigest()
    assert observed["execution"]["freshSubprocessPerCase"] is True
    assert observed["execution"]["parallel"] is False
    assert observed["execution"]["temporaryInputsRetained"] is False
    assert [(case["format"], case["rowsOrElements"]) for case in observed["cases"]] == [
        ("CSV", 10),
        ("NPZ", 10),
    ]
    assert all(len(case["inputSha256"]) == 64 for case in observed["cases"])


def test_readme_documents_the_input_contract_without_claim_inflation() -> None:
    readme = public_documentation_text(REPOSITORY_ROOT)
    normalized = " ".join(readme.split())
    assert "## Load bounded CSV and NPZ inputs" in readme
    assert "strict UTF-8 two-field CSV" in readme
    assert "quoted fields cannot contain CR or LF" in readme
    assert "`source`/`source.npy` and `target`/`target.npy`" in readme
    for reason, limit in EXPECTED_LIMITS.items():
        assert f"`{reason}` | {limit:,}" in readme
    assert "INPUT_RESOURCE_LIMIT_EXCEEDED_V1 reason=<CODE> limit=<INTEGER>" in readme
    assert (
        "These input-loading limits are independent of and not interchangeable with the "
        "`B`, `C`, `N`, and `S` in-memory execution budget." in normalized
    )
    assert (
        "This is not streaming or out-of-core support, a security certification, M6 "
        "scientific evidence, release readiness, or SoftwareX submission readiness." in normalized
    )


@pytest.mark.parametrize("instrumentation", ["trace", "profile", "tracemalloc"])
def test_memory_worker_rejects_active_instrumentation(instrumentation: str) -> None:
    worker = """
import runpy
import sys
import tracemalloc

namespace = runpy.run_path(sys.argv[1])
if sys.argv[2] == "trace":
    def trace(frame, event, arg):
        return trace
    sys.settrace(trace)
elif sys.argv[2] == "profile":
    sys.setprofile(lambda frame, event, arg: None)
else:
    tracemalloc.start()
namespace["_emit_isolated_verifier_memory_sample"](64, 8)
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            worker,
            str(REPOSITORY_ROOT / "tests/test_verifier_integration_v2.py"),
            instrumentation,
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
        timeout=30,
    )
    assert completed.returncode != 0
    assert "RuntimeError: measurement environment is dirty" in completed.stderr
    assert "peak_bytes" not in completed.stdout


def test_resource_and_coverage_selections_partition_verifier_tests() -> None:
    def collect(selection: str | None) -> set[str]:
        command = [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
            "tests/test_verifier_integration_v2.py",
        ]
        if selection is not None:
            command.extend(["-k", selection])
        completed = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            errors="replace",
            check=False,
            timeout=30,
        )
        assert completed.returncode == 0, completed.stderr
        return {
            line
            for line in completed.stdout.splitlines()
            if line.startswith("tests/test_verifier_integration_v2.py::")
        }

    all_tests = collect(None)
    resource_tests = collect("test_public_verifier_scratch_peak")
    coverage_tests = collect("not test_public_verifier_scratch_peak")
    assert resource_tests == {
        "tests/test_verifier_integration_v2.py::"
        "test_public_verifier_scratch_peak_is_b_bounded_with_fixed_c",
        "tests/test_verifier_integration_v2.py::"
        "test_public_verifier_scratch_peak_has_bounded_c_scaling_with_fixed_b",
    }
    assert coverage_tests
    assert not resource_tests & coverage_tests
    assert resource_tests | coverage_tests == all_tests


def test_readme_separates_memory_measurement_from_coverage() -> None:
    readme = public_documentation_text(REPOSITORY_ROOT)
    assert (
        "python -m pytest tests/test_verifier_integration_v2.py "
        "-k test_public_verifier_scratch_peak" in readme
    )
    assert (
        "python -m pytest tests/test_verifier_integration_v2.py "
        "--cov=src -k 'not test_public_verifier_scratch_peak'" in readme
    )
    assert "uv run --locked --all-extras python -m pytest\n" in readme
    assert "measurement environment is dirty" in readme
    assert "Python-tracked allocations, not process RSS or native allocations" in readme
