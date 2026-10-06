"""Replay the bounded-input pressure observation without writing repository files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import threading
import time
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np

FORMATS = ("CSV", "NPZ")
SIZES = (10_000, 100_000, 250_000, 1_000_000)
DTYPE = "float64"
NPZ_STORAGE = "NPZ_STORED"
REPEAT_COUNT = 1
RSS_SAMPLE_INTERVAL_SECONDS = 0.001
INPUT_GENERATION_CONTRACT = (
    "For N rows, source=float64 arange(0,N), target=2*source; CSV is strict UTF-8 "
    "with header source,target and base-10 integer rows; NPZ_STORED is numpy.savez with "
    "exact source and target keys. Each generated file is SHA-256 hashed before one public "
    "load in its own fresh worker process."
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATHS = (
    "src/selcal/input_resources.py",
    "src/selcal/inputs.py",
    "src/selcal/npz_safe.py",
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _generate_input(input_format: str, size: int, path: Path) -> None:
    source = np.arange(size, dtype=np.float64)
    target = source * 2.0
    if input_format == "CSV":
        with path.open("w", encoding="utf-8", newline="") as stream:
            stream.write("source,target\n")
            for index in range(size):
                stream.write(f"{index},{index * 2}\n")
        return
    if input_format == "NPZ":
        np.savez(path, source=source, target=target)
        return
    raise ValueError(f"unsupported pressure input format: {input_format}")


def _measure_case(input_format: str, size: int, workspace: Path) -> dict[str, Any]:
    import psutil

    from selcal.inputs import load_csv, load_npz

    suffix = ".csv" if input_format == "CSV" else ".npz"
    path = workspace / f"{input_format.lower()}-{size}{suffix}"
    _generate_input(input_format, size, path)
    input_sha256 = _sha256_file(path)
    raw_bytes = path.stat().st_size
    process = psutil.Process()
    rss_baseline = process.memory_info().rss
    rss_samples = [rss_baseline]
    stop_sampling = threading.Event()

    def sample_rss() -> None:
        while not stop_sampling.wait(RSS_SAMPLE_INTERVAL_SECONDS):
            rss_samples.append(process.memory_info().rss)

    sampler = threading.Thread(target=sample_rss, name="selcal-rss-sampler")
    sampler.start()
    tracemalloc.start()
    started = time.perf_counter()
    try:
        if input_format == "CSV":
            loaded = load_csv(
                path,
                source_column="source",
                target_column="target",
                candidates=(1,),
            )
        else:
            loaded = load_npz(path, candidates=(1,))
        wall_seconds = time.perf_counter() - started
        _current, tracemalloc_peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
        stop_sampling.set()
        sampler.join()
    rss_samples.append(process.memory_info().rss)
    assert loaded.pair.source.dtype == np.dtype(DTYPE)
    assert loaded.pair.target.dtype == np.dtype(DTYPE)
    assert loaded.pair.source.size == loaded.pair.target.size == size
    rss_peak = max(rss_samples)
    contract_sha256 = _sha256_bytes(INPUT_GENERATION_CONTRACT.encode("utf-8"))
    return {
        "arrayPayloadBytes": loaded.pair.source.nbytes + loaded.pair.target.nbytes,
        "dtype": DTYPE,
        "format": input_format,
        "generationContractSha256": contract_sha256,
        "inputSha256": input_sha256,
        "rawBytes": raw_bytes,
        "repeatIndex": 1,
        "rowsOrElements": size,
        "rssBaselineBytes": rss_baseline,
        "rssDeltaBytes": rss_peak - rss_baseline,
        "rssPeakBytes": rss_peak,
        "status": "PASS",
        "tracemallocPeakBytes": tracemalloc_peak,
        "wallSeconds": wall_seconds,
    }


def _worker(input_format: str, size: int, workspace: Path) -> int:
    print(json.dumps(_measure_case(input_format, size, workspace), sort_keys=True))
    return 0


def _outside_repository(path: Path) -> bool:
    resolved = path.resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT)
    except ValueError:
        return True
    return False


def _run_all(formats: tuple[str, ...], sizes: tuple[int, ...], output: Path) -> int:
    import psutil

    if not _outside_repository(output):
        raise ValueError("pressure output must be outside the repository")
    harness_sha256 = _sha256_file(Path(__file__).resolve())
    contract_sha256 = _sha256_bytes(INPUT_GENERATION_CONTRACT.encode("utf-8"))
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPOSITORY_ROOT / "src")
    cases: list[dict[str, Any]] = []
    temporary_path: Path
    with tempfile.TemporaryDirectory(prefix="selcal-input-pressure-") as directory:
        temporary_path = Path(directory)
        for input_format in formats:
            for size in sizes:
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--worker",
                        input_format,
                        str(size),
                        str(temporary_path),
                    ],
                    cwd=REPOSITORY_ROOT,
                    env=environment,
                    check=False,
                    capture_output=True,
                    text=True,
                    errors="replace",
                )
                if completed.returncode != 0:
                    raise RuntimeError(
                        f"pressure worker failed for {input_format}/{size}: "
                        f"{completed.stderr.strip()}"
                    )
                cases.append(json.loads(completed.stdout))
    if temporary_path.exists():
        raise RuntimeError("temporary pressure workspace was not removed")
    payload = {
        "cases": cases,
        "claimCeiling": (
            "LOCAL_DESIGN_PRESSURE_OBSERVATION_ONLY_NOT_PERFORMANCE_GUARANTEE_"
            "NOT_STREAMING_NOT_SECURITY_CERTIFICATION_NOT_CROSS_PLATFORM_NOT_M6"
        ),
        "environment": {
            "architecture": platform.machine(),
            "numpy": np.__version__,
            "os": platform.system(),
            "osRelease": platform.release(),
            "psutil": psutil.__version__,
            "python": platform.python_version(),
            "pythonImplementation": platform.python_implementation(),
        },
        "execution": {
            "freshSubprocessPerCase": True,
            "parallel": False,
            "temporaryInputsRetained": False,
        },
        "method": {
            "arrayPayloadDefinition": (
                "source.nbytes plus target.nbytes after the public loader returns"
            ),
            "dtype": DTYPE,
            "example_command": (
                "PYTHONPATH=src uv run --no-project --python 3.11 --with numpy==2.4.6 "
                "--with psutil==7.0.0 python scripts/verify_input_resource_pressure.py "
                "--output <temporary-receipt-path>"
            ),
            "invocation": {
                "executable": sys.executable,
                # uv may delete this interpreter when its process exits. Capture
                # evidence now without substituting the parent launcher's path.
                "executable_exists_during_run": Path(sys.executable).is_file(),
                "executable_sha256": _sha256_file(Path(sys.executable)),
                "argv": list(sys.orig_argv),
                "cwd": str(Path.cwd()),
                "scope": "python_process_not_parent_launcher",
            },
            "formats": list(formats),
            "harnessPath": "scripts/verify_input_resource_pressure.py",
            "harnessSha256": harness_sha256,
            "inputGenerationContract": INPUT_GENERATION_CONTRACT,
            "inputGenerationContractSha256": contract_sha256,
            "npzStorage": NPZ_STORAGE,
            "repeatCount": REPEAT_COUNT,
            "rssSampleIntervalSeconds": RSS_SAMPLE_INTERVAL_SECONDS,
            "rssSamplingLimitation": (
                "Loads shorter than the 0.001-second sampling interval may miss a transient "
                "RSS peak; sampled RSS is observational and not a performance guarantee."
            ),
            "sizes": list(sizes),
            "temporaryWorkspace": (
                "Created with tempfile.TemporaryDirectory outside the repository and "
                "removed before the receipt was written"
            ),
            "tracemalloc": "tracemalloc peak around the public load only",
            "wallTimer": "time.perf_counter around the public load only",
        },
        "sourceIdentity": [
            {"path": relative, "sha256": _sha256_file(REPOSITORY_ROOT / relative)}
            for relative in SOURCE_PATHS
        ],
        "status": "LOCAL_DESIGN_PRESSURE_OBSERVATION_ONLY",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formats", nargs="+", choices=FORMATS, default=list(FORMATS))
    parser.add_argument("--sizes", nargs="+", type=int, default=list(SIZES))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", nargs=3, metavar=("FORMAT", "SIZE", "WORKSPACE"))
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    if arguments.worker is not None:
        input_format, size, workspace = arguments.worker
        return _worker(input_format, int(size), Path(workspace))
    if arguments.output is None:
        raise ValueError("--output is required")
    if any(size <= 0 for size in arguments.sizes):
        raise ValueError("pressure sizes must be positive integers")
    return _run_all(tuple(arguments.formats), tuple(arguments.sizes), arguments.output)


if __name__ == "__main__":
    raise SystemExit(main())
