"""Build current SelCal bytes and smoke a wheel-only clean installation."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_MODULES = (
    "selcal/input_resources.py",
    "selcal/inputs.py",
    "selcal/npz_safe.py",
)
SUPERSEDED_WHEEL = REPOSITORY_ROOT / "dist/selcal-0.1.0.dev0-py3-none-any.whl"
SUPERSEDED_WHEEL_SHA256 = (
    "41ff1ea7c303008e9eebc1f6c64c9324e2eb8645b3fdc23174da27456f2d27f7"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _outside_repository(path: Path) -> bool:
    try:
        path.resolve().relative_to(REPOSITORY_ROOT)
    except ValueError:
        return True
    return False


def _run(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed


def _verify_wheel_record(wheel: Path) -> list[str]:
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        if not set(SOURCE_MODULES).issubset(names):
            raise RuntimeError("current wheel omits required input source modules")
        record_name = next(
            (name for name in names if name.endswith(".dist-info/RECORD")),
            None,
        )
        if record_name is None:
            raise RuntimeError("current wheel omits RECORD")
        rows = list(csv.reader(io.StringIO(archive.read(record_name).decode("utf-8"))))
        row_by_name = {row[0]: row for row in rows}
        for name in SOURCE_MODULES:
            row = row_by_name[name]
            algorithm, encoded = row[1].split("=", 1)
            if algorithm != "sha256":
                raise RuntimeError(f"unexpected RECORD algorithm for {name}")
            observed = hashlib.sha256(archive.read(name)).digest()
            expected = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
            if observed != expected or int(row[2]) != len(archive.read(name)):
                raise RuntimeError(f"RECORD mismatch for {name}")
    return list(SOURCE_MODULES)


def _smoke_code() -> str:
    return """
import importlib.metadata
import json
import pathlib
import platform
import sys
import tempfile
import zipfile
import numpy as np
import selcal
import selcal.npz_safe as npz_safe
from selcal.inputs import load_csv, load_npz

repository = pathlib.Path(sys.argv[1]).resolve()
package_file = pathlib.Path(selcal.__file__).resolve()
assert importlib.metadata.version('selcal') == '0.1.0.dev0'
assert not package_file.is_relative_to(repository)
assert all(pathlib.Path(item or '.').resolve() != repository for item in sys.path)
with tempfile.TemporaryDirectory(prefix='selcal-wheel-smoke-') as directory:
    root = pathlib.Path(directory)
    csv_path = root / 'input.csv'
    stored_path = root / 'stored.npz'
    deflated_path = root / 'deflated.npz'
    csv_path.write_bytes(b'source,target\\n1,10\\n2,20\\n3,30\\n')
    source = np.array([1.0, 2.0, 3.0])
    target = np.array([10.0, 20.0, 30.0])
    np.savez(stored_path, source=source, target=target)
    np.savez_compressed(deflated_path, source=source, target=target)
    loaded_csv = load_csv(
        csv_path,
        source_column='source',
        target_column='target',
        candidates=(1,),
    )
    assert loaded_csv.source_format == 'csv'
    assert load_npz(stored_path, candidates=(1,)).source_format == 'npz'
    original_zip_file = zipfile.ZipFile
    original_decompressobj = npz_safe.zlib.decompressobj
    constructions = []
    def forbidden_zip_file(*args, **kwargs):
        raise AssertionError('stdlib ZipFile forbidden in public NPZ load')
    def observe_decompressobj(*args, **kwargs):
        constructions.append((args, kwargs))
        return original_decompressobj(*args, **kwargs)
    zipfile.ZipFile = forbidden_zip_file
    npz_safe.zlib.decompressobj = observe_decompressobj
    try:
        assert load_npz(deflated_path, candidates=(1,)).source_format == 'npz'
    finally:
        zipfile.ZipFile = original_zip_file
        npz_safe.zlib.decompressobj = original_decompressobj
    assert len(constructions) == 2
print(json.dumps({
    'cwd': str(pathlib.Path.cwd().resolve()),
    'deflateDecompressorConstructions': len(constructions),
    'environment': {
        'architecture': platform.machine(),
        'numpy': np.__version__,
        'os': platform.system(),
        'osRelease': platform.release(),
        'python': platform.python_version(),
        'pythonImplementation': platform.python_implementation(),
    },
    'packageFile': str(package_file),
    'packageMetadata': {'name': 'selcal', 'version': '0.1.0.dev0'},
    'publicSmoke': {'csv': 'PASS', 'npzDeflatedPublicZlib': 'PASS', 'npzStored': 'PASS'},
    'repositoryPathOnSysPath': False,
    'sourceTreeExcluded': True,
}, sort_keys=True))
"""


def _build_and_smoke(output: Path) -> int:
    if not _outside_repository(output):
        raise ValueError("package-install output must be outside the repository")
    if not SUPERSEDED_WHEEL.is_file():
        raise RuntimeError("expected superseded dist wheel is absent")
    if _sha256_file(SUPERSEDED_WHEEL) != SUPERSEDED_WHEEL_SHA256:
        raise RuntimeError("superseded dist wheel bytes changed")
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv executable is required")
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    workspace: Path
    with tempfile.TemporaryDirectory(prefix="selcal-task4-package-") as directory:
        workspace = Path(directory)
        artifacts = workspace / "artifacts"
        artifacts.mkdir()
        build = _run(
            [sys.executable, "-m", "build", "--outdir", str(artifacts)],
            cwd=REPOSITORY_ROOT,
            environment=environment,
        )
        wheel = next(artifacts.glob("selcal-*.whl"))
        sdist = next(artifacts.glob("selcal-*.tar.gz"))
        source_modules = _verify_wheel_record(wheel)
        clean_venv = workspace / "clean-venv"
        _run(
            [uv, "venv", "--python", "3.11", str(clean_venv)],
            cwd=workspace,
            environment=environment,
        )
        _run(
            [
                uv,
                "pip",
                "install",
                "--python",
                str(clean_venv / "bin/python"),
                str(wheel),
            ],
            cwd=workspace,
            environment=environment,
        )
        smoke = _run(
            [
                str(clean_venv / "bin/python"),
                "-c",
                _smoke_code(),
                str(REPOSITORY_ROOT),
            ],
            cwd=workspace,
            environment=environment,
        )
        clean = json.loads(smoke.stdout)
        clean["cwdWasTemporaryWorkspace"] = Path(clean["cwd"]) == workspace.resolve()
        clean["exitCode"] = 0
        clean["installedFrom"] = "CURRENT_WHEEL_ONLY_NOT_EDITABLE_NOT_SOURCE_TREE"
        clean["sourceModulePresence"] = source_modules
        clean["wheelRecordCheck"] = "PASS"
        artifact_rows = [
            {
                "bytes": artifact.stat().st_size,
                "name": artifact.name,
                "sha256": _sha256_file(artifact),
            }
            for artifact in (wheel, sdist)
        ]
        payload: dict[str, Any] = {
            "artifactDisposition": "DELETED_AFTER_TEMPORARY_VERIFICATION",
            "artifactHashEvidence": (
                "EXECUTION_TIME_OBSERVATION_NOT_INDEPENDENTLY_REINSPECTABLE"
            ),
            "build": {
                "artifacts": artifact_rows,
                "command": (
                    "<verification-python> -m build --outdir <temporary-build-dir>/artifacts"
                ),
                "environment": {
                    "architecture": platform.machine(),
                    "os": platform.system(),
                    "osRelease": platform.release(),
                    "python": platform.python_version(),
                    "pythonImplementation": platform.python_implementation(),
                },
                "exitCode": 0,
                "result": build.stdout.strip().splitlines()[-1],
            },
            "claimCeiling": (
                "LOCAL_PACKAGE_BUILD_AND_CLEAN_INSTALL_EVIDENCE_NOT_RELEASE_NOT_LICENCE_"
                "NOT_CROSS_PLATFORM_NOT_M6_NOT_SUBMISSION"
            ),
            "cleanInstall": clean,
            "exactArtifactsRetained": False,
            "harness": {
                "path": "scripts/verify_input_package_install.py",
                "sha256": _sha256_file(Path(__file__).resolve()),
            },
            "status": "PASS_LOCAL_CURRENT_WHEEL_ONLY",
            "supersededDistArtifact": {
                "path": "dist/selcal-0.1.0.dev0-py3-none-any.whl",
                "sha256": SUPERSEDED_WHEEL_SHA256,
                "status": "SUPERSEDED_NOT_CURRENT_NOT_USED",
            },
            "temporaryWorkspace": (
                "Created with tempfile.TemporaryDirectory; current artifacts and clean venv "
                "were deleted after execution-time hashes and smoke observations were recorded"
            ),
        }
    if workspace.exists():
        raise RuntimeError("temporary package workspace was not removed")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    return _build_and_smoke(arguments.output)


if __name__ == "__main__":
    raise SystemExit(main())
