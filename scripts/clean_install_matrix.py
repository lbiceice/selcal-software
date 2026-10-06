"""First-time-user acceptance: build, install from the artifact only, then use SelCal.

Usage: python scripts/clean_install_matrix.py OUTPUT_DIR [--pythons 3.11 3.12 3.13]

For each Python version it builds nothing itself beyond one wheel and one sdist, creates a fresh
virtual environment, installs SelCal from the built artifact (runtime dependencies only, no
development extras, no editable source), and then runs the acceptance check from a folder outside
the repository, so nothing can be picked up from the checkout. The wheel is used for every
version and the sdist for the last one, which also proves the source distribution installs.

Each run writes its complete output, the built artefacts' digests and a JSON receipt. The script
exits 0 only if every environment passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _run(argv: list[str], *, cwd: Path, log: Path) -> int:
    done = subprocess.run(argv, cwd=cwd, capture_output=True, text=True,
        errors="replace", check=False)
    log.write_text(
        f"$ {' '.join(argv)}\n(cwd {cwd})\n\n{done.stdout}\n{done.stderr}\n",
        encoding="utf-8",
        newline="\n",
    )
    return done.returncode


def _build(out: Path) -> dict[str, Any]:
    """Build one wheel and one sdist from a pristine copy of the tracked sources."""
    dist = out / "dist"
    with tempfile.TemporaryDirectory(prefix="selcal-build-") as folder:
        source = Path(folder) / "source"
        listing = subprocess.run(
            ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True
        ).stdout.decode()
        for name in filter(None, listing.split("\0")):
            target = source / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
        code = _run(
            [sys.executable, "-m", "build", "--outdir", str(dist)],
            cwd=source,
            log=out / "build.log",
        )
    artefacts = sorted(dist.glob("*")) if dist.is_dir() else []
    return {
        "exit_code": code,
        "artefacts": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in artefacts
        },
        "wheel": next((p for p in artefacts if p.suffix == ".whl"), None),
        "sdist": next((p for p in artefacts if p.name.endswith(".tar.gz")), None),
    }


def _environment(out: Path, version: str, artefact: Path, kind: str) -> dict[str, Any]:
    """Install the artefact into a fresh environment and use SelCal from outside the repository."""
    folder = out / f"py{version}-{kind}"
    folder.mkdir(parents=True)
    venv = folder / "venv"
    python = (
        venv
        / ("Scripts" if os.name == "nt" else "bin")
        / ("python.exe" if os.name == "nt" else "python")
    )
    steps: dict[str, int] = {}
    steps["create venv"] = _run(
        ["uv", "venv", "--python", version, str(venv)], cwd=folder, log=folder / "venv.log"
    )
    steps["install artefact"] = _run(
        ["uv", "pip", "install", "--python", str(python), str(artefact)],
        cwd=folder,
        log=folder / "install.log",
    )
    steps["environment"] = _run(
        [
            str(python),
            "-c",
            "import sys, platform, numpy, selcal; "
            "print('python', sys.version.split()[0]); "
            "print('platform', platform.platform(), platform.machine()); "
            "print('numpy', numpy.__version__); print('selcal', selcal.__version__); "
            "print('imported from', selcal.__file__)",
        ],
        cwd=folder,
        log=folder / "environment.log",
    )
    # The acceptance check runs from a folder outside the repository; only the script and the
    # example inputs are copied in, so nothing else can be picked up from the checkout.
    workspace = folder / "workspace"
    (workspace / "scripts").mkdir(parents=True)
    (workspace / "examples" / "workflow").mkdir(parents=True)
    shutil.copy2(ROOT / "scripts" / "acceptance_check.py", workspace / "scripts")
    for name in ("series.csv", "pearson.json"):
        shutil.copy2(ROOT / "examples" / "workflow" / name, workspace / "examples" / "workflow")
    steps["acceptance"] = _run(
        [
            str(python),
            "scripts/acceptance_check.py",
            "--precision",
            "--installed",
            "--out",
            str(folder / "acceptance"),
        ],
        cwd=workspace,
        log=folder / "acceptance.log",
    )
    summary = sorted((folder / "acceptance").glob("*/summary.txt"))
    verdict = summary[0].read_text(encoding="utf-8").strip().splitlines()[-1] if summary else ""
    return {
        "python": version,
        "artefact": artefact.name,
        "kind": kind,
        "steps": steps,
        "verdict": verdict,
        "passed": all(code == 0 for code in steps.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("--pythons", nargs="+", default=["3.11", "3.12", "3.13"])
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    print(f"building artefacts in {out}", flush=True)
    build = _build(out)
    if build["exit_code"] != 0 or not build["wheel"] or not build["sdist"]:
        print("BUILD FAILED; see build.log", flush=True)
        return 1
    for name, digest in build["artefacts"].items():
        print(f"  {name}  sha256 {digest[:16]}...", flush=True)
    results = []
    for version in args.pythons:
        print(f"wheel, Python {version} ...", flush=True)
        results.append(_environment(out, version, build["wheel"], "wheel"))
        print(f"  {results[-1]['verdict']}", flush=True)
    print(f"sdist, Python {args.pythons[-1]} ...", flush=True)
    results.append(_environment(out, args.pythons[-1], build["sdist"], "sdist"))
    print(f"  {results[-1]['verdict']}", flush=True)
    receipt = {
        "schema": "selcal.clean-install-matrix.v1",
        "host": {"platform": platform.platform(), "machine": platform.machine()},
        "artefacts": build["artefacts"],
        "environments": results,
        "all_passed": all(item["passed"] for item in results),
    }
    (out / "receipt.json").write_text(
        json.dumps(receipt, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    print("ALL PASSED" if receipt["all_passed"] else "FAILED", flush=True)
    return 0 if receipt["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
