"""Build and test the actual delivered artifacts; requires Python 3.11-3.13 and build."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import time
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path, PurePosixPath


def digest(data: bytes) -> dict:
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def inventory(package: Path) -> dict:
    if package.is_symlink():
        raise ValueError(f"product link refused: {package}")
    result = {}
    for path in sorted(package.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"product link refused: {path}")
        if (path.is_file() and "__pycache__" not in path.parts
                and path.suffix not in (".pyc", ".pyo")):
            result[path.relative_to(package).as_posix()] = digest(path.read_bytes())
    return result


def require_inventory(expected: dict, actual: dict, label: str) -> None:
    if actual != expected:
        changed = sorted(k for k in actual.keys() & expected.keys() if actual[k] != expected[k])
        raise ValueError(f"{label} inventory mismatch: "
                         f"missing={sorted(expected.keys()-actual.keys())}, "
                         f"extra={sorted(actual.keys()-expected.keys())}, changed={changed}")


def require_generated_sdist_metadata(before: bytes, after: bytes) -> None:
    """Allow setuptools' one setup.cfg entry; preserve every other byte and line order.

    SOURCES.txt is generated metadata, not a globally sorted file. This exception
    does not permit product/source changes, duplicates, deletion, or reordering.
    Callers must still check all original files and product inventories.
    """
    if before == after:
        return
    old = before.splitlines(keepends=True)
    new = after.splitlines(keepends=True)
    inserted = [line for line in new if line.rstrip(b"\r\n") == b"setup.cfg"]
    if (any(line.rstrip(b"\r\n") == b"setup.cfg" for line in old)
            or len(inserted) != 1
            or b"".join(line for line in new if line.rstrip(b"\r\n") != b"setup.cfg") != before):
        raise ValueError("generated metadata changed beyond one setup.cfg insertion")


def mypy_applies(numpy_version: str) -> bool:
    """Static typing is checked only against NumPy 2 stubs (the locked runtime is 2.4.6)."""
    return int(numpy_version.split(".", 1)[0]) >= 2


def archive_inventory(path: Path) -> dict:
    result = {}
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            members = []
            for member in archive.infolist():
                if stat.S_ISLNK(member.external_attr >> 16):
                    raise ValueError("archive link refused")
                if not member.is_dir():
                    members.append((member.filename, archive.read(member)))
    else:
        with tarfile.open(path) as archive:
            members = []
            for member in archive:
                if not member.isdir() and not member.isfile():
                    raise ValueError("archive link or special member refused")
                if member.isfile():
                    members.append((member.name, archive.extractfile(member).read()))
    seen = set()
    for name, data in members:
        parts = PurePosixPath(name).parts
        if name in seen or name.startswith("/") or ".." in parts or "\\" in name:
            raise ValueError("unsafe or duplicate archive member")
        seen.add(name)
        product = parts[1:] if path.suffix == ".whl" and parts[0] == "selcal" else ()
        if path.suffix != ".whl" and parts[1:3] == ("src", "selcal"):
            product = parts[3:]
        if product:
            key = "/".join(product)
            if key in result:
                raise ValueError("duplicate product inventory member")
            result[key] = digest(data)
    return result


def clean_environment(executable: str) -> dict:
    environment = {key: value for key, value in os.environ.items()
                   if "PYTHON" not in key.upper() and key.upper() not in {
                       "__PYVENV_LAUNCHER__", "VIRTUAL_ENV", "PYTEST_ADDOPTS", "PYTEST_PLUGINS",
                       "COVERAGE_PROCESS_START", "COVERAGE_RCFILE"}}
    # Keep the selected venv's tools ahead of similarly named host executables.
    environment["PATH"] = str(Path(executable).absolute().parent) + os.pathsep + environment.get(
        "PATH", "")
    return environment


def require_installation(info: dict, environment: Path, numpy_version: str) -> None:
    environment = environment.resolve()
    package, library = Path(info["package"]).resolve(), Path(info["purelib"]).resolve()
    if (Path(info["prefix"]).resolve() != environment or not library.is_relative_to(environment)
            or package != library / "selcal" or info["numpy"] != numpy_version):
        raise ValueError(f"installed origin or NumPy mismatch: {info}")


class InvalidTests(ValueError):
    def __init__(self, counts: dict):
        self.counts = counts
        super().__init__(f"shipped tests invalid: {counts}")


def read_junit(path: Path) -> dict:
    cases = list(ET.parse(path).getroot().iter("testcase"))
    failures = sum(case.find("failure") is not None for case in cases)
    errors = sum(case.find("error") is not None for case in cases)
    skips = [{"name": case.get("classname", "") + "." + case.get("name", ""),
              "reason": case.find("skipped").get("message", "")}
             for case in cases if case.find("skipped") is not None]
    counts = {"collected": len(cases), "passed": len(cases)-len(skips)-failures-errors,
              "skipped": len(skips), "failures": failures, "errors": errors, "skip_details": skips}
    if not cases or failures or errors or len(skips) == len(cases):
        raise InvalidTests(counts)
    return counts


IDENTITY = (
    "import importlib.metadata,json,pathlib,platform,sys,sysconfig,selcal,numpy; "
    "print(json.dumps({'package':str(pathlib.Path(selcal.__file__).resolve().parent),"
    "'prefix':sys.prefix,'purelib':sysconfig.get_path('purelib'),'python':sys.version,"
    "'executable':sys.executable,'platform':platform.platform(),'numpy':numpy.__version__,"
    "'selcal_version':importlib.metadata.version('selcal')}))"
)


class Run:
    def __init__(self, root: Path, output: Path):
        self.root = root.resolve()
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"output already exists: {output}")
        self.out = output.resolve()
        if self.out.is_relative_to(self.root):
            raise ValueError("output must be outside the source tree")
        self.out.mkdir(parents=True)
        for name in ("evidence", "artifacts", "work"):
            (self.out / name).mkdir()
        self.evidence = self.out / "evidence"
        self.receipt = {"schema": "selcal.ci-delivery.v1", "status": "FAIL",
                        "host": platform.platform(), "python": sys.version,
                        "executable": sys.executable, "source": str(self.root),
                        "scope": "ordinary artifact tests/installations; no hosted execution claim",
                        "steps": []}
        self.save()

    def save(self) -> None:
        (self.evidence / "receipt.json").write_text(
            json.dumps(self.receipt, indent=2) + "\n", encoding="utf-8")

    def command(self, name: str, argv: list, cwd: Path, timeout: float = 1800) -> str:
        stem = f"{len(self.receipt['steps']):02d}-{name}"
        step = {"name": name, "argv": list(map(str, argv)), "cwd": str(cwd),
                "stdout": stem + ".stdout", "stderr": stem + ".stderr",
                "status": "FAIL", "exit_code": None, "timeout_seconds": timeout}
        started = time.monotonic()
        try:
            with (self.evidence / step["stdout"]).open("wb") as stdout, \
                    (self.evidence / step["stderr"]).open("wb") as stderr:
                with subprocess.Popen(step["argv"], cwd=cwd,
                                      env=clean_environment(step["argv"][0]),
                                      stdout=stdout, stderr=stderr,
                                      start_new_session=os.name != "nt",
                                      creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                                      if os.name == "nt" else 0) as child:
                    try:
                        step["exit_code"] = child.wait(timeout=timeout)
                        step["status"] = "PASS" if child.returncode == 0 else "FAIL"
                    except subprocess.TimeoutExpired:
                        step["status"] = "TIMEOUT"
                        try:
                            if os.name == "nt":
                                subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"],
                                               stdout=stdout, stderr=stderr, check=True, timeout=30)
                            else:
                                os.killpg(child.pid, signal.SIGKILL)
                        except (OSError, subprocess.SubprocessError) as error:
                            step["tree_cleanup_error"] = str(error)
                        finally:
                            child.kill()
                            step["exit_code"] = child.wait(timeout=30)
        except (OSError, subprocess.SubprocessError) as error:
            step["error"] = str(error)
        finally:
            step["seconds"] = time.monotonic() - started
            self.receipt["steps"].append(step)
            self.save()
        if step["status"] != "PASS":
            raise RuntimeError(f"{name}: {step['status']} (exit {step['exit_code']})")
        return (self.evidence / step["stdout"]).read_text(encoding="utf-8", errors="replace")

    def install(self, name: str, artifact: Path, numpy_version: str, extras: str = "") -> tuple:
        environment = self.out / "work" / name
        self.command(name + "-venv", [sys.executable, "-I", "-m", "venv", environment], self.out)
        python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        self.command(name + "-install", [python, "-I", "-m", "pip", "--isolated", "install",
                                        str(artifact) + extras, "numpy==" + numpy_version],
                     self.out)
        self.command(name + "-dependencies", [python, "-I", "-m", "pip", "check"], self.out)
        self.command(name + "-freeze", [python, "-I", "-m", "pip", "freeze", "--all"], self.out)
        if name == "test-env":
            self.command(name + "-uv-install", [python, "-I", "-m", "pip", "--isolated",
                                               "install", "uv==0.7.6"], self.out)
            probe = ("import json,shutil,subprocess; path=shutil.which('uv'); "
                     "print(json.dumps({'path':path,'version':subprocess.check_output("
                     "[path,'--version'],text=True).strip()}))")
            info = json.loads(self.command(name + "-uv-version", [python, "-I", "-c", probe],
                                           self.out))
            uv = python.parent / ("uv.exe" if os.name == "nt" else "uv")
            if Path(info["path"]) != uv or info["version"].split()[:2] != ["uv", "0.7.6"]:
                raise ValueError("private test uv origin or version mismatch")
            self.receipt["test_tools"] = {"uv": info | digest(uv.read_bytes()),
                "pressure_dependency_boundary": "existing pressure subprocess pins NumPy 2.4.6 "
                "and psutil 7.0.0 independently of the outer NumPy matrix"}
            self.save()
            self.command(name + "-tool-freeze", [python, "-I", "-m", "pip", "freeze", "--all"],
                         self.out)
        return environment, python

    def identity(self, name: str, environment: Path, python: Path, numpy_version: str) -> None:
        raw = self.command(name + "-identity", [python, "-I", "-c", IDENTITY], self.out)
        info = json.loads(raw)
        require_installation(info, environment, numpy_version)
        actual = inventory(Path(info["package"]))
        require_inventory(self.receipt["source_inventory"], actual, name)
        self.receipt.setdefault("installations", {})[name] = info | {"product_inventory": actual}
        self.save()

    def record_tests(self, junit: Path) -> None:
        try:
            self.receipt["tests"] = read_junit(junit)
        except InvalidTests as error:
            self.receipt["tests"] = error.counts
            raise
        finally:
            self.save()

    def pipeline(self, numpy_version: str) -> None:
        artifacts, work = self.out / "artifacts", self.out / "work"
        self.command("build", [sys.executable, "-I", "-m", "build", "--outdir", artifacts,
                               self.root], self.out)
        wheels, sdists = list(artifacts.glob("*.whl")), list(artifacts.glob("*.tar.gz"))
        if len(wheels) != 1 or len(sdists) != 1:
            raise ValueError("build must produce exactly one wheel and one sdist")
        wheel, sdist = wheels[0], sdists[0]
        self.receipt["artifacts"] = {
            path.name: digest(path.read_bytes()) for path in (wheel, sdist)}
        for path in (wheel, sdist):
            require_inventory(self.receipt["source_inventory"], archive_inventory(path), path.name)
        extracted = work / "extracted"
        extracted.mkdir()
        with tarfile.open(sdist) as archive:
            archive.extractall(extracted, filter="data")
        roots = list(extracted.iterdir())
        if len(roots) != 1 or not roots[0].is_dir():
            raise ValueError("sdist must have one source root")
        shipped = roots[0]
        require_inventory(self.receipt["source_inventory"], inventory(shipped / "src/selcal"),
                          "extracted")
        environment, python = self.install("test-env", sdist, numpy_version, "[dev,docs,benchmark]")
        self.identity("test-before", environment, python, numpy_version)
        junit = self.evidence / "shipped-pytest.xml"
        # Tests import scripts.* from this shipped root; the src-layout product is installed.
        try:
            self.command("pytest", [python, "-m", "pytest", "-p", "no:cacheprovider", "-q",
                                    "--junitxml=" + str(junit)], shipped)
        finally:
            self.record_tests(junit)
        self.command("ruff", [python, "-I", "-m", "ruff", "check", "src", "tests", "scripts"],
                     shipped)
        if mypy_applies(numpy_version):
            self.command("mypy", [python, "-I", "-m", "mypy", "src"], shipped)
        else:
            # Type annotations are checked against the locked NumPy 2 stubs. NumPy 1.26 ships
            # older stubs (untyped numpy.lib.format readers, no show_config(mode=)) although
            # the runtime APIs exist; this lane proves runtime compatibility through pytest.
            self.receipt["steps"].append({
                "name": "mypy", "status": "NOT_APPLICABLE", "exit_code": None,
                "reason": f"NumPy {numpy_version} type stubs predate the typed APIs; "
                          "static types are checked in the NumPy 2 lanes"})
            self.save()
        self.identity("test-after", environment, python, numpy_version)
        require_inventory(self.receipt["source_inventory"], inventory(shipped / "src/selcal"),
                          "tested")
        for name, artifact in (("wheel", wheel), ("sdist", sdist)):
            environment, python = self.install(name + "-runtime", artifact, numpy_version)
            self.identity(name + "-before", environment, python, numpy_version)
            workspace = work / (name + "-acceptance")
            for relative in ("scripts/acceptance_check.py", "examples/workflow/series.csv",
                             "examples/workflow/pearson.json"):
                target = workspace / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(shipped / relative, target)
            self.command(name + "-acceptance",
                         [python, "-I", workspace / "scripts/acceptance_check.py",
                          "--precision", "--installed", "--out",
                          self.evidence / (name + "-acceptance")], workspace)
            self.identity(name + "-after", environment, python, numpy_version)
        for path in (wheel, sdist):
            if digest(path.read_bytes()) != self.receipt["artifacts"][path.name]:
                raise ValueError("built artifact changed during acceptance")

    def execute(self, numpy_version: str) -> int:
        self.receipt["numpy_requested"] = numpy_version
        try:
            if platform.python_implementation() != "CPython" or sys.version_info[:2] not in (
                    (3, 11), (3, 12), (3, 13)):
                raise ValueError("use CPython 3.11-3.13")
            if not re.fullmatch(r"\d+\.\d+\.\d+", numpy_version):
                raise ValueError("NumPy must be an exact X.Y.Z version")
            if (self.root / "src").is_symlink():
                raise ValueError("source link refused")
            self.receipt["source_inventory"] = inventory(self.root / "src/selcal")
            if not self.receipt["source_inventory"]:
                raise ValueError("empty product inventory")
            self.save()
            self.pipeline(numpy_version)
            self.receipt["status"] = "PASS"
        except Exception as error:
            self.receipt["error"] = f"{type(error).__name__}: {error}"
        finally:
            try:
                require_inventory(self.receipt["source_inventory"],
                                  inventory(self.root / "src/selcal"), "final source")
                self.receipt["source_unchanged"] = True
            except Exception as error:
                self.receipt["source_unchanged"] = False
                self.receipt["source_recheck_error"] = str(error)
                self.receipt["status"] = "FAIL"
            self.save()
        print(f"{self.receipt['status']}: {self.evidence / 'receipt.json'}", flush=True)
        return 0 if self.receipt["status"] == "PASS" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--numpy-version", default="2.4.6")
    args = parser.parse_args()
    try:
        run = Run(Path(__file__).resolve().parents[1], args.output)
    except (OSError, ValueError) as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 2
    return run.execute(args.numpy_version)


if __name__ == "__main__":
    raise SystemExit(main())
