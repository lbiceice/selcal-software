"""Offline CI-runner contracts; never recursively run the complete CI pipeline."""

from __future__ import annotations

import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import types
import venv
import zipfile
from pathlib import Path

import pytest
from _documentation import public_documentation_text

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "scripts/ci_check.py"
WORKFLOW = ROOT / ".github/workflows/ci.yml"


def _runner():
    assert ENTRY.is_file(), "artifact-first CI entry is absent"
    spec = importlib.util.spec_from_file_location("selcal_ci_check", ENTRY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ci_delivery_entry_exists() -> None:
    assert ENTRY.is_file(), "artifact-first CI entry is absent"


@pytest.mark.parametrize("before,after", [
    (b"README.md\nsrc/pkg.py\n", b"README.md\nsetup.cfg\nsrc/pkg.py\n"),
    (b"README.md\nsrc/pkg.py\n", b"README.md\nsrc/pkg.py\n"),
    (b"README.md\r\nsrc/pkg.py\r\n", b"README.md\r\nsetup.cfg\r\nsrc/pkg.py\r\n"),
])
def test_generated_metadata_accepts_only_the_single_setup_cfg_insertion(before, after):
    _runner().require_generated_sdist_metadata(before, after)


@pytest.mark.parametrize("after", [
    b"README.md\nsetup.cfg\nsetup.cfg\nsrc/pkg.py\n",
    b"README.md\nother.cfg\nsrc/pkg.py\n",
    b"src/pkg.py\nsetup.cfg\nREADME.md\n",
    b"README.md\nsetup.cfg\n",
    b"README.md\nsetup.cfg\nsrc/changed.py\n",
    b"README.md\r\nsetup.cfg\r\nsrc/pkg.py\r\n",
])
def test_generated_metadata_refuses_other_changes(after):
    with pytest.raises(ValueError):
        _runner().require_generated_sdist_metadata(b"README.md\nsrc/pkg.py\n", after)


def test_ci_workflow_exists() -> None:
    assert WORKFLOW.is_file(), "CI definition is absent"


def test_release_inventory_declares_ci_workflow() -> None:
    from scripts.audit_repository_hygiene import _release_surface_paths

    assert WORKFLOW in _release_surface_paths(ROOT)


@pytest.mark.parametrize("kind", ["directory", "file", "inside", "root"])
def test_output_refusal_preserves_existing_bytes(tmp_path: Path, kind: str) -> None:
    runner = _runner()
    source = tmp_path / "source"
    source.mkdir()
    marker = source / "keep.txt"
    marker.write_text("untouched", encoding="utf-8")
    target = tmp_path / "output"
    if kind == "directory":
        target.mkdir()
        (target / "keep.txt").write_text("also untouched", encoding="utf-8")
    elif kind == "file":
        target.write_text("also untouched", encoding="utf-8")
    else:
        target = source / "new" if kind == "inside" else source
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with pytest.raises((ValueError, FileExistsError)):
        runner.Run(source, target)
    assert {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before
    if kind == "inside":
        assert not target.exists()


def test_real_child_has_no_python_source_injection(tmp_path: Path, monkeypatch) -> None:
    runner = _runner()
    source = tmp_path / "source"
    source.mkdir()
    for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE", "PYTHONSTARTUP",
                 "_PYTHON_SYSCONFIGDATA_NAME", "__PYVENV_LAUNCHER__", "PYTEST_ADDOPTS"):
        monkeypatch.setenv(name, "invalid-source-injection")
    run = runner.Run(source, tmp_path / "output")
    output = run.command("environment", [sys.executable, "-I", "-c",
                         "import os,json; print(json.dumps(dict(os.environ)))"], source)
    environment = json.loads(output)
    assert not any("PYTHON" in key.upper() for key in environment)
    assert "__PYVENV_LAUNCHER__" not in environment
    assert "PYTEST_ADDOPTS" not in environment


@pytest.mark.parametrize("host_tool", [False, True])
def test_real_child_uses_tools_beside_selected_python(tmp_path: Path, monkeypatch,
                                                   host_tool: bool) -> None:
    runner = _runner()
    source = tmp_path / "source"
    source.mkdir()
    environment = tmp_path / "environment"
    venv.EnvBuilder(with_pip=False, symlinks=os.name != "nt").create(environment)
    binary = environment / ("Scripts" if os.name == "nt" else "bin")
    python = binary / ("python.exe" if os.name == "nt" else "python")
    name = "uv.exe" if os.name == "nt" else "uv"
    private_tool = binary / name
    arguments = ["/c", "exit", "0"] if os.name == "nt" else ["tool-ran"]
    if os.name == "nt":
        shutil.copy(Path(os.environ["SystemRoot"]) / "System32/cmd.exe", private_tool)
    else:
        private_tool.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        private_tool.chmod(0o755)
    host = tmp_path / "host"
    host.mkdir()
    if host_tool:
        shutil.copy(private_tool, host / name)
    monkeypatch.setenv("PATH", str(host))
    run = runner.Run(source, tmp_path / "output")
    code = ("import json,shutil,subprocess; path=shutil.which('uv'); "
            f"code=subprocess.run([path]+{arguments!r},capture_output=True).returncode "
            "if path else None; print(json.dumps({'path':path,'exit_code':code}))")
    observed = json.loads(run.command("private-tool", [python, "-I", "-c", code], source))
    selected = Path(observed["path"])
    assert selected.parent.samefile(binary)
    assert selected.samefile(private_tool)
    if host_tool:
        assert not selected.samefile(host / name)
    assert observed["exit_code"] == 0


@pytest.mark.parametrize("name,extras", [
    ("test-env", "[dev,docs,benchmark]"), ("wheel-runtime", ""), ("sdist-runtime", ""),
])
def test_only_test_environment_gets_pinned_private_uv(tmp_path: Path, monkeypatch,
                                                    name: str, extras: str) -> None:
    runner = _runner()
    source = tmp_path / "source"
    source.mkdir()
    run = runner.Run(source, tmp_path / "output")
    binary = run.out / "work" / name / ("Scripts" if os.name == "nt" else "bin")
    binary.mkdir(parents=True)
    uv = binary / ("uv.exe" if os.name == "nt" else "uv")
    uv.write_bytes(b"private tool fixture; network installation is stubbed")
    observed = []

    def no_network(step, argv, cwd, **kwargs):
        observed.append((step, list(map(str, argv))))
        return json.dumps({"path": str(uv), "version": "uv 0.7.6"})

    monkeypatch.setattr(run, "command", no_network)
    run.install(name, tmp_path / "product.tar.gz", "1.26.4", extras)
    installs = [argv for _, argv in observed if "uv==0.7.6" in argv]
    assert len(installs) == (1 if name == "test-env" else 0)
    if name == "test-env":
        assert "-I" in installs[0] and "--isolated" in installs[0]
        assert run.receipt["test_tools"]["uv"]["version"] == "uv 0.7.6"
        assert run.receipt["test_tools"]["uv"]["path"] == str(uv)
        assert observed[-1][0] == "test-env-tool-freeze"
    else:
        assert "test_tools" not in run.receipt


@pytest.mark.parametrize("timeout", [False, True])
def test_actual_child_failure_retains_failed_receipt(tmp_path: Path, timeout: bool) -> None:
    runner = _runner()
    source = tmp_path / "source"
    source.mkdir()
    out = tmp_path / "output"
    run = runner.Run(source, out)
    code = "import sys,time; print('retained stdout',flush=True); " \
           "print('retained stderr',file=sys.stderr,flush=True); "
    code += "time.sleep(30)" if timeout else "sys.exit(7)"
    with pytest.raises(RuntimeError):
        run.command("failure", [sys.executable, "-I", "-c", code], source,
                    timeout=0.5 if timeout else 10)
    receipt = json.loads((out / "evidence/receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "FAIL"
    step = receipt["steps"][-1]
    assert step["status"] == ("TIMEOUT" if timeout else "FAIL")
    assert step["exit_code"] != 0
    assert step["argv"][-1] == code
    assert "retained stdout" in (out / "evidence" / step["stdout"]).read_text(encoding="utf-8")
    assert "retained stderr" in (out / "evidence" / step["stderr"]).read_text(encoding="utf-8")


def test_inventory_includes_untracked_product_files_without_git(tmp_path: Path) -> None:
    runner = _runner()
    (tmp_path / "web").mkdir()
    (tmp_path / "__init__.py").write_text("# no repository metadata", encoding="utf-8")
    (tmp_path / "untracked.py").write_text("VALUE = 7", encoding="utf-8")
    (tmp_path / "web/untracked.js").write_text("true", encoding="utf-8")
    assert set(runner.inventory(tmp_path)) == {"__init__.py", "untracked.py", "web/untracked.js"}


def test_inventory_refuses_symlink(tmp_path: Path) -> None:
    runner = _runner()
    original = tmp_path / "original.py"
    original.write_text("pass", encoding="utf-8")
    try:
        (tmp_path / "linked.py").symlink_to(original)
    except OSError as error:
        pytest.skip(f"symbolic links unavailable: {error}")
    with pytest.raises(ValueError, match="link"):
        runner.inventory(tmp_path)


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
@pytest.mark.parametrize("delta", ["same", "missing", "extra", "changed"])
def test_actual_archives_require_exact_product_inventory(tmp_path: Path, kind, delta) -> None:
    runner = _runner()
    package = tmp_path / "package"
    package.mkdir()
    (package / "__init__.py").write_bytes(b"# product\n")
    expected = runner.inventory(package)
    members = {"__init__.py": b"# product\n"}
    if delta == "missing":
        members.clear()
    elif delta == "extra":
        members["unexpected.txt"] = b"unexpected product member"
    elif delta == "changed":
        members["__init__.py"] = b"# stale product\n"
    path = tmp_path / ("candidate.whl" if kind == "wheel" else "candidate.tar.gz")
    if kind == "wheel":
        with zipfile.ZipFile(path, "w") as archive:
            for name, payload in members.items():
                archive.writestr("selcal/" + name, payload)
    else:
        with tarfile.open(path, "w:gz") as archive:
            for name, payload in members.items():
                member = tarfile.TarInfo("candidate/src/selcal/" + name)
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
    if delta == "same":
        runner.require_inventory(expected, runner.archive_inventory(path), kind)
    else:
        with pytest.raises(ValueError, match="inventory"):
            runner.require_inventory(expected, runner.archive_inventory(path), kind)


@pytest.mark.parametrize("delta", ["missing", "extra", "changed"])
def test_installed_inventory_mismatch_is_rejected(tmp_path: Path, delta: str) -> None:
    runner = _runner()
    package = tmp_path / "selcal"
    package.mkdir()
    init = package / "__init__.py"
    init.write_text("pass", encoding="utf-8")
    expected = runner.inventory(package)
    if delta == "missing":
        init.unlink()
    elif delta == "extra":
        (package / "unexpected.py").write_text("pass", encoding="utf-8")
    else:
        init.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="inventory"):
        runner.require_inventory(expected, runner.inventory(package), "installed")


def test_installation_origin_requires_this_venv(tmp_path: Path) -> None:
    runner = _runner()
    environment = tmp_path / "venv"
    legitimate = environment / "lib/site-packages/selcal"
    legitimate.mkdir(parents=True)
    (legitimate / "__init__.py").write_text("pass", encoding="utf-8")
    info = {"prefix": str(environment), "package": str(legitimate),
            "purelib": str(legitimate.parent), "numpy": "2.4.6"}
    runner.require_installation(info, environment, "2.4.6")
    wrong = tmp_path / "old/site-packages/selcal"
    wrong.mkdir(parents=True)
    info["package"] = str(wrong)
    with pytest.raises(ValueError, match="origin"):
        runner.require_installation(info, environment, "2.4.6")


@pytest.mark.parametrize("xml", ["<testsuites/>",
                                 '<testsuite><testcase><failure/></testcase></testsuite>',
                                 '<testsuite><testcase><error/></testcase></testsuite>'])
def test_empty_or_failed_junit_never_passes(tmp_path: Path, xml: str) -> None:
    runner = _runner()
    path = tmp_path / "pytest.xml"
    path.write_text(xml, encoding="utf-8")
    with pytest.raises(ValueError, match="tests"):
        runner.read_junit(path)


def test_junit_preserves_skips(tmp_path: Path) -> None:
    runner = _runner()
    path = tmp_path / "pytest.xml"
    path.write_text('<testsuite><testcase name="ok"/><testcase name="optional">'
                    '<skipped message="browser unavailable"/></testcase></testsuite>',
                    encoding="utf-8",
                )
    result = runner.read_junit(path)
    assert result["passed"] == 1 and result["skipped"] == 1
    assert result["skip_details"][0]["reason"] == "browser unavailable"


def test_failed_junit_counts_are_saved_in_failed_receipt(tmp_path: Path) -> None:
    runner = _runner()
    source = tmp_path / "source"
    source.mkdir()
    run = runner.Run(source, tmp_path / "output")
    junit = run.evidence / "pytest.xml"
    junit.write_text(
        '<testsuite><testcase name="bad"><failure/></testcase>'
        '<testcase name="error"><error/></testcase><testcase name="optional">'
        '<skipped message="native only"/></testcase></testsuite>',
        encoding="utf-8",
    )
    assert hasattr(run, "record_tests"), "failed test counts need a retained receipt"
    with pytest.raises(ValueError, match="tests"):
        run.record_tests(junit)
    receipt = json.loads((run.evidence / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "FAIL"
    assert receipt["tests"]["failures"] == 1
    assert receipt["tests"]["errors"] == 1
    assert receipt["tests"]["skipped"] == 1
    assert receipt["tests"]["passed"] == 0


def test_successful_test_process_without_junit_is_refused(tmp_path: Path, monkeypatch) -> None:
    runner = _runner()
    source = tmp_path / "source"
    source.mkdir()
    run = runner.Run(source, tmp_path / "output")
    content = b"# product\n"
    run.receipt["source_inventory"] = {"__init__.py": runner.digest(content)}
    with zipfile.ZipFile(run.out / "artifacts/product.whl", "w") as wheel:
        wheel.writestr("selcal/__init__.py", content)
    with tarfile.open(run.out / "artifacts/product.tar.gz", "w:gz") as sdist:
        member = tarfile.TarInfo("product/src/selcal/__init__.py")
        member.size = len(content)
        sdist.addfile(member, io.BytesIO(content))
    # Network/install boundaries are disabled; real archive extraction and control flow remain.
    monkeypatch.setattr(run, "command", lambda *args, **kwargs: "")
    monkeypatch.setattr(run, "install", lambda *args: (tmp_path / "venv", Path(sys.executable)))
    monkeypatch.setattr(run, "identity", lambda *args: None)
    with pytest.raises(FileNotFoundError, match=r"shipped-pytest\.xml"):
        run.pipeline("2.4.6")


def test_pipeline_installs_all_declared_shipped_test_dependencies(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = _runner()
    source = tmp_path / "source"
    source.mkdir()
    run = runner.Run(source, tmp_path / "output")
    content = b"# product\n"
    run.receipt["source_inventory"] = {"__init__.py": runner.digest(content)}
    with zipfile.ZipFile(run.out / "artifacts/product.whl", "w") as wheel:
        wheel.writestr("selcal/__init__.py", content)
    with tarfile.open(run.out / "artifacts/product.tar.gz", "w:gz") as sdist:
        member = tarfile.TarInfo("product/src/selcal/__init__.py")
        member.size = len(content)
        sdist.addfile(member, io.BytesIO(content))
    selected = []

    def stop_before_install(name, artifact, numpy_version, extras=""):
        selected.append((name, extras))
        raise RuntimeError("stop before network installation")

    monkeypatch.setattr(run, "command", lambda *args, **kwargs: "")
    monkeypatch.setattr(run, "install", stop_before_install)
    with pytest.raises(RuntimeError, match="stop before network"):
        run.pipeline("2.4.6")
    assert selected == [("test-env", "[dev,docs,benchmark]")]


def test_taskkill_failure_still_kills_actual_timed_out_child(tmp_path: Path, monkeypatch) -> None:
    runner = _runner()
    source = tmp_path / "source"
    source.mkdir()
    run = runner.Run(source, tmp_path / "output")
    monkeypatch.setattr(
        runner, "os", types.SimpleNamespace(name="nt", environ=os.environ, pathsep=os.pathsep)
    )
    monkeypatch.setattr(runner.subprocess, "CREATE_NEW_PROCESS_GROUP", 0, raising=False)

    def unavailable_taskkill(*args, **kwargs):
        raise subprocess.TimeoutExpired("taskkill", 30)

    monkeypatch.setattr(runner.subprocess, "run", unavailable_taskkill)
    started = time.monotonic()
    with pytest.raises(RuntimeError):
        run.command(
            "timeout",
            [sys.executable, "-I", "-c", "import time; time.sleep(3)"],
            source,
            timeout=0.2,
        )
    assert time.monotonic() - started < 2, "timed-out child survived taskkill failure"
    step = json.loads((run.evidence / "receipt.json").read_text(encoding="utf-8"))["steps"][-1]
    assert step["status"] == "TIMEOUT" and step["exit_code"] is not None


def test_workflow_contract_and_sdist_inclusion() -> None:
    assert WORKFLOW.is_file(), "CI definition is absent"
    text = WORKFLOW.read_text(encoding="utf-8")
    for required in (
        "d23441a48e516b6c34aea4fa41551a30e30af803",
        "ece7cb06caefa5fff74198d8649806c4678c61a1",
        "ea165f8d65b6e75b540449e92b4886f43607fa02",
        "contents: read",
        "persist-credentials: false",
        "fail-fast: false",
        "timeout-minutes:",
        "ubuntu-latest",
        "macos-latest",
        "windows-latest",
        '"3.11"',
        '"3.12"',
        '"3.13"',
        '"2.4.6"',
        '"1.26.4"',
        "include:",
        "if: always()",
        "scripts/ci_check.py",
        "/evidence",
        "/artifacts",
        "workflow_dispatch:",
    ):
        assert required in text, required
    for forbidden in ("pull_request_target", "continue-on-error", "secrets.", "/work\n"):
        assert forbidden not in text
    # Manual runs only, and no matrix job in a private repository (2026-10-06 local-first
    # plan). The whole file is frozen so that another trigger, a moved job guard or a smaller
    # matrix cannot pass; an intended workflow change must update this contract on review.
    expected_workflow = """name: Artifact-first CI

on:
  workflow_dispatch:

permissions:
  contents: read

jobs:
  artifacts:
    if: ${{ github.event.repository.private == false }}
    runs-on: ${{ matrix.os }}
    timeout-minutes: 90
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python: ["3.11", "3.12", "3.13"]
        numpy: ["2.4.6"]
        include:
          - os: ubuntu-latest
            python: "3.11"
            numpy: "1.26.4"
    steps:
      - uses: actions/checkout@d23441a48e516b6c34aea4fa41551a30e30af803 # v6
        with:
          persist-credentials: false
      - uses: actions/setup-python@ece7cb06caefa5fff74198d8649806c4678c61a1 # v6
        with:
          python-version: ${{ matrix.python }}
      - name: Install build frontend
        run: python -I -m pip install "build>=1.2,<2"
      - name: Build and check delivered artifacts
        run: python -I scripts/ci_check.py "${{ runner.temp }}/selcal-ci" \
--numpy-version "${{ matrix.numpy }}"
      - name: Preserve results and built artifacts
        if: always()
        uses: actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02 # v4
        with:
          name: selcal-${{ matrix.os }}-py${{ matrix.python }}-np${{ matrix.numpy }}
          path: |
            ${{ runner.temp }}/selcal-ci/evidence
            ${{ runner.temp }}/selcal-ci/artifacts
          if-no-files-found: warn
          retention-days: 14
"""
    assert text == expected_workflow
    assert "include .github/workflows/ci.yml" in (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    assert "Artifact-first CI" in public_documentation_text(ROOT)


def test_source_mutation_and_build_failure_stay_failed(tmp_path: Path, monkeypatch) -> None:
    runner = _runner()
    source = tmp_path / "source"
    package = source / "src/selcal"
    package.mkdir(parents=True)
    target = package / "__init__.py"
    target.write_text("pass", encoding="utf-8")
    run = runner.Run(source, tmp_path / "output")
    expected = runner.inventory(package)

    def failed_build(*args):
        target.write_text("changed during build", encoding="utf-8")
        raise RuntimeError("build failed")

    monkeypatch.setattr(run, "pipeline", failed_build)
    assert run.execute("2.4.6") == 1
    receipt = json.loads((run.out / "evidence/receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "FAIL"
    assert "build failed" in receipt["error"]
    assert receipt["source_unchanged"] is False
    assert receipt["source_inventory"] == expected


def test_static_typing_runs_against_numpy2_stubs_only() -> None:
    """NumPy 1.26 stubs lack typed APIs that exist at runtime; that lane records why."""
    runner = _runner()
    assert runner.mypy_applies("2.4.6") and runner.mypy_applies("2.0.0")
    assert not runner.mypy_applies("1.26.4")
    source = (Path(__file__).resolve().parents[1] / "scripts/ci_check.py").read_text(
        encoding="utf-8"
    )
    assert '"status": "NOT_APPLICABLE"' in source and "static types are checked" in source
