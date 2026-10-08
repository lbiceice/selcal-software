"""P19-01: short physical Windows runner paths do not weaken long-path tests.

Native probes use Windows PowerShell 5.1 and a real CPython venv; static contracts
also run elsewhere. Every probe retains its uniquely owned short evidence folder.
A custom root close to the length limit can leave no room for a link/file negative
fixture; that specific probe then skips before I/O, without relaxing its assertion.
"""

from __future__ import annotations

import inspect
import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path, PurePath, PureWindowsPath
from zipfile import ZipFile

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "windows_check.ps1"
WINDOWS = sys.platform == "win32"
native = pytest.mark.skipif(not WINDOWS, reason="native Windows PowerShell 5.1 path probe")


def _source() -> str:
    assert RUNNER.is_file(), "the source package must carry its Windows check runner"
    return RUNNER.read_text(encoding="utf-8")


def _block() -> str:
    source = _source()
    start = source.index("# BEGIN short-test-work")
    end = source.index("# END short-test-work") + len("# END short-test-work")
    return source[start:end]


def _test_root() -> Path:
    selected = os.environ.get("SELCAL_WINDOWS_TEST_ROOT")
    return Path(selected) if selected else Path(ROOT.anchor) / "SelCal-Test"


def _short_root(label: str) -> Path:
    # Only negative link/file fixtures need a separate child path. Keep enough room for
    # that component: every native probe still enforces the runner's 100/64 limits.
    return _test_root() / (label + uuid.uuid4().hex[:8])


def _require_negative_fixture_budget(root: PurePath) -> None:
    # Mirror only the existing layout lengths, before mkdir/write/link. Ordinary
    # probes use the selected root directly and never call this negative-fixture guard.
    run = root / ("r-" + "0" * 16)
    # PowerShell/.NET String.Length counts UTF-16 units, including non-BMP path names.
    python_length = len(str(run / "venv" / "Scripts" / "python.exe").encode("utf-16-le")) // 2
    pytest_length = len(str(run / "pytest").encode("utf-16-le")) // 2
    if python_length > 100 or pytest_length > 64:
        pytest.skip(
            "negative fixture not executed: selected test root leaves insufficient "
            f"path space (python length {python_length}/100, pytest length "
            f"{pytest_length}/64); NOT_PHYSICAL refusal remains unverified in this run"
        )


def _probe(tmp_path: Path, *, requested: str = "", environment: str = "",
           make_venv: bool = False, console_code_page: int | None = None) -> tuple[int, dict]:
    """Run the runner's layout block in Windows PowerShell 5.1 and return (exit, result).

    The result travels as an explicit UTF-8 JSON file, not through stdout: stdout is encoded
    with the console output code page, which need not be UTF-8 (R19 v3 Windows return, item 10).
    With console_code_page the probe runs in its own new console set to that code page, so the
    caller's console is never changed.
    """
    shell = shutil.which("powershell")
    assert shell is not None, "Windows PowerShell 5.1 is required for this native check"
    case = tmp_path / "case.json"
    case.write_text(json.dumps({"repository": str(ROOT), "requested": requested,
                               "environment": environment, "python": sys.executable,
                               "make_venv": make_venv}), encoding="utf-8")
    script = tmp_path / "probe.ps1"
    result_file = tmp_path / "result.json"
    script.write_text(r'''param([string]$CaseFile, [string]$ResultFile, [int]$CodePage = 0)
$ErrorActionPreference = "Stop"
if ($CodePage) { [Console]::OutputEncoding = [Text.Encoding]::GetEncoding($CodePage) }
function Save-Result($Value) {
    $json = $Value | ConvertTo-Json -Depth 6 -Compress
    [IO.File]::WriteAllText($ResultFile, $json, (New-Object Text.UTF8Encoding($false)))
    $json
}
''' + _block() + r'''
$caseData = [IO.File]::ReadAllText($CaseFile) | ConvertFrom-Json
try {
    $options = @{
        Repository = $caseData.repository
        RequestedRoot = $caseData.requested
        EnvironmentRoot = $caseData.environment
    }
    $layout = New-SelCalWindowsTestWork @options
    $child = $null
    if ($caseData.make_venv) {
        & $caseData.python -I -m venv --without-pip $layout.venv
        if ($LASTEXITCODE -ne 0) { throw "native venv creation failed: $LASTEXITCODE" }
        $code = 'import json,sys; ' +
            'print(json.dumps(dict(executable=sys.executable,prefix=sys.prefix)))'
        $child = (& $layout.python -I -c $code) | ConvertFrom-Json
        if ($LASTEXITCODE -ne 0) { throw "short physical venv failed to start: $LASTEXITCODE" }
    }
    $result = [ordered]@{
        status="PASS"; layout=$layout; child=$child
        ps_version=$PSVersionTable.PSVersion.ToString()
        console_code_page=[Console]::OutputEncoding.CodePage
    }
    Save-Result $result
    exit 0
} catch {
    Save-Result ([ordered]@{status="ERROR";error=[string]$_})
    exit 3
}
''', encoding="ascii")
    env = os.environ.copy()
    # An inherited PS7 module path can load PS7-only modules in PS5. Let native
    # Windows PowerShell build its own default module path; no product env is cleared.
    for key in list(env):
        if key.lower() == "psmodulepath":
            env.pop(key)
    argv = [shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script),
            "-CaseFile", str(case), "-ResultFile", str(result_file)]
    flags = 0
    if console_code_page is not None:
        argv += ["-CodePage", str(console_code_page)]
        flags = subprocess.CREATE_NEW_CONSOLE  # type: ignore[attr-defined]
    done = subprocess.run(argv, capture_output=True, env=env, timeout=180, creationflags=flags)
    # stdout/stderr only diagnose a missing result; they are never parsed.
    assert result_file.is_file(), (done.stdout + done.stderr).decode(errors="backslashreplace")
    return done.returncode, json.loads(result_file.read_text(encoding="utf-8"))


def test_runner_restores_temporary_and_cache_environment_and_keeps_full_suite() -> None:
    source = _source()
    for variable in ("Temp", "Tmp", "PipCacheDir", "UvCacheDir"):
        assert f"$previous{variable} = $env:" in source
        assert f"= $previous{variable}" in source.split("} finally {", 1)[1]
    assert '"--basetemp=$pytestBase"' in source
    assert '"-m", "pytest", "-p", "no:cacheprovider", "-q", "-rfEs"' in source
    assert '"-I", "tests\\_windows_acceptance.py"' in source
    assert "-k " not in source and "--ignore" not in source
    assert '$env:PYTHONPATH =' not in source and '$env:PYTHONUTF8 =' not in source


def test_runner_keeps_outer_archive_contract_and_returns_short_work_evidence() -> None:
    source = _source()
    assert '"windows-check-$stamp"' in source
    assert 'Join-Path $out "test-work-evidence.zip"' in source
    assert "-EvidenceDirectory $testWork.run" in source
    assert "-ExcludedDirectories @($venv)" in source
    assert "-EvidenceDirectory $out -DestinationPath $archive\n" in source
    assert "test_work_archive = $null" in source
    assert "test_work = $testWork" in source
    assert "process paths: executable length" in source
    assert "runner_cwd_length = $repo.Length" in source


def test_runner_passes_resolved_test_root_to_children_and_restores_environment() -> None:
    source = _source()
    capture = "$previousTestRoot = $env:SELCAL_WINDOWS_TEST_ROOT"
    propagate = "$env:SELCAL_WINDOWS_TEST_ROOT = $testWork.root"
    restore = "$env:SELCAL_WINDOWS_TEST_ROOT = $previousTestRoot"
    assert capture in source
    assert propagate in source
    assert source.index(capture) < source.index(propagate) < source.index('"full test suite"')
    assert restore in source.rsplit("} finally {", 1)[1]


def test_short_root_uses_selected_root_without_creating_it(tmp_path: Path, monkeypatch) -> None:
    chosen = tmp_path / "chosen"
    monkeypatch.setenv("SELCAL_WINDOWS_TEST_ROOT", str(chosen))
    first, second = _short_root("e-"), _short_root("e-")
    assert first.parent == second.parent == chosen
    assert first != second
    assert not chosen.exists()


def test_default_root_construction_is_same_volume_without_filesystem_changes(monkeypatch) -> None:
    monkeypatch.delenv("SELCAL_WINDOWS_TEST_ROOT", raising=False)
    assert _short_root("e-").parent == Path(ROOT.anchor) / "SelCal-Test"
    assert 'Join-Path ([IO.Path]::GetPathRoot($Repository)) "SelCal-Test"' in _block()


def test_ordinary_native_probes_do_not_lengthen_the_selected_root() -> None:
    for probe in (
        test_native_explicit_root_wins_over_environment_and_never_reuses,
        test_native_environment_root_is_respected,
        test_native_short_venv_python_really_starts,
        test_native_failed_setup_still_returns_nested_short_work_evidence,
    ):
        assert "_short_root(" not in inspect.getsource(probe)


@pytest.mark.parametrize(
    "selected_root, expected_pytest_length",
    [
        ("F:/SelCal-Test", 51),
        ("F:/" + "x" * 24, 64),
        ("F:/" + "x" * 25, 65),
        ("F:/" + "\U0001f600" * 12 + "x", 65),
    ],
)
def test_negative_fixture_budget_covers_normal_exact_and_over_limit(
    selected_root: str, expected_pytest_length: int
) -> None:
    selected = PureWindowsPath(selected_root)
    fixture = selected / ("f-" + "a" * 8)
    path = str(fixture / ("r-" + "0" * 16) / "pytest")
    assert len(path.encode("utf-16-le")) // 2 == expected_pytest_length
    if expected_pytest_length > 64:
        with pytest.raises(pytest.skip.Exception, match="negative fixture not executed"):
            _require_negative_fixture_budget(fixture)
    else:
        _require_negative_fixture_budget(fixture)


@pytest.mark.parametrize("kind", ["link", "existing-file"])
def test_overlong_negative_fixture_skips_before_filesystem_changes(
    tmp_path: Path, monkeypatch, kind: str
) -> None:
    module = sys.modules[__name__]
    # A long selected root may fit the ordinary run yet leave no room for a negative
    # fixture's extra component. This test must not create even its parent directory.
    monkeypatch.setenv("SELCAL_WINDOWS_TEST_ROOT", "F:/" + "x" * 25)

    def refuse_io(*args, **kwargs):
        pytest.fail("negative fixture performed I/O before checking its path budget")

    monkeypatch.setattr(Path, "mkdir", refuse_io)
    monkeypatch.setattr(Path, "write_bytes", refuse_io)
    monkeypatch.setattr(module, "_probe", refuse_io)
    with pytest.raises(pytest.skip.Exception, match="negative fixture not executed"):
        if kind == "link":
            test_native_linked_root_is_refused_before_run_creation(tmp_path)
        else:
            test_native_unsafe_root_fails_without_fallback(tmp_path, "existing-file")


def test_default_native_probe_skips_before_io_when_another_root_is_selected(
    tmp_path: Path, monkeypatch
) -> None:
    chosen = tmp_path / "chosen"
    monkeypatch.setenv("SELCAL_WINDOWS_TEST_ROOT", str(chosen))

    def refuse_default_probe(*args, **kwargs):
        pytest.fail("default-volume native probe ran despite a different selected root")

    monkeypatch.setattr(sys.modules[__name__], "_probe", refuse_default_probe)
    with pytest.raises(pytest.skip.Exception, match="different test root"):
        test_native_default_is_same_volume_short_physical_and_owned(tmp_path)
    assert not chosen.exists()


@native
def test_native_default_is_same_volume_short_physical_and_owned(tmp_path: Path) -> None:
    if _test_root() != Path(ROOT.anchor) / "SelCal-Test":
        pytest.skip(
            "different test root selected; default-volume filesystem probe not executed; "
            "default path construction is checked separately without filesystem changes"
        )
    code, payload = _probe(tmp_path)
    assert code == 0 and payload["ps_version"].startswith("5.1"), payload
    layout = payload["layout"]
    assert layout["root_source"] == "repository_volume_default"
    assert Path(layout["root"]) == Path(ROOT.anchor) / "SelCal-Test"
    assert layout["python_length"] <= 100 and layout["pytest_base_length"] <= 64
    assert not Path(layout["pytest_base"]).exists(), "pytest owns its fresh base"
    owner = Path(layout["run"]) / "test-work-layout.json"
    assert json.loads(owner.read_bytes()) == layout
    assert len(layout["run_id"]) == 32
    assert Path(layout["temporary"]).is_dir() and Path(layout["cache"]).is_dir()
    assert Path(layout["run"]).resolve() == Path(layout["run"])


@native
def test_native_explicit_root_wins_over_environment_and_never_reuses(tmp_path: Path) -> None:
    root = _test_root()
    first_code, first = _probe(tmp_path, requested=str(root), environment="relative-refused")
    assert first_code == 0, first
    assert first["layout"]["root_source"] == "parameter"
    keep = Path(first["layout"]["run"]) / "keep.txt"
    keep.write_bytes(b"retain previous evidence")
    second_code, second = _probe(tmp_path, requested=str(root))
    assert second_code == 0, second
    assert second["layout"]["run"] != first["layout"]["run"]
    assert keep.read_bytes() == b"retain previous evidence"


@native
def test_native_environment_root_is_respected(tmp_path: Path) -> None:
    root = _test_root()
    code, payload = _probe(tmp_path, environment=str(root))
    assert code == 0, payload
    assert payload["layout"]["root_source"] == "SELCAL_WINDOWS_TEST_ROOT"
    assert Path(payload["layout"]["root"]) == root


@native
def test_native_linked_root_is_refused_before_run_creation(tmp_path: Path) -> None:
    from _platform_support import directory_link_or_skip

    target = _short_root("t-")
    link = _short_root("l-")
    _require_negative_fixture_budget(target)
    _require_negative_fixture_budget(link)
    target.mkdir(parents=True)
    directory_link_or_skip(link, target)
    code, payload = _probe(tmp_path, requested=str(link))
    assert code == 3 and "TEST_WORK_ROOT_NOT_PHYSICAL" in payload["error"], payload
    assert list(target.iterdir()) == [], "a linked root must not get a run or venv"


@native
@pytest.mark.parametrize("kind", ["relative", "volume-root", "too-deep", "existing-file"])
def test_native_unsafe_root_fails_without_fallback(tmp_path: Path, kind: str) -> None:
    if kind == "relative":
        root = "not-an-absolute-test-root"
        expected = "TEST_WORK_ROOT_INVALID"
    elif kind == "volume-root":
        root = ROOT.anchor
        expected = "TEST_WORK_ROOT_INVALID"
    elif kind == "too-deep":
        root = str(Path(ROOT.anchor) / ("too-deep-" + "x" * 100))
        expected = "TEST_WORK_ROOT_TOO_LONG"
        assert not Path(root).exists()
    else:
        target = _short_root("f-")
        _require_negative_fixture_budget(target)
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(b"existing user bytes")
        root = str(target)
        expected = "TEST_WORK_ROOT_NOT_PHYSICAL"
    code, payload = _probe(tmp_path, requested=root)
    assert code == 3 and payload["status"] == "ERROR", payload
    assert expected in payload["error"]
    if kind == "too-deep":
        assert not Path(root).exists(), "rejected before any path creation"
    elif kind == "existing-file":
        assert Path(root).read_bytes() == b"existing user bytes"


@native
def test_native_short_venv_python_really_starts(tmp_path: Path) -> None:
    code, payload = _probe(tmp_path, requested=str(_test_root()), make_venv=True)
    assert code == 0, payload
    layout = payload["layout"]
    assert Path(payload["child"]["executable"]) == Path(layout["python"])
    assert Path(payload["child"]["prefix"]) == Path(layout["venv"])
    assert Path(layout["venv"]).is_dir(), "short venv is retained, not moved or deleted"


@native
def test_native_failed_setup_still_returns_nested_short_work_evidence(tmp_path: Path) -> None:
    # Execute the full runner, but fail its first venv setup deliberately; no pip,
    # scientific computation or full pytest starts. This tests the actual finally/archive.
    repo = tmp_path / "source"
    (repo / "scripts").mkdir(parents=True)
    (repo / "src" / "selcal").mkdir(parents=True)
    (repo / "src" / "selcal" / "workflow.py").write_bytes(b"# owned stub, LF only\n")
    for name in ("windows_check.ps1", "windows_native_capture.ps1", "windows_evidence_archive.ps1"):
        shutil.copyfile(ROOT / "scripts" / name, repo / "scripts" / name)
    shell = shutil.which("powershell")
    assert shell is not None
    env = {key: value for key, value in os.environ.items() if key.lower() != "psmodulepath"}
    command = [shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
               str(repo / "scripts" / "windows_check.ps1"),
               "-Python", "__selcal_missing_executable__", "-TestWorkRoot", str(_test_root())]
    done = subprocess.run(command, env=env, capture_output=True, timeout=120)
    folders = list(tmp_path.glob("windows-check-*"))
    (folder,) = [path for path in folders if path.is_dir()]
    summary = json.loads((folder / "runner-summary.json").read_bytes())
    assert done.returncode == 1 and summary["status"] == "FAIL", summary
    assert summary["first_environment_failure"] == "create venv"
    assert "full test suite" in summary["blocked_steps"]
    assert summary["test_work_archive"] == "test-work-evidence.zip"
    assert Path(summary["test_work"]["run"]).is_dir(), "owned diagnostics stay in place"
    with ZipFile(folder / "test-work-evidence.zip") as inner:
        assert inner.testzip() is None
        assert "test-work-layout.json" in inner.namelist()
        assert all(not name.startswith("venv/") for name in inner.namelist())
        omissions = json.loads(inner.read("archive-omissions.json"))
        assert omissions["excluded_venvs"] == ["venv"]
    with ZipFile(Path(str(folder) + ".zip")) as outer:
        assert outer.testzip() is None
        assert "test-work-evidence.zip" in outer.namelist()
        assert "runner-summary.json" in outer.namelist()


@native
def test_native_probe_result_is_exact_under_a_non_utf8_console(tmp_path: Path) -> None:
    """R19 v3 Windows return, item 10: a non-UTF-8 console must not change the probe result."""
    name = "\u7f16\u7801 " + uuid.uuid4().hex[:6]  # two Chinese characters, a space, hex
    root = _test_root() / name
    run = root / ("r-" + "0" * 16)
    if len(str(run / "pytest").encode("utf-16-le")) // 2 > 64:
        pytest.skip("selected test root leaves no room for this probe; default root covers it")
    code, payload = _probe(tmp_path, requested=str(root), console_code_page=936)
    assert code == 0, payload
    assert payload["console_code_page"] == 936, "the probe ran in its own cp936 console"
    assert payload["layout"]["root"] == str(root), payload["layout"]["root"]
    assert payload["layout"]["root_source"] == "parameter"
