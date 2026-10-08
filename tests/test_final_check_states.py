"""R17 Windows item 3: automatic.json must say RUNNING while running and a verdict only at the end.

The outer FINAL_CHECK.ps1 runs with stub steps: natively on Windows in Windows PowerShell 5.1 (the
installed Python is an empty venv of this Python), elsewhere in a Linux PowerShell 7 container when
SELCAL_POWERSHELL_DOCKER_IMAGE is set (not Windows evidence). Only its Windows guard is removed in
the copy. Stubs are ASCII because Windows PowerShell 5.1 reads BOM-less scripts in the ANSI code
page. A source tree without the outer script skips every test by name (R18 Windows return: a
module-level skip produced a JUnit record without a test identity, which the native gate refuses).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OUTER = ROOT / "docs" / "status" / "windows_outer"
GUARD = 'if ([Environment]::OSVersion.Platform -ne "Win32NT") { throw "Run on native Windows" }'
WINDOWS = sys.platform == "win32"

pytestmark = pytest.mark.skipif(
    not (OUTER / "FINAL_CHECK.ps1").is_file(),
    reason="the outer FINAL_CHECK.ps1 is not part of this source tree",
)

STUB_RUNNER = r"""param([string]$Python)
$pkg = Split-Path $PSScriptRoot -Parent
$mode = [IO.File]::ReadAllText((Join-Path $pkg "mode.txt")).Trim()
$receipt = Get-ChildItem -LiteralPath $pkg -Directory -Filter "final-run-*" | Select-Object -First 1
$snapshot = Join-Path $pkg "while-running.json"
Copy-Item -LiteralPath (Join-Path $receipt.FullName "automatic.json") -Destination $snapshot
if ($mode -eq "interrupt") { Start-Sleep -Seconds 60 }
if ($mode -eq "suite-fails") { exit 7 }
exit 0
"""

STUB_INSTALL = r"""param([string]$Python)
$mode = [IO.File]::ReadAllText((Join-Path (Split-Path $PSScriptRoot -Parent) "mode.txt")).Trim()
if ($mode -eq "install-fails") { exit 5 }
$state = Join-Path $PSScriptRoot ".selcal-user"
New-Item -ItemType Directory -Path $state | Out-Null
if ($mode -eq "python-missing") { exit 0 }
if ([Environment]::OSVersion.Platform -eq "Win32NT") {
    & $Python -I -m venv --without-pip (Join-Path $state "venv")
    exit $LASTEXITCODE
}
$folder = Join-Path $state "venv/Scripts"
$scripts = New-Item -ItemType Directory -Force -Path $folder
$python = Join-Path $scripts.FullName "python.exe"
[IO.File]::WriteAllText($python, "#!/bin/sh`nexit 0`n")
& chmod 755 $python
exit 0
"""

# Non-ASCII text built from code points, so the script itself stays ASCII.
STUB_REUSE = (
    "$note = [string][char]0x4E2D + [char]0x6587 + ' x'\n"
    "Write-Output ('{\"status\":\"REUSE_OK\",\"note\":\"' + $note + '\"}')\n"
    "exit 0\n"
)


def _package(tmp_path: Path, mode: str, *, installed: bool = False) -> Path:
    pkg = tmp_path / "包 根"
    (pkg / "software").mkdir(parents=True)
    (pkg / "acceptance" / "private_cases").mkdir(parents=True)
    script = (OUTER / "FINAL_CHECK.ps1").read_text(encoding="utf-8")
    assert script.count(GUARD) == 1
    (pkg / "FINAL_CHECK.ps1").write_text(script.replace(GUARD, "# guard removed"), "utf-8")
    members = {
        "software/RUN_WINDOWS_TEST.ps1": STUB_RUNNER,
        "software/INSTALL_WINDOWS.ps1": STUB_INSTALL,
        "software/verify_installed_identity.py": "# stub\n",
        "acceptance/CHECK_REUSE.ps1": STUB_REUSE,
        "acceptance/research_case_flow.py": "# stub\n",
        "acceptance/check_saved_downloads.py": "# stub\n",
        "acceptance/private_cases/cases.json": "{}\n",
    }
    for name, text in members.items():
        assert text.isascii()
        (pkg / name).write_text(text, encoding="utf-8")
    (pkg / "mode.txt").write_text(mode, encoding="utf-8")
    if installed:
        state = pkg / "software" / ".selcal-user"
        if WINDOWS:
            venv.EnvBuilder(with_pip=False).create(state / "venv")
        else:
            python = state / "venv" / "Scripts" / "python.exe"
            python.parent.mkdir(parents=True)
            python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            python.chmod(0o755)
    rows = []
    for name in ["FINAL_CHECK.ps1", *members]:
        rows.append(f"{hashlib.sha256((pkg / name).read_bytes()).hexdigest()}  {name}\n")
    (pkg / "FINAL_SHA256SUMS.txt").write_text("".join(rows), encoding="utf-8")
    return pkg


def _command(tmp_path: Path, pkg: Path, *, interrupt: bool) -> list[str]:
    if WINDOWS:
        if interrupt:
            pytest.skip("Ctrl-C cannot be delivered reliably to a background Windows process")
        shell = shutil.which("powershell")
        if shell is None:
            pytest.fail("Windows PowerShell 5.1 is required for the native receipt check")
        return [shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                str(pkg / "FINAL_CHECK.ps1"), "-Python", sys.executable]
    image = os.environ.get("SELCAL_POWERSHELL_DOCKER_IMAGE")
    if image is None:
        pytest.skip("Linux PowerShell image not opted in; the Windows run checks this natively")
    inner = "/probe/" + pkg.relative_to(tmp_path).as_posix()
    command = f"pwsh -NoProfile -File '{inner}/FINAL_CHECK.ps1' -Python stub"
    if interrupt:
        # Ctrl-C equivalent: SIGINT to PowerShell while the first step runs. PowerShell runs in
        # the foreground through exec (a background job of sh would ignore SIGINT); $$ is its PID.
        command = (
            f"(while [ ! -f '{inner}/while-running.json' ]; do sleep 0.2; done; sleep 1; "
            f"kill -INT $$) & exec {command}"
        )
    return ["docker", "run", "--rm", "--pull", "never", "--network", "none", "--mount",
            f"type=bind,src={tmp_path},dst=/probe", image, "sh", "-c", command]


def _run(tmp_path: Path, pkg: Path, *, interrupt: bool = False) -> tuple[int, dict, dict | None]:
    done = subprocess.run(
        _command(tmp_path, pkg, interrupt=interrupt), capture_output=True, timeout=600
    )
    receipts = list(pkg.glob("final-run-*/automatic.json"))
    assert len(receipts) == 1, done.stdout.decode(errors="replace") + done.stderr.decode(
        errors="replace"
    )
    assert not list(pkg.glob("final-run-*/automatic.json.tmp"))
    running = pkg / "while-running.json"
    snapshot = json.loads(running.read_bytes()) if running.exists() else None
    return done.returncode, json.loads(receipts[0].read_bytes()), snapshot


def _statuses(receipt: dict) -> dict[str, str]:
    return {step["name"]: step["status"] for step in receipt["steps"]}


def test_running_receipt_then_pass(tmp_path: Path) -> None:
    code, receipt, running = _run(tmp_path, _package(tmp_path, "pass"))
    assert running is not None
    assert running["status"] == "RUNNING"
    assert running["current_step"] == "native full acceptance"
    assert "finished_at_utc" not in running
    assert running["steps"][-1]["status"] == "RUNNING"
    assert code == 0 and receipt["status"] == "PASS_AUTOMATED_ONLY", receipt
    assert receipt["current_step"] is None and receipt["finished_at_utc"]
    assert set(_statuses(receipt).values()) == {"PASS"}
    assert all(step["duration_seconds"] >= 0 for step in receipt["steps"])
    assert receipt["final_release"] is False and receipt["browser_manual_acceptance"] == "NOT_RUN"


def test_failed_suite_is_fail_but_independent_steps_continue(tmp_path: Path) -> None:
    code, receipt, _ = _run(tmp_path, _package(tmp_path, "suite-fails"))
    assert code == 1 and receipt["status"] == "FAIL"
    assert _statuses(receipt) == {
        "native full acceptance": "FAIL",
        "ordinary user install": "PASS",
        "two real research workflows": "PASS",
    }
    assert receipt["steps"][0]["exit_code"] == 7
    assert receipt["failed"] == ["native full acceptance"]
    assert receipt["ordinary_install_ready"] is True


def test_failed_install_blocks_only_its_dependent(tmp_path: Path) -> None:
    code, receipt, _ = _run(tmp_path, _package(tmp_path, "install-fails"))
    assert code == 1 and receipt["status"] == "FAIL"
    assert _statuses(receipt) == {
        "native full acceptance": "PASS",
        "ordinary user install": "FAIL",
        "two real research workflows": "BLOCKED",
    }
    assert receipt["ordinary_install_ready"] is False


def test_program_that_cannot_start_is_its_own_status(tmp_path: Path) -> None:
    code, receipt, _ = _run(tmp_path, _package(tmp_path, "python-missing"))
    assert code == 1 and receipt["status"] == "FAIL"
    research = receipt["steps"][-1]
    assert research["status"] == "CANNOT_START" and research["exit_code"] == -1
    assert research["error"]


def test_verified_reuse_is_recorded_with_exact_output(tmp_path: Path) -> None:
    code, receipt, _ = _run(tmp_path, _package(tmp_path, "pass", installed=True))
    assert code == 0 and receipt["status"] == "PASS_AUTOMATED_ONLY", receipt
    assert _statuses(receipt)["ordinary user install"] == "REUSED_VERIFIED_INSTALL"
    (folder,) = (tmp_path / "包 根").glob("final-run-*")
    if _console_can_carry("中文"):
        assert "中文 x" in (folder / "reuse-check.json").read_text(encoding="utf-8")


def _console_can_carry(text: str) -> bool:
    """A child PowerShell writes with the console output code page; 437 cannot carry Chinese."""
    if not WINDOWS:
        return True
    import ctypes

    page = ctypes.windll.kernel32.GetConsoleOutputCP()  # type: ignore[attr-defined]
    if not page:
        return False
    try:
        text.encode("utf-8" if page == 65001 else f"cp{page}")
    except (LookupError, UnicodeEncodeError):
        return False
    return True


def test_interrupted_run_is_incomplete_not_running_or_failed(tmp_path: Path) -> None:
    code, receipt, running = _run(tmp_path, _package(tmp_path, "interrupt"), interrupt=True)
    assert running is not None and running["status"] == "RUNNING"
    # After Ctrl-C PowerShell stops the script before its own "exit"; the process exit code is
    # PowerShell's (0 here), so the receipt, not the exit code, is the record of the stop.
    del code
    assert receipt["status"] == "INCOMPLETE", receipt
    assert receipt["steps"][0]["status"] == "INTERRUPTED"
    assert receipt["current_step"] is None and receipt["finished_at_utc"]
