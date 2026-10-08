"""R17 Windows item 13: native output keeps every character (scripts/windows_native_capture.ps1).

Natively on Windows the check runs in Windows PowerShell 5.1 with the registry ANSI code page and
Python as the child. Elsewhere it runs only when a Linux PowerShell image is opted in
(SELCAL_POWERSHELL_DOCKER_IMAGE); that is not Windows evidence.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "scripts" / "windows_native_capture.ps1"
CHECK = ROOT / "tests" / "ps" / "native_capture_check.ps1"


def _block(text: str) -> str:
    start = text.index("# BEGIN native-capture")
    end = text.index("# END native-capture") + len("# END native-capture")
    return text[start:end]


@pytest.mark.parametrize(
    "name",
    [
        "p3_windows_round3/INSTALL_WINDOWS.ps1",
        "p3_windows_round3/RUN_EXAMPLE.ps1",
        "windows_outer/FINAL_CHECK.ps1",
    ],
)
def test_standalone_scripts_carry_the_canonical_capture_block(name: str) -> None:
    path = ROOT / "docs" / "status" / name
    if not path.is_file() or not CANONICAL.is_file():
        pytest.skip("Windows entry scripts are repository-only")
    canonical = _block(CANONICAL.read_text(encoding="utf-8"))
    assert _block(path.read_text(encoding="utf-8")) == canonical


def test_windows_check_uses_the_capture_helper_not_console_decoding() -> None:
    path = ROOT / "scripts" / "windows_check.ps1"
    if not path.is_file():
        pytest.skip("Windows runner is in the repository and the Windows source package only")
    runner = path.read_text(encoding="utf-8")
    assert CANONICAL.is_file(), "windows_check.ps1 dot-sources the capture helper"
    assert '. (Join-Path $PSScriptRoot "windows_native_capture.ps1")' in runner
    assert "Invoke-NativeCapture" in runner
    assert "& $Exe @Arguments 2>&1" not in runner


def test_capture_helper_keeps_every_character() -> None:
    if not CHECK.is_file() or not CANONICAL.is_file():
        pytest.skip("PowerShell checks are in the repository and the Windows source package only")
    if sys.platform == "win32":
        shell = shutil.which("powershell") or shutil.which("pwsh")
        if shell is None:
            pytest.fail("Windows PowerShell is required for the native capture check")
        command = [shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(CHECK),
                   "-Python", sys.executable]
    else:
        image = os.environ.get("SELCAL_POWERSHELL_DOCKER_IMAGE")
        if image is None:
            pytest.skip("Linux PowerShell image not opted in; native Windows run checks this")
        command = ["docker", "run", "--rm", "--pull", "never", "--network", "none", "--mount",
                   f"type=bind,src={ROOT},dst=/repo,readonly", image, "pwsh", "-NoProfile",
                   "-File", "/repo/tests/ps/native_capture_check.ps1"]
    done = subprocess.run(command, capture_output=True, timeout=300)
    output = done.stdout.decode("utf-8", "backslashreplace")
    assert done.returncode == 0 and "NATIVE_CAPTURE_OK" in output, output + done.stderr.decode(
        "utf-8", "backslashreplace"
    )


def test_every_shipped_powershell_script_is_ascii() -> None:
    """Windows PowerShell 5.1 reads a .ps1 without BOM in the ANSI code page (cp936 here)."""
    scripts = [
        *ROOT.glob("scripts/*.ps1"),
        *ROOT.glob("tests/**/*.ps1"),
        *ROOT.glob("docs/status/p3_windows_round3/*.ps1"),
        *ROOT.glob("docs/status/windows_outer/**/*.ps1"),
    ]
    if not scripts:
        pytest.skip("this source tree ships no PowerShell scripts")
    for path in scripts:
        assert path.read_bytes().isascii(), path


AUTOMATIC = ("OutputEncoding", "Input", "Args", "Error", "Host", "PID", "PWD", "Home", "Profile",
             "PSScriptRoot", "PSCommandPath", "LastExitCode", "ExecutionContext", "Event", "This",
             "Sender", "ShellId", "StackTrace", "MyInvocation", "PSItem", "_", "IsWindows")


def test_no_shipped_script_assigns_a_powershell_automatic_variable() -> None:
    """Variable names are case-insensitive: "$outputEncoding = @(...)" fails on Windows."""
    pattern = re.compile(
        r"\$(?:script:|global:|local:)?(" + "|".join(AUTOMATIC) + r")\s*(?:\[[^\]]*\])?\s*=(?!=)",
        re.IGNORECASE,
    )
    allowed = re.compile(r"\$global:LASTEXITCODE\s*=", re.IGNORECASE)  # deliberate exit code
    scripts = [
        *ROOT.glob("scripts/*.ps1"),
        *ROOT.glob("tests/**/*.ps1"),
        *ROOT.glob("docs/status/p3_windows_round3/*.ps1"),
        *ROOT.glob("docs/status/windows_outer/**/*.ps1"),
    ]
    if not scripts:
        pytest.skip("this source tree ships no PowerShell scripts")
    found = [
        f"{path.name}:{number}: {line.strip()}"
        for path in scripts
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if pattern.search(line) and not allowed.search(line)
    ]
    assert found == []
