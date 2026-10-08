"""R18 Windows item 1: every test keeps its identity in JUnit, also in a release source tree.

A module-level pytest.skip produced a JUnit record with an empty classname; the native Windows gate
(tests/_windows_acceptance.py) refuses any record without an identity, so the whole run failed.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULES = ("test_final_check_states.py", "test_windows_native_capture.py")


def test_no_test_module_skips_at_collection_time() -> None:
    offenders = [
        path.name
        for path in ROOT.glob("tests/**/*.py")
        if path.name != Path(__file__).name
        and re.search(r"allow_module_level\s*=\s*True", path.read_text(encoding="utf-8"))
    ]
    assert offenders == []


def test_release_tree_without_repository_files_keeps_every_identity(tmp_path: Path) -> None:
    tree = tmp_path / "release"
    (tree / "tests").mkdir(parents=True)
    for name in MODULES:  # no docs/status, no scripts/*.ps1, no tests/ps: the release layout
        shutil.copyfile(ROOT / "tests" / name, tree / "tests" / name)
    report = tmp_path / "junit.xml"
    done = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rs",
         f"--junitxml={report}", "tests"],
        cwd=tree, capture_output=True, text=True, errors="replace", timeout=300,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    cases = list(ET.parse(report).getroot().iter("testcase"))
    assert len(cases) >= 10
    assert all(case.get("classname") and case.get("name") for case in cases), [
        (case.get("classname"), case.get("name")) for case in cases
    ]
    assert all(case.find("skipped") is not None for case in cases)
    sys.path.insert(0, str(ROOT / "tests"))
    try:
        import _windows_acceptance as gate
    finally:
        sys.path.pop(0)
    keys = [(case.get("classname"), case.get("name")) for case in cases]
    assert all(all(key) for key in keys) and gate.REQUIRED  # the gate's identity rule holds
