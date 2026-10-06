"""Require concrete native Windows JUnit outcomes; fixture parsing is portable."""

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

REQUIRED = (
    ("tests.test_crash_process",
     "test_windows_nested_launcher_terminates_real_writer_before_reaping_launcher"),
    ("tests.test_crash_windows_process", "test_native_retained_identity_wait_and_readonly_refusal"),
    ("tests.test_ui_child_runtime",
     "test_native_child_is_owned_pid_with_same_installed_environment"),
    *(("tests.test_ui_child_runtime",
       "test_native_stop_before_python_initialization_reaps_only_owned_worker[" + case + "]")
      for case in ("cancel-report", "cancel-export", "close-report", "close-export")),
)


def check_required_tests(path):
    """Inspect testcase nodes, never aggregate attributes or a total-count claim."""
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as error:
        raise ValueError(f"Cannot read JUnit evidence: {error}") from error
    if root.tag not in {"testsuite", "testsuites"}:
        raise ValueError("Not a JUnit testsuite report")
    cases = list(root.iter("testcase"))
    keys = [(case.get("classname"), case.get("name")) for case in cases]
    counts = Counter(keys)
    if not cases or any(not all(key) or count != 1 for key, count in counts.items()):
        raise ValueError("Empty report, missing testcase identity, or duplicate testcase")
    if any(node.tag in {"failure", "error"} for node in root.iter()):
        raise ValueError("JUnit contains a failure or error")
    for suite in root.iter():
        if suite.tag not in {"testsuite", "testsuites"}:
            continue
        children = list(suite.iter("testcase"))
        actual = {"tests": len(children), "failures": 0, "errors": 0,
                  "skipped": sum(case.find("skipped") is not None for case in children)}
        for name, count in actual.items():
            supplied = suite.get(name)
            if supplied is not None and (not supplied.isdecimal() or int(supplied) != count):
                raise ValueError(f"Contradictory JUnit {name}: {supplied}; actual {count}")
    by_key = dict(zip(keys, cases, strict=True))
    missing = ["::".join(key) for key in REQUIRED if key not in by_key]
    skipped = ["::".join(key) for key in REQUIRED
               if key in by_key and by_key[key].find("skipped") is not None]
    if missing or skipped:
        raise ValueError(f"Required native outcomes missing={missing}, skipped={skipped}")
    skip_count = sum(case.find("skipped") is not None for case in cases)
    return {"status": "PASS", "testcases": len(cases), "passed": len(cases) - skip_count,
            "skipped": skip_count, "required_passed": len(REQUIRED),
            "required_tests": ["::".join(key) for key in REQUIRED]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("junit", type=Path)
    args = parser.parse_args()
    try:
        if sys.platform != "win32":
            raise ValueError(
                "This CLI requires native Windows; portable fixture parsing is not evidence")
        result = check_required_tests(args.junit)
    except ValueError as error:
        print(json.dumps({"status": "FAIL", "error": str(error)}))
        return 1
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
