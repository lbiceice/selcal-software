"""Require concrete native Windows JUnit outcomes; fixture parsing is portable."""

import argparse
import json
import sys
import xml.etree.ElementTree as ET
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


class GateFailure(ValueError):
    """A failed gate, with the required-node coverage and the whole-suite result kept apart."""

    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details or {}


# pytest writes a second testcase entry with the same identity when a test that already has a
# call outcome also errors in setup or teardown; only that form is an expected repeat.
_PHASE_ERROR_PREFIXES = ("failed on setup with", "failed on teardown with")


def _is_phase_error(case):
    error = case.find("error")
    return (error is not None and case.find("failure") is None
            and (error.get("message") or "").startswith(_PHASE_ERROR_PREFIXES))


def check_required_tests(path):
    """Inspect testcase nodes, never aggregate attributes or a total-count claim.

    Passes only if every testcase passed or was skipped and the seven required native nodes
    passed. A failure states the required-node coverage and the suite result separately.
    """
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as error:
        raise ValueError(f"Cannot read JUnit evidence: {error}") from error
    if root.tag not in {"testsuite", "testsuites"}:
        raise ValueError("Not a JUnit testsuite report")
    cases = list(root.iter("testcase"))
    keys = [(case.get("classname"), case.get("name")) for case in cases]
    if not cases or not all(all(key) for key in keys):
        raise ValueError("Empty report or missing testcase identity")
    entries = {}
    for key, case in zip(keys, cases, strict=True):
        entries.setdefault(key, []).append(case)
    unexpected = ["::".join(key) for key, group in entries.items()
                  if len(group) > 1 and sum(not _is_phase_error(case) for case in group) != 1]
    if unexpected:
        raise ValueError(f"Unexpected duplicate testcase: {unexpected[:5]}")
    failed = sorted("::".join(key) for key, group in entries.items()
                    if any(case.find("failure") is not None for case in group))
    errored = sorted("::".join(key) for key, group in entries.items()
                     if any(case.find("error") is not None for case in group))
    required_failed = ["::".join(key) for key in REQUIRED if key in entries and any(
        case.find("failure") is not None or case.find("error") is not None
        or case.find("skipped") is not None for case in entries[key])]
    missing = ["::".join(key) for key in REQUIRED if key not in entries]
    if failed or errored:
        details = {
            "required": {"passed": len(REQUIRED) - len(missing) - len(required_failed),
                         "of": len(REQUIRED), "missing": missing,
                         "failed_errored_or_skipped": required_failed},
            "suite": {"tests": len(entries), "failed": len(failed), "errored": len(errored),
                      "failed_tests": failed[:20], "errored_tests": errored[:20]},
        }
        raise GateFailure(
            f"Suite not clean: {len(failed)} tests failed and {len(errored)} had errors "
            f"(of {len(entries)}); required native nodes passed "
            f"{details['required']['passed']}/{len(REQUIRED)}", details)
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
    by_key = {key: group[0] for key, group in entries.items()}
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
        print(json.dumps({"status": "FAIL", "error": str(error),
                          **getattr(error, "details", {})}))
        return 1
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
