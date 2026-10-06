"""JUnit evidence gating fixtures; these are not native Windows execution evidence."""

import importlib.util
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

CHECKER = Path(__file__).with_name("_windows_acceptance.py")
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


def checker():
    assert CHECKER.is_file(), "native evidence checker implementation is missing"
    spec = importlib.util.spec_from_file_location("native_acceptance_checker", CHECKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def report(tmp_path, keys=REQUIRED):
    root = ET.Element("testsuites")
    suite = ET.SubElement(root, "testsuite")
    for classname, name in keys:
        ET.SubElement(suite, "testcase", classname=classname, name=name)
    path = tmp_path / "pytest.xml"
    ET.ElementTree(root).write(path, encoding="utf-8")
    return path


@pytest.mark.parametrize("include_counts", [False, True])
def test_exact_keys_pass_from_testcase_nodes(tmp_path, include_counts):
    path = report(tmp_path)
    if include_counts:
        tree = ET.parse(path)
        for suite in (tree.getroot(), tree.find("testsuite")):
            suite.attrib.update(tests="7", failures="0", errors="0", skipped="0")
        tree.write(path)
    result = checker().check_required_tests(path)
    assert result["status"] == "PASS"
    assert result["required_passed"] == 7
    assert result["testcases"] == 7


@pytest.mark.parametrize("level", ["root", "suite"])
@pytest.mark.parametrize("attribute,value", [
    ("tests", "99"), ("failures", "99"), ("errors", "99"), ("skipped", "99"),
    ("tests", "not-a-number"), ("tests", "-1"),
])
def test_aggregate_counts_cannot_contradict_testcase_nodes(tmp_path, level, attribute, value):
    module = checker()
    path = report(tmp_path)
    tree = ET.parse(path)
    parent = tree.getroot() if level == "root" else tree.find("testsuite")
    parent.set(attribute, value)
    tree.write(path)
    with pytest.raises(ValueError):
        module.check_required_tests(path)


@pytest.mark.parametrize("index", range(7))
def test_each_required_key_must_be_present(tmp_path, index):
    module = checker()
    with pytest.raises(ValueError):
        module.check_required_tests(report(tmp_path, REQUIRED[:index] + REQUIRED[index + 1:]))


@pytest.mark.parametrize("change", ["wrong-module", "duplicate", "skipped", "failure", "error"])
def test_required_nodes_must_be_unique_and_pass(tmp_path, change):
    module = checker()
    path = report(tmp_path)
    tree = ET.parse(path)
    first = tree.find(".//testcase")
    if change == "wrong-module":
        first.set("classname", "unrelated.test_crash_process")
    elif change == "duplicate":
        ET.SubElement(tree.find(".//testsuite"), "testcase", **first.attrib)
    else:
        ET.SubElement(first, change)
    tree.write(path)
    with pytest.raises(ValueError):
        module.check_required_tests(path)


@pytest.mark.parametrize("outcome", ["failure", "error"])
def test_unrelated_failures_cannot_be_hidden_by_required_passes(tmp_path, outcome):
    module = checker()
    path = report(tmp_path, (*REQUIRED, ("tests.other", "test_other")))
    tree = ET.parse(path)
    ET.SubElement(tree.findall(".//testcase")[-1], outcome)
    tree.write(path)
    with pytest.raises(ValueError):
        module.check_required_tests(path)


def test_unrelated_skip_is_reported_without_weakening_required_keys(tmp_path):
    module = checker()
    path = report(tmp_path, (*REQUIRED, ("tests.other", "test_posix_only")))
    tree = ET.parse(path)
    ET.SubElement(tree.findall(".//testcase")[-1], "skipped")
    tree.write(path)
    result = module.check_required_tests(path)
    assert result["required_passed"] == 7 and result["testcases"] == 8
    assert result["skipped"] == 1


@pytest.mark.parametrize("raw", [b"", b"not xml", b"<testsuite>", b"<testsuite/>", b"<other/>"])
def test_malformed_and_empty_reports_fail(tmp_path, raw):
    module = checker()
    path = tmp_path / "bad.xml"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        module.check_required_tests(path)


def test_missing_report_fails(tmp_path):
    module = checker()
    with pytest.raises(ValueError):
        module.check_required_tests(tmp_path / "absent.xml")


def test_cli_does_not_turn_fixture_xml_into_nonwindows_native_evidence(tmp_path):
    checker()
    path = report(tmp_path)
    result = subprocess.run([sys.executable, "-I", str(CHECKER), str(path)],
                            text=True, errors="replace", capture_output=True, check=False)
    payload = json.loads(result.stdout)
    if sys.platform == "win32":
        assert result.returncode == 0 and payload["status"] == "PASS"
    else:
        assert result.returncode != 0 and payload["status"] == "FAIL"
        assert "native Windows" in payload["error"]
