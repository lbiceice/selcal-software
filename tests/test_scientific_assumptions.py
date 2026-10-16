"""Visible scientific scope is guidance, never an empirical validity certificate.

These checks exercise actual validation, report and browser rendering paths. They check
the interpretation contract; passing them does not test the distributional assumptions.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from test_report_presentation import Report
from test_workflow_preflight import write

from selcal import workflow

ROOT = Path(__file__).resolve().parents[1]
CAP = 1_048_576
COMBINATIONS = [
    ("lagged_pearson_v1", "circular_shift_exact_v1"),
    ("lagged_pearson_v1", "circular_shift_v2"),
    ("lagged_pearson_v1", "block_shuffle_v2"),
    ("equal_width_binned_nette_v1", "circular_shift_v2"),
]


def plan_files(tmp_path, statistic, null):
    data, config = write(
        tmp_path, n=64, lags=[1, 2],
        replicates=63 if null == "circular_shift_exact_v1" else 19, null=null,
    )
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["plan"]["statistic_name"] = statistic
    payload["plan"]["statistic_params"] = (
        {"bins": 3} if statistic == "equal_width_binned_nette_v1" else {}
    )
    if null == "block_shuffle_v2":
        payload["plan"]["null_params"] = {"block_length": 8}
    config.write_text(json.dumps(payload), encoding="utf-8")
    return data, config


@pytest.mark.parametrize("null", ["circular_shift_exact_v1", "circular_shift_v2"])
def test_preflight_distinguishes_group_enumeration_from_error_rate_control(tmp_path, null):
    data, config = plan_files(tmp_path, "lagged_pearson_v1", null)
    checked = workflow.validate_files(data, config)["preflight"]
    assert checked["plan"]["status"] == "EXECUTABLE"
    science = checked["scientific_assumptions"]
    assert science["status"] == "DECLARED_NOT_VERIFIED"
    statement = next(
        row["statement"] for row in science["assumptions"]
        if row["code"] == "circular_shift_exchangeability"
    )
    assert "joint distribution of (X, Y)" in statement
    assert "invariant under every circular shift of X while Y is held fixed" in statement
    assert "Complete enumeration traverses the reference distribution" in statement
    assert "does not establish error-rate control" in statement
    assert "Stationarity or matching endpoints alone is not sufficient" in statement
    assert "exact for circular series" not in statement
    assert "approximate for stationary series" not in statement


@pytest.mark.parametrize(
    ("statistic", "null", "warning"),
    [
        ("equal_width_binned_nette_v1", "circular_shift_v2", "not been statistically validated"),
        ("lagged_pearson_v1", "block_shuffle_v2", "did not meet the study's adoption criterion"),
    ],
)
def test_experimental_adapter_warning_reaches_cli_validate(tmp_path, statistic, null, warning):
    data, config = plan_files(tmp_path, statistic, null)
    direct = workflow.validate_files(data, config)["preflight"]["scientific_assumptions"]
    command = [sys.executable, "-m", "selcal", "validate", str(data), str(config)]
    done = subprocess.run(
        command, capture_output=True, text=True, encoding="utf-8", timeout=30
    )
    assert done.returncode == 0, done.stderr
    delivered = json.loads(done.stdout)["data"]["preflight"]["scientific_assumptions"]
    assert delivered == direct
    assert delivered["status"] == "DECLARED_NOT_VERIFIED"
    assert warning in " ".join(row["statement"] for row in delivered["assumptions"])


@pytest.mark.parametrize(("statistic", "null"), COMBINATIONS)
def test_report_displays_the_same_declared_scope_without_changing_the_record(
    tmp_path, statistic, null
):
    data, config = plan_files(tmp_path, statistic, null)
    science = workflow.validate_files(data, config)["preflight"]["scientific_assumptions"]
    record_path, report_path = tmp_path / "record.sqlite", tmp_path / "report.html"
    workflow.run_files(data, config, record_path, max_bytes=CAP)
    before = record_path.read_bytes()
    summary = workflow.report_record(record_path, report_path, max_bytes=CAP)
    report = Report(report_path.read_text(encoding="utf-8"))
    visible = " ".join(report.visible)
    assert "Scientific assumptions: DECLARED_NOT_VERIFIED" in visible
    assert "Current guidance for the recorded plan" in visible
    assert science["note"] in visible
    for row in science["assumptions"]:
        assert row["statement"] in visible
    assert record_path.read_bytes() == before
    assert "scientific_assumptions" not in summary, "Result output schema must stay unchanged"
    assert workflow.verify_record(record_path, max_bytes=CAP, replay=True)["replay"] == "MATCH"


def test_ui_scientific_guidance_is_visible_without_opening_terminal_details():
    html = (ROOT / "src/selcal/web/index.html").read_text(encoding="utf-8")
    parsed = Report(html)
    assert any(
        attrs.get("id") == "scientific-assumptions" and "hidden" not in attrs
        for _, attrs in parsed.elements
    ), "The existing result area needs a visible scientific-assumptions container"
    visible = " ".join(parsed.visible)
    assert "joint distribution of (X, Y)" in visible
    assert "not been statistically validated" in visible
    assert "did not meet the study's adoption criterion" in visible


def test_actual_javascript_renders_validate_assumptions_and_clears_stale_guidance(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("JavaScript runtime unavailable; browser acceptance remains separate")
    data, config = plan_files(tmp_path, "equal_width_binned_nette_v1", "circular_shift_v2")
    validation = workflow.validate_files(data, config)
    # Only the DOM and transport are small substitutes. The actual app.js consumes a
    # complete, freshly generated validation payload; no scientific service is mocked.
    harness = r"""
const fs = require('node:fs'), vm = require('node:vm'), assert = require('node:assert/strict');
const elements = new Map();
function element(id) {
  if (!elements.has(id)) elements.set(id, {value:'', checked:false, disabled:false,
    hidden:false, options:[], files:[], handlers:{}, textContent:'',
    classList:{toggle(){}}, addEventListener(k,f){this.handlers[k]=f},
    replaceChildren(...children){this.options=[...children];this.textContent=''},
    append(...children){this.options.push(...children)}});
  return elements.get(id);
}
const rendered = e => e.textContent + e.options.map(rendered).join(' ');
const context = vm.createContext({URLSearchParams, Blob, Object, JSON, console,
  location:{hash:'#token=synthetic', pathname:'/'}, history:{replaceState(){}},
  sessionStorage:{setItem(){},getItem(){return 'synthetic'}}, setInterval(){},
  document:{getElementById:element,createElement(){return element(Symbol())}},
  fetch:async()=>({ok:true,json:async()=>({jobs:[]})})});
context.checked = {id:'checked', state:'ready', input_status:'complete', operation:'validate',
  result:{data:PAYLOAD}, record_target:{available:false}};
vm.runInContext(fs.readFileSync(SCRIPT,'utf8'),context);
(async()=>{
  await new Promise(setImmediate);
  vm.runInContext('selectedId=viewedId="checked";showJob(checked)',context);
  const expected = context.checked.result.data.preflight.scientific_assumptions;
  const output = rendered(element('scientific-assumptions'));
  assert(output.includes(expected.status),'The scientific status must be visible');
  assert(output.includes(expected.note),'The analyst justification boundary must be visible');
  for (const row of expected.assumptions) assert(output.includes(row.statement),row.code);
  vm.runInContext('invalidate()',context);
  const cleared = rendered(element('scientific-assumptions'));
  assert(!cleared.includes(expected.assumptions[0].statement),
    'Edited plans must not retain the previous validation guidance');
  assert(cleared.includes('Validate'),'The empty state must direct the user to Validate');
  context.updated = {...context.checked, progress:'Updated in another window'};
  context.fetch = async()=>({ok:true,json:async()=>({jobs:[context.updated]})});
  await vm.runInContext('refreshJobs()',context);
  assert.equal(vm.runInContext('selectedId',context),null);
  assert(!rendered(element('scientific-assumptions')).includes(expected.assumptions[0].statement),
    'A background update must not restore guidance from the old plan after an edit');
  vm.runInContext('selectedId="checked";showJob(updated)',context);
  assert(rendered(element('scientific-assumptions')).includes(expected.assumptions[0].statement),
    'Explicitly reselecting the saved job must restore its own declared guidance');
})().catch(error=>{console.error(error);process.exitCode=1});
"""
    harness = harness.replace("PAYLOAD", json.dumps(validation))
    harness = harness.replace("SCRIPT", json.dumps(str(ROOT / "src/selcal/web/app.js")))
    done = subprocess.run(
        [node, "-e", harness], capture_output=True, text=True, encoding="utf-8", timeout=30
    )
    assert done.returncode == 0, done.stderr
