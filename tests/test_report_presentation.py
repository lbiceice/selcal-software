"""Report presentation preserves checked results and the complete recorded evidence."""

from __future__ import annotations

import json
from html.parser import HTMLParser

import pytest
from test_workflow import CAP, files

from selcal import workflow
from selcal.workflow_config import encode_workflow_config

FIELDS = (
    "status",
    "selected_candidate",
    "decision_statistic",
    "p_value",
    "reject_null",
    "planned_replicates",
    "retained_replicates",
    "failure_count",
    "failure_stage",
    "exceedance_count",
    "exceedance_bound_low",
    "exceedance_bound_high",
)


def test_saved_pre_u01_report_style_remains_allowed_without_relaxing_csp():
    from selcal.ui import _HEADERS

    # Exact CSS digest of already-saved pre-U01 reports, not an inline-style wildcard.
    policy = _HEADERS["Content-Security-Policy"]
    assert "'sha256-/UEtROarG+pZ6KzBT6Fh5yp27PejSEFa07Oftkmpals='" in policy
    assert "unsafe-inline" not in policy and "unsafe-eval" not in policy
    assert "script-src 'self'" in policy


class Report(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.details = []
        self.fields = {}
        self.visible = []
        self.elements = []
        self.detail = None
        self.field = None
        self.in_summary = False
        self.in_pre = False
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        self.elements.append((tag, attrs))
        if tag == "details":
            assert self.detail is None, "Evidence sections need one clear expansion each"
            self.detail = {"attrs": attrs, "label": "", "pres": [], "replicates": []}
            self.details.append(self.detail)
        if tag == "summary":
            self.in_summary = True
        if tag == "pre" and self.detail is not None:
            self.in_pre = True
            self.detail["pres"].append("")
        if "data-replicate-id" in attrs and self.detail is not None:
            self.detail["replicates"].append(attrs["data-replicate-id"])
        if "data-summary-key" in attrs:
            assert self.detail is None, "Decision and failure fields must remain visible"
            self.field = attrs["data-summary-key"]
            self.fields[self.field] = ""

    def handle_endtag(self, tag):
        if tag == "details":
            self.detail = None
        if tag == "summary":
            self.in_summary = False
        if tag == "pre":
            self.in_pre = False
        if tag == "dd":
            self.field = None

    def handle_data(self, text):
        if self.detail is None:
            self.visible.append(text)
        if self.field is not None:
            self.fields[self.field] += text
        if self.detail is not None:
            if self.in_summary:
                self.detail["label"] += text
            if self.in_pre:
                self.detail["pres"][-1] += text


def saved_report(tmp_path, mode="complete"):
    inp, cfg = files(tmp_path, mode=mode)
    record_path = tmp_path / "record.sqlite"
    workflow.run_files(inp, cfg, record_path, max_bytes=CAP, allow_unattainable=True)
    before = record_path.read_bytes()
    output = tmp_path / "report.html"
    summary = workflow.report_record(record_path, output, max_bytes=CAP)
    assert record_path.read_bytes() == before
    record = workflow.read_workflow(record_path, max_bytes=CAP)
    return record, summary, output.read_bytes()


@pytest.mark.parametrize(
    "mode", ["complete", "null_bind", "observed_statistic_scan", "replicate_execution"]
)
def test_summary_fields_and_failure_scope_remain_visible(tmp_path, mode):
    _, summary, payload = saved_report(tmp_path, mode)
    report = Report(payload.decode())
    assert report.fields == {key: json.dumps(summary[key]) for key in FIELDS}
    visible = " ".join(report.visible)
    assert "Input/plan/result consistency checked; replay NOT_PERFORMED." in visible
    assert "Recorded content is not authenticated historical execution." in visible
    assert "NOT_EVALUABLE is not evidence of no effect or non-significance." in visible
    if mode == "complete":
        assert report.fields["failure_count"] == "0"
        assert report.fields["reject_null"] == "false"
        assert report.fields["failure_stage"] == "null"
    else:
        assert report.fields["status"] == '"not_evaluable"'
        assert report.fields["failure_stage"] == json.dumps(mode)
        assert report.fields["p_value"] == report.fields["reject_null"] == "null"


def test_complete_evidence_is_preserved_in_closed_native_sections(tmp_path):
    record, summary, payload = saved_report(tmp_path)
    report = Report(payload.decode())
    expected = {
        "Summary": [json.dumps(summary, indent=2)],
        "Configuration": [encode_workflow_config(record.config).decode()],
        "Observed candidates": [repr(record.result.observed_results)],
        "Observed selection": [repr(record.result.observed_selection)],
        "Replicates": [repr(outcome) for outcome in record.result.replicates],
        "Diagnostics": [repr(record.result.diagnostics)],
        "Recorded software identity (not authentication)": [
            json.dumps(record.metadata["software"], indent=2)
        ],
    }
    assert [section["label"] for section in report.details] == list(expected)
    for section in report.details:
        assert "open" not in section["attrs"]
        assert [text.strip() for text in section["pres"]] == [
            text.strip() for text in expected[section["label"]]
        ]
    assert report.details[4]["replicates"] == [
        str(outcome.replicate_id) for outcome in record.result.replicates
    ]
    assert any(
        tag == "meta"
        and attrs.get("name") == "viewport"
        and attrs.get("content") == "width=device-width, initial-scale=1"
        for tag, attrs in report.elements
    )


def test_new_summary_values_are_escaped_without_active_content(tmp_path):
    record, summary, _ = saved_report(tmp_path)
    # Rendering-only probe: deliberately hostile text is not a scientific result.
    hostile = '<img src="https://example.invalid/x" onerror="alert(1)">&'
    summary["selected_candidate"] = hostile
    report = Report(workflow._render_report(record, summary, CAP).decode())
    assert report.fields["selected_candidate"] == json.dumps(hostile)
    assert not {"script", "img", "link", "iframe"} & {tag for tag, _ in report.elements}
    assert not any(
        key.startswith("on") or key in {"src", "href"}
        for _, attrs in report.elements
        for key in attrs
    )


def test_report_byte_limit_preserves_exact_boundary(tmp_path):
    record, summary, payload = saved_report(tmp_path)
    assert workflow._render_report(record, summary, len(payload)) == payload
    with pytest.raises(workflow.WorkflowError, match="report_size_limit"):
        workflow._render_report(record, summary, len(payload) - 1)
