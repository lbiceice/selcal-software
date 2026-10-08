from __future__ import annotations

import csv
import hashlib
import importlib
import importlib.util
import io
import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest
from _platform_support import directory_link_or_skip, symlink_or_skip
from test_calibration_v2_progress import _case
from test_workflow import CAP, files

from selcal import workflow
from selcal.canonical import canonical_json_bytes
from selcal.canonical_v2 import scientific_plan_v2_payload
from selcal.workflow_config import WorkflowConfig, encode_workflow_config
from selcal.workflow_store import read_record, write_record

SUMMARY = (
    "status,failure_stage,alpha,planned_replicates,retained_replicates,exceedance_count,"
    "failure_count,p_value,selected_candidate,decision_statistic,tied_candidates,reject_null,"
    "exceedance_bound_low,exceedance_bound_high,semantic_input_sha256,scientific_plan_sha256,"
    "diagnostics"
).split(",")
CANDIDATES = (
    "scope,replicate_id,candidate_id,estimate,selection_score,support_n,validity,diagnostics,"
    "backend_identity,preprocessing_identity"
).split(",")
REPLICATES = (
    "replicate_id,seed_digest_sha256,status,failure_stage,selected_candidate,selected_index,"
    "decision_statistic,tied_candidates,transform_token,diagnostics"
).split(",")
DERIVED = (
    "plan.json",
    "software.json",
    "summary.csv",
    "candidates.csv",
    "replicates.csv",
    "report.html",
)


def export_api():
    assert importlib.util.find_spec("selcal.workflow_export") is not None, "missing export service"
    return importlib.import_module("selcal.workflow_export")


def make_record(tmp_path, *, form="csv", mode="complete"):
    inp, cfg = files(tmp_path, form=form, mode=mode)
    record = tmp_path / "run.sqlite"
    result = workflow.run_files(inp, cfg, record, max_bytes=CAP, allow_unattainable=True)
    return record, result, read_record(record, max_bytes=CAP)


def bundle(tmp_path):
    api = export_api()
    record, result, members = make_record(tmp_path)
    output = tmp_path / "bundle"
    summary = api.export_record(record, output, max_bytes=CAP)
    return api, output, summary, result, members


def rows(output, name, columns):
    raw = (output / name).read_bytes()
    assert b"\r\n" not in raw
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8"), newline=""))
    assert reader.fieldnames == columns
    return list(reader)


def exact_float(cell, value):
    assert cell == "" if value is None else float(cell).hex() == value.hex()


def rehash(output, name, payload):
    (output / name).write_bytes(payload)
    manifest = json.loads((output / "manifest.json").read_bytes())
    manifest["members"][name] = {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def assert_projection(output, result, saved):
    summary = rows(output, "summary.csv", SUMMARY)
    assert len(summary) == 1
    row = summary[0]
    expected = workflow.result_summary(result)
    for field in SUMMARY:
        value = (
            result.alpha
            if field == "alpha"
            else (list(result.diagnostics) if field == "diagnostics" else expected[field])
        )
        if value is None:
            assert row[field] == ""
        elif type(value) is float:
            exact_float(row[field], value)
        elif type(value) is bool:
            assert row[field] == str(value).lower()
        elif type(value) is list:
            assert json.loads(row[field]) == value
        else:
            assert row[field] == str(value)
    candidates = rows(output, "candidates.csv", CANDIDATES)
    actual = [("observed", None, item) for item in result.observed_results]
    actual += [
        ("replicate", rep.replicate_id, item)
        for rep in result.replicates
        for item in rep.statistic_results
    ]
    assert len(candidates) == len(actual)
    for row, (scope, rid, item) in zip(candidates, actual, strict=True):
        assert row["scope"] == scope
        assert row["replicate_id"] == ("" if rid is None else str(rid))
        assert int(row["candidate_id"]) == item.candidate_id
        exact_float(row["estimate"], item.estimate)
        exact_float(row["selection_score"], item.selection_score)
        assert int(row["support_n"]) == item.support_n
        assert row["validity"] == item.validity.value
        assert json.loads(row["diagnostics"]) == list(item.diagnostics)
        assert row["backend_identity"] == item.backend_identity
        assert row["preprocessing_identity"] == item.preprocessing_identity
    replicates = rows(output, "replicates.csv", REPLICATES)
    wire = json.loads(saved["result"])
    assert len(replicates) == len(result.replicates)
    for row, rep, raw in zip(replicates, result.replicates, wire["replicates"], strict=True):
        assert int(row["replicate_id"]) == rep.replicate_id
        assert row["seed_digest_sha256"] == rep.seed_digest_sha256
        assert row["status"] == rep.status.value
        assert row["failure_stage"] == (
            "" if rep.failure_stage is None else rep.failure_stage.value
        )
        assert json.loads(row["transform_token"]) == raw["transform_token"]
        assert json.loads(row["diagnostics"]) == list(rep.diagnostics)
        selection = rep.selection
        for field in (
            "selected_candidate",
            "selected_index",
            "decision_statistic",
            "tied_candidates",
        ):
            value = None if selection is None else getattr(selection, field)
            if value is None:
                assert row[field] == ""
            elif type(value) is float:
                exact_float(row[field], value)
            elif type(value) is tuple:
                assert json.loads(row[field]) == list(value)
            else:
                assert row[field] == str(value)


@pytest.mark.parametrize("form", ["csv", "npz"])
@pytest.mark.parametrize(
    "mode", ["complete", "null_bind", "observed_statistic_scan", "replicate_execution"]
)
def test_bundle_preserves_exact_checked_record(tmp_path, form, mode):
    api = export_api()
    record, result, saved = make_record(tmp_path, form=form, mode=mode)
    before = record.read_bytes()
    output = tmp_path / "证据 bundle"
    summary = api.export_record(record, output, max_bytes=CAP)
    names = {"input." + form, "request.json", "result.json", "metadata.json", *DERIVED}
    assert {path.name for path in output.iterdir()} == names | {"manifest.json"}
    assert (output / ("input." + form)).read_bytes() == saved["input"]
    for name in ("request", "result", "metadata"):
        assert (output / (name + ".json")).read_bytes() == saved[name]
    assert record.read_bytes() == before
    assert summary["status"] == result.status.value
    assert summary["member_count"] == 11
    assert summary["bundle_schema"] == "selcal.evidence-bundle.v1"
    assert summary["replay"] == "NOT_PERFORMED"
    assert summary["historical_execution_authenticated"] is False
    assert "attainability" not in summary
    assert api.verify_export(output, max_bytes=CAP) == summary
    manifest = json.loads((output / "manifest.json").read_bytes())
    assert set(manifest) == {
        "schema",
        "members",
        "raw_input_sha256",
        "semantic_input_sha256",
        "scientific_plan_sha256",
        "replay",
        "historical_execution_authenticated",
    }
    assert set(manifest["members"]) == names
    for name, descriptor in manifest["members"].items():
        payload = (output / name).read_bytes()
        assert descriptor == {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    assert (
        hashlib.sha256((output / "plan.json").read_bytes()).hexdigest()
        == result.scientific_plan_sha256
    )
    resolution = workflow.resolve_plan_v2(
        workflow.read_workflow(record, max_bytes=CAP).config.request
    )
    assert (output / "plan.json").read_bytes() == canonical_json_bytes(
        scientific_plan_v2_payload(resolution.plan)
    )
    assert not (output / "plan.json").read_bytes().endswith(b"\n")
    assert_projection(output, result, saved)


@pytest.mark.parametrize("case", ["sampled_pearson", "exact_pearson", "block_pearson", "binned"])
def test_registered_plans_keep_complete_transform_state(tmp_path, case):
    api = export_api()
    pair, request = _case(case)
    inp, cfg, record, output = [
        tmp_path / name for name in ("input.npz", "config", "record", "bundle")
    ]
    np.savez(inp, source=pair.source, target=pair.target)
    cfg.write_bytes(encode_workflow_config(WorkflowConfig(request, "npz")))
    result = workflow.run_files(inp, cfg, record, max_bytes=CAP, allow_unattainable=True)
    saved = read_record(record, max_bytes=CAP)
    api.export_record(record, output, max_bytes=CAP)
    assert_projection(output, result, saved)
    assert api.verify_export(output, max_bytes=CAP)["status"] == result.status.value


def test_single_snapshot_and_no_scientific_or_environment_recomputation(tmp_path, monkeypatch):
    api = export_api()
    record, result, saved = make_record(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    replacement, _, _ = make_record(other, mode="null_bind")
    calls = []
    original = api.read_record

    def read_once(*args, **kwargs):
        calls.append(args[0])
        captured = original(*args, **kwargs)
        record.write_bytes(replacement.read_bytes())
        return captured

    def forbidden(*args, **kwargs):
        pytest.fail("export/verify-export must not recompute or inspect exporter software")

    monkeypatch.setattr(api, "read_record", read_once)
    for name in (
        "calibrate_selected_family",
        "attainability",
        "verify_record",
        "report_record",
        "_software_identity",
    ):
        monkeypatch.setattr(workflow, name, forbidden)
    output = tmp_path / "bundle"
    api.export_record(record, output, max_bytes=CAP)
    assert calls == [record]
    assert (output / "result.json").read_bytes() == saved["result"]
    monkeypatch.setattr(workflow, "read_record", forbidden)
    monkeypatch.setattr(api, "read_record", forbidden)
    monkeypatch.setattr(sqlite3, "connect", forbidden)
    assert api.verify_export(output, max_bytes=CAP)["status"] == result.status.value
    assert_projection(output, result, saved)


def test_verifier_reads_each_member_once_and_never_writes(tmp_path, monkeypatch):
    api, output, summary, _, _ = bundle(tmp_path)
    original = api.read_regular_file_snapshot
    calls = []
    before = {p.name: p.read_bytes() for p in output.iterdir()}

    def counted(path, **kwargs):
        calls.append(Path(path).name)
        return original(path, **kwargs)

    monkeypatch.setattr(api, "read_regular_file_snapshot", counted)
    assert api.verify_export(output, max_bytes=CAP) == summary
    assert sorted(calls) == sorted(before)
    assert {p.name: p.read_bytes() for p in output.iterdir()} == before


def test_recorded_v1_software_and_html_escape(tmp_path, monkeypatch):
    api = export_api()
    record, _, saved = make_record(tmp_path)
    metadata = json.loads(saved["metadata"])
    metadata["schema"] = "selcal.workflow-record.v1"
    del metadata["software"]["platform"]
    metadata["software"]["numpy_version"] = '<script>&" 中文</script>'
    saved["metadata"] = json.dumps(metadata).encode()
    altered = tmp_path / "old.sqlite"
    write_record(altered, saved, max_bytes=CAP)
    monkeypatch.setattr(
        workflow, "_software_identity", lambda: pytest.fail("not exporter identity")
    )
    output = tmp_path / "bundle"
    api.export_record(altered, output, max_bytes=CAP)
    assert (output / "software.json").read_bytes() == canonical_json_bytes(metadata["software"])
    assert "platform" not in json.loads((output / "software.json").read_bytes())
    html = (output / "report.html").read_text(encoding="utf-8")
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert api.verify_export(output, max_bytes=CAP)["replay"] == "NOT_PERFORMED"
    assert record.is_file()


@pytest.mark.parametrize("prefix", ["=", "+", "-", "@", "\t", "\r"])
def test_csv_text_is_neutralized_without_changing_numeric_sign(prefix):
    api = export_api()
    assert api._csv_cell(prefix + "text") == "'" + prefix + "text"
    assert api._csv_cell(-0.0) == "-0.0"
    assert api._csv_cell(-1.25) == "-1.25"
    assert api._csv_cell(False) == "false"
    assert api._csv_cell(None) == ""
    assert api._csv_cell([prefix + "text"]) == json.dumps([prefix + "text"], separators=(",", ":"))


@pytest.mark.parametrize(
    "text",
    [
        "\rformula",
        "\tformula",
        "=SUM(1,2)",
        '中<&"\nline',
        "inside\rreturn",
        "trailing\r",
        "two\r\nlines",
        "bare\nline",
        "comma,value",
        'double"quote',
        '混合,\r\n"字\r尾\n',
        "-1",
        "+1",
    ],
)
def test_csv_round_trip_preserves_embedded_delimiters_and_safe_text(text):
    api = export_api()
    raw = api._csv(("text", "number"), [{"text": text, "number": -0.0}], CAP)
    parsed = list(csv.DictReader(io.StringIO(raw.decode(), newline="")))
    assert parsed == [{"text": api._csv_cell(text), "number": "-0.0"}]
    assert raw.endswith(b"\n") and not raw.endswith(b"\r\n")


def test_csv_multiline_utf8_budget_is_exact_and_keeps_signed_numbers():
    api = export_api()
    source = [
        {"text": '混合\r\n,\r"字\n', "number": -0.0},
        {"text": "-1", "number": -1.25},
        {"text": "+1", "number": 0.0},
    ]
    raw = api._csv(("text", "number"), source, CAP)
    assert list(csv.DictReader(io.StringIO(raw.decode(), newline=""))) == [
        {"text": api._csv_cell(row["text"]), "number": repr(row["number"])} for row in source
    ]
    assert api._csv(("text", "number"), source, len(raw)) == raw
    with pytest.raises(api.ExportError) as refused:
        api._csv(("text", "number"), source, len(raw) - 1)
    assert refused.value.code == "bundle_size_limit"
    assert api._csv(("text", "number"), [{"text": "plain", "number": -0.0}], CAP) == (
        b"text,number\nplain,-0.0\n"
    )


def test_html_budget_stops_before_later_sections(tmp_path, monkeypatch):
    export_api()
    path, _, _ = make_record(tmp_path)
    record = workflow.read_workflow(path, max_bytes=CAP)
    calls = []
    original = workflow.html.escape

    def escape(text):
        calls.append(text)
        return original(text)

    monkeypatch.setattr(workflow.html, "escape", escape)
    with pytest.raises(workflow.WorkflowError, match="report_size_limit"):
        workflow._render_report(record, {"large": "<&中" * 10000}, 2000)
    assert len(calls) == 1


def test_valid_large_diagnostics_expand_past_cap_without_creating_destination(tmp_path):
    api = export_api()
    _, _, members = make_record(tmp_path)
    value = json.loads(members["result"])
    value["diagnostics"] = ["<&中" * 10000]
    members["result"] = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode()
    source = tmp_path / "large.sqlite"
    write_record(source, members, max_bytes=CAP)
    output = tmp_path / "large-export"
    with pytest.raises(api.ExportError, match="exceeds max_bytes"):
        api.export_record(source, output, max_bytes=source.stat().st_size)
    assert not output.exists()


@pytest.mark.parametrize("mutation", ["missing", "extra", "empty", "bytearray", "overflow"])
def test_direct_context_seam_validates_authoritative_member_shape(tmp_path, mutation):
    export_api()
    _, _, members = make_record(tmp_path)
    cap = CAP
    if mutation == "missing":
        del members["input"]
    elif mutation == "extra":
        members["extra"] = b"extra"
    elif mutation == "empty":
        members["input"] = b""
    elif mutation == "bytearray":
        members["input"] = bytearray(members["input"])
    else:
        cap = sum(map(len, members.values())) - 1
    with pytest.raises(workflow.WorkflowError):
        workflow._context_from_members(members, cap)


@pytest.mark.parametrize("name", DERIVED)
def test_rehashed_wrong_projection_is_rejected(tmp_path, name):
    api, output, _, _, _ = bundle(tmp_path)
    rehash(output, name, b"rehashed but not the retained projection\n")
    with pytest.raises(api.ExportError):
        api.verify_export(output, max_bytes=CAP)


@pytest.mark.parametrize("name", ["input.csv", "request.json", "result.json", "metadata.json"])
def test_rehashed_wrong_authoritative_content_is_rejected(tmp_path, name):
    api, output, _, _, _ = bundle(tmp_path)
    payload = (output / name).read_bytes()
    payload = payload.replace(b"0,0", b"1,0") if name == "input.csv" else b"{}"
    rehash(output, name, payload)
    with pytest.raises(api.ExportError):
        api.verify_export(output, max_bytes=CAP)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "extra",
        "empty",
        "digest",
        "directory",
        "duplicate",
        "type",
        "schema",
        "bool_size",
        "zero_size",
        "declared_overflow",
        "actual_overflow",
        "traversal",
        "hash_type",
        "uppercase_hash",
        "identity",
        "replay",
        "authentication",
        "metadata_duplicate",
        "utf8",
    ],
)
def test_hostile_bundle_is_rejected(tmp_path, mutation):
    api, output, _, _, _ = bundle(tmp_path)
    member = output / "summary.csv"
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    if mutation == "missing":
        member.unlink()
    elif mutation == "extra":
        (output / "extra").write_bytes(b"extra")
    elif mutation == "empty":
        member.write_bytes(b"")
    elif mutation == "digest":
        member.write_bytes(member.read_bytes() + b"changed")
    elif mutation == "directory":
        member.unlink()
        member.mkdir()
    elif mutation == "duplicate":
        manifest_path.write_bytes(b'{"schema":"duplicate",' + manifest_path.read_bytes()[1:])
    elif mutation == "type":
        manifest_path.write_bytes(b"[]")
    elif mutation == "utf8":
        manifest_path.write_bytes(b"\xff")
    elif mutation == "actual_overflow":
        member.write_bytes(b"a" * CAP)
    elif mutation == "metadata_duplicate":
        raw = (output / "metadata.json").read_bytes()
        rehash(output, "metadata.json", b'{"schema":"duplicate",' + raw[1:])
    else:
        if mutation == "schema":
            manifest["schema"] = "unknown"
        elif mutation == "bool_size":
            manifest["members"]["summary.csv"]["bytes"] = True
        elif mutation == "zero_size":
            manifest["members"]["summary.csv"]["bytes"] = 0
        elif mutation == "declared_overflow":
            manifest["members"]["summary.csv"]["bytes"] = CAP + 1
        elif mutation == "traversal":
            manifest["members"]["../summary.csv"] = manifest["members"].pop("summary.csv")
        elif mutation == "hash_type":
            manifest["members"]["summary.csv"]["sha256"] = 42
        elif mutation == "uppercase_hash":
            manifest["members"]["summary.csv"]["sha256"] = "A" * 64
        elif mutation == "identity":
            manifest["raw_input_sha256"] = "0" * 64
        elif mutation == "replay":
            manifest["replay"] = "MATCH"
        else:
            manifest["historical_execution_authenticated"] = 0
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(api.ExportError):
        api.verify_export(output, max_bytes=CAP)


def test_aggregate_size_includes_manifest_and_refuses_before_creation(tmp_path):
    api, output, _, _, _ = bundle(tmp_path)
    record = tmp_path / "run.sqlite"
    size = sum(p.stat().st_size for p in output.iterdir())
    assert size > record.stat().st_size
    for cap in (1, record.stat().st_size, size - 1):
        target = tmp_path / ("limited-" + str(cap))
        with pytest.raises(api.ExportError):
            api.export_record(record, target, max_bytes=cap)
        assert not target.exists()
    target = tmp_path / "exact"
    api.export_record(record, target, max_bytes=size)
    api.verify_export(target, max_bytes=size)
    with pytest.raises(api.ExportError):
        api.verify_export(target, max_bytes=size - 1)


def test_manifest_formatting_does_not_change_actual_byte_budget(tmp_path):
    api, output, _, _, _ = bundle(tmp_path)
    manifest = output / "manifest.json"
    manifest.write_bytes(manifest.read_bytes().removesuffix(b"\n"))
    actual_bytes = sum(path.stat().st_size for path in output.iterdir())
    assert api.verify_export(output, max_bytes=actual_bytes)["status"] == "complete"


def test_input_filename_must_match_the_validated_configuration(tmp_path):
    api, output, _, _, _ = bundle(tmp_path)
    (output / "input.csv").rename(output / "input.npz")
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["members"]["input.npz"] = manifest["members"].pop("input.csv")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(api.ExportError):
        api.verify_export(output, max_bytes=CAP)


@pytest.mark.parametrize(
    "kind", ["file", "directory", "dangling", "missing_parent", "file_parent", "symlink_parent"]
)
def test_destination_is_exclusive_and_parent_is_real(tmp_path, kind):
    api = export_api()
    record, _, _ = make_record(tmp_path)
    target = tmp_path / "output"
    if kind == "file":
        target.write_bytes(b"preserve")
    elif kind == "directory":
        target.mkdir()
        (target / "user").write_bytes(b"preserve")
    elif kind == "dangling":
        symlink_or_skip(target, tmp_path / "missing")
    elif kind == "missing_parent":
        target = tmp_path / "missing" / "output"
    elif kind == "file_parent":
        target.write_bytes(b"preserve")
        target = target / "child"
    else:
        parent = tmp_path / "real"
        parent.mkdir()
        directory_link_or_skip(target, parent)
        target = target / "child"
    before = record.read_bytes()
    with pytest.raises((api.ExportError, OSError)):
        api.export_record(record, target, max_bytes=CAP)
    assert record.read_bytes() == before
    assert not (tmp_path / "real" / "child").exists()
    if kind == "file":
        assert target.read_bytes() == b"preserve"
    if kind == "directory":
        assert (target / "user").read_bytes() == b"preserve"


@pytest.mark.parametrize("kind", ["root", "member"])
def test_symlinks_cannot_be_bundle_or_member(tmp_path, kind):
    api, output, _, _, _ = bundle(tmp_path)
    if kind == "root":
        link = tmp_path / "link"
        directory_link_or_skip(link, output)
        output = link
    else:
        member = output / "summary.csv"
        original = tmp_path / "original"
        original.write_bytes(member.read_bytes())
        member.unlink()
        symlink_or_skip(member, original)
    with pytest.raises(api.ExportError):
        api.verify_export(output, max_bytes=CAP)


@pytest.mark.parametrize(
    "stage", ["member_write", "member_close", "manifest_write", "manifest_close", "short_write"]
)
def test_failed_write_retains_directory_and_close_failure_is_not_success(
    tmp_path, monkeypatch, stage
):
    api = export_api()
    record, _, _ = make_record(tmp_path)
    target = tmp_path / "bundle"
    chosen = "manifest.json" if stage.startswith("manifest") else "candidates.csv"
    original = Path.open

    class FailingStream:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def write(self, payload):
            if stage.endswith("write"):
                self.stream.write(payload[:3])
                if stage == "short_write":
                    return 3
                raise OSError("injected write failure")
            return self.stream.write(payload)

        def __exit__(self, *args):
            self.stream.close()
            if stage.endswith("close"):
                raise OSError("injected close failure")

    def open_file(path, mode="r", *args, **kwargs):
        stream = original(path, mode, *args, **kwargs)
        return FailingStream(stream) if path == target / chosen and mode == "xb" else stream

    monkeypatch.setattr(Path, "open", open_file)
    with pytest.raises(OSError):
        api.export_record(record, target, max_bytes=CAP)
    assert target.is_dir()
    if stage == "manifest_close":
        assert api.verify_export(target, max_bytes=CAP)["status"] == "complete"
    else:
        with pytest.raises(api.ExportError):
            api.verify_export(target, max_bytes=CAP)


def test_file_browser_metadata_file_in_bundle_is_ignored_but_links_are_not(tmp_path):
    """R19 v3 review: opening the bundle folder in Finder made verify-export refuse it."""
    from _platform_support import symlink_or_skip

    api, output, summary, _, _ = bundle(tmp_path)
    (output / ".DS_Store").write_bytes(b"\0" * 8)
    (output / "desktop.ini").write_bytes(b"[.ShellClassInfo]\r\n")
    assert api.verify_export(output, max_bytes=CAP) == summary
    (output / "notes.txt").write_bytes(b"x")
    with pytest.raises(api.ExportError, match="exactly its eleven"):
        api.verify_export(output, max_bytes=CAP)
    (output / "notes.txt").unlink()
    (output / "desktop.ini").unlink()
    symlink_or_skip(output / "desktop.ini", output / "manifest.json")
    with pytest.raises(api.ExportError, match="exactly its eleven"):
        api.verify_export(output, max_bytes=CAP)
