"""Download responses name their job and operation and state their checked identity.

R12 Windows download retest (2026-10-04): in ordinary Chrome the evidence ZIP saved as 0 bytes
while the same backend served 6/6 complete responses, and an older "evidence (3).zip" in the
same folder could be mistaken for the current one. The page now refuses a short or altered
body before saving (see test_ui_artifacts), using these headers, and file names carry the job
and operation. Real-browser saving still needs a native Windows check.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile

from test_ui_http import request as http_request
from test_ui_jobs import admission
from test_ui_jobs import physical_tmp as _physical_tmp
from test_ui_resume import helper, poll

physical_tmp = _physical_tmp


def test_downloads_are_named_by_job_and_operation_and_declare_their_identity(physical_tmp):
    inp, request = admission(physical_tmp)
    request["max_bytes"] = "8388608"
    workspace = physical_tmp / "work"
    with helper(workspace) as server:
        status, _, body = http_request(server, "POST", "/api/jobs", json.dumps(request))
        assert status == 201
        job_id = json.loads(body)["id"]
        assert http_request(server, "PUT", f"/api/jobs/{job_id}/input", inp.read_bytes())[0] == 200
        for action in ("run", "report", "export"):
            assert http_request(server, "POST", f"/api/jobs/{job_id}/{action}", b"")[0] == 200
            poll(server, job_id, lambda job: job["state"] != "running")
        job = workspace / "jobs" / job_id
        operations = {
            "record": json.loads((job / "record.json").read_bytes())["operation_id"],
            "report": json.loads((job / "report.json").read_bytes())["operation_id"],
            "bundle": json.loads((job / "bundle.json").read_bytes())["operation_id"],
        }
        files = {"record": "result.sqlite", "report": "report.html", "bundle": "evidence.zip"}
        for kind, base in files.items():
            path = f"/api/jobs/{job_id}/download/{kind}"
            status, headers, raw = http_request(server, "GET", path)
            assert status == 200 and raw
            name = f"selcal-{job_id[:8]}-{operations[kind][:8]}-{base}"
            assert headers["Content-Disposition"] == f'attachment; filename="{name}"'
            assert re.fullmatch(r"selcal-[0-9a-f]{8}-[0-9a-f]{8}-[a-z]+\.[a-z]+", name)
            assert int(headers["Content-Length"]) == int(headers["X-SelCal-Bytes"]) == len(raw)
            assert headers["X-SelCal-SHA256"] == hashlib.sha256(raw).hexdigest()
            original = job / "operations" / operations[kind] / base
            assert original.read_bytes() == raw
            # Download-manager-independent fallback: the checked original in the workspace.
            assert workspace / headers["X-SelCal-Workspace-Path"] == original
            if kind == "bundle":
                with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                    assert len(archive.infolist()) == 11 and archive.testzip() is None


class _BrokenStream:
    def write(self, data):
        raise OSError("connection reset by the browser")

    def flush(self):
        raise OSError("connection reset by the browser")


class _ResetAfter:
    """Accepts ``limit`` bytes, then fails as a connection reset by the browser would."""

    def __init__(self, limit):
        self.limit, self.sent = limit, bytearray()

    def write(self, data):
        room = self.limit - len(self.sent)
        self.sent += bytes(data)[: max(room, 0)]
        if len(data) > room:
            raise OSError("connection reset by the browser")
        return len(data)

    def flush(self):
        pass


def _handler(dispatch, stream=None):
    from selcal import ui

    handler = object.__new__(ui._Handler)
    handler.request_version, handler.requestline = "HTTP/1.1", "GET /download HTTP/1.1"
    handler.command, handler.client_address = "GET", ("127.0.0.1", 0)
    handler.wfile = _BrokenStream() if stream is None else stream
    written = []
    handler._json = lambda status, value: written.append(status)
    handler._dispatch = lambda: dispatch(handler)
    return handler, written


BODY = b"PK\x03\x04" + bytes(range(256)) * 64


def _send_zip(handler):
    handler._send(200, BODY, "application/zip", "selcal-a-b-evidence.zip")


def _split(sent):
    head, _, body = bytes(sent).partition(b"\r\n\r\n")
    return head.decode("latin-1"), body


def test_no_error_document_is_appended_after_a_download_response_started():
    """A failure while sending a body must close, not write a JSON error into the binary."""
    handler, written = _handler(
        lambda h: h._send(200, b"PK\x03\x04payload", "application/zip", "selcal-a-b-evidence.zip")
    )
    handler._handle()
    assert written == []


def test_errors_before_any_response_still_get_a_json_reply():
    def refuse(handler):
        raise OSError("local file operation failed")

    handler, written = _handler(refuse)
    handler._handle()
    assert written == [400]


def test_headers_sent_then_body_fails_leaves_a_detectably_short_response():
    """Headers state the full size and digest; the body fails at once; nothing is appended."""
    stream = _ResetAfter(0)
    handler, written = _handler(_send_zip, stream)
    handler._handle()
    assert written == [] and stream.sent == b""
    probe = _ResetAfter(10**9)
    _handler(_send_zip, probe)[0]._handle()
    head, body = _split(probe.sent)
    assert body == BODY
    assert f"Content-Length: {len(BODY)}" in head and f"X-SelCal-Bytes: {len(BODY)}" in head
    assert f"X-SelCal-SHA256: {hashlib.sha256(BODY).hexdigest()}" in head
    header_bytes = len(probe.sent) - len(BODY)
    stream = _ResetAfter(header_bytes)
    handler, written = _handler(_send_zip, stream)
    handler._handle()
    head, body = _split(stream.sent)
    assert written == [] and body == b"" and f"X-SelCal-Bytes: {len(BODY)}" in head


def test_partial_body_is_a_prefix_shorter_than_the_stated_size_and_not_extended():
    """A reset mid-body leaves a strict prefix; the page refuses it by X-SelCal-Bytes."""
    probe = _ResetAfter(10**9)
    _handler(_send_zip, probe)[0]._handle()
    header_bytes = len(probe.sent) - len(BODY)
    for cut in (1, len(BODY) // 2, len(BODY) - 1):
        stream = _ResetAfter(header_bytes + cut)
        handler, written = _handler(_send_zip, stream)
        handler._handle()
        head, body = _split(stream.sent)
        assert written == [], "no JSON error may follow a started download"
        assert body == BODY[:cut] and len(body) < len(BODY)
        assert f"X-SelCal-Bytes: {len(BODY)}" in head


def test_stream_failing_before_any_response_is_closed_quietly():
    """A refused request whose error reply cannot be written either does not raise."""

    def refuse(handler):
        raise OSError("local file operation failed")

    from selcal import ui

    handler = object.__new__(ui._Handler)
    handler.request_version, handler.requestline = "HTTP/1.1", "GET /download HTTP/1.1"
    handler.command, handler.client_address = "GET", ("127.0.0.1", 0)
    handler.wfile = _BrokenStream()
    handler._dispatch = lambda: refuse(handler)
    handler._handle()
    assert handler.close_connection is True
