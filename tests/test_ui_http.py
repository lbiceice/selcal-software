from __future__ import annotations

import http.client
import importlib
import importlib.util
import json
import re
import socket
import threading
import time

import pytest
from test_ui_jobs import admission
from test_ui_jobs import physical_tmp as _physical_tmp

physical_tmp = _physical_tmp


@pytest.fixture
def server(physical_tmp):
    if importlib.util.find_spec("selcal.ui") is None:
        yield None
        return
    ui = importlib.import_module("selcal.ui")
    server = ui.create_server(physical_tmp / "work", port=0)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def request(server, method, path, body=None, headers=None):
    assert server is not None, "missing local HTTP UI"
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)
    supplied = {"X-SelCal-Token": server.token, **(headers or {})}
    connection.request(method, path, body, supplied)
    response = connection.getresponse()
    result = response.status, dict(response.getheaders()), response.read()
    connection.close()
    return result


_RETAINED = re.compile(r"retained=(\d+)/")


def wait_while_progressing(fetch, done, *, stall=60.0, cap=900.0, what="job"):
    """Poll until done(job); fail if nothing observable advances for `stall` seconds.

    R17 items 4-5 (R16 Windows return): every replicate the interface runs is a fully
    synchronized checkpoint commit, which takes a large part of a second on some Windows disks,
    so a fixed total deadline mixed up "slow" and "stuck". Progress is the latest committed or
    replayed replicate count from the CLI's own progress lines, together with the state.
    """
    started = last_change = time.monotonic()
    last = None
    while True:
        job = fetch()
        if done(job):
            return job
        assert job["state"] == "running", job
        found = _RETAINED.findall(job.get("progress", ""))
        marker = (job["state"], found[-1] if found else None, len(job.get("progress", "")))
        now = time.monotonic()
        if marker != last:
            last, last_change = marker, now
        if now - last_change > stall:
            pytest.fail(f"{what} made no observable progress for {stall:.0f} s "
                        f"(last retained={marker[1]}, {now - started:.0f} s in total)")
        if now - started > cap:
            pytest.fail(f"{what} still progressing after {cap:.0f} s (retained={marker[1]})")
        time.sleep(0.05)


def test_static_and_example_routes_are_fixed_and_secured(server):
    for path, content_type in [
        ("/", "text/html"),
        ("/app.js", "text/javascript"),
        ("/style.css", "text/css"),
    ]:
        status, headers, body = request(server, "GET", path)
        assert status == 200 and body
        assert content_type in headers["Content-Type"]
        assert headers["Cache-Control"] == "no-store"
        assert headers["X-Content-Type-Options"] == "nosniff"
        assert headers["Referrer-Policy"] == "no-referrer"
        assert "default-src 'self'" in headers["Content-Security-Policy"]
        assert server.token.encode() not in body
    status, _, body = request(server, "GET", "/api/example")
    assert status == 200
    assert set(json.loads(body)) == {"input_text", "config_text"}


def test_unedited_page_defaults_complete_the_packaged_example_over_http(server):
    status, _, shell = request(server, "GET", "/")
    assert status == 200
    matched = re.search(rb'<input id="budget"[^>]*value="([0-9]+)"', shell)
    assert matched is not None, "The actual page must expose its default byte budget"
    status, _, example_body = request(server, "GET", "/api/example")
    assert status == 200
    example = json.loads(example_body)
    admission_body = {
        "config_text": example["config_text"],
        "max_bytes": matched.group(1).decode("ascii"),
        "allow_unattainable": False,
    }
    status, _, job_body = request(server, "POST", "/api/jobs", json.dumps(admission_body))
    assert status == 201
    job_id = json.loads(job_body)["id"]
    assert (
        request(server, "PUT", f"/api/jobs/{job_id}/input", example["input_text"].encode("utf-8"))[
            0
        ]
        == 200
    )

    def fetch():
        status, _, body = request(server, "GET", f"/api/jobs/{job_id}")
        assert status == 200
        return json.loads(body)

    def finish():
        return wait_while_progressing(fetch, lambda job: job["state"] != "running",
                                      what="The unchanged packaged example")

    assert request(server, "POST", f"/api/jobs/{job_id}/validate", b"")[0] == 200
    assert finish()["state"] == "validated"
    assert request(server, "POST", f"/api/jobs/{job_id}/run", b"")[0] == 200
    job = finish()
    assert job["state"] == "complete", job["result"]
    assert job["result"]["exit_code"] == 0
    assert job["result"]["data"]["planned_replicates"] == 199
    assert job["result"]["data"]["selected_candidate"] == 2
    assert job["result"]["data"]["p_value"] == 0.015
    records = list((server.jobs.workspace / "jobs" / job_id / "operations").glob("*/result.sqlite"))
    assert len(records) == 1 and records[0].stat().st_size > 0


@pytest.mark.parametrize(
    "headers",
    [
        {"Host": "evil.test"},
        {"Origin": "https://evil.test"},
        {"X-SelCal-Token": "wrong"},
        {"X-SelCal-Token": ""},
    ],
)
def test_api_trust_refusals(server, headers):
    assert request(server, "GET", "/api/jobs", headers=headers)[0] == 403


@pytest.mark.parametrize(
    "path",
    [
        "/../request.json",
        "/%2e%2e/",
        "/api/jobs/a%2fb",
        "/app.js?token=bad",
        "/series.csv",
        "/api/jobs/nope/export",
    ],
)
def test_routes_do_not_serve_arbitrary_files_or_future_actions(server, path):
    assert request(server, "GET", path)[0] == 404


def test_http_admission_input_and_saved_listing(server, physical_tmp):
    inp, body = admission(physical_tmp)
    status, _, data = request(server, "POST", "/api/jobs", json.dumps(body))
    assert status == 201
    job_id = json.loads(data)["id"]
    assert json.loads(data)["config_text"] == body["config_text"]
    assert request(server, "PUT", f"/api/jobs/{job_id}/input", inp.read_bytes())[0] == 200
    assert request(server, "PUT", f"/api/jobs/{job_id}/input", b"x")[0] == 409
    status, _, data = request(server, "GET", "/api/jobs")
    assert status == 200 and json.loads(data)["jobs"][0]["input_status"] == "complete"
    assert request(server, "POST", f"/api/jobs/{job_id}/validate", b"not empty")[0] == 400
    # Resume is routed, but this job has no checkpoint yet.
    assert request(server, "POST", f"/api/jobs/{job_id}/resume", b"")[0] == 400


def test_resume_keeps_the_existing_request_and_route_boundaries(server, physical_tmp):
    inp, body = admission(physical_tmp)
    _, _, data = request(server, "POST", "/api/jobs", json.dumps(body))
    job_id = json.loads(data)["id"]
    assert request(server, "PUT", f"/api/jobs/{job_id}/input", inp.read_bytes())[0] == 200
    route = f"/api/jobs/{job_id}/resume"
    for headers in (
        {"Host": "evil.test"},
        {"Origin": "https://evil.test"},
        {"X-SelCal-Token": "wrong"},
    ):
        assert request(server, "POST", route, b"", headers)[0] == 403
    assert request(server, "POST", route, b'{"checkpoint":"elsewhere"}')[0] == 400
    assert request(server, "GET", route)[0] == 404
    assert request(server, "POST", f"/api/jobs/{'0' * 32}/resume", b"")[0] == 404
    assert request(server, "POST", f"/api/jobs/{job_id}/verify", b"")[0] == 400
    for action in ("report", "export"):
        assert request(server, "POST", f"/api/jobs/{job_id}/{action}", b"")[0] == 400
    for action in ("artifacts", "download"):
        assert request(server, "POST", f"/api/jobs/{job_id}/{action}", b"")[0] == 404
    assert list((server.jobs.workspace / "jobs" / job_id / "operations").iterdir()) == []


def test_verify_route_authentication_empty_body_and_actual_child(server, physical_tmp):
    import io

    from test_ui_process import wait_job

    inp, body = admission(physical_tmp)
    job = server.jobs.admit(body)
    job_id = job["id"]
    server.jobs.upload(job_id, io.BytesIO(inp.read_bytes()), inp.stat().st_size)
    server.jobs.start(job_id, "run")
    assert wait_job(server.jobs, job_id)["state"] == "complete"
    route = f"/api/jobs/{job_id}/verify"
    for headers in (
        {"Host": "evil.test"},
        {"Origin": "https://evil.test"},
        {"X-SelCal-Token": "wrong"},
    ):
        assert request(server, "POST", route, b"", headers)[0] == 403
    assert request(server, "POST", route, b'{"record":"elsewhere"}')[0] == 400
    assert request(server, "GET", route)[0] == 404
    assert request(server, "POST", f"/api/jobs/{'0' * 32}/verify", b"")[0] == 404
    assert request(server, "POST", route, b"")[0] == 200
    result = wait_job(server.jobs, job_id)
    assert result["result"]["command"] == "verify"
    assert result["result"]["data"]["replay"] == "NOT_PERFORMED"


def test_duplicate_admission_keys_fail(server):
    assert request(server, "POST", "/api/jobs", b'{"a":1,"a":2}')[0] == 400


@pytest.mark.parametrize(
    "framing",
    [
        "Content-Length: -1",
        "Content-Length: 1.0",
        "Content-Length: 1\r\nContent-Length: 1",
        "Transfer-Encoding: chunked\r\nContent-Length: 2",
        "Content-Length: 524289",
    ],
)
def test_bad_framing_refused_before_body(server, framing):
    assert server is not None, "missing local HTTP UI"
    with socket.create_connection(("127.0.0.1", server.server_port), timeout=5) as connection:
        text = (
            f"POST /api/jobs HTTP/1.1\r\nHost: 127.0.0.1:{server.server_port}\r\n"
            f"X-SelCal-Token: {server.token}\r\n{framing}\r\nConnection: close\r\n\r\n"
        )
        connection.sendall(text.encode())
        response = connection.recv(2048)
    assert response.startswith((b"HTTP/1.0 400", b"HTTP/1.0 413"))


def test_short_raw_upload_retained_invalid(server, physical_tmp):
    _, body = admission(physical_tmp)
    _, _, data = request(server, "POST", "/api/jobs", json.dumps(body))
    job_id = json.loads(data)["id"]
    with socket.create_connection(("127.0.0.1", server.server_port), timeout=5) as connection:
        text = (
            f"PUT /api/jobs/{job_id}/input HTTP/1.1\r\n"
            f"Host: 127.0.0.1:{server.server_port}\r\nX-SelCal-Token: {server.token}\r\n"
            "Content-Length: 20\r\n\r\nx,y\n"
        )
        connection.sendall(text.encode())
        connection.shutdown(socket.SHUT_WR)
        assert connection.recv(2048).startswith(b"HTTP/1.0 400")
    assert server.jobs.get(job_id)["input_status"] == "incomplete"
