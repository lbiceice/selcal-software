"""Offline, authenticated loopback HTTP shell for the basic local workflow."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from typing import Any

from selcal.ui_jobs import JobManager, UIError, strict_json

_STATIC = {
    "/": ("index.html", "text/html"),
    "/app.js": ("app.js", "text/javascript"),
    "/style.css": ("style.css", "text/css"),
}
_JOB = re.compile(
    r"/api/jobs/([0-9a-f]{32})(?:/(input|validate|run|resume|verify|report|export|cancel))?\Z"
)
_DOWNLOAD = re.compile(r"/api/jobs/([0-9a-f]{32})/download/(record|report|bundle)\Z")
_JSON_LIMIT = 524288
_HEADERS = {
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; style-src 'self' "
        "'sha256-/UEtROarG+pZ6KzBT6Fh5yp27PejSEFa07Oftkmpals=' "
        "'sha256-IG6y3kzG8m0FBU79C5uCbENCks908o0SFn8uV6xz6p0='; connect-src 'self'; "
        "img-src 'none'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; "
        "form-action 'self'"
    ),
}


class LocalServer(ThreadingHTTPServer):
    """The server owns its job helper; closing retains all work and reaps its child."""

    daemon_threads = True
    block_on_close = False

    def __init__(self, workspace: str | Path, port: int) -> None:
        self.jobs = JobManager(workspace)
        self.token = secrets.token_urlsafe(32)
        try:
            super().__init__(("127.0.0.1", port), _Handler)
        except BaseException:
            self.jobs.close()
            raise
        self.host = f"127.0.0.1:{self.server_port}"
        self.origin = "http://" + self.host

    def server_close(self) -> None:
        self.jobs.close()
        super().server_close()


class _Handler(BaseHTTPRequestHandler):
    server: LocalServer

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(5)

    def log_message(self, format: str, *args: Any) -> None:
        # Never record browser secrets, requested paths, raw input or config.
        pass

    def _send(
        self,
        status: int,
        body: bytes,
        content_type: str = "application/json",
        filename: str | None = None,
        location: str | None = None,
    ) -> None:
        # Once a status line is out, an error must not append a second response to the body.
        self._response_started = True
        self.send_response(status)
        charset = (
            "; charset=utf-8"
            if content_type.startswith("text/") or content_type == "application/json"
            else ""
        )
        self.send_header("Content-Type", content_type + charset)
        self.send_header("Content-Length", str(len(body)))
        if filename is not None:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            # The checked artifact's identity, so the page can refuse a short or altered body.
            self.send_header("X-SelCal-Bytes", str(len(body)))
            self.send_header("X-SelCal-SHA256", hashlib.sha256(body).hexdigest())
        if location is not None:
            # Workspace-relative (checked ASCII by the job manager): a copy needing no download.
            self.send_header("X-SelCal-Workspace-Path", location)
        for name, value in _HEADERS.items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, value: object) -> None:
        self._send(status, json.dumps(value, allow_nan=False, ensure_ascii=True).encode("utf-8"))

    def _one_header(self, name: str) -> str | None:
        values = self.headers.get_all(name, [])
        if len(values) > 1:
            raise UIError("Repeated request headers are refused.", 400)
        return values[0] if values else None

    def _authorize(self) -> None:
        if self._one_header("Host") != self.server.host:
            raise UIError("Use the exact loopback address printed by SelCal.", 403)
        origin = self._one_header("Origin")
        if origin is not None and origin != self.server.origin:
            raise UIError("Cross-origin requests are refused.", 403)
        if self.path.startswith("/api/"):
            supplied = self._one_header("X-SelCal-Token") or ""
            if not hmac.compare_digest(supplied.encode("utf-8"), self.server.token.encode("ascii")):
                raise UIError(
                    "Session token is missing or expired. Reopen the printed launch URL.", 403
                )

    def _length(self, cap: int) -> int:
        if self.headers.get_all("Transfer-Encoding"):
            raise UIError("Transfer encoding is unsupported; send an exact Content-Length.")
        value = self._one_header("Content-Length")
        if value is None:
            if self.command in {"POST", "PUT"}:
                raise UIError("An exact Content-Length is required.")
            return 0
        if not re.fullmatch(r"[0-9]{1,20}", value):
            raise UIError("Content-Length must be a nonnegative decimal integer.")
        length = int(value)
        if length > cap:
            raise UIError("The request exceeds this operation's byte limit.", 413)
        return length

    def _body(self, length: int) -> bytes:
        chunks = []
        while length:
            chunk = self.rfile.read(min(65536, length))
            if not chunk:
                raise UIError("Request body ended before its declared length.")
            chunks.append(chunk)
            length -= len(chunk)
        return b"".join(chunks)

    def _dispatch(self) -> None:
        self._authorize()
        if any(character in self.path for character in ("%", "?", "#", "\\")) or ".." in self.path:
            raise UIError("Unknown route.", 404)
        if self.command == "GET" and self.path in _STATIC:
            if self._length(0):
                raise UIError("GET requests cannot contain a body.")
            name, content_type = _STATIC[self.path]
            self._send(200, files("selcal").joinpath("web", name).read_bytes(), content_type)
            return
        if self.command == "GET" and self.path == "/api/example":
            self._length(0)
            assets = files("selcal").joinpath("web")
            self._json(
                200,
                {
                    "input_text": assets.joinpath("series.csv").read_text(encoding="utf-8"),
                    "config_text": assets.joinpath("pearson.json").read_text(encoding="utf-8"),
                },
            )
            return
        if self.path == "/api/jobs":
            if self.command == "GET":
                self._length(0)
                self._json(200, {"jobs": self.server.jobs.list_jobs()})
                return
            if self.command == "POST":
                data = strict_json(self._body(self._length(_JSON_LIMIT)))
                self._json(201, self.server.jobs.admit(data))
                return
        download = _DOWNLOAD.fullmatch(self.path)
        if download and self.command == "GET":
            self._length(0)
            raw, content_type, filename, location = self.server.jobs.download(*download.groups())
            self._send(200, raw, content_type, filename, location)
            return
        matched = _JOB.fullmatch(self.path)
        if matched:
            job_id, action = matched.groups()
            if self.command == "GET" and action is None:
                self._length(0)
                self._json(200, self.server.jobs.get(job_id))
                return
            if self.command == "PUT" and action == "input":
                length = self._length(self.server.jobs.input_limit(job_id))
                self._json(200, self.server.jobs.upload(job_id, self.rfile, length))
                return
            if self.command == "POST" and action in {
                "validate",
                "run",
                "resume",
                "verify",
                "report",
                "export",
                "cancel",
            }:
                # A nonempty action body is a request error, not a data-size restriction.
                if self._length(_JSON_LIMIT) != 0:
                    raise UIError("Action requests require an empty body.")
                value = (
                    self.server.jobs.cancel(job_id)
                    if action == "cancel"
                    else self.server.jobs.start(job_id, action)
                )
                self._json(200, value)
                return
        raise UIError(
            "Unknown route or method.",
            404,
        )

    def _handle(self) -> None:
        self.close_connection = True
        self._response_started = False
        try:
            self._dispatch()
        except UIError as error:
            if not self._response_started:
                self._json(error.status, {"error": str(error)})
        except (TimeoutError, OSError):
            if self._response_started:
                # The status line and part of the body may be out; close instead of writing
                # an error document into a binary download.
                return
            try:
                self._json(
                    400,
                    {
                        "error": "The request or local file operation could not finish. "
                        "Retain the workspace and retry with a new job if the upload is incomplete."
                    },
                )
            except OSError:
                pass

    do_GET = _handle
    do_POST = _handle
    do_PUT = _handle
    do_DELETE = _handle
    do_OPTIONS = _handle
    do_HEAD = _handle


def create_server(workspace: str | Path, *, port: int = 0) -> LocalServer:
    """Bind only 127.0.0.1; port zero lets the operating system choose a free port."""
    if type(port) is not int or not 0 <= port <= 65535:
        raise UIError("Port must be an integer from 0 to 65535.")
    return LocalServer(workspace, port)


def serve(workspace: str | Path, *, port: int = 0, no_browser: bool = False) -> int:
    """Run until Ctrl-C; closing a browser tab does not stop the helper."""
    with create_server(workspace, port=port) as server:
        url = server.origin + "/#token=" + server.token
        sys.stdout.write("SelCal local UI: " + url + "\n")
        sys.stdout.write(
            "Keep this terminal open. Ctrl-C stops this helper and its own active child; "
            "all work is retained. Closing a browser tab does not stop it.\n",
        )
        sys.stdout.flush()
        if server.jobs.unloadable_jobs:
            sys.stderr.write(
                f"{len(server.jobs.unloadable_jobs)} saved job folder(s) could not be read and "
                "were left untouched (for example an interrupted job creation): "
                + ", ".join(server.jobs.unloadable_jobs)
                + ". Other jobs are available.\n"
            )
            sys.stderr.flush()
        if not no_browser:
            webbrowser.open(url)
        try:
            server.serve_forever(poll_interval=0.2)
        except KeyboardInterrupt:
            pass
    return 0
