from __future__ import annotations

import json
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_connection_and_creation_guidance_is_in_the_input_step():
    class Elements(HTMLParser):
        def __init__(self):
            super().__init__()
            self.by_id = {}

        def handle_starttag(self, tag, attrs):
            attributes = dict(attrs)
            if attributes.get("id"):
                self.by_id[attributes["id"]] = (tag, attributes)

    html = (ROOT / "src/selcal/web/index.html").read_text(encoding="utf-8")
    parsed = Elements()
    parsed.feed(html)
    for identity in ("connection-notice", "retry-connection", "input-notice", "create-reason"):
        assert identity in parsed.by_id, f"Missing nearby UI guidance: {identity}"
        assert html.index(f'id="{identity}"') < html.index('id="run-heading"')
    assert html.index('id="connection-notice"') < html.index('id="input-file"')
    assert parsed.by_id["retry-connection"][0] == "button"
    assert parsed.by_id["retry-connection"][1].get("type") == "button"
    assert parsed.by_id["connection-notice"][1].get("role") in {"status", "alert"}
    assert parsed.by_id["input-notice"][1].get("role") in {"status", "alert"}


@pytest.mark.parametrize(
    "scenario",
    [
        "missing_token",
        "blocked_storage_without_token",
        "blocked_storage_with_url",
        "url_token_precedence",
        "no_input",
        "network_retry_preserves_edits",
        "offline_initial_retry",
        "expired_token",
        "example_create",
        "pasted_launch_url_same_tab",
    ],
)
def test_actual_javascript_connection_guidance_and_read_only_retry(scenario):
    node = shutil.which("node")
    if node is None:
        pytest.skip("JavaScript runtime unavailable; real browser acceptance is separate")
    # This is a small DOM/state harness for the shipped app.js. It verifies client
    # transitions and exact requests; it does not claim browser rendering or a
    # successful scientific calculation. Every token below is a synthetic fixture.
    harness = r"""
const fs = require("node:fs"), vm = require("node:vm"), assert = require("node:assert/strict");
const scenario = SCENARIO, example = EXAMPLE, elements = new Map(), requests = [];
const launchToken = "synthetic-launch-session", oldToken = "synthetic-old-session";
const noToken = ["missing_token", "blocked_storage_without_token",
  "pasted_launch_url_same_tab"].includes(scenario);
const globalHandlers = {};
const blockedStorage = scenario.startsWith("blocked_storage");
let network = scenario === "offline_initial_retry" ? "offline" : "ok";
let storedToken = noToken ? null : oldToken, saved = null, uploaded = null, posted = null;
function element(id) {
  if (!elements.has(id)) elements.set(id, {
    value: "", checked: false, disabled: false, hidden: false, title: "",
    textContent: "", options: [], files: [], handlers: {}, attributes: {},
    classList: {toggle() {}, add() {}, remove() {}},
    addEventListener(event, handler) { this.handlers[event] = handler; },
    setAttribute(key, value) { this.attributes[key] = String(value); },
    removeAttribute(key) { delete this.attributes[key]; },
    replaceChildren(...children) { this.options = [...children]; },
    append(...children) { this.options.push(...children); },
  });
  return elements.get(id);
}
element("budget").value = "8388608";
element("source-column").value = "x";
element("target-column").value = "y";
const response = (value, status = 200) => ({ok: status >= 200 && status < 300, status,
  json: async () => value});
const job = inputStatus => ({id: "fixture-job",
  state: inputStatus === "complete" ? "ready" : "created",
  input_status: inputStatus, format: "csv", config_text: posted.config_text,
  config_sha256: "c".repeat(64), input_sha256: inputStatus === "complete" ? "a".repeat(64) : null,
  max_bytes: posted.max_bytes, allow_unattainable: posted.allow_unattainable,
  created_at: "fixture-time", message: "Fixture input saved", record_target: {available: false}});
const context = vm.createContext({URLSearchParams, Blob, Object, JSON, console,
  location: {hash: noToken ? "" : "#token=" + launchToken, pathname: "/"},
  history: {replaceState() {}},
  sessionStorage: {
    setItem(key, value) {
      if (blockedStorage) throw new Error("SecurityError: session storage blocked");
      storedToken = value;
    },
    getItem() {
      if (blockedStorage) throw new Error("SecurityError: session storage blocked");
      return storedToken;
    },
  },
  setInterval() {},
  addEventListener(event, handler) { globalHandlers[event] = handler; },
  document: {getElementById: element, createElement() {return element(Symbol());}},
  fetch: async (path, options = {}) => {
    const method = options.method || "GET";
    requests.push({path, method, options});
    assert.equal(options.headers["X-SelCal-Token"], launchToken,
      "A launch fragment must take precedence over an old stored session");
    if (network === "offline") throw new TypeError("Failed to fetch");
    if (network === "expired") return response({
      error: "Session token is missing or expired. Reopen the printed launch URL."
    }, 403);
    if (path === "/api/jobs" && method === "GET") return response({jobs: saved ? [saved] : []});
    if (path === "/api/example" && method === "GET") return response({
      input_text: "x,y\n0,1\n1,2\n", config_text: example
    });
    if (path === "/api/jobs" && method === "POST") {
      posted = JSON.parse(options.body); saved = job("missing"); return response(saved);
    }
    if (path === "/api/jobs/fixture-job/input" && method === "PUT") {
      uploaded = options.body; saved = job("complete"); return response(saved);
    }
    throw new Error("Unexpected route " + method + " " + path);
  },
});
const read = expression => vm.runInContext(expression, context);
const tick = () => new Promise(setImmediate);
const inputStatus = () => element("connection-notice").textContent;
const writes = () => requests.filter(request => request.method !== "GET");
const click = async id => {
  assert.equal(typeof element(id).handlers.click, "function", "Click listener missing: " + id);
  await element(id).handlers.click(); await tick();
};
const assertMirroredNotice = () => assert.equal(element("input-notice").textContent,
  element("notice").textContent, "Operation messages must also appear in the input step");
async function createExample() {
  assert.equal(element("create").disabled, false);
  await click("create");
  assert.deepEqual(writes().map(request => [request.method, request.path]), [
    ["POST", "/api/jobs"], ["PUT", "/api/jobs/fixture-job/input"]
  ]);
  assert.equal(posted.config_text, example, "The original configuration must remain exact");
  assert.equal(await uploaded.text(), "x,y\n0,1\n1,2\n");
  assert.equal(read("selectedId"), "fixture-job");
  assert.equal(element("validate").disabled, false);
}
vm.runInContext(fs.readFileSync(SCRIPT, "utf8"), context);
(async () => {
  await tick();
  assert.equal(typeof element("input-file").handlers.change, "function",
    "Storage denial must not interrupt listener registration");
  if (noToken) {
    assert.match(inputStatus(), /token|session/i);
    assert.match(inputStatus(), /launch|printed|reopen/i);
    assert.equal(element("create").disabled, true);
    const ownInput = new Blob(["x,y\n1,2\n"]);
    element("input-file").files = [ownInput];
    element("input-file").handlers.change();
    element("config-text").value = example;
    assert.equal(element("create").disabled, true,
      "Selecting input must not conceal an absent session credential");
    assert.match(element("create-reason").textContent, /token|session|connection/i);
    await click("create");
    await click("retry-connection");
    assert.equal(requests.length, 0, "Absent credentials must never be sent to an API");
    assertMirroredNotice();
    if (scenario !== "pasted_launch_url_same_tab") return;
    // The user pastes the complete launch URL into this same tab: only the fragment changes.
    assert.equal(typeof globalHandlers.hashchange, "function", "hashchange listener missing");
    context.location.hash = "#token=" + launchToken;
    globalHandlers.hashchange(); await tick(); await tick();
    assert.deepEqual(requests.map(request => [request.method, request.path]),
      [["GET", "/api/jobs"]], "The pasted token is used for one read only; nothing is submitted");
    assert.equal(read("sessionToken"), launchToken);
    assert.equal(read("inputBytes"), ownInput, "A chosen input is kept across the reconnection");
    assert.equal(element("config-text").value, example);
    assert.doesNotMatch(inputStatus(), /no local session/i);
    assert.equal(element("create").disabled, false);
    await click("create");
    assert.deepEqual(writes().map(request => [request.method, request.path]), [
      ["POST", "/api/jobs"], ["PUT", "/api/jobs/fixture-job/input"]
    ]);
    assert.equal(await uploaded.text(), "x,y\n1,2\n", "The user's own input is what gets saved");
    assert.equal(posted.config_text, example);
    return;
  }
  if (scenario === "offline_initial_retry") {
    assert.match(inputStatus(), /connect|reach|server|helper|terminal/i);
    assert.equal(read("inputBytes"), null);
    assert.equal(element("create").disabled, true);
    const before = requests.length;
    network = "ok"; await click("retry-connection");
    assert.deepEqual(requests.slice(before).map(request => [request.method, request.path]),
      [["GET", "/api/jobs"]]);
    assert.equal(read("inputBytes"), null, "Retry must not silently select the example");
    assert.equal(element("create").disabled, true);
    assert.match(element("create-reason").textContent, /input|example/i);
    await click("example"); await createExample();
    return;
  }
  assert.match(element("input-description").textContent, /Included example/);
  assert.equal(element("create").disabled, false);
  assertMirroredNotice();
  if (scenario === "blocked_storage_with_url") {
    assert.match(inputStatus(), /storage|store|remember|retain/i);
    assert.match(inputStatus(), /launch|URL|link/i);
    await click("example"); await createExample();
    return;
  }
  if (scenario === "url_token_precedence" || scenario === "example_create") {
    await createExample(); return;
  }
  if (scenario === "no_input") {
    element("input-file").files = [];
    element("input-file").handlers.change();
    assert.equal(read("inputBytes"), null);
    assert.equal(element("create").disabled, true);
    assert.match(element("create-reason").textContent, /input|example/i);
    await click("create");
    assert.equal(writes().length, 0);
    assertMirroredNotice(); return;
  }
  if (scenario === "expired_token") {
    network = "expired"; await click("retry-connection");
    assert.match(inputStatus(), /token|session/i);
    assert.match(inputStatus(), /launch|printed|reopen/i);
    assert.match(inputStatus(), /expired|refused|403/i);
    assert.equal(element("create").disabled, true);
    assert.equal(writes().length, 0);
    assertMirroredNotice(); return;
  }
  assert.equal(scenario, "network_retry_preserves_edits");
  const ownInput = new Blob(["x,y\n11,22\n"]);
  element("input-file").files = [ownInput]; element("input-file").handlers.change();
  element("seed").value = "9007199254740993"; element("seed").handlers.input();
  element("budget").value = "16777216"; element("budget").handlers.input();
  element("override").checked = true; element("override").handlers.input();
  const config = element("config-text").value;
  const description = element("input-description").textContent;
  const revision = read("revision");
  network = "offline"; await click("example");
  assert.match(inputStatus(), /connect|reach|server|helper|terminal/i);
  assert.equal(element("create").disabled, true);
  assert.equal(read("inputBytes"), ownInput, "A failed example fetch must preserve selected bytes");
  assert.equal(element("config-text").value, config);
  assertMirroredNotice();
  const before = requests.length;
  network = "ok"; await click("retry-connection");
  assert.deepEqual(requests.slice(before).map(request => [request.method, request.path]),
    [["GET", "/api/jobs"]], "Retry is only a read; it must not load examples or submit operations");
  assert.equal(writes().length, 0);
  assert.equal(read("inputBytes"), ownInput);
  assert.equal(element("config-text").value, config);
  assert.equal(element("input-description").textContent, description);
  assert.equal(element("seed").value, "9007199254740993");
  assert.equal(element("budget").value, "16777216");
  assert.equal(element("override").checked, true);
  assert.equal(read("revision"), revision);
  assert.equal(read("selectedId"), null);
  assert.equal(element("create").disabled, false);
})().catch(error => {console.error(error); process.exitCode = 1;});
"""
    for key, value in {
        "SCENARIO": scenario,
        "EXAMPLE": (ROOT / "examples/workflow/pearson.json").read_text(encoding="utf-8"),
        "SCRIPT": str(ROOT / "src/selcal/web/app.js"),
    }.items():
        harness = harness.replace(key, json.dumps(value))
    result = subprocess.run(
        [node, "-e", harness], capture_output=True, text=True, errors="replace", timeout=60
    )
    assert result.returncode == 0, result.stderr
