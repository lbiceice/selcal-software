"use strict";
const $ = (id) => document.getElementById(id);
const fragment = new URLSearchParams(location.hash.slice(1));
if (fragment.has("token")) sessionStorage.setItem("selcal-token", fragment.get("token"));
history.replaceState(null, "", location.pathname);
const sessionToken = sessionStorage.getItem("selcal-token") || "";
let inputBytes = null, selectedId = null, viewedId = null, latestJob = null, busy = false, revision = 0;
let viewGeneration = 0, pollInFlight = false, actionSerial = 0;
// R12 download retest: a 0 ms revocation, or the next action revoking every URL, could end a
// download before the browser read the Blob. Download URLs now live for a bounded time.
const downloadURLs = new Map();
const DOWNLOAD_URL_LIFETIME_MS = 120000, MAX_DOWNLOAD_URLS = 6;
function releaseDownloadURL(url) {
  if (!downloadURLs.has(url)) return;
  clearTimeout(downloadURLs.get(url));
  downloadURLs.delete(url);
  URL.revokeObjectURL(url);
}
if (typeof globalThis.addEventListener === "function") {
  globalThis.addEventListener("pagehide", () => { for (const url of [...downloadURLs.keys()]) releaseDownloadURL(url); });
}
async function sha256Hex(blob) {
  if (!(globalThis.crypto && crypto.subtle)) return null;
  const digest = await crypto.subtle.digest("SHA-256", await blob.arrayBuffer());
  return [...new Uint8Array(digest)].map((value) => value.toString(16).padStart(2, "0")).join("");
}
function attachmentName(response, fallback) {
  const match = /filename="([^"]+)"/.exec(response.headers.get("Content-Disposition") || "");
  return match ? match[1] : fallback;
}

function notice(message, error = false) {
  $("notice").textContent = message;
  $("notice").classList.toggle("error", error);
}
async function api(path, options = {}) {
  const response = await fetch(path, {...options, headers: {"X-SelCal-Token": sessionToken, ...(options.headers || {})}, cache: "no-store"});
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || `Operation refused (${response.status}).`);
  return value;
}
function buttons() {
  for (const id of ["example", "input-file", "config-file", "format", "source-column", "target-column", "candidates", "statistic", "null-model", "statistic-params", "null-params", "selection", "replicates", "alpha", "tolerance", "seed", "budget", "override", "config-text", "saved-jobs"]) $(id).disabled = busy;
  const running = latestJob && latestJob.state === "running";
  $("create").disabled = busy || !inputBytes;
  $("validate").disabled = busy || !selectedId || running || !latestJob || latestJob.input_status !== "complete";
  $("run").disabled = $("validate").disabled || (latestJob && ["complete", "not_evaluable", "interrupted"].includes(latestJob.state));
  $("resume").disabled = $("validate").disabled;
  $("verify").disabled = $("validate").disabled || !latestJob.record_target || !latestJob.record_target.available;
  for (const id of ["report", "export", "download-record"]) $(id).disabled = $("verify").disabled;
  const artifacts = latestJob && latestJob.artifacts;
  for (const id of ["preview-report", "download-report"]) $(id).disabled = $("verify").disabled || !artifacts || !artifacts.report || !artifacts.report.available;
  $("download-bundle").disabled = $("verify").disabled || !artifacts || !artifacts.bundle || !artifacts.bundle.available;
  $("cancel").disabled = busy || !running;
}
function clearCheckDisplay() {
  clearArtifactDisplay();
  $("verification-status").textContent = "No current content-check result.";
  $("result-summary").replaceChildren();
  $("terminal").textContent = "No result for this request. Reselect a saved job to inspect its earlier observations.";
}
function clearArtifactDisplay() {
  $("report-preview").srcdoc = "";
  $("report-preview").hidden = true;
  $("artifact-status").textContent = "No artifact requested in this view.";
}
function invalidate() {
  revision++;
  viewGeneration++;
  selectedId = null;
  clearCheckDisplay();
  $("selection-note").textContent = "Input or plan changed. Create a new job before validation or running, or reselect the saved job to resume its original plan.";
  buttons();
}
function integer(text, label) {
  const value = text.trim();
  if (!/^-?(0|[1-9][0-9]*)$/.test(value)) throw new Error(`${label} must be an exact decimal integer.`);
  return value;
}
function floating(text, label) {
  const value = text.trim();
  if (!/^-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?$/.test(value)) throw new Error(`${label} must be a JSON number.`);
  return /[.eE]/.test(value) ? value : `${value}.0`;
}
function parameters(text) {
  const checked = JSON.parse(text);
  if (!checked || Array.isArray(checked) || typeof checked !== "object") throw new Error("Parameters must be JSON objects.");
  return text; // Original spelling, including large integers, remains authoritative.
}
function formText() {
  const input = $("format").value === "csv"
    ? `{"format":"csv","source_column":${JSON.stringify($("source-column").value)},"target_column":${JSON.stringify($("target-column").value)}}`
    : '{"format":"npz"}';
  const candidates = $("candidates").value.split(",").map((x) => integer(x, "Candidate lag")).join(",");
  return `{
  "schema": "selcal.workflow-config.v1",
  "input": ${input},
  "plan": {
    "candidates": [${candidates}],
    "statistic_name": ${JSON.stringify($("statistic").value)},
    "statistic_params": ${parameters($("statistic-params").value)},
    "selection_rule": ${JSON.stringify($("selection").value)},
    "null_name": ${JSON.stringify($("null-model").value)},
    "null_params": ${parameters($("null-params").value)},
    "replicates": ${integer($("replicates").value, "Repeats")},
    "alpha": ${floating($("alpha").value, "Alpha")},
    "tie_tolerance": ${floating($("tolerance").value, "Tie tolerance")},
    "root_seed": ${integer($("seed").value, "Seed")}
  }
}`;
}
// Extract original JSON value spans. JSON.parse is used for syntax and strings
// only; no numeric value from it is ever serialized back into the plan.
function rawObject(text) {
  const parsed = JSON.parse(text);
  if (!parsed || Array.isArray(parsed) || typeof parsed !== "object") throw new Error("Expected a JSON object.");
  let i = 0;
  const result = {};
  const space = () => { while (/\s/.test(text[i] || "") && i < text.length) i++; };
  space(); i++;
  while (true) {
    space(); if (text[i] === "}") return result;
    const start = i++; let escaped = false;
    while (i < text.length) {
      const c = text[i++];
      if (escaped) escaped = false;
      else if (c === "\\") escaped = true;
      else if (c === '"') break;
    }
    const key = JSON.parse(text.slice(start, i));
    if (Object.hasOwn(result, key)) throw new Error("Duplicate configuration keys are refused.");
    space(); i++; space(); const valueStart = i;
    let depth = 0, quoted = false; escaped = false;
    for (; i < text.length; i++) {
      const c = text[i];
      if (quoted) {
        if (escaped) escaped = false;
        else if (c === "\\") escaped = true;
        else if (c === '"') quoted = false;
      } else if (c === '"') quoted = true;
      else if (c === "{" || c === "[") depth++;
      else if (c === "}" || c === "]") { if (depth === 0) break; depth--; }
      else if (c === "," && depth === 0) break;
    }
    result[key] = text.slice(valueStart, i).trim();
    if (text[i] === ",") i++;
  }
}
function setOption(id, value) {
  const select = $(id);
  if (![...select.options].some((option) => option.value === value)) {
    const option = document.createElement("option"); option.value = value; option.textContent = value; select.append(option);
  }
  select.value = value;
}
function syncForm(text) {
  const root = rawObject(text), input = rawObject(root.input), plan = rawObject(root.plan);
  $("format").value = JSON.parse(input.format);
  if (input.source_column) $("source-column").value = JSON.parse(input.source_column);
  if (input.target_column) $("target-column").value = JSON.parse(input.target_column);
  $("candidates").value = plan.candidates.slice(1, -1);
  for (const [id, key] of [["statistic", "statistic_name"], ["null-model", "null_name"], ["selection", "selection_rule"]]) setOption(id, JSON.parse(plan[key]));
  for (const [id, key] of [["statistic-params", "statistic_params"], ["null-params", "null_params"], ["replicates", "replicates"], ["alpha", "alpha"], ["tolerance", "tie_tolerance"], ["seed", "root_seed"]]) $(id).value = plan[key];
}
function showJob(job) {
  latestJob = job;
  const data = job.result && job.result.data;
  const rows = [["Job", job.id], ["Operation state", job.state], ["Input", job.input_status], ["Raw input SHA-256", job.input_sha256 || "Not admitted"], ["Configuration SHA-256", job.config_sha256]];
  $("verification-status").textContent = job.record_target && job.record_target.available
    ? "Saved record target available. Availability is not verification."
    : (job.record_target && job.record_target.reason) || "No UI verification target.";
  if (data) {
    for (const [label, key] of [["Scientific status", "status"], ["Failure stage", "failure_stage"], ["Selected lag", "selected_candidate"], ["p-value", "p_value"], ["Reject null", "reject_null"], ["Semantic input SHA-256", "semantic_input_sha256"], ["Scientific plan SHA-256", "scientific_plan_sha256"]]) {
      if (Object.hasOwn(data, key)) rows.push([label, data[key] === null ? "Not available" : String(data[key])]);
    }
    if (data.checkpoint) {
      for (const [label, key] of [["Execution", "execution_id"], ["Replayed repeats", "replayed_replicates"], ["Newly retained repeats", "appended_replicates"]]) rows.push([label, String(data.checkpoint[key])]);
    }
    if (job.operation === "verify") {
      for (const [label, key] of [["Verification scope", "verification_scope"], ["Replay", "replay"], ["Historical execution authenticated", "historical_execution_authenticated"]]) rows.push([label, String(data[key])]);
      $("verification-status").textContent = "Saved content-check observation, not current-byte verification. No replay; historical execution is not authenticated.";
    }
  }
  $("result-summary").replaceChildren();
  for (const [label, value] of rows) {
    const term = document.createElement("dt"), definition = document.createElement("dd");
    term.textContent = label; definition.textContent = value; $("result-summary").append(term, definition);
  }
  $("progress").textContent = job.progress || "No progress output.";
  $("terminal").textContent = job.result ? JSON.stringify(job.result, null, 2) : "No terminal result. Progress alone does not establish completion.";
  notice(job.message, job.state === "failed" || job.state === "interrupted");
  buttons();
}
async function refreshJobs() {
  const generation = viewGeneration, result = await api("/api/jobs");
  if (generation !== viewGeneration) return;
  const chosen = viewedId;
  $("saved-jobs").replaceChildren();
  const empty = document.createElement("option"); empty.value = ""; empty.textContent = "Choose a saved job"; $("saved-jobs").append(empty);
  for (const job of result.jobs) {
    const option = document.createElement("option"); option.value = job.id; option.textContent = `${job.created_at} · ${job.state} · ${job.id.slice(0, 8)}`; $("saved-jobs").append(option);
  }
  $("saved-jobs").value = chosen || "";
}
async function perform(work) {
  const generation = ++viewGeneration, owner = ++actionSerial;
  clearCheckDisplay();
  busy = true; buttons();
  try { await work(generation); } catch (error) {
    if (generation === viewGeneration) { clearArtifactDisplay(); notice(error.message, true); }
  } finally { if (owner === actionSerial) { busy = false; buttons(); } }
}
async function loadExample(generation = viewGeneration) {
  const example = await api("/api/example");
  if (generation !== viewGeneration) return;
  inputBytes = new Blob([example.input_text], {type: "text/csv"});
  $("config-text").value = example.config_text; syncForm(example.config_text);
  $("override").checked = false; $("input-file").value = "";
  $("input-description").textContent = "Included example: series.csv (100 paired observations).";
  invalidate(); notice("Example loaded. Create a job, then validate and run.");
}
$("example").addEventListener("click", () => perform(loadExample));
$("input-file").addEventListener("change", () => {
  inputBytes = $("input-file").files[0] || null;
  $("input-description").textContent = inputBytes ? `${inputBytes.name} · ${inputBytes.size} bytes` : "Choose an input.";
  invalidate();
});
$("config-file").addEventListener("change", () => {
  invalidate();
  return perform(async (generation) => {
    const file = $("config-file").files[0]; if (!file) return;
    if (file.size > 65536) throw new Error("Configuration exceeds 65,536 bytes.");
    const text = await file.text();
    if (generation !== viewGeneration) return;
    $("config-text").value = text; syncForm(text); notice("Original configuration loaded. Check the input format before creating a new job.");
  });
});
for (const id of ["format", "source-column", "target-column", "candidates", "statistic", "null-model", "statistic-params", "null-params", "selection", "replicates", "alpha", "tolerance", "seed"]) {
  $(id).addEventListener("input", () => { invalidate(); try { $("config-text").value = formText(); notice("Plan edited. Create a new job to use it."); } catch (error) { $("config-text").value = ""; notice(error.message, true); } });
}
for (const id of ["budget", "override"]) $(id).addEventListener("input", invalidate);
$("config-text").addEventListener("input", invalidate);
$("config-text").addEventListener("change", () => { try { syncForm($("config-text").value); } catch (error) { notice(error.message, true); } });
$("create").addEventListener("click", () => perform(async (generation) => {
  if (!inputBytes) throw new Error("Choose an input or load the example first.");
  const admittedInput = inputBytes, admittedRevision = revision;
  const text = $("config-text").value;
  const budget = integer($("budget").value, "Byte budget");
  const job = await api("/api/jobs", {method: "POST", body: JSON.stringify({config_text: text, max_bytes: budget, allow_unattainable: $("override").checked})});
  if (generation === viewGeneration) {
    viewedId = job.id;
    selectedId = revision === admittedRevision ? job.id : null;
    showJob(job);
  }
  // Finish saving the already admitted immutable snapshot even if the form changed.
  const uploaded = await api(`/api/jobs/${job.id}/input`, {method: "PUT", body: admittedInput});
  if (generation !== viewGeneration) return;
  showJob(uploaded);
  await refreshJobs();
  if (generation !== viewGeneration) return;
  $("selection-note").textContent = revision === admittedRevision
    ? "This job uses its saved immutable input and configuration. Editing requires a new job."
    : "Input or plan changed during admission. The original snapshot was saved; create a new job to use your edits.";
}));
$("saved-jobs").addEventListener("change", () => perform(async (generation) => {
  const requestedId = $("saved-jobs").value || null;
  selectedId = viewedId = null;
  latestJob = null;
  if (!requestedId) { buttons(); return; }
  const job = await api(`/api/jobs/${requestedId}`);
  if (generation !== viewGeneration) return;
  $("config-text").value = job.config_text; syncForm(job.config_text);
  $("budget").value = job.max_bytes; $("override").checked = job.allow_unattainable;
  inputBytes = null; $("input-file").value = "";
  $("input-description").textContent = `Saved immutable ${job.format.toUpperCase()} input · ${job.input_sha256 || "upload incomplete"}. Choose new input to create another job.`;
  $("selection-note").textContent = "Selected saved job: original configuration shown. Saved observations have not been independently verified.";
  selectedId = viewedId = job.id;
  showJob(job);
}));
for (const action of ["validate", "run", "resume", "verify", "report", "export", "cancel"]) $(action).addEventListener("click", () => perform(async () => {
  const id = action === "cancel" ? viewedId : selectedId;
  if (!id) throw new Error("Create or select an unchanged saved job first.");
  const generation = viewGeneration;
  const job = await api(`/api/jobs/${id}/${action}`, {method: "POST", body: ""});
  if (generation !== viewGeneration || id !== (action === "cancel" ? viewedId : selectedId)) return;
  showJob(job);
  await refreshJobs();
}));
for (const [id, kind, filename] of [["preview-report", "report", "report.html"], ["download-report", "report", "report.html"], ["download-record", "record", "result.sqlite"], ["download-bundle", "bundle", "evidence.zip"]]) {
  $(id).addEventListener("click", () => perform(async (generation) => {
    const jobId = selectedId;
    if (!jobId) throw new Error("Select an unchanged saved job first.");
    const current = () => generation === viewGeneration && jobId === selectedId && jobId === viewedId;
    const response = await fetch(`/api/jobs/${jobId}/download/${kind}`, {headers: {"X-SelCal-Token": sessionToken}, cache: "no-store"});
    if (!current()) return;
    if (!response.ok) {
      const value = await response.json();
      if (!current()) return;
      throw new Error(value.error || `Download refused (${response.status}).`);
    }
    const blob = await response.blob();
    if (!current()) return;
    if (id === "preview-report") {
      const text = await blob.text();
      if (!current()) return;
      const previous = $("report-preview");
      const preview = document.createElement("iframe");
      preview.id = previous.id;
      preview.title = previous.title;
      preview.setAttribute("sandbox", "");
      preview.srcdoc = text;
      preview.hidden = false;
      previous.replaceWith(preview);
      $("artifact-status").textContent = "Checked HTML snapshot shown in a script-disabled sandbox. No replay or historical execution authentication.";
    } else {
      const name = attachmentName(response, filename);
      const expectedBytes = Number(response.headers.get("X-SelCal-Bytes"));
      const expectedSha = response.headers.get("X-SelCal-SHA256") || "";
      if (!(expectedBytes > 0) || !/^[0-9a-f]{64}$/.test(expectedSha)) {
        throw new Error(`Download not handed to the browser: the server response for ${name} did not state the checked size and SHA-256. The saved artifact is unchanged; retry.`);
      }
      if (!(blob.size > 0) || blob.size !== expectedBytes) {
        throw new Error(`Download not handed to the browser: received ${blob.size} bytes of ${name}, expected ${expectedBytes}. The saved artifact is unchanged; retry.`);
      }
      let digest;
      try {
        digest = await sha256Hex(blob);
      } catch (error) {
        throw new Error(`Download not handed to the browser: this browser could not compute the SHA-256 of ${name} (${error.message}). The saved artifact is unchanged; retry.`);
      }
      if (!current()) return;
      if (digest !== null && digest !== expectedSha) {
        throw new Error(`Download not handed to the browser: the received bytes of ${name} differ from the checked artifact. The saved artifact is unchanged; retry.`);
      }
      const url = URL.createObjectURL(blob);
      downloadURLs.set(url, undefined);
      const timer = setTimeout(() => releaseDownloadURL(url), DOWNLOAD_URL_LIFETIME_MS);
      if (downloadURLs.has(url)) downloadURLs.set(url, timer);
      while (downloadURLs.size > MAX_DOWNLOAD_URLS) releaseDownloadURL(downloadURLs.keys().next().value);
      const anchor = document.createElement("a");
      anchor.href = url; anchor.download = name;
      document.body.append(anchor);
      if (current()) anchor.click();
      anchor.remove();
      const saved = /^jobs\/[0-9a-f]{32}\/operations\/[0-9a-f]{32}\/[a-z.]+$/.test(response.headers.get("X-SelCal-Workspace-Path") || "") ? response.headers.get("X-SelCal-Workspace-Path") : "";
      const receipt = digest === null ? "the SHA-256 was not checked in this browser (Web Crypto unavailable), only the size" : "the bytes this page received have that SHA-256";
      if (current()) $("artifact-status").textContent = `Handed ${name} to the browser. The server re-checked the saved artifact and stated ${blob.size} bytes, SHA-256 ${expectedSha}; ${receipt}. This page cannot see whether the file was saved: look for ${name} with ${blob.size} bytes in the browser's downloads list. If a download manager such as IDM is installed, also check its completed list and save folder (ZIP files may be saved under Downloads\\Compressed).${saved ? ` If no download appears anywhere, copy the checked original from the workspace folder: ${saved}.` : ""}`;
    }
  }));
}
setInterval(async () => {
  if (pollInFlight || busy || !viewedId || !latestJob || latestJob.state !== "running") return;
  const id = viewedId, generation = viewGeneration;
  pollInFlight = true;
  try {
    const job = await api(`/api/jobs/${id}`);
    if (generation === viewGeneration && id === viewedId) {
      showJob(job);
      if (job.state !== "running") await refreshJobs();
    }
  } catch (error) {
    if (generation === viewGeneration && id === viewedId) notice(`Status unavailable: ${error.message}. Completion is unknown.`, true);
  } finally { pollInFlight = false; }
}, 600);
perform(async (generation) => { await refreshJobs(); if (generation === viewGeneration) await loadExample(generation); });
