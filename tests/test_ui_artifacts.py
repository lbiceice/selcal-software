"""Saved UI artifacts use actual CLI operations and checked, fixed downloads."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

import pytest
from _platform_support import directory_link_or_skip, symlink_or_skip
from test_ui_http import request
from test_ui_http import server as _server
from test_ui_jobs import admission, jobs_module
from test_ui_jobs import physical_tmp as _physical_tmp
from test_ui_process import wait_job
from test_ui_verify import files_at, saved_job

physical_tmp = _physical_tmp
server = _server


@pytest.mark.parametrize("action", ["report", "export"])
def test_actual_saved_job_generates_artifact(physical_tmp, action):
    with saved_job(physical_tmp) as (manager, job_id, original):
        manager.start(job_id, action)
        result = wait_job(manager, job_id)
        assert result["state"] == original["state"]
        assert result["result"]["command"] == action
        assert result["result"]["data"]["replay"] == "NOT_PERFORMED"
        if action == "export":
            assert result["result"]["data"]["member_count"] == 11


def generated(manager, job_id, action):
    manager.start(job_id, action)
    value = wait_job(manager, job_id)
    assert value["state"] in {"complete", "not_evaluable"}, value
    kind = "bundle" if action == "export" else "report"
    path = manager.workspace / "jobs" / job_id
    reference = json.loads((path / (kind + ".json")).read_bytes())
    return path, reference, path / "operations" / reference["operation_id"], value


@pytest.mark.parametrize("form", ["csv", "npz"])
@pytest.mark.parametrize(
    "mode", ["complete", "null_bind", "observed_statistic_scan", "replicate_execution"]
)
def test_real_artifacts_match_separate_cli_and_checked_downloads(physical_tmp, form, mode):
    with saved_job(physical_tmp, form=form, mode=mode) as (manager, job_id, original):
        path = manager.workspace / "jobs" / job_id
        source_ref = json.loads((path / "record.json").read_bytes())
        source = path / "operations" / source_ref["operation_id"] / "result.sqlite"
        before = files_at(path / "operations")
        raw, media, name, location = manager.download(job_id, "record")
        assert raw == source.read_bytes() and media == "application/octet-stream"
        assert manager.workspace / location == source
        # R12 download retest: names carry the job and operation, never a bare shared name.
        assert name == f"selcal-{job_id[:8]}-{source_ref['operation_id'][:8]}-result.sqlite"
        for action in ("report", "export"):
            _, reference, operation, observed = generated(manager, job_id, action)
            kind = "report" if action == "report" else "bundle"
            raw, media, name, location = manager.download(job_id, kind)
            assert (manager.workspace / location).read_bytes() == raw
            assert location.startswith(f"jobs/{job_id}/operations/{operation.name}/")
            assert reference == {
                "schema": "selcal.ui-artifact-reference.v1",
                "kind": kind,
                "operation_id": operation.name,
                "record": source_ref,
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
            direct_path = physical_tmp / (action + "-direct")
            direct = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "selcal",
                    action,
                    str(source),
                    str(direct_path),
                    "--max-bytes",
                    original["max_bytes"],
                ],
                capture_output=True,
                timeout=30,
            )
            assert direct.returncode == (0 if mode == "complete" else 7), direct.stderr
            assert json.loads(direct.stdout) == observed["result"]
            assert observed["state"] == original["state"]
            if mode != "complete":
                assert observed["result"]["data"]["p_value"] is None
                assert observed["result"]["data"]["reject_null"] is None
            if kind == "report":
                assert media == "text/html"
                assert name == f"selcal-{job_id[:8]}-{operation.name[:8]}-report.html"
                assert raw == direct_path.read_bytes() == (operation / "report.html").read_bytes()
            else:
                assert media == "application/zip"
                assert name == f"selcal-{job_id[:8]}-{operation.name[:8]}-evidence.zip"
                with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                    assert not archive.comment and len(archive.infolist()) == 11
                    assert {i.filename: archive.read(i) for i in archive.infolist()} == files_at(
                        direct_path
                    )
                    assert all(
                        i.compress_type == zipfile.ZIP_STORED
                        and not i.extra
                        and not i.comment
                        and not i.flag_bits & 1
                        and "/" not in i.filename
                        for i in archive.infolist()
                    )
            assert "replay" in observed["message"].lower()
            assert "Run and record saving" not in observed["message"]
        assert json.loads((path / "record.json").read_bytes()) == source_ref
        assert len(list(path.glob("operations/*/result.sqlite"))) == 1
        assert all((path / "operations" / n).read_bytes() == b for n, b in before.items())


@pytest.mark.parametrize("action", ["report", "export"])
def test_artifact_repeat_restart_resume_preserves_old_files(physical_tmp, action):
    with saved_job(physical_tmp) as (manager, job_id, _):
        path, first, _, _ = generated(manager, job_id, action)
        before = files_at(path / "operations")
        _, second, _, _ = generated(manager, job_id, action)
        assert first["operation_id"] != second["operation_id"]
        kind = "report" if action == "report" else "bundle"
        manager.close()
        reopened = jobs_module().JobManager(manager.workspace)
        try:
            assert reopened.download(job_id, kind)[0]
            reopened.start(job_id, "resume")
            assert wait_job(reopened, job_id)["state"] == "complete"
            with pytest.raises(jobs_module().UIError):
                reopened.download(job_id, kind)
            assert json.loads((path / (kind + ".json")).read_bytes()) == second
            generated(reopened, job_id, action)
            assert reopened.download(job_id, kind)[0]
            assert all((path / "operations" / n).read_bytes() == b for n, b in before.items())
        finally:
            reopened.close()


@pytest.mark.parametrize(
    "action,field,value",
    [
        ("report", "replay", "MATCH"),
        ("report", "replay", None),
        ("export", "replay", "MATCH"),
        ("export", "verification_scope", "input_plan_result_consistency"),
        ("export", "bundle_schema", "other"),
        ("export", "member_count", True),
        ("export", "member_count", 11.0),
        ("export", "member_count", 10),
        ("export", "historical_execution_authenticated", 0),
        ("export", "historical_execution_authenticated", True),
    ],
)
def test_artifact_terminal_requires_actual_command_specific_fields(
    physical_tmp, action, field, value
):
    with saved_job(physical_tmp) as (manager, job_id, _):
        path = manager.workspace / "jobs" / job_id
        (source,) = path.glob("operations/*/result.sqlite")
        direct = subprocess.run(
            [
                sys.executable,
                "-m",
                "selcal",
                action,
                str(source),
                str(physical_tmp / "direct"),
                "--max-bytes",
                "1048576",
            ],
            capture_output=True,
            timeout=30,
        )
        assert direct.returncode == 0, direct.stderr
        terminal = json.loads(direct.stdout)
        assert jobs_module()._terminal(direct.stdout, action, 0) == terminal
        if value is None:
            terminal["data"].pop(field)
        else:
            terminal["data"][field] = value
        with pytest.raises(jobs_module().UIError):
            jobs_module()._terminal(json.dumps(terminal).encode(), action, 0)


@pytest.mark.parametrize("kind", ["record", "report", "bundle"])
@pytest.mark.parametrize(
    "damage",
    ["input", "config", "record", "ref", "missing", "linked", "parent", "bytes", "foreign_ref"],
)
def test_download_damaged_subject_or_artifact_is_refused(physical_tmp, kind, damage):
    with saved_job(physical_tmp) as (manager, job_id, _):
        path = manager.workspace / "jobs" / job_id
        if kind != "record":
            _, ref, operation, _ = generated(
                manager, job_id, "report" if kind == "report" else "export"
            )
        else:
            ref = json.loads((path / "record.json").read_bytes())
            operation = path / "operations" / ref["operation_id"]
        artifact = (
            operation
            / {"record": "result.sqlite", "report": "report.html", "bundle": "evidence.zip"}[kind]
        )
        reference = path / (kind + ".json")
        if damage == "input":
            (path / "input.csv").write_bytes(b"changed")
        elif damage == "config":
            (path / "request.json").write_bytes(b"{}")
        elif damage == "record":
            (next(path.glob("operations/*/result.sqlite"))).write_bytes(b"changed")
        elif damage == "ref":
            ref["extra"] = True
            reference.write_text(json.dumps(ref), encoding="utf-8")
        elif damage == "bytes":
            artifact.write_bytes(artifact.read_bytes() + b"changed")
        elif damage == "foreign_ref":
            if kind == "record":
                ref["operation_id"] = "0" * 32
            else:
                ref["record"]["operation_id"] = "0" * 32
            reference.write_text(json.dumps(ref), encoding="utf-8")
        elif damage == "parent":
            retained = operation.with_name("retained")
            operation.rename(retained)
            directory_link_or_skip(operation, retained)
        else:
            retained = artifact.with_suffix(".retained")
            artifact.rename(retained)
            if damage == "linked":
                symlink_or_skip(artifact, retained)
        with pytest.raises(jobs_module().UIError):
            manager.download(job_id, kind)
        assert manager._active is None


@pytest.mark.parametrize("damage", ["missing", "extra", "linked", "content", "zip", "foreign"])
def test_bundle_download_rechecks_members_source_and_zip(physical_tmp, damage):
    with saved_job(physical_tmp) as (manager, job_id, _):
        path, ref, operation, _ = generated(manager, job_id, "export")
        bundle = operation / "bundle"
        if damage == "missing":
            (bundle / "result.json").rename(operation / "retained-result")
        elif damage == "extra":
            (bundle / "extra").write_bytes(b"retained")
        elif damage == "linked":
            (bundle / "result.json").rename(operation / "retained-result")
            symlink_or_skip(bundle / "result.json", operation / "retained-result")
        elif damage == "content":
            (bundle / "summary.csv").write_bytes(b"changed")
        elif damage == "foreign":
            from selcal.workflow import read_workflow
            from selcal.workflow_export import export_record, verify_export
            from selcal.workflow_store import read_record, write_record

            foreign = operation / "foreign"
            source = path / "operations" / ref["record"]["operation_id"] / "result.sqlite"
            members = read_record(source, max_bytes=1048576)
            members["metadata"] = json.dumps(json.loads(members["metadata"]), indent=2).encode()
            foreign_record = operation / "foreign.sqlite"
            write_record(foreign_record, members, max_bytes=1048576)
            assert (
                read_workflow(foreign_record, max_bytes=1048576).metadata
                == read_workflow(source, max_bytes=1048576).metadata
            )
            exported = export_record(foreign_record, foreign, max_bytes=1048576)
            assert verify_export(foreign, max_bytes=1048576) == exported
            # Valid content with identical raw/plan identities is still not these four source bytes.
            assert (foreign / "metadata.json").read_bytes() != (
                bundle / "metadata.json"
            ).read_bytes()
            bundle.rename(operation / "retained-bundle")
            foreign.rename(bundle)
            raw = jobs_module()._transport_zip(files_at(bundle), 1048576)
            (operation / "evidence.zip").write_bytes(raw)
            ref.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
            (path / "bundle.json").write_text(json.dumps(ref), encoding="utf-8")
        else:
            with zipfile.ZipFile(operation / "evidence.zip", "w") as archive:
                archive.writestr("unverified", b"not the verified bundle")
            raw = (operation / "evidence.zip").read_bytes()
            ref.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
            (path / "bundle.json").write_text(json.dumps(ref), encoding="utf-8")
        with pytest.raises(jobs_module().UIError):
            manager.download(job_id, "bundle")


@pytest.mark.parametrize("action", ["report", "export"])
@pytest.mark.parametrize("mutation", ["input", "config", "record", "close", "cap"])
def test_generation_failure_cannot_promote_artifact(physical_tmp, monkeypatch, action, mutation):
    with saved_job(physical_tmp) as (manager, job_id, _):
        path, _, _, _ = generated(manager, job_id, action)
        kind = "report" if action == "report" else "bundle"
        previous = (path / (kind + ".json")).read_bytes()
        original = jobs_module()._terminal

        def checked(raw, command, code):
            value = original(raw, command, code)
            if command == action:
                if mutation == "close":
                    raise OSError("injected final close failure")
                if mutation == "cap":
                    metadata = json.loads((path / "metadata.json").read_bytes())
                    metadata["max_bytes"] = str(int(metadata["max_bytes"]) + 1)
                    (path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
                    return value
                target = {
                    "input": path / "input.csv",
                    "config": path / "request.json",
                    "record": next(path.glob("operations/*/result.sqlite")),
                }[mutation]
                target.write_bytes(target.read_bytes() + b"changed before publication")
            return value

        monkeypatch.setattr(jobs_module(), "_terminal", checked)
        manager.start(job_id, action)
        # Damaged configuration correctly makes ordinary GET refuse; inspect the
        # retained operation observation after the owned worker has finished.
        deadline = time.monotonic() + 30
        while manager._active is not None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert manager._active is None
        observed = manager._observations[job_id]
        assert observed["state"] == "failed" and observed["result"] is None
        assert (path / (kind + ".json")).read_bytes() == previous


@pytest.mark.parametrize("cap", [sys.maxsize, sys.maxsize + 1, 10**100])
def test_generation_and_download_large_legal_cap(physical_tmp, cap):
    inp, config = admission(physical_tmp)
    config["max_bytes"] = str(cap)
    manager = jobs_module().JobManager(physical_tmp / "work")
    try:
        job_id = manager.admit(config)["id"]
        manager.upload(job_id, io.BytesIO(inp.read_bytes()), inp.stat().st_size)
        manager.start(job_id, "run")
        assert wait_job(manager, job_id)["state"] == "complete"
        for action, kind in (("report", "report"), ("export", "bundle")):
            generated(manager, job_id, action)
            assert manager.download(job_id, kind)[0]
    finally:
        manager.close()


def test_authenticated_artifact_routes_headers_and_methods(server, physical_tmp):
    inp, config = admission(physical_tmp)
    job_id = server.jobs.admit(config)["id"]
    server.jobs.upload(job_id, io.BytesIO(inp.read_bytes()), inp.stat().st_size)
    server.jobs.start(job_id, "run")
    wait_job(server.jobs, job_id)
    for action in ("report", "export"):
        route = f"/api/jobs/{job_id}/{action}"
        assert request(server, "POST", route, b"")[0] == 200
        assert wait_job(server.jobs, job_id)["state"] == "complete"
    for kind, mime in (
        ("record", "application/octet-stream"),
        ("report", "text/html"),
        ("bundle", "application/zip"),
    ):
        route = f"/api/jobs/{job_id}/download/{kind}"
        status, headers, raw = request(server, "GET", route)
        assert status == 200 and raw
        assert headers["Content-Type"].split(";")[0] == mime
        assert int(headers["Content-Length"]) == len(raw)
        assert headers["Content-Disposition"].startswith("attachment;")
        assert headers["Cache-Control"] == "no-store"
        if kind != "report":
            assert "charset" not in headers["Content-Type"]
        for h in (
            {"X-SelCal-Token": ""},
            {"X-SelCal-Token": "bad"},
            {"Host": "localhost"},
            {"Origin": "https://foreign.test"},
        ):
            assert request(server, "GET", route, headers=h)[0] == 403
        assert request(server, "GET", route, b"x")[0] == 413
        assert request(server, "GET", route + "?token=bad")[0] == 404
        assert request(server, "POST", route, b"")[0] == 404


def test_artifact_ui_controls_sandbox_and_exact_report_style_hash(physical_tmp):
    import base64
    import re

    import selcal
    from selcal.ui import _HEADERS

    web = Path(selcal.__file__).resolve().parent / "web"
    html = (web / "index.html").read_text(encoding="utf-8")
    for label in (
        "Generate HTML report",
        "Export evidence bundle",
        "Preview HTML report",
        "Download HTML report",
        "Download result record",
        "Download evidence ZIP",
    ):
        assert label in html
    assert re.search(r'<iframe[^>]*sandbox=""', html)
    assert "allow-scripts" not in html and "allow-same-origin" not in html
    # Check actual CLI HTML, not Python source syntax or a copied style string.
    with saved_job(physical_tmp) as (manager, job_id, _):
        _, _, operation, _ = generated(manager, job_id, "report")
        actual = (operation / "report.html").read_text(encoding="utf-8")
    style = re.search(r"<style>(.*?)</style>", actual, re.S).group(1)
    digest = base64.b64encode(hashlib.sha256(style.encode()).digest()).decode()
    assert "'sha256-" + digest + "'" in _HEADERS["Content-Security-Policy"]
    assert "unsafe-inline" not in _HEADERS["Content-Security-Policy"]


@pytest.mark.parametrize("action", ["report", "export"])
def test_nested_record_reference_is_strict_not_numeric_equality(physical_tmp, action):
    with saved_job(physical_tmp) as (manager, job_id, _):
        path, ref, _, _ = generated(manager, job_id, action)
        kind = "report" if action == "report" else "bundle"
        ref["record"]["bytes"] = float(ref["record"]["bytes"])
        (path / (kind + ".json")).write_text(json.dumps(ref), encoding="utf-8")
        assert manager.get(job_id)["artifacts"][kind]["available"] is False
        with pytest.raises(jobs_module().UIError):
            manager.download(job_id, kind)


@pytest.mark.parametrize(
    "scenario",
    [
        "success",
        "late_blob",
        "late_text",
        "late_text_after_new_preview",
        "late_error",
        "late_action_error",
        "fresh_error",
        "config_oversize",
        "config_read_error",
        "config_late",
        # R12 download retest (empty evidence ZIP in Chrome): refuse short or altered bodies,
        # and never revoke a download URL at click time or on the next action.
        "short_body",
        "wrong_digest",
        "deferred_revoke",
        # R13 Windows return (W13-02, W13-05): the link policy is "at most six, each released
        # after 120 s, on eviction or on page close"; header, Crypto and view-change edges.
        "url_cap_and_pagehide",
        "missing_headers",
        "no_crypto",
        "digest_error",
        "switch_during_digest",
    ],
)
def test_actual_javascript_artifact_async_guards(scenario):
    import selcal

    node = shutil.which("node")
    if node is None:
        pytest.skip("JavaScript runtime unavailable; browser acceptance remains separate")
    web = Path(selcal.__file__).resolve().parent / "web"
    harness = r"""
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const elements=new Map();let downloads=[],revoked=[],urls=[],timers=[],delays=[];
let windowHandlers={},release,posted;
let now=0,deadlines=[];
function advanceClock(ms) {
 const until=now+ms;
 while(true) {
  let next=-1;
  for(let i=0;i<timers.length;i++)
   if(timers[i]&&deadlines[i]<=until&&(next<0||deadlines[i]<deadlines[next]))next=i;
  if(next<0)break;
  now=deadlines[next];const callback=timers[next];timers[next]=null;callback();
 }
 now=until;
}
function connect(node,connected) {
 node.isConnected=connected;
 if(node.id){if(connected)elements.set(node.id,node);
  else if(elements.get(node.id)===node)elements.delete(node.id)}
 for(const child of node.options)connect(child,connected);
}
function createElement(tag) {
 const attributes=new Map();
 return {tagName:tag.toUpperCase(),isConnected:false,srcdoc:'',hidden:false,
 value:'',checked:false,disabled:false,options:[],files:[],
 textContent:'',handlers:{},classList:{toggle(){}},addEventListener(k,f){this.handlers[k]=f},
 get id(){return this.getAttribute('id')||''},set id(value){this.setAttribute('id',value)},
 get title(){return this.getAttribute('title')||''},
 set title(value){this.setAttribute('title',value)},
 setAttribute(name,value){
  if(name==='id'&&this.isConnected&&elements.get(this.id)===this)elements.delete(this.id);
  attributes.set(name,String(value));
  if(name==='id'&&this.isConnected)elements.set(this.id,this);
 },
 getAttribute(name){return attributes.has(name)?attributes.get(name):null},
 replaceChildren(){for(const child of this.options)connect(child,false);this.options=[]},
 append(...children){this.options.push(...children);if(this.isConnected)
  for(const child of children)connect(child,true)},
 replaceWith(next){assert(this.isConnected);assert(!next.isConnected);
  connect(this,false);connect(next,true)},
 remove(){connect(this,false)},
 click(){downloads.push({name:this.download,url:this.href})}};
}
// Only attached fixture nodes are discoverable by ID; new elements stay detached
// until append/replaceWith, including when they have the same ID as an old frame.
for(const match of fs.readFileSync(INDEX,'utf8').matchAll(/<([a-z][\w-]*)\b([^>]*)>/g)){
 const id=/\bid="([^"]*)"/.exec(match[2]);if(!id)continue;
 const node=createElement(match[1]);
 for(const attr of match[2].matchAll(/([\w-]+)="([^"]*)"/g))node.setAttribute(attr[1],attr[2]);
 node.hidden=/\bhidden\b/.test(match[2]);connect(node,true);
}
const element=id=>elements.get(id)||null;
const body=createElement('body');connect(body,true);
const example=EXAMPLE,scenario=SCENARIO;
const saved={id:'saved',state:'complete',input_status:'complete',format:'csv',config_text:example,
 max_bytes:'1048576',allow_unattainable:false,record_target:{available:true},
 artifacts:{report:{available:true},bundle:{available:true}},message:'saved'};
const newer={...saved,id:'new-job',message:'new job selected'};
const response=x=>({ok:true,json:async()=>x});
const context=vm.createContext({URLSearchParams,Blob,Object,JSON,console,
 URL:{createObjectURL(b){urls.push(b);return 'blob:checked-'+urls.length},
  revokeObjectURL(u){revoked.push(u)}},
 setTimeout(f,delay){delays.push(delay);
  if(scenario==='deferred_revoke'||scenario==='url_cap_and_pagehide'){
   timers.push(f);deadlines.push(now+delay);return timers.length}f()},
 clearTimeout(id){if(id)timers[id-1]=null},
 addEventListener(kind,f){windowHandlers[kind]=f},
 crypto:scenario==='no_crypto'?undefined:scenario==='digest_error'?
  {subtle:{digest:async()=>{throw new Error('digest unavailable')}}}:
  scenario==='switch_during_digest'?{subtle:{digest:(kind,bytes)=>new Promise(resolve=>{
   release=()=>resolve(globalThis.crypto.subtle.digest(kind,bytes))})}}:globalThis.crypto,
 location:{hash:'#token=local',pathname:'/'},history:{replaceState(){}},
 sessionStorage:{setItem(){},getItem(){return 'local'}},setInterval(){},
 document:{getElementById:element,createElement,body},
 fetch:async(path,options)=>{
  if(path==='/api/example')return response({input_text:'x,y\n0,1\n',config_text:example});
  if(path==='/api/jobs')return response({jobs:[saved]});
  if(path==='/api/jobs/new-job')return response(newer);
  posted={path,options};assert.equal(options.headers['X-SelCal-Token'],'local');
  assert(!path.includes('?')&&!path.includes('token'));
  if(scenario==='late_action_error')return new Promise((_,reject)=>{
   release=()=>reject(new Error('old action error'))});
  if(scenario==='fresh_error')return {ok:false,json:async()=>({error:'changed bytes refused'})};
  if(scenario==='late_error')return {ok:false,json:()=>new Promise(resolve=>{
   release=()=>resolve({error:'old error'})})};
  if(scenario==='late_text_after_new_preview')return {ok:true,blob:async()=>
   path==='/api/jobs/saved/download/report'?{text:()=>new Promise(resolve=>{
    release=()=>resolve('<h1>old</h1>')})}:new Blob(['<h1>new checked</h1>'])};
  const kind=path.split('/').pop(),body='<h1>checked</h1>';
  const file={record:'result.sqlite',report:'report.html',bundle:'evidence.zip'}[kind];
  const sha=require('node:crypto').createHash('sha256').update(body).digest('hex');
  const headers={'Content-Disposition':`attachment; filename="selcal-saved000-op000000-${file}"`,
   'X-SelCal-Bytes':String(body.length+(scenario==='short_body'?5:0)),
   'X-SelCal-SHA256':scenario==='wrong_digest'?'0'.repeat(64):sha,
   'X-SelCal-Workspace-Path':`jobs/${'a'.repeat(32)}/operations/${'b'.repeat(32)}/${file}`};
  if(scenario==='missing_headers'&&kind==='bundle')delete headers['X-SelCal-SHA256'];
  if(scenario==='missing_headers'&&kind==='record')delete headers['X-SelCal-Bytes'];
  return {ok:true,headers:{get:name=>headers[name]??null},
   blob:()=>scenario==='late_blob'?new Promise(resolve=>{
   release=()=>resolve(new Blob(['checked HTML']))}):
    Promise.resolve(scenario==='late_text'?{text:()=>new Promise(resolve=>{
     release=()=>resolve('<h1>old</h1>')})}:new Blob(['<h1>checked</h1>']))};
 }});
context.saved=saved;vm.runInContext(fs.readFileSync(SCRIPT,'utf8'),context);
(async()=>{
 await new Promise(setImmediate);
 vm.runInContext('selectedId=viewedId="saved";showJob(saved)',context);
 if(scenario.startsWith('config_')){
  element('config-file').files=[{size:scenario==='config_oversize'?65537:5,
   text:()=>scenario==='config_late'?new Promise(resolve=>{release=()=>resolve(example)}):
    Promise.reject(new Error('read failed'))}];
  const pending=element('config-file').handlers.change();await new Promise(setImmediate);
  if(scenario==='config_late'){
   vm.runInContext('viewGeneration++;$("config-text").value="new plan";notice("new view")',context);
   release();
  }
  await pending;
  if(scenario==='config_late'){
   assert.equal(element('config-text').value,'new plan');
   assert.equal(element('notice').textContent,'new view');
  }else assert(element('notice').textContent.includes(
   scenario==='config_oversize'?'65,536':'read failed'));
  return;
 }
 const controls=['report','export','preview-report','download-report',
  'download-record','download-bundle'];
 for(const id of controls){
  assert(element(id).handlers.click,'Missing actual handler '+id);
  assert.equal(element(id).disabled,false,id);
 }
 vm.runInContext('invalidate()',context);
 for(const id of controls)assert(element(id).disabled);
 vm.runInContext('selectedId=viewedId="saved";showJob(saved)',context);
 if(scenario==='success'){
  async function preview(){
   const previous=element('report-preview');
   await element('preview-report').handlers.click();
   const current=element('report-preview');
   assert.notEqual(current,previous,'Each successful preview needs a fresh iframe');
   assert.equal(previous.isConnected,false);assert.equal(current.isConnected,true);
   assert.equal(previous.srcdoc,'');assert.equal(previous.hidden,true);
   assert.equal(current.tagName,'IFRAME');assert.equal(current.id,'report-preview');
   assert.equal(current.title,'Checked saved HTML report');
   assert.equal(current.getAttribute('sandbox'),'');
   assert.equal(current.srcdoc,'<h1>checked</h1>');assert.equal(current.hidden,false);
   return current;
  }
  await preview();await preview();
  for(const [id,name]of [['download-record','selcal-saved000-op000000-result.sqlite'],
   ['download-report','selcal-saved000-op000000-report.html'],
   ['download-bundle','selcal-saved000-op000000-evidence.zip']]){
   const current=await preview();
   await element(id).handlers.click();assert.equal(downloads.at(-1).name,name);
   assert.equal(element('report-preview'),current);
   assert.equal(current.srcdoc,'');assert.equal(current.hidden,true);
  }
  assert.equal(revoked.length,urls.length);assert.equal(downloads.length,3);
  const status=element('artifact-status').textContent;
  const sha=require('node:crypto').createHash('sha256').update('<h1>checked</h1>').digest('hex');
  assert(status.startsWith('Handed selcal-saved000-op000000-evidence.zip to the browser.'),status);
  assert(status.includes(
   `The server re-checked the saved artifact and stated 16 bytes, SHA-256 ${sha}`),status);
  assert(status.includes('the bytes this page received have that SHA-256'),status);
  assert(status.includes('IDM')&&status.includes('Downloads\\Compressed'),status);
  assert(status.endsWith('If no download appears anywhere, copy the checked original from the '+
   `workspace folder: jobs/${'a'.repeat(32)}/operations/${'b'.repeat(32)}/evidence.zip.`),status);
  assert(delays.every(delay=>delay===120000),String(delays));
  const current=await preview();
  element('budget').handlers.input();
  assert.equal(element('report-preview'),current);
  assert.equal(current.srcdoc,'');assert.equal(current.hidden,true);
  assert.equal(element('artifact-status').textContent,'No artifact requested in this view.');
 }else if(scenario==='short_body'||scenario==='wrong_digest'){
  await element('download-bundle').handlers.click();
  assert.equal(downloads.length,0,'a short or altered body must not reach the browser');
  assert.equal(urls.length,0);
  assert(element('notice').textContent.includes('Download not handed to the browser'),
   element('notice').textContent);
 }else if(scenario==='deferred_revoke'){
  await element('download-bundle').handlers.click();
  assert.equal(downloads.length,1);assert.equal(revoked.length,0,'revoked at click time');
  await element('download-record').handlers.click();
  assert.equal(downloads.length,2);
  assert.equal(revoked.length,0,'the next action revoked a download still in progress');
  element('budget').handlers.input();
  assert.equal(revoked.length,0,'invalidating the view revoked a download in progress');
  assert.deepEqual(delays,[120000,120000],'each download URL lives exactly 120 s');
  advanceClock(119999);
  assert.equal(revoked.length,0,'download URLs must survive until the 120 s deadline');
  assert.equal(timers.filter(Boolean).length,2,'both expiry timers must still be pending');
  advanceClock(1);
  assert.deepEqual(revoked,['blob:checked-1','blob:checked-2'],
   'the 120 s deadline must release every download URL');
  assert(timers.every(f=>f===null),'expiry must clear every pending timer');
 }else if(scenario==='url_cap_and_pagehide'){
  const order=['download-bundle','download-record','download-report'];
  for(let i=0;i<6;i++)await element(order[i%3]).handlers.click();
  assert.equal(downloads.length,6);assert.equal(revoked.length,0,'six links must all stay');
  await element('download-bundle').handlers.click();
  assert.deepEqual(revoked,['blob:checked-1'],'the seventh link evicts only the oldest');
  assert.equal(timers[0],null,'an evicted link must cancel its own timer');
  assert.deepEqual(delays,Array(7).fill(120000));
  windowHandlers.pagehide();
  assert.deepEqual(revoked,[1,2,3,4,5,6,7].map(n=>'blob:checked-'+n),'page close releases all');
  assert(timers.every(f=>f===null),'page close must cancel every pending timer');
 }else if(scenario==='missing_headers'){
  for(const id of ['download-bundle','download-record']){
   await element(id).handlers.click();
   assert(element('notice').textContent.includes('did not state the checked size and SHA-256'),
    element('notice').textContent);
  }
  assert.equal(downloads.length,0);assert.equal(urls.length,0);
 }else if(scenario==='no_crypto'){
  await element('download-bundle').handlers.click();
  assert.equal(downloads.length,1,'without Web Crypto a size-checked file is still delivered');
  const status=element('artifact-status').textContent;
  assert(status.includes(
   'the SHA-256 was not checked in this browser (Web Crypto unavailable), only the size'),status);
 }else if(scenario==='digest_error'){
  await element('download-bundle').handlers.click();
  assert.equal(downloads.length,0);assert.equal(urls.length,0);
  assert(element('notice').textContent.includes('could not compute the SHA-256')&&
   element('notice').textContent.includes('digest unavailable'),element('notice').textContent);
 }else if(scenario==='switch_during_digest'){
  const before=element('artifact-status').textContent;
  const pending=element('download-bundle').handlers.click();
  for(let i=0;i<20&&typeof release!=='function';i++)await new Promise(setImmediate);
  assert.equal(typeof release,'function','the download must be pending inside the digest');
  vm.runInContext('selectedId=viewedId="new-job";viewGeneration++;notice("new view");',context);
  release();await pending;
  assert.equal(downloads.length,0,'an old job download must not start after a view change');
  assert.equal(urls.length,0);assert.equal(element('notice').textContent,'new view');
  assert.equal(element('artifact-status').textContent,before);
 }else if(scenario==='late_text_after_new_preview'){
  const pending=element('preview-report').handlers.click();await new Promise(setImmediate);
  assert.equal(typeof release,'function','Old response must be pending at blob.text()');
  element('budget').handlers.input();
  element('saved-jobs').value='new-job';await element('saved-jobs').handlers.change();
  await element('preview-report').handlers.click();
  const current=element('report-preview'),status=element('artifact-status').textContent,
   notice=element('notice').textContent;
  assert.equal(current.srcdoc,'<h1>new checked</h1>');assert.equal(current.hidden,false);
  assert(status.includes('Checked HTML snapshot'));assert.equal(notice,'new job selected');
  release();await pending;
  assert.equal(element('report-preview'),current,'Stale text replaced the newer iframe');
  assert.equal(current.srcdoc,'<h1>new checked</h1>');assert.equal(current.hidden,false);
  assert.equal(element('artifact-status').textContent,status);
  assert.equal(element('notice').textContent,notice);
  assert.equal(downloads.length,0);
 }else{
  element('report-preview').srcdoc='old green report';element('report-preview').hidden=false;
  const id=scenario==='late_action_error'?'report':
   scenario==='late_blob'?'download-bundle':'preview-report';
  const pending=element(id).handlers.click();await new Promise(setImmediate);
  if(scenario!=='fresh_error'){
   vm.runInContext('selectedId=viewedId="new-job";viewGeneration++;notice("new view");',context);
   release();
  }
  await pending;
  assert.equal(element('report-preview').srcdoc,'');assert.equal(downloads.length,0);
  if(scenario==='fresh_error')
   assert(element('notice').textContent.includes('changed bytes refused'));
  else assert.equal(element('notice').textContent,'new view',
   'stale completion or error polluted new job');
 }
})().catch(e=>{console.error(e);process.exitCode=1});
"""
    example = (web / "pearson.json").read_text(encoding="utf-8")
    harness = harness.replace("EXAMPLE", json.dumps(example))
    harness = (
        harness.replace("SCENARIO", json.dumps(scenario))
        .replace("SCRIPT", json.dumps(str(web / "app.js")))
        .replace("INDEX", json.dumps(str(web / "index.html")))
    )
    result = subprocess.run([node, "-e", harness], capture_output=True, text=True,
        errors="replace", timeout=10)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("action", ["report", "export"])
@pytest.mark.parametrize("stop", ["cancel", "close"])
def test_actual_owned_artifact_child_cancellation_preserves_previous(physical_tmp, action, stop):
    with saved_job(physical_tmp) as (manager, job_id, _):
        path, _, _, _ = generated(manager, job_id, action)
        kind = "report" if action == "report" else "bundle"
        previous = (path / (kind + ".json")).read_bytes()
        previous_files = files_at(path / "operations")
        manager.start(job_id, action)
        with pytest.raises(jobs_module().UIError) as refused:
            manager.download(job_id, kind)
        assert refused.value.status == 409
        if stop == "cancel":
            manager.cancel(job_id)
        else:
            manager.close()
        assert manager._active is None
        assert manager._observations[job_id]["state"] == "interrupted"
        assert (path / (kind + ".json")).read_bytes() == previous
        assert all(
            (path / "operations" / name).read_bytes() == raw for name, raw in previous_files.items()
        )


@pytest.mark.parametrize("action", ["report", "export"])
def test_reference_stream_close_failure_prevents_publication(physical_tmp, monkeypatch, action):
    with saved_job(physical_tmp) as (manager, job_id, _):
        path, _, _, _ = generated(manager, job_id, action)
        kind = "report" if action == "report" else "bundle"
        previous = (path / (kind + ".json")).read_bytes()
        original = Path.open

        class CloseFailure:
            def __init__(self, stream):
                self.stream = stream

            def __enter__(self):
                return self.stream

            def __exit__(self, *args):
                self.stream.close()
                raise OSError("injected reference stream close failure")

        def opened(target, *args, **kwargs):
            stream = original(target, *args, **kwargs)
            if (
                target.parent == path
                and target.name.startswith(kind + "-")
                and target.suffix == ".tmp"
            ):
                return CloseFailure(stream)
            return stream

        monkeypatch.setattr(Path, "open", opened)
        manager.start(job_id, action)
        observed = wait_job(manager, job_id)
        assert observed["state"] == "failed" and "close failure" in observed["message"]
        assert (path / (kind + ".json")).read_bytes() == previous
        assert list(path.glob(kind + "-*.tmp")), "Failed output retained, not published"


@pytest.mark.parametrize("kind", ["report", "bundle"])
@pytest.mark.parametrize(
    "damage", ["missing", "linked", "large", "duplicate", "float", "traversal"]
)
def test_derivative_reference_not_guessed_or_adopted(physical_tmp, kind, damage):
    with saved_job(physical_tmp) as (manager, job_id, _):
        path, ref, _, _ = generated(manager, job_id, "report" if kind == "report" else "export")
        target = path / (kind + ".json")
        previous_files = files_at(path / "operations")
        if damage in {"missing", "linked"}:
            target.rename(path / "retained-reference")
            if damage == "linked":
                symlink_or_skip(target, path / "retained-reference")
        elif damage == "large":
            target.write_bytes(b" " * 8193)
        elif damage == "duplicate":
            target.write_text(
                target.read_text(encoding="utf-8").replace('"kind":', '"kind":"report","kind":'),
                encoding="utf-8",
            )
        else:
            if damage == "float":
                ref["bytes"] = float(ref["bytes"])
            else:
                ref["operation_id"] = "../outside"
            target.write_text(json.dumps(ref), encoding="utf-8")
        assert not manager.get(job_id)["artifacts"][kind]["available"]
        with pytest.raises(jobs_module().UIError):
            manager.download(job_id, kind)
        assert files_at(path / "operations") == previous_files


def test_exact_member_aggregate_and_transport_overhead_bounds(physical_tmp):
    with saved_job(physical_tmp) as (manager, job_id, _):
        path, ref, operation, _ = generated(manager, job_id, "export")
        members = files_at(operation / "bundle")
        total = sum(map(len, members.values()))
        source = path / "operations" / ref["record"]["operation_id"] / "result.sqlite"
        assert total >= source.stat().st_size
        subject = (source, ref["record"])
        assert manager._bundle_snapshot(operation, subject, total, "csv")[0] == members
        with pytest.raises(ValueError):
            manager._bundle_snapshot(operation, subject, total - 1, "csv")
        transport = jobs_module()._transport_zip(members, total)
        assert total < len(transport) <= total + 65536
        exact = physical_tmp / "snapshot"
        exact.write_bytes(transport)
        assert jobs_module()._artifact_bytes(exact, len(transport)) == transport
        with pytest.raises(ValueError):
            jobs_module()._artifact_bytes(exact, len(transport) - 1)
        # The allowance belongs to headers only; an exact-limit aggregate is not rejected.
        assert manager.download(job_id, "bundle")[0] == transport


def test_download_unknown_job_preserves_404_and_resource_failure_is_json(server, monkeypatch):
    for kind in ("record", "report", "bundle"):
        assert request(server, "GET", f"/api/jobs/{'0' * 32}/download/{kind}")[0] == 404

    def failed(*args):
        raise MemoryError("bounded snapshot allocation unavailable")

    monkeypatch.setattr(server.jobs, "_checked_subject", failed)
    status, headers, raw = request(server, "GET", f"/api/jobs/{'0' * 32}/download/record")
    assert status == 400 and "application/json" in headers["Content-Type"]
    assert "allocation unavailable" in json.loads(raw)["error"]


def test_oversized_saved_report_is_http_json_refusal(server, physical_tmp):
    inp, config = admission(physical_tmp)
    job_id = server.jobs.admit(config)["id"]
    server.jobs.upload(job_id, io.BytesIO(inp.read_bytes()), inp.stat().st_size)
    server.jobs.start(job_id, "run")
    wait_job(server.jobs, job_id)
    _, _, operation, _ = generated(server.jobs, job_id, "report")
    (operation / "report.html").write_bytes(b"x" * (int(config["max_bytes"]) + 1))
    status, headers, raw = request(server, "GET", f"/api/jobs/{job_id}/download/report")
    assert status == 400 and "application/json" in headers["Content-Type"]
    assert json.loads(raw)["error"]


def test_oversized_child_report_finishes_failed_not_running(physical_tmp, monkeypatch):
    with saved_job(physical_tmp) as (manager, job_id, _):
        path, _, _, _ = generated(manager, job_id, "report")
        previous = (path / "report.json").read_bytes()
        original = jobs_module()._artifact_bytes

        def oversized(target, limit):
            if target.name == "report.html":
                target.write_bytes(b"x" * (limit + 1))
            return original(target, limit)

        monkeypatch.setattr(jobs_module(), "_artifact_bytes", oversized)
        manager.start(job_id, "report")
        deadline = time.monotonic() + 30
        while manager._active is not None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert manager._active is None
        assert manager._observations[job_id]["state"] == "failed"
        assert (path / "report.json").read_bytes() == previous


@pytest.mark.parametrize("action", ["report", "export"])
def test_artifact_actions_reject_request_paths_and_auth_before_new_operation(
    server, physical_tmp, action
):
    inp, body = admission(physical_tmp)
    job_id = server.jobs.admit(body)["id"]
    server.jobs.upload(job_id, io.BytesIO(inp.read_bytes()), inp.stat().st_size)
    route = f"/api/jobs/{job_id}/{action}"
    for headers in (
        {"Host": "evil.test"},
        {"Origin": "https://evil.test"},
        {"X-SelCal-Token": "wrong"},
    ):
        assert request(server, "POST", route, b"", headers)[0] == 403
    assert request(server, "POST", route, b'{"path":"elsewhere"}')[0] == 400
    for method in ("GET", "PUT", "DELETE", "OPTIONS"):
        assert request(server, method, route, b"")[0] == 404
    for suffix in ("?file=elsewhere", "/extra", "%2f..", "/../record"):
        assert request(server, "POST", route + suffix, b"")[0] == 404
    assert request(server, "POST", route, b"")[0] == 400  # No explicit saved record.
    assert not list((server.jobs.workspace / "jobs" / job_id / "operations").iterdir())


@pytest.mark.parametrize("header", ["Host", "Origin", "X-SelCal-Token", "Content-Length"])
def test_download_rejects_duplicate_headers(server, header):
    values = {
        "Host": server.host,
        "Origin": server.origin,
        "X-SelCal-Token": server.token,
        "Content-Length": "0",
    }
    text = f"GET /api/jobs/{'0' * 32}/download/record HTTP/1.1\r\n"
    text += "".join(f"{k}: {v}\r\n" for k, v in values.items())
    text += f"{header}: {values[header]}\r\nConnection: close\r\n\r\n"
    with socket.create_connection(("127.0.0.1", server.server_port), timeout=5) as connection:
        connection.sendall(text.encode())
        response = connection.recv(2048)
    assert response.startswith(b"HTTP/1.0 400")


def test_export_summary_must_match_shared_verifier_before_publication(physical_tmp, monkeypatch):
    with saved_job(physical_tmp) as (manager, job_id, _):
        path, _, _, _ = generated(manager, job_id, "export")
        previous = (path / "bundle.json").read_bytes()
        original = jobs_module()._terminal

        def changed(raw, command, code):
            terminal = original(raw, command, code)
            terminal["data"]["raw_input_sha256"] = "0" * 64
            return terminal

        monkeypatch.setattr(jobs_module(), "_terminal", changed)
        manager.start(job_id, "export")
        observed = wait_job(manager, job_id)
        assert observed["state"] == "failed"
        assert "terminal differs" in observed["message"]
        assert (path / "bundle.json").read_bytes() == previous


def test_real_chrome_sandbox_style_and_authenticated_downloads(server, physical_tmp):
    node = shutil.which("node")
    if (
        not node
        or not os.environ.get("SELCAL_TEST_PLAYWRIGHT_MODULE")
        or not os.environ.get("SELCAL_TEST_BROWSER_EXECUTABLE")
    ):
        pytest.skip("Optional actual Chrome test requires configured browser/runtime")
    inp, config = admission(physical_tmp)
    job_id = server.jobs.admit(config)["id"]
    server.jobs.upload(job_id, io.BytesIO(inp.read_bytes()), inp.stat().st_size)
    server.jobs.start(job_id, "run")
    wait_job(server.jobs, job_id)
    path, report, report_op, _ = generated(server.jobs, job_id, "report")
    _, _, bundle_op, _ = generated(server.jobs, job_id, "export")
    source = path / "operations" / report["record"]["operation_id"] / "result.sqlite"
    harness = r"""
const assert=require('node:assert/strict'),fs=require('node:fs');
const {chromium}=require(process.env.SELCAL_TEST_PLAYWRIGHT_MODULE);
const [url,id,record,report,bundle,recordName,reportName,bundleName]=process.argv.slice(1);
(async()=>{
 const browser=await chromium.launch({headless:true,
  executablePath:process.env.SELCAL_TEST_BROWSER_EXECUTABLE});
 try{
  const page=await browser.newPage({acceptDownloads:true});
  await page.goto(url);await page.waitForLoadState('networkidle');
  assert(await page.getByRole('button',{name:'Preview HTML report',exact:true}).count());
  await page.locator('#saved-jobs').selectOption(id);
  await page.getByRole('button',{name:'Preview HTML report',exact:true}).click();
  const frame=page.frameLocator('#report-preview');
  await frame.getByRole('heading',{name:'SelCal calibration report',exact:true}).waitFor();
  assert.equal(await page.locator('#report-preview').getAttribute('sandbox'),'');
  const actual=await frame.locator('body').evaluate(el=>({
   color:getComputedStyle(el).color,maxWidth:getComputedStyle(el).maxWidth,
   isolated:(()=>{try{return !parent.document}catch(e){return e.name==='SecurityError'}})()}));
  assert.deepEqual(actual,{color:'rgb(23, 43, 77)',maxWidth:'960px',isolated:true});
  // Waiting on the preview heading scrolls the page to the frame; Playwright's own scroll then
  // missed the button (2026-10-04 diagnosis: no click event reached the page, a harness effect,
  // not a product fault). Bring each button into view before clicking, as a user would.
  const button=label=>{const b=page.getByRole('button',{name:label,exact:true});
   return {click:async()=>{await b.scrollIntoViewIfNeeded();await b.click()}}};
  for(const [label,name,path] of [['Download result record',recordName,record],
   ['Download HTML report',reportName,report],['Download evidence ZIP',bundleName,bundle]]){
   const pending=page.waitForEvent('download');
   await button(label).click();
   const download=await pending;
   assert.equal(download.suggestedFilename(),name);
   assert.deepEqual(fs.readFileSync(await download.path()),fs.readFileSync(path));
   const status=await page.locator('#artifact-status').textContent();
   assert(status.includes(name)&&status.includes(`${fs.statSync(path).size} bytes`),status);
   assert(/workspace folder: jobs\/[0-9a-f]{32}\/operations\/[0-9a-f]{32}\/[a-z.]+\.$/.test(status),
    status);
  }
  // R12 download retest: the evidence ZIP saved empty in Chrome. Start the ZIP download and
  // immediately start the next one; the saved ZIP must still be complete, byte for byte.
  const downloads=[];page.on('download',d=>downloads.push(d));
  await button('Download evidence ZIP').click();
  await page.waitForFunction(()=>!document.getElementById('download-record').disabled);
  await button('Download result record').click();
  await page.waitForFunction(()=>!document.getElementById('download-record').disabled);
  for(let i=0;i<100&&downloads.length<2;i++)await page.waitForTimeout(50);
  assert.equal(downloads.length,2);
  const byName=new Map(downloads.map(d=>[d.suggestedFilename(),d]));
  assert.deepEqual(fs.readFileSync(await byName.get(bundleName).path()),fs.readFileSync(bundle));
  assert.deepEqual(fs.readFileSync(await byName.get(recordName).path()),fs.readFileSync(record));
  console.log(JSON.stringify({styledSandbox:actual,exactDownloads:5}));
 }finally{await browser.close()}
})().catch(error=>{console.error(error);process.exitCode=1});
"""
    completed = subprocess.run(
        [
            node,
            "-e",
            harness,
            server.origin + "/#token=" + server.token,
            job_id,
            str(source),
            str(report_op / "report.html"),
            str(bundle_op / "evidence.zip"),
            f"selcal-{job_id[:8]}-{report['record']['operation_id'][:8]}-result.sqlite",
            f"selcal-{job_id[:8]}-{report_op.name[:8]}-report.html",
            f"selcal-{job_id[:8]}-{bundle_op.name[:8]}-evidence.zip",
        ],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=90,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("action", ["report", "export"])
@pytest.mark.parametrize("stop", ["cancel", "close", "close_gap"])
@pytest.mark.parametrize("mode", ["complete", "replicate_execution"])
def test_stop_after_actual_cli_exit_prevents_artifact_publication(
    physical_tmp, monkeypatch, action, stop, mode
):
    with saved_job(physical_tmp, mode=mode) as (manager, job_id, _):
        path, _, _, _ = generated(manager, job_id, action)
        kind = "report" if action == "report" else "bundle"
        reference = path / (kind + ".json")
        previous = reference.read_bytes()
        old_files = files_at(path / "operations")
        old_operations = set((path / "operations").iterdir())
        drained, release, stop_processed = (threading.Event() for _ in range(3))
        allow_cancel = threading.Event()
        real_join, real_cancel = threading.Thread.join, manager.cancel
        stop_errors = []

        def joined(thread, timeout=None):
            active = manager._active
            if (
                active
                and threading.current_thread().name == "artifact-stop"
                and thread is active[2]
            ):
                stop_processed.set()  # cancel has accepted the request before waiting.
            result = real_join(thread, timeout)
            active = manager._active
            if active and threading.current_thread() is active[2] and not drained.is_set():
                # Pause only scheduling after the real CLI exit; do not fake
                # commands, terminal data, members, references or child status.
                assert active[1].poll() == (0 if mode == "complete" else 7)
                drained.set()
                assert release.wait(10), "scheduler release missing"
            return result

        def deferred_cancel(identifier):
            assert manager._closed  # close released its first lock, not yet in cancel.
            stop_processed.set()
            assert allow_cancel.wait(10), "close-gap release missing"
            return real_cancel(identifier)

        def stopping():
            try:
                manager.cancel(job_id) if stop == "cancel" else manager.close()
            except BaseException as error:
                stop_errors.append(error)

        with monkeypatch.context() as patch:
            patch.setattr(threading.Thread, "join", joined)
            if stop == "close_gap":
                patch.setattr(manager, "cancel", deferred_cancel)
            manager.start(job_id, action)
            assert drained.wait(20), "actual CLI did not exit"
            assert manager.get(job_id)["state"] == "running"
            active = manager._active
            assert active and active[1].poll() == (0 if mode == "complete" else 7)
            stopper = threading.Thread(name="artifact-stop", target=stopping)
            stopper.start()
            try:
                assert stop_processed.wait(5), "stop did not reach its controlled window"
                release.set()
                if stop == "close_gap":
                    # The worker settles while close has not yet called the real cancel.
                    real_join(active[2], 10)
                    assert not active[2].is_alive()
            finally:
                release.set()
                allow_cancel.set()
                real_join(stopper, 10)
            assert not stopper.is_alive() and not stop_errors

        observed = wait_job(manager, job_id)
        assert reference.read_bytes() == previous, "Accepted stop published a new artifact"
        assert observed["state"] == "interrupted", observed
        assert manager._active is None
        (new_operation,) = set((path / "operations").iterdir()) - old_operations
        terminal = json.loads((new_operation / "terminal.json").read_bytes())
        assert terminal["command"] == action
        assert terminal["exit_code"] == (0 if mode == "complete" else 7)
        assert observed["result"] == terminal, "Real child terminal must remain inspectable"
        if mode != "complete":
            assert terminal["data"]["p_value"] is None
            assert terminal["data"]["reject_null"] is None
        output = new_operation / ("report.html" if action == "report" else "bundle")
        assert output.exists(), "Actual child output must be retained"
        assert all(
            (path / "operations" / name).read_bytes() == raw for name, raw in old_files.items()
        )
        # Restart retains the cancellation observation and the earlier good reference.
        manager.close()
        reopened = jobs_module().JobManager(manager.workspace)
        try:
            assert reopened.get(job_id)["state"] == "interrupted"
            assert reopened.download(job_id, kind)[0]
        finally:
            reopened.close()


@pytest.mark.parametrize("action", ["report", "export"])
@pytest.mark.parametrize("mode", ["complete", "replicate_execution"])
def test_cancel_after_artifact_settlement_is_a_noop(physical_tmp, action, mode):
    with saved_job(physical_tmp, mode=mode) as (manager, job_id, _):
        path, _, _, observed = generated(manager, job_id, action)
        before = files_at(path)
        assert manager._active is None
        assert manager.cancel(job_id) == observed
        assert files_at(path) == before
        manager.close()
        assert files_at(path) == before
