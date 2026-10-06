from __future__ import annotations

import json
import os
import shutil
import subprocess
from html.parser import HTMLParser
from importlib.resources import files
from pathlib import Path

import pytest


def test_ui_assets_are_packaged_and_demo_bytes_exact():
    assets = files("selcal").joinpath("web")
    for name in ("index.html", "app.js", "style.css", "series.csv", "pearson.json"):
        assert assets.joinpath(name).is_file(), f"missing UI asset {name}"
    root = Path(__file__).resolve().parents[1]
    for name in ("series.csv", "pearson.json"):
        assert (
            assets.joinpath(name).read_bytes()
            == (root / "examples" / "workflow" / name).read_bytes()
        )


def test_ui_has_no_remote_assets_or_html_injection_and_names_pending_scope():
    assets = files("selcal").joinpath("web")
    assert assets.joinpath("app.js").is_file(), "missing UI script"
    script = assets.joinpath("app.js").read_text(encoding="utf-8")
    html = assets.joinpath("index.html").read_text(encoding="utf-8")
    assert "innerHTML" not in script
    assert "textContent" in script and "sessionStorage" in script
    assert "X-SelCal-Token" in script and "replaceState" in script
    for text in (html, script, assets.joinpath("style.css").read_text(encoding="utf-8")):
        assert "https://" not in text and "http://" not in text
    for label in ("Resume", "replayed and checked", "export", "pending", "Ctrl-C", "NOT_EVALUABLE"):
        assert label in html


def test_verification_button_and_explicit_scope_are_visible():
    assets = files("selcal").joinpath("web")
    html = assets.joinpath("index.html").read_text(encoding="utf-8")
    assert 'id="verify"' in html and "Verify saved result" in html
    assert "input, plan and result consistency" in html
    assert "not current-byte verification" in html
    assert "historical execution" in html


def test_actual_javascript_verification_target_edit_and_scope_guards():
    node = shutil.which("node")
    if node is None:
        pytest.skip("JavaScript runtime unavailable; real browser acceptance is separate")
    root = Path(__file__).resolve().parents[1]
    harness = r"""
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const elements=new Map(), example=EXAMPLE;
function element(id) {
 if(!elements.has(id)) elements.set(id,{value:'',checked:false,disabled:false,options:[],
  files:[],handlers:{},textContent:'',classList:{toggle(){}},
  addEventListener(k,f){this.handlers[k]=f},replaceChildren(){this.options=[]},
  append(...v){this.options.push(...v)}});
 return elements.get(id);
}
let release, posted, refuse=false;
const saved={id:'saved',state:'complete',input_status:'complete',format:'csv',
 config_text:example,max_bytes:'8388608',allow_unattainable:false,record_target:{available:true}};
const checked={...saved,operation:'verify',message:'Saved content check completed',
 result:{command:'verify',data:{status:'complete',p_value:0.2,reject_null:false,
 verification_scope:'input_plan_result_consistency',replay:'NOT_PERFORMED',
 historical_execution_authenticated:false}}};
const response=value=>({ok:true,json:async()=>value});
const context=vm.createContext({URLSearchParams,Blob,Object,JSON,console,
 location:{hash:'#token=local',pathname:'/'},history:{replaceState(){}},
 sessionStorage:{setItem(){},getItem(){return 'local'}},setInterval(){},
 document:{getElementById:element,createElement(){return element(Math.random())}},
 fetch:async(path,options)=>{
  if(path==='/api/example') return response({input_text:'x,y\n0,1\n',config_text:example});
  if(path==='/api/jobs') return response({jobs:[saved]});
  if(path==='/api/jobs/saved') return response(checked);
  if(path==='/api/jobs/saved/verify') {posted=options;
   if(refuse) return {ok:false,json:async()=>({error:'Saved record changed; refused'})};
   return new Promise(resolve=>{release=()=>resolve(response(checked))});}
  throw new Error('Unexpected route '+path);
 }});
context.saved=saved;context.checked=checked;
vm.runInContext(fs.readFileSync(SCRIPT,'utf8'),context);
(async()=>{
 await new Promise(setImmediate);
 vm.runInContext('selectedId=viewedId="saved"; showJob(saved)',context);
 assert.equal(element('verify').disabled,false);
 for(const expression of ['busy=true','latestJob={...saved,state:"running"}',
  'latestJob={...saved,record_target:{available:false}}','selectedId=null']) {
  vm.runInContext('busy=false;selectedId="saved";latestJob=saved;'+expression+';buttons()',context);
  assert.equal(element('verify').disabled,true,expression);
 }
 vm.runInContext('selectedId="saved";showJob(saved);invalidate()',context);
 assert.equal(element('verify').disabled,true,'Edits must disable verify');
 vm.runInContext('selectedId=viewedId="saved";showJob(checked)',context);
 const text=element('result-summary').options.map(x=>x.textContent).join(' ');
 assert(text.includes('input_plan_result_consistency'));
 assert(text.includes('NOT_PERFORMED'));assert(text.includes('false'));
 assert(element('verification-status').textContent.includes('not current-byte verification'));
 const pending=element('verify').handlers.click();await new Promise(setImmediate);
 assert.equal(posted.method,'POST');assert.equal(posted.body,'');
 assert.equal(element('verify').disabled,true);
 assert(element('verification-status').textContent.includes('No current'));
 vm.runInContext('invalidate()',context);release();await pending;
 assert.equal(element('verify').disabled,true);
 assert(element('verification-status').textContent.includes('No current'),
  'An edited selection must not inherit a late successful verification');
 vm.runInContext('selectedId=viewedId="saved";showJob(checked)',context);
 refuse=true;await element('verify').handlers.click();
 assert(element('notice').textContent.includes('Saved record changed'));
 assert(element('verification-status').textContent.includes('No current'));
 assert.equal(element('result-summary').options.length,0,
  'A refused fresh request must not leave an old successful check on display');
 assert(element('terminal').textContent.includes('No result'));
})().catch(error=>{console.error(error);process.exitCode=1});
"""
    harness = harness.replace(
        "EXAMPLE", json.dumps((root / "examples/workflow/pearson.json").read_text(encoding="utf-8"))
    )
    harness = harness.replace("SCRIPT", json.dumps(str(root / "src/selcal/web/app.js")))
    result = subprocess.run([node, "-e", harness], capture_output=True, text=True,
        errors="replace", timeout=60)  # Node starts slowly on hosted Windows runners
    assert result.returncode == 0, result.stderr


def test_actual_javascript_preserves_lexemes_and_invalidates_edits_during_admission():
    node = shutil.which("node")
    if node is None:
        pytest.skip("JavaScript runtime unavailable; real browser acceptance is separate")
    root = Path(__file__).resolve().parents[1]
    script = root / "src" / "selcal" / "web" / "app.js"
    example = (root / "examples" / "workflow" / "pearson.json").read_text(encoding="utf-8")
    # This minimal DOM exercises actual client state/serialization, never scientific success.
    harness = r"""
const fs = require("fs"), vm = require("vm"), assert = require("assert");
const elements = new Map();
function element(id) {
 if (!elements.has(id)) elements.set(id, {value:"", checked:false, disabled:false,
  options:[], files:[], handlers:{}, classList:{toggle(){}},
  addEventListener(k,f){this.handlers[k]=f}, replaceChildren(){this.options=[]},
  append(...v){this.options.push(...v)}});
 return elements.get(id);
}
 let admissionResolve, posted, uploaded, resumedPath, resumedBody;
const example = EXAMPLE;
const context = vm.createContext({URLSearchParams, Blob, Object, JSON, console,
 location:{hash:"#token=local", pathname:"/"}, history:{replaceState(){}},
 sessionStorage:{setItem(){},getItem(){return "local"}}, setInterval(){},
 document:{getElementById:element,createElement(){return element(Math.random())}},
 fetch:async(path,options)=>{
  if(path==="/api/jobs/unavailable")
   return {ok:false,json:async()=>({error:"Saved job unavailable"})};
  if(path==="/api/jobs/saved")
   return {ok:true,json:async()=>({id:"saved",state:"interrupted",input_status:"complete",
    format:"npz",input_sha256:"a".repeat(64),config_text:context.exact,
    max_bytes:"8388608",allow_unattainable:true})};
  if(path.endsWith("/resume")) {
   resumedPath=path; resumedBody=options.body;
   return {ok:true,json:async()=>({id:"saved",state:"running",input_status:"complete"})};
  }
  if(path==="/api/example")
   return {ok:true,json:async()=>({input_text:"x,y\n0,1\n",config_text:example})};
  if(path==="/api/jobs" && options.method==="POST") {
    posted = JSON.parse(options.body);
    return new Promise(resolve=>{admissionResolve=()=>resolve({ok:true,json:async()=>({
     id:"job",state:"created",input_status:"missing",config_text:posted.config_text})})});
  }
  if(path.endsWith("/input")){uploaded=options.body;return {ok:true,json:async()=>({
   id:"job",state:"ready",input_status:"complete"})};}
  return {ok:true,json:async()=>({jobs:[]})};
 }});
vm.runInContext(fs.readFileSync(SCRIPT,"utf8"), context);
async function test(){
 await new Promise(setImmediate);
 const exact = example.replace('"root_seed": 17','"root_seed": 9007199254740993')
  .replace('"statistic_params": {}','"statistic_params": {"large":9007199254740995}');
 context.exact=exact;
 vm.runInContext('syncForm(exact); $("config-text").value=exact;'+
  ' $("budget").value="1048576";',context);
 const serialized = vm.runInContext('formText()',context);
 assert(serialized.includes('"root_seed": 9007199254740993'));
 assert(serialized.includes('"large":9007199254740995'));
 vm.runInContext('$("alpha").value="1"',context);
 assert(vm.runInContext('formText()',context).includes('"alpha": 1.0'));
 vm.runInContext('selectedId="existing";'+
  ' latestJob={state:"ready",input_status:"complete"}; buttons()',context);
 assert.equal(element("run").disabled,false);
 assert.equal(element("resume").disabled,false);
 element("seed").handlers.input();
 assert.equal(element("run").disabled,true);
 assert.equal(element("resume").disabled,true);
 vm.runInContext('$("config-text").value=exact; inputBytes=new Blob(["original-input"]);',context);
 const creating = element("create").handlers.click();
 await new Promise(setImmediate);
 assert.equal(posted.config_text,exact);
 vm.runInContext('inputBytes=new Blob(["edited-input"]); invalidate();',context);
 admissionResolve(); await creating;
 assert.equal(await uploaded.text(),"original-input", "Upload must use the admitted snapshot");
 assert.equal(vm.runInContext('selectedId',context),null,
  "Late admission must not reactivate an edited plan");
 assert.equal(element("run").disabled,true);
 assert.equal(element("resume").disabled,true);
 vm.runInContext('selectedId="previous"; viewedId="previous";'+
  ' latestJob={state:"ready",input_status:"complete"};'+
  ' $("saved-jobs").value="unavailable";',context);
 await element("saved-jobs").handlers.change();
 assert.equal(vm.runInContext('selectedId',context),null,
  "Failed selection must not target an unseen plan");
 assert.equal(element("run").disabled,true);
 assert.equal(element("resume").disabled,true);
 vm.runInContext('$("saved-jobs").value="saved";',context);
 await element("saved-jobs").handlers.change();
 assert.equal(vm.runInContext('selectedId',context),"saved");
 assert.equal(element("config-text").value,exact);
 assert.equal(element("seed").value,"9007199254740993");
 assert.equal(element("budget").value,"8388608");
 assert.equal(element("override").checked,true);
 assert.equal(element("resume").disabled,false);
 await element("resume").handlers.click();
 assert.equal(resumedPath,"/api/jobs/saved/resume");
 assert.equal(resumedBody,"");
 assert.equal(element("resume").disabled,true);
 assert.equal(element("cancel").disabled,false);
}
test().catch(error=>{console.error(error);process.exitCode=1});
"""
    harness = harness.replace("EXAMPLE", json.dumps(example)).replace(
        "SCRIPT", json.dumps(str(script))
    )
    result = subprocess.run([node, "-e", harness], capture_output=True, text=True,
        errors="replace", timeout=60)  # Node starts slowly on hosted Windows runners
    assert result.returncode == 0, result.stderr


def test_form_lists_every_existing_registered_statistic_and_null():
    from selcal.resolution_v2 import _NULL_REGISTRY_V2, _STATISTIC_REGISTRY_V2

    class Options(HTMLParser):
        def __init__(self):
            super().__init__()
            self.current = None
            self.options = {}
            self.in_option = False

        def handle_starttag(self, tag, attrs):
            if tag == "select":
                self.current = dict(attrs).get("id")
                self.options[self.current] = []
            if tag == "option":
                self.in_option = True

        def handle_endtag(self, tag):
            if tag == "select":
                self.current = None
            if tag == "option":
                self.in_option = False

        def handle_data(self, data):
            if self.current and self.in_option:
                self.options[self.current].append(data)

    parser = Options()
    parser.feed(files("selcal").joinpath("web", "index.html").read_text(encoding="utf-8"))
    assert set(parser.options["statistic"]) == set(_STATISTIC_REGISTRY_V2.names)
    assert set(parser.options["null-model"]) == set(_NULL_REGISTRY_V2.names)


@pytest.mark.parametrize(
    "scenario",
    [
        "overlap",
        "resume_success",
        "resume_error",
        "verify_success",
        "verify_error",
        "view_success",
        "view_error",
        "edit_error",
        "late_list",
    ],
)
def test_actual_javascript_rejects_stale_poll_responses(scenario):
    node = shutil.which("node")
    if node is None:
        pytest.skip("JavaScript runtime unavailable; real browser acceptance is separate")
    root = Path(__file__).resolve().parents[1]
    harness = r"""
const fs = require("node:fs"), vm = require("node:vm"), assert = require("node:assert/strict");
const scenario = SCENARIO, example = EXAMPLE, elements = new Map();
function element(id) {
 if (!elements.has(id)) elements.set(id, {value:"", checked:false, disabled:false,
  options:[], files:[], handlers:{}, classList:{toggle(){}},
  addEventListener(k,f){this.handlers[k]=f}, replaceChildren(){this.options=[]},
  append(...values){this.options.push(...values)}});
 return elements.get(id);
}
const job = (id,state,message) => ({id,state,message,input_status:"complete",format:"csv",
 config_text:example,max_bytes:"8388608",allow_unattainable:false,created_at:"saved",
 record_target:{available:true}});
const response = value => ({ok:true,json:async()=>value});
const tick = () => new Promise(setImmediate);
let poll, releaseOld, rejectOld, holdList=false, statusRequests=0, currentState="complete";
const context = vm.createContext({URLSearchParams,Blob,Object,JSON,console,
 location:{hash:"#token=local",pathname:"/"},history:{replaceState(){}},
 sessionStorage:{setItem(){},getItem(){return "local"}},setInterval(fn){poll=fn},
 document:{getElementById:element,createElement(){return element(Math.random())}},
 fetch:async(path,options)=>{
  if(path==="/api/example") return response({input_text:"x,y\n0,1\n",config_text:example});
  if(path==="/api/jobs") {
   if(holdList) {holdList=false;return new Promise(resolve=>{
    releaseOld=()=>resolve(response({jobs:[job("saved","complete","Old list")]}));
   });}
   return response({jobs:[job("saved",currentState,"Current list")]});
  }
  if(path==="/api/jobs/saved/resume" || path==="/api/jobs/saved/verify") {
   assert.equal(options.method,"POST");assert.equal(options.body,"");
   currentState="running";return response(job("saved","running","Current resume"));
  }
  if(path==="/api/jobs/other") return response(job("other","complete","Other view"));
  if(path==="/api/jobs/saved") {
   statusRequests++;
   if(statusRequests===1) {
    if(scenario==="late_list") {
     holdList=true;return response(job("saved","complete","Old complete"));
    }
    return new Promise((resolve,reject)=>{
     releaseOld=()=>resolve(response(job("saved","complete","Old complete")));
     rejectOld=()=>reject(new Error("Old status failure"));
    });
   }
   const message=currentState==="running"?"Current poll":"Saved complete";
   return response(job("saved",currentState,message));
  }
  throw new Error("Unexpected route "+path);
 }});
vm.runInContext(fs.readFileSync(SCRIPT,"utf8"),context);
(async()=>{
 await tick();
 vm.runInContext('selectedId=viewedId="saved"; showJob({id:"saved",state:"running",'+
  'input_status:"complete",message:"Old run"})',context);
 const oldPoll=poll();await tick();
 if(scenario==="overlap") {
  const overlap=poll();await tick();
  assert.equal(statusRequests,1,"Only one status poll may be in flight");
  releaseOld();await oldPoll;await overlap;
  return;
 }
 if(scenario.startsWith("resume") || scenario.startsWith("verify") || scenario==="late_list") {
  if(scenario!=="late_list") {
   // Reselecting is allowed while a status poll is pending. The current saved
   // response completes the prior run and enables a real Resume click.
   element("saved-jobs").value="saved";await element("saved-jobs").handlers.change();
  }
  const action=scenario.startsWith("verify")?"verify":"resume";
  assert.equal(element(action).disabled,false);
  await element(action).handlers.click();
  assert.equal(element("cancel").disabled,false);
 } else if(scenario.startsWith("view")) {
  element("saved-jobs").value="other";await element("saved-jobs").handlers.change();
 } else {
  element("seed").value="18";element("seed").handlers.input();
 }
 const expectedMessage=element("notice").textContent;
 if(scenario.endsWith("error")) rejectOld();else releaseOld();
 await oldPoll;
 assert.equal(element("notice").textContent,expectedMessage,
  "Stale polling must not replace the current message");
 if(scenario.startsWith("resume") || scenario.startsWith("verify") || scenario==="late_list") {
  assert.equal(vm.runInContext("latestJob.state",context),"running");
  assert.equal(element("cancel").disabled,false,"Current running child must remain cancellable");
  assert.equal(element("resume").disabled,true);
  if(scenario==="late_list")
   assert(element("saved-jobs").options[1].textContent.includes("running"));
  const before=statusRequests;await poll();
  assert.equal(statusRequests,before+1,"Polling must continue after the stale request settles");
  assert.equal(element("notice").textContent,"Current poll");
 } else if(scenario.startsWith("view")) {
  assert.equal(vm.runInContext("latestJob.id",context),"other");
 } else {
  assert.equal(vm.runInContext("selectedId",context),null);
  assert.equal(element("resume").disabled,true);
 }
})().catch(error=>{console.error(error);process.exitCode=1});
"""
    for key, value in {
        "SCENARIO": scenario,
        "EXAMPLE": (root / "examples" / "workflow" / "pearson.json").read_text(encoding="utf-8"),
        "SCRIPT": str(root / "src" / "selcal" / "web" / "app.js"),
    }.items():
        harness = harness.replace(key, json.dumps(value))
    result = subprocess.run([node, "-e", harness], capture_output=True, text=True,
        errors="replace", timeout=60)  # Node starts slowly on hosted Windows runners
    assert result.returncode == 0, result.stderr


def test_real_browser_wraps_saved_input_hash_in_narrow_layout():
    node = shutil.which("node")
    module = os.environ.get("SELCAL_TEST_PLAYWRIGHT_MODULE")
    executable = os.environ.get("SELCAL_TEST_BROWSER_EXECUTABLE")
    if node is None or not module or not executable:
        pytest.skip(
            "Optional real-browser layout test needs configured Node Playwright and browser"
        )
    assets = Path(__file__).resolve().parents[1] / "src" / "selcal" / "web"
    script = r"""
const assert = require("node:assert/strict"), fs = require("node:fs");
const {chromium} = require(process.env.SELCAL_TEST_PLAYWRIGHT_MODULE);
const [htmlPath, cssPath] = process.argv.slice(1);
(async () => {
 const browser = await chromium.launch({headless:true,
  executablePath:process.env.SELCAL_TEST_BROWSER_EXECUTABLE});
 try {
  const page = await browser.newPage({viewport:{width:390,height:844}});
  const html = fs.readFileSync(htmlPath,"utf8")
   .replace(/<script[^>]*>.*?<\/script>/gs,"")
   .replace(/<link[^>]*rel="stylesheet"[^>]*>/g,"");
  await page.setContent(html);
  await page.addStyleTag({content:fs.readFileSync(cssPath,"utf8")});
  await page.locator("#input-description").evaluate(element => {
   element.textContent = "Saved immutable CSV input · " + "a".repeat(64) +
    ". Choose new input to create another job.";
  });
  const measurements = await page.evaluate(() => {
   const description = document.getElementById("input-description");
   return {viewport:innerWidth, document:document.documentElement.scrollWidth,
    description:description.scrollWidth, available:description.clientWidth,
    wrapping:getComputedStyle(description).overflowWrap};
  });
  console.log(JSON.stringify(measurements));
  assert(measurements.document <= measurements.viewport,
   "Saved input hash must not widen the document");
  assert(measurements.description <= measurements.available,
   "Saved input description must remain within its container");
  assert.equal(measurements.wrapping,"anywhere");
 } finally { await browser.close(); }
})().catch(error => {console.error(error);process.exitCode=1});
"""
    result = subprocess.run(
        [node, "-e", script, str(assets / "index.html"), str(assets / "style.css")],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
