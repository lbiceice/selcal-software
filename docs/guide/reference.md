# SelCal user reference

The complete behaviour of the command line, the local interface, exports, input limits, the
bundled examples and the Python API. Start with the [README](../../README.md) and
[Interpreting SelCal results](interpreting_results.md); this page is the detailed reference.

## Input encoding, refusals and replay

Input text files (CSV data and JSON configuration) are read as UTF-8; a leading UTF-8 byte-order
mark, as written by spreadsheet programs and Windows PowerShell 5.1, is accepted. The raw input
digest in a record covers the file bytes exactly as read, including such a mark. When a request is
refused, the JSON result names the error in `error` and explains it in `detail` (for example which
columns the file has and which the configuration names).

`selcal verify --replay` is bit-exact and needs the recording code, Python, NumPy and platform
(operating system, architecture, C library, BLAS); on another platform it reports
`environment_mismatch`. `selcal verify --replay-decision` is supported with the recording code and
the same NumPy version: seeds, surrogate states, selections, exceedance count, p-value and decision
must match exactly, and statistic
values within 64 units in the last place of max(|a|, |b|, 1). A macOS record replayed this way on
Linux matched (largest difference: 1 unit).
Cross-NumPy record verification, reporting and replay are not currently supported: the strict
result verifier binds each saved statistic to its NumPy backend identity. A genuine record made
with a different NumPy version currently fails with `invalid_record_content`, even if its
decisions would agree. Use the NumPy version recorded in the original environment; do not edit
the record or disable integrity checks to force a replay.

## Run a file-to-report workflow

After installation, run these commands from the source snapshot. Every command
uses the same application service as the Python API:

```console
selcal validate examples/workflow/series.csv examples/workflow/pearson.json
selcal run examples/workflow/series.csv examples/workflow/pearson.json run.sqlite --max-bytes 1048576
selcal verify run.sqlite --max-bytes 1048576
selcal verify run.sqlite --max-bytes 1048576 --replay
selcal verify run.sqlite --max-bytes 1048576 --replay-decision
selcal report run.sqlite report.html --max-bytes 1048576
selcal doctor
```

`python -m selcal` is equivalent to `selcal`. The input is a fixed synthetic
100-row example (y depends on x at lag 2), not research validation; its result has
199 retained replicates, selects lag 2 and gives p=0.015. Replace the input and
configuration to analyse your own supported data.

`validate` also reports `attainability`, computed from the plan and the sample count
with the same float comparison the calibrator uses (`p = (1 + E)/(B + 1)`, reject iff
`p <= alpha`). No run can reach p below `1/(B+1)`. For the full cyclic group
(`min_shift = 1`) with
lagged Pearson, shifts mapping a searched lag onto a lag attaining the observed maximum
reproduce that candidate's score; reselecting across all lags can only reach or exceed it.
Thus at least a share (number of searched lags)/(number of null states) of the states is
always an exceedance: 0.03 for 3 lags and 100 samples. With exact enumeration
(`circular_shift_exact_v1`) that share is a hard floor on p. With the sampling null
(`circular_shift_v2`) a single run can fall below it when the draws happen to miss those
states (the example above gives p = 0.015), so there it caps power instead; the reported
`monte_carlo_power_cap` is an upper bound on the rejection rate B draws allow even for a very strong
signal. `run` refuses plans whose share exceeds alpha and exact-enumeration plans whose
`replicates` is not `n - 1`, with exit 2 and error `unattainable_plan`, unless
`--allow-unattainable-plan` is given.

`min_shift` > 1 is refused for every statistic and any number of lags. For 1 < `min_shift` < n/2
the remaining shifts are not a group and the test then has no level guarantee (`min_shift` = n/2
leaves {0, n/2}, a group, but it is unsupported too): in the Windows core audit
(2026-10-03) one lag with `min_shift = 2` rejected 2 of 24 phases of an exactly cyclic null
at alpha = 0.05 (about 6.1% with B = 999). This is a validity refusal, reported as
`REFUSE_NON_GROUP_NULL` and in `preflight.plan.null_validity`; `run`, `resume` and the
interface refuse it with exit 2 and error `invalid_null_for_inference`, and
`--allow-unattainable-plan` does not lift it. Use `min_shift = 1`.

The Monte Carlo grid is not an error bar: with B = 199 the possible p-values are spaced by
1/200 = 0.005, which is the resolution, not a ±0.005 uncertainty. The `low`/`high` bounds of a
NOT_EVALUABLE result are the p-values all failed surrogates could reach, not a confidence
interval. The scientific plan identity includes B, alpha and the tie tolerance, so with a fixed
`root_seed` changing any of them can also change the Monte Carlo stream; compare plans by their
recorded identity, not by seed alone.

`attainability` is computed only for lagged Pearson with the circular nulls; other statistics
report `NOT_ASSESSED`, and `null_validity` reports `NOT_ASSESSED` for block shuffle. Treat
`NOT_ASSESSED` as "not checked", never as a pass.

Records made before the Pearson numeric-method change still pass `verify`; their `--replay`
and `--replay-decision` are refused with `environment_mismatch` because the code differs.

Lagged Pearson scales each window by a power of two (exact), centres it in two passes and forms
every sum with `math.fsum`, the correctly rounded exact sum. A large common offset therefore does
not change the correlation, and the result no longer depends on BLAS summation order: the bytes
matched on Apple Accelerate and on OpenBLAS in Linux arm64 and x86_64. This is a tested result,
not a guarantee for every platform or build, so cross-platform checks continue. This numeric method is recorded as
the result's `preprocessing_identity`. The earlier one-pass method lost exactly representable
differences under offsets near 1e16, and BLAS sums differed in the last place between platforms
(Windows audits, 2026-10-03). Correctly rounded sums cost time, and the cost grows with series
length. In eight complete calibrations measured on one Apple arm64 laptop, each at or below the
in-memory execution budget, runs were 1.07-2.77 times slower than before (the largest ratio at
N = 26,315 with B = 19 and four lags: 0.27 s to 0.74 s), peak memory changed by at most 1 MiB and
p-values and counts were identical; a direct kernel call was 9-13 times slower at N = 10,000-100,000
on Windows. The budget limits the combination of N, B and the number of lags, not N alone, so these
ratios describe the measured configurations, not a bound for every admissible plan.

The public Python entry `selcal.calibrate_selected_family` applies the same admission rule as the
command line. The rule currently refuses `circular_shift_v2` with `min_shift` > 1, raising
`selcal.InvalidNullForInferenceError` (`code == "invalid_null_for_inference"`) before any
computation. It does not assess the scientific validity of other nulls; their assumptions remain
the analyst's responsibility. `selcal.calibration_v2` is the internal kernel used for record replay; it is not a
supported entry for new analyses.

`validate` reports a three-level `preflight` so that different questions are not merged:
`input` (the file loaded and passed the input limits; loader errors are reported earlier with
their own codes), `plan` (`EXECUTABLE` or `NOT_EXECUTABLE`, with every reason listed:
`resource_budget_exceeded` when the executor's in-memory caps would refuse the run,
`REFUSE_NON_GROUP_NULL` from `null_validity`, or a `REFUSE_*` attainability status; `warnings` flag plans that will end NOT_EVALUABLE because the
null has no states), and `scientific_assumptions` (the assumptions this plan relies on, e.g. lags
declared before seeing the data, circular-shift exchangeability; marked
`DECLARED_NOT_VERIFIED` because SelCal cannot check them from the data). When the plan is not
executable, `validate` exits 2 with outcome `PLAN_NOT_EXECUTABLE` and the first reason as its
error, and `run` refuses before any calibration (`resource_budget_exceeded` or
`unattainable_plan`). The resource check runs first, so an oversized replicate count is refused
without plan-time arithmetic that grows with it.
The record captures complete raw input, normalized request, full result and
recorded software identity. Reports and records must use new output paths.

`verify` checks byte, actual input, plan and result consistency; it does not
recompute by default. `--replay` explicitly recomputes the complete calibration
and compares exact result bytes under matching recorded versions/source files.
Neither check authenticates past execution. The saved record can be read without
the original input paths; terminal-record reading is separate from checkpoint recovery. Interrupted
writes can leave a rejected partial file; no cross-platform power-loss durability
guarantee is made.

To retain a new run for local recovery, choose a new checkpoint directory:

```console
selcal run examples/workflow/series.csv examples/workflow/pearson.json run.sqlite --max-bytes 8388608 --checkpoint run-checkpoint
selcal resume run-checkpoint recovered.sqlite --max-bytes 8388608
```

`resume` uses the retained input and original plan, seed, replicate IDs, override and
execution UUID. It recomputes the complete calibration and checks every retained outcome
byte for byte before appending missing outcomes. This mode is `replay_before_continue`;
it does not save computation on the retained prefix. It requires the same source files,
Python, NumPy, platform identity and original `--max-bytes`; it accepts no replacement
input, plan or override. Even a finalized checkpoint is fully replayed before another export.

One writer holds the local checkpoint lock through replay, finalization and export.
Progress counts go to stderr; stdout contains one final JSON document with a nested
`checkpoint` summary. A busy writer, changed identity or replay mismatch fails with exit 4
and a specific error code. An observed interrupt uses exit 130; a killed process may
produce no final JSON. Initialization may be incomplete, so merely specifying
`--checkpoint` does not guarantee recovery. After checking the retained directory, use
`resume` with a new output filename. Existing output and checkpoint files are preserved.
If export fails after finalization, the finalized checkpoint can be replayed to another
filename. Inside the checkpoint directory, use an ordinary new filename; the internal
database, lock and SQLite sidecar names are reserved, including case and directory aliases.
Checkpoint and terminal-record caps apply separately; fitting one does not
guarantee fitting the other. Checkpoints require a supported local filesystem and do not
provide distributed locking, environment migration or a general power-loss guarantee.

The JSON config schema is `selcal.workflow-config.v1` with exactly `input` and
`plan` in addition to `schema`. All ten plan parameters are explicit in the example.
For NPZ use `"input": {"format": "npz"}`. CSV requires two distinct named columns.
Both the incoming config and its normalized encoding plus LF must fit 65,536
bytes, with nesting depth at most 8. Duplicate keys and nonfinite values reject.
`--max-bytes` bounds stored payload/database and report bytes, not process memory;
the example's 1 MiB cap is a caller choice. Existing scientific input/execution
limits remain unchanged.

Normal commands emit one JSON line. Exit 0 means the requested operation completed
(or validation/inspection passed); **exit 7 means scientific NOT_EVALUABLE** with
null p/decision, not an I/O failure or evidence of no effect. Exit 2 is invalid
request, exit 4 operation/integrity failure, and exit 130 an observed interruption.
Help/argument syntax diagnostics use the standard command-line parser. Runtime
record reading requires SQLite serialize/deserialize and no-follow file opening;
unsupported capabilities fail explicitly. The basic local UI loop, reports,
export and downloads are implemented. F4C UI integration and F5A artifact-first CI
have local acceptance; container acceptance and final release remain separate gates.

## Basic local interface

```console
selcal ui --workspace selcal-work
```

This starts an offline helper on `127.0.0.1` at an operating-system-selected port
and opens the browser. Use `--no-browser` to print the launch URL, or `--port 8765`
to choose a port. Keep the terminal open; press Ctrl-C there to stop the helper
and its own active calculation. Closing a browser tab does not stop the helper.
The session token is in the launch URL fragment, retained only for this browser
session and sent in a request header. Reopen the printed URL after restarting.
There is no configurable public host or remote service.

Choose a new directory under an existing real parent, an empty real directory,
or a workspace previously created by this UI. Linked and unknown nonempty
directories are refused. Load the included example or choose CSV/NPZ input and
an editable plan. Click **Create job and save input**, **Validate**, then **Run**.
The example has 100 paired observations; the result shows the selected lag,
p-value, decision, scientific status and input/plan hashes. The advanced editor
submits original JSON text, preserving integer seeds and parameter spellings.
The UI starts with a caller-selected budget of 8 MiB (8,388,608 bytes), which
allows the included example's checkpoint and result record to finish. This
budget applies separately to each checkpoint and terminal record; it is not a
process-memory bound or a universal input-size limit. Larger work may need a
different explicit budget within the existing workflow's resource limits.

Each job has immutable configuration, budget, override and raw input. Any edit
requires a new job; saved jobs remain available for inspection. Jobs contain
`request.json`, `metadata.json`, fixed `input.csv`/`input.npz`, an observation and
fresh operation directories. Runs also retain a fixed checkpoint directory.
Only one child operation is active in a helper. Cancellation retains files and
reports interruption, never scientific NOT_EVALUABLE. Initialization may have
stopped before a recoverable checkpoint existed. An existing checkpoint cannot
be overwritten by Run. After restarting, reopen the printed launch URL, select
the saved job, then click **Resume**. Its original input, configuration, override
and byte budget are restored. Retained replicates are replayed and checked before
continuing, using the same execution identity. Every attempt uses a new operation
directory and result destination. Resume requires the original code/environment
and a valid matching checkpoint; a refusal retains the saved files. Editing the
form disables Resume until you reselect the saved job or create a new job.

After a successful Run or Resume, click **Verify saved result** to check the
explicitly referenced record's input, plan and result consistency. The helper
passes its original byte budget to the existing CLI and performs no replay.
Historical execution is not authenticated. Scientific NOT_EVALUABLE remains
NOT_EVALUABLE with null p/decision. Checking again creates an operation observation,
not a new calculation or result file; previous files remain unchanged.

The job-local `record.json` identifies one owned result by operation, byte count
and SHA-256. A missing, changed, linked or mismatched target is refused. The helper
does not guess the latest result. Older jobs without this reference remain readable;
use `selcal verify PATH_TO_RECORD --max-bytes ORIGINAL_BUDGET` for their old records.
Editing the input or plan disables verification until the saved job is reselected.
Saved checks are observations of an earlier operation, not current-byte verification.

Use **Generate HTML report** or **Export evidence bundle** for the selected saved
record. Each action invokes the existing CLI with the original budget, without
replay, and retains a fresh operation directory. Report and bundle references
bind the complete `record.json` descriptor; a newer Run/Resume record makes old
artifacts ineligible without deleting them. Failed or cancelled actions retain
earlier references and partial files but never publish partial outputs.

**Preview HTML report** opens a checked snapshot in a script-disabled sandbox.
The three download controls use authenticated fixed routes for the record,
report or evidence ZIP; no token or user-selected path is placed in the URL.
Every download rechecks input/configuration, source record and artifact bytes.
The ZIP contains exactly the eleven checked F3 members, including the four exact
source members. The original cap bounds source, report and aggregate bundle bytes;
only ZIP transport headers receive an additional 65,536-byte allowance. The cap
is not a process-memory bound. Downloads are named
`selcal-<job>-<operation>-report.html`, `…-result.sqlite` and `…-evidence.zip` (the first eight
characters of each identifier), so earlier files in the same folder are not mistaken for them.
The server re-checks the saved artifact and states its size and SHA-256 in the `X-SelCal-Bytes`
and `X-SelCal-SHA256` headers; the page checks the received bytes against both before handing a
file to the browser, and refuses a missing header or an empty, short or altered body with an error.
Where the browser offers no Web Crypto, only the size is checked and the status says so. The page
keeps at most the six most recent download links; each is released after two minutes, when a
seventh replaces it, or when the page is closed, and the next action no longer withdraws them
(added after a Windows Chrome retest left a 0-byte evidence ZIP in the default folder while a
download manager saved a complete copy elsewhere, 2026-10-04; the cause is not established). Use the
browser with its existing download settings; no download manager is needed. The page cannot see
whether or where a file was saved; the status names the file, its size and full SHA-256 so it can be
found in the browser's downloads list or, if a download manager such as IDM is installed and takes
over, in its completed list and save folder (ZIP files may go to `Downloads\Compressed`). The status
also gives the checked original's path inside the workspace (`jobs/<job>/operations/<operation>/…`,
from the `X-SelCal-Workspace-Path` header), which can be copied directly when no download appears.
The evidence ZIP holds `summary.csv`, `candidates.csv` and `replicates.csv` for spreadsheets; the
SQLite record is for `selcal verify`. Editing or changing actions clears the old preview.
Older unreferenced artifacts remain available through the command line, not
automatic UI adoption. Saved availability and observations never authorize a download.

Status: `PARTIAL_UI_BASIC_LOOP` (including resume, content checks, reports, evidence
export and downloads). F4C integration passed installed-package and visible-browser
acceptance locally on macOS on 2026-09-30. Broader usability, container acceptance,
renewed native Windows validation and final release remain separate gates.
Saved UI observations are not calculation authority.
Neither validation nor a completed calculation establishes scientific validity;
non-rejection is not absence of association, and NOT_EVALUABLE is not p = 1.

## Export and verify portable evidence

Export an existing terminal record to a new directory, then verify it without
the original SQLite file:

```console
selcal export run.sqlite evidence --max-bytes 8388608
selcal verify-export evidence --max-bytes 8388608
```

The directory contains exactly eleven files: the original `input.csv` or
`input.npz`, exact `request.json`, `result.json` and `metadata.json`, canonical
`plan.json` and recorded `software.json`, `summary.csv`, `candidates.csv`,
`replicates.csv`, `report.html` and `manifest.json`. Every observed/replicate
candidate, including failures and unselected rows, is retained. The original
result JSON is authoritative and preserves exact binary64 values and complete
transformation tokens. CSV uses UTF-8 and LF, round-trippable numbers, lowercase
booleans, empty cells for missing values, and compact JSON for lists/objects.
Formula-leading text receives an apostrophe for spreadsheet safety; numeric
negative values keep their sign. Raw recorded JSON is unchanged.

`--max-bytes` caps both the source record and the aggregate exported bytes,
including the manifest. The example uses a caller-selected 8 MiB cap (8,388,608
bytes). CSV/HTML expansion may need a larger limit than the
record alone. This is a serialized-byte limit, not a peak-memory guarantee.
The parent must be an existing real directory and the destination must be new.
No file or directory is replaced. A write or close failure preserves the output;
inspect it with `verify-export` or use a fresh destination. A final close failure
can leave content that verifies, but the original operation still failed and
verification does not establish durable storage.

Both commands check input/plan/result consistency and every derived projection;
they perform no calibration, attainability calculation or replay. Rehashing an
altered CSV/HTML file does not make its projection valid. Hashes and consistency
checks do not authenticate historical execution or establish scientific validity.
`NOT_EVALUABLE` keeps **exit 7** even when the bundle was successfully written or
verified; the diagnostic makes that distinction explicit. Corrupt or oversized
bundles and I/O failures use exit 4; invalid command arguments use exit 2.
Hostile same-user mutation racing export/verification is outside this contract.

## Load bounded CSV and NPZ inputs

The candidate loaders use a fixed input resource envelope with no user override. `load_csv`
accepts a strict UTF-8 two-field CSV: one nonempty unique two-name header, exactly two numeric
fields per later physical record, and no blank, repeated-header, missing, extra, nonfinite, or
float64-underflowing data cells. Quoted fields are accepted within one physical record, but
quoted fields cannot contain CR or LF. `LF`, `CRLF`, and `CR` terminators normalize to the same
record boundary.

`load_npz` accepts one conventional single-disk ZIP with exactly two non-directory numeric NPY
members named `source`/`source.npy` and `target`/`target.npy`. Member compression is limited to
stored or deflate. Encryption, data descriptors, archive-level ZIP64, path-like or duplicate
names, overlapping ranges, unsupported NPY versions, multidimensional arrays, and non-real or
object dtypes are rejected. The supported NPY versions are 1.0 and 2.0; arrays must be
one-dimensional signed integer, unsigned integer, or real float with item size at most 8 bytes.

The exact envelope-v1 limits are:

| Reason code | Exact limit |
|---|---:|
| `CSV_RAW_BYTES` | 67,108,864 |
| `CSV_RECORD_CHARACTERS` | 1,024 |
| `CSV_FIELD_CHARACTERS` | 256 |
| `CSV_DATA_ROWS` | 1,000,000 |
| `NPZ_RAW_BYTES` | 33,554,432 |
| `NPZ_CENTRAL_DIRECTORY_BYTES` | 16,384 |
| `NPY_HEADER_BYTES` | 4,096 |
| `NPY_ELEMENTS` | 1,000,000 |
| `NPZ_MEMBER_UNCOMPRESSED_BYTES` | 8,004,108 |
| `NPZ_TOTAL_UNCOMPRESSED_BYTES` | 16,008,216 |

Resource failures use a path-free stable prefix followed by an exact observation:

```text
INPUT_RESOURCE_LIMIT_EXCEEDED_V1 reason=<CODE> limit=<INTEGER> observed=<INTEGER>
INPUT_RESOURCE_LIMIT_EXCEEDED_V1 reason=<CODE> limit=<INTEGER> observed_at_least=<INTEGER>
```

These input-loading limits are independent of and not interchangeable with the `B`, `C`, `N`,
and `S` in-memory execution budget. The loading envelope bounds construction of a `SeriesPair`;
the execution budget separately bounds calibration work after an input has been accepted. This
is not streaming or out-of-core support, a security certification, M6 scientific evidence,
release readiness, or SoftwareX submission readiness.

## Run the deterministic example

The repository includes a small end-to-end example that resolves a v2 plan, performs
selection-aware surrogate calibration, verifies the returned result, and emits stable JSON:

```console
python examples/basic_selection_aware_calibration.py
```

The example selects candidate lag 2 and reports a full-reselection global `p_value` of
`0.2` for its fixed synthetic input and seed. This example is an executable contract demonstration, not M6 scientific evidence.

## Inspect fail-closed behavior

The second example deliberately supplies a series and a valid null model that cannot produce
any surrogate (five samples form a single shuffle block):

```console
python examples/fail_closed_not_evaluable.py
```

SelCal returns `status="not_evaluable"`, `failure_stage="null_bind"`, and null decision
fields without executing any surrogate replicates. `NOT_EVALUABLE is not evidence of no effect or non-significance.`

## Exercise binned NetTE with block shuffle

Binned NetTE uses equal-width bins between the observed minimum and maximum, with edges computed
exactly and rounded once, so they do not depend on the NumPy version. A value lying exactly on an
inner edge goes to the upper bin. Integer or otherwise discrete data often lie on edges; rescaling
such data by a factor that is not exact in binary floating point (for example 0.1) moves those values
off the edges and can change the result. Choose the number of bins with the data's values in mind.

The third example uses the other registered statistic/null combination: three-bin
equal-width NetTE with strict length-two block shuffling.

```console
python examples/binned_nette_block_shuffle.py
```

It runs exact-B full reselection, verifies the result, and emits deterministic JSON for
the fixed synthetic input. This broadens executable feature coverage; it is not a comparator or M6 result.

## Save and read complete result content

Save all candidate results, transformation tokens, retained failures and decisions
from the small synthetic example, then read the record in a separate process:

```console
python examples/save_and_read_result.py save result.json --max-bytes 65536
python examples/save_and_read_result.py read result.json --max-bytes 65536
```

The destination must not exist. The explicit byte budget is for this example;
oversized or malformed records are rejected rather than truncated. The read
summary is labelled `content_only_not_replay`: strict content reconstruction is
not authenticated execution, checkpoint recovery or a calibration replay.
Interrupted writes can leave a partial file; this example is not an atomic
evidence-bundle exporter. The API reference documents the direct development
imports from `selcal.result_wire` and their verification limits.

## Python API

The historical v1 public path remains available for plan identity and migration. It is
not executable calibration semantics:

```python
from selcal import PlanRequest, resolve_plan
from selcal.canonical import scientific_plan_sha256

request = PlanRequest(
    candidates=(1, 2, 3),
    statistic_name="lagged_pearson_v1",
    statistic_params={},
    selection_rule="max_absolute",
    null_name="circular_shift_v1",
    null_params={"min_shift": 1},
    replicates=999,
    alpha=0.05,
    tie_tolerance=1e-12,
    root_seed=17,
    failure_policy="fail_closed_v1",
)
resolution = resolve_plan(request)
print(scientific_plan_sha256(resolution.plan))
```

The candidate v2 in-memory execution path is:

```python
import numpy as np

from selcal import PlanRequestV2, calibrate_selected_family, resolve_plan_v2
from selcal.contracts import SeriesPair

pair = SeriesPair(
    source=np.asarray((0.0, 3.0, 1.0, 2.0, 4.0, 5.0), dtype=np.float64),
    target=np.asarray((0.0, 1.0, 3.0, 2.0, 5.0, 4.0), dtype=np.float64),
)

request_v2 = PlanRequestV2(
    candidates=(1, 2, 3),
    statistic_name="lagged_pearson_v1",
    statistic_params={},
    selection_rule="max_absolute",
    null_name="circular_shift_v2",
    null_params={"min_shift": 1},
    replicates=9,
    alpha=0.05,
    tie_tolerance=1e-12,
    root_seed=17,
)
resolution_v2 = resolve_plan_v2(request_v2)
result = calibrate_selected_family(pair, resolution_v2)
```

A v2 plan with up to 1,000,000 replicates may resolve successfully because that is a
plan-representability cap. Resolution does not imply that the current in-memory executor budget
admits the plan. The versioned candidate executor requires `B <= 1000`,
`B*C <= 5000`, `B*S <= 25000`, and `B*(C*N + S + N) <= 2500000`, where `S=1` for
circular shift and `S=N/block_length` for block shuffle. An over-budget call raises
`ResourceLimitError` before replicate allocation and returns no calibration result.

### Exact enumeration instead of Monte Carlo draws

`null_name="circular_shift_exact_v1"` (with `null_params={"min_shift": 1}`) uses every
non-identity circular state exactly once instead of drawing `B` states with replacement.
Set `replicates` to `n - 1`. Through the file workflow any other value is refused by `run`
(`unattainable_plan`); through the Python API it yields a NOT_EVALUABLE result at stage
`null_bind` with diagnostic `enumeration_replicate_count_mismatch_v1`. `verify` checks from the
retained tokens alone that replicate `i` used shift `i + 1`, so a record with repeated, missing
or reordered states is rejected even without `--replay`.
The p-value formula is unchanged, and with the complete group `(1 + E)/(B + 1)` equals the
exact share `#{s : T_s >= T_0}/n`, so the Monte Carlo sampling term disappears for this
enumerated state space. This is an algorithmic statement; finite-sample operating results
from internal studies are not independently reproducible from this source candidate.
The sampling null `circular_shift_v2` is unchanged, so existing plans and records keep their
behaviour.

**Recommended for new circular-shift plans:** `circular_shift_exact_v1` with `replicates = n - 1`,
whenever `selcal validate` reports `attainability.status` `PASS` and the circular-shift
exchangeability assumption is defensible for the task. When the guard refuses a plan because
the attainable p-value is too coarse, SelCal offers no calibrated fallback. Do not switch to
block shuffle merely to obtain a p-value; use a longer series or search fewer lags. The
internal operating-characteristic studies and non-adopted alternative are not part of this
source candidate's independently reproducible evidence.

Current implementation: immutable inputs, v1/v2 plan resolution, explicit
v1-to-v2 migration, common support, selection rules, binned NetTE, lagged Pearson,
owned circular-shift (sampled or exactly enumerated) and strict block-shuffle
transformations, exact-B full reselection, fail-closed Monte Carlo calibration, result
verification, and an internal finite-state algorithm oracle.

This is local algorithm-contract implementation, not scientific validation. Exact-oracle
agreement does not establish exchangeability, Type-I-error control, power, or validity
for a particular data-generating process.
