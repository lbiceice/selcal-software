Usage guide
===========

Basic local interface
---------------------

.. code-block:: console

   selcal ui --workspace selcal-work

The helper binds only ``127.0.0.1`` and opens a local browser. The default port
is chosen by the operating system; ``--port 8765`` selects a port and
``--no-browser`` prints the launch URL without opening it. The fresh token is
in the URL fragment and then browser session storage. Reopen the printed URL
after restarting the helper. Keep its terminal open. Ctrl-C stops the helper
and its own child; closing the browser tab does not stop it.

On Windows, UI operations require unfrozen CPython with identifiable absolute
interpreter paths. The helper starts the actual base interpreter directly while
preserving the active virtual environment; unsupported runtime layouts are
refused before launch. This uses CPython's internal virtual-environment startup
convention. Native tests check worker ownership and environment identity; local
tests on another operating system do not establish Windows acceptance.

Use a new directory under a real existing parent, an empty real directory, or
an existing owned workspace. Linked and unknown nonempty directories are
refused. Load the included example or select CSV/NPZ input and edit the plan.
Create a job, validate, then run. The advanced JSON text is admitted unchanged,
preserving exact integer seeds. Each job's input, plan, byte budget and override
are immutable; any edit requires a new job before another run.
The initial UI byte budget is 8 MiB (8,388,608 bytes), sufficient for the
included example's checkpoint and terminal record. It is a caller-selected
storage budget applied separately to each, not a process-memory bound or a
universal input limit. Other plans may require a different explicit budget;
the existing scientific resource limits remain unchanged.

Progress is provisional until the child exits with a matching terminal result.
Exit 7 remains scientific NOT_EVALUABLE, with its original failure stage and
null p/decision. Input/guard/resource refusals remain operational outcomes.
Non-rejection does not prove absence of association or causality. Check the
circular exchangeability assumption and the prespecified candidate family.

Capability is ``PARTIAL_UI_BASIC_LOOP``. Saved jobs are observations, not
verified calculations. Formerly live jobs become interrupted/unknown after a
restart. Files are retained; interrupted initialization need not yield a
recoverable checkpoint. A Run cannot overwrite an existing checkpoint. Use the
Resume control after selecting the saved job. It restores the original input,
configuration, override and byte budget. Retained replicates are replayed and
checked before continuing under the same execution identity; they are not skipped.
Each attempt keeps earlier operations and writes to a fresh result destination.
Resume requires the original code/environment and a valid matching checkpoint.
Editing the form disables Resume until the saved job is selected again.
After successful Run or Resume, use Verify saved result to check input, plan and
result consistency through CLI ``verify`` with the original byte budget. It does
not replay calculations or authenticate historical execution. NOT_EVALUABLE keeps
its failure stage and null p/decision. Repeated checks save new observations, not
new result files. Editing disables the control until the saved job is reselected.
The strict ``record.json`` reference selects one owned record; missing, changed,
linked or mismatched records are refused without guessing another result. Older
jobs without a reference remain readable and can use ``selcal verify RECORD
--max-bytes ORIGINAL_BUDGET``. Saved check observations are not current-byte verification.
Generate HTML report and Export evidence bundle are separate actions through the
existing CLI with the original budget, no replay, and fresh operation directories.
Preview HTML report uses a script-disabled sandbox. Download HTML report,
Download result record and Download evidence ZIP use authenticated fixed routes,
never arbitrary paths or token URLs. Files are named ``selcal-<job>-<operation>-report.html``,
``…-result.sqlite`` and ``…-evidence.zip``; the page checks the received size and SHA-256 against
the ``X-SelCal-Bytes`` and ``X-SelCal-SHA256`` headers before handing a file to the browser. Each download rechecks original input and
configuration, the exact selected record and the saved artifact. The ZIP contains
only the eleven checked F3 files, with all four source members byte-identical to
the selected record. Its transport headers alone may use 65,536 extra bytes; the
original budget still bounds the source, report and aggregate exported members.
The budget does not bound process memory. Download initiated is not confirmation
of a file saved on disk. Edits, job changes and new requests clear the old preview.
Only successful actions update strict ``report.json`` / ``bundle.json`` references;
failed, partial or cancelled operations preserve earlier files and references.
References bind the entire original ``record.json`` descriptor, so Resume makes
old derivatives ineligible without deleting them. No missing reference is guessed
from timestamps or adopted from old directories. Old records can use CLI report
and export. Availability is advisory, not verification or download authorization.
UI reports, export and downloads are implemented; final integration,
current installed/browser acceptance and new native Windows validation remain separate gates.

Development installation
------------------------

Use Python 3.11 or newer in an isolated environment and install the checked-out
snapshot from the repository root:

.. code-block:: console

   python -m pip install .

This installs a development snapshot, not a public release.

Complete calibration example
----------------------------

Run the deterministic complete route with:

.. code-block:: console

   python examples/basic_selection_aware_calibration.py

The script resolves the request, performs complete surrogate reselection, verifies
the result, and prints machine-readable JSON. For its fixed synthetic input it
selects candidate 2 and reports ``p_value=0.2``. That number is an example-specific
contract result, not scientific validation or evidence for a domain claim.

.. literalinclude:: ../../examples/basic_selection_aware_calibration.py
   :language: python
   :linenos:
   :caption: examples/basic_selection_aware_calibration.py

Interpret a complete result only after ``verify_calibration_result`` succeeds.
The ``scientific_plan_sha256`` binds the result to the resolved plan;
``planned_replicates`` states the requested Monte Carlo count; and
``failure_count`` records retained replicate failures rather than silently
shortening the denominator.

Binned NetTE and block shuffle
------------------------------

The third example covers the other registered statistic/null combination:
equal-width binned NetTE with strict block shuffling.

.. code-block:: console

   python examples/binned_nette_block_shuffle.py

The fixed input has length 10, three bins, candidate lags 1 and 2, block length
2, and nine planned replicates. The verified output selects candidate 1 and
reports ``p_value=0.4``. This broadens executable feature coverage; the example
does not compare SelCal with another package or establish an M6 result.

.. literalinclude:: ../../examples/binned_nette_block_shuffle.py
   :language: python
   :linenos:
   :caption: examples/binned_nette_block_shuffle.py

Fail-closed example
-------------------

Run the deliberately non-evaluable route with:

.. code-block:: console

   python examples/fail_closed_not_evaluable.py

The fixed request has a single shuffle block, so the null cannot produce any
surrogate. SelCal returns
``status="not_evaluable"``, ``failure_stage="null_bind"``, no p-value, no reject
decision, and no surrogate replicates. ``NOT_EVALUABLE`` is not evidence of no
effect or non-significance.

.. literalinclude:: ../../examples/fail_closed_not_evaluable.py
   :language: python
   :linenos:
   :caption: examples/fail_closed_not_evaluable.py

Save and read complete result content
--------------------------------------

The following commands execute in separate processes. The first calculates the
small synthetic basic example and writes every observed candidate, replicate,
transformation token, failure field and decision. The second strictly reads the
saved record; it does **not** rerun the calibration.

.. code-block:: console

   python examples/save_and_read_result.py save result.json --max-bytes 65536
   python examples/save_and_read_result.py read result.json --max-bytes 65536

The output file must not already exist. The example cap of 65,536 bytes is a
caller-selected budget for this small record, not a tested peak-memory bound or
a scientific threshold. Oversized or malformed input is rejected, not truncated.
An I/O failure is reported as an operation failure, never as a scientific result;
an interrupted write can leave a partial file, which must not be treated as a
completed record. The example is not an atomic evidence-bundle publisher.

The wire uses schema ``selcal.calibration-result-wire.v1``, UTF-8 JSON, exact
hexadecimal float tags and one terminal newline. Duplicate, missing and extra
fields, non-finite numbers and noncanonical encodings are rejected. Null values
and analytical failures are retained. The API returns a restored
``CalibrationResult`` whose content is internally checked, not authenticated
historical execution. A fresh resolver and ``verify_calibration_result`` can
check the documented result/plan consistency; neither substitutes for checking
the real input or executing a replay. No checkpoint/resume or full evidence
bundle is supplied by this example.

.. literalinclude:: ../../examples/save_and_read_result.py
   :language: python
   :linenos:
   :caption: examples/save_and_read_result.py

File-driven workflow
---------------------

For the actual file workflow, use these commands after installation:

.. code-block:: console

   selcal validate examples/workflow/series.csv examples/workflow/pearson.json
   selcal run examples/workflow/series.csv examples/workflow/pearson.json run.sqlite --max-bytes 1048576
   selcal verify run.sqlite --max-bytes 1048576 --replay
   selcal report run.sqlite report.html --max-bytes 1048576

The CSV/config fixtures are synthetic demonstrations. ``python -m selcal`` is an
equivalent entrypoint. Ordinary verification and reports do not recompute; only
explicit ``--replay`` performs a full calibration and exact-byte comparison.
Records capture their own input and request, without relying on original paths.
Reading captured terminal data is not checkpoint/resume or historical execution
authentication. Output paths must not exist. Scientific NE exits with code 7;
invalid requests use 2 and operation/integrity failures 4, with no invented p-value.

Portable evidence export
------------------------

Export the complete captured terminal to a new directory and check it without
the SQLite source:

.. code-block:: console

   selcal export run.sqlite evidence --max-bytes 8388608
   selcal verify-export evidence --max-bytes 8388608

``selcal.workflow_export.export_record`` and ``verify_export`` are the shared
Python services. The eleven flat files comprise original ``input.csv`` or
``input.npz``; exact stored ``request.json``, ``result.json`` and ``metadata.json``;
canonical ``plan.json`` and recorded ``software.json``; ``summary.csv``,
``candidates.csv``, ``replicates.csv``, ``report.html`` and ``manifest.json``.
The plan file SHA-256 is the scientific-plan identity. The software identity is
the recorded identity, including absent platform information in old v1 records.

All failed and unselected candidates and every retained replicate remain in
their stored order. Empty tables still have headers. CSV has UTF-8/LF encoding,
fixed columns, round-trippable floats, lowercase booleans, empty missing values,
and compact JSON for object/array cells. Formula-leading text receives an
apostrophe; negative numeric values remain numeric. The authoritative result
JSON retains exact binary64 encodings and complete transform tokens unchanged.

``max_bytes`` bounds both the source and aggregate bundle including manifest;
the example uses a caller-selected 8 MiB cap (8,388,608 bytes). Expanded CSV/HTML
can need a larger cap. This does not bound peak process memory.
The parent must exist as a real directory and the output must be new. No
overwriting or recursive cleanup occurs. On a write/close failure, preserve the
directory, inspect it with ``verify-export``, or use a fresh destination. If all
manifest bytes exist before its close fails, later content verification can
succeed even though export failed. Content verification does not prove durability.

Both services validate one captured set of bytes and regenerate all derived
files, so an altered projection with a recomputed manifest digest still fails.
They do not calculate attainability or perform calibration/replay. Ordinary
``report`` retains its existing attainability summary. Neither service proves
historical execution authenticity, scientific validity or independent replication.
Scientific ``NOT_EVALUABLE`` retains exit 7 after successful export/verification;
invalid arguments use exit 2 and content/I/O failures exit 4. Hostile same-user
concurrent mutation is outside this guarantee.

Local checkpoint recovery
-------------------------

Opt in when starting a run, using a new directory. After an interruption, inspect
the retained directory and request recovery to a new output filename:

.. code-block:: console

   selcal run examples/workflow/series.csv examples/workflow/pearson.json run.sqlite --max-bytes 8388608 --checkpoint run-checkpoint
   selcal resume run-checkpoint recovered.sqlite --max-bytes 8388608

``run_checkpointed_files`` and ``resume_checkpoint`` in
``selcal.workflow_recovery`` return ``RecoveryRun``: the actual result computed
in this call, the original execution UUID, and replayed/appended counts. An optional
callback receives immutable ``RecoveryProgress`` operational counters after each
successfully matched replay or committed append, and after finalization. Callback
errors propagate as execution interruptions/errors, never scientific NE.

Recovery uses the retained raw CSV/NPZ and normalized configuration, not the
original paths. It resolves the same plan and rechecks admission, then invokes the
complete real calibration exactly once. Every saved replicate ID and full outcome
must match exactly before missing outcomes may be appended. It does not skip the
retained computation or change B, seed, candidates, input, override or plan.
``resume`` accepts only the directory, a new output path and the original byte cap.
The entire source/Python/NumPy/platform identity must match; no tolerance or
environment migration is offered. A finalized checkpoint is also fully replayed
before another export.

The same local writer lock is retained until export finishes. Busy, damaged,
identity-mismatched or replay-mismatched checkpoints fail with exit 4 and a
specific code. Scientific NE remains exit 7; request/admission refusal remains
exit 2. Progress milestones go to stderr, and stdout has one final JSON document
with a nested ``checkpoint`` summary and mode ``replay_before_continue``.
Observed interruption is exit 130; a killed process need not emit final JSON.

Initialization can fail before a usable checkpoint exists. Merely passing the
flag does not establish recoverability. Existing files are never overwritten or
deleted. Export failure may leave a valid finalized checkpoint that can be
replayed to a new filename. A new ordinary result filename inside an existing
checkpoint directory is allowed. Its internal names ``state.sqlite``,
``writer.lock``, ``state.sqlite-journal``, ``state.sqlite-wal`` and
``state.sqlite-shm`` are reserved case-insensitively; directory aliases cannot
be used to export over them. Such a request fails before replay or mutation.
The byte cap separately bounds the checkpoint and
terminal record; neither implies the other fits, and it is not a peak-memory or
journal-size bound. Supported local filesystem locking is required; network or
distributed recovery and general power-loss durability are not promised.
Portable evidence commands ``export`` and ``verify-export`` are implemented;
they check retained content without replay. UI status is ``PARTIAL_UI_BASIC_LOOP``.
UI resume and record content checks are implemented. UI reports, export and downloads are implemented. Neither
recovery nor portable export has received new native Windows validation; older
Windows results do not validate these changes.

Reading the output
-------------------

``selected_candidate`` is the reported lag label; ``tied_candidates`` preserves
the tolerance-defined tie set. The family decision statistic is distinct from
the lag label. It does not identify a causal edge. Every surrogate rescans the
declared candidate family; the result is not a test performed only at the lag
chosen from the observed data.

``planned_replicates`` is B, not the number of successful replicates. When the
result is complete, the implemented Monte Carlo tail is (1 + E) / (B + 1), where
E is the recorded exceedance count. For a replicate-stage failure, p and the
decision remain null and the retained diagnostics/bounds are not a replacement
p-value. Pre-replicate NE retains its distinct failure stage and no invented
replicate outcomes. Increasing B cannot repair an inappropriate null model.

The example parameters are explicit test-fixture choices, not recommended
defaults for arbitrary research data. Bounded synthetic operating-characteristic
studies have been executed internally, but they do not establish validity for
arbitrary data-generating processes or task-specific parameter guidance. Their
research evidence is not shipped in this source candidate.

Current claim boundary
----------------------

These examples demonstrate executable API and failure contracts. They do not
establish null-model exchangeability, Type-I-error control, statistical power,
workflow superiority, independent reuse, or SoftwareX readiness. Bounded
synthetic studies have been executed internally, while independent reuse,
cross-domain validity and the public research-evidence package remain open.
