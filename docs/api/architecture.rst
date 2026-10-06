Architecture
============

Scope and status
----------------

SelCal currently implements an in-memory M0--M2 core. It resolves an immutable
scientific plan, evaluates a candidate family on observed data, applies an owned
null transformation, repeats the complete candidate-selection operation, and
returns a result bound to the resolved plan. Terminal persistence and a thin CLI
are implemented through a shared application service. Opt-in local checkpoints
support same-environment replay before continuing. Portable evidence bundles
support checked export and read-only verification through ``export`` and
``verify-export``. The basic local UI is a loopback-only HTTP shell around one
owned CLI subprocess. Its saved observations do not verify calculations.
General scientific validation and a public release remain unfinished.

Component boundaries
--------------------

.. list-table::
   :header-rows: 1
   :widths: 18 33 49

   * - Layer
     - Modules
     - Responsibility
   * - M0 contracts
     - ``selcal.contracts_v2``, ``selcal.canonical_v2``
     - Validate request fields, freeze plan identity, and provide canonical
       serialization for scientific hashes.
   * - M0 resolution
     - ``selcal.resolution_v2``, ``selcal.migration_v1_to_v2``
     - Resolve named statistics, selection rules, and null models without
       silently changing historical v1 identities.
   * - M1 statistics
     - ``selcal.statistics``
     - Evaluate binned NetTE or lagged Pearson on explicitly constructed common
       support.
   * - M1 null models
     - ``selcal.nulls``
     - Bind circular-shift or strict block-shuffle transformations to immutable
       observed inputs and SelCal-owned random tokens.
   * - M2 selection
     - ``selcal.selection_v2``
     - Select one candidate using the plan's rule and tie contract.
   * - M2 calibration
     - ``selcal.calibration_v2``
     - Recompute and reselect the complete candidate family for exactly the
       planned number of surrogates, then create a verifiable result.
   * - Internal oracle
     - ``selcal.exact_oracle_v0``
     - Check finite-state algorithm behavior independently of the production
       calibration path. It is not a domain-validity oracle.
   * - File application
     - ``selcal.workflow``, ``selcal.workflow_config``
     - Run actual CSV/NPZ requests through the core; bind captured input, config
       and complete result; separate read-only consistency from explicit replay.
   * - Terminal store
     - ``selcal.workflow_store``, ``selcal.result_wire``
     - Strict complete byte records in SQLite with exclusive writes and bounded
       read-only snapshots, separate from incremental execution state.
   * - Local recovery
     - ``selcal.workflow_recovery``, ``selcal.checkpoint_store``, ``selcal.checkpoint_lock``
     - Keep one writer session through retained-input loading, exact scientific
       prefix replay, transactional append/finalization and exclusive export.
   * - User entrypoint
     - ``selcal.cli``
     - Thin validate/run/resume/verify/report/export/verify-export/doctor commands
       sharing the application API.
   * - Portable evidence
     - ``selcal.workflow_export``
     - One checked terminal snapshot, bounded CSV/HTML projections, exclusive
       directory export and read-only raw/derived consistency verification.

Execution sequence
------------------

.. code-block:: text

   PlanRequestV2 + immutable SeriesPair
                |
                v
        resolve_plan_v2
                |
                v
   ResolvedScientificPlanV2 -- scientific_plan_sha256
                |
                v
   bind statistic + bind null transformation
                |
                v
   observed candidate scan -> observed selection
                |
                v
   for each planned replicate:
       transform full source -> rebuild support
       -> rescan all candidates -> reselect
                |
                v
   CalibrationResultV2 -> verify_calibration_result

Identity and determinism
------------------------

Plan serialization is canonical and versioned. The resolved plan hash identifies
scientific choices; it is not a hash of runtime outputs or a claim of correctness.
Randomness is derived from the frozen root seed and owned replicate tokens. The
examples therefore emit byte-identical JSON for their fixed inputs on the
currently verified environment. Cross-platform reproducibility remains a release
gate until a supported-platform CI matrix is executed.

Failure boundary
----------------

Invalid contracts fail during resolution. A scientifically impossible bound null
configuration returns a verified ``NOT_EVALUABLE`` result with null decision
fields. Integrity failures and execution defects are not converted into evidence
of no effect. The in-memory resource budget is checked before replicate
allocation.

Recovery boundary
-----------------

The store validates structure, sequence, canonical bytes and hashes. It grants no
scientific execution authority. The recovery service loads the retained raw input
through the ordinary strict loader, resolves and admits the original plan, and
runs the same observed calibration kernel once with the original B and seed.
Saved IDs and outcome bytes are compared exactly. Only after the complete retained
prefix matches may missing rows be committed; no prefix computation is skipped.
The returned result is always the result from that invocation.

The original UUID is operational and never enters scientific plan or seed
derivation. Python, NumPy, all package source bytes, platform identity and cap
must match. No cross-environment tolerance is used. The writer lock spans
snapshot through export. Finalized checkpoints are replayed and compared before
re-export; a failed export preserves the checkpoint. A nonempty saved prefix
followed by zero observed callbacks is rejected before terminal commit. The
checkpoint and exported record are independently bounded. This local recovery
does not establish historical authenticity, scientific validity, distributed
locking or a general power-loss guarantee.

Planned downstream layers
-------------------------

Terminal persistence and a thin CLI are implemented, but M3/M4/M5 are not fully
accepted modules. M3 now has transactionally tested local replay-before-continue;
M4 has complete portable export/manifest and read-only projection checking;
M5 needs final compatibility testing. These capabilities do not establish
historical authentication or scientific validity. Bounded synthetic calibration
and comparator studies have since been
executed internally; their evidence is not in this source candidate. General
scientific validity, an independent user evaluation and
the public research-evidence package remain open. UI recovery and evidence controls,
CI, containers, public
release and SoftwareX submission remain unfinished; a draft manuscript exists.

The 2026-09-28 R5 return includes native Windows 11/AMD64 tests on one host:
Python 3.11/3.12/3.13 with PowerShell 5.1 and Python 3.12 with PowerShell 7,
all using NumPy 2.4.6. Each configuration reports
3,466 passed / 20 skipped / 0 failed, including successful saved-record readback.
That return covers the earlier 40-file source snapshot. Subsequent checkpoint
kernel, store and recovery changes have no new native Windows execution. The
13 symlink-permission skips remain unverified Windows branches; the other seven
skips concern platform-specific capabilities. Actual visible browser rendering,
new machines and broader filesystem compatibility remain unverified. Repeated
configurations must not be counted as disjoint tests or as independent user evidence.

Basic UI boundary
-----------------

``selcal ui`` binds only to ``127.0.0.1``. Packaged HTML/CSS/JavaScript has no
external assets. Exact Host, same Origin when present, and a fresh session-token
header protect fixed API routes. Inputs use the existing raw CSV/NPZ limits;
the outer JSON transport cap is separate from the existing configuration codec.

Jobs freeze original configuration text, budget, override and raw input. Fresh
operation directories retain each command's output. The helper drains bounded
stdout and stderr, waits for process exit, and checks the single matching CLI
terminal record before showing success. Finalized checkpoint progress does not
establish saved-record success. Cancellation reaps only the owned child and
preserves work; restart marks formerly live observations interrupted or unknown.

Capability is ``PARTIAL_UI_BASIC_LOOP``: input, validation, run, resume, result
display, saved-record content checks, cancel and saved-job inspection. Resume binds the selected input and
canonical request to the owned checkpoint using the existing checkpoint store,
then delegates replay and continuation to the existing CLI with the original
byte budget and a fresh result path. Saved observations do not grant authority.
Successful run/resume operations atomically update a job-local ``record.json``
reference with schema, operation UUID, byte count and SHA-256. The fixed result
path remains under that job's operation directory. Verify binds its bytes, raw
input identity and canonical configuration using the existing workflow reader,
then delegates to CLI ``verify`` with the original byte budget and no replay.
The subject hash is checked again before accepting the matching terminal result.
Its scope is input/plan/result consistency, without historical authentication.
Missing or invalid references never trigger guessing or adoption of older records;
those records remain accessible through the CLI. Checks do not create new result
files, replace the target or grant authority to a saved observation.
Report and export use fixed authenticated empty-body POST actions and the existing
CLI, one owned child, original byte budget and new operation directories. Report
requires the real COMPLETE/NOT_EVALUABLE terminal and no replay. Export additionally
checks the exact eleven-member bundle schema, projection-consistency scope and
strict false historical-authentication flag. Before publication the shared
``verify_export`` summary must equal the child summary and all four source members
must equal ``read_record`` bytes. A fixed-metadata ZIP_STORED transport contains
only these eleven flat members. Only its headers may exceed the original budget
by at most 65,536 bytes; source/report/aggregate limits remain unchanged.

Atomic ``report.json`` and ``bundle.json`` references bind the entire record
descriptor and artifact bytes/SHA-256. Input, canonical configuration, original
metadata and record are rechecked before promotion; failure retains prior
references and files. Fixed authenticated GET routes
``/api/jobs/{id}/download/{record|report|bundle}`` reject busy/closing helpers and
return only fully checked bounded snapshots with fixed attachment names. Bundle
downloads repeat shared verification, exact four-source matching and exact ZIP
content checks. Availability never authorizes access. Missing references cannot
adopt old files. Binary responses omit charset; no-store/nosniff remain in force.

The browser fetches authenticated bytes and uses temporary Blob URLs for downloads,
revoking them after use. HTML previews are sandboxed without scripts or same-origin
permission; the CSP admits only the existing renderer's fixed inline CSS hash.
Generation/job guards reject late success, failure and body-read completions.
The interface adds no second scientific algorithm or result-verification engine.
Final integration, wider usability and release validation remain separate gates.
