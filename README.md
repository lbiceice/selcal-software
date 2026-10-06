# SelCal

**Selection-aware calibration of scanned-lag dependence tests between two time series.**

Analysts who scan several lags and report the strongest cross-correlation often test that winner as if
its lag had been fixed in advance, and often ignore autocorrelation; both inflate false positives.
SelCal implements the published remedy (repeat the whole lag search inside every circular-shift
surrogate; Cannistra et al., eLife 2025; Yuan & Shou, PLoS Biology 2024) as reusable software, and adds:

- **a frozen analysis plan**: candidate lags, statistic, selection rule, null model, number of
  surrogates and alpha are fixed before any data-dependent step and stored with every result;
- **exact enumeration** of the complete circular group (`circular_shift_exact_v1`), removing Monte Carlo
  loss;
- **an attainable-p guard**: in the circular Pearson lag scan, at least L of the n circular states
  reach or exceed the observed maximum, so the exact p-value cannot fall below L/n;
  `selcal validate` reports this and `selcal run` refuses unattainable exact-enumeration plans;
- **a three-level preflight** separating input validity, plan executability and the scientific
  assumptions the analyst must justify;
- **verifiable, replayable records**: one SQLite file with input, plan, full result and software
  identity; `selcal verify --replay` recomputes it byte for byte.

SelCal tests one pair of series over a small, pre-declared lag set. It does not perform causal
discovery, estimate effect sizes, align or detrend series, or correct across many pairs.

## At a glance

| | Command line or Python | Local browser interface (`selcal ui`) |
|---|---|---|
| **You provide** | Two aligned numeric series (CSV with two columns, or NPZ) and a JSON plan: candidate lags, statistic, selection rule, null model, number of surrogates, alpha, seed | The same two files, chosen in the page; or the included example |
| **You get** | One SQLite record (input, plan, full result, software identity); an HTML report; an eleven-file evidence folder with `summary.csv`, `candidates.csv` and `replicates.csv` for spreadsheets | The same three files, downloaded from the page after a size and SHA-256 check |
| **Main result** | Selected lag, its statistic, exceedance count E over B surrogates, `p = (1 + E) / (B + 1)`, decision at alpha, and the plan's smallest attainable p | Shown on the page and in the report |
| **Typical time** | Included example (n = 100, 3 lags, 199 surrogates) on an Apple M4 Pro: validate 0.11 s, run 0.31 s, `verify --replay` 0.33 s, report 0.13 s, export 0.14 s (median of 3) | Same computations; one click per step |

How to read the numbers, when a plan is refused and what a non-significant result does and does not
mean: [Interpreting SelCal results](docs/guide/interpreting_results.md). The browser interface is
shown in [docs/guide/interface.md](docs/guide/interface.md).

![SelCal local interface: completed run of the included example](docs/guide/images/ui_2_result.png)

## Quick start

The source repository is currently private; the clone commands below require authorized access.
There is no public install route for the exact evaluated revision yet. Use a fresh virtual environment
(Python 3.11-3.13). The macOS/Linux instructions have local test evidence. Native Windows
results are available for the limited configurations recorded in
[docs/guide/development.md](docs/guide/development.md#platform-test-record); they are not a claim of
compatibility with every Windows installation.

macOS / Linux:

```console
git clone https://github.com/lbiceice/selcal-software.git
cd selcal
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
```

Windows (PowerShell 5.1 or 7; put the folder at a short path such as `C:\src\selcal`):

```console
git clone -c core.autocrlf=false https://github.com/lbiceice/selcal-software.git
cd selcal
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install .
```

(`core.autocrlf=false` keeps source-file bytes unchanged; a downloaded source ZIP also works.
If PowerShell blocks activation, no policy change is needed: run
`.venv\Scripts\python.exe -m pip install .` and use `.venv\Scripts\python.exe` in place of
`python`, or `.venv\Scripts\python.exe -m selcal` in place of `selcal` below.)

The runtime installation needs Python and NumPy, but no API key, GPU, Java, or paid service.
The first installation needs network access unless the dependencies are already cached.
The commands below use files in the source folder; keep that folder available after installation.

After installation (check your environment with `selcal doctor`):

```console
selcal doctor
python scripts/acceptance_check.py
selcal validate examples/workflow/series.csv examples/workflow/pearson.json
selcal run examples/workflow/series.csv examples/workflow/pearson.json run.sqlite --max-bytes 1048576
selcal verify run.sqlite --max-bytes 1048576 --replay
selcal report run.sqlite report.html --max-bytes 1048576
```

`selcal doctor` saves and reads back a small record in the temporary folder and exits with 4 if
records cannot be used; `selcal doctor --folder DIR` checks the folder where you will keep records
(for example a synchronised or network folder).
`scripts/acceptance_check.py` runs the whole workflow on both example plans and checks the results
(selected lag 2; p = 0.015 with 199 sampled surrogates, p = 0.03 with exact enumeration; null
rejected; replay MATCH; the HTML report shows the same values). It prints PASS/FAIL per check and
exits 0 only if everything passed. Open `report.html` in a browser to read the result.

### Use your own two series

1. Prepare a UTF-8 CSV with **exactly two numeric columns** and one header, for example
   `temperature,cases`. Each row must be one aligned observation at the same sampling interval.
   Resolve missing observations, alignment and preprocessing before using SelCal; it does not
   silently drop rows, fill gaps, resample or detrend.
2. Copy `examples/workflow/pearson.json` to `my-plan.json`. Set `source_column` to
   `temperature` and `target_column` to `cases` (or your actual header names). Choose the lag set,
   null model and other plan settings **before inspecting which result is strongest**. Lag units
   are rows: with monthly data, lag 2 means two months.
3. Run `selcal validate my-data.csv my-plan.json`. Correct any named input or plan problem first.
   `EXECUTABLE` is not evidence that the null-model assumptions hold: review the separate
   `scientific_assumptions` field. Do not bypass a refusal just to obtain a p-value.
4. Run and retain the record, then make a report:

   ```console
   selcal run my-data.csv my-plan.json my-run.sqlite --max-bytes 1048576
   selcal verify my-run.sqlite --max-bytes 1048576 --replay
   selcal report my-run.sqlite my-report.html --max-bytes 1048576
   ```

Use new output filenames for another run. An existing-file error protects the previous result;
it is not a reason to delete that record. Exit 7 (`NOT_EVALUABLE`) means no valid p-value was
computed, not evidence of no relationship. A non-significant valid result also does not prove
absence of a relationship. Reuse across domains requires the same input structure **and** a
defensible null model, not just renamed columns.

Requirements: Python 3.11 or newer (tested: 3.11-3.13) and NumPy (`numpy>=1.26,<2.5`). Platform
test rounds, including the native Windows results and what they do not establish, are recorded in
[docs/guide/development.md](docs/guide/development.md#platform-test-record).

## Validation

- **Correctness checks shipped with the source**: the product test suite (run it with
  `uv run --locked --all-extras python -m pytest`), the runnable examples in `examples/` and
  `scripts/acceptance_check.py`, which checks the documented workflow end to end. The test suite
  includes an internal finite-state algorithm oracle; agreement with it is an algorithm check,
  not scientific validation.
- **Records**: `selcal verify` checks a saved record; `--replay` recomputes it byte for byte in
  the recording environment, and `--replay-decision` across platforms with the same NumPy.
- **Statistical operating characteristics**: the bounded synthetic studies were executed
  internally and are reported in the accompanying manuscript; see the evidence boundary below.

## Research evidence boundary

This source candidate contains the software, runnable examples, API documentation, product
tests and benchmark fixtures. It does **not** include the unpublished manuscript or its
internal research protocols, scripts, results and third-party data. The bounded synthetic
studies have been executed internally, but their manuscript numbers cannot be independently
recomputed from this package alone. A rights-reviewed evidence package must be attached and
checked before any public research-reproducibility claim or SoftwareX submission.

## Documentation

| Page | Contents |
|---|---|
| [Interpreting results](docs/guide/interpreting_results.md) | What the numbers mean, refusals, exit codes, what a result does not show |
| [Local interface](docs/guide/interface.md) | The browser interface, with screenshots |
| [User reference](docs/guide/reference.md) | File workflow, attainability guard, recovery, interface details, evidence export, input limits, examples, Python API |
| [Development and platform record](docs/guide/development.md) | Locked environment, CI, containers, API reference build, performance, platform test rounds |
| API reference | Build with Sphinx from `docs/api` (see the development page) |

## Citation, licence and support

- Cite the software with `CITATION.cff`. A DOI for the exact public version is pending.
- Licence: BSD 3-Clause (`LICENSE.txt`; `Licence.txt` is an identical copy required by SoftwareX).
- The historical GitHub repository is currently private; public issues are unavailable.
  Until a clean public repository is verified, use the maintainer contact in `CITATION.cff`.

## Project status

Version 0.1.0 is a candidate for the first tagged release. The historical internal milestone label
`M0-M2 IMPLEMENTATION CANDIDATE / FINAL VERIFICATION PENDING` is not a current test-count report;
the full status record is in [docs/guide/development.md](docs/guide/development.md#project-status).

`SCIENTIFIC EVIDENCE: BOUNDED SYNTHETIC STUDIES EXECUTED / GENERAL IMPACT AND INDEPENDENT-USER BENEFIT HOLD`

`LICENSE: BSD-3-CLAUSE / RELEASE: 0.1.0 LOCAL CANDIDATE (PUBLIC RELEASE PENDING) / DOI: PENDING /
MANUSCRIPT: IN PREPARATION, NOT SUBMITTED`

`LOCAL RESUME: REPLAY_BEFORE_CONTINUE / EVIDENCE BUNDLES: CONTENT-CHECKED / UI: PARTIAL_UI_BASIC_LOOP`

SelCal is not a causal-edge inference product. Records are terminal results that can be
verified and replayed. Opt-in checkpoints support local same-environment recovery by
replaying the retained prefix before continuing. Portable evidence bundles can be
exported and checked without replay. The graphical interface implements the basic
input/validate/run/resume/result-content-check loop plus reports, evidence export
and checked downloads. F4C integration is locally accepted on macOS; container,
renewed native Windows and release gates remain separate.
