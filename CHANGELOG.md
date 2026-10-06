# Changelog

All notable changes to SelCal are listed here. Versions follow [Semantic Versioning](https://semver.org/);
the format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## 0.1.0 - unreleased

First public version, in preparation. No public release, tag or DOI exists yet; the version
number in `pyproject.toml` and `CITATION.cff` is the candidate for that release.

### Added

- Selection-aware calibration of a scanned-lag dependence test: the complete lag search is repeated
  inside every circular-shift surrogate, with `p = (1 + E) / (B + 1)`, or exact enumeration of the
  whole circular group.
- Plans whose smallest attainable p-value exceeds alpha are refused unless explicitly allowed;
  nulls that are not a group under the declared support are refused for inference.
- Lagged Pearson with power-of-two scaling, two-pass centring and correctly rounded sums.
- Verifiable SQLite records that bind input, plan, result and the exact source files;
  `selcal verify`, `--replay` and `--replay-decision`; HTML reports; eleven-file evidence export
  with `verify-export` (CSV tables for spreadsheets are inside the export).
- Bounded CSV and NPZ inputs, checkpoints with same-environment resume, a basic offline local
  interface, and a Python API (`selcal.calibrate_selected_family`).
- Offline Windows installation and acceptance entry points, and tools to check files saved from
  the local interface (`scripts/check_saved_downloads.py`) and to run named research cases with
  failure evidence (`scripts/research_case_flow.py`).

### Fixed during pre-release Windows testing

- Text resources and test files are read as UTF-8 on Windows with a cp936 locale.
- Browser downloads are checked for size and SHA-256 before they are handed to the browser, are
  named by job and operation, and can be copied from the workspace if no download appears.
- The Windows acceptance runner pins the locked pip from the offline wheelhouse before reading
  requirements, writes the lock export without a path header, and reports environment failures
  once, marking dependent steps as blocked.

### Fixed before release (2026-10-06)

- Exceedance counting: a surrogate whose decision statistic is within 64 units in the last place of
  max(|observed|, 1) of the observed one now counts as reaching it. Values that are equal in exact
  arithmetic could differ in their last bits, so ties were lost: on degenerate data a p-value fell
  below its L/n floor and a positive rescaling of the data changed the decision. One frozen rule is
  shared by the calculation, the result verifier and the exact oracle. Ordinary continuous data are
  unaffected (1,200 study-like runs and both re-analyses gave identical results); the bundled binned
  NetTE example changes from p = 0.4 to the correct 0.9 (six exact ln(2)/4 ties).
- Binned NetTE computes equal-width bin edges in exact rational arithmetic. np.linspace rounded them
  differently in NumPy 1.26 and 2.x, so data lying on an edge could change bins and results between
  NumPy versions. The preprocessing identity is renamed accordingly.
- Outputs refuse a name occupied by a dangling symbolic link (Windows exclusive creation followed it).
- The local interface holds a lock on its workspace: a second helper on the same workspace is refused
  instead of marking the first helper's running jobs as interrupted.
- A job folder that cannot be read (for example after an interrupted job creation) is left untouched
  and reported; it no longer stops the workspace and its healthy jobs from opening.
- Interface child processes run in their own operation folder, so a module named selcal in the folder
  where the helper was started cannot replace the installed package.
- `scripts/verify_installed_identity.py` checks an installation against its RECORD hashes and,
  optionally, against the wheel it should come from.

### Documentation and metadata

- The README is a short user page; the complete user reference and the development and platform
  record are in `docs/guide/`, with result-interpretation and interface guides.
- `codemeta.json` added. The author confirmed on 2026-10-05 the right to release the source under
  BSD-3-Clause (`docs/provenance/SOURCE_ORIGIN.tsv`).
- Source repository: https://github.com/lbiceice/selcal-software
