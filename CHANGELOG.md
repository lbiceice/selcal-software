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

### Documentation and metadata

- The README is a short user page; the complete user reference and the development and platform
  record are in `docs/guide/`, with result-interpretation and interface guides.
- `codemeta.json` added. The author confirmed on 2026-10-05 the right to release the source under
  BSD-3-Clause (`docs/provenance/SOURCE_ORIGIN.tsv`).
- Source repository: https://github.com/lbiceice/selcal-software
