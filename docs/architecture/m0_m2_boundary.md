# SelCal M0–M2 boundary

> Historical record of the 2026-08 M0-M2 checkpoint, retained for contract provenance.
> Its statuses below are superseded, not current. BSD-3-Clause was selected later;
> version 0.1.0 remains a local
> candidate and had no verified public release as of 2026-09-27. Consult README.md for
> current product and evidence boundaries.

SelCal is a pre-release scientific-plan and statistic core. It is not a causal-edge
inference product and is not SoftwareX submission-ready. M3–M8 are unpassed. No license
has been selected, so redistribution is not authorized by this development snapshot.

`M0-M2 IMPLEMENTATION CANDIDATE / FINAL VERIFICATION PENDING`

The controller design authorities are `2026-08-25-p3-selcal-stage3-design.md` and
`2026-08-26-p3-selcal-multi-adapter-calibration-audit-design.md`; controller commits
`f914ad059` and `2aa257c51`; no machine-local controller path is part of portable contract.

## Target scope: M0–M2

- Immutable v1/v2 contracts and data (implementation candidate).
- Statistic adapters and owned v2 null transformations (implementation candidate).
- Deterministic, selection-aware, exact-B full-reselection calibration
  (implementation candidate; final verification pending).
- Resolution-bound result verification and an internal finite-state algorithm oracle
  (implementation candidate; final verification pending).
- Failure, determinism, resource-cap, and common-support tests (implemented locally;
  final verification pending).

## Current M0–M2 capability boundary

| Capability | Status |
| --- | --- |
| Unresolved request → registered contracts | Implemented and locally tested |
| v1 scientific hash identity | Golden vectors preserved locally |
| Binned NetTE statistic | Implemented and locally tested |
| Lagged Pearson statistic | Implemented and locally tested |
| Circular-shift/block-shuffle preservation contracts | M0-M2 implementation candidate; final verification pending |
| Full surrogate reselection calibration | M0-M2 implementation candidate; final verification pending |
| Resolution-bound result verifier | M0-M2 implementation candidate; final verification pending |
| Independent finite-state algorithm oracle | Internal implementation candidate; not top-level public API |
| M6 calibration and comparator evidence | HOLD / not executed |
| SoftwareX scientific impact | HOLD |

V2 requests with up to 1,000,000 replicates may resolve successfully under the plan
representability contract. Resolution does not imply admission by the current
in-memory executor budget. Execution is separately bounded by `B <= 1000`, `B*C <= 5000`,
`B*S <= 25000`, and `B*(C*N + S + N) <= 2500000`; over-budget execution raises a typed
resource error before the replicate loop and produces no result object.

## Out of scope: M3–M8

- Immutable execution, resume, and verification.
- Reporting.
- A command-line interface.
- A scientific comparison study.
- A thin user interface.
- Public release.
- A SoftwareX manuscript or submission.

`M6 SCIENTIFIC IMPACT: HOLD / NOT EXECUTED`

`PERSISTENCE, EVIDENCE BUNDLES, UI, RELEASE, LICENSE, DOI, MANUSCRIPT, SOFTWAREX
READINESS, SUBMISSION: HOLD / NOT DUE / NOT READY`

## 2026-08-27 Task-7 stop/go record

`CONTRACT_RESOLUTION_AND_PEARSON: PASS`

This is a local engineering PASS for contract resolution and lagged Pearson only. It is
not scientific validation, calibration evidence, completion of M0–M2, M6 evidence,
release evidence, SoftwareX readiness, or submission evidence.

### Provenance and retained initial HOLD

- The initial audit baseline was
  `c2c69add1c113300a969db417c96ccb88554097e`; the initial HOLD record was committed as
  `cdac49331fc2006748be50eab1a454acb523c60b`.
- On that unchanged baseline, `.venv/bin/python -m mypy src` exited 1:
  `binned_nette.py:192` passed an `ndarray[..., floating[Any]]` where an
  `ndarray[..., float64]` was declared. The finding was reproduced with mypy 1.11.2,
  1.17.1, and 1.20.2.
- On the same baseline, `.venv/bin/python -m ruff check src tests` exited 1 with two
  `UP038` findings, at `contracts.py:184` and `test_support.py:32`. Ruff 0.6.9 and
  0.12.12 both reproduced the findings.
- Those results remain adverse evidence. Their disposition is
  **initial HOLD, superseded by fix**. Commit
  `59a1ab0c64915ee0fab39af42e33c2a63c3a44b9` made the minimal static
  type/lint repairs and is the source-and-test byte identity verified for this PASS. Two
  independent specification/quality reviews found the repair in scope and passed it.

The adverse environment history is also retained: the prior ignored environment used
CPython 3.14.3 and NumPy 2.5.2, and the exact mypy command exited 2 because NumPy
2.5.2 stubs use Python 3.12 type-statement syntax while this project targets Python
3.11. That environment was preserved recoverably outside the tracked worktree; it was
not deleted or treated as a source defect.

### Evidence closing the local gate

The primary evidence environment was CPython 3.12.10, NumPy 2.2.6, pytest 8.4.2,
pytest-cov 6.3.0, mypy 1.20.2, Ruff 0.6.9, and build 1.3.0:

- Ruff exited 0, and mypy checked all 17 source files with exit 0.
- The full coverage run passed 594 tests and skipped 1 platform-dependent extended-float
  test. Total coverage was 95.49%, with `resolution.py` at 100% and
  `lagged_pearson.py` at 91%.
- Wheel and sdist builds exited 0. The wheel installed into a fresh environment, and the
  README example produced
  `fbded1df530825b5f95664f809c7b11e0f00a860f2fbb195cb364e16e250acc7`.
- Two independent processes using candidate orders `(3, 1, 2)` and `(2, 3, 1)` produced
  that same digest.
- The two legacy golden digests remained
  `41089d28130ba5e8f42b3205618c147d7dde3d7ebf32173c750d2c545f0db150` and
  `43430161e87194afef9cebb3c5b278e5784632690e7d7a8a8ccf5932870b9cdf`; the Pearson
  mixed maximum/subnormal-scale regressions passed all 4 targeted tests.

The repaired bytes were also verified under CPython 3.11.12 and NumPy 2.4.6: 594 tests
passed, 1 platform-dependent test was skipped, and Ruff and mypy both exited 0. An
independent quality review additionally used NumPy 1.26.4, obtained mypy exit 0 and 254
targeted test passes, and found runtime diagnostic bytes equivalent between the baseline
and repaired HEAD across 30 test groups.

The tracked worktree and diff checks were clean at source verification. Generated build
and tool-cache outputs are ignored; they are local verification by-products, not release
artifacts or release evidence.

`M0-M2: IMPLEMENTATION CANDIDATE / FINAL VERIFICATION PENDING`

`M6 SCIENTIFIC_IMPACT: HOLD / NOT EXECUTED`

`UI, RELEASE, SOFTWAREX MANUSCRIPT, SUBMISSION: NOT DUE / NOT READY`

## 2026-08-27 SoftwareX contract refresh

```json
{
  "schema": "selcal.softwarex-contract-checkpoint.v1",
  "as_of": "2026-08-27",
  "status": "historical_superseded_checkpoint",
  "contract_validation": "HOLD",
  "closed_as_current_blockers": [
    "v2_design_written_approval",
    "full_reselection_implementation"
  ],
  "current_gate_status": {
    "m0_m2": "IMPLEMENTATION_CANDIDATE_FINAL_VERIFICATION_PENDING",
    "m6_impact_and_comparators": "HOLD",
    "public_release": "HOLD_NOT_DUE",
    "license": "HOLD_NOT_DUE",
    "research_data": "HOLD_NOT_DUE",
    "clean_install_and_documentation": "HOLD_NOT_DUE",
    "rights_and_author_facts": "HOLD_NOT_DUE",
    "live_portal": "HOLD_NOT_DUE"
  }
}
```

This is a historical checkpoint recording the contract state as of 2026-08-27. Its
design-approval and implementation items are superseded as current blockers by the
Task-10 state below; the adverse contract evidence and downstream gates remain retained.

At that checkpoint, the SoftwareX Guide for Authors, Version 6 OSP templates, and
official reviewer form had been re-verified and frozen in
`docs/official_contract/softwarex_product_contract_20260827.json`. The contract
validator returned `HOLD` with zero structural errors and zero warnings. The live Guide
was no longer an unavailable-source blocker. The blocker ledger then included written
approval of the v2 null-reselection design, implementation of full reselection, M6
impact and comparator evidence, public GitHub release, an author-selected code license,
research-data handling, independent clean-install and documentation evidence, rights
and author facts, and a future live portal draft.

Historical reconciliation as of 2026-08-31: written design approval and full-reselection implementation were
no longer blockers to the then-current
`M0-M2 IMPLEMENTATION CANDIDATE / FINAL VERIFICATION PENDING` state. This does not close
final verification or scientific validity. A current dependency lock, same-machine
CPython 3.11 source installation, CPython 3.12 wheel installation, README, examples and
API documentation existed. The then-current internal contract and validation report were
`docs/official_contract/softwarex_product_contract_20260831.json` and
`docs/official_contract/softwarex_product_contract_validation_20260831.json`; the
2026-08-27 pair was historical evidence. The remaining HOLD gates at that checkpoint were Task-10 authority
scope-v2 closure; M3, M4 and the M5 CLI; M6 impact and comparator evidence; public GitHub
release; an author-selected code license; research-data handling; independent or
cross-platform installation and usability evidence; rights and author facts; and a
future live portal draft.

## Current downstream order constraint

The reviewer form independently confirms the current order constraint: empirical impact,
install/reproduction, testing, portability, documentation and licensing must be proven
before UI, release and manuscript work. A manuscript-quality improvement cannot close a
software-quality or scientific-impact gate.
