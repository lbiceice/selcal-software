# Changelog

All notable changes to SelCal are listed here. Versions follow [Semantic Versioning](https://semver.org/);
the format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Unreleased

### Fixed

- `scripts/verify_installed_identity.py` read installed files through its case-folded comparison
  key; under the simulated-Windows test on a case-sensitive file system (hosted Linux runners) it
  reported wheel files as missing. Files are now read at their real paths and the folded key is used
  only for comparison. Real installations verified correctly before and after.
- `scripts/acceptance_check.py` stopped with an encoding error when the console code page could not
  encode a character of the output path (hosted Windows runners, cp1252); unencodable characters are
  now written as escapes.
- Two installer-launcher tests assumed the current pip's launcher layout; they now accept the CRLF that
  pip <= 24.0 writes on Windows and skip the text-launcher case there.

## 0.1.0 - 2026-10-09

First public version: tag `v0.1.0` of <https://github.com/lbiceice/selcal-software>, with the
Windows and macOS user packages attached to the GitHub release and the source archived on Zenodo
(DOI in `CITATION.cff`).

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
  NumPy versions. Its information is computed from the exact integer counts with 50-digit decimal
  logarithms and rounded once, so NetTE values are the same on every platform (np.log and np.sum
  differed in the last bit between NumPy versions and CPUs). The preprocessing identity is renamed.
  All of this decimal arithmetic runs in one private, fully specified context, so a caller's decimal
  precision, rounding or traps (in any thread) no longer change the result: before, the bundled
  binned case gave E = 4, 2, 3, 3 at caller precision 2, 6, 28, 50 instead of the exact 3.
- Outputs refuse a name occupied by a dangling symbolic link (Windows exclusive creation followed it).
- The local interface holds a lock on its workspace: a second helper on the same workspace is refused
  instead of marking the first helper's running jobs as interrupted.
- A job folder that cannot be read (for example after an interrupted job creation) is left untouched
  and reported; it no longer stops the workspace and its healthy jobs from opening.
- Interface child processes run in their own operation folder, so a module named selcal in the folder
  where the helper was started cannot replace the installed package.
- `scripts/verify_installed_identity.py` checks an installation against the bytes of the wheel it
  should come from (data files at their installed location; pip's bytecode, entry-point launchers
  (the exact text pip 22.3-26.2.1 writes, and on Windows the launcher stub, interpreter line and
  ZIP layout, including the extra CRLF that pip 22.3-24.0 put before the ZIP) and installer metadata each checked by kind; extra files in the package folders
  refused), or,
  without a wheel, against its RECORD hashes. It previously compared the two RECORD files, which
  rejected every normal pip installation and missed an edit made together with RECORD. The Windows
  installer runs it before writing READY.json.
- The Windows runner no longer states that disabled long paths are the cause of a failure; it lists
  the order in which to diagnose one.

### Fixed after the R16 Windows test (2026-10-06)

- The installed-identity check classified files by the lowercased path key Windows uses for
  comparison, so a normal installation was refused for its own `INSTALLER` and `REQUESTED` files;
  files are now classified by their real names.
- Interface child processes keep the helper's working directory and start with `-P` (or `-I`), so
  the working directory is not searched for modules; starting them in a deep operation folder
  failed on Windows, where a process's working directory is limited to about 258 characters.
- When a download manager such as IDM takes over a download (HTTP 204), the page says so and that
  the saved state is unknown, with the path of the checked original, instead of reporting a failure.
- When another window's operation is running, the page disables starting actions and explains a
  refused request (HTTP 409) instead of reporting a failure of the current job.
- Test and tool fixes for the Chinese Windows environment: the pressure test runs in the locked test
  environment (psutil 7.0.0 added to the `dev` extra) instead of a uv overlay whose `.pth` file
  could not be read under the cp936 code page; child-process output that tests parse is
  ASCII-escaped JSON; the native-test gate reports call failures with teardown errors as such,
  not as a duplicate testcase; interface tests wait while progress advances instead of a fixed
  deadline, and stop running jobs through the interface before ending the helper.
- The documentation states that the exceedance threshold is computed in binary64 and differs from
  `tie_tolerance`, and gives the NetTE time indices with a worked example.

### Fixed after the R17 Windows test (2026-10-07)

- The acceptance and research-case tools start every child command in a short, new, empty folder
  and pass files as absolute paths. Windows cannot start a process whose working directory is
  longer than about 258 characters, so deep result folders failed with WinError 267. A command
  that cannot start is now recorded (`started=false`) and counted as a failed check.
- The local interface follows the job it shows when another window or tab runs it: the state and
  the starting actions update without reselecting the job.
- The Windows install, example and acceptance scripts keep each program's output as raw bytes and
  decode it line by line (UTF-8, otherwise the system ANSI code page) instead of through the console
  code page, which had replaced Chinese folder names in the logs. The console code page and the
  programs' environment are not changed.
- The final Windows acceptance receipt reads `RUNNING` (with the current step) until the run ends,
  then `PASS_AUTOMATED_ONLY` or `FAIL`; a run stopped early reads `INCOMPLETE`.

### Fixed after the R18 Windows test (2026-10-07)

- Tests that need repository-only files are skipped one by one, so every test keeps its name in the
  JUnit report; a whole-module skip had produced a record without a name, which the native Windows
  evidence check rightly refuses.
- Native output is decoded with the encoding its producer states (Python reports its own, for the
  same executable and flags), not guessed: cp936 bytes such as C3 A6 are also valid UTF-8, so the
  guess could silently show another character. Lines that are not valid in the stated encoding are
  marked; the exact bytes are kept.
- The precision acceptance check compares records only when all five runs exited 0 and left a
  readable record, marks the relocation and tamper checks as failed when the first record is
  missing, and ends with a failure summary and a nonzero exit after any unexpected error. Child
  commands start in the system temporary folder, or at the root of the output drive when that
  folder's path is very long.

### Fixed after the R22 Windows test (2026-10-09)

- Local interface: a browser that refuses session storage no longer aborts the page script before
  the buttons are wired; a token in the launch URL is used from memory in that case. The first step
  shows a connection line ("Connected" only after a real answer), separate messages for a missing
  session, a refused request (403) and an unreachable helper, a read-only **Retry connection (keep
  input)** button, the reason the create button is disabled, and a no-JavaScript notice. A
  complete launch URL pasted into a tab that already shows the page (only the fragment changes,
  so the page does not reload) is taken over and used to reconnect, keeping a chosen input. The
  three web resources changed; no Python product file did, so the software identity and the
  reference records are unchanged.
- `scripts/check_saved_downloads.py`: an archive whose required ZIP version is unsupported now
  yields a structured FAIL receipt instead of an uncaught `NotImplementedError`; unrelated errors
  still propagate.
- Windows installer: without `-Python`, the first installed standard 64-bit CPython 3.13, 3.12 or
  3.11 is used and every attempt is logged; free-threaded, ARM64 and non-CPython builds are refused
  before pip runs with the reason; a machine without Python is pointed to one fixed official
  download. `READY.json` records the exact interpreter version and path. The macOS installer gains
  the same free-threaded refusal and link.
- User guides: install once and start directly afterwards; the complete launch URL rule when
  switching browsers; the connection line and retry; default downloads; never delete
  `writer.lock`. README and the development guide describe the user packages consistently.
- Both user packages now ship `check_saved_downloads.py`, and the guides give the exact command
  to check the three files the browser saved (stopped helper, an empty folder with the three
  files, `--count 3`): byte equality with the workspace original, record replay, ZIP content and
  a regenerated-report comparison; exit 0 and `"status": "PASS"` mean all three are correct.

### Fixed after the R21 Windows test (2026-10-09)

- The saved-download checker also reports a ZIP whose central directory reads but whose member data
  does not (CRC or decompression error, encrypted, patched-data or unsupported compression) as a
  failed file in its receipt, and reads every member within the 32 MiB cap before it creates the
  output folder, so a failed archive leaves no partial extraction.
- The test that covered file-browser metadata in an evidence bundle is split so that the regular-file
  part no longer skips with the file-symlink part on a Windows account without symlink privilege; a
  native junction case was added. Product files are unchanged.
- The test session refuses to start on POSIX when SIGINT is ignored in the pytest process (a
  background job of a non-interactive shell inherits that), because the real-helper cancel tests
  send SIGINT and would otherwise fail two seconds late with a misleading message.
- macOS user entry points (`INSTALL_MACOS.sh`, `RUN_EXAMPLE.sh`, a one-page guide) mirror the
  Windows ones: offline installation from verified wheels into `.selcal-user/venv`, installed files
  compared with the wheel, `READY.json` bound to the delivery manifest, and the two examples
  checked for lag 2, p = 0.015 and 0.03 with replay MATCH. Tested on CPython 3.11, 3.12 and 3.13
  (arm64).

### Fixed after the R19 v3 Windows acceptance review (2026-10-08)

- A `.DS_Store`, `desktop.ini` or `Thumbs.db` file that a file browser writes into the workspace, its
  `jobs/` folder or an evidence bundle no longer makes the interface refuse the whole workspace or the
  bundle. Only those three names, as regular non-link files, are ignored; every other unexpected
  member is still refused, and links under those names are still refused.
- The saved-download checker reports a partial or corrupt evidence ZIP as a failed file in its receipt
  instead of stopping with a traceback.
- Records written by earlier development snapshots are refused by the software identity check, as
  documented; the 64 scaled-ULP tie rule of this version is not applied to them retroactively.

### Documentation and metadata

- The README is a short user page; the complete user reference and the development and platform
  record are in `docs/guide/`, with result-interpretation and interface guides.
- `codemeta.json` added. The author confirmed on 2026-10-05 the right to release the source under
  BSD-3-Clause (`docs/provenance/SOURCE_ORIGIN.tsv`).
- Source repository: https://github.com/lbiceice/selcal-software
