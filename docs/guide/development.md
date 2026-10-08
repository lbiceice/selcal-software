# SelCal development and platform record

Installation details for developers, the locked environment, continuous integration,
containers, the API reference build, performance measurement, and the record of platform test
rounds. Users need only the [README](../../README.md).

## Platform test record

Requirements: Python 3.11 or newer (tested: 3.11-3.13) and NumPy (`numpy>=1.26,<2.5`). On macOS arm64
the wheel was installed into fresh environments on CPython 3.11, 3.12 and 3.13, and the sdist on
CPython 3.13; the documented workflow was run from outside the source folder; the three versions give the same lag,
correlation, p-value and decision, and a record is reproducible byte for byte, survives being moved
and is refused if one byte is altered. These are internal checks on earlier candidate
snapshots; their reports are not included in this product source package. The product
suite was also run on Linux aarch64 and x86_64 on earlier snapshots, which does not
establish the current candidate's cross-platform status.
The 2026-09-28 R5 return contains native Windows 11/AMD64 results from one host: Python 3.11,
3.12 and 3.13 with PowerShell 5.1, plus Python 3.12 with PowerShell 7, all using NumPy 2.4.6.
Each configuration recorded **3,466 passed / 20 skipped / 0 failed**; the four configurations
repeat the same product suite, not four disjoint sets of tests. Saved-record readback and
installed-package workflows passed in those returned logs, after the September 24 readback
failure. That R5 return covers the earlier 40-file product source; the subsequent
checkpoint kernel, store and recovery changes were not covered by that R5 return.
Of the 20 skips, 13 symlink-permission tests remain unverified on Windows; the other seven
concern platform-specific capabilities. Visible browser rendering, an independent user evaluation,
new machines without Python, and other filesystems remain unverified. These results do not
establish universal Windows compatibility. Original execution logs are retained internally,
not bundled into the product source package.

The 2026-10-03 R11 return, from one Windows 11 Pro AMD64 host (build 26200, ordinary user,
NTFS), ran the R11 candidate with Python 3.11, 3.12 and 3.13 under PowerShell 5.1 and Python
3.12 under PowerShell 7. Each configuration collected 4,457 tests: 4,388 passed, 67 skipped and
2 failed. Both failures were reference fixtures whose Pearson values differed by 1-2 units in the
last place between OpenBLAS and Apple Accelerate, with identical p-values, counts and decisions;
the correctly rounded sums described below were introduced after R11 to remove that difference.
Wheel and sdist installs on Python 3.12 had 423 passed, 21 skipped and 0 failed.

The 2026-10-04 R12 return, from the same host and the same four configurations, ran the R12
candidate: each configuration collected 4,470 tests, with 4,403
passed, 67 skipped and 0 failed or errored. The 67 skips are 40 symlink-permission tests
(WinError 1314), 11 FIFO, 7 other POSIX-specific, 5 question-mark file names, 2 unconfigured
browser tests, 1 extended float type and 1 non-Windows host check. Wheel and sdist installs on
Python 3.12 had 423 passed, 21 skipped and 0 failed, and the four reference cases matched the
saved results byte for byte. In the built-in Codex browser on that host, creating, validating,
running, reporting, previewing and resuming a job worked; the three file downloads were not
confirmed, and browser downloads in ordinary Chrome or Edge, offline use with the proxy off, other
Windows machines and machines without Python remain unverified.

The 2026-10-07 R19 v2 return from that Windows 11 Pro AMD64 host used CPython 3.12.0,
NumPy 2.4.6 and Windows PowerShell 5.1. The shipped full suite recorded **4,509 passed,
6 failed, 0 errors and 73 skipped** (4,588 unique test cases); the final automated gate
correctly failed even though its seven mandatory native nodes passed. The six failures
were traced to nested test executables with 260- or 279-character paths that Windows
could not start. A short-physical-path diagnostic passed the relevant eight tests;
that diagnostic did not replace or repair the original full-suite result.

The same return confirmed ordinary offline installation without upgrading its bundled
pip, installed-file identity, and three actual local in-app-browser workflows. Each of
those workflows saved HTML, SQLite and evidence ZIP downloads; downloaded SQLite replay,
HTML regeneration, ZIP integrity and independently recounted CSV summaries agreed.
A separate 499-repeat run was cancelled and resumed, replaying 24 retained repeats and
adding 475. A malformed JSON plan was refused. These are one host, one Python version
and one browser-engine result, not ordinary Chrome/Edge or macOS matrix acceptance.

The original saved-download acceptance script also failed when reading the live Windows
workspace's byte-locked runtime lock. This is distinct from downloaded-content failure.
The local repair candidate addresses test paths, offline saved-download checking and
packaging documentation; its new results must be recorded separately after execution.
See [saved-download verification](saved-download-verification.md) for the stopped-helper
contract. Offline use with the proxy off, native console Ctrl-C, ordinary Chrome/Edge,
desktop Excel and other machines still require their own execution evidence.

## Project status

Version 0.1.0 is a candidate for the first tagged release. The historical internal milestone label
`M0-M2 IMPLEMENTATION CANDIDATE / FINAL VERIFICATION PENDING` is not a current test-count report.
The R5 and R11 native Windows results above cover their own candidates and environments only.
The private full-checkout suite also includes research, manuscript and governance tests; its
denominator is not the product source-package denominator. Final candidate packaging, the
remaining Windows checks, an independent user evaluation and public release are separate gates.

## Install from source

SelCal currently requires Python 3.11 or newer. From an activated, isolated Python
environment at the repository root, install the exact checked-out snapshot with:

```console
python -m pip install .
```

This installs the checked-out candidate snapshot from source. SelCal 0.1.0 is not published on PyPI.

### Reproduce the locked development environment

The checked-in `uv.lock` resolves the runtime, development, documentation, and benchmark
dependency surfaces declared in `pyproject.toml`. With `uv` installed, create or update an
exact project environment from that lock without changing it:

```console
uv sync --locked --all-extras
```

Verify that `pyproject.toml` and the lock remain synchronized before running release-facing
checks:

```console
uv lock --check
```

Run the test suite through the locked project environment with:

```console
uv run --locked --all-extras python -m pytest
```

The lock makes dependency resolution repeatable for the declared Python dependency range;
it does not prove that every interpreter or operating-system combination in that range is
supported. It is not a cross-platform test result.

For strict development type checking, run `python -m mypy` from the repository root
with the development dependencies installed. The configuration checks `src/selcal`,
not an installed wheel. It can also be checked explicitly with
`python -m mypy --platform win32 src/selcal` (or `darwin` / `linux`). These are static
checks, not execution on those operating systems; this release does not promise a
downstream `py.typed` package interface.

### Separate verifier memory measurement from coverage

The following are targeted verifier checks, not a whole-product coverage result.
From an activated project environment at the repository root, run the resource
measurement without coverage, profiling, a debugger, or other active tracing:

```console
python -m pytest tests/test_verifier_integration_v2.py -k test_public_verifier_scratch_peak
```

Run coverage separately, excluding only the two resource-measurement tests from
this verifier file:

```console
python -m pytest tests/test_verifier_integration_v2.py --cov=src -k 'not test_public_verifier_scratch_peak'
```

Both resource tests remain in the default unfiltered suite. Coverage can start
tracing in child processes; it must not be combined with the memory-measurement
lane. The worker fails with `measurement environment is dirty` when a trace or
profile callback is active, or tracemalloc is already running. It does not clear
instrumentation or raise the frozen memory limits to make a measurement pass.
The check detects those CPython trace/profile/tracemalloc states, not every form
of external instrumentation. These tests measure Python-tracked allocations, not process RSS or native allocations,
and do not establish asymptotic memory bounds or scientific validity. Coverage
results do not substitute for a clean resource measurement.

### Artifact-first CI

With CPython 3.11, 3.12 or 3.13, install the build frontend and run from the
source root, choosing a **new directory outside the checkout**:

```console
python -m pip install "build>=1.2,<2"
python scripts/ci_check.py /outside/checkout/new-ci-output --numpy-version 2.4.6
```

On Windows, use an equivalent absolute output path, such as `C:\SelCalChecks\new-ci-output`.
This developer route uses ordinary network dependency installation in fresh environments;
it does not use the separate offline Windows end-user handoff or require Git metadata.
The test environment installs the project's declared `dev`, `docs` and `benchmark`
extras so the complete shipped suite, including its plotting tests, can run.
No preinstalled uv is needed: the runner installs and records `uv==0.7.6` privately in
the test environment for the lock check, placing its tools first on PATH.
The two runtime-only environments do not receive uv. The pressure test runs with the test
environment's own Python, using psutil 7.0.0 from the `dev` extra and that lane's NumPy.
It builds the current working tree, checks complete product inventories, tests the extracted
source package, then separately installs the wheel and source package with runtime dependencies
and runs the shipped examples and ordinary installed acceptance checks.

`evidence/receipt.json` records actual outcomes, environment identities and skips; command logs,
JUnit and installation checks remain in `evidence/`, distributions in `artifacts/`, and temporary
environments in `work/`. Failed steps retain their output and return nonzero. Existing destinations
are refused. The workflow uploads only evidence and artifacts, not work environments or private
research documents. It runs only when started by hand (workflow_dispatch) and runs no matrix job
while the repository is private. Its declared matrix covers Linux/macOS/Windows and the three Python versions
with NumPy 2.4.6, plus Linux/Python 3.11 with NumPy 1.26.4. A workflow definition or local successful
run does not establish that hosted CI or native Windows has run. This ordinary artifact acceptance
does not close the separate full-engineering, research-value or release gates.

The F5A artifact-first CI route passed local acceptance on 2026-09-30. That local
run exercised the built wheel and extracted source package. Hosted CI remained unverified
in that return; the later R19 v2 native Windows result is recorded separately above and
does not establish that the hosted artifact-first matrix ran.

## Run with containers

Docker with the Compose plugin can build and run this source snapshot without
host Python. Run the following from the extracted source package or checkout.
The first build needs network access (or an already populated build cache).
The image builds an installed wheel using pinned build tools, the pinned official
Python 3.12 image and NumPy 2.4.6. Those pins do not promise bit-identical image builds.
The build context includes only product code, declared packaged documentation,
licences and the bundled example; private research and local outputs are excluded.

Create the output directory before starting. Both input and output bind mounts
refuse missing host directories. On Linux, use your ordinary non-root account's
numeric IDs so newly written files belong to you:

```console
mkdir -p selcal-container-output
export SELCAL_UID="$(id -u)"
export SELCAL_GID="$(id -g)"
docker compose build
docker compose run --rm selcal doctor --folder /output
docker compose run --rm selcal validate /input/series.csv /input/pearson.json
docker compose run --rm selcal run /input/series.csv /input/pearson.json /output/run.sqlite --max-bytes 8388608
docker compose run --rm selcal verify /output/run.sqlite --max-bytes 8388608 --replay
docker compose run --rm selcal report /output/run.sqlite /output/report.html --max-bytes 8388608
docker compose run --rm selcal export /output/run.sqlite /output/evidence --max-bytes 8388608
docker compose run --rm selcal verify-export /output/evidence --max-bytes 8388608
```

The examples use 8 MiB (8,388,608 bytes), matching the UI and recovery examples.
This accommodates the complete eleven-file evidence bundle as well as the example's
record and checkpoint. The cap applies to stored payload, database, report and
aggregate bundle bytes according to each operation; it is not a process-memory bound.

The default image user is 1000:1000; the two ID variables override that for bind
mount ownership. Do not select UID 0 or change ownership of existing data recursively.
Open `selcal-container-output/report.html` on the host. The fixed synthetic example
selects lag 2 with p = 0.015; this is a runnable example, not scientific validation.
Each command uses a fresh container, while records, reports and evidence stay in
the host output directory after container removal. Choose fresh output filenames
for another run; existing records and bundle destinations are refused.

Set `SELCAL_INPUT_DIR` and `SELCAL_OUTPUT_DIR` to existing host directories to use
your own files. `/input` is read-only, `/output` is writable, and the root filesystem
is read-only with temporary `/tmp` storage. The CLI service is networkless, drops
all Linux capabilities and does not mount the Docker socket or your home directory.
It runs the installed program with isolated Python; no host source mount is needed.
The bundled files are also available inside the image at `/opt/selcal/examples`.

For local recovery, choose a new checkpoint directory and a new record filename:

```console
docker compose run --rm selcal run /input/series.csv /input/pearson.json /output/checkpoint-run.sqlite --max-bytes 8388608 --checkpoint /output/run-checkpoint
docker compose run --rm selcal resume /output/run-checkpoint /output/recovered.sqlite --max-bytes 8388608
```

The small example usually finishes before it can be interrupted. For a longer
run, Ctrl-C requests an orderly stop; the image and Compose use SIGINT, with
30 seconds of Compose stop grace before forced termination. Initialization might
stop before a valid checkpoint exists. Resume in a new container using the same image,
original budget and retained output directory. The original code, Python, NumPy and
platform identity must match; a macOS host checkpoint is not promised to resume in
a Linux container. Resume replays the retained prefix before continuing and does
not promise saved computation or recovery from every forced kill.

To open the existing local interface, use the override for the same single service:

```console
docker compose -f compose.yaml -f compose.ui.yaml up
```

Open the printed `127.0.0.1` URL including its session token fragment in your host
browser. Keep the terminal open; Ctrl-C stops the helper. Restart with the same
command and open the newly printed URL to use saved jobs under
`selcal-container-output/ui`. Set `SELCAL_UI_PORT` if port 8765 is occupied. The
interface retains its loopback-only listener and strict Host, Origin and token
checks. There is no published port or separate frontend service.

The UI override requires host networking: native Linux Engine supports this route,
and OrbStack provides a macOS route. It shares host connectivity, so use it only for
this trusted local application. Docker Desktop host networking requires explicit
opt-in and is untested here. Local execution was checked on 2026-09-30 using
OrbStack's linux/arm64 engine, Python 3.12.14, NumPy 2.4.6 and host Chrome:
build, networkless commands, same-image interruption/recovery, saved-job reopening
and actual report/evidence downloads. These are bounded execution checks, not a
human usability study. Other engines and architectures remain untested.
No new Windows, hosted CI, public-release or scientific-validity claim follows
from this route.

## Build the API reference

Install the documentation dependency group from the repository root:

```console
python -m pip install ".[docs]"
```

Build the HTML reference with warnings treated as errors:

```console
python -m sphinx --fail-on-warning --keep-going -b html docs/api docs/api/_build/html
```

Open `docs/api/_build/html/index.html` in a browser after the command succeeds. The
generated reference describes the supported API of this candidate snapshot; it is not evidence of
scientific validity.

## Characterize in-memory performance

Run the bounded standard benchmark from an isolated development installation:

```console
python scripts/benchmark_in_memory.py --preset standard > benchmark.json
```

The JSON records the benchmark contract hash, environment identity, B/C/N case matrix,
repeat-level wall times, result statuses, scientific-plan hashes, and maximum Python
allocation peaks reported by `tracemalloc`. Input construction, process startup, native
allocations, parallel scaling, GPU execution, and cross-machine comparisons are outside the
measurement boundary. This is single-process characterization, not a cross-platform performance guarantee or M6 scientific evidence.
