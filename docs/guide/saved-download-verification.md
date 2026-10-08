# Checking files saved by the browser

Use the browser's normal download behavior. A successful page handoff is not proof that
the browser or an installed download manager saved the file. Check the completed downloads
list and the actual save folder. Do not change security settings to force a download.

## Generate and download

1. Create, validate and run a job. Wait for a completed result.
2. Generate the HTML report and export the evidence bundle.
3. Click Download HTML report, Download result record and Download evidence ZIP.
4. Confirm all three files have finished downloading. Keep their job/operation-based
   filenames; use a dedicated folder containing the files being checked.

The ZIP contains eleven files, including `summary.csv`, `candidates.csv` and
`replicates.csv`. Extract to a new folder. Open `summary.csv` with a spreadsheet application
to see the result overview, `candidates.csv` for candidate statistics and `replicates.csv`
for repeat-level details. The CSV columns and full precision are authoritative projections
of the checked saved content; display rounding in a spreadsheet is not a change to the data.
An empty field is not zero. The HTML's expanded technical records remain available for
the full recorded identities and diagnostics. SelCal does not require Excel or launch it
automatically. Application-specific Excel behavior requires separate testing.

## Verify a stopped workspace

`scripts/check_saved_downloads.py` is an **offline workspace acceptance tool**. Finish all
downloads and normally stop the local helper before invoking it. On Windows, use Ctrl-C
in its terminal and wait for the process to exit; closing a browser tab does not stop the
helper. Do not delete `writer.lock`: the file can remain after a normal stop, and its
presence alone does not mean a writer is active. Do not run this acceptance tool while
any process is writing to the source workspace.

From the same installed SelCal environment, use a new output directory:

```console
python scripts/check_saved_downloads.py --python /path/to/environment/python --workspace /path/to/stopped-workspace --saved /path/to/downloads --out /path/to/new-evidence --count 3
```

Use the same installed environment's Python executable for `--python`, and equivalent
absolute paths on Windows. The tool takes a private snapshot for checking;
it must not change the source workspace or the downloaded originals. Required files that
cannot be read must cause failure, not be skipped. A baseline that could not be established
is NOT_CHECKED, not proof that the inputs stayed unchanged. Read the overall status, the
first error and the per-file checks; an early workspace failure is not a finding that the
three downloads were corrupt.

The check validates filename/job/operation binding, actual bytes and digests, and the
saved contents. SQLite replay uses the current implementation; neither replay nor an
HTML/ZIP consistency check authenticates historical execution. Keep the receipt and
the exact downloaded files when reporting a problem.
