# Exact-enumeration example

`examples/workflow/pearson_exact.json` runs the shipped `examples/workflow/series.csv` with the
exact circular null (`circular_shift_exact_v1`): all 99 non-identity shifts of the 100-sample
source are enumerated once. It is the example of Section 3.1 of the SelCal software paper and has
shipped with the package since 0.1.2; it also runs unchanged with 0.1.1. Install the release
wheel in an isolated environment first; all paths below are relative to the repository root.

```console
python -m selcal validate examples/workflow/series.csv examples/workflow/pearson_exact.json
python -m selcal run examples/workflow/series.csv examples/workflow/pearson_exact.json exact.sqlite --max-bytes 1048576
python -m selcal verify exact.sqlite --max-bytes 1048576 --replay
python -m selcal report exact.sqlite exact.html --max-bytes 1048576
```

Expected: COMPLETE, selected lag 2, E = 2 of 99, p = 0.03; same-environment replay MATCH.
The separate `pearson.json` example uses 199 sampled shifts and gives p = 0.015; do not
substitute its p-value for the enumerated example.

Here L/n = 3/100 is a deterministic lower bound for the enumerated reference p-value.
Error-rate control additionally needs the joint distribution of (X, Y) to be invariant under
source-only circular shifts with the target held fixed. Stationarity or similar endpoints do not
establish that assumption, and passing preflight does not verify it. Release 0.1.1 printed a
weaker advisory; 0.1.2 states this condition in `validate`, the report and the interface.

For a clean-directory check, copy just `series.csv` and `pearson_exact.json` into an empty
directory and use those two basenames in the commands above. No other input files are required.
Keep each SQLite record with the runtime that created it; source digests and report projections
intentionally detect runtime changes.
