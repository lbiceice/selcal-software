# Interpreting SelCal results

This guide answers the questions a user has after running SelCal: what the numbers mean, why a
plan was refused, and what a result does and does not show. The precise rules are in the README
sections "Run a file-to-report workflow" and "Save and read complete result content".

## What question does SelCal answer?

You scanned a small, pre-declared set of lags between a source series and a target series and
picked the lag with the strongest dependence (by default the largest absolute lagged Pearson
correlation). SelCal asks: **is a maximum this strong unusual if the two series were unrelated
apart from their own autocorrelation?** It answers by circularly shifting the source series,
repeating the *whole* lag search on every shifted copy, and comparing the observed maximum with
the maxima of the shifted copies.

Two common shortcuts answer a different question and give p-values that are too small:

| Shortcut | What goes wrong | What SelCal does instead |
|---|---|---|
| Report the standard p-value of the winning lag | The winner was chosen as the best of L lags, so its p-value ignores the search | Repeats the full lag search inside every surrogate |
| Shuffle samples, or use the textbook correlation test | Autocorrelated series produce large correlations by chance; shuffling destroys autocorrelation | Circular shifts keep each series' own autocorrelation |

## Reading the main numbers

| Field | Meaning |
|---|---|
| `selected_candidate` | The lag with the strongest dependence in the observed data. Ties are listed in `tied_candidates`. The test is about the whole scan, not a confidence statement that this particular lag is the true one. |
| `decision_statistic` | The observed maximum (for Pearson with `max_absolute`, the largest absolute correlation over the searched lags). |
| `exceedance_count` (E), `planned_replicates` (B) | E of the B surrogates reached or exceeded the observed maximum. |
| `p_value` | `(1 + E) / (B + 1)`: the share of null states, counting the observed alignment itself, whose maximum reaches the observed one. A surrogate value within 64 units in the last place of max(\|observed\|, 1) counts as reaching it, so values that are equal in exact arithmetic but rounded differently are treated as the ties they are. The threshold `observed − 64 × ulp(max(\|observed\|, 1))` is computed in binary64 floating point and rounded once, and a value at the rounded threshold counts. This fixed tolerance is unrelated to the plan's `tie_tolerance`, which only decides which candidate lags tie when the maximum is selected. |
| `reject_null` | `p_value <= alpha`. |
| `attainability` (from `validate`) | `null_state_p_floor`: the smallest exact p the plan can reach (L/n for the full circular Pearson scan). `monte_carlo_p_floor`: `1/(B+1)`. `monte_carlo_power_cap`: the most often a sampled plan can reject, even for a very strong signal. |

Example (`examples/workflow`): 100 samples, lags 1–3, 199 sampled circular shifts. The result is
lag 2, E = 2, p = 3/200 = 0.015, rejected at alpha = 0.05. The exact version of the same plan
(all 99 non-identity shifts) gives p = 0.03.

## Why was my plan refused?

`validate` names the reason as a status; `run` stops with exit 2 and an error code.

| `validate` status | `run` error | Reason | What to do |
|---|---|---|---|
| `REFUSE_NULL_STATES_TOO_FEW` | `unattainable_plan` | With L contiguous lags and n samples, L of the n circular states always tie or beat the observed maximum, so p cannot fall below L/n. If L/n > alpha, the plan can never reject. | Use a longer series or fewer lags: you need n ≥ L/alpha (for 5 lags at alpha = 0.05, n ≥ 100). Decide this before looking at the data. |
| `REFUSE_REPLICATES_TOO_FEW` | `unattainable_plan` | 1/(B+1) > alpha: too few surrogates for any p to reach alpha. | Use B ≥ 1/alpha − 1 (B = 199 or 999 are common). |
| `REFUSE_ENUMERATION_REPLICATE_COUNT` | `unattainable_plan` | Exact enumeration needs B = n − 1, one surrogate per non-identity shift. | Set B = n − 1, or use `circular_shift_v2`. |
| `REFUSE_NON_GROUP_NULL` | `invalid_null_for_inference` | `min_shift` > 1 removes shifts, and the remaining shifts are no longer a group, so the test has no level guarantee. | Use `min_shift = 1`. |
| `within_caps: false` in `resource_budget` | `resource_budget_exceeded` | B, the number of lags and n exceed the in-memory caps (B ≤ 1000, B × lags ≤ 5000; the message lists every cap). | Reduce B or the number of lags. |

Numbers in the plan keep their JSON type: write `alpha` as a decimal such as `0.05`, and lags,
`replicates` and `root_seed` as integers.

`--allow-unattainable-plan` runs a plan that cannot reach alpha (for example to document a
published design); it never lifts the `min_shift` refusal.

Example: the bundled 100-sample series with lags 1–10 gives `REFUSE_NULL_STATES_TOO_FEW`
(`null_state_p_floor` = 10/100 = 0.1 > 0.05).

## Exact or sampled null?

| Choose | When | Result |
|---|---|---|
| `circular_shift_exact_v1` | n is small or moderate (B = n − 1 surrogates is affordable) | The exact p over all circular states; no Monte Carlo error |
| `circular_shift_v2` | Long series, where all n − 1 shifts would be too many | A sampled p on a grid of step `1/(B+1)`; the step is the resolution, not an error bar |

## What a result does not show

- **Not significant is not "no association".** It means the observed maximum was not unusual
  under the circular-shift null with this n, lag set and alpha. Check `monte_carlo_power_cap` and
  the length of the series before concluding anything about absence.
- **Significant is not causation**, and SelCal does not estimate an effect size.
- **The null is circular-shift exchangeability.** It is exact for circular series and approximate
  otherwise. Remove trends and seasonality first: with seasonal data, shifts near multiples of the
  period keep the seasonal phase and make the test less informative.
- **One pair of series.** SelCal does not correct across many pairs, align or resample series, or
  impute missing values.
- **The lag set must be fixed before seeing the data.** If lags were chosen after looking, the
  p-value does not cover that extra choice.

## Status and exit codes

| Exit | Outcome | Meaning |
|---|---|---|
| 0 | complete | The analysis finished; read the numbers above. |
| 1 | internal error | An unexpected error inside SelCal (a defect, not a data or plan problem). The traceback is printed on purpose; please report it with the command and the files. |
| 2 | refused | Invalid or unattainable request (see the refusal table); nothing was computed. |
| 4 | operational or integrity failure | A file, environment or record check failed; keep the files and the message. |
| 7 | NOT_EVALUABLE | The analysis ran but could not be evaluated, for example a correlation was undefined because a series is constant over the compared window, or its values overflow. The `low`/`high` bounds are the p-values the failed surrogates could reach, not a confidence interval. |
| 130 | interrupted | Ctrl-C or a stop request was observed; no result is claimed. Resume from a checkpoint if one was requested. |

## Checking a saved result

| Command | What it establishes |
|---|---|
| `selcal verify RECORD` | The record's input, plan and result are mutually consistent. |
| `selcal verify RECORD --replay` | Recomputes the whole calibration and compares bytes; needs the recorded code, Python, NumPy and platform. |
| `selcal verify RECORD --replay-decision` | Recomputes on another platform with the recorded code and NumPy; seeds, surrogate states, selections, E, p and the decision must match exactly, statistic values within 64 scaled ULPs. |
| `selcal verify-export FOLDER` | The exported files agree with each other; it does not recompute. |

None of these authenticates when or by whom the original analysis was run.
