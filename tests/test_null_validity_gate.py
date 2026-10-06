"""Restricted circular shifts are refused for inference (R10v3 Windows core audit, ALG-02).

With p = (1 + E) / (B + 1) a randomization test keeps its level when the reference states form
a group acting on the data (Hemerik and Goeman 2018). Circular shifts with min_shift = 1 are the
whole cyclic group; for 1 < min_shift < n/2 the shifts are not a group, for one lag or several and
for any statistic. Every min_shift > 1 is refused as a policy (min_shift = n/2 gives {0, n/2}).
The audit's counterexample is recomputed here from first principles with exact arithmetic.
"""

from __future__ import annotations

import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from selcal.contracts_v2 import PlanRequestV2
from selcal.workflow import null_validity, preflight, run_files, validate_files
from selcal.workflow_config import WorkflowConfigError

CAP = 1_048_576
X = [
    -2,
    94,
    7,
    -90,
    -34,
    30,
    24,
    3,
    100,
    -23,
    22,
    -9,
    49,
    -45,
    29,
    -65,
    -28,
    -65,
    93,
    -76,
    58,
    -36,
    36,
    80,
]
Y = [
    0,
    78,
    92,
    101,
    -83,
    -124,
    -4,
    54,
    27,
    103,
    77,
    -1,
    13,
    40,
    4,
    -16,
    -36,
    -93,
    -93,
    28,
    17,
    -18,
    22,
    0,
]


def _corr_key(x: list[int], y: list[int]) -> Fraction:
    """Monotone in the Pearson correlation for a fixed y: sign(cov) * cov^2 / var(x)."""
    n = len(x)
    mx, my = Fraction(sum(x), n), Fraction(sum(y), n)
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True))
    var = sum((a - mx) ** 2 for a in x)
    return (1 if cov >= 0 else -1) * cov * cov / var


def _statistic(phase: int) -> Fraction:
    rolled = X[-phase:] + X[:-phase] if phase else X[:]
    return _corr_key(rolled[:-1], Y[1:])  # lag 1, max_upper


def _exact_rejection_rate(shifts: list[int]) -> Fraction:
    """Share of the 24 equally likely null phases rejected at alpha = 0.05 by the exact
    enumeration p = #{s in S: T(H + s) >= T(H)} / |S| over the reference set S."""
    n = len(X)
    rejected = 0
    for phase in range(n):
        observed = _statistic(phase)
        exceed = sum(_statistic((phase + s) % n) >= observed for s in shifts)
        rejected += Fraction(exceed, len(shifts)) <= Fraction(1, 20)
    return Fraction(rejected, n)


def test_audit_counterexample_restricted_shifts_exceed_the_level_full_group_does_not():
    n = len(X)
    restricted = [0, *range(2, n - 1)]  # min_shift = 2: identity and 2..22
    assert len(restricted) == n - 2 * 2 + 2 == 22
    assert _exact_rejection_rate(restricted) == Fraction(2, 24)
    assert _exact_rejection_rate(list(range(n))) == Fraction(1, 24)


def _request(
    *, min_shift=1, candidates=(1,), statistic="lagged_pearson_v1", null=None, replicates=999
):
    null = null or "circular_shift_v2"
    params = {"block_length": 2} if null == "block_shuffle_v2" else {"min_shift": min_shift}
    return PlanRequestV2(
        candidates=candidates,
        statistic_name=statistic,
        statistic_params={"bins": 3} if statistic != "lagged_pearson_v1" else {},
        selection_rule="max_upper",
        null_name=null,
        null_params=params,
        replicates=(23 if null == "circular_shift_exact_v1" else replicates),
        alpha=0.05,
        tie_tolerance=0.0,
        root_seed=17,
    )


@pytest.mark.parametrize("statistic", ["lagged_pearson_v1", "equal_width_binned_nette_v1"])
@pytest.mark.parametrize("candidates", [(1,), (1, 2)])
@pytest.mark.parametrize("min_shift", [2, 3])
def test_preflight_refuses_restricted_shifts_for_every_statistic_and_lag_count(
    statistic, candidates, min_shift
):
    report = preflight(
        _request(min_shift=min_shift, candidates=candidates, statistic=statistic), 24
    )
    assert report["plan"]["status"] == "NOT_EXECUTABLE"
    assert report["plan"]["reasons"].count("REFUSE_NON_GROUP_NULL") == 1
    assert report["plan"]["null_validity"]["status"] == "REFUSE_NON_GROUP_NULL"


@pytest.mark.parametrize(
    ("kwargs", "status"),
    [
        ({"min_shift": 1}, "PASS"),
        ({"null": "circular_shift_exact_v1"}, "PASS"),
        ({"null": "block_shuffle_v2"}, "NOT_ASSESSED"),
    ],
)
def test_group_nulls_pass_and_unchecked_nulls_are_not_reported_as_valid(kwargs, status):
    assert null_validity(_request(**kwargs))["status"] == status


def _files(tmp_path: Path, min_shift: int) -> tuple[Path, Path]:
    data = tmp_path / "data.csv"
    rows = "".join(f"{a},{b}\n" for a, b in zip(X, Y, strict=True))
    data.write_text("x,y\n" + rows, encoding="utf-8")
    cfg = tmp_path / "config.json"
    template = json.loads(
        (Path(__file__).resolve().parents[1] / "examples/workflow/pearson.json").read_text(
            encoding="utf-8"
        )
    )
    template["plan"].update(
        candidates=[1],
        selection_rule="max_upper",
        null_name="circular_shift_v2",
        null_params={"min_shift": min_shift},
        replicates=999,
        alpha=0.05,
        root_seed=17,
    )
    cfg.write_text(json.dumps(template), encoding="utf-8")
    return data, cfg


def test_override_does_not_lift_the_validity_refusal(tmp_path):
    data, cfg = _files(tmp_path, 2)
    assert validate_files(data, cfg)["preflight"]["plan"]["status"] == "NOT_EXECUTABLE"
    with pytest.raises(WorkflowConfigError) as error:
        run_files(data, cfg, tmp_path / "out.sqlite", max_bytes=CAP, allow_unattainable=True)
    assert error.value.code == "invalid_null_for_inference"
    assert not (tmp_path / "out.sqlite").exists()


def test_cli_refuses_with_exit_2_even_with_override(tmp_path):
    data, cfg = _files(tmp_path, 2)
    out = tmp_path / "out.sqlite"
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "selcal",
            "run",
            str(data),
            str(cfg),
            str(out),
            "--max-bytes",
            str(CAP),
            "--allow-unattainable-plan",
        ],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=60,
    )
    assert run.returncode == 2, run.stdout + run.stderr
    assert json.loads(run.stdout)["error"] == "invalid_null_for_inference"
    assert not out.exists()


def test_full_group_audit_plan_still_runs(tmp_path):
    data, cfg = _files(tmp_path, 1)
    result = run_files(data, cfg, tmp_path / "out.sqlite", max_bytes=8 * CAP)
    assert result.p_value is not None and 0 < result.p_value <= 1


# R11 Windows return (R11-01): the same N24 plan was refused by the CLI but returned complete,
# p = 0.047, reject = true through the public Python API. The public entry now applies the
# same rule before any computation; the kernel stays unchanged for replay.
@pytest.mark.parametrize("statistic", ["lagged_pearson_v1", "equal_width_binned_nette_v1"])
@pytest.mark.parametrize("candidates", [(1,), (1, 2)])
@pytest.mark.parametrize("rule", ["max_upper", "max_absolute"])
def test_public_api_refuses_restricted_shifts(statistic, candidates, rule):
    import dataclasses

    import numpy as np

    import selcal
    from selcal import calibration_v2
    from selcal.contracts import SeriesPair

    request = dataclasses.replace(
        _request(min_shift=2, candidates=candidates, statistic=statistic), selection_rule=rule
    )
    resolution = selcal.resolve_plan_v2(request)
    pair = SeriesPair(
        source=np.asarray(X, dtype=np.float64), target=np.asarray(Y, dtype=np.float64)
    )
    with pytest.raises(selcal.InvalidNullForInferenceError) as error:
        selcal.calibrate_selected_family(pair, resolution)
    assert error.value.code == "invalid_null_for_inference"
    assert selcal.calibrate_selected_family is not calibration_v2.calibrate_selected_family


def test_public_api_runs_the_full_group_and_matches_the_kernel():
    import numpy as np

    import selcal
    from selcal import calibration_v2
    from selcal.contracts import SeriesPair

    resolution = selcal.resolve_plan_v2(_request(min_shift=1, replicates=199))
    pair = SeriesPair(
        source=np.asarray(X, dtype=np.float64), target=np.asarray(Y, dtype=np.float64)
    )
    public = selcal.calibrate_selected_family(pair, resolution)
    kernel = calibration_v2.calibrate_selected_family(pair, resolution)
    assert public.status.value == "complete" and public.p_value == kernel.p_value
    assert public.exceedance_count == kernel.exceedance_count
