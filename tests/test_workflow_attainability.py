"""Plan-time attainable-p guard for the file workflow (2026-09-19 comparator finding).

Circular shifts s = c* - c map searched lag c onto a maximizing lag c*. The witness score is
reproduced exactly; a reselected maximum can be larger. Their share bounds the exact p-value.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from selcal import calibrate_selected_family, resolve_plan_v2
from selcal.contracts import SeriesPair
from selcal.contracts_v2 import PlanRequestV2
from selcal.nulls.circular_shift_exact_v1 import CircularShiftExactNullV1
from selcal.statistics.lagged_pearson import LaggedPearsonAdapter
from selcal.workflow import attainability, run_files, validate_files
from selcal.workflow_config import WorkflowConfigError

CAP = 1_048_576


def request(
    *,
    candidates=(1, 2),
    min_shift=1,
    replicates=199,
    alpha=0.05,
    null_name="circular_shift_v2",
    statistic_name="lagged_pearson_v1",
):
    null_params = (
        {"block_length": 2} if null_name == "block_shuffle_v2" else {"min_shift": min_shift}
    )
    return PlanRequestV2(
        candidates=candidates,
        statistic_name=statistic_name,
        statistic_params={},
        selection_rule="max_upper",
        null_name=null_name,
        null_params=null_params,
        replicates=replicates,
        alpha=alpha,
        tie_tolerance=0.0,
        root_seed=7,
    )


@pytest.mark.parametrize(
    ("lags", "status", "floor", "cap"),
    [
        ((1, 2), "PASS", 2 / 64, 0.904),
        ((1, 2, 3, 4), "REFUSE_NULL_STATES_TOO_FEW", 4 / 64, 0.198),
        (tuple(range(1, 9)), "REFUSE_NULL_STATES_TOO_FEW", 8 / 64, 0.0),
    ],
)
def test_floor_and_power_cap_match_the_audited_study_values(lags, status, floor, cap):
    report = attainability(request(candidates=lags), 64)
    assert report["status"] == status
    assert report["null_state_count"] == 64
    assert report["null_state_p_floor"] == pytest.approx(floor, abs=0)
    assert report["monte_carlo_power_cap"] == pytest.approx(cap, abs=5e-4)


def test_noncontiguous_candidates_collide_once_per_candidate():
    report = attainability(request(candidates=(1, 3)), 8)
    assert report["null_state_p_floor"] == 2 / 8
    assert report["status"] == "REFUSE_NULL_STATES_TOO_FEW"


def test_too_few_replicates_are_refused_even_with_enough_states():
    report = attainability(request(candidates=(1,), replicates=9), 400)
    assert report["monte_carlo_p_floor"] == 1 / 10
    assert report["status"] == "REFUSE_REPLICATES_TOO_FEW"


def test_min_shift_above_one_under_lag_search_is_refused_as_non_group():
    report = attainability(request(candidates=(1, 2, 3, 4), min_shift=3), 64)
    assert report["status"] == "REFUSE_NON_GROUP_NULL"
    assert report["null_state_p_floor"] == 1 / 60


def test_single_candidate_min_shift_above_one_is_also_non_group():
    """Changed 2026-10-03 (R10v3 Windows core audit, ALG-02): the group condition does not
    depend on lag search; one lag with min_shift = 2 rejected 2/24 phases of a cyclic null."""
    assert attainability(request(candidates=(1,), min_shift=3), 64)["status"] == (
        "REFUSE_NON_GROUP_NULL"
    )


def test_empty_shift_space_defers_to_the_calibration_terminal():
    report = attainability(request(candidates=(1,), min_shift=3), 5)
    assert report["status"] == "NOT_ASSESSED_EMPTY_NULL_STATE_SPACE"


def test_exact_enumeration_null_has_no_monte_carlo_loss():
    report = attainability(
        request(candidates=(1, 2), replicates=63, null_name="circular_shift_exact_v1"), 64
    )
    assert report["status"] == "PASS"
    assert report["monte_carlo_p_floor"] == 1 / 64
    assert report["null_state_p_floor"] == 2 / 64
    assert report["monte_carlo_power_cap"] == 1.0


def test_exact_enumeration_requires_the_complete_state_space_as_replicates():
    report = attainability(
        request(candidates=(1, 2), replicates=199, null_name="circular_shift_exact_v1"), 64
    )
    assert report["status"] == "REFUSE_ENUMERATION_REPLICATE_COUNT"


def test_unassessed_null_or_statistic_is_reported_not_refused():
    report = attainability(request(null_name="block_shuffle_v2"), 64)
    assert report["status"] == "NOT_ASSESSED"
    assert report["monte_carlo_p_floor"] == 1 / 200


def _profile_count(x, y, lags, observed_max):
    count = 0
    n = x.size
    top = max(lags)
    for s in range(n):
        shifted = x[(np.arange(n) - s) % n]
        value = max(np.corrcoef(shifted[top - c : n - c], y[top:])[0, 1] for c in lags)
        count += value >= observed_max
    return count


@pytest.mark.parametrize("lags", [(1, 2), (1, 3), (1, 2, 3, 4), (2, 5, 6)])
def test_reported_floor_is_a_true_lower_bound_on_random_data(lags):
    rng = np.random.default_rng(20260919)
    report = attainability(request(candidates=lags), 32)
    for _ in range(20):
        x, y = rng.standard_normal(32), rng.standard_normal(32)
        top = max(lags)
        observed = max(np.corrcoef(x[top - c : 32 - c], y[top:])[0, 1] for c in lags)
        assert _profile_count(x, y, lags, observed) / 32 >= report["null_state_p_floor"]


def test_production_mapped_states_can_exceed_observed_without_tying():
    rng = np.random.default_rng(1)
    n, candidates = 16, (1, 2, 3, 4)
    pair = SeriesPair(source=rng.normal(size=n), target=rng.normal(size=n))
    adapter = LaggedPearsonAdapter().bind(pair, candidates)
    observed = adapter.evaluate(pair, candidates, "max_upper")
    winner = max(observed, key=lambda value: value.selection_score)
    null = CircularShiftExactNullV1(min_shift=1).bind(
        pair, semantic_input_sha256="8" * 64, scientific_plan_sha256="d" * 64
    ).bound
    maxima = []
    for shift in range(n):
        token = null.identity_token() if shift == 0 else null.enumerate_token(shift - 1)
        values = adapter.evaluate(null.apply(token).pair, candidates, "max_upper")
        maxima.append(max(value.selection_score for value in values))
    mapped = [maxima[(winner.candidate_id - lag) % n] for lag in candidates]
    assert all(value >= winner.selection_score for value in mapped)
    assert any(value > winner.selection_score for value in mapped)
    count = sum(value >= winner.selection_score for value in maxima)
    plan = request(candidates=candidates, null_name="circular_shift_exact_v1", replicates=n - 1)
    result = calibrate_selected_family(pair, resolve_plan_v2(plan))
    assert result.p_value == count / n == 0.875
    assert result.p_value >= attainability(plan, n)["null_state_p_floor"] == len(candidates) / n


def _write(tmp_path: Path, n: int, lags: list[int], replicates: int) -> tuple[Path, Path]:
    rng = np.random.default_rng(3)
    data = tmp_path / "data.csv"
    x, y = rng.standard_normal(n), rng.standard_normal(n)
    data.write_text(
        "x,y\n" + "".join(f"{float(a)!r},{float(b)!r}\n" for a, b in zip(x, y, strict=True)),
        newline="\n", encoding="utf-8"
    )
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "schema": "selcal.workflow-config.v1",
                "input": {"format": "csv", "source_column": "x", "target_column": "y"},
                "plan": {
                    "candidates": lags,
                    "statistic_name": "lagged_pearson_v1",
                    "statistic_params": {},
                    "selection_rule": "max_upper",
                    "null_name": "circular_shift_v2",
                    "null_params": {"min_shift": 1},
                    "replicates": replicates,
                    "alpha": 0.05,
                    "tie_tolerance": 0.0,
                    "root_seed": 11,
                },
            }
        ), newline="\n", encoding="utf-8"
    )
    return data, config


def test_validate_reports_attainability(tmp_path):
    data, config = _write(tmp_path, 64, [1, 2, 3, 4], 199)
    assert validate_files(data, config)["attainability"]["status"] == "REFUSE_NULL_STATES_TOO_FEW"


def test_run_refuses_an_unattainable_plan_before_writing_a_record(tmp_path):
    data, config = _write(tmp_path, 64, [1, 2, 3, 4], 199)
    with pytest.raises(WorkflowConfigError) as caught:
        run_files(data, config, tmp_path / "r.sqlite", max_bytes=CAP)
    assert caught.value.code == "unattainable_plan"
    assert not (tmp_path / "r.sqlite").exists()


def test_run_accepts_an_explicit_override_and_an_attainable_plan(tmp_path):
    data, config = _write(tmp_path, 64, [1, 2, 3, 4], 19)
    run_files(data, config, tmp_path / "o.sqlite", max_bytes=CAP, allow_unattainable=True)
    good = tmp_path / "good"
    good.mkdir()
    good_data, good_config = _write(good, 64, [1, 2], 19)
    run_files(good_data, good_config, tmp_path / "ok.sqlite", max_bytes=CAP)


def test_cli_run_exit_code_and_override_flag(tmp_path):
    data, config = _write(tmp_path, 64, [1, 2, 3, 4], 19)
    base = [sys.executable, "-m", "selcal", "run", str(data), str(config)]
    refused = subprocess.run(
        [*base, str(tmp_path / "a.sqlite"), "--max-bytes", str(CAP)],
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
    )
    assert refused.returncode == 2
    assert json.loads(refused.stdout)["error"] == "unattainable_plan"
    allowed = subprocess.run(
        [*base, str(tmp_path / "b.sqlite"), "--max-bytes", str(CAP), "--allow-unattainable-plan"],
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
    )
    assert allowed.returncode == 0, allowed.stderr


# --- Guard arithmetic must match the calibrator's float comparison p <= alpha ---
# (code review 2026-09-22)


def test_floor_equal_to_alpha_in_float_arithmetic_is_attainable():
    # n = 10, three lags: floor 3/10; the calibrator computes (1 + E)/(B + 1) = 0.3 <= 0.3.
    report = attainability(
        request(candidates=(1, 2, 3), replicates=9, alpha=0.3, null_name="circular_shift_exact_v1"),
        10,
    )
    assert report["status"] == "PASS"
    assert report["monte_carlo_power_cap"] == 1.0


def test_power_cap_counts_rejections_the_way_the_calibrator_does():
    from math import comb

    report = attainability(request(candidates=(1,), replicates=9, alpha=0.3), 10)
    floor = 1 / 10
    allowed = max(k for k in range(10) if (1 + k) / 10 <= 0.3)  # = 2 in float arithmetic
    expected = sum(comb(9, k) * floor**k * (1 - floor) ** (9 - k) for k in range(allowed + 1))
    assert report["monte_carlo_power_cap"] == pytest.approx(expected, rel=1e-12)


def test_wrong_enumeration_count_reports_no_power_cap():
    report = attainability(
        request(candidates=(1, 2), replicates=199, null_name="circular_shift_exact_v1"), 64
    )
    assert report["status"] == "REFUSE_ENUMERATION_REPLICATE_COUNT"
    assert report["monte_carlo_power_cap"] is None
