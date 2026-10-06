"""Three-level preflight: input valid / plan executable / scientific assumptions declared.

Also guards against expensive plan-time arithmetic before the resource check (review 2026-09-22).
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from fractions import Fraction
from math import comb
from pathlib import Path

import numpy as np
import pytest

from selcal.contracts_v2 import PlanRequestV2
from selcal.workflow import attainability, run_files, validate_files
from selcal.workflow_config import WorkflowConfigError

CAP = 1_048_576


def request(*, candidates=(1, 2), replicates=199, null_name="circular_shift_v2"):
    return PlanRequestV2(
        candidates=candidates,
        statistic_name="lagged_pearson_v1",
        statistic_params={},
        selection_rule="max_upper",
        null_name=null_name,
        null_params={"min_shift": 1},
        replicates=replicates,
        alpha=0.05,
        tie_tolerance=0.0,
        root_seed=5,
    )


def write(
    tmp_path: Path, *, n: int, lags: list[int], replicates: int, null: str
) -> tuple[Path, Path]:
    rng = np.random.default_rng(11)
    x, y = rng.standard_normal(n), rng.standard_normal(n)
    data = tmp_path / "data.csv"
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
                    "null_name": null,
                    "null_params": {"min_shift": 1},
                    "replicates": replicates,
                    "alpha": 0.05,
                    "tie_tolerance": 0.0,
                    "root_seed": 3,
                },
            }
        ), newline="\n", encoding="utf-8"
    )
    return data, config


def test_power_cap_is_fast_for_the_largest_representable_plan():
    started = time.monotonic()
    report = attainability(request(replicates=1_000_000), 64)
    assert time.monotonic() - started < 2.0
    assert 0.0 <= report["monte_carlo_power_cap"] <= 1.0


@pytest.mark.parametrize("replicates", [19, 199, 999])
def test_power_cap_matches_exact_rational_binomial(replicates):
    floor = Fraction(2, 64)
    allowed = sum(1 for k in range(replicates + 1) if (1 + k) / (replicates + 1) <= 0.05)
    exact = sum(
        comb(replicates, k) * floor**k * (1 - floor) ** (replicates - k) for k in range(allowed)
    )
    report = attainability(request(replicates=replicates), 64)
    assert report["monte_carlo_power_cap"] == pytest.approx(float(exact), rel=1e-9, abs=1e-12)


def test_validate_separates_the_three_levels(tmp_path):
    data, config = write(
        tmp_path, n=100, lags=[1, 2, 3], replicates=99, null="circular_shift_exact_v1"
    )
    preflight = validate_files(data, config)["preflight"]
    assert preflight["input"]["status"] == "VALID"
    assert preflight["plan"]["status"] == "EXECUTABLE"
    assert preflight["plan"]["reasons"] == []
    science = preflight["scientific_assumptions"]
    assert science["status"] == "DECLARED_NOT_VERIFIED"
    codes = {item["code"] for item in science["assumptions"]}
    assert "lags_declared_before_seeing_data" in codes
    assert "circular_shift_exchangeability" in codes


def test_resource_budget_is_reported_as_a_specific_plan_reason(tmp_path):
    # Exact enumeration with 8 lags needs B = n - 1; at n = 700 the work cap is exceeded.
    data, config = write(
        tmp_path, n=700, lags=list(range(1, 9)), replicates=699, null="circular_shift_exact_v1"
    )
    plan = validate_files(data, config)["preflight"]["plan"]
    assert plan["status"] == "NOT_EXECUTABLE"
    assert "resource_budget_exceeded" in plan["reasons"]
    assert plan["resource_budget"]["within_caps"] is False


def test_huge_replicate_count_is_refused_quickly_without_calibration(tmp_path):
    data, config = write(
        tmp_path, n=64, lags=[1, 2], replicates=1_000_000, null="circular_shift_v2"
    )
    started = time.monotonic()
    plan = validate_files(data, config)["preflight"]["plan"]
    assert time.monotonic() - started < 3.0
    assert "resource_budget_exceeded" in plan["reasons"]
    started = time.monotonic()
    with pytest.raises(WorkflowConfigError) as caught:
        run_files(data, config, tmp_path / "r.sqlite", max_bytes=CAP)
    assert time.monotonic() - started < 3.0
    assert caught.value.code == "resource_budget_exceeded"
    assert not (tmp_path / "r.sqlite").exists()


def test_attainability_refusal_is_listed_with_its_reason(tmp_path):
    data, config = write(
        tmp_path, n=64, lags=[1, 2, 3, 4], replicates=199, null="circular_shift_v2"
    )
    plan = validate_files(data, config)["preflight"]["plan"]
    assert plan["status"] == "NOT_EXECUTABLE"
    assert "REFUSE_NULL_STATES_TOO_FEW" in plan["reasons"]


def test_cli_validate_exits_2_with_the_first_reason_when_the_plan_is_not_executable(tmp_path):
    data, config = write(
        tmp_path, n=64, lags=[1, 2, 3, 4], replicates=199, null="circular_shift_v2"
    )
    done = subprocess.run(
        [sys.executable, "-m", "selcal", "validate", str(data), str(config)],
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
    )
    payload = json.loads(done.stdout)
    assert done.returncode == 2
    assert payload["outcome"] == "PLAN_NOT_EXECUTABLE"
    assert payload["error"] == "REFUSE_NULL_STATES_TOO_FEW"
    assert payload["data"]["preflight"]["input"]["status"] == "VALID"


@pytest.mark.parametrize("null_name", ["circular_shift_v2", "circular_shift_exact_v1"])
def test_core_api_rejects_a_huge_plan_before_any_replicate(null_name):
    from selcal import calibrate_selected_family, resolve_plan_v2
    from selcal.contracts import SeriesPair
    from selcal.contracts_v2 import ResourceLimitError

    rng = np.random.default_rng(2)
    pair = SeriesPair(source=rng.standard_normal(64), target=rng.standard_normal(64))
    resolution = resolve_plan_v2(request(replicates=1_000_000, null_name=null_name))
    started = time.monotonic()
    if null_name == "circular_shift_exact_v1":
        # B != n - 1: refused as NOT_EVALUABLE at null_bind before the budget is reached.
        result = calibrate_selected_family(pair, resolution)
        assert result.replicates == ()
    else:
        with pytest.raises(ResourceLimitError):
            calibrate_selected_family(pair, resolution)
    assert time.monotonic() - started < 2.0
