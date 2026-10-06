"""Lagged Pearson accuracy against an exact rational reference (R10v3 Windows core audit, ALG-01).

The Windows core audit (2026-10-03) found that a large common offset with a small, exactly
representable spread produced a valid-looking but wrong correlation (0.943 and 0.853 where the
exact value is 1), and changed an end-to-end p-value from 0.30 to 0.40. The reference here is
computed from the actual float inputs with Fraction sums and a 50-digit square root; it never
calls product helpers.
"""

from __future__ import annotations

import json
import math
import random
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from selcal.statistics.lagged_pearson import _lagged_pearson
from selcal.workflow import run_files

EPS = float(np.finfo(np.float64).eps)


def exact_pearson(source: list[float], target: list[float]) -> float:
    a = [Fraction(value) for value in source]
    b = [Fraction(value) for value in target]
    n = len(a)
    mean_a, mean_b = sum(a) / n, sum(b) / n
    sab = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b, strict=True))
    saa = sum((x - mean_a) ** 2 for x in a)
    sbb = sum((y - mean_b) ** 2 for y in b)
    with localcontext() as context:
        context.prec = 50
        denominator = (Decimal(saa.numerator) / Decimal(saa.denominator)).sqrt() * (
            Decimal(sbb.numerator) / Decimal(sbb.denominator)
        ).sqrt()
        return float(Decimal(sab.numerator) / Decimal(sab.denominator) / denominator)


def product(source: list[float], target: list[float]) -> float:
    return _lagged_pearson(np.array(source, dtype=np.float64), np.array(target, dtype=np.float64))


# Lag-1 common support of the audit's two minimal cases: source[0:3], target[1:4].
@pytest.mark.parametrize(
    ("source", "target"),
    [
        ([1e16, 1e16, 1e16 + 2], [0.0, 0.0, 1.0]),
        ([1e16, 1e16 + 2, 1e16 + 4], [0.0, 1.0, 2.0]),
    ],
)
def test_audit_minimal_cases_match_the_exact_correlation(source, target):
    assert len(set(source)) >= 2
    assert exact_pearson(source, target) == 1.0
    assert product(source, target) == pytest.approx(1.0, abs=4 * EPS)


@pytest.mark.parametrize("offset", [1e16, -1e16, 2.0**53, -(2.0**53), 3e15])
def test_large_common_offsets_preserve_the_exact_value(offset):
    rng = random.Random(9137)
    checked = 0
    for _ in range(60):
        base = [2 * rng.randint(-50, 50) for _ in range(20)]
        target = [float(rng.randint(-100, 100)) for _ in range(20)]
        source = [offset + value for value in base]
        # The offset must keep each value exact, as in the audit (float spacing <= 2 here).
        assert all(Fraction(v) - Fraction(offset) == b for v, b in zip(source, base, strict=True))
        if len(set(base)) < 2 or len(set(target)) < 2:
            continue
        reference = exact_pearson(source, target)
        assert abs(product(source, target) - reference) <= 64 * EPS, (base, target)
        checked += 1
    assert checked >= 50


@pytest.mark.parametrize("exponent", [-300, -150, -20, 0, 20, 150, 300])
def test_generic_inputs_across_scales_match_the_exact_value(exponent):
    rng = random.Random(exponent)
    scale = 10.0**exponent
    for _ in range(40):
        n = rng.randint(3, 40)
        source = [rng.uniform(-1, 1) * scale for _ in range(n)]
        target = [rng.uniform(-1, 1) * scale + 0.5 * source[i] for i in range(n)]
        reference = exact_pearson(source, target)
        assert abs(product(source, target) - reference) <= 64 * EPS


def test_offset_does_not_change_the_end_to_end_record(tmp_path: Path):
    """The audit's file-chain case: adding an exact offset changed p from 0.30 to 0.40."""
    rng = random.Random(9137)
    x = [2 * rng.randint(-50, 50) for _ in range(20)]
    y = [rng.randint(-100, 100) for _ in range(20)]
    config = {
        "plan": {
            "candidates": [1],
            "statistic_name": "lagged_pearson_v1",
            "statistic_params": {},
            "selection_rule": "max_absolute",
            "null_name": "circular_shift_exact_v1",
            "null_params": {"min_shift": 1},
            "replicates": 19,
            "alpha": 0.05,
            "tie_tolerance": 0.0,
            "root_seed": 17,
        }
    }
    template = json.loads(
        (Path(__file__).resolve().parents[1] / "examples/workflow/pearson.json").read_text(
            encoding="utf-8"
        )
    )
    template["plan"].update(config["plan"])
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(template), encoding="utf-8")
    results = {}
    for name, shift in (("baseline", 0), ("offset", 10**16)):
        data = tmp_path / f"{name}.csv"
        data.write_text(
            "x,y\n" + "".join(f"{a + shift},{b}\n" for a, b in zip(x, y, strict=True)),
            encoding="utf-8",
        )
        results[name] = run_files(
            data, cfg, tmp_path / f"{name}.sqlite", max_bytes=1 << 20, allow_unattainable=True
        )
    base, moved = results["baseline"], results["offset"]
    assert (base.p_value, moved.p_value) == (base.p_value, base.p_value)
    assert base.exceedance_count == moved.exceedance_count
    statistic = base.observed_selection.decision_statistic
    assert math.isclose(moved.observed_selection.decision_statistic, statistic, abs_tol=64 * EPS)
