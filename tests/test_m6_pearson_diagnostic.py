from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[1] / "scripts" / "m6_pearson_diagnostic.py"


def reference():
    assert MODULE.is_file(), "Independent Pearson diagnostic implementation is absent"
    spec = importlib.util.spec_from_file_location("m6_pearson_diagnostic", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "target,expected",
    [
        ((0.0, 1.0, 2.0), 1.0),
        ((2.0, 1.0, 0.0), -1.0),
        ((1.0, 0.0, 1.0), 0.0),
    ],
)
def test_pearson_hand_algebra(target, expected):
    assert reference().pearson((0.0, 1.0, 2.0), target) == pytest.approx(expected)


def test_constant_failure_precedence():
    ref = reference()
    with pytest.raises(ref.AnalyticFailure, match="ZERO_SOURCE_VARIANCE"):
        ref.pearson((1.0, 1.0), (2.0, 2.0))
    with pytest.raises(ref.AnalyticFailure, match="ZERO_TARGET_VARIANCE"):
        ref.pearson((1.0, 2.0), (2.0, 2.0))


def test_true_maximum_is_not_representative_score():
    out = reference().select((0.875, 1.0), (1, 2), "max_upper", 0.125)
    assert out == dict(
        selected_index=0,
        selected_candidate=1,
        tied_candidates=(1, 2),
        decision_statistic=1.0,
        selected_estimate=0.875,
    )


def test_signed_estimate_retained_under_absolute_rule():
    out = reference().select((-1.0, 0.5), (1, 2), "max_absolute", 0.0)
    assert out["decision_statistic"] == 1.0
    assert out["selected_estimate"] == -1.0


def test_common_support_and_full_failed_vector():
    out = reference().scan((0.0, 0.0, 0.0, 1.0), (1.0, 2.0, 3.0, 4.0), (1, 2), "max_upper", 0.0)
    assert out["selection"] is None
    assert len(out["candidate_records"]) == 2
    assert [r["support_n"] for r in out["candidate_records"]] == [2, 2]
    assert [r["failure_reason"] for r in out["candidate_records"]] == [
        "ZERO_SOURCE_VARIANCE",
        "ZERO_SOURCE_VARIANCE",
    ]


@pytest.mark.parametrize("candidates", [(2, 1), (1, 1), (True,), (0,), (), (3,)])
def test_bad_candidates_rejected(candidates):
    with pytest.raises(ValueError):
        reference().scan((1.0, 2.0, 3.0, 4.0), (2.0, 1.0, 4.0, 3.0), candidates, "max_upper", 0.0)


@pytest.mark.parametrize("value", [math.inf, math.nan, -1.0, True])
def test_bad_tolerance_rejected(value):
    with pytest.raises(ValueError):
        reference().select((0.5,), (1,), "max_upper", value)


def test_finite_extremes_and_subnormal():
    ref = reference()
    large = float.fromhex("0x1.fffffffffffffp+1023")
    tiny = math.ulp(0.0)
    assert ref.pearson((-large, 0.0, large), (-large, 0.0, large)) == pytest.approx(1.0)
    assert ref.pearson((0.0, tiny, 2 * tiny), (2 * tiny, tiny, 0.0)) == pytest.approx(-1.0)


X = (0.0, 3.0, 1.0, 2.0, 4.0, 5.0)
Y = (0.0, 1.0, 3.0, 2.0, 5.0, 4.0)


def test_six_state_pearson_kills_fixed_candidate_shortcut():
    ref = reference()
    assert callable(getattr(ref, "exact", None)), "complete labelled diagnostic is absent"
    out = ref.exact(X, Y, (1, 2))
    assert out["status"] == "COMPLETE_TABLE"
    assert [r["selection"]["selected_candidate"] for r in out["states"]] == [1, 2, 1, 2, 2, 1]
    assert [r["exceeds_observed"] for r in out["states"]] == [True, False, False, False, True, True]
    assert out["nonidentity_exceedance_count"] == 2
    assert (out["numerator"], out["denominator"], out["p_exact"]) == (3, 6, 0.5)
    observed = out["states"][0]["candidate_records"][0]["estimate"]
    fixed = sum(r["candidate_records"][0]["estimate"] >= observed for r in out["states"])
    assert fixed == 2
    assert fixed / 6 != out["p_exact"]


def test_labelled_multiplicity_not_numeric_deduplication():
    ref = reference()
    x = (0.0, 1.0, 0.0, 1.0, 2.0, 3.0)
    y = (0.0, 2.0, 1.0, 4.0, 3.0, 5.0)
    assert callable(getattr(ref, "exact", None)), "complete labelled diagnostic is absent"
    out = ref.exact(x, y, (1, 2), null_name="block_shuffle_v2", null_parameter=2)
    assert len(out["states"]) == out["denominator"] == 6
    assert sum(r["is_identity"] for r in out["states"]) == 1
    from collections import Counter

    arrays = [
        tuple(x[block * 2 + j] for block in row["state"] for j in range(2)) for row in out["states"]
    ]
    assert sorted(Counter(arrays).values()) == [2, 2, 2]
    assert len(set(arrays)) != out["denominator"]


def test_analytic_failure_keeps_states_and_never_selects_valid_subset():
    ref = reference()
    assert callable(getattr(ref, "exact", None)), "complete labelled diagnostic is absent"
    out = ref.exact((0.0, 0.0, 1.0, 2.0, 0.0, 0.0), (0.0, 1.0, 2.0, 3.0, 4.0, 5.0), (1, 2))
    assert out["status"] == "ANALYTICALLY_UNEVALUABLE"
    assert len(out["states"]) == 6
    assert out["failure_count"] == 2
    assert out["p_exact"] is out["numerator"] is None
    for index, failed_lag in [(2, 2), (3, 1)]:
        row = out["states"][index]
        assert row["selection"] is row["exceeds_observed"] is None
        assert [r["candidate_id"] for r in row["candidate_records"]] == [1, 2]
        assert row["candidate_records"][failed_lag - 1]["failure_reason"] == "ZERO_SOURCE_VARIANCE"


@pytest.mark.parametrize(
    "kind,param", [("circular_shift_v2", 4), ("block_shuffle_v2", 4), ("block_shuffle_v2", 6)]
)
def test_disabled_nulls_do_not_scan(kind, param, monkeypatch):
    ref = reference()
    assert callable(getattr(ref, "exact", None)), "complete labelled diagnostic is absent"

    def forbidden(*args):
        pytest.fail("disabled null scanned")

    monkeypatch.setattr(ref, "scan", forbidden)
    assert (
        ref.exact(X, Y, (1, 2), null_name=kind, null_parameter=param)["status"] == "NULL_DISABLED"
    )


def test_state_cap_boundary_is_admitted():
    ref = reference()
    assert callable(getattr(ref, "exact", None)), "complete labelled diagnostic is absent"
    out = ref.exact(X, Y, (1, 2), null_name="block_shuffle_v2", null_parameter=1)
    assert len(out["states"]) == 720
    assert out["status"] != "REFERENCE_RESOURCE_LIMIT"


def test_over_budget_is_rejected_before_scan(monkeypatch):
    ref = reference()
    assert callable(getattr(ref, "exact", None)), "complete labelled diagnostic is absent"

    def forbidden(*args):
        pytest.fail("over-budget input scanned")

    monkeypatch.setattr(ref, "scan", forbidden)
    x = tuple(float(i) for i in range(8))
    out = ref.exact(x, x, (1,), null_name="block_shuffle_v2", null_parameter=1)
    assert out["status"] == "REFERENCE_RESOURCE_LIMIT"
    assert out["failure_reason"] == "STATE_COUNT_LIMIT"
    assert out["states"] == () and out["p_exact"] is None
    x = tuple(float(i) for i in range(60))
    out = ref.exact(x, x, tuple(range(1, 9)), null_name="block_shuffle_v2", null_parameter=10)
    assert out["failure_reason"] == "WORK_LIMIT"
    assert out["states"] == ()


def test_restricted_circular_states_and_orientation():
    ref = reference()
    assert callable(getattr(ref, "exact", None)), "complete labelled diagnostic is absent"
    out = ref.exact(X, Y, (1, 2), null_parameter=2)
    assert [r["state"] for r in out["states"]] == [0, 2, 3, 4]
    expected = ref.scan(tuple(X[(j - 2) % 6] for j in range(6)), Y, (1, 2), "max_upper", 0.0)
    assert out["states"][1]["candidate_records"] == expected["candidate_records"]
    wrong = ref.scan(tuple(X[(j + 2) % 6] for j in range(6)), Y, (1, 2), "max_upper", 0.0)
    assert out["states"][1]["candidate_records"] != wrong["candidate_records"]


def test_schedule_retains_planned_denominator_and_failed_positions():
    ref = reference()
    assert callable(getattr(ref, "summarize_tail", None)), "finite-B arithmetic is absent"
    out = ref.summarize_tail(3.0, (3.0, 4.0, 4.0, 3.0, 2.0))
    assert (out["B"], out["E"], out["F"], out["p"]) == (5, 4, 0, 5 / 6)
    out = ref.summarize_tail(3.0, (3.0, None, 2.0))
    assert (out["B"], out["E"], out["F"], out["p"]) == (3, 1, 1, None)
    assert out["bounds"] == (0.5, 0.75)
    assert out["indicators"] == (True, None, False)


def test_tail_is_inclusive_without_tolerance():
    ref = reference()
    assert callable(getattr(ref, "summarize_tail", None)), "finite-B arithmetic is absent"
    out = ref.summarize_tail(1.0, (1.0, math.nextafter(1.0, 0.0)))
    assert out["indicators"] == (True, False)
    assert out["E"] == 1
