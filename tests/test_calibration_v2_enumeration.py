"""Calibration with the exact-enumeration null: every state once, fail closed on B mismatch."""

from __future__ import annotations

import numpy as np
import pytest

from selcal import calibrate_selected_family, resolve_plan_v2, verify_calibration_result
from selcal.contracts import RunStatus, SeriesPair
from selcal.contracts_v2 import PlanRequestV2, RunFailureStage

N = 12


def resolution(
    *, replicates: int = N - 1, null: str = "circular_shift_exact_v1", candidates=(1, 2)
):
    return resolve_plan_v2(
        PlanRequestV2(
            candidates=candidates,
            statistic_name="lagged_pearson_v1",
            statistic_params={},
            selection_rule="max_upper",
            null_name=null,
            null_params={"min_shift": 1},
            replicates=replicates,
            alpha=0.05,
            tie_tolerance=0.0,
            root_seed=19,
        )
    )


def pair(n: int = N) -> SeriesPair:
    rng = np.random.default_rng(5)
    x = rng.standard_normal(n)
    return SeriesPair(source=x, target=np.roll(x, 2) + 0.05 * rng.standard_normal(n))


def test_every_nonidentity_state_is_used_exactly_once():
    result = calibrate_selected_family(pair(), resolution())
    assert result.status is RunStatus.COMPLETE
    shifts = sorted(outcome.transform_token.state.shift for outcome in result.replicates)
    assert shifts == list(range(1, N))
    assert all(
        outcome.transform_token.null_name == "circular_shift_exact_v1"
        for outcome in result.replicates
    )


def test_p_value_equals_the_exact_enumeration_share():
    result = calibrate_selected_family(pair(), resolution())
    assert result.p_value == (1 + result.exceedance_count) / N
    verify_calibration_result(result, resolution())


def test_replicate_count_other_than_the_state_space_is_not_evaluable():
    result = calibrate_selected_family(pair(), resolution(replicates=N))
    assert result.status is RunStatus.NOT_EVALUABLE
    assert result.failure_stage is RunFailureStage.NULL_BIND
    assert "enumeration_replicate_count_mismatch_v1" in result.diagnostics
    assert result.p_value is None and result.replicates == ()


def test_sampling_null_is_unchanged_and_still_samples_with_replacement():
    result = calibrate_selected_family(
        pair(), resolution(null="circular_shift_v2", replicates=N - 1)
    )
    assert result.status is RunStatus.COMPLETE
    shifts = sorted(outcome.transform_token.state.shift for outcome in result.replicates)
    assert shifts != list(range(1, N))


@pytest.mark.parametrize("candidates", [(1, 2), (1, 3), (2, 4)])
def test_enumerated_decisions_match_an_independent_enumeration(candidates):
    observed = pair()
    result = calibrate_selected_family(observed, resolution(candidates=candidates))
    top = max(candidates)
    x, y = observed.source, observed.target

    def statistic(series: np.ndarray) -> float:
        return max(float(np.corrcoef(series[top - c : N - c], y[top:])[0, 1]) for c in candidates)

    reference = statistic(x)
    count = sum(statistic(x[(np.arange(N) - shift) % N]) >= reference for shift in range(N))
    assert result.p_value == count / N


# --- The verifier must enforce the enumeration contract from retained tokens alone ---
# Code review 2026-09-22: per-token checks accepted repeated, missing, identity and
# reordered states.

import dataclasses as _dc  # noqa: E402

from selcal.contracts_v2 import CircularShiftStateV2, SelCalV2Error  # noqa: E402


def _genuine():
    resolved = resolution()
    return resolved, calibrate_selected_family(pair(), resolved)


def _rebuild(result, outcomes):
    observed = result.observed_selection.decision_statistic
    exceed = sum(o.selection.decision_statistic >= observed for o in outcomes)
    p = (1 + exceed) / (len(outcomes) + 1)
    return _dc.replace(
        result,
        replicates=tuple(outcomes),
        exceedance_count=exceed,
        p_value=p,
        reject_null=p <= result.alpha,
    )


def _split(result):
    observed = result.observed_selection.decision_statistic
    outs = list(result.replicates)
    exceed = [o for o in outs if o.selection.decision_statistic >= observed]
    below = [o for o in outs if o.selection.decision_statistic < observed]
    return outs, exceed, below


def test_verifier_rejects_a_repeated_state_that_replaces_a_missing_one():
    resolved, result = _genuine()
    outs, exceed, below = _split(result)
    assert exceed and below
    slot = exceed[0].replicate_id
    forged = list(outs)
    forged[slot] = _dc.replace(
        below[0], replicate_id=slot, seed_digest_sha256=outs[slot].seed_digest_sha256
    )
    with pytest.raises(SelCalV2Error):
        verify_calibration_result(_rebuild(result, forged), resolved)


def test_verifier_rejects_every_replicate_reusing_one_state():
    resolved, result = _genuine()
    outs, _, below = _split(result)
    forged = [
        _dc.replace(below[0], replicate_id=i, seed_digest_sha256=o.seed_digest_sha256)
        for i, o in enumerate(outs)
    ]
    with pytest.raises(SelCalV2Error):
        verify_calibration_result(_rebuild(result, forged), resolved)


def test_verifier_rejects_the_identity_state_as_a_replicate():
    resolved, result = _genuine()
    outs = list(result.replicates)
    identity = _dc.replace(
        outs[0].transform_token,
        is_identity=True,
        state=CircularShiftStateV2(schema="selcal.circular-shift-state.v2", shift=0),
    )
    outs[0] = _dc.replace(
        outs[0],
        transform_token=identity,
        statistic_results=result.observed_results,
        selection=result.observed_selection,
    )
    with pytest.raises(SelCalV2Error):
        verify_calibration_result(_rebuild(result, outs), resolved)


def test_verifier_rejects_states_out_of_enumeration_order():
    resolved, result = _genuine()
    outs = list(result.replicates)
    forged = [
        _dc.replace(o, replicate_id=i, seed_digest_sha256=outs[i].seed_digest_sha256)
        for i, o in enumerate(reversed(outs))
    ]
    with pytest.raises(SelCalV2Error):
        verify_calibration_result(_rebuild(result, forged), resolved)


def test_verifier_rejects_a_complete_result_for_a_partial_enumeration_plan():
    from selcal.nulls.circular_shift_exact_v1 import CircularShiftExactNullV1
    from selcal.randomness import ReplicateRandomSource

    _, result = _genuine()
    partial = resolution(replicates=5)
    refused = calibrate_selected_family(pair(), partial)
    assert refused.status is RunStatus.NOT_EVALUABLE
    bound = (
        CircularShiftExactNullV1(min_shift=1)
        .bind(
            pair(),
            semantic_input_sha256=refused.semantic_input_sha256,
            scientific_plan_sha256=refused.scientific_plan_sha256,
        )
        .bound
    )
    outcomes = [
        _dc.replace(
            result.replicates[i],
            transform_token=bound.enumerate_token(i),
            seed_digest_sha256=ReplicateRandomSource(
                refused.scientific_plan_sha256, i, 5
            ).seed_digest_sha256,
        )
        for i in range(5)
    ]
    observed = result.observed_selection.decision_statistic
    exceed = sum(o.selection.decision_statistic >= observed for o in outcomes)
    p = (1 + exceed) / 6
    forged = _dc.replace(
        result,
        semantic_input_sha256=refused.semantic_input_sha256,
        scientific_plan_sha256=refused.scientific_plan_sha256,
        planned_replicates=5,
        replicates=tuple(outcomes),
        exceedance_count=exceed,
        p_value=p,
        reject_null=p <= result.alpha,
    )
    with pytest.raises(SelCalV2Error):
        verify_calibration_result(forged, partial)


def test_verifier_still_accepts_the_genuine_enumerated_result():
    resolved, result = _genuine()
    verify_calibration_result(result, resolved)
