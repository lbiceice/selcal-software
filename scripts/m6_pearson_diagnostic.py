"""Bounded same-team independent arithmetic diagnostic; not the full M6 reference.

Only the standard library is used. No public SelCal interfaces are changed.
The limits below bound this small diagnostic's coverage, not product capacity.
"""

from __future__ import annotations

import math
from itertools import pairwise, permutations

MAX_N = 64
MAX_CANDIDATES = 8
MAX_STATES = 720
MAX_SUPPORT_VISITS = 100_000


class AnalyticFailure(ValueError):
    """A specified analytical failure, distinct from malformed inputs."""


def _vector(values: tuple[float, ...]) -> None:
    if type(values) is not tuple or not 2 <= len(values) <= MAX_N:
        raise ValueError("finite float tuple length must be between 2 and 64")
    if any(type(x) is not float or not math.isfinite(x) for x in values):
        raise ValueError("finite built-in floats required")


def _candidates(candidates: tuple[int, ...]) -> None:
    if type(candidates) is not tuple or not 1 <= len(candidates) <= MAX_CANDIDATES:
        raise ValueError("one to eight candidates required")
    if any(type(c) is not int or c <= 0 for c in candidates):
        raise ValueError("positive built-in candidate integers required")
    if any(a >= b for a, b in pairwise(candidates)):
        raise ValueError("strictly increasing candidates required")


def _rule(rule: str, tolerance: float) -> None:
    if type(rule) is not str or rule not in ("max_upper", "max_absolute"):
        raise ValueError("unknown selection rule")
    if type(tolerance) is not float or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("finite nonnegative float tolerance required")


def _center(values: tuple[float, ...], role: str) -> tuple[tuple[float, ...], float]:
    scale = max(abs(x) for x in values)
    if scale == 0.0:
        raise AnalyticFailure("ZERO_" + role + "_VARIANCE")
    scaled = tuple(x / scale for x in values)
    mean = math.fsum(scaled) / len(scaled)
    centered = tuple(x - mean for x in scaled)
    norm = math.sqrt(math.fsum(x * x for x in centered))
    if norm == 0.0:
        raise AnalyticFailure("ZERO_" + role + "_VARIANCE")
    return centered, norm


def pearson(source: tuple[float, ...], target: tuple[float, ...]) -> float:
    """Separately summed centered Pearson correlation on the supplied support."""
    _vector(source)
    _vector(target)
    if len(source) != len(target):
        raise ValueError("equal lengths required")
    u, unorm = _center(source, "SOURCE")
    v, vnorm = _center(target, "TARGET")
    value = math.fsum(a * b for a, b in zip(u, v, strict=True)) / (unorm * vnorm)
    if not math.isfinite(value) or abs(value) > 1.0 + 32 * 2**-52:
        raise AnalyticFailure("NUMERIC_FAILURE")
    return min(1.0, max(-1.0, value))


def select(
    estimates: tuple[float, ...], candidates: tuple[int, ...], rule: str, tolerance: float
) -> dict:
    """Retain the true family maximum separately from its canonical tie label."""
    _candidates(candidates)
    _rule(rule, tolerance)
    if (
        type(estimates) is not tuple
        or len(estimates) != len(candidates)
        or any(type(x) is not float or not math.isfinite(x) for x in estimates)
    ):
        raise ValueError("complete finite estimate vector required")
    scores = tuple(x if rule == "max_upper" else abs(x) for x in estimates)
    maximum = max(scores)
    if maximum == 0.0:
        maximum = 0.0
    tied = tuple(i for i, score in enumerate(scores) if maximum - score <= tolerance)
    index = tied[0]
    return dict(
        selected_index=index,
        selected_candidate=candidates[index],
        tied_candidates=tuple(candidates[i] for i in tied),
        decision_statistic=maximum,
        selected_estimate=estimates[index],
    )


def _scan_inputs(
    source: tuple[float, ...],
    target: tuple[float, ...],
    candidates: tuple[int, ...],
    rule: str,
    tolerance: float,
) -> None:
    _vector(source)
    _vector(target)
    _candidates(candidates)
    _rule(rule, tolerance)
    if len(source) != len(target) or len(source) - candidates[-1] < 2:
        raise ValueError("equal lengths and max-lag support >= 2 required")


def scan(
    source: tuple[float, ...],
    target: tuple[float, ...],
    candidates: tuple[int, ...],
    rule: str,
    tolerance: float,
) -> dict:
    """Evaluate every candidate on one max-lag support, retaining every failure."""
    _scan_inputs(source, target, candidates, rule, tolerance)
    h = candidates[-1]
    records = []
    for lag in candidates:
        failure = None
        try:
            estimate = pearson(tuple(source[t - lag] for t in range(h, len(source))), target[h:])
        except AnalyticFailure as error:
            estimate, failure = None, str(error)
        records.append(
            dict(
                candidate_id=lag,
                estimate=estimate,
                support_n=len(source) - h,
                failure_reason=failure,
            )
        )
    selection = None
    if all(r["failure_reason"] is None for r in records):
        selection = select(tuple(r["estimate"] for r in records), candidates, rule, tolerance)
    return dict(candidate_records=tuple(records), selection=selection)


def _empty(status: str, reason: str) -> dict:
    return dict(
        status=status,
        failure_reason=reason,
        states=(),
        nonidentity_exceedance_count=None,
        failure_count=None,
        numerator=None,
        denominator=None,
        p_exact=None,
    )


def exact(
    source: tuple[float, ...],
    target: tuple[float, ...],
    candidates: tuple[int, ...],
    *,
    rule: str = "max_upper",
    tolerance: float = 0.0,
    null_name: str = "circular_shift_v2",
    null_parameter: int = 1,
) -> dict:
    """Recompute all admitted labelled states, independently of product code.

    Aggregate p is absent if any required candidate scan fails. There is no
    random sampler, fallback, truncation, or deduplication of numerical arrays.
    """
    _scan_inputs(source, target, candidates, rule, tolerance)
    if type(null_parameter) is not int or null_parameter < 1:
        raise ValueError("positive built-in null parameter required")
    if type(null_name) is not str or null_name not in ("circular_shift_v2", "block_shuffle_v2"):
        raise ValueError("unsupported null")
    n = len(source)
    if null_name == "circular_shift_v2":
        if 2 * null_parameter > n:
            return _empty("NULL_DISABLED", "EMPTY_NONIDENTITY_SPACE")
        total = n - 2 * null_parameter + 2
    else:
        if n % null_parameter or n // null_parameter < 2:
            return _empty("NULL_DISABLED", "INVALID_BLOCK_PARTITION")
        total = 1
        for factor in range(2, n // null_parameter + 1):
            if total > MAX_STATES // factor:
                return _empty("REFERENCE_RESOURCE_LIMIT", "STATE_COUNT_LIMIT")
            total *= factor
    if total > MAX_STATES:
        return _empty("REFERENCE_RESOURCE_LIMIT", "STATE_COUNT_LIMIT")
    if total * len(candidates) * (n - candidates[-1]) > MAX_SUPPORT_VISITS:
        return _empty("REFERENCE_RESOURCE_LIMIT", "WORK_LIMIT")

    # Only allocate the bounded state iterator and output rows after admission.
    if null_name == "circular_shift_v2":
        states = iter((0, *range(null_parameter, n - null_parameter + 1)))
    else:
        states = permutations(range(n // null_parameter))
    rows = []
    observed_maximum = None
    exceedances = failures = 0
    for ordinal, state in enumerate(states):
        if null_name == "circular_shift_v2":
            transformed = tuple(source[(j - state) % n] for j in range(n))
        else:
            transformed = tuple(
                source[label * null_parameter + j] for label in state for j in range(null_parameter)
            )
        scanned = scan(transformed, target, candidates, rule, tolerance)
        selected = scanned["selection"]
        if ordinal == 0 and selected is not None:
            observed_maximum = selected["decision_statistic"]
        indicator = None
        if selected is None:
            failures += 1
        elif observed_maximum is not None:
            indicator = selected["decision_statistic"] >= observed_maximum
            if ordinal != 0:
                exceedances += int(indicator)
        rows.append(
            dict(
                state_id=ordinal,
                state=state,
                is_identity=(ordinal == 0),
                **scanned,
                exceeds_observed=indicator,
            )
        )
    complete = failures == 0
    return dict(
        status="COMPLETE_TABLE" if complete else "ANALYTICALLY_UNEVALUABLE",
        failure_reason=None if complete else "REQUIRED_SCAN_FAILED",
        states=tuple(rows),
        failure_count=failures,
        nonidentity_exceedance_count=(exceedances if observed_maximum is not None else None),
        numerator=(1 + exceedances if complete else None),
        denominator=total,
        p_exact=((1 + exceedances) / total if complete else None),
    )


def summarize_tail(observed: float, maxima: tuple[float | None, ...]) -> dict:
    """Conditional finite-schedule arithmetic; None means failed, never false."""
    if type(observed) is not float or not math.isfinite(observed):
        raise ValueError("finite observed maximum required")
    if type(maxima) is not tuple or not 1 <= len(maxima) <= MAX_STATES:
        raise ValueError("one to 720 planned schedule positions required")
    if any(x is not None and (type(x) is not float or not math.isfinite(x)) for x in maxima):
        raise ValueError("each maximum must be finite float or None")
    indicators = tuple(None if x is None else x >= observed for x in maxima)
    b = len(indicators)
    e = sum(x is True for x in indicators)
    f = sum(x is None for x in indicators)
    bounds = ((1 + e) / (b + 1), (1 + e + f) / (b + 1))
    return dict(
        B=b, E=e, F=f, indicators=indicators, bounds=bounds, p=(bounds[0] if f == 0 else None)
    )
