"""Deterministic candidate selection for complete statistic vectors."""

from __future__ import annotations

import math
from dataclasses import replace
from itertools import pairwise
from numbers import Real

from selcal.contracts import SelectionResult, SelectionRule, StatisticResult, Validity

__all__ = ["select_candidate"]


def _finite_real(value: object, *, error_message: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(error_message)
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(error_message) from error
    if not math.isfinite(normalized):
        raise ValueError(error_message)
    if normalized == 0.0:
        return 0.0
    return normalized


def select_candidate(
    results: tuple[StatisticResult, ...],
    rule: SelectionRule | str,
    tie_tolerance: float,
) -> tuple[SelectionResult, tuple[StatisticResult, ...]]:
    """Score a canonical result vector and select its first tolerance tie."""

    if isinstance(rule, SelectionRule):
        normalized_rule = rule
    elif type(rule) is str:
        try:
            normalized_rule = SelectionRule(rule)
        except ValueError as error:
            raise ValueError("selection rule is unsupported") from error
    else:
        raise ValueError("selection rule is unsupported")

    normalized_tolerance = _finite_real(
        tie_tolerance,
        error_message="tie_tolerance must be a finite real number >= 0",
    )
    if normalized_tolerance < 0:
        raise ValueError("tie_tolerance must be a finite real number >= 0")

    if type(results) is not tuple:
        raise ValueError("results must be an exact tuple")
    if not results:
        raise ValueError("results must be non-empty")
    if any(type(result) is not StatisticResult for result in results):
        raise ValueError("results must contain only exact StatisticResult records")

    candidate_ids = tuple(result.candidate_id for result in results)
    if any(type(candidate_id) is not int or candidate_id <= 0 for candidate_id in candidate_ids):
        raise ValueError("results must have canonical candidate IDs")
    if any(current >= following for current, following in pairwise(candidate_ids)):
        raise ValueError("results must have canonical candidate IDs")

    scored_results: list[StatisticResult] = []
    scores: list[float] = []
    for result in results:
        if result.validity is not Validity.VALID:
            raise ValueError("selection requires only VALID statistic results")
        estimate = _finite_real(
            result.estimate,
            error_message="selection requires a finite estimate for every result",
        )

        score = estimate if normalized_rule is SelectionRule.MAX_UPPER else abs(estimate)
        scores.append(score)
        scored_results.append(replace(result, estimate=estimate, selection_score=score))

    top = max(scores)
    tied_indices = tuple(
        index for index, score in enumerate(scores) if top - score <= normalized_tolerance
    )
    selected_index = tied_indices[0]
    scored_tuple = tuple(scored_results)
    selection = SelectionResult(
        selected_candidate=scored_tuple[selected_index].candidate_id,
        selected_index=selected_index,
        decision_statistic=scores[selected_index],
        tied_candidates=tuple(scored_tuple[index].candidate_id for index in tied_indices),
    )
    return selection, scored_tuple
