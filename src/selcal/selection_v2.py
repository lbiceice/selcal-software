"""Fail-closed v2 family selection over an exact raw statistic vector."""

from __future__ import annotations

import math
from dataclasses import replace
from itertools import pairwise
from typing import NoReturn, cast

from selcal.contracts import SelectionResult, SelectionRule, StatisticResult, Validity
from selcal.contracts_v2 import V2IntegrityError

__all__ = ["select_family_v2"]


def _integrity(message: str) -> NoReturn:
    raise V2IntegrityError(message)


def _exact_rule(value: object) -> SelectionRule:
    if type(value) is SelectionRule:
        return value
    if type(value) is str:
        try:
            return SelectionRule(value)
        except ValueError as error:
            raise V2IntegrityError("selection rule is unsupported") from error
    _integrity("selection rule is unsupported")


def _exact_candidates(value: object) -> tuple[int, ...]:
    if type(value) is not tuple:
        _integrity("expected_candidates must be an exact canonical tuple")
    candidates = cast(tuple[object, ...], value)
    if not candidates or any(
        type(candidate) is not int or candidate <= 0 for candidate in candidates
    ):
        _integrity("expected_candidates must be an exact canonical tuple")
    exact_candidates = cast(tuple[int, ...], candidates)
    if any(current >= following for current, following in pairwise(exact_candidates)):
        _integrity("expected_candidates must be an exact canonical tuple")
    return exact_candidates


def select_family_v2(
    results: tuple[StatisticResult, ...],
    expected_candidates: tuple[int, ...],
    rule: SelectionRule | str,
    tie_tolerance: float,
) -> tuple[SelectionResult, tuple[StatisticResult, ...]]:
    """Select a canonical label while retaining the true family maximum."""

    candidates = _exact_candidates(expected_candidates)
    normalized_rule = _exact_rule(rule)
    if type(tie_tolerance) is not float or not math.isfinite(tie_tolerance):
        _integrity("tie_tolerance must be an exact finite built-in float >= 0")
    if tie_tolerance < 0.0:
        _integrity("tie_tolerance must be an exact finite built-in float >= 0")
    normalized_tolerance = 0.0 if tie_tolerance == 0.0 else tie_tolerance

    if type(results) is not tuple:
        _integrity("results must be an exact tuple")
    if any(type(result) is not StatisticResult for result in results):
        _integrity("results must contain only exact StatisticResult records")
    if len(results) != len(candidates):
        _integrity("result candidate vector must exactly match expected_candidates")

    exact_results = results
    scores_list: list[float] = []
    for expected, result in zip(candidates, exact_results, strict=True):
        if type(result.candidate_id) is not int or result.candidate_id != expected:
            _integrity("result candidate vector must exactly match expected_candidates")
        if result.validity is not Validity.VALID:
            _integrity("v2 family selection requires only VALID statistic results")
        if result.selection_score is not None:
            _integrity("raw v2 statistic results must have selection_score None")
        estimate = result.estimate
        if type(estimate) is not float or not math.isfinite(estimate):
            _integrity("v2 family selection requires a finite estimate for every result")
        score = estimate if normalized_rule is SelectionRule.MAX_UPPER else abs(estimate)
        scores_list.append(0.0 if score == 0.0 else score)

    scores = tuple(scores_list)
    decision_statistic = max(scores)
    tied_indices = tuple(
        index
        for index, score in enumerate(scores)
        if decision_statistic - score <= normalized_tolerance
    )
    selected_index = tied_indices[0]
    scored_results = tuple(
        replace(result, selection_score=score)
        for result, score in zip(exact_results, scores, strict=True)
    )
    selection = SelectionResult(
        selected_candidate=candidates[selected_index],
        selected_index=selected_index,
        decision_statistic=decision_statistic,
        tied_candidates=tuple(candidates[index] for index in tied_indices),
    )
    return selection, scored_results
