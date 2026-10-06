"""Statistic adapter contracts and reference implementations."""

from selcal.statistics.base import BoundStatisticAdapter, StatisticAdapter
from selcal.statistics.binned_nette import (
    BinnedNetTEAdapter,
    conditional_mutual_information,
)
from selcal.statistics.lagged_pearson import LaggedPearsonAdapter

__all__ = [
    "BinnedNetTEAdapter",
    "BoundStatisticAdapter",
    "LaggedPearsonAdapter",
    "StatisticAdapter",
    "conditional_mutual_information",
]
