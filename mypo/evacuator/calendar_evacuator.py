"""Optimizer for weights of portfolio."""
from datetime import datetime
from typing import List

import numpy as np

from mypo import Market
from mypo.common import safe_cast
from mypo.evacuator import BaseEvacuator


class CalendarEvacuator(BaseEvacuator):
    """Weighted rebalance strategy by monthly applying."""

    _months: List[int]
    _risk_off: np.float64
    _risk_on: np.float64

    def __init__(self, months: List[int], risk_off: float = 0.9, risk_on: float = 1.0):
        """Construct this object.

        Args:
            months: Risk off months.
            risk_off: Risk off rate.
            risk_on: Risk on rate.
        """
        self._months = months
        self._risk_off = np.float64(risk_off)
        self._risk_on = np.float64(risk_on)

    def evacuate(
        self, at: datetime, market: Market, assets: np.ndarray, cash: np.float64, weights: np.ndarray
    ) -> np.ndarray:
        """Apply risk off strategy to current situation.

        Args:
            at: Current date.
            market: Market data.
            assets: Assets
            cash: Cash
            weights: Weights.
        Returns:
            New assets.
        """
        weights = safe_cast(weights)
        if at.month in self._months:
            new_assets = (np.sum(assets) + cash) * self._risk_off * weights
        else:
            new_assets = (np.sum(assets) + cash) * self._risk_on * weights  # pragma: no cover
        return safe_cast(new_assets)
