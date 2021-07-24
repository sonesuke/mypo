"""Optimizer for weights of portfolio."""
from datetime import datetime

import numpy as np
import numpy.typing as npt
import pandas as pd

from mypo import Market
from mypo.common import safe_cast
from mypo.evacuator import BaseEvacuator


class MovingAverageEvacuator(BaseEvacuator):
    """Weighted rebalance strategy by monthly applying."""

    _span: int
    _risk_off: np.float64
    _risk_on: np.float64

    def __init__(self, span: int = 20, risk_on: np.float64 = np.float64(1.0), risk_off: np.float64 = np.float64(0.0)):
        """Construct this object.

        Args:
            span: Span for evaluation.
            risk_off: Risk off ratio. (Max : 1.0)
            risk_on: Risk on ratio. (Max : 1.0)
        """
        self._span = span
        self._risk_off = risk_off
        self._risk_on = risk_on

    def is_risk_off(self, at: datetime, market: Market, weights: npt.ArrayLike) -> bool:
        """Apply risk off strategy to current situation.

        Args:
            at: Current date.
            market: Market data.
            weights: Portfolio weights.

        Returns:
            Deal
        """
        weights = safe_cast(weights)
        historical_data = market.extract(market.get_index() <= pd.to_datetime(at)).get_rate_of_change()
        prices = historical_data.tail(n=self._span).to_numpy() + 1.0
        if prices.size < self._span:
            return False
        prices = np.cumprod(np.dot(prices, weights.T))
        moving_average = np.sum(prices) / self._span
        current_asserts = prices[-1]
        return bool(current_asserts < moving_average)

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
        if self.is_risk_off(at, market, weights):
            new_assets = (np.sum(assets) + cash) * self._risk_off * weights
        else:
            new_assets = (np.sum(assets) + cash) * self._risk_on * weights
        return safe_cast(new_assets)
