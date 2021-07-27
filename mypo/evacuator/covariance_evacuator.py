"""Optimizer for weights of portfolio."""
from datetime import datetime

import numpy as np
import pandas as pd

from mypo import Market
from mypo.common import covariance, safe_cast
from mypo.evacuator import BaseEvacuator


class CovarianceEvacuator(BaseEvacuator):
    """Weighted rebalance strategy by monthly applying."""

    _long_span: int
    _shortspan: int
    _factor: np.float64

    def __init__(self, long_span: int = 260, short_span: int = 20, factor: np.float64 = np.float64(1.0)):
        """Construct this object.

        Args:
            long_span: Long span for covariance.
            short_span: Short span for covariance.
            factor: Risk off ratio. (Max : 1.0)
        """
        self._long_span = long_span
        self._short_span = short_span
        self._factor = factor

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
        historical_data = market.extract(market.get_index() <= pd.to_datetime(at)).get_rate_of_change()
        long_span_prices = historical_data.tail(n=self._long_span).to_numpy() + 1.0
        short_span_prices = historical_data.tail(n=self._short_span).to_numpy() + 1.0
        long_Q = covariance(long_span_prices)
        short_Q = covariance(short_span_prices)

        long_covariance = np.dot(np.dot(weights, long_Q), weights.T)
        short_covariance = np.dot(np.dot(weights, short_Q), weights.T)

        diff_rate = np.min([np.abs(long_covariance - short_covariance) / long_covariance, 1]) * self._factor
        new_assets = (np.sum(assets) + cash) * (1 - diff_rate) * weights

        return safe_cast(new_assets)
