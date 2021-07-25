"""Optimizer for weights of portfolio."""

from datetime import datetime

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from mypo.common import safe_cast, sharpe_ratio
from mypo.market import Market
from mypo.optimizer import BaseOptimizer


class SharpeRatioOptimizer(BaseOptimizer):
    """Minimum variance optimizer."""

    _span: int
    _risk_free_rate: float

    def __init__(self, risk_free_rate: float = 0.02, span: int = 260, do_re_optimize: bool = False):
        """Construct this object.

        Args:
            risk_free_rate: Risk free rate
            span: Span for evaluation.
            do_re_optimize: Do re-optimize.
        """
        self._risk_free_rate = risk_free_rate
        self._span = span
        super().__init__([1], do_re_optimize)

    def optimize(self, market: Market, at: datetime) -> np.float64:
        """Optimize weights.

        Args:
            market: Past market stock prices.
            at: Current date.

        Returns:
            Optimized weights
        """
        historical_data = market.extract(market.get_index() <= pd.to_datetime(at)).get_rate_of_change()
        prices = historical_data.tail(n=self._span).to_numpy()
        n = prices.shape[1]
        x = np.ones(n) / n

        def fn(x: np.ndarray) -> np.float64:
            return -sharpe_ratio(np.dot(prices, x.T), np.float64(self._risk_free_rate))

        cons = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
        bounds = [[0.0, 1.0] for i in range(n)]
        minout = minimize(
            fn,
            x,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
        )
        self._weights = safe_cast(minout.x)
        return np.float64(minout.fun)
