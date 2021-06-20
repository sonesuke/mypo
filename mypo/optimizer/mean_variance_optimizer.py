"""Optimizer for weights of portfolio."""

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from mypo.common import safe_cast
from mypo.market import Market
from mypo.optimizer.base_optimizer import BaseOptimizer
from mypo.optimizer.objective import covariance


class MeanVarianceOptimizer(BaseOptimizer):
    """Mean variance optimizer."""

    _span: int
    _risk_tolerance: Optional[float]

    def __init__(
        self,
        span: int = 260,
        risk_tolerance: float = 0,
    ):
        """Construct this object.

        Args:
            span: Span for evaluation.
            risk_tolerance: Risk tolerance.
        """
        self._span = span
        self._risk_tolerance = risk_tolerance
        super().__init__([1])

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
        Q = covariance(prices)
        R = prices.mean(axis=0)
        n = len(historical_data.columns)
        x = np.ones(n) / n

        def fn(x: np.ndarray, Q: np.ndarray) -> np.float64:
            ret: np.float64 = np.dot(np.dot(x, Q), x.T) - self._risk_tolerance * np.dot(R, x)
            return ret

        cons = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]

        bounds = [[0.0, 1.0] for i in range(n)]
        minout = minimize(
            fn, x, args=(Q), method="SLSQP", bounds=bounds, constraints=cons, tol=1e-6 * np.max(np.abs(Q))
        )
        self._weights = safe_cast(minout.x)
        return np.float64(minout.fun)
