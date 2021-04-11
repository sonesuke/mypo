"""Optimizer for weights of portfolio."""

from datetime import datetime
from typing import List, Optional

import numpy as np
from scipy.optimize import minimize

from mypo.common import safe_cast
from mypo.market import Market
from mypo.optimizer import BaseOptimizer


class RiskParityOptimizer(BaseOptimizer):
    """Minimum variance optimizer."""

    _span: int
    _risk_target: Optional[np.ndarray]

    def __init__(self, span: int = 260, risk_target: Optional[List[float]] = None):
        """Construct this object.

        Args:
            span: Span for evaluation.
            risk_target: Risk target.
        """
        self._span = span
        self._risk_target = safe_cast(risk_target) if risk_target is not None else None
        super().__init__([1])

    def optimize(self, market: Market, at: datetime) -> np.float64:
        """Optimize weights.

        Args:
            market: Past market stock prices.
            at: Current date.

        Returns:
            Optimized weights
        """
        historical_data = market.extract(market.get_index() < at).get_rate_of_change()
        prices = historical_data.tail(n=self._span).to_numpy()
        Q = np.cov(prices.T)
        n = Q.shape[0]
        x = np.ones(n) / n
        if self._risk_target is None:
            risk_t = x
        else:
            risk_t = self._risk_target

        def fn(x: np.ndarray, Q: np.ndarray, risk_t: np.ndarray) -> np.float64:
            sigma = np.sqrt(np.dot(np.dot(x, Q), x.T))
            RC = np.multiply(np.dot(Q, x), x.T) / sigma
            risk_target = sigma * risk_t
            return np.float64(sum(np.sqrt((RC - risk_target) ** 2)))

        cons = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
        bounds = [[0.0, 1.0] for i in range(n)]
        minout = minimize(
            fn,
            x,
            args=(Q, risk_t),
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
        )
        self._weights = safe_cast(minout.x)
        return np.float64(minout.fun)
