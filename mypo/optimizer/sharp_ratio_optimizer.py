"""Optimizer for weights of portfolio."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from mypo.common import safe_cast
from mypo.market import Market
from mypo.optimizer import Optimizer
from mypo.optimizer.objective import sharp_ratio


class SharpRatioOptimizer(Optimizer):
    """Minimum variance optimizer."""

    _historical_data: pd.DataFrame
    _span: int

    def __init__(
        self,
        market: Market,
        risk_free_rate: np.float64 = np.float64(0.02),
        span: int = 260,
    ):
        """
        Construct this object.

        Parameters
        ----------
        market
            Past market stock prices.
        risk_free_rate
            Risk free rate

        span
            Span for evaluation.
        """
        self._historical_data = market.get_prices()
        self._risk_free_rate = risk_free_rate
        self._span = span

    def optimize_weight(self) -> np.ndarray:
        """
        Optimize weights.

        Returns
        -------
        Optimized weights
        """
        prices = self._historical_data.tail(n=self._span).to_numpy()
        Q = np.cov(prices.T)
        R = prices.mean(axis=0).T
        n = Q.shape[0]
        x = np.ones(n) / n
        daily_risk_free_rate = (1.0 + self._risk_free_rate) ** (1 / 252) - 1.0

        def fn(
            x: np.ndarray,
            R: np.ndarray,
            Q: np.ndarray,
            daily_risk_free_rate: np.float64,
        ) -> np.float64:
            return -sharp_ratio(
                np.dot(x, R), np.dot(np.dot(x, Q), x.T), daily_risk_free_rate
            )

        cons = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]
        bounds = [[0.0, 1.0] for i in range(n)]
        minout = minimize(
            fn,
            x,
            args=(R, Q, daily_risk_free_rate),
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
        )
        return safe_cast(minout.x)
