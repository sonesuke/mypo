"""Optimizer for weights of portfolio."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from mypo.common import safe_cast
from mypo.market import Market
from mypo.optimizer.objective import CovarianceModel, covariance
from mypo.optimizer.optimizer import Optimizer


class MinimumVarianceOptimizer(Optimizer):
    """Minimum variance optimizer."""

    _historical_data: pd.DataFrame
    _span: int

    def __init__(
        self,
        market: Market,
        span: int = 260,
        covariance_model: CovarianceModel = covariance,
        minimum_return: float = None,
    ):
        """
        Construct this object.

        Parameters
        ----------
        market
            Past market stock prices.

        span
            Span for evaluation.
        """
        self._historical_data = market.get_prices()
        self._span = span
        self._covariance_model = covariance_model
        self._minimum_return = minimum_return

    def optimize_weight(self) -> np.ndarray:
        """
        Optimize weights.

        Returns
        -------
        Optimized weights
        """
        prices = self._historical_data.tail(n=self._span).to_numpy()
        Q = self._covariance_model(prices)
        n = len(self._historical_data.columns)
        x = np.ones(n) / n

        def fn(x: np.ndarray, Q: np.ndarray) -> np.float64:
            ret: np.float64 = np.dot(np.dot(x, Q), x.T) / np.max(np.abs(Q))
            return ret

        cons = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]
        if self._minimum_return is not None:
            ret = prices.mean(axis=0)
            daily_risk_free_rate = (1.0 + self._minimum_return) ** (1 / 252) - 1.0
            print(ret)
            print(daily_risk_free_rate)
            cons += [
                {
                    "type": "ineq",
                    "fun": lambda x: np.dot(ret, x) - daily_risk_free_rate,
                }
            ]

        bounds = [[0.0, 1.0] for i in range(n)]
        minout = minimize(
            fn, x, args=(Q), method="SLSQP", bounds=bounds, constraints=cons
        )
        return safe_cast(minout.x)
