"""Optimizer for weights of portfolio."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..common import safe_cast
from ..market import Market


class Optimizer(object):
    """Base Optimizer."""

    pass


class MinimumVarianceOptimizer(Optimizer):
    """Minimum variance optimizer."""

    _historical_data: pd.DataFrame
    _span: int

    def __init__(self, market: Market, span: int = 260):
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

    def optimize_weight(self, minimum_return: float = None) -> np.ndarray:
        """
        Optimize weights.

        Parameters
        ----------
        minimum_return
            Add minimum return constraints.

        Returns
        -------
        Optimized weights
        """
        prices = self._historical_data.tail(n=self._span).to_numpy()
        Q = np.cov(prices.T)
        Q = Q / np.max(np.abs(Q))
        n = Q.shape[0]
        x = np.ones(n) / n

        def fn(x: np.ndarray, Q: np.ndarray) -> np.float64:
            ret: np.float64 = np.dot(np.dot(x, Q), x.T)
            return ret

        cons = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]
        if minimum_return is not None:
            ret = np.prod(prices, axis=0)
            print(ret)
            cons += [
                {
                    "type": "ineq",
                    "fun": lambda x: np.dot(ret, x) - (1.0 + minimum_return),
                }
            ]

        bounds = [[0.0, 1.0] for i in range(n)]
        minout = minimize(
            fn, x, args=(Q), method="SLSQP", bounds=bounds, constraints=cons
        )
        return safe_cast(minout.x)


class SharpRatioOptimizer(Optimizer):
    """Minimum variance optimizer."""

    _historical_data: pd.DataFrame
    _span: int

    def __init__(self, market: Market, span: int = 260):
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

    def optimize_weight(
        self, risk_free_rate: np.float64 = np.float64(0.02)
    ) -> np.ndarray:
        """
        Optimize weights.

        Parameters
        ----------
        risk_free_rate
            Risk free rate

        Returns
        -------
        Optimized weights
        """
        prices = self._historical_data.tail(n=self._span).to_numpy()
        Q = np.cov(prices.T)
        R = prices.prod(axis=0).T
        Q = Q / np.max(np.abs(Q))
        n = Q.shape[0]
        x = np.ones(n) / n

        def fn(
            x: np.ndarray, R: np.ndarray, Q: np.ndarray, risk_free_rate: np.float64
        ) -> np.float64:
            ret: np.float64 = (np.dot(x, R) - (1.0 + risk_free_rate)) / np.dot(
                np.dot(x, Q), x.T
            )
            return -ret

        cons = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]
        bounds = [[0.0, 1.0] for i in range(n)]
        minout = minimize(
            fn,
            x,
            args=(R, Q, risk_free_rate),
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
        )
        return safe_cast(minout.x)
