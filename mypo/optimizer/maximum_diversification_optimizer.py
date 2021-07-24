"""Optimizer for weights of portfolio."""

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from mypo.common import safe_cast
from mypo.market import Market
from mypo.optimizer import BaseOptimizer
from mypo.optimizer.objective import covariance


class MaximumDiversificationOptimizer(BaseOptimizer):
    """Minimum variance optimizer."""

    _span: int
    _risk_free_rate: float
    _cost_tolerance: Optional[float]

    def __init__(self, span: int = 260, cost_tolerance: float = 0, do_re_optimize: bool = False):
        """Construct this object.

        Args:
            span: Span for evaluation.
            cost_tolerance: Cost tolerance.
            do_re_optimize: Do re-optimize.
        """
        self._span = span
        self._cost_tolerance = cost_tolerance
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
        Q = covariance(prices)
        Q_diag = np.diag(Q)
        n = Q.shape[0]
        x = np.ones(n) / n

        def fn(x: np.ndarray) -> np.float64:
            ret: np.float64 = -np.dot(x, Q_diag) / np.dot(np.dot(x, Q), x.T)
            return ret

        cons = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
        bounds = [[0.0, 1.0] for i in range(n)]
        minout = minimize(
            fn,
            x,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
        )

        if self._cost_tolerance != 0:
            # Cn = CEo - CEn + TC`|wn - wc| + d(wn - wo)`(wn - wo)
            previous_weights = safe_cast(self._weights)
            optimized_weights = safe_cast(minout.x)
            optimized_value = np.float64(minout.fun)

            def cn(
                x: np.ndarray,
            ) -> np.float64:
                wo_diff = x - optimized_weights
                wp_diff = x - previous_weights
                ret: np.float64 = (
                    optimized_value
                    - fn(x)
                    + self._cost_tolerance * np.dot(wp_diff, wp_diff.T)
                    + np.dot(wo_diff, wo_diff.T)
                )
                return ret

            minout = minimize(
                cn,
                previous_weights,
                method="SLSQP",
                bounds=bounds,
                constraints=cons,
                tol=1e-6 * np.max(np.abs(Q)),
            )
        self._weights = safe_cast(minout.x)
        return np.float64(minout.fun)
