"""Optimizer for weights of portfolio."""
from datetime import datetime
from typing import Optional

import numpy as np
from scipy.optimize import minimize

from mypo.common import safe_cast
from mypo.market import Market
from mypo.optimizer.base_optimizer import BaseOptimizer
from mypo.sampler import Sampler


class CVaROptimizer(BaseOptimizer):
    """Minimum variance optimizer."""

    _span: int
    _beta: float
    _samples: int
    _sampler: Optional[Sampler]

    def __init__(self, span: int = 260, beta: float = 0.05, samples: int = 10, sampler: Optional[Sampler] = None):
        """Construct this object.

        Args:
            sampler: Sampler.
            span: Span for evaluation.
            beta: Confidence.
            samples: Count of scenarios.
            sampler: Sampler.
        """
        self._span = span
        self._beta = beta
        self._samples = samples
        self._sampler = sampler
        super().__init__([1])

    def optimize(self, market: Market, at: datetime) -> np.float64:
        """Optimize weights.

        Args:
            market: Past market stock prices.
            at: Current date.

        Returns:
            Optimized weights
        """
        historical_data = market.extract(market.get_index() < at).tail(self._span)
        sampler = Sampler(market=historical_data, samples=self._samples) if self._sampler is None else self._sampler
        sample = sampler.sample(self._span).to_numpy()

        n = len(historical_data.get_tickers())
        x = np.ones(n) / n

        def fn(x: np.ndarray, sequence: np.ndarray) -> np.float64:
            r = np.dot(sequence, x.T)
            return -np.float64(np.mean(np.where(r < np.quantile(r, self._beta), r, 0)))

        cons = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
        bounds = [[0.0, 1.0] for i in range(n)]
        minout = minimize(
            fn, x, args=(sample), method="SLSQP", bounds=bounds, constraints=cons, tol=1e-6 * np.max(sample)
        )
        self._weights = safe_cast(minout.x)
        return np.float64(minout.fun)
