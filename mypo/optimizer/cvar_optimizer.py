"""Optimizer for weights of portfolio."""
from datetime import datetime
from typing import List, Optional

import numpy as np
from scipy.optimize import minimize

from mypo.common import safe_cast
from mypo.market import Market
from mypo.optimizer.base_optimizer import BaseOptimizer
from mypo.sampler import Sampler


class CVaROptimizer(BaseOptimizer):
    """Minimum variance optimizer."""

    _span: int
    _scenarios: int
    _sampler: Optional[Sampler]

    def __init__(self, span: int = 260, scenarios: int = 10, sampler: Optional[Sampler] = None):
        """Construct this object.

        Args:
            sampler: Sampler.
            span: Span for evaluation.
            scenarios: Count of scenarios.
        """
        self._span = span
        self._scenarios = scenarios
        self._sampler = sampler
        super().__init__()

    def optimize(self, market: Market, at: datetime) -> None:
        """Optimize weights.

        Args:
            market: Past market stock prices.
            at: Current date.

        Returns:
            Optimized weights
        """
        historical_data = market.extract(market.get_index() < at).tail(self._span)
        sampler = Sampler(market=historical_data, scenarios=self._scenarios) if self._sampler is None else self._sampler
        samples = sampler.sample(200, self._span)
        scenarios = [sample.to_numpy() for sample in samples]

        n = len(historical_data.get_tickers())
        x = np.ones(n) / n
        take_bad_scenarios = int(self._scenarios * 0.05 + 1)

        def fn(x: np.ndarray, scenarios: List[np.ndarray]) -> np.float64:
            ret = []
            for scenario in scenarios:
                assets = np.ones(n)
                for i in range(self._span):
                    assets = (1.0 + scenario[i]) * assets
                    assets = x * np.sum(assets)
                ret += [np.sum(assets)]
            return np.float64(np.mean(np.array(sorted(ret)[:take_bad_scenarios])))

        cons = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]
        bounds = [[0.0, 1.0] for i in range(n)]
        minout = minimize(fn, x, args=(scenarios), method="SLSQP", bounds=bounds, constraints=cons)
        self._weights = safe_cast(minout.x)
