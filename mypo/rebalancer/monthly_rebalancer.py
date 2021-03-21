"""Rebalance strategies."""

import numpy as np

from mypo.optimizer import BaseOptimizer
from mypo.rebalancer import BaseRebalancer
from mypo.trigger import MonthlyTrigger


class MonthlyRebalancer(BaseRebalancer):
    """Weighted rebalance strategy by monthly applying."""

    _old_month: int
    _weights: np.ndarray

    def __init__(self, optimizer: BaseOptimizer, old_month: int = 0) -> None:
        """Construct object.

        Args:
            optimizer: Optimizer.
            old_month: Previous month.
        """
        super().__init__(trigger=MonthlyTrigger(old_month), optimizer=optimizer)
        self._old_month = old_month
