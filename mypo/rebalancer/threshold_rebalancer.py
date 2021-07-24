"""Rebalance strategies."""

import numpy as np

from mypo.optimizer import BaseOptimizer
from mypo.rebalancer import BaseRebalancer
from mypo.trigger import ThresholdTrigger


class ThresholdRebalancer(BaseRebalancer):
    """Weighted rebalance strategy by monthly applying."""

    _threshold: np.float64

    def __init__(self, optimizer: BaseOptimizer, threshold: np.float64 = np.float64(0.05)) -> None:
        """Construct object.

        Args:
            optimizer: Optimizer.
            threshold: Threshold of fire.
        """
        super().__init__(trigger=ThresholdTrigger(threshold=threshold), optimizer=optimizer)
