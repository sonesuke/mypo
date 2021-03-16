"""Rebalance strategies."""

import numpy as np
import numpy.typing as npt

from mypo.rebalancer.base_rebalancer import BaseRebalancer
from mypo.trigger.threshold_trigger import ThresholdTrigger


class ThresholdRebalancer(BaseRebalancer):
    """Weighted rebalance strategy by monthly applying."""

    _threshold: np.float64

    def __init__(
        self, weights: npt.ArrayLike, threshold: np.float64 = np.float64(0.05)
    ) -> None:
        """
        Construct object.

        Parameters
        ----------
        weights
            Weight for applying rebalance.

        threshold
            Threshold of fire.

        """
        super().__init__(trigger=ThresholdTrigger(threshold=threshold), weights=weights)
