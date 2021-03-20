"""Rebalance strategies."""
import datetime

import numpy as np
import numpy.typing as npt

from mypo.common import safe_cast
from mypo.trigger.base_trigger import BaseTrigger


class ThresholdTrigger(BaseTrigger):
    """Weighted rebalance strategy by monthly applying."""

    _threshold: np.float64
    _weights: np.ndarray

    def __init__(self, threshold: np.float64 = np.float64(0.05)) -> None:
        """Construct object.

        Args:
            threshold: Threshold of fire.
        """
        self._threshold = threshold

    def is_fire(
        self,
        index: datetime.datetime,
        assets: npt.ArrayLike,
        cash: np.float64,
        weights: npt.ArrayLike,
    ) -> bool:
        """Apply rebalance strategy to current situation.

        Args:
            index: Current date for applying rebalance.
            assets: Current assets for applying rebalance.
            cash: Current cash for applying rebalance.
            weights: Weights of assets.

        Returns:
            Deal
        """
        assets = safe_cast(assets)
        if np.max(np.abs(assets / np.sum(assets) - weights)) > self._threshold:
            return True
        else:
            return False
