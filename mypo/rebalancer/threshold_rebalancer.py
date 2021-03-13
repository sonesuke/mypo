"""Rebalance strategies."""
import datetime

import numpy as np
import numpy.typing as npt

from mypo.common import safe_cast
from mypo.rebalancer.rebalancer import Rebalancer


class ThresholdRebalancer(Rebalancer):
    """Weighted rebalance strategy by monthly applying."""

    _threshold: np.float64
    _weights: np.ndarray

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
        super().__init__()
        self._threshold = threshold
        self._weights = safe_cast(weights)

    def apply(
        self, index: datetime.datetime, assets: npt.ArrayLike, cash: np.float64
    ) -> np.ndarray:
        """
        Apply rebalance strategy to current situation.

        Parameters
        ----------
        index
            Current date for applying rebalance.

        assets
            Current assets for applying rebalance.

        cash
            Current cash for applying rebalance.

        Returns
        -------
        Deal
        """
        assets = safe_cast(assets)
        if np.max(np.abs(assets / np.sum(assets) - self._weights)) > self._threshold:
            diff: np.ndarray = self._weights * np.sum(assets) - assets
            return diff
        else:
            zero: np.ndarray = assets - assets
            return zero
