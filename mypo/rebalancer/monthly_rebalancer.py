"""Rebalance strategies."""
import datetime

import numpy as np
import numpy.typing as npt

from mypo.common import safe_cast
from mypo.rebalancer.rebalancer import Rebalancer


class MonthlyRebalancer(Rebalancer):
    """Weighted rebalance strategy by monthly applying."""

    _old_month: int
    _weights: np.ndarray

    def __init__(self, weights: npt.ArrayLike, old_month: int = 0) -> None:
        """
        Construct object.

        Parameters
        ----------
        weights
            Weight for applying rebalance.

        old_month
            Previous month.
        """
        super().__init__()
        self._old_month = old_month
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
        if self._old_month != index.month:
            self._old_month = index.month
            diff: np.ndarray = self._weights * np.sum(assets) - assets
            return diff
        else:
            zero: np.ndarray = assets - assets
            return zero
