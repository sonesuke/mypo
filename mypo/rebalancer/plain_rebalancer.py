"""Rebalance strategies."""
import datetime

import numpy as np
import numpy.typing as npt

from mypo.common import safe_cast
from mypo.rebalancer.rebalancer import Rebalancer


class PlainRebalancer(Rebalancer):
    """Simple weighted rebalance strategy."""

    _weights: np.ndarray

    def __init__(self, weights: npt.ArrayLike) -> None:
        """
        Construct object.

        Parameters
        ----------
        weights
            Weight for applying rebalance.
        """
        super().__init__()
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
        diff: np.ndarray = self._weights * np.sum(assets) - safe_cast(assets)
        return diff
