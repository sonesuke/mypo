"""Rebalance strategies."""

import numpy as np
import numpy.typing as npt

from mypo.common import safe_cast
from mypo.rebalancer.base_rebalancer import BaseRebalancer
from mypo.trigger.monthly_trigger import MonthlyTrigger


class MonthlyRebalancer(BaseRebalancer):
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
        super().__init__(trigger=MonthlyTrigger(old_month), weights=weights)
        self._old_month = old_month
        self._weights = safe_cast(weights)
