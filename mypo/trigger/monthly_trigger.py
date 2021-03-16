"""Rebalance strategies."""
import datetime

import numpy as np
import numpy.typing as npt

from mypo.common import safe_cast
from mypo.trigger.base_trigger import BaseTrigger


class MonthlyTrigger(BaseTrigger):
    """Weighted rebalance strategy by monthly applying."""

    _old_month: int
    _weights: np.ndarray

    def __init__(self, old_month: int = 0) -> None:
        """
        Construct object.

        Parameters
        ----------
        old_month
            Previous month.
        """
        super().__init__()
        self._old_month = old_month

    def is_fire(
        self,
        index: datetime.datetime,
        assets: npt.ArrayLike,
        cash: np.float64,
        weights: npt.ArrayLike,
    ) -> bool:
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

        weights
            Weights of assets.

        Returns
        -------
        Deal
        """
        assets = safe_cast(assets)
        if self._old_month != index.month:
            self._old_month = index.month
            return True
        else:
            return False
