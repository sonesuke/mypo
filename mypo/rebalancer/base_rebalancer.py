"""Rebalance strategies."""
import datetime

import numpy as np
import numpy.typing as npt

from mypo.common import safe_cast
from mypo.trigger.base_trigger import BaseTrigger


class BaseRebalancer(object):
    """Interface class of Rebalance stragegy class."""

    _trigger: BaseTrigger

    def __init__(self, trigger: BaseTrigger, weights: npt.ArrayLike) -> None:
        """Construct object.

        Args:
            trigger: Trigger.
            weights: Weight for rebalance.
        """
        self._trigger = trigger
        self._weights = safe_cast(weights)

    def apply(self, index: datetime.datetime, assets: npt.ArrayLike, cash: np.float64) -> np.ndarray:
        """Apply rebalance strategy to current situation.

        Args:
            index: Current date for applying rebalance.
            assets: Current assets for applying rebalance.
            cash: Current cash for applying rebalance.

        Returns:
            Deal
        """
        assets = safe_cast(assets)
        if self._trigger.is_fire(index, assets, cash, self._weights):
            diff = self._rebalance(assets)
        else:
            diff = self._do_nothing(assets)
        if cash < 0 and np.sum(assets + diff) > 0:
            diff = diff - np.abs(cash) * self._weights
        return diff

    def _rebalance(self, assets: np.ndarray) -> np.ndarray:
        diff: np.ndarray = self._weights * np.sum(assets) - assets
        return diff

    def _do_nothing(self, assets: np.ndarray) -> np.ndarray:
        zero: np.ndarray = assets - assets
        return zero
