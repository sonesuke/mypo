"""Rebalance strategies."""
from datetime import datetime

import numpy as np
import numpy.typing as npt

from mypo import Market
from mypo.common import safe_cast
from mypo.optimizer import BaseOptimizer, NoOptimizer
from mypo.trigger import BaseTrigger


class BaseRebalancer(object):
    """Interface class of Rebalance strategy class."""

    _trigger: BaseTrigger
    _optimizer: BaseOptimizer
    _do_re_optimize: bool

    def __init__(
        self, trigger: BaseTrigger, optimizer: BaseOptimizer = NoOptimizer(), do_re_optimize: bool = False
    ) -> None:
        """Construct object.

        Args:
            trigger: Trigger.
            optimizer: Optimizer.
            do_re_optimize: Re-optimize if it's True. The default is False.
        """
        self._trigger = trigger
        self._optimizer = optimizer
        self._do_re_optimize = do_re_optimize

    def apply(self, at: datetime, market: Market, assets: npt.ArrayLike, cash: np.float64) -> np.ndarray:
        """Apply rebalance strategy to current situation.

        Args:
            at: Current date for applying rebalance.
            market: Market data for all spans.
            assets: Current assets for applying rebalance.
            cash: Current cash for applying rebalance.

        Returns:
            Deal
        """
        assets = safe_cast(assets)
        if self._trigger.is_fire(at, assets, cash, self._optimizer.get_weights()):
            if self._do_re_optimize:
                self._optimizer.optimize(market, at)
            diff = self._rebalance(assets)
        else:
            diff = self._do_nothing(assets)
        if cash < 0 and np.sum(assets + diff) > 0:
            diff = diff - np.abs(cash) * self._optimizer.get_weights()
        return diff

    def _rebalance(self, assets: np.ndarray) -> np.ndarray:
        diff: np.ndarray = self._optimizer.get_weights() * np.sum(assets) - assets
        return diff

    def _do_nothing(self, assets: np.ndarray) -> np.ndarray:
        zero: np.ndarray = assets - assets
        return zero

    def get_optimizer(self) -> BaseOptimizer:
        """Get optimizer.

        Returns:
            Optimizer
        """
        return self._optimizer
