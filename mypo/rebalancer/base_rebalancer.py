"""Rebalance strategies."""
from datetime import datetime

import numpy as np
import numpy.typing as npt

from mypo import Market
from mypo.common import safe_cast
from mypo.evacuator import BaseEvacuator, NoEvacuator
from mypo.optimizer import BaseOptimizer, NoOptimizer
from mypo.trigger import BaseTrigger, NoTrigger


class BaseRebalancer(object):
    """Interface class of Rebalance strategy class."""

    _trigger: BaseTrigger
    _optimizer: BaseOptimizer
    _evacuate_trigger: BaseTrigger
    _evacuator: BaseEvacuator
    _first_time: bool

    def __init__(
        self,
        trigger: BaseTrigger,
        optimizer: BaseOptimizer = NoOptimizer(),
        evacuate_trigger: BaseTrigger = NoTrigger(),
        evacuator: BaseEvacuator = NoEvacuator(),
    ) -> None:
        """Construct object.

        Args:
            trigger: Trigger.
            optimizer: Optimizer.
            evacuate_trigger: Trigger for evacuator.
            evacuator: Evacuator.
        """
        self._trigger = trigger
        self._optimizer = optimizer
        self._evacuate_trigger = evacuate_trigger
        self._evacuator = evacuator
        self._first_time = True

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
        new_assets = assets

        # Weights
        if self._trigger.is_fire(at, market, assets, cash, self._optimizer.get_weights()):
            if self._optimizer.do_re_optimize() or self._first_time:
                self._first_time = False
                self._optimizer.optimize(market, at)
            new_assets = self._optimizer.get_weights() * np.sum(assets)

        # Evacuate from market crush
        if self._evacuate_trigger.is_fire(at, market, new_assets, cash, self._optimizer.get_weights()):
            new_assets = self._evacuator.evacuate(at, market, new_assets, cash, self._optimizer.get_weights())

        # Adjust cash
        if cash < 0 and np.sum(new_assets) > 0:
            new_assets = new_assets - np.abs(cash) * self._optimizer.get_weights()

        return safe_cast(new_assets - assets)

    def get_optimizer(self) -> BaseOptimizer:
        """Get optimizer.

        Returns:
            Optimizer
        """
        return self._optimizer
