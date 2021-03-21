"""Rebalance strategies."""

import numpy.typing as npt

from mypo.optimizer import NoOptimizer
from mypo.rebalancer import BaseRebalancer
from mypo.trigger import AlwaysTrigger


class PlainRebalancer(BaseRebalancer):
    """Simple weighted rebalance strategy."""

    def __init__(self, weights: npt.ArrayLike) -> None:
        """Construct object.

        Args:
            weights: Weights.
        """
        super().__init__(trigger=AlwaysTrigger(), optimizer=NoOptimizer(weights))
