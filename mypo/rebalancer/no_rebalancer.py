"""Rebalance strategies."""

from mypo.optimizer import NoOptimizer
from mypo.rebalancer import BaseRebalancer
from mypo.trigger import NoTrigger


class NoRebalancer(BaseRebalancer):
    """Simple weighted rebalance strategy."""

    def __init__(self) -> None:
        """Construct object."""
        super().__init__(trigger=NoTrigger(), optimizer=NoOptimizer())
