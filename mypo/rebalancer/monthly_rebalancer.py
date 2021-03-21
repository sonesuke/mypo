"""Rebalance strategies."""

from mypo.optimizer import BaseOptimizer
from mypo.rebalancer import BaseRebalancer
from mypo.trigger import MonthlyTrigger


class MonthlyRebalancer(BaseRebalancer):
    """Weighted rebalance strategy by monthly applying."""

    def __init__(self, optimizer: BaseOptimizer, do_re_optimize: bool = False, old_month: int = 0) -> None:
        """Construct object.

        Args:
            optimizer: Optimizer.
            do_re_optimize: Re-optimize if it's True. The default is False.
            old_month: Previous month.
        """
        super().__init__(trigger=MonthlyTrigger(old_month), optimizer=optimizer, do_re_optimize=do_re_optimize)
