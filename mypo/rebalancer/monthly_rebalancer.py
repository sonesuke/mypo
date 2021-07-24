"""Rebalance strategies."""
from mypo.evacuator import BaseEvacuator, NoEvacuator
from mypo.optimizer import BaseOptimizer
from mypo.rebalancer import BaseRebalancer
from mypo.trigger import AlwaysTrigger, BaseTrigger, MonthlyTrigger


class MonthlyRebalancer(BaseRebalancer):
    """Weighted rebalance strategy by monthly applying."""

    def __init__(
        self,
        optimizer: BaseOptimizer,
        evacuate_trigger: BaseTrigger = AlwaysTrigger(),
        evacuator: BaseEvacuator = NoEvacuator(),
        old_month: int = 0,
    ) -> None:
        """Construct object.

        Args:
            optimizer: Optimizer.
            evacuate_trigger: Evacuate trigger.
            evacuator: Evacuator.
            old_month: Previous month.
        """
        super().__init__(
            trigger=MonthlyTrigger(old_month),
            optimizer=optimizer,
            evacuate_trigger=evacuate_trigger,
            evacuator=evacuator,
        )
