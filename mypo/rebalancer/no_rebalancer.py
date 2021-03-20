"""Rebalance strategies."""
import numpy as np

from mypo.rebalancer.base_rebalancer import BaseRebalancer
from mypo.trigger.no_trigger import NoTrigger


class NoRebalancer(BaseRebalancer):
    """Simple weighted rebalance strategy."""

    def __init__(self) -> None:
        """Construct object."""
        super().__init__(trigger=NoTrigger(), weights=np.zeros(1))
