"""Rebalance strategies."""
import numpy.typing as npt

from mypo.rebalancer.base_rebalancer import BaseRebalancer
from mypo.trigger.always_trigger import AlwaysTrigger


class PlainRebalancer(BaseRebalancer):
    """Simple weighted rebalance strategy."""

    def __init__(self, weights: npt.ArrayLike) -> None:
        """
        Construct object.

        Parameters
        ----------
        weights
            Weight for applying rebalance.
        """
        super().__init__(trigger=AlwaysTrigger(), weights=weights)
