"""Rebalance strategies."""
import datetime

import numpy as np
import numpy.typing as npt

from mypo.trigger.base_trigger import BaseTrigger


class AlwaysTrigger(BaseTrigger):
    """Always fire trigger."""

    def __init__(self) -> None:
        """Construct object."""
        super().__init__()

    def is_fire(
        self,
        index: datetime.datetime,
        assets: npt.ArrayLike,
        cash: np.float64,
        weights: npt.ArrayLike,
    ) -> bool:
        """Apply rebalance strategy to current situation.

        Args:
        index: Current date for applying rebalance.
            assets: Current assets for applying rebalance.
            cash: Current cash for applying rebalance.
            weights: Weights of assets.

        Returns:
            Deal
        """
        return True
