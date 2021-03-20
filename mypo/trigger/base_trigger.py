"""Rebalance strategies."""
import datetime

import numpy as np
import numpy.typing as npt


class BaseTrigger(object):
    """Weighted rebalance strategy by monthly applying."""

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
        pass  # pragma: no cover
