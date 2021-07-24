"""Optimizer for weights of portfolio."""
from datetime import datetime

import numpy as np

from mypo import Market


class BaseEvacuator(object):
    """Weighted rebalance strategy by monthly applying."""

    def evacuate(
        self, at: datetime, market: Market, assets: np.ndarray, cash: np.float64, weights: np.ndarray
    ) -> np.ndarray:
        """Apply risk off strategy to current situation.

        Args:
            at: Current date.
            market: Market data.
            assets: Assets.
            cash: Cash.
            weights: Weights.

        Returns:
            Trades for risk off.
        """
        pass  # pragma: no cover
