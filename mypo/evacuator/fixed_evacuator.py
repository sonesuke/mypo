"""Optimizer for weights of portfolio."""
from datetime import datetime
from typing import Optional

import numpy as np

from mypo import Market
from mypo.common import safe_cast
from mypo.evacuator import BaseEvacuator


class FixedEvacuator(BaseEvacuator):
    """Weighted rebalance strategy by monthly applying."""

    _level: Optional[np.float64]
    _ratio: Optional[np.float64]

    def __init__(self, ratio: Optional[float] = None, level: Optional[float] = None):
        """Construct this object.

        Args:
            ratio: ratio.
            level: level.
        """
        self._level = np.float64(level) if level is not None else None
        self._ratio = np.float64(ratio) if ratio is not None else None
        if ratio is None and level is None:
            raise ValueError("You cannot specify both ratio and level at same time.")  # pragma: no cover
        elif ratio is not None and level is not None:
            raise ValueError("You have to specify ratio or level.")  # pragma: no cover

    def evacuate(
        self, at: datetime, market: Market, assets: np.ndarray, cash: np.float64, weights: np.ndarray
    ) -> np.ndarray:
        """Apply risk off strategy to current situation.

        Args:
            at: Current date.
            market: Market data.
            assets: Assets
            cash: Cash
            weights: Weights.
        Returns:
            New assets.
        """
        weights = safe_cast(weights)

        total_assets = np.sum(assets)
        if self._ratio is not None:
            evacuation = max((total_assets + cash) * self._ratio, 0)
        else:
            evacuation = self._level

        diff = cash - evacuation
        new_assets = assets + np.dot(diff, weights)
        return safe_cast(new_assets)
