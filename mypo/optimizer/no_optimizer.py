"""Optimizer for weights of portfolio."""
from datetime import datetime
from typing import Optional

import numpy.typing as npt

from mypo import Market
from mypo.optimizer import BaseOptimizer


class NoOptimizer(BaseOptimizer):
    """Base Optimizer."""

    def __init__(self, weights: Optional[npt.ArrayLike] = None):
        """Construct this object.

        Args:
            weights: Weight.
        """
        super().__init__(weights)

    def optimize(self, market: Market, at: datetime) -> None:
        """Optimize weights.

        Args:
            market: Market data.
            at: Current date.
        """
        pass
