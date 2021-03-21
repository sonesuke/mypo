"""Optimizer for weights of portfolio."""
import numpy.typing as npt

from mypo import Market
from mypo.optimizer import BaseOptimizer


class NoOptimizer(BaseOptimizer):
    """Base Optimizer."""

    def __init__(self, weights: npt.ArrayLike = None):
        """Construct this object.

        Args:
            weights: Weight.
        """
        super().__init__(weights)

    def optimize(self, market: Market) -> None:
        """Optimize weights.

        Args:
            market: Market data.
        """
        pass
