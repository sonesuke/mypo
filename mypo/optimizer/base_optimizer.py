"""Optimizer for weights of portfolio."""
from datetime import datetime
from typing import Optional

import numpy as np
import numpy.typing as npt

from mypo import Market
from mypo.common import safe_cast


class BaseOptimizer(object):
    """Base Optimizer."""

    _weights: np.ndarray

    def __init__(self, weights: Optional[npt.ArrayLike]):
        """Construct this object.

        Args:
            weights: Weight.
        """
        if weights is None:
            weights = [1]
        self._weights = safe_cast(weights)

    def get_weights(self) -> np.ndarray:
        """Get weights.

        Returns:
            Weights.
        """
        return self._weights

    def optimize(self, market: Market, at: datetime) -> np.float64:
        """Optimize weights.

        Args:
            market: Market data.
            at: Current date.
        """
        pass  # pragma: no cover
