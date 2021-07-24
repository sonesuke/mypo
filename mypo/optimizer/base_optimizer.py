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
    _do_re_optimize: bool

    def __init__(self, weights: Optional[npt.ArrayLike], do_re_optimize: bool = False):
        """Construct this object.

        Args:
            weights: Weight.
        """
        if weights is None:
            weights = [1]
        self._weights = safe_cast(weights)
        self._do_re_optimize = do_re_optimize

    def do_re_optimize(self) -> bool:
        """Do re optimize?

        Returns:
            whether re optimize.
        """
        return self._do_re_optimize

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
