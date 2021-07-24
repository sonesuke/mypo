"""Optimizer for weights of portfolio."""
from datetime import datetime
from typing import Optional

import numpy as np
import numpy.typing as npt

from mypo import Market
from mypo.optimizer import BaseOptimizer


class NoOptimizer(BaseOptimizer):
    """Base Optimizer."""

    def __init__(self, weights: Optional[npt.ArrayLike] = None, do_re_optimize: bool = False):
        """Construct this object.

        Args:
            weights: Weight.
            do_re_optimize: Do re-optimize.
        """
        super().__init__(weights, do_re_optimize)

    def optimize(self, market: Market, at: datetime) -> np.float64:
        """Optimize weights.

        Args:
            market: Market data.
            at: Current date.
        """
        pass
