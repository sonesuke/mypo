"""Rebalance strategies."""
import datetime

import numpy as np
import numpy.typing as npt

from .common import safe_cast


class Rebalancer(object):
    """Interface class of Rebalance stragegy class."""

    def __init__(self) -> None:
        pass

    def apply(self, index: datetime.datetime, assets: npt.ArrayLike, cash: np.float64) -> np.ndarray:
        """
        Apply rebalance strategy to current situation.

        Parameters
        ----------
        index
            Current date for applying rebalance.

        assets
            Current assets for applying rebalance.

        cash
            Current cash for applying rebalance.

        Returns
        -------
        Deal
        """
        pass


class PlainRebalancer(Rebalancer):
    """Simple weighted rebalance strategy."""

    _weights: np.ndarray

    def __init__(self, weights: npt.ArrayLike) -> None:
        """
        Construct object.

        Parameters
        ----------
        weights
            Weight for applying rebalance.
        """
        super().__init__()
        self._weights = safe_cast(weights)

    def apply(self, index: datetime.datetime, assets: npt.ArrayLike, cash: np.float64) -> np.ndarray:
        """
        Apply rebalance strategy to current situation.

        Parameters
        ----------
        index
            Current date for applying rebalance.

        assets
            Current assets for applying rebalance.

        cash
            Current cash for applying rebalance.

        Returns
        -------
        Deal
        """
        diff: np.ndarray = self._weights * np.sum(assets) - safe_cast(assets)
        return diff


class MonthlyRebalancer(Rebalancer):
    """Weighted rebalance strategy by monthly applying."""

    old_month: int

    def __init__(self, weights: npt.ArrayLike, old_month: int = 0) -> None:
        """
        Construct object.

        Parameters
        ----------
        weights
            Weight for applying rebalance.

        old_month
            Previous month.
        """
        super().__init__()
        self.old_month = old_month
        self.weights = safe_cast(weights)

    def apply(self, index: datetime.datetime, assets: npt.ArrayLike, cash: np.float64) -> np.ndarray:
        """
        Apply rebalance strategy to current situation.

        Parameters
        ----------
        index
            Current date for applying rebalance.

        assets
            Current assets for applying rebalance.

        cash
            Current cash for applying rebalance.

        Returns
        -------
        Deal
        """
        assets = safe_cast(assets)
        if self.old_month != index.month:
            self.old_month = index.month
            diff: np.ndarray = self.weights * np.sum(assets) - assets
            return diff
        else:
            zero: np.ndarray = assets - assets
            return zero
