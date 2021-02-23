import datetime

import numpy as np
import numpy.typing as npt

from .common import safe_cast


class Rebalancer(object):
    def __init__(self) -> None:
        pass

    def apply(self, index: datetime.datetime, assets: npt.ArrayLike, cash: np.float64) -> np.ndarray:
        pass


class PlainRebalancer(Rebalancer):
    weights: np.ndarray

    def __init__(self, weights: npt.ArrayLike) -> None:
        self.weights = safe_cast(weights)

    def apply(self, index: datetime.datetime, assets: npt.ArrayLike, cash: np.float64) -> np.ndarray:
        diff: np.ndarray = self.weights * np.sum(assets) - safe_cast(assets)
        return diff


class MonthlyRebalancer(Rebalancer):
    old_month: int

    def __init__(self, weights: npt.ArrayLike, old_month: int = 0) -> None:
        self.old_month = old_month
        self.weights = safe_cast(weights)

    def apply(self, index: datetime.datetime, assets: npt.ArrayLike, cash: np.float64) -> np.ndarray:
        assets = safe_cast(assets)
        if self.old_month != index.month:
            self.old_month = index.month
            diff: np.ndarray = self.weights * np.sum(assets) - assets
            return diff
        else:
            zero: np.ndarray = assets - assets
            return zero
