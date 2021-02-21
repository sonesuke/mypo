import numpy as np
from .common import safe_cast


class Rebalancer(object):

    def __init__(self):
        pass

    def apply(self, index, assets, cash):
        pass


class PlainRebalancer(Rebalancer):

    def __init__(self, weights):
        self.weights = safe_cast(weights)

    def apply(self, index, assets, cash):
        assets = safe_cast(assets)
        diff = self.weights * np.sum(assets) - assets
        return diff


class MonthlyRebalancer(Rebalancer):

    def __init__(self, weights, old_month=0):
        self.old_month = old_month
        self.weights = safe_cast(weights)

    def apply(self, index, assets, cash):
        assets = safe_cast(assets)
        if self.old_month != index.month:
            self.old_month = index.month
            diff = self.weights * np.sum(assets) - assets
            return diff
        else:
            return assets - assets
