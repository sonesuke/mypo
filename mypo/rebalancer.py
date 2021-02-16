import numpy as np


class Rebalancer(object):

    def __init__(self):
        pass

    def apply(self, index, assets, cash):
        pass


class PlainRebalancer(Rebalancer):

    def __init__(self, weights):
        self.weights = weights

    def apply(self, index, assets, cash):
        diff = self.weights * np.sum(assets) - assets
        return diff


class MonthlyRebalancer(Rebalancer):

    def __init__(self, weights, old_month=0):
        self.old_month = old_month
        self.weights = weights

    def apply(self, index, assets, cash):
        if self.old_month != index.month:
            self.old_month = index.month
            diff = self.weights * np.sum(assets) - assets
            return diff
        else:
            return assets - assets
