import numpy as np


class Rebalancer(object):

    def __init__(self, weights):
        self.weights = weights

    def apply(self, assets):
        diff = self.weights * np.sum(assets) - assets
        return diff
