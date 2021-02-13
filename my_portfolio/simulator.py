import numpy as np
from .rebalance import rebalance
from .rebalance import calc_capital_gain_tax
from .rebalance import calc_income_gain_tax
from .rebalance import calc_fee


class Simulator(object):

    def __init__(
            self,
            assets,
            weights,
            cash,
            spending
            ):
        self.initial_assets = assets
        self.assets = assets
        self.weights = weights
        self.cash = cash
        self.spending = spending
        self.tax_rate = 0.2
        self.fee_rate = 0.005

    def total_assets(self):
        return np.sum(self.assets) + self.cash

    def apply(self, market, price_dividends_yield, expense_ratio):

        self.assets = self.assets * market
        diff = rebalance(self.assets, self.weights)

        # process of capital gain
        capital_gain_tax = calc_capital_gain_tax(
            self.initial_assets,
            self.assets,
            diff,
            self.tax_rate)
        self.cash -= capital_gain_tax
        fee = calc_fee(
            diff,
            self.fee_rate)
        self.cash -= fee
        self.assets += diff

        # process of income gain
        dividends = np.sum(self.assets * price_dividends_yield)

        income_gain_tax = calc_income_gain_tax(
            self.assets,
            price_dividends_yield,
            self.tax_rate)
        self.cash += dividends
        self.cash -= income_gain_tax

        # process of others
        self.assets = (1.0 - expense_ratio) * self.assets
