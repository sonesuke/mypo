import numpy as np
from .common import calc_capital_gain_tax
from .common import calc_income_gain_tax
from .common import calc_fee
from .common import safe_cast


class Runner(object):

    def __init__(
            self,
            assets,
            rebalancer,
            cash,
            spending
            ):
        self.assets = safe_cast(assets)
        self.initial_assets = self.assets
        self.rebalancer = rebalancer
        self.cash = cash
        self.spending = spending
        self.tax_rate = 0.2
        self.fee_rate = 0.005

    def total_assets(self):
        return np.sum(self.assets) + self.cash

    def apply(self, index, market, price_dividends_yield, expense_ratio):
        market = safe_cast(market)
        price_dividends_yield = safe_cast(price_dividends_yield)
        expense_ratio = safe_cast(expense_ratio)

        self.assets = self.assets * market
        diff = self.rebalancer.apply(index, self.assets, self.cash)

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
        self.cash -= np.sum(diff)
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

    def run(self, market, expense_ratio):
        index = market.get_index()
        markets = market.get_prices().to_records(index=False)
        price_dividends_yield = market.get_price_dividends_yield().to_records(index=False)
        expense_ratio = safe_cast(expense_ratio)
        for i in range(len(markets)):
            self.apply(
                index[i],
                markets[i],
                price_dividends_yield[i],
                expense_ratio
            )
