import datetime

import numpy as np
import numpy.typing as npt
import pandas as pd

from .common import calc_capital_gain_tax, calc_fee, calc_income_gain_tax, safe_cast
from .market import Market
from .rebalancer import Rebalancer
from .reporter import Reporter


class Runner(object):
    """Runner of simulation."""

    _assets: np.ndarray
    _initial_assets: np.ndarray
    _rebalancer: Rebalancer
    _reporter: Reporter
    _cash: np.float64
    _tax_rate: np.float64
    _spending: np.float64
    _fee_rate: np.float64

    def __init__(self, assets: npt.ArrayLike, rebalancer: Rebalancer, cash: np.float64, spending: np.float64):
        """
        Construct this object.

        Parameters
        ----------
        assets
            Initial asset.

        rebalancer
            Reblance strategy.

        cash
            Initial cash.

        spending
            Monthly spending.
        """
        self._assets = safe_cast(assets)
        self._initial_assets = self._assets
        self.assets = safe_cast(assets)
        self.initial_assets = self.assets
        self._rebalancer = rebalancer
        self._reporter = Reporter()
        self._cash = cash
        self._spending = spending
        self._tax_rate = 0.2  # type: ignore
        self._fee_rate = 0.005  # type: ignore

    def total_assets(self) -> np.float64:
        """
        Get current total assets. Total asset is addition of stock assets and cash.

        Returns
        -------
        Total assets.
        """
        return np.float64(np.sum(self._assets) + self._cash)

    def apply(
        self,
        index: datetime.datetime,
        market: npt.ArrayLike,
        price_dividends_yield: npt.ArrayLike,
        expense_ratio: npt.ArrayLike,
    ) -> None:
        """
        Apply current market situation.

        Parameters
        ----------
        index
            Current date.

        market
            Current market situation.

        price_dividends_yield
            Current dividends yield this date.

        expense_ratio
            Expense ratio of holding assets.
        """
        previous_assets = np.sum(self.assets)
        market = safe_cast(market)
        price_dividends_yield = safe_cast(price_dividends_yield)
        expense_ratio = safe_cast(expense_ratio)

        # apply market prices
        self.assets = self.assets * market
        diff = self.rebalancer.apply(index, self.assets, self.cash)
        deal: np.float64 = np.abs(diff)

        # process of capital gain
        capital_gain_tax = calc_capital_gain_tax(self._initial_assets, self._assets, diff, self._tax_rate)
        self._cash -= capital_gain_tax
        fee = calc_fee(diff, self._fee_rate)
        self._cash -= fee
        self._cash -= np.float64(np.sum(diff))
        self._assets += diff

        # process of income gain
        income_gain = np.sum(self._assets * price_dividends_yield)
        income_gain_tax = calc_income_gain_tax(self._assets, price_dividends_yield, self._tax_rate)
        self._cash += income_gain
        self._cash -= income_gain_tax

        # process of others
        self._assets = (1.0 - expense_ratio) * self._assets

        # record to reporter
        capital_gain: np.float64 = np.float64(np.sum(self._assets) - previous_assets)
        self.reporter.record(index, capital_gain, income_gain, self._cash, deal, fee, capital_gain_tax, income_gain_tax)

    def run(self, market: Market, expense_ratio: npt.ArrayLike) -> None:
        """
        Run simulation.

        Parameters
        ----------
        market
            Market data.

        expense_ratio
            Expense ratio of holding stocks.
        """
        index = market.get_index()
        markets = market.get_prices().to_records(index=False)
        price_dividends_yield = market.get_price_dividends_yield().to_records(index=False)
        expense_ratio = safe_cast(expense_ratio)
        for i in range(len(markets)):
            self.apply(index[i], markets[i], price_dividends_yield[i], expense_ratio)

    def report(self) -> pd.DataFrame:
        """
        Report simulation.

        Returns
        -------
        result
        """
        return self._reporter.report()
