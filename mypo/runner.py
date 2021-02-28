"""Simulation."""

import datetime

import numpy as np
import numpy.typing as npt
import pandas as pd

from .common import calc_capital_gain_tax, calc_fee, calc_income_gain_tax, safe_cast
from .market import Market
from .rebalancer import Rebalancer
from .reporter import Reporter
from .settings import Settings

WEEK_DAYS = int(365 * 5 / 7)


class Runner(object):
    """Runner of simulation."""

    _assets: np.ndarray
    _averagel_assets_price: np.ndarray
    _rebalancer: Rebalancer
    _reporter: Reporter
    _cash: np.float64
    _settings: Settings

    def __init__(
        self,
        assets: npt.ArrayLike,
        rebalancer: Rebalancer,
        cash: np.float64,
        settings: Settings,
    ):
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
        self._averagel_assets_price = np.ones(len(self._assets))
        self._rebalancer = rebalancer
        self._reporter = Reporter()
        self._cash = cash
        self._settings = settings
        self._reporter.record(
            pd.NaT,
            self.total_assets(),
            1.0,  # type: ignore
            0.0,  # type: ignore
            self._cash,
            0.0,  # type: ignore
            0.0,  # type: ignore
            0.0,  # type: ignore
            0.0,  # type: ignore
        )

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
        prices: npt.ArrayLike,
        price_dividends_yield: npt.ArrayLike,
        expense_ratio: npt.ArrayLike,
    ) -> None:
        """
        Apply current market situation.

        Parameters
        ----------
        index
            Current date.

        prices
            Current market situation.

        price_dividends_yield
            Current dividends yield this date.

        expense_ratio
            Expense ratio of holding assets.
        """
        previous_assets = np.sum(self._assets)
        prices = safe_cast(prices)
        price_dividends_yield = safe_cast(price_dividends_yield)
        expense_ratio = safe_cast(expense_ratio)

        # apply market prices
        self._assets = self._assets * prices
        diff = self._rebalancer.apply(index, self._assets, self._cash)
        deal: np.float64 = np.abs(diff)

        # process of capital gain
        capital_gain_tax = calc_capital_gain_tax(
            self._averagel_assets_price, self._assets, diff, self._settings.tax_rate
        )
        self._cash -= capital_gain_tax
        fee = calc_fee(diff, self._settings.fee_rate)
        self._cash -= fee
        self._cash -= np.float64(np.sum(diff))
        self._assets += diff
        trading_prices = np.where(diff > 0, prices, self._averagel_assets_price)
        self._averagel_assets_price = (
            self._averagel_assets_price * previous_assets + diff * trading_prices
        ) / self._assets

        # process of income gain
        income_gain = np.sum(self._assets * price_dividends_yield)
        income_gain_tax = calc_income_gain_tax(
            self._assets, price_dividends_yield, self._settings.tax_rate
        )
        self._cash += income_gain
        self._cash -= income_gain_tax

        # process of others
        self._assets = (1.0 - expense_ratio / WEEK_DAYS) * self._assets

        # record to reporter
        capital_gain: np.float64 = np.float64(
            np.sum(self._assets) / np.sum(previous_assets)
        )
        self._reporter.record(
            index,
            self.total_assets(),
            capital_gain,
            income_gain,
            self._cash,
            deal,
            fee,
            capital_gain_tax,
            income_gain_tax,
        )

    def run(self, market: Market) -> None:
        """
        Run simulation.

        Parameters
        ----------
        market
            Market data.
        """
        index = market.get_index()
        markets = market.get_prices().to_records(index=False)
        price_dividends_yield = market.get_price_dividends_yield().to_records(
            index=False
        )
        expense_ratio = market.get_expense_ratio()
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
