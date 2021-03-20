"""Simulation."""

import datetime

import numpy as np
import numpy.typing as npt
import pandas as pd

from mypo.common import calc_capital_gain_tax, calc_fee, calc_income_gain_tax, safe_cast
from mypo.market import Market
from mypo.rebalancer.base_rebalancer import BaseRebalancer
from mypo.rebalancer.no_rebalancer import NoRebalancer
from mypo.reporter import Reporter
from mypo.settings import DEFAULT_SETTINGS, Settings

WEEK_DAYS = int(365 * 5 / 7)


class Runner(object):
    """Runner of simulation."""

    _assets: np.ndarray
    _average_assets_prices: np.ndarray
    _rebalancer: BaseRebalancer
    _reporter: Reporter
    _cash: np.float64
    _withdraw: np.float64
    _settings: Settings
    _month: int

    def __init__(
        self,
        assets: npt.ArrayLike = np.array([0]),
        cash: np.float64 = np.float64(0),
        withdraw: np.float64 = np.float64(0),
        rebalancer: BaseRebalancer = NoRebalancer(),
        settings: Settings = DEFAULT_SETTINGS,
    ):
        """Construct this object.

        Args:
            assets: Initial asset.
            rebalancer: Rebalance strategy.
            cash: Initial cash.
            withdraw: Withdraw.
            settings: Settings.
        """
        self._assets = safe_cast(assets)
        self._average_assets_prices = np.ones(len(self._assets))
        self._rebalancer = rebalancer
        self._reporter = Reporter()
        self._cash = cash
        self._withdraw = withdraw
        self._settings = settings
        self._month = 0
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
        """Get current total assets. Total asset is addition of stock assets and cash.

        Returns:
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
        """Apply current market situation.

        Args:
            index: Current date.
            prices: Current market situation.
            price_dividends_yield: Current dividends yield this date.
            expense_ratio: Expense ratio of holding assets.
        """
        previous_assets = np.sum(self._assets)
        prices = safe_cast(prices)
        price_dividends_yield = safe_cast(price_dividends_yield)
        expense_ratio = safe_cast(expense_ratio)

        # apply withdraw
        if self._month != index.month:
            self._cash -= self._withdraw / 12
        self._month = index.month

        # apply market prices
        self._assets = self._assets * (1.0 + prices)
        diff = self._rebalancer.apply(index, self._assets, self._cash)
        deal: np.float64 = np.abs(diff)

        # process of capital gain
        capital_gain_tax = calc_capital_gain_tax(
            self._average_assets_prices, self._assets, diff, self._settings.tax_rate
        )
        self._cash -= capital_gain_tax
        fee = calc_fee(diff, self._settings.fee_rate)
        self._cash -= np.float64(np.sum(diff) + fee)
        self._assets += diff
        trading_prices = np.where(diff > 0, 1.0 + prices, self._average_assets_prices)
        self._average_assets_prices = np.where(
            self._assets != 0,
            (self._average_assets_prices * previous_assets + diff * trading_prices) / self._assets,
            self._assets,
        )

        # process of income gain
        income_gain = np.sum(self._assets * price_dividends_yield)
        income_gain_tax = calc_income_gain_tax(self._assets, price_dividends_yield, self._settings.tax_rate)
        self._cash += income_gain
        self._cash -= income_gain_tax

        # process of others
        self._assets = (1.0 - expense_ratio / WEEK_DAYS) * self._assets

        # record to reporter
        capital_gain: np.float64 = np.float64(np.sum(self._assets) / np.sum(previous_assets))
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
        """Run simulation.

        Args:
            market: Market data.
        """
        index = market.get_index()
        markets = market.get_rate_of_change().to_records(index=False)
        price_dividends_yield = market.get_price_dividends_yield().to_records(index=False)
        expense_ratio = market.get_expense_ratio()
        for i in range(len(markets)):
            self.apply(index[i], markets[i], price_dividends_yield[i], expense_ratio)

    def report(self) -> pd.DataFrame:
        """Report simulation.

        Returns:
            result
        """
        return self._reporter.report()
