"""Market object for store and loading stock prices data."""

from __future__ import annotations

import pickle
from datetime import datetime
from enum import Enum
from typing import Dict, List

import numpy as np
import pandas as pd

from mypo.common import safe_cast


class SamplingMethod(Enum):
    """Enum of method for resampling."""

    YEAR = "Y"
    MONTH = "M"


class Market(object):
    """Market class for store and loading stock prices data."""

    _names: Dict[str, str]
    _expense_ratio: Dict[str, float]
    _closes: pd.DataFrame
    _price_dividends_yield: pd.DataFrame

    def __init__(
        self,
        names: Dict[str, str],
        closes: pd.DataFrame,
        price_dividends_yield: pd.DataFrame,
        expense_ratio: Dict[str, float],
    ):
        """Construct this object.

        Args:
            names: Names.
            closes: Tickers.
            price_dividends_yield: Price dividends yield.
            expense_ratio: Expense ratio.
        """
        self._names = names
        self._expense_ratio = expense_ratio
        self._closes = closes
        self._price_dividends_yield = price_dividends_yield

    def save(self, filepath: str) -> None:  # pragma: no cover
        """Save market data to file.

        Args:
            filepath: Path to file for storing data.
        """
        with open(filepath, "wb") as bin_file:
            pickle.dump(self, bin_file)

    @staticmethod
    def load(filepath: str) -> Market:
        """Load market data from file.

        Args:
            filepath: Path to file for loading data.

        Returns:
            Market object
        """
        with open(filepath, "rb") as bin_file:
            value: Market = pickle.load(bin_file)
            return value

    @staticmethod
    def create(start_date: str, end_date: str, yearly_gain: float, ticker: str = "None") -> Market:
        """Load market data from file.

        Args:
            ticker: Ticker.
            start_date: Start date.
            end_date: End date.
            yearly_gain: Yearly gain.

        Returns:
            Market object
        """
        index = pd.date_range(start_date, end_date, freq="D")
        n = len(index)
        daily_gain = (1.0 + yearly_gain) ** (1 / 365)
        prices = np.ones(n) * (daily_gain ** np.arange(n))
        return Market.create_from_ticker(
            names={ticker: "None"},
            tickers={ticker: pd.DataFrame({"Close": prices, "Dividends": np.zeros(n)}, index=index)},
            expense_ratio={ticker: 0.0},
        )

    @staticmethod
    def create_from_ticker(
        names: Dict[str, str], tickers: Dict[str, pd.DataFrame], expense_ratio: Dict[str, float]
    ) -> Market:
        """Construct this object.

        Args:
            names: Names.
            tickers: Tickers.
            expense_ratio: Expense ratio.
        """
        return Market(
            names=names,
            closes=Market.calc_raw(tickers),
            price_dividends_yield=Market.calc_price_dividends_yield(tickers),
            expense_ratio=expense_ratio,
        )

    @staticmethod
    def calc_raw(tickers: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Get price data from stored market data.

        Args:
            tickers: Tickers

        Returns:
            Prices
        """
        rs = [tickers[ticker][["Close"]] for ticker in tickers.keys()]
        df = pd.concat(rs, axis=1, join="inner")
        df.columns = tickers.keys()
        return df

    @staticmethod
    def calc_price_dividends_yield(tickers: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Get price dividends yield.

        Returns:
            price dividends yield data
        """
        rs = [tickers[ticker]["Dividends"] / tickers[ticker]["Close"] for ticker in tickers.keys()]
        df = pd.concat(rs, axis=1, join="inner")
        df.columns = tickers.keys()
        return df

    def resample(self, method: SamplingMethod) -> Market:
        """Resample market data.

        Args:
            method: Frequency of sampling.

        Returns:
            Resampled data.
        """
        market = self.trim(method)
        if method == SamplingMethod.YEAR:
            rule = "Y"
        elif method == SamplingMethod.MONTH:
            rule = "M"
        else:
            assert False  # pragma: no cover
        return Market(
            names=market._names,
            closes=market._closes.resample(str(rule)).last(),
            price_dividends_yield=market._price_dividends_yield.resample(rule).sum(),
            expense_ratio=market._expense_ratio,
        )

    def trim(self, method: SamplingMethod) -> Market:
        """Trim market data.

        Args:
            method: Frequency of sampling.

        Returns:
            Trim data.
        """
        if method == SamplingMethod.YEAR:
            rule = "Y"
        elif method == SamplingMethod.MONTH:
            rule = "M"
        else:
            assert False  # pragma: no cover
        df = self._closes.groupby(pd.Grouper(freq=rule)).sum()
        if method == SamplingMethod.YEAR:
            first = df.index[0] if self._closes.index.is_year_start[0] else df.index[1]
            last = df.index[-1] if self._closes.index.is_year_end[-1] else df.index[-2]
        elif method == SamplingMethod.MONTH:
            first = df.index[0] if self._closes.index.is_month_start[0] else df.index[1]
            last = df.index[-1] if self._closes.index.is_month_end[-1] else df.index[-2]
        else:
            assert False  # pragma: no cover
        return Market(
            names=self._names,
            closes=self._closes[first:last],
            price_dividends_yield=self._price_dividends_yield[first:last],
            expense_ratio=self._expense_ratio,
        )

    def extract(self, index: pd.Series) -> Market:
        """Extract market data.

        Args:
            index: Index what you want to extract.

        Returns:
            Extracted Data
        """
        return Market(
            names=self._names,
            closes=self._closes.loc[index],
            price_dividends_yield=self._price_dividends_yield.loc[index],
            expense_ratio=self._expense_ratio,
        )

    def filter(self, tickers: List[str]) -> Market:
        """Filter tickers.

        Args:
            tickers: Remaining tickers.

        Returns:
            Filtered market data.
        """
        return Market(
            names=self._names,
            closes=self._closes[tickers],
            price_dividends_yield=self._price_dividends_yield[tickers],
            expense_ratio={
                ticker: expense_ratio for ticker, expense_ratio in self._expense_ratio.items() if ticker in tickers
            },
        )

    def tail(self, n: int) -> Market:
        """Extract market data.

        Args:
            n: Last n records.

        Returns:
            Extracted Data
        """
        return Market(
            names=self._names,
            closes=self._closes.loc[self._closes.tail(n).index],
            price_dividends_yield=self._price_dividends_yield.loc[self._price_dividends_yield.tail(n).index],
            expense_ratio=self._expense_ratio,
        )

    def head(self, n: int) -> Market:
        """Extract market data.

        Args:
            n: First n records.

        Returns:
            Extracted Data
        """
        return Market(
            names=self._names,
            closes=self._closes.loc[self._closes.head(n).index],
            price_dividends_yield=self._price_dividends_yield.loc[self._price_dividends_yield.head(n).index],
            expense_ratio=self._expense_ratio,
        )

    def get_summary(self) -> pd.DataFrame:
        """Get summary.

        Returns:
            Summary
        """
        rate_of_change = self.get_rate_of_change()
        df = pd.DataFrame(
            {
                "daily return": rate_of_change.mean(),
                "variance": rate_of_change.var(),
                "sharp ratio": rate_of_change.mean() / rate_of_change.var(),
                "expense ratio": self.get_expense_ratio(),
            },
            index=self.get_tickers(),
        )
        return df.sort_index()

    def get_raw(self) -> pd.DataFrame:
        """Get price data from stored market data.

        Returns:
            Prices
        """
        return self._closes

    def get_tickers(self) -> List[str]:
        """Get tickers.

        Returns:
            Tickers.
        """
        return list(self._closes.columns)

    def get_length(self) -> int:
        """Get length.

        Returns:
            length.
        """
        return len(self.get_rate_of_change())

    def get_index(self) -> pd.Index:
        """Get index date from stored market data.

        Returns:
            index date
        """
        return self._closes.index

    def get_first_date(self) -> datetime:
        """Get first date.

        Returns:
            First date.
        """
        first_date: datetime = self.get_index()[0]
        return first_date

    def get_last_date(self) -> datetime:
        """Get Last date.

        Returns:
            Last date.
        """
        last_date: datetime = self.get_index()[-1]
        return last_date

    def get_normalized_prices(self) -> pd.DataFrame:
        """Get normalized prices.

        Returns:
            Normalized prices
        """
        df = self._closes
        for c in df.columns:
            df[c] = df[c] / df[c][0]
        return df

    def get_rate_of_change(self) -> pd.DataFrame:
        """Get rate of change of prices.

        Returns:
            Rate of change
        """
        df = self._closes
        df = df.pct_change(axis=0)
        df.dropna(inplace=True)
        return df

    def get_price_dividends_yield(self) -> pd.DataFrame:
        """Get price dividends yield.

        Returns:
            price dividends yield data
        """
        return self._price_dividends_yield

    def get_expense_ratio(self) -> np.ndarray:
        """Get expense ratio.

        Returns:
            Expense ratio
        """
        rs = [self._expense_ratio[ticker] for ticker in self._closes.columns]
        return safe_cast(rs)

    def get_relative(self, ticker: str, n: int = 10) -> pd.DataFrame:
        """Get relative ticker.

        Args:
            ticker: Ticker.

        Returns:
            Relative tickers.
        """
        df = self.get_rate_of_change()
        df = df.corr()
        df = df.sort_values(ticker, ascending=False)[:n]
        df = df[[ticker]]
        df.columns = ["correlation"]
        return df
