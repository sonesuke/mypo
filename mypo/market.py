"""Market object for store and loading stock prices data."""

from __future__ import annotations

import pickle
from typing import Dict

import numpy as np
import pandas as pd

from mypo.common import safe_cast


class Market(object):
    """Market class for store and loading stock prices data."""

    _tickers: Dict[str, pd.DataFrame]
    _expense_ratio: Dict[str, float]

    def __init__(self, tickers: Dict[str, pd.DataFrame], expense_ratio: Dict[str, float]):
        """Construct this object.

        Args:
            tickers: Tickers.
            expense_ratio: Expense ratio.
        """
        self._tickers = tickers
        self._expense_ratio = expense_ratio

    def save(self, filepath: str) -> None:
        """Save market data to file.

        Args:
            filepath: Path to file for storing data.
        """
        with open(filepath, "wb") as bin_file:
            pickle.dump(self, bin_file)

    @classmethod
    def load(cls, filepath: str) -> Market:
        """Load market data from file.

        Args:
            filepath: Path to file for loading data.

        Returns:
            Market object
        """
        with open(filepath, "rb") as bin_file:
            value: Market = pickle.load(bin_file)
            return value

    @classmethod
    def create(cls, start_date: str, end_date: str, yearly_gain: float, ticker: str = "None") -> Market:
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
        return Market(
            tickers={ticker: pd.DataFrame({"Close": prices, "Dividends": np.zeros(n)}, index=index)},
            expense_ratio={ticker: 0.0},
        )

    def extract(self, index: pd.Series) -> Market:
        """Extract market data.

        Args:
            index: Index what you want to extract.

        Returns:
            Extracted Data
        """
        return Market(
            tickers={ticker: data.loc[index] for ticker, data in self._tickers.items()},
            expense_ratio=self._expense_ratio,
        )

    def get_index(self) -> pd.Series:
        """Get index date from stored market data.

        Returns:
            index date
        """
        df = self.get_raw()
        return df.index

    def get_raw(self) -> pd.DataFrame:
        """Get price data from stored market data.

        Returns:
            Prices
        """
        rs = [self._tickers[ticker][["Close"]] for ticker in self._tickers.keys()]
        df = pd.concat(rs, axis=1, join="inner")
        df.columns = self._tickers.keys()
        return df

    def get_normalized_prices(self) -> pd.DataFrame:
        """Get normalized prices.

        Returns:
            Normalized prices
        """
        df = self.get_raw()
        for c in df.columns:
            df[c] = df[c] / df[c][0]
        return df

    def get_rate_of_change(self) -> pd.DataFrame:
        """Get rate of change of prices.

        Returns:
            Rate of change
        """
        df = self.get_raw()
        df = df.pct_change(axis=0)
        df.dropna(inplace=True)
        df.columns = self._tickers.keys()
        return df

    def get_price_dividends_yield(self) -> pd.DataFrame:
        """Get price dividends yield.

        Returns:
            price dividends yield data
        """
        rs = [self._tickers[ticker]["Dividends"] / self._tickers[ticker]["Close"] for ticker in self._tickers.keys()]
        df = pd.concat(rs, axis=1, join="inner")
        df.columns = self._tickers.keys()
        return df

    def get_expense_ratio(self) -> np.ndarray:
        """Get expense ratio.

        Returns:
            Expense ratio
        """
        rs = [self._expense_ratio[ticker] for ticker in self._tickers.keys()]
        return safe_cast(rs)
