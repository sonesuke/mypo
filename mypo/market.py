"""Market object for store and loading stock prices data."""

from __future__ import annotations

import pickle
from typing import Dict

import numpy as np
import pandas as pd

from .common import safe_cast


class Market(object):
    """Market class for store and loading stock prices data."""

    _tickers: Dict[str, pd.DataFrame]
    _expense_ratio: Dict[str, float]

    def __init__(
        self, tickers: Dict[str, pd.DataFrame], expense_ratio: Dict[str, float]
    ):
        self._tickers = tickers
        self._expense_ratio = expense_ratio

    def save(self, filepath: str) -> None:
        """
        Save market data to file.

        Parameters
        ----------
        filepath
            Path to file for storing data.
        """
        with open(filepath, "wb") as bin_file:
            pickle.dump(self, bin_file)

    @classmethod
    def load(cls, filepath: str) -> Market:
        """
        Load market data from file.

        Parameters
        ----------
        filepath
            Path to file for loading data.

        Returns
        -------
        Market object

        """
        with open(filepath, "rb") as bin_file:
            value: Market = pickle.load(bin_file)
            return value

    def extract(self, index: pd.Series) -> Market:
        """
        Extract market data.

        Parameters
        ----------
        index
            Index what you want to extract.

        Returns
        -------
            Extracted Data
        """
        return Market(
            tickers={ticker: data.loc[index] for ticker, data in self._tickers.items()},
            expense_ratio=self._expense_ratio,
        )

    def get_index(self) -> pd.Series:
        """
        Get index date from stored market data.

        Returns
        -------
        index date
        """
        rs = [self._tickers[ticker][["r"]] for ticker in self._tickers.keys()]
        df = pd.concat(rs, axis=1, join="inner")
        return df.index

    def get_prices(self) -> pd.DataFrame:
        """
        Get price data from stored market data.

        Returns
        -------
        Price data
        """
        rs = [self._tickers[ticker][["r"]] for ticker in self._tickers.keys()]
        df = pd.concat(rs, axis=1, join="inner")
        df.columns = self._tickers.keys()
        return df

    def get_price_dividends_yield(self) -> pd.DataFrame:
        """
        Get price dividends yield from stored market data.

        Returns
        -------
        price dividends yield data
        """
        rs = [self._tickers[ticker][["ir"]] for ticker in self._tickers.keys()]
        df = pd.concat(rs, axis=1, join="inner")
        df.columns = self._tickers.keys()
        return df

    def get_expense_ratio(self) -> np.ndarray:
        """
        Get expense ratio.

        Returns
        -------
        Expense ratio
        """
        rs = [self._expense_ratio[ticker] for ticker in self._tickers.keys()]
        return safe_cast(rs)
