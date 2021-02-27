"""Market object for store and loading stock prices data."""

from __future__ import annotations

import datetime
import pickle
from typing import Dict

import pandas as pd


class Market(object):
    """Market class for store and loading stock prices data."""

    _tickers: Dict[str, pd.DataFrame]
    _period_end: datetime.datetime

    def __init__(self, tickers: Dict[str, pd.DataFrame]):
        self._tickers = tickers
        self._period_end = datetime.datetime.now()

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

    def set_period_end(self, date: datetime.datetime) -> None:
        """
        Set the end of period.

        Parameters
        ----------
        date
            end date of period.
        """
        self._period_end = date

    def get_index(self) -> pd.Series:
        """
        Get index date from stored market data.

        Returns
        -------
        index date
        """
        rs = [self._tickers[ticker][["r"]] for ticker in self._tickers.keys()]
        df = pd.concat(rs, axis=1, join="inner")
        df = df[df.index < self.period_end]
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
        df = df[df.index < self._period_end]
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
        df = df[df.index < self._period_end]
        df.columns = self._tickers.keys()
        return df
