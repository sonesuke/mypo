"""Loader class for downloading stock data.

Download stock data from yahoo finance.

"""

from __future__ import annotations

import pickle
from datetime import datetime
from typing import Dict, List

import pandas as pd
import yfinance as yf

from mypo.market import Market


class Loader(object):
    """Loader class for downloading stock."""

    _tickers: Dict[str, pd.DataFrame]
    _names: Dict[str, str]
    _expense_ratio: Dict[str, float]
    _total_assets: Dict[str, int]
    _daily_volume_10_days: Dict[str, int]

    def __init__(
        self,
        tickers: Dict[str, pd.DataFrame] = {},
        names: Dict[str, str] = {},
        expense_ratio: Dict[str, float] = {},
        total_assets: Dict[str, int] = {},
        daily_volume_10_days: Dict[str, int] = {},
    ) -> None:  # pragma: no cover
        """Construct this object.

        Args:
            tickers: Tickers.
            names: Names.
            expense_ratio: Expense ratio.
            total_assets: Total assets.
            daily_volume_10_days: Daily volume 10 days.
        """
        self._tickers = tickers
        self._names = names
        self._expense_ratio = expense_ratio
        self._total_assets = total_assets
        self._daily_volume_10_days = daily_volume_10_days

    def save(self, filepath: str) -> None:  # pragma: no cover
        """Save market data to file.

        Args:
            filepath: Path to file for storing data.
        """
        with open(filepath, "wb") as bin_file:
            pickle.dump(self, bin_file)

    @staticmethod
    def load(filepath: str) -> Loader:
        """Load market data from file.

        Args:
            filepath: Path to file for loading data.

        Returns:
            Market object
        """
        with open(filepath, "rb") as bin_file:
            value: Loader = pickle.load(bin_file)
            return value

    def get(self, ticker: str, expense_ratio: float = 0.0) -> None:  # pragma: no cover
        """Get stock data of specified ticker.

        Args:
            ticker: Ticker that you want to download stock data.
            expense_ratio: Expense ratio of ticker. The default value is 0.0.

        Returns:
            Nothing
        """
        ticker = ticker.upper()
        t = yf.Ticker(ticker)
        df = t.history(period="max")
        df.index = pd.to_datetime(df.index)
        self._tickers[ticker] = df
        self._names[ticker] = t.info["longName"] if "longName" in t.info else ""
        self._total_assets[ticker] = t.info["totalAssets"] if "totalAssets" in t.info else None
        self._daily_volume_10_days[ticker] = (
            t.info["averageDailyVolume10Day"] if "averageDailyVolume10Day" in t.info else None
        )
        self._expense_ratio[ticker] = expense_ratio

    def get_market(self) -> Market:  # pragma: no cover
        """Get market data.

        Returns:
            Market data
        """
        sorted_total_assert = [item[0] for item in sorted(self._total_assets.items(), key=lambda x: x[1], reverse=True)]
        return Market.create_from_ticker(
            {k: self._names[k] for k in sorted_total_assert},
            {k: self._tickers[k] for k in sorted_total_assert},
            {k: self._expense_ratio[k] for k in sorted_total_assert},
        )

    def summary(self) -> pd.DataFrame:
        """Get summary.

        Returns:
            Summary.
        """
        df = pd.DataFrame(
            {
                "established": [self._tickers[t].index[0] for t in self._tickers.keys()],
                "names": [self._names[t] for t in self._tickers.keys()],
                "total_assets": [self._total_assets[t] for t in self._tickers.keys()],
                "volume": [self._daily_volume_10_days[t] for t in self._tickers.keys()],
                "expense_ratio": [self._expense_ratio[t] for t in self._tickers.keys()],
            },
            index=self._tickers.keys(),
        )
        return df.sort_index()

    def since(self, since: datetime) -> Loader:
        """Filter market data.

        Args:
            since: Established date.

        Returns:
            Filtered data.
        """
        tickers = [ticker for ticker in self._tickers.keys() if self._tickers[ticker].index[0] <= since]
        return self._filter(tickers)

    def volume_than(self, volume: int) -> Loader:
        """Filter market data.

        Args:
            volume: Volume for comparing.

        Returns:
            Filtered data.
        """
        tickers = [ticker for ticker in self._tickers.keys() if self._daily_volume_10_days[ticker] >= volume]
        return self._filter(tickers)

    def total_assets_than(self, total: int) -> Loader:
        """Filter market data.

        Args:
            total: Total assets for comparing.

        Returns:
            Filtered data.
        """
        tickers = [ticker for ticker in self._tickers.keys() if self._total_assets[ticker] >= total]
        return self._filter(tickers)

    def _filter(self, tickers: List[str]) -> Loader:
        """Filter market data.

        Args:
            since: Established date.

        Returns:
            Filtered data.
        """
        return Loader(
            tickers={key: value for key, value in self._tickers.items() if key in tickers},
            names={key: value for key, value in self._names.items() if key in tickers},
            expense_ratio={key: value for key, value in self._expense_ratio.items() if key in tickers},
            total_assets={key: value for key, value in self._total_assets.items() if key in tickers},
            daily_volume_10_days={key: value for key, value in self._daily_volume_10_days.items() if key in tickers},
        )
