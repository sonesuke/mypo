"""Loader class for downloading stock data.

Download stock data from yahoo finance.

"""


from collections import OrderedDict
from typing import Dict

import pandas as pd
import yfinance as yf

from mypo.market import Market


class Loader(object):
    """Loader class for downloading stock."""

    _tickers: Dict[str, pd.DataFrame]
    _expense_ratio: Dict[str, float]

    def __init__(self) -> None:  # pragma: no cover
        """Construct this object."""
        self._tickers = OrderedDict()
        self._expense_ratio = OrderedDict()

    def get(self, ticker: str, expense_ratio: float = 0.0) -> None:  # pragma: no cover
        """Get stock data of specified ticker.

        Args:
            ticker: Ticker that you want to download stock data.
            expense_ratio: Expense ratio of ticker. The default value is 0.0.

        Returns:
            Nothing
        """
        ticker = ticker.upper()
        df = yf.Ticker(ticker).history(period="max")
        df.index = pd.to_datetime(df.index)
        self._tickers[ticker] = df
        self._expense_ratio[ticker] = expense_ratio

    def get_market(self) -> Market:  # pragma: no cover
        """Get Market data.

        Returns:
            Market data
        """
        return Market.create_from_ticker(self._tickers, self._expense_ratio)
