"""Class for report result."""
import datetime
import itertools
from typing import List

import numpy as np
import pandas as pd

from mypo.indicator import max_drawdown, max_drawdown_span, yearly_total_return


class Reporter(object):
    """Reporter class of simulation."""

    _index: List[datetime.datetime]
    _weights: List[np.ndarray]
    _tickers: List[List[str]]
    _total_assets: List[np.float64]
    _capital_gain: List[np.float64]
    _income_gain: List[np.float64]
    _cash: List[np.float64]
    _deal: List[np.float64]
    _fee: List[np.float64]
    _capital_gain_tax: List[np.float64]
    _income_gain_tax: List[np.float64]

    def __init__(self) -> None:
        """Construct this object."""
        self._index = []
        self._weights = []
        self._tickers = []
        self._total_assets = []
        self._capital_gain = []
        self._income_gain = []
        self._cash = []
        self._deal = []
        self._fee = []
        self._capital_gain_tax = []
        self._income_gain_tax = []

    def record(
        self,
        at: datetime.datetime,
        weights: np.ndarray,
        tickers: List[str],
        total_assets: np.float64,
        capital_gain: np.float64,
        income_gain: np.float64,
        cash: np.float64,
        deal: np.float64,
        fee: np.float64,
        capital_gain_tax: np.float64,
        income_gain_tax: np.float64,
    ) -> None:
        """Record current situation.

        Args:
            at: Current date.
            weights: Weight of asserts.
            tickers: Tickers.
            total_assets: Current total assets.
            capital_gain: Current capital gain.
            income_gain: Current income gain.
            cash: Current cash.
            deal: Current deal.
            fee: Current fee.
            capital_gain_tax: Current capital gain tax.
            income_gain_tax: Current income gain tax.
        """
        self._index += [at]
        self._weights += [weights]
        self._tickers += [tickers]
        self._total_assets += [total_assets]
        self._capital_gain += [capital_gain]
        self._income_gain += [income_gain]
        self._cash += [cash]
        self._deal += [deal]
        self._fee += [fee]
        self._capital_gain_tax += [capital_gain_tax]
        self._income_gain_tax += [income_gain_tax]

    def history(self) -> pd.DataFrame:
        """Report the result.

        Returns:
            result
        """
        return pd.DataFrame(
            {
                "total_assets": self._total_assets,
                "capital_gain": self._capital_gain,
                "income_gain": self._income_gain,
                "cash": self._cash,
                "deal": self._deal,
                "fee": self._fee,
                "capital_gain_tax": self._capital_gain_tax,
                "income_gain_tax": self._income_gain_tax,
            },
            index=self._index,
        )

    def get_tickers(self) -> List[str]:
        """Get tickers.

        Returns:
            Tickers.
        """
        return list(set(itertools.chain.from_iterable(self._tickers)))

    def history_weights(self) -> pd.DataFrame:
        """Report weights.

        Returns:
            Report of weights.
        """
        tickers = self.get_tickers()
        ret = []
        for i in range(len(self._weights)):
            weight_record = {ticker: 0.0 for ticker in tickers}
            for j in range(len(self._tickers[i])):
                weight_record[self._tickers[i][j]] = self._weights[i][j]
            ret += [weight_record]
        return pd.DataFrame.from_records(ret, index=self._index)

    def summary(self) -> pd.DataFrame:
        """Get summary.

        Returns:
            Summary.
        """
        report = self.history()
        return pd.DataFrame(
            {
                "tickers": [self.get_tickers()],
                "yearly total return": [yearly_total_return(report)],
                "max draw down": [max_drawdown(report)],
                "max draw down span": [max_drawdown_span(report)],
            }
        )
