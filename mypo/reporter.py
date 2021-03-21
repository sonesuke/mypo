"""Class for report result."""
import datetime
from typing import List

import numpy as np
import pandas as pd


class Reporter(object):
    """Reporter class of simulation."""

    _index: List[datetime.datetime]
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
        self._total_assets += [total_assets]
        self._capital_gain += [capital_gain]
        self._income_gain += [income_gain]
        self._cash += [cash]
        self._deal += [deal]
        self._fee += [fee]
        self._capital_gain_tax += [capital_gain_tax]
        self._income_gain_tax += [income_gain_tax]

    def report(self) -> pd.DataFrame:
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
