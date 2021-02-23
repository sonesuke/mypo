from typing import List
import numpy as np
import pandas as pd
import datetime


class Reporter(object):
    index: List[datetime.datetime]
    capital_gain: List[np.float64]
    income_gain: List[np.float64] = []
    cash: List[np.float64] = []
    deal: List[np.float64] = []
    fee: List[np.float64] = []
    capital_gain_tax: List[np.float64] = []
    income_gain_tax: List[np.float64] = []

    def __init__(self) -> None:
        self.index = []
        self.capital_gain = []
        self.income_gain = []
        self.cash = []
        self.deal = []
        self.fee = []
        self.capital_gain_tax = []
        self.income_gain_tax = []

    def record(
        self,
        index: datetime.datetime,
        capital_gain: np.float64,
        income_gain: np.float64,
        cash: np.float64,
        deal: np.float64,
        fee: np.float64,
        capital_gain_tax: np.float64,
        income_gain_tax: np.float64,
    ) -> None:
        self.index += [index]
        self.capital_gain += [capital_gain]
        self.income_gain += [income_gain]
        self.cash += [cash]
        self.deal += [deal]
        self.fee += [fee]
        self.capital_gain_tax += [capital_gain_tax]
        self.income_gain_tax += [income_gain_tax]

    def report(self) -> pd.DataFrame:
        return pd.DataFrame({
            'capital_gain': self.capital_gain,
            'income_gain': self.income_gain,
            'cash': self.cash,
            'deal': self.deal,
            'fee': self.fee,
            'capital_gain_tax': self.capital_gain_tax,
            'income_gain_tax': self.income_gain_tax
        }, index=self.index)