from __future__ import annotations

import datetime
import pickle
from typing import Dict

import pandas as pd


class Market(object):

    tikcers: Dict[str, pd.DataFrame]
    period_end: datetime.datetime

    def __init__(self, tickers: Dict[str, pd.DataFrame]):
        self.tickers = tickers
        self.period_end = datetime.datetime.now()

    def save(self, filepath: str) -> None:
        with open(filepath, "wb") as bin_file:
            pickle.dump(self, bin_file)

    @classmethod
    def load(cls, filepath: str) -> Market:
        with open(filepath, "rb") as bin_file:
            value: Market
            value = pickle.load(bin_file)
            return value

    def set_period_end(self, date: datetime.datetime) -> None:
        self.period_end = date

    def get_index(self) -> pd.Series:
        rs = [self.tickers[ticker][["r"]] for ticker in self.tickers.keys()]
        df = pd.concat(rs, axis=1, join="inner")
        return df.index

    def get_prices(self) -> pd.DataFrame:
        rs = [self.tickers[ticker][["r"]] for ticker in self.tickers.keys()]
        df = pd.concat(rs, axis=1, join="inner")
        df = df[df.index < self.period_end]
        df.columns = self.tickers.keys()
        return df

    def get_price_dividends_yield(self) -> pd.DataFrame:
        rs = [self.tickers[ticker][["ir"]] for ticker in self.tickers.keys()]
        df = pd.concat(rs, axis=1, join="inner")
        df = df[df.index < self.period_end]
        df.columns = self.tickers.keys()
        return df
