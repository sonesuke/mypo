from collections import OrderedDict
from typing import Dict

import pandas as pd
import yfinance as yf

from .market import Market


def normalized_raw(df: pd.DataFrame) -> pd.DataFrame:
    df["r"] = df["Close"].pct_change() + 1.0
    df.dropna(inplace=True)
    df["ir"] = df["Dividends"] / df["Close"]
    return df


class Loader(object):
    tickers: Dict[str, pd.DataFrame]

    def __init__(self) -> None:
        self.tickers = OrderedDict()

    def get(self, ticker: str) -> None:
        ticker = ticker.upper()
        df = yf.Ticker(ticker).history(period="max")
        df.index = pd.to_datetime(df.index)
        self.tickers[ticker] = normalized_raw(df)

    def get_market(self) -> Market:
        return Market(self.tickers)
