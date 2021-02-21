import yfinance as yf
from collections import OrderedDict
from .market import Market
import pandas as pd


def normalized_raw(df):
    df['r'] = df['Close'].pct_change() + 1.0
    df.dropna(inplace=True)
    df['ir'] = df['Dividends']/df['Close']
    return df


class Loader(object):

    def __init__(self):
        self.tickers = OrderedDict()

    def get(self, ticker):
        ticker = ticker.upper()
        df = yf.Ticker(ticker).history(period='max')
        df.index = pd.to_datetime(df.index)
        self.tickers[ticker] = normalized_raw(df)

    def get_market(self):
        return Market(self.tickers)
