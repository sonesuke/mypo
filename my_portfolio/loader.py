import numpy as np
import yfinance as yf
import pandas as pd
from .rebalance import rebalance
from .rebalance import calc_capital_gain_tax
from .rebalance import calc_income_gain_tax
from .rebalance import calc_fee

def normalized_raw(df):
    df['r'] = df['Close'].pct_change() + 1.0
    df.dropna(inplace=True)
    df['ir'] = df['Dividends']/df['Close']
    return df

class Loader(object):

    def __init__(self):
        self.tickers = []
        self.raw = {}

    def get(self, ticker):
        ticker = ticker.upper()
        df = yf.Ticker(ticker).history(period='max')
        self.tickers += [ticker]
        self.raw[ticker] = normalized_raw(df)
        return df

    def get_market(self):
        rs = [self.raw[ticker][['r']] for ticker in self.tickers]
        df = pd.concat(rs, axis=1, join='inner')
        df.columns = self.tickers
        return df

    def get_price_dividend_yield(self):
        rs = [self.raw[ticker][['ir']] for ticker in self.tickers]
        df = pd.concat(rs, axis=1, join='inner')
        df.columns = self.tickers
        return df