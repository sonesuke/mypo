import yfinance as yf
import pandas as pd
import pickle
from collections import OrderedDict


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
        self.tickers[ticker] = normalized_raw(df)

    def get_market(self):
        rs = [self.tickers[ticker][['r']] for ticker in self.tickers.keys()]
        df = pd.concat(rs, axis=1, join='inner')
        df.columns = self.tickers.keys()
        return df

    def get_price_dividend_yield(self):
        rs = [self.tickers[ticker][['ir']] for ticker in self.tickers.keys()]
        df = pd.concat(rs, axis=1, join='inner')
        df.columns = self.tickers.keys()
        return df

    def save(self, filepath):
        with open(filepath, 'wb') as bin_file:
            pickle.dump(self,  bin_file)

    @classmethod
    def load(cls, filepath):
        with open(filepath, 'rb') as bin_file:
            return pickle.load(bin_file)
