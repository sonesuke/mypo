import pandas as pd
import pickle
import datetime


class Market(object):

    def __init__(self, tickers):
        self.tickers = tickers
        self.period_end = datetime.datetime.now()

    def save(self, filepath):
        with open(filepath, 'wb') as bin_file:
            pickle.dump(self,  bin_file)

    @classmethod
    def load(cls, filepath):
        with open(filepath, 'rb') as bin_file:
            return pickle.load(bin_file)

    def set_period_end(self, date):
        self.period_end = date

    def get_index(self):
        rs = [self.tickers[ticker][['r']] for ticker in self.tickers.keys()]
        df = pd.concat(rs, axis=1, join='inner')
        return df.index

    def get_prices(self):
        rs = [self.tickers[ticker][['r']] for ticker in self.tickers.keys()]
        df = pd.concat(rs, axis=1, join='inner')
        df = df[df.index < self.period_end]
        df.columns = self.tickers.keys()
        return df

    def get_price_dividends_yield(self):
        rs = [self.tickers[ticker][['ir']] for ticker in self.tickers.keys()]
        df = pd.concat(rs, axis=1, join='inner')
        df = df[df.index < self.period_end]
        df.columns = self.tickers.keys()
        return df
