

class Simulator(object):

    def __init__(
            self,
            market_data,
            rebalance,
            settings,
            assets,
            ):
        self.market_data = market_data
        self.rebalance = rebalance
        self.settings = settings
        self.assets = assets

    def exec(self):
        periods = self.get_index()
        for period in periods:
            assets, dividends = self.apply(period)
            assets, tax = self.rebalance(assets, self.settings)
