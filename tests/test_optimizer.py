import datetime
import os

import numpy.testing as npt

from mypo import Market, MinimumVarianceOptimizer

TEST_DATA = os.path.join(os.path.dirname(__file__), "data", "test.bin")


def test_minimum_variance_optimizer():
    market = Market.load(TEST_DATA)
    market.set_period_end(datetime.datetime(2021, 2, 16))
    df = market.get_prices()
    optimizer = MinimumVarianceOptimizer(df)
    weights = optimizer.optimize_weight()
    npt.assert_almost_equal(weights, [0.11692006, 0.88307994])
