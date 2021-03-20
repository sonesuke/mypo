import datetime
import os

import numpy.testing as npt

from mypo import Market
from mypo.optimizer import MinimumVarianceOptimizer, SharpRatioOptimizer, semi_covariance

TEST_DATA = os.path.join(os.path.dirname(__file__), "data", "test.bin")


def test_minimum_variance_optimizer():
    market = Market.load(TEST_DATA)
    market = market.extract(market.get_index()[:300])
    optimizer = MinimumVarianceOptimizer(market)
    weights = optimizer.optimize_weight()
    npt.assert_almost_equal(weights, [0.2501864, 0.7498136])


def test_minimum_variance_optimizer_with_minimum_return():
    market = Market.load(TEST_DATA)
    market = market.extract(market.get_index()[:200])
    optimizer = MinimumVarianceOptimizer(market=market, minimum_return=0.10)
    weights = optimizer.optimize_weight()
    npt.assert_almost_equal(weights, [0.3313992, 0.6686008])


def test_minimum_sharp_ratio_optimizer():
    market = Market.load(TEST_DATA)
    market = market.extract(market.get_index()[:200])
    optimizer = SharpRatioOptimizer(market, risk_free_rate=0.02)
    weights = optimizer.optimize_weight()
    npt.assert_almost_equal(weights, [0.4304742, 0.5695258])


def test_semi_minimum_variance_optimizer():
    market = Market.load(TEST_DATA)
    market = market.extract(market.get_index()[:200])
    optimizer = MinimumVarianceOptimizer(market, covariance_model=semi_covariance)
    weights = optimizer.optimize_weight()
    npt.assert_almost_equal(weights, [0.3243259, 0.6756741])
