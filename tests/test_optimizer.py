import os

import numpy.testing as npt
import pytest

from mypo import Market
from mypo.optimizer import (
    CVaROptimizer,
    MaximumDiversificationOptimizer,
    MeanVarianceOptimizer,
    MinimumVarianceOptimizer,
    RiskParityOptimizer,
    SharpRatioOptimizer,
)
from mypo.sampler import Sampler

skip_long_tests = pytest.mark.skipif(True, reason="This test takes long time.")
TEST_DATA = os.path.join(os.path.dirname(__file__), "data", "test.bin")
MODEL_DATA = os.path.join(os.path.dirname(__file__), "data", "sample.bin")


def test_minimum_variance_optimizer() -> None:
    market = Market.load(TEST_DATA)
    market = market.head(300)
    optimizer = MinimumVarianceOptimizer()
    optimizer.optimize(market, market.get_last_date())
    weights = optimizer.get_weights()
    npt.assert_almost_equal(weights, [0.2493157, 0.7506843])


def test_minimum_variance_optimizer_with_minimum_return() -> None:
    market = Market.load(TEST_DATA)
    market = market.head(200)
    optimizer = MinimumVarianceOptimizer(minimum_return=0.10)
    optimizer.optimize(market, market.get_last_date())
    weights = optimizer.get_weights()
    npt.assert_almost_equal(weights, [0.44973, 0.55027], decimal=5)


def test_mean_variance_optimizer() -> None:
    market = Market.load(TEST_DATA)
    market = market.head(300)
    optimizer = MeanVarianceOptimizer(risk_tolerance=0)
    optimizer.optimize(market, market.get_last_date())
    weights = optimizer.get_weights()
    npt.assert_almost_equal(weights, [0.2493157, 0.7506843])


def test_mean_variance_optimizer_with_risk_tolerance() -> None:
    market = Market.load(TEST_DATA)
    market = market.head(300)
    optimizer = MeanVarianceOptimizer(risk_tolerance=1)
    optimizer.optimize(market, market.get_last_date())
    weights = optimizer.get_weights()
    npt.assert_almost_equal(weights, [0.331342, 0.668658])


def test_mean_variance_optimizer_with_cost_tolerance() -> None:
    market = Market.load(TEST_DATA)
    market = market.head(300)
    optimizer = MeanVarianceOptimizer(cost_tolerance=0.1)
    v = optimizer.optimize(market, market.get_last_date())
    print(v)
    weights = optimizer.get_weights()
    npt.assert_almost_equal(weights, [0.2721083, 0.7278917])


def test_minimum_sharp_ratio_optimizer() -> None:
    market = Market.load(TEST_DATA)
    market = market.head(200)
    optimizer = SharpRatioOptimizer(risk_free_rate=0.02)
    optimizer.optimize(market, market.get_last_date())
    weights = optimizer.get_weights()
    npt.assert_almost_equal(weights, [0.47694, 0.52306], decimal=5)


def test_semi_minimum_variance_optimizer() -> None:
    market = Market.load(TEST_DATA)
    market = market.head(200)
    optimizer = MinimumVarianceOptimizer(with_semi_covariance=True)
    optimizer.optimize(market, market.get_last_date())
    weights = optimizer.get_weights()
    npt.assert_almost_equal(weights, [0.33742, 0.66258], decimal=5)


def test_maximum_diversification_optimizer() -> None:
    market = Market.load(TEST_DATA)
    market = market.head(200)
    optimizer = MaximumDiversificationOptimizer()
    optimizer.optimize(market, market.get_last_date())
    weights = optimizer.get_weights()
    npt.assert_almost_equal(weights, [0.3867, 0.6133], decimal=5)


def test_maximum_diversification_optimizer_with_cost_tolerance() -> None:
    market = Market.load(TEST_DATA)
    market = market.head(200)
    optimizer = MaximumDiversificationOptimizer(cost_tolerance=0.01)
    optimizer.optimize(market, market.get_last_date())
    weights = optimizer.get_weights()
    npt.assert_almost_equal(weights, [0.93929, 0.06071], decimal=5)


def test_risk_parity_optimizer() -> None:
    market = Market.load(TEST_DATA)
    market = market.head(200)
    optimizer = RiskParityOptimizer()
    optimizer.optimize(market, market.get_last_date())
    weights = optimizer.get_weights()
    npt.assert_almost_equal(weights, [0.3867, 0.6133], decimal=5)


def test_risk_parity_optimizer_with_target() -> None:
    market = Market.load(TEST_DATA)
    market = market.head(200)
    optimizer = RiskParityOptimizer(risk_target=[0.75, 0.25])
    optimizer.optimize(market, market.get_last_date())
    weights = optimizer.get_weights()
    npt.assert_almost_equal(weights, [0.46959, 0.53041], decimal=5)


def test_cvar_optimizer() -> None:
    market = Market.load(TEST_DATA)
    market = market.head(200)
    optimizer = CVaROptimizer(sampler=Sampler.load(MODEL_DATA))
    optimizer.optimize(market, market.get_last_date())
    weights = optimizer.get_weights()
    npt.assert_almost_equal(weights, [0.44782, 0.55218], decimal=5)


@skip_long_tests
def test_cvar_optimizer_with_sampling() -> None:
    market = Market.load(TEST_DATA)
    market = market.head(200)
    optimizer = CVaROptimizer(samples=200)
    optimizer.optimize(market, market.get_last_date())
    weights = optimizer.get_weights()
    npt.assert_almost_equal(weights, [0.37332, 0.62668], decimal=5)
