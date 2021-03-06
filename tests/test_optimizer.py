import os

import numpy.testing as npt
import pytest

from mypo import Market
from mypo.optimizer import (
    CVaROptimizer,
    MaximumDiversificationOptimizer,
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
    npt.assert_almost_equal(weights, [0.2497605, 0.7502395])


def test_minimum_variance_optimizer_with_minimum_return() -> None:
    market = Market.load(TEST_DATA)
    market = market.head(200)
    optimizer = MinimumVarianceOptimizer(minimum_return=0.10)
    optimizer.optimize(market, market.get_last_date())
    weights = optimizer.get_weights()
    npt.assert_almost_equal(weights, [0.33055, 0.66945], decimal=5)


def test_minimum_sharp_ratio_optimizer() -> None:
    market = Market.load(TEST_DATA)
    market = market.head(200)
    optimizer = SharpRatioOptimizer(risk_free_rate=0.02)
    optimizer.optimize(market, market.get_last_date())
    weights = optimizer.get_weights()
    npt.assert_almost_equal(weights, [0.4380607, 0.5619393], decimal=5)


def test_semi_minimum_variance_optimizer() -> None:
    market = Market.load(TEST_DATA)
    market = market.head(200)
    optimizer = MinimumVarianceOptimizer(with_semi_covariance=True)
    optimizer.optimize(market, market.get_last_date())
    weights = optimizer.get_weights()
    npt.assert_almost_equal(weights, [0.3239604, 0.6760396], decimal=5)


def test_maximum_diversification_optimizer() -> None:
    market = Market.load(TEST_DATA)
    market = market.head(200)
    optimizer = MaximumDiversificationOptimizer()
    optimizer.optimize(market, market.get_last_date())
    weights = optimizer.get_weights()
    npt.assert_almost_equal(weights, [0.38076, 0.61924], decimal=5)


def test_risk_parity_optimizer() -> None:
    market = Market.load(TEST_DATA)
    market = market.head(200)
    optimizer = RiskParityOptimizer()
    optimizer.optimize(market, market.get_last_date())
    weights = optimizer.get_weights()
    npt.assert_almost_equal(weights, [0.38077, 0.61923], decimal=5)


def test_risk_parity_optimizer_with_target() -> None:
    market = Market.load(TEST_DATA)
    market = market.head(200)
    optimizer = RiskParityOptimizer(risk_target=[0.75, 0.25])
    optimizer.optimize(market, market.get_last_date())
    weights = optimizer.get_weights()
    npt.assert_almost_equal(weights, [0.4626, 0.5374], decimal=5)


def test_cvar_optimizer() -> None:
    market = Market.load(TEST_DATA)
    market = market.head(200)
    optimizer = CVaROptimizer(sampler=Sampler.load(MODEL_DATA))
    optimizer.optimize(market, market.get_last_date())
    weights = optimizer.get_weights()
    npt.assert_almost_equal(weights, [0.1942, 0.8058], decimal=5)


@skip_long_tests
def test_cvar_optimizer_with_sampling() -> None:
    market = Market.load(TEST_DATA)
    market = market.head(200)
    optimizer = CVaROptimizer(samples=200)
    optimizer.optimize(market, market.get_last_date())
    weights = optimizer.get_weights()
    npt.assert_almost_equal(weights, [0.36126, 0.63874], decimal=5)
