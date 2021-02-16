import numpy as np
import numpy.testing as npt
import os
from datetime import date
from mypo import Runner
from mypo import PlainRebalancer, MonthlyRebalancer
from mypo import Loader


TEST_DATA = os.path.join(os.path.dirname(__file__), 'data', 'test.bin')


def test_simulator():
    runner = Runner(
        assets=np.array([1, 1]),
        rebalancer=PlainRebalancer(np.array([0.6, 0.4])),
        cash=0.5,
        spending=0.06
    )
    npt.assert_almost_equal(
        runner.total_assets(),
        2.5
    )


def test_apply():
    runner = Runner(
        assets=np.array([1.2, 0.8]),
        rebalancer=PlainRebalancer(np.array([0.6, 0.4])),
        cash=0.5,
        spending=0.06
    )
    runner.apply(
        index=date.today(),
        market=np.array([1.1, 0.9]),
        price_dividends_yield=np.array([0.05, 0.01]),
        expense_ratio=np.array([0.0007, 0.0007])
    )
    npt.assert_almost_equal(
        runner.total_assets(),
        2.625068
    )


def test_run():
    loader = Loader.load(TEST_DATA)
    runner = Runner(
        assets=np.array([1.2, 0.8]),
        rebalancer=PlainRebalancer(np.array([0.8, 0.2])),
        cash=0.5,
        spending=0.06
    )
    runner.run(
        index=loader.get_index(),
        market=loader.get_market(),
        price_dividends_yield=loader.get_price_dividend_yield(),
        expense_ratio=np.array([0.0007, 0.0007])
    )
    npt.assert_almost_equal(
        runner.total_assets(),
        2.2437560
    )


def test_monthly_run():
    loader = Loader.load(TEST_DATA)
    runner = Runner(
        assets=np.array([1.2, 0.8]),
        rebalancer=MonthlyRebalancer(np.array([0.8, 0.2])),
        cash=0.5,
        spending=0.06
    )
    runner.run(
        index=loader.get_index(),
        market=loader.get_market(),
        price_dividends_yield=loader.get_price_dividend_yield(),
        expense_ratio=np.array([0.0007, 0.0007])
    )
    npt.assert_almost_equal(
        runner.total_assets(),
        2.1252432
    )
