import numpy.testing as npt
import os
from datetime import date
from mypo import Runner
from mypo import PlainRebalancer, MonthlyRebalancer
from mypo import Loader


TEST_DATA = os.path.join(os.path.dirname(__file__), 'data', 'test.bin')


def test_simulator():
    runner = Runner(
        assets=[1, 1],
        rebalancer=PlainRebalancer([0.6, 0.4]),
        cash=0.5,
        spending=0.06
    )
    npt.assert_almost_equal(
        runner.total_assets(),
        2.5
    )


def test_apply():
    runner = Runner(
        assets=[1.2, 0.8],
        rebalancer=PlainRebalancer([0.6, 0.4]),
        cash=0.5,
        spending=0.06
    )
    runner.apply(
        index=date.today(),
        market=[1.1, 0.9],
        price_dividends_yield=[0.05, 0.01],
        expense_ratio=[0.0007, 0.0007]
    )
    npt.assert_almost_equal(
        runner.total_assets(),
        2.625068
    )


def test_run():
    loader = Loader.load(TEST_DATA)
    runner = Runner(
        assets=[1.2, 0.8],
        rebalancer=PlainRebalancer([0.8, 0.2]),
        cash=0.5,
        spending=0.06
    )
    runner.run(
        market=loader.get_market(),
        price_dividends_yield=loader.get_price_dividend_yield(),
        expense_ratio=[0.0007, 0.0007]
    )
    npt.assert_almost_equal(
        runner.total_assets(),
        2.2437560
    )


def test_monthly_run():
    loader = Loader.load(TEST_DATA)
    runner = Runner(
        assets=[1.2, 0.8],
        rebalancer=MonthlyRebalancer([0.8, 0.2]),
        cash=0.5,
        spending=0.06
    )
    runner.run(
        market=loader.get_market(),
        price_dividends_yield=loader.get_price_dividend_yield(),
        expense_ratio=[0.0007, 0.0007]
    )
    npt.assert_almost_equal(
        runner.total_assets(),
        2.1252432
    )
