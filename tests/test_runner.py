import datetime
import os

import numpy.testing as npt
import pandas as pd

from mypo import Market, MonthlyRebalancer, PlainRebalancer, Runner

TEST_DATA = os.path.join(os.path.dirname(__file__), "data", "test.bin")


def test_simulator():
    runner = Runner(
        assets=[1, 1], rebalancer=PlainRebalancer([0.6, 0.4]), cash=0.5, spending=0.06
    )
    npt.assert_almost_equal(runner.total_assets(), 2.5)


def test_apply():
    runner = Runner(
        assets=[1.2, 0.8],
        rebalancer=PlainRebalancer([0.6, 0.4]),
        cash=0.5,
        spending=0.06,
    )
    runner.apply(
        index=datetime.datetime(2021, 2, 17),
        prices=[1.1, 0.9],
        price_dividends_yield=[0.05, 0.01],
        expense_ratio=[0.0007, 0.0007],
    )
    npt.assert_almost_equal(runner.total_assets(), 2.6303305)


def test_run_and_report():
    market = Market.load(TEST_DATA)
    market = market.extract(market.get_index()[:100])
    runner = Runner(
        assets=[1.2, 0.8],
        rebalancer=PlainRebalancer([0.8, 0.2]),
        cash=0.5,
        spending=0.06,
    )
    runner.run(market=market)
    report = runner.report()
    assert report.index[1] == pd.Timestamp("2010-09-10")
