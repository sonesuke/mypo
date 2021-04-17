import os

import numpy.testing as npt
import pandas as pd

from mypo import Market, Runner, split_k_folds
from mypo.rebalancer import PlainRebalancer

TEST_DATA = os.path.join(os.path.dirname(__file__), "data", "test.bin")


def test_simulator() -> None:
    runner = Runner(assets=[1, 1], rebalancer=PlainRebalancer([0.6, 0.4]), cash=0.5, withdraw=0.06)
    npt.assert_almost_equal(runner.total_assets(), 2.5)


def test_apply() -> None:
    market = Market.load(TEST_DATA)
    market = market.extract(market.get_index()[:100])
    runner = Runner(
        assets=[1.2, 0.8],
        rebalancer=PlainRebalancer([0.6, 0.4]),
        cash=0.5,
        withdraw=0.0,
    )
    runner.apply(market, 0)
    npt.assert_almost_equal(runner.total_assets(), 2.50309807503)


def test_run_and_report() -> None:
    market = Market.load(TEST_DATA)
    market = market.extract(market.get_index()[:100])
    runner = Runner(
        assets=[1.2, 0.8],
        rebalancer=PlainRebalancer([0.8, 0.2]),
        cash=0.5,
        withdraw=0.06,
    )
    runner.run(market=market)
    report = runner.report()
    assert report.index[0] == pd.Timestamp("2010-09-09")


def test_run_fold_and_report() -> None:
    market = Market.load(TEST_DATA)
    market = market.extract(market.get_index()[:100])
    folds = split_k_folds(market, 1, 10)
    runner = Runner(
        assets=[1.2, 0.8],
        rebalancer=PlainRebalancer([0.8, 0.2]),
        cash=0.5,
        withdraw=0.06,
    )
    runner.run_fold(folds[0])
    report = runner.report()
    assert report.index[0] == pd.Timestamp("2010-09-23")
