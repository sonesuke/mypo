import os

import pandas as pd

from mypo import Market, Runner, split_k_folds
from mypo.rebalancer import PlainRebalancer

TEST_DATA = os.path.join(os.path.dirname(__file__), "data", "test.bin")


def test_runner() -> None:
    runner = Runner(rebalancer=PlainRebalancer([0.6, 0.4]))
    assert runner is not None


def test_run_and_report() -> None:
    market = Market.load(TEST_DATA)
    market = market.extract(market.get_index()[:100])
    runner = Runner(
        rebalancer=PlainRebalancer([0.8, 0.2]),
    )
    runner.run(
        market=market,
        assets=[1.2, 0.8],
        cash=0.5,
        withdraw=0.06,
    )
    report = runner.report().history()
    assert report.index[0] == pd.Timestamp("2010-09-09")


def test_run_fold_and_report() -> None:
    market = Market.load(TEST_DATA)
    market = market.extract(market.get_index()[:100])
    folds = split_k_folds(market, 1, 10)
    runner = Runner(
        rebalancer=PlainRebalancer([0.8, 0.2]),
    )
    runner.run(fold=folds[0], cash=0.5, withdraw=0.06)
    report = runner.report().history()
    assert report.index[0] == pd.Timestamp("2010-09-23")
