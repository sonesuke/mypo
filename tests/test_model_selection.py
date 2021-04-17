import os

import numpy.testing as npt

from mypo import (
    Market,
    clustering_tickers,
    evaluate_combinations,
    select_by_correlation,
    select_by_regression,
    split_k_folds,
)
from mypo.optimizer import MaximumDiversificationOptimizer

TEST_DATA = os.path.join(os.path.dirname(__file__), "data", "test.bin")
TEST_ALL_DATA = os.path.join(os.path.dirname(__file__), "data", "all.bin")


def test_split_k_folds() -> None:
    market = Market.load(TEST_DATA)
    market = market.head(90)
    folds = split_k_folds(market=market, k=4, train_span=10)

    for f in folds:
        assert len(f.get_train().get_index()) == 10
        assert len(f.get_valid().get_index()) == 30
        filtered = f.filter(["VOO"])
        assert filtered.get_train().get_tickers() == ["VOO"]


def test_clustering_tickers() -> None:
    market = Market.load(TEST_DATA)
    c = clustering_tickers(market, n=2)
    df = evaluate_combinations(market, c, MaximumDiversificationOptimizer(span=200))
    assert df is not None


def test_select_by_variance() -> None:
    from datetime import datetime

    from mypo import Loader

    loader = Loader.load(TEST_ALL_DATA)
    loader = loader.since(datetime(2005, 1, 1))

    market = loader.get_market()
    tickers = select_by_correlation(market, 0.8)
    npt.assert_equal(len(tickers), 66)


def test_select_by_regression() -> None:
    from datetime import datetime

    from mypo import Loader

    loader = Loader.load(TEST_ALL_DATA)
    loader = loader.since(datetime(2005, 1, 1))

    market = loader.get_market()
    tickers = select_by_regression(market, 0.9)
    npt.assert_equal(len(tickers), 5)
