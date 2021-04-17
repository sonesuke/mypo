import os

from mypo import Market, clustering_tickers, evaluate_combinations, split_k_folds
from mypo.optimizer import MaximumDiversificationOptimizer

TEST_DATA = os.path.join(os.path.dirname(__file__), "data", "test.bin")


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
