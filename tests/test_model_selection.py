import os

from mypo import Market, clustering_tickers, evaluate_combinations, split_n_periods
from mypo.optimizer import MaximumDiversificationOptimizer

TEST_DATA = os.path.join(os.path.dirname(__file__), "data", "test.bin")


def test_split_n_periods() -> None:
    market = Market.load(TEST_DATA)
    market = market.head(90)
    train, eval = split_n_periods(market=market, n=4, train_span=10)

    for t in train:
        assert len(t.get_index()) == 10

    for e in eval:
        assert len(e.get_index()) == 20


def test_clustering_tickers() -> None:
    market = Market.load(TEST_DATA)
    c = clustering_tickers(market, n=2)
    df = evaluate_combinations(market, c, MaximumDiversificationOptimizer(span=200))
    assert df is not None
