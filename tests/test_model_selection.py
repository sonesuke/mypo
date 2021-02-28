import os

from mypo import Market, split_n_periods
from mypo import MinimumVarianceOptimizer, ThresholdRebalancer, Runner, negative_total_return

TEST_DATA = os.path.join(os.path.dirname(__file__), "data", "test.bin")


def test_split_n_periods():
    market = Market.load(TEST_DATA)
    market = market.extract(market.get_index()[:90])
    train, eval = split_n_periods(market=market, n=4, train_span=10)

    for t in train:
        assert len(t.get_index()) == 10

    for e in eval:
        assert len(e.get_index()) == 20


def test_cross_validation():
    market = Market.load(TEST_DATA)
    market = market.extract(market.get_index()[:1200])
    trains, evals = split_n_periods(market=market, n=10)

    for train, eval in zip(trains, evals):
        optimizer=MinimumVarianceOptimizer(train.get_prices())
        weights = optimizer.optimize_weight()
        print(weights)
        runner = Runner(
            assets=[1.2, 0.8], rebalancer=ThresholdRebalancer(weights=weights, threshold=0.05), cash=0.5, spending=0.06)
        runner.run(market=eval)
        report = runner.report()
        print(negative_total_return(report))
    assert False