import os

from mypo import Market, split_n_periods

TEST_DATA = os.path.join(os.path.dirname(__file__), "data", "test.bin")


def test_split_n_periods() -> None:
    market = Market.load(TEST_DATA)
    market = market.extract(market.get_index()[:90])
    train, eval = split_n_periods(market=market, n=4, train_span=10)

    for t in train:
        assert len(t.get_index()) == 10

    for e in eval:
        assert len(e.get_index()) == 20


def test_split_n_periods_without_specified_train_span() -> None:
    market = Market.load(TEST_DATA)
    market = market.extract(market.get_index()[:100])
    train, eval = split_n_periods(market=market, n=4)

    for t in train:
        assert len(t.get_index()) == 20

    for e in eval:
        assert len(e.get_index()) == 20
