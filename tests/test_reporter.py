import datetime

import numpy as np
import numpy.testing as npt
import pandas as pd

from mypo import Market, Reporter, Runner


def test_report():
    reporter = Reporter()
    reporter.record(
        datetime.datetime(2021, 2, 22),
        np.float64(1.0),
        np.float64(0.1),
        np.float64(0.2),
        np.float64(0.5),
        np.float64(0.1),
        np.float64(0.2),
        np.float64(0.3),
        np.float64(0.4),
    )
    df = reporter.report()
    assert df.index[0] == pd.Timestamp("2021-02-22")


def test_run_and_report():
    market = Market.create(start_date="2021-01-01", end_date="2021-12-31", yearly_gain=0.00)

    initial_cash = 0.5
    withdraw = 0.05
    runner = Runner(cash=initial_cash, withdraw=withdraw)
    runner.run(market=market)
    df = runner.report()
    npt.assert_almost_equal(initial_cash - df["total_assets"].min(), withdraw)
    assert not df.isnull().to_numpy().any()
    npt.assert_almost_equal(df["capital_gain"].sum(), 0)
    npt.assert_almost_equal(df["income_gain"].sum(), 0)
    npt.assert_almost_equal(df["deal"].sum(), 0)
    npt.assert_almost_equal(df["fee"].sum(), 0)
