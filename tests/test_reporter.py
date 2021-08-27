import numpy as np
import numpy.testing as npt

from mypo import Market, Runner, Settings


def test_run_and_report() -> None:
    market = Market.create(start_date="2021-01-01", end_date="2021-12-31", yearly_gain=0.00)

    runner = Runner()
    runner.run(assets=[0], market=market, cash=0.5, withdraw=0.05)
    df = runner.report().history()
    assert not df.isnull().to_numpy().any()
    npt.assert_almost_equal(0.5 - df["total_assets"].min(), 0.05)
    npt.assert_almost_equal(df["capital_gain"].sum(), 0)
    npt.assert_almost_equal(df["income_gain"].sum(), 0)
    npt.assert_almost_equal(df["deal"].sum(), 0)
    npt.assert_almost_equal(df["fee"].sum(), 0)


def test_run_and_report_run_out_cash() -> None:
    market = Market.create(start_date="2021-01-01", end_date="2021-12-31", yearly_gain=0.00)

    runner = Runner(
        settings=Settings(tax_rate=np.float64(0.0), fee_rate=np.float64(0.0)),
    )
    runner.run(assets=[0.6], market=market, cash=0.6, withdraw=1.2)
    report = runner.report()
    assert report.summary() is not None
    assert report.history_weights() is not None
    assert report.history_cost() is not None
    assert report.history_assets() is not None
    assert report.history_cash_vs_assets() is not None


def test_run_and_report_annual() -> None:
    market = Market.create(start_date="2021-01-01", end_date="2023-12-31", yearly_gain=0.01)

    runner = Runner(
        settings=Settings(tax_rate=np.float64(0.0), fee_rate=np.float64(0.0)),
    )
    runner.run(assets=[1.0], market=market, cash=0.0, withdraw=0)
    report = runner.report()
    assert report.annual_summary() is not None
