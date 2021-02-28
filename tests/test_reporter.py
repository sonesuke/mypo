import datetime

import pandas as pd

from mypo import Reporter


def test_report():
    reporter = Reporter()
    reporter.record(datetime.datetime(2021, 2, 22), 0.1, 0.2, 0.5, 0.1, 0.2, 0.3, 0.4)
    df = reporter.report()
    assert df.index[0] == pd.Timestamp("2021-02-22")
