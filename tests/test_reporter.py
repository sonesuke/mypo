import datetime
import os
import pandas as pd

from mypo import Reporter

TEST_DATA = os.path.join(os.path.dirname(__file__), "data", "test.bin")


def test_report():
    reporter = Reporter()
    reporter.record(datetime.datetime(2021, 2, 22), 0.1, 0.2, 0.5, 0.1, 0.2, 0.3, 0.4)
    df = reporter.report()
    assert df.index[0] == pd.Timestamp("2021-02-22")



