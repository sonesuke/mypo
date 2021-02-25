import numpy as np
import pandas as pd


def negative_total_return(report: pd.DataFrame) -> np.float64:
    ret: np.float64
    ret = np.sum(report["capital_gain"]) - np.sum(report["capital_gain_tax"])
    ret += np.sum(report["income_gain"]) - np.sum(report["income_gain_tax"])
    ret += np.sum(report["fee"])
    return -ret
