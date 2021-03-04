"""Loss functions."""
import numpy as np
import pandas as pd


def negative_total_return(report: pd.DataFrame) -> np.float64:
    """
    Get negative total return.

    Parameters
    ----------
    report
        Result of simulation.

    Returns
    -------
        Negative tatal return.
    """
    total_assets = report["total_assets"]
    return -np.float64(total_assets[len(total_assets) - 1] / total_assets[0])


def max_drawdown(report: pd.DataFrame) -> np.float64:
    """
    Get negative total return.

    Parameters
    ----------
    report
        Result of simulation.

    Returns
    -------
        Negative tatal return.
    """
    max_assets = report["total_assets"].cummax()
    return np.float64(np.min(report["total_assets"] / max_assets))


def max_drawdown_span(report: pd.DataFrame) -> int:
    """
    Get negative total return.

    Parameters
    ----------
    report
        Result of simulation.

    Returns
    -------
        Negative tatal return.
    """
    df = report[["total_assets"]].copy()
    df["max_total"] = df["total_assets"] < df["total_assets"].cummax()
    df["continuous"] = (
        df.groupby((df["max_total"] != df["max_total"].shift()).cumsum()).cumcount() + 1
    )
    df = df[df["max_total"]]
    ret: int = np.max(df["continuous"])
    return ret
