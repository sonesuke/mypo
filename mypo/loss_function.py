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
    total_assets = list(report["total_assets"])
    minimum_return = 1e6
    r = 1.0
    previous_assets = total_assets[0]
    for a in total_assets:
        this_return = a / previous_assets
        if this_return > 1.0:
            r = 1.0
        else:
            r *= this_return
        if r < minimum_return:
            minimum_return = r
    return np.float64(minimum_return)


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
    total_assets = list(report["total_assets"])
    max_assets = total_assets[0]
    not_update_max_of_asset = 0
    for a in total_assets:
        if a > max_assets:
            not_update_max_of_asset = 0
            max_assets = a
        else:
            not_update_max_of_asset += 1

    return not_update_max_of_asset
