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
    max_assets = total_assets[0]
    minimum_drawdown = np.float64(1e6)
    for a in total_assets:
        if a >= max_assets:
            max_assets = a
        this_drawdown = a / max_assets
        if this_drawdown < minimum_drawdown:
            minimum_drawdown = this_drawdown
    return np.float64(np.min(minimum_drawdown))


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
    drawdown_span = 0
    for a in total_assets:
        if a >= max_assets:
            not_update_max_of_asset = 0
            max_assets = a
        else:
            not_update_max_of_asset += 1

        if not_update_max_of_asset > drawdown_span:
            drawdown_span = not_update_max_of_asset

    return drawdown_span
