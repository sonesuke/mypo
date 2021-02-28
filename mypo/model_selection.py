"""Utility functions for model selection."""
from typing import List, Tuple

from .market import Market


def split_n_periods(
    market: Market, n: int, train_span: int = None
) -> Tuple[List[Market], List[Market]]:
    """
    Split market to n periods.

    Parameters
    ----------
    market
        Market data
    n
        Count of split

    train_span
        Train span if you want to specify.

    Returns
    -------
    Split market data
    """
    index = market.get_index()
    len_index = len(index)
    if train_span is None:
        train_span = int(len_index / (n + 1))

    eval_span = int((len_index - train_span) / n)
    train_market: List[Market] = []
    eval_market: List[Market] = []
    for i in range(n):
        start_eval_index = i * eval_span + train_span
        train_market += [
            market.extract(index[start_eval_index - train_span : start_eval_index])
        ]
        eval_market += [
            market.extract(index[start_eval_index : start_eval_index + eval_span])
        ]

    return train_market, eval_market
