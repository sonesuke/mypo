"""Utility functions for model selection."""
import itertools
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2

from mypo.market import Market


def split_n_periods(market: Market, n: int, train_span: int) -> Tuple[List[Market], List[Market]]:
    """Split market to n periods.

    Args:
        market: Market data
        n: Count of split
        train_span: Train span if you want to specify.

    Returns:
        Split market data
    """
    index = market.get_index()
    len_index = len(index)

    eval_span = int((len_index - train_span) / n)
    train_market: List[Market] = []
    eval_market: List[Market] = []
    for i in range(n):
        start_eval_index = i * eval_span + train_span
        train_market += [market.extract(index[start_eval_index - train_span : start_eval_index])]
        eval_market += [market.extract(index[start_eval_index : start_eval_index + eval_span])]

    return train_market, eval_market


def clustering_tickers(market: Market, n: int, seed: int = 32) -> pd.DataFrame:
    """Enumerate tickers.

    Args:
        market: Market.
        n: Count of tickers.
        seed: Seed.

    Returns:
        Clustered tickers.
    """
    np.random.seed(seed)
    corr = market.get_rate_of_change().corr()

    _, label = kmeans2(corr.to_numpy(), k=n, minit="++")
    return pd.DataFrame({"class": label}, index=corr.index)


def make_combinations(cluster: pd.DataFrame) -> List[Tuple[str]]:
    """Make combinations.

    Args:
        cluster: Cluster data.

    Returns:
        Combinations.
    """
    k = cluster["class"].max() + 1
    clusters = []
    for i in range(k):
        clusters += [list(cluster[cluster["class"] == i].index)]
    return list(itertools.product(*clusters))
