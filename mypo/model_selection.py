"""Utility functions for model selection."""
import itertools
from datetime import datetime
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.cluster.vq import kmeans2
from tqdm import tqdm

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
    df = pd.DataFrame({"class": label}, index=corr.index)
    return df.sort_values("class")


def _make_combinations(cluster: pd.DataFrame) -> List[Tuple[str]]:
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


def evaluate_combinations(market: Market, cluster: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Evaluate combinations.

    Args:
        market: Market.
        cluster: Cluster data.
        verbose: Verbose mode.

    Returns:
        Result.
    """

    combinations = _make_combinations(cluster)

    def proc(c: Any) -> Any:  # pragma: no cover
        from mypo.optimizer import MinimumVarianceOptimizer

        optimizer = MinimumVarianceOptimizer(span=50000)
        target_market = market.filter(list(c))
        optimizer.optimize(target_market, at=datetime.today())
        w = optimizer.get_weights()
        r = target_market.get_rate_of_change().to_numpy()
        q = np.dot(np.dot(w, np.cov(r.T)), w.T)
        r = np.dot(w, r.mean(axis=0))
        return target_market.get_tickers(), r, q, w

    def wrap(x: Any, total: Any) -> Any:
        """Wrapper for tqdm."""
        return tqdm(x, total=total) if verbose else x

    result = Parallel(n_jobs=-1)(delayed(proc)(c) for c in wrap(combinations, total=len(combinations)))

    df = pd.DataFrame(
        {
            "c": [r[0] for r in result],
            "r": [r[1] for r in result],
            "q": [r[2] for r in result],
            "w": [r[3] for r in result],
        }
    )
    df["sharp ratio"] = df["r"] / df["q"]
    return df
