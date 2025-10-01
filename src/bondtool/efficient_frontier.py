# src/bondtool/efficient_frontier.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from .portfolio import PortfolioInputs, efficient_frontier


def plot_frontier(
    exp_returns: pd.Series,
    cov: pd.DataFrame,
    rf: float = 0.0,
    n_points: int = 100,
    out_path: str = "efficient_frontier.png",
    show: bool = False,
) -> pd.DataFrame:
    """
    Generate efficient frontier given expected returns and covariance.

    Parameters
    ----------
    exp_returns : pd.Series
        Expected returns, indexed by asset.
    cov : pd.DataFrame
        Covariance matrix of returns.
    rf : float, optional
        Risk-free rate. Default = 0.0.
    n_points : int, optional
        Number of points to plot along the frontier. Default = 100.
    out_path : str, optional
        Output path for saving the plot (default: 'efficient_frontier.png').
    show : bool, optional
        If True, displays the plot interactively.

    Returns
    -------
    pd.DataFrame
        DataFrame of efficient frontier points (expected return, vol, weights).
    """

    # Build inputs and compute efficient frontier
    inputs = PortfolioInputs(exp_returns=exp_returns, cov=cov, rf=rf)
    df = efficient_frontier(inputs, n_points=n_points, short=False)

    # Plot
    plt.figure(figsize=(7, 5))
    plt.plot(df["vol"], df["exp_return"], color="blue", lw=2)
    plt.xlabel("Volatility (Std Dev)")
    plt.ylabel("Expected Return")
    plt.title("Efficient Frontier")
    plt.grid(True, linestyle="--", alpha=0.6)

    if out_path:
        plt.savefig(out_path, bbox_inches="tight")
        print(f"Saved plot: {out_path}")

    if show:
        plt.show()

    return df
