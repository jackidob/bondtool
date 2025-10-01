import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .portfolio import PortfolioInputs, efficient_frontier

def plot_frontier(exp_returns: pd.Series, cov: pd.DataFrame, rf: float = 0.0, n_points: int = 100, out_path: str = None):
    inputs = PortfolioInputs(exp_returns=exp_returns, cov=cov, rf=rf)
    df = efficient_frontier(inputs, n_points=n_points, short=False)
    plt.figure()
    plt.plot(df['vol'], df['exp_return'])
    plt.xlabel('Volatility (Std Dev)')
    plt.ylabel('Expected Return')
    plt.title('Efficient Frontier')
    if out_path:
        plt.savefig(out_path, bbox_inches='tight')
    return df
