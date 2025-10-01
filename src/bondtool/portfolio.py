from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Sequence

@dataclass
class PortfolioInputs:
    exp_returns: pd.Series      # expected returns (annualized, decimals)
    cov: pd.DataFrame           # covariance matrix (annualized)
    rf: float = 0.0             # risk-free rate (decimal)

def normalize_weights(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    if (w < -1e-12).any():
        raise ValueError("Weights must be non-negative for this simple model.")
    s = w.sum()
    if s <= 0:
        raise ValueError("Sum of weights must be positive.")
    return w / s

def portfolio_stats(inputs: PortfolioInputs, weights: Sequence[float]) -> dict:
    w = normalize_weights(np.array(weights))
    mu = inputs.exp_returns.values
    cov = inputs.cov.values
    exp_ret = float(w @ mu)
    var = float(w @ cov @ w)
    vol = float(np.sqrt(var))
    sharpe = (exp_ret - inputs.rf) / vol if vol > 0 else np.nan
    return {"exp_return": exp_ret, "vol": vol, "var": var, "sharpe": sharpe, "weights": w}

def efficient_frontier(inputs: PortfolioInputs, n_points: int = 100, short: bool = False) -> pd.DataFrame:
    # Quadratic programming via numpy pseudo-inverse (no external solvers).
    mu = inputs.exp_returns.values
    cov = inputs.cov.values
    ones = np.ones_like(mu)
    inv = np.linalg.pinv(cov)
    A = ones @ inv @ ones
    B = ones @ inv @ mu
    C = mu @ inv @ mu
    # Target returns between min and max expected returns
    mus = np.linspace(mu.min(), mu.max(), n_points)
    rows = []
    for m in mus:
        # Lagrange multipliers for min variance at target return m
        # w* = inv(Σ) (λ1 * 1 + λ2 * μ)
        denom = A * C - B * B
        lam1 = (C - B * m) / denom
        lam2 = (A * m - B) / denom
        w = inv @ (lam1 * ones + lam2 * mu)
        if not short:
            w = np.clip(w, 0, None)
            w = w / w.sum()
        exp_ret = float(w @ mu)
        var = float(w @ cov @ w)
        vol = float(np.sqrt(var))
        rows.append({"target_mu": m, "exp_return": exp_ret, "vol": vol, **{f"w_{i}": wi for i, wi in enumerate(w)}})
    return pd.DataFrame(rows)
