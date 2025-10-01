# src/bondtool/main_cli.py
import argparse
import ast
import os
import sys
from typing import List, Optional

import numpy as np
import pandas as pd

# package-relative imports (works when installed as bondtool)
from .bonds import Bond
from .portfolio import PortfolioInputs, portfolio_stats
from .efficient_frontier import plot_frontier


# ---------- helpers ----------
def _require_file(path: str) -> None:
    if not os.path.isfile(path):
        raise SystemExit(f"Error: file not found: {path}")


def _print_kv(title: str, mapping: dict) -> None:
    print(f"\n{title}:")
    for k, v in mapping.items():
        if isinstance(v, (float, np.floating)):
            print(f"{k} {float(v):.6f}")
        elif isinstance(v, (list, np.ndarray, pd.Series)):
            arr = np.array(v)
            print(f"{k} {np.round(arr, 6)}")
        else:
            print(f"{k} {v}")


# ---------- commands ----------
def cmd_bond(file: str, out: str) -> None:
    _require_file(file)
    df = pd.read_csv(file)

    required_cols = {"face", "coupon_rate", "maturity_years"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise SystemExit(f"Error: missing required columns: {sorted(missing)}")

    # at least one of price / ytm must exist (as columns), rows can leave one blank
    if not ({"price"} <= set(df.columns) or {"ytm"} <= set(df.columns)):
        raise SystemExit("Error: CSV must include at least one of the columns: price or ytm")

    print("Loaded bonds:")
    print(df)

    rows = []
    for idx, r in df.iterrows():
        name = r.get("name", f"bond_{idx}")

        face = float(r["face"])
        coupon_rate = float(r["coupon_rate"])
        maturity_years = float(r["maturity_years"])
        freq = int(r.get("freq", 2))

        # may be NaN
        price = None if "price" not in r or pd.isna(r["price"]) else float(r["price"])
        ytm = None if "ytm" not in r or pd.isna(r["ytm"]) else float(r["ytm"])

        if price is None and ytm is None:
            raise SystemExit(f"Row {idx} ({name}): need price or ytm")

        b = Bond(
            face=face,
            coupon_rate=coupon_rate,
            price=price,
            ytm=ytm,
            maturity_years=maturity_years,
            freq=freq,
        )

        # compute the missing leg
        if b.ytm is None:
            y = b.ytm_from_price()
            p = b.price  # given
        elif b.price is None:
            y = b.ytm
            p = b.price_from_ytm(y)
        else:
            # both provided; trust y, recompute p to ensure consistent metrics
            y = b.ytm
            p = b.price_from_ytm(y)

        dur_mac = b.macaulay_duration(y)
        dur_mod = b.modified_duration(y)
        conv = b.convexity(y)

        row = {
            "name": name,
            "ytm": y,
            "price": p,
            "macaulay_dur": dur_mac,
            "modified_dur": dur_mod,
            "convexity": conv,
        }

        # optional DV01 if your Bond implements it
        dv01 = None
        dv01_fn = getattr(b, "dv01", None)
        if callable(dv01_fn):
            try:
                dv01 = float(dv01_fn(y))
                row["DV01"] = dv01
            except Exception:
                # keep going even if DV01 fails
                pass

        rows.append(row)

    out_df = pd.DataFrame(rows)
    print("\nBond analytics:")
    print(out_df.round(6))

    # write results
    out_path = out or "bond_analytics_output.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


def cmd_frontier(file: str, points: int, rf: float, out: str) -> None:
    _require_file(file)
    df = pd.read_csv(file, index_col=0)

    if "exp_return" not in df.columns:
        raise SystemExit("Error: CSV must include an 'exp_return' column (expected returns index-aligned with assets).")

    mu = df["exp_return"]
    cov = df.drop(columns=["exp_return"])

    # minimal sanity: covariance should be square & match asset count
    if cov.shape[0] != cov.shape[1] or cov.shape[0] != mu.shape[0]:
        raise SystemExit("Error: covariance matrix must be square and match the length/order of exp_return.")

    out_df = plot_frontier(mu, cov, rf=rf, n_points=points, out_path=out or "efficient_frontier.png")
    print(out_df.head())

    print(f"\nSaved plot: {out or 'efficient_frontier.png'}")


def cmd_portfolio(file: str, weights: str, rf: float, out: Optional[str]) -> None:
    _require_file(file)
    df = pd.read_csv(file, index_col=0)

    if "exp_return" not in df.columns:
        raise SystemExit("Error: CSV must include an 'exp_return' column (expected returns index-aligned with assets).")

    mu = df["exp_return"]
    cov = df.drop(columns=["exp_return"])

    try:
        w = np.array(ast.literal_eval(weights), dtype=float)
    except Exception:
        raise SystemExit('Error: --weights must be a Python list, e.g. "[0.3,0.4,0.3]"')

    if w.shape[0] != mu.shape[0]:
        raise SystemExit(f"Error: weights length {w.shape[0]} must match number of assets {mu.shape[0]}")

    stats = portfolio_stats(PortfolioInputs(mu, cov, rf), w)
    _print_kv("Portfolio stats", stats)

    if out:
        # write a tiny CSV with the stats; weights as JSON-ish string
        to_write = {k: (np.array(v).tolist() if isinstance(v, (np.ndarray, list)) else v) for k, v in stats.items()}
        pd.DataFrame([to_write]).to_csv(out, index=False)
        print(f"Saved: {out}")


# ---------- cli ----------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Bond & Portfolio Toolkit")
    sub = p.add_subparsers(dest="cmd", required=True)

    # bond
    b = sub.add_parser("bond", help="Analyze bonds from CSV")
    b.add_argument("--file", required=True, help="CSV with columns: name,face,coupon_rate,price,ytm,maturity_years,freq")
    b.add_argument("--out", default="bond_analytics_output.csv", help="Output CSV filename (default: bond_analytics_output.csv)")

    # frontier
    f = sub.add_parser("frontier", help="Plot efficient frontier from CSV")
    f.add_argument("--file", required=True, help="CSV with exp_return column and covariance matrix to the right")
    f.add_argument("--points", type=int, default=100, help="Number of points along the frontier (default: 100)")
    f.add_argument("--rf", type=float, default=0.0, help="Risk-free rate (default: 0.0)")
    f.add_argument("--out", default="efficient_frontier.png", help="Output PNG filename (default: efficient_frontier.png)")

    # portfolio
    po = sub.add_parser("portfolio", help="Portfolio stats from weights")
    po.add_argument("--file", required=True, help="CSV with exp_return column and covariance matrix to the right")
    po.add_argument("--weights", required=True, help='Python list, e.g. "[0.3,0.4,0.3]"')
    po.add_argument("--rf", type=float, default=0.0, help="Risk-free rate (default: 0.0)")
    po.add_argument("--out", default=None, help="Optional output CSV filename")

    return p


def main(argv: Optional[List[str]] = None) -> None:
    p = build_parser()
    args = p.parse_args(argv)

    if args.cmd == "bond":
        cmd_bond(args.file, args.out)
    elif args.cmd == "frontier":
        cmd_frontier(args.file, args.points, args.rf, args.out)
    elif args.cmd == "portfolio":
        cmd_portfolio(args.file, args.weights, args.rf, args.out)
    else:
        p.print_help()
        raise SystemExit(2)


if __name__ == "__main__":
    main()
