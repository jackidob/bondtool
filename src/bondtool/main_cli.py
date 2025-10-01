import argparse, ast, pandas as pd, numpy as np
from .bonds import Bond
from .portfolio import PortfolioInputs, portfolio_stats
from .efficient_frontier import plot_frontier

def cmd_bond(file: str):
    df = pd.read_csv(file)
    print("Loaded bonds:")
    print(df)
    rows = []
    for _, r in df.iterrows():
        bond = Bond(face=r['face'], coupon_rate=r['coupon_rate'], price=r['price'], ytm=None if pd.isna(r.get('ytm', float('nan'))) else r['ytm'], maturity_years=r['maturity_years'], freq=int(r.get('freq', 2)))
        y = bond.ytm if bond.ytm is not None else bond.ytm_from_price()
        p = bond.price_from_ytm(y)
        dur_mac = bond.macaulay_duration(y)
        dur_mod = bond.modified_duration(y)
        conv = bond.convexity(y)
        rows.append({"name": r.get('name', f'bond_{_}'), "ytm": y, "price": p, "macaulay_dur": dur_mac, "modified_dur": dur_mod, "convexity": conv})
    out = pd.DataFrame(rows)
    print("\nBond analytics:")
    print(out.round(6))
    out.to_csv('bond_analytics_output.csv', index=False)
    print("\nSaved: bond_analytics_output.csv")

def cmd_frontier(file: str, points: int, rf: float):
    df = pd.read_csv(file, index_col=0)
    mu = df['exp_return']
    cov = df.drop(columns=['exp_return'])
    out = plot_frontier(mu, cov, rf=rf, n_points=points, out_path='efficient_frontier.png')
    print(out.head())
    print("\nSaved plot: efficient_frontier.png")

def cmd_portfolio(file: str, weights: str, rf: float):
    df = pd.read_csv(file, index_col=0)
    mu = df['exp_return']
    cov = df.drop(columns=['exp_return'])
    w = ast.literal_eval(weights)
    stats = portfolio_stats(PortfolioInputs(mu, cov, rf), w)
    print("\nPortfolio stats:")
    for k, v in stats.items():
        if k == 'weights':
            print(k, np.round(v, 6))
        else:
            print(k, round(float(v), 6))

def main():
    p = argparse.ArgumentParser(description="Bond & Portfolio Toolkit")
    sub = p.add_subparsers(dest='cmd', required=True)

    b = sub.add_parser('bond', help='Analyze bonds from CSV')
    b.add_argument('--file', required=True)

    f = sub.add_parser('frontier', help='Plot efficient frontier from CSV')
    f.add_argument('--file', required=True)
    f.add_argument('--points', type=int, default=100)
    f.add_argument('--rf', type=float, default=0.0)

    po = sub.add_parser('portfolio', help='Portfolio stats from weights')
    po.add_argument('--file', required=True)
    po.add_argument('--weights', required=True, help='Python list, e.g. "[0.3,0.4,0.3]"')
    po.add_argument('--rf', type=float, default=0.0)

    args = p.parse_args()
    if args.cmd == 'bond':
        cmd_bond(args.file)
    elif args.cmd == 'frontier':
        cmd_frontier(args.file, args.points, args.rf)
    elif args.cmd == 'portfolio':
        cmd_portfolio(args.file, args.weights, args.rf)

if __name__ == '__main__':
    main()
