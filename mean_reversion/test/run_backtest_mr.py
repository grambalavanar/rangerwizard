"""
run_backtest_mr.py
==================
Example backtest runner for AlphaMeanReversionStrategy. Mirrors
momentum/test/run_backtest_example.py in structure and output.

Run from repo root:
    python mean_reversion/test/run_backtest_mr.py
    python mean_reversion/test/run_backtest_mr.py --symbol MSFT --years 5
    python mean_reversion/test/run_backtest_mr.py --source ibkr --symbol SPY
"""

import argparse
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from mean_reversion.strategies.alpha_mean_reversion import AlphaMeanReversionStrategy
from mean_reversion.test import (
    run_backtest, BacktestConfig, load_price_data, print_full_report,
)
from mean_reversion.mean_reversion_tools import avg_holding_period, mean_reversion_speed

parser = argparse.ArgumentParser(description="MR backtest example runner")
parser.add_argument("--symbol",  default="AAPL")
parser.add_argument("--years",   default=3, type=int)
parser.add_argument("--source",  default="auto",
                    choices=["auto", "yfinance", "ibkr", "csv", "synthetic"])
parser.add_argument("--csv-path", default=None)
parser.add_argument("--capital",  default=100_000, type=float)
args = parser.parse_args()

print(f"\n  Mean Reversion Backtest  |  {args.symbol}  |  {args.years}y")
print(f"  Source: {args.source}")

print("\n  Loading data ...")
df, source = load_price_data(
    symbol=args.symbol, years=args.years,
    source=args.source, csv_path=args.csv_path,
)

cfg = BacktestConfig(
    initial_capital      = args.capital,
    commission_per_share = 0.005,
    slippage_pct         = 0.001,
    position_sizing      = "atr",
    atr_risk_pct         = 0.01,
    allow_short          = False,
)

print(f"\n  Running AlphaMeanReversionStrategy on {args.symbol} ...")
strategy = AlphaMeanReversionStrategy()
result   = run_backtest(strategy, df, cfg, symbol=args.symbol)
print_full_report(result, source)

# MR-specific stats
avg_hold = avg_holding_period(result.trades)
mr_speed = mean_reversion_speed(result.daily_returns)
print(f"  Mean Reversion Stats:")
print(f"    Avg holding period : {avg_hold:.1f} days  (< 10 = fast MR, > 20 = slow)")
print(f"    Return autocorr    : {mr_speed:+.3f}      (< 0 confirms MR regime)\n")
