"""
backtest_optimized_mr.py
========================
Historical backtest runner for AlphaMeanReversionStrategy using
optimised parameters from the genetic optimizer.
Mirrors momentum/strategies/backtest_optimized.py.

Usage
-----
  python mean_reversion/strategies/backtest_optimized_mr.py
  python mean_reversion/strategies/backtest_optimized_mr.py --symbol MSFT --years 5
  python mean_reversion/strategies/backtest_optimized_mr.py --universe movers
  python mean_reversion/strategies/backtest_optimized_mr.py --params-file momentum/test/runs/AAPL_mr_opt.json
"""

import argparse
import json
import math
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from mean_reversion.strategies.alpha_mean_reversion import AlphaMeanReversionStrategy
from mean_reversion.test import (
    run_backtest, BacktestConfig, compare_results, load_price_data, print_full_report,
)
from mean_reversion.mean_reversion_tools import avg_holding_period, mean_reversion_speed

DEFAULT_PARAMS_FILE = "momentum/test/runs/alpha_mr_opt.json"

UNIVERSES = {
    "megacap": ["AAPL", "NVDA", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "LLY", "BRK-B"],
    "movers":  ["TSLA", "NVDA", "AMD", "PLTR", "MSTR", "SMCI", "COIN", "MRNA", "SHOP", "CELH"],
    "volume":  ["SPY", "QQQ", "AAPL", "TSLA", "NVDA", "AMZN", "AMD", "SOXL", "BAC", "F"],
}
UNIVERSES["all"] = sorted(set(s for v in UNIVERSES.values() for s in v))

parser = argparse.ArgumentParser()
parser.add_argument("--symbol",      default=None)
parser.add_argument("--symbols",     nargs="+")
parser.add_argument("--universe",    default=None, choices=["megacap", "movers", "volume", "all"])
parser.add_argument("--years",       default=3, type=int)
parser.add_argument("--params-file", default=DEFAULT_PARAMS_FILE)
args = parser.parse_args()

# Load params
with open(args.params_file) as f:
    data = json.load(f)
best_params = data["best_params"]
fitness     = data.get("best_fitness", float("nan"))
print(f"\n  Loaded  : {args.params_file}")
print(f"  Fitness : {fitness:.4f}")

cfg = BacktestConfig(
    initial_capital      = 100_000,
    commission_per_share = 0.005,
    position_sizing      = "atr",
    atr_risk_pct         = 0.01,
    allow_short          = False,
)

# Resolve symbols
if args.universe:
    symbols = UNIVERSES[args.universe]
elif args.symbols:
    symbols = args.symbols
elif args.symbol:
    symbols = [args.symbol]
else:
    symbols = ["AAPL"]

print(f"  Symbols : {symbols}  |  {args.years}y\n")

# Single symbol → full ASCII report
if len(symbols) == 1:
    sym = symbols[0]
    df, source = load_price_data(sym, years=args.years)
    strategy   = AlphaMeanReversionStrategy(params=best_params)
    result     = run_backtest(strategy, df, cfg, symbol=sym)
    print_full_report(result, source)

    # MR-specific stats
    avg_hold = avg_holding_period(result.trades)
    mr_speed = mean_reversion_speed(result.daily_returns)
    print(f"  MR Stats:")
    print(f"    Avg holding period  : {avg_hold:.1f} days")
    print(f"    Return autocorr lag1: {mr_speed:+.3f}  (< 0 = MR regime)\n")

    df_out = strategy.generate_signals(df.copy())
    print(f"  Current composite   : {df_out['_composite'].iloc[-1]:.3f}")
    print(f"  Current signal      : {'LONG (oversold)' if df_out['signal'].iloc[-1] == 1 else 'FLAT'}\n")

# Multiple symbols → comparison table
else:
    results = {}
    for sym in symbols:
        print(f"  Backtesting {sym:<8}", end=" ", flush=True)
        try:
            df, _ = load_price_data(sym, years=args.years)
            r     = run_backtest(AlphaMeanReversionStrategy(best_params), df, cfg, symbol=sym)
            results[sym] = r
            print(f"Sharpe={r.sharpe:+.2f}  CAGR={r.cagr_val:.1%}  "
                  f"MaxDD={r.max_dd:.1%}  Trades={r.total_trades}")
        except Exception as exc:
            print(f"FAILED: {exc}")

    if results:
        table = compare_results(results)
        W = 74
        print(f"\n{'═' * W}")
        print(f"  COMPARISON — AlphaMeanReversion (optimised)  |  {args.years}y")
        print(f"{'═' * W}")
        print(table.to_string(index=False))
        print(f"{'═' * W}\n")

        # Current signal table
        print(f"  {'Symbol':<8}  {'Composite':>10}  {'Signal':<20}")
        print(f"  {'─'*8}  {'─'*10}  {'─'*20}")
        for sym in sorted(results):
            try:
                df, _ = load_price_data(sym, years=args.years)
                df_out = AlphaMeanReversionStrategy(best_params).generate_signals(df.copy())
                comp = df_out["_composite"].iloc[-1]
                sig  = "LONG (oversold)" if df_out["signal"].iloc[-1] == 1 else "flat"
                print(f"  {sym:<8}  {comp:>10.3f}  {sig}")
            except Exception:
                pass
        print()
