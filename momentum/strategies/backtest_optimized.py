"""
Run a historical backtest of AlphaCompositeMomentumStrategy
using the parameters saved by the genetic optimizer.

Single symbol:
    python3.11 momentum/strategies/backtest_optimized.py --symbol AAPL --years 3

Multiple symbols (comparison table):
    python3.11 momentum/strategies/backtest_optimized.py --symbols AAPL MSFT SPY --years 3

Pre-defined universes:
    python3.11 momentum/strategies/backtest_optimized.py --universe megacap
    python3.11 momentum/strategies/backtest_optimized.py --universe movers
    python3.11 momentum/strategies/backtest_optimized.py --universe volume
    python3.11 momentum/strategies/backtest_optimized.py --universe all
"""

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from momentum.strategies.alpha_composite import AlphaCompositeMomentumStrategy
from momentum.test.strategy_tester import run_backtest, BacktestConfig, compare_results
from momentum.test.run_backtest_example import load_price_data, print_full_report

PARAMS_FILE = "momentum/test/runs/alpha_composite_opt.json"

# ── Pre-defined universes ────────────────────────────────────────────────────
UNIVERSES = {
    # Top 10 by market cap (July 2026)
    "megacap": ["AAPL", "NVDA", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "LLY", "BRK-B"],
    # Top 10 biggest movers / high-beta momentum stocks
    "movers":  ["TSLA", "NVDA", "AMD", "PLTR", "MSTR", "SMCI", "COIN", "MRNA", "SHOP", "CELH"],
    # Top 10 most actively traded by volume
    "volume":  ["SPY", "QQQ", "AAPL", "TSLA", "NVDA", "AMZN", "AMD", "SOXL", "BAC", "F"],
}
UNIVERSES["all"] = sorted(set(s for v in UNIVERSES.values() for s in v))

parser = argparse.ArgumentParser()
parser.add_argument("--symbol",      default=None,  help="Single ticker")
parser.add_argument("--symbols",     nargs="+",     help="Multiple tickers for comparison")
parser.add_argument("--universe",    default=None,
                    choices=["megacap", "movers", "volume", "all"],
                    help="Pre-defined universe")
parser.add_argument("--years",       default=3, type=int)
parser.add_argument("--params-file", default=PARAMS_FILE,
                    help="Path to optimizer JSON (default: alpha_composite_opt.json)")
args = parser.parse_args()

# ── Resolve symbol list ───────────────────────────────────────────────────────
if args.universe:
    symbols = UNIVERSES[args.universe]
elif args.symbols:
    symbols = args.symbols
elif args.symbol:
    symbols = [args.symbol]
else:
    symbols = ["AAPL"]

# ── Load optimised parameters ────────────────────────────────────────────────
with open(args.params_file) as f:
    data = json.load(f)

best_params = data["best_params"]
fitness     = data["best_fitness"]
print(f"\n  Loaded  : {PARAMS_FILE}")
print(f"  Fitness : {fitness:.4f}")
print(f"  Symbols : {symbols}")
print(f"  Years   : {args.years}\n")

cfg = BacktestConfig(
    initial_capital      = 100_000,
    commission_per_share = 0.005,
    position_sizing      = "atr",
    atr_risk_pct         = 0.01,
    allow_short          = False,
)

# ── Single symbol — full ASCII report ────────────────────────────────────────
if len(symbols) == 1:
    sym = symbols[0]
    df, source = load_price_data(sym, years=args.years)
    strategy   = AlphaCompositeMomentumStrategy(params=best_params)
    result     = run_backtest(strategy, df, cfg, symbol=sym)
    print_full_report(result, source)
    df_out = strategy.generate_signals(df.copy())
    print(f"  Current composite : {df_out['_composite'].iloc[-1]:.3f}")
    print(f"  Current signal    : {'LONG' if df_out['signal'].iloc[-1] == 1 else 'FLAT'}\n")

# ── Multiple symbols — comparison table ──────────────────────────────────────
else:
    results = {}
    for sym in symbols:
        print(f"  Backtesting {sym:<8}", end=" ", flush=True)
        try:
            df, _ = load_price_data(sym, years=args.years)
            strategy = AlphaCompositeMomentumStrategy(params=best_params)
            result   = run_backtest(strategy, df, cfg, symbol=sym)
            results[sym] = result
            print(f"Sharpe={result.sharpe:+.2f}  CAGR={result.cagr_val:.1%}  "
                  f"MaxDD={result.max_dd:.1%}  Trades={result.total_trades}")
        except Exception as exc:
            print(f"FAILED: {exc}")

    if results:
        table = compare_results(results)
        W = 74
        print(f"\n{'═' * W}")
        print(f"  COMPARISON — AlphaComposite (optimised)  |  {args.years}y  |  {len(results)} symbols")
        print(f"{'═' * W}")
        print(table.to_string(index=False))
        print(f"{'═' * W}\n")

        # Current signal for each symbol
        print(f"  {'Symbol':<8}  {'Composite':>10}  {'Signal':<6}")
        print(f"  {'─'*8}  {'─'*10}  {'─'*6}")
        for sym, result in sorted(results.items()):
            try:
                df, _ = load_price_data(sym, years=args.years)
                strategy = AlphaCompositeMomentumStrategy(params=best_params)
                df_out   = strategy.generate_signals(df.copy())
                comp     = df_out["_composite"].iloc[-1]
                sig      = "LONG" if df_out["signal"].iloc[-1] == 1 else "flat"
                print(f"  {sym:<8}  {comp:>10.3f}  {sig:<6}")
            except Exception:
                pass
        print()
