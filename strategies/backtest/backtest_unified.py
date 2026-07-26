"""
backtest_unified.py
===================
Historical backtest runner for UnifiedAlphaStrategy using optimised
parameters. Prints full ASCII report + regime breakdown chart.

Usage
-----
  python strategies/backtest/backtest_unified.py
  python strategies/backtest/backtest_unified.py --symbol NVDA --years 5
  python strategies/backtest/backtest_unified.py --universe all --years 3
  python strategies/backtest/backtest_unified.py --params-file momentum/test/runs/unified_opt.json
"""

import argparse
import json
import math
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from strategies.unified_alpha import UnifiedAlphaStrategy
from strategies.test import (
    run_backtest, BacktestConfig, compare_results,
    load_price_data, print_full_report,
)

DEFAULT_PARAMS_FILE = "momentum/test/runs/unified_opt.json"

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
parser.add_argument("--default-params", action="store_true",
                    help="Use default (un-optimised) parameters instead of loading JSON")
args = parser.parse_args()

# ── Load params ────────────────────────────────────────────────────────────────
if args.default_params or not os.path.exists(args.params_file):
    best_params = {}
    fitness = float("nan")
    print(f"\n  Using default UnifiedAlphaStrategy parameters.")
else:
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

# ── Resolve symbols ─────────────────────────────────────────────────────────────
if args.universe:
    symbols = UNIVERSES[args.universe]
elif args.symbols:
    symbols = args.symbols
elif args.symbol:
    symbols = [args.symbol]
else:
    symbols = ["AAPL"]

print(f"  Symbols : {symbols}  |  {args.years}y\n")

# ── Single symbol — full report ────────────────────────────────────────────────
if len(symbols) == 1:
    sym = symbols[0]
    df, source = load_price_data(sym, years=args.years)
    strategy   = UnifiedAlphaStrategy(params=best_params)
    result     = run_backtest(strategy, df, cfg, symbol=sym)
    print_full_report(result, source)

    df_out = strategy.generate_signals(df.copy())
    print("  Regime breakdown:")
    vc    = {}
    for r in df_out["_regime"].values:
        vc[r] = vc.get(r, 0) + 1
    total = len(df_out)
    for regime in ("MOMENTUM", "MEAN_REVERSION", "CASH"):
        count = vc.get(regime, 0)
        bar   = "█" * int(count / total * 35)
        print(f"    {regime:<20} {bar:<35}  {count:4d} bars  ({count/total:.0%})")

    print(f"\n  Current regime  : {df_out['_regime'].iloc[-1]}")
    print(f"  Current signal  : {'LONG' if df_out['signal'].iloc[-1] == 1 else 'FLAT'}\n")

# ── Multiple symbols — comparison table ────────────────────────────────────────
else:
    results = {}
    for sym in symbols:
        print(f"  Backtesting {sym:<8}", end=" ", flush=True)
        try:
            df, _ = load_price_data(sym, years=args.years)
            r     = run_backtest(UnifiedAlphaStrategy(best_params), df, cfg, symbol=sym)
            results[sym] = r
            print(f"Sharpe={r.sharpe:+.2f}  CAGR={r.cagr_val:.1%}  "
                  f"MaxDD={r.max_dd:.1%}  Trades={r.total_trades}")
        except Exception as exc:
            print(f"FAILED: {exc}")

    if results:
        table = compare_results(results)
        W = 74
        print(f"\n{'═' * W}")
        print(f"  UNIFIED STRATEGY COMPARISON  |  {args.years}y  |  {len(results)} symbols")
        print(f"{'═' * W}")
        print(table.to_string(index=False))
        print(f"{'═' * W}\n")

        print(f"  {'Symbol':<8}  {'Regime':<20}  {'Composite':>10}  {'Signal'}")
        print(f"  {'─'*8}  {'─'*20}  {'─'*10}  {'─'*6}")
        for sym in sorted(results):
            try:
                df, _  = load_price_data(sym, years=args.years)
                strat  = UnifiedAlphaStrategy(best_params)
                df_out = strat.generate_signals(df.copy())
                regime = df_out["_regime"].iloc[-1]
                if regime == "MOMENTUM":
                    comp = df_out["_mom_composite"].iloc[-1]
                else:
                    comp = df_out["_mr_composite"].iloc[-1]
                sig = "LONG" if df_out["signal"].iloc[-1] == 1 else "flat"
                print(f"  {sym:<8}  {regime:<20}  {comp:>10.3f}  {sig}")
            except Exception:
                pass
        print()
