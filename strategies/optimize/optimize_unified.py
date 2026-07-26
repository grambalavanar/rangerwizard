"""
optimize_unified.py
===================
Genetic optimizer runner for UnifiedAlphaStrategy.
Optimises regime classifier thresholds + both sub-strategy parameters
simultaneously across a diverse multi-stock basket.

Usage
-----
  python strategies/optimize/optimize_unified.py             # full run
  python strategies/optimize/optimize_unified.py --fast      # smoke test
  python strategies/optimize/optimize_unified.py --target AAPL --train-years 4 --val-years 2
  python strategies/optimize/optimize_unified.py --validate SPY
"""

import argparse
import math
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from strategies.unified_alpha import UnifiedAlphaStrategy, UNIFIED_PARAM_SPACE
from strategies.test import (
    GeneticOptimizer, GAConfig, print_optimization_report, save_result,
    run_backtest, BacktestConfig, print_result, load_price_data,
)

parser = argparse.ArgumentParser(description="GA optimiser for UnifiedAlphaStrategy")
parser.add_argument("--fast",        action="store_true")
parser.add_argument("--target",      default=None,  help="Single symbol (security-specific mode)")
parser.add_argument("--train-years", default=4, type=int)
parser.add_argument("--val-years",   default=1, type=int)
parser.add_argument("--validate",    default=None,  help="Post-run out-of-sample validation symbol")
args = parser.parse_args()

bt_cfg = BacktestConfig(
    initial_capital      = 100_000,
    commission_per_share = 0.005,
    position_sizing      = "atr",
    atr_risk_pct         = 0.01,
    allow_short          = False,
)

# ── Security-specific mode ─────────────────────────────────────────────────────
if args.target:
    total_years = args.train_years + args.val_years
    print(f"\n  Security-specific Unified optimisation for {args.target}")
    print(f"  Train: {args.train_years}y  |  Validate (held-out): {args.val_years}y\n")

    df_all, _ = load_price_data(args.target, years=total_years)
    split_idx  = int(len(df_all) * args.train_years / total_years)
    df_train   = df_all.iloc[:split_idx].copy()
    df_val     = df_all.iloc[split_idx:].copy()
    print(f"  Train : {df_train.index[0].date()} → {df_train.index[-1].date()}  ({len(df_train)} bars)")
    print(f"  Val   : {df_val.index[0].date()} → {df_val.index[-1].date()}  ({len(df_val)} bars)\n")

    cfg = GAConfig(
        population_size     = 10  if args.fast else 30,
        n_generations       = 5   if args.fast else 100,
        fitness_metric      = "sharpe",
        consistency_penalty = 0.0,
        min_trades          = 3,
        n_workers           = 4,
        verbose             = True,
        years_of_data       = args.train_years,
    )

    opt = GeneticOptimizer(
        strategy_factory = UnifiedAlphaStrategy,
        param_space      = UNIFIED_PARAM_SPACE,
        symbols          = [args.target],
        config           = cfg,
        price_data       = {args.target: df_train},
    )
    result = opt.run()
    print_optimization_report(result)

    # Walk-forward validation
    print(f"\n{'═' * 60}")
    print(f"  WALK-FORWARD VALIDATION  ({args.target})")
    print(f"{'═' * 60}")
    r_train = run_backtest(UnifiedAlphaStrategy(result.best_params), df_train, bt_cfg,
                           symbol=f"{args.target}[train]")
    r_val   = run_backtest(UnifiedAlphaStrategy(result.best_params), df_val,   bt_cfg,
                           symbol=f"{args.target}[val]")

    def _fmt(r):
        sh  = f"{r.sharpe:+.2f}" if not math.isnan(r.sharpe) else " n/a"
        cal = f"{r.calmar:+.2f}" if not math.isnan(r.calmar) else " n/a"
        return f"Sharpe={sh}  Calmar={cal}  CAGR={r.cagr_val:.1%}  MaxDD={r.max_dd:.1%}  Trades={r.total_trades}"

    print(f"  In-sample  (train) : {_fmt(r_train)}")
    print(f"  Out-of-sample (val): {_fmt(r_val)}")
    if not math.isnan(r_train.sharpe) and not math.isnan(r_val.sharpe) and r_train.sharpe != 0:
        decay = r_val.sharpe / r_train.sharpe
        print(f"\n  Sharpe decay : {decay:.0%}  "
              f"({'⚠ possible overfit' if decay < 0.5 else '✓ robust'})")

    out = f"momentum/test/runs/{args.target}_unified_opt.json"
    save_result(result, out)
    print(f"\n  Saved → {out}")

# ── Basket mode ────────────────────────────────────────────────────────────────
else:
    # Diverse 10-stock basket covering multiple sectors and volatility regimes.
    # Spanning multiple regime types ensures the classifier is trained on all
    # conditions and doesn't overfit to a single market environment.
    if args.fast:
        symbols = ["AAPL", "SPY", "AMD"]
        cfg = GAConfig(
            population_size = 10, n_generations = 5,
            fitness_metric  = "robust", verbose = True,
        )
    else:
        symbols = ["AAPL", "MSFT", "NVDA", "SPY", "QQQ",
                   "JPM", "XOM", "AMD", "AMZN", "V"]
        cfg = GAConfig(
            population_size     = 30,
            n_generations       = 100,
            fitness_metric      = "robust",
            consistency_penalty = 0.30,
            n_workers           = 4,
            verbose             = True,
        )

    opt = GeneticOptimizer(
        strategy_factory = UnifiedAlphaStrategy,
        param_space      = UNIFIED_PARAM_SPACE,
        symbols          = symbols,
        config           = cfg,
    )
    result = opt.run()
    print_optimization_report(result)

    out = "momentum/test/runs/unified_opt.json"
    save_result(result, out)
    print(f"\n  Saved → {out}")

    if args.validate:
        print(f"\n{'═' * 60}")
        print(f"  POST-RUN VALIDATION on {args.validate}")
        df_v, _ = load_price_data(args.validate, years=3)
        r_v = run_backtest(UnifiedAlphaStrategy(result.best_params),
                           df_v, bt_cfg, symbol=args.validate)
        print_result(r_v)
