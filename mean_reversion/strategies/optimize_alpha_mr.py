"""
optimize_alpha_mr.py
====================
Genetic optimizer runner for AlphaMeanReversionStrategy.
Mirrors momentum/strategies/optimize_alpha.py in structure.

Usage
-----
  # Basket (multi-stock, robust fitness):
  python mean_reversion/strategies/optimize_alpha_mr.py
  python mean_reversion/strategies/optimize_alpha_mr.py --fast

  # Security-specific with walk-forward validation:
  python mean_reversion/strategies/optimize_alpha_mr.py --target AAPL
  python mean_reversion/strategies/optimize_alpha_mr.py --target MSFT --train-years 4 --val-years 2
"""

import argparse
import math
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from mean_reversion.strategies.alpha_mean_reversion import (
    AlphaMeanReversionStrategy, TUNABLE_PARAM_SPACE,
)
from mean_reversion.test import (
    GeneticOptimizer, GAConfig, print_optimization_report, save_result,
    run_backtest, BacktestConfig, print_result, load_price_data,
)

parser = argparse.ArgumentParser(description="GA optimiser for AlphaMeanReversionStrategy")
parser.add_argument("--fast",        action="store_true")
parser.add_argument("--target",      default=None,  help="Single symbol (security-specific mode)")
parser.add_argument("--train-years", default=4, type=int)
parser.add_argument("--val-years",   default=1, type=int)
parser.add_argument("--validate",    default=None,  help="Post-run validation symbol")
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
    print(f"\n  Security-specific MR optimisation for {args.target}")
    print(f"  Train: {args.train_years}y  |  Validate: {args.val_years}y\n")

    df_all, _ = load_price_data(args.target, years=total_years)
    split_idx  = int(len(df_all) * args.train_years / total_years)
    df_train   = df_all.iloc[:split_idx].copy()
    df_val     = df_all.iloc[split_idx:].copy()

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
        strategy_factory = AlphaMeanReversionStrategy,
        param_space      = TUNABLE_PARAM_SPACE,
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
    r_train = run_backtest(AlphaMeanReversionStrategy(result.best_params), df_train, bt_cfg,
                           symbol=f"{args.target}[train]")
    r_val   = run_backtest(AlphaMeanReversionStrategy(result.best_params), df_val,   bt_cfg,
                           symbol=f"{args.target}[val]")

    def _fmt(r):
        sh  = f"{r.sharpe:+.2f}" if not math.isnan(r.sharpe) else " n/a"
        return f"Sharpe={sh}  CAGR={r.cagr_val:.1%}  MaxDD={r.max_dd:.1%}  Trades={r.total_trades}"

    print(f"  In-sample  (train) : {_fmt(r_train)}")
    print(f"  Out-of-sample (val): {_fmt(r_val)}")
    if not math.isnan(r_train.sharpe) and not math.isnan(r_val.sharpe) and r_train.sharpe != 0:
        decay = r_val.sharpe / r_train.sharpe
        print(f"\n  Sharpe decay (val/train): {decay:.0%}  "
              f"({'⚠ possible overfit' if decay < 0.5 else '✓ robust'})")

    out = f"momentum/test/runs/{args.target}_mr_opt.json"
    save_result(result, out)
    print(f"\n  Saved → {out}")

# ── Basket mode ────────────────────────────────────────────────────────────────
else:
    # Mean-reverting stocks: prefer high-beta, mean-reverting names
    # Include both MR-friendly names and broad indices for robustness
    if args.fast:
        symbols = ["AAPL", "SPY", "AMD"]
        cfg = GAConfig(population_size=10, n_generations=5, fitness_metric="robust", verbose=True)
    else:
        # Diverse MR basket: includes volatile (AMD, TSLA) and stable (SPY, V)
        # to find parameters robust across different MR dynamics
        symbols = ["AAPL", "MSFT", "AMD", "SPY", "QQQ", "V", "TSLA", "JPM", "XOM", "AMZN"]
        cfg = GAConfig(
            population_size     = 30,
            n_generations       = 100,
            fitness_metric      = "robust",
            consistency_penalty = 0.30,
            n_workers           = 4,
            verbose             = True,
        )

    opt = GeneticOptimizer(
        strategy_factory = AlphaMeanReversionStrategy,
        param_space      = TUNABLE_PARAM_SPACE,
        symbols          = symbols,
        config           = cfg,
    )
    result = opt.run()
    print_optimization_report(result)

    out = "momentum/test/runs/alpha_mr_opt.json"
    save_result(result, out)
    print(f"\n  Saved → {out}")

    if args.validate:
        print(f"\n{'═' * 60}")
        print(f"  POST-RUN VALIDATION on {args.validate}")
        df_v, _ = load_price_data(args.validate, years=3)
        r_v = run_backtest(AlphaMeanReversionStrategy(result.best_params), df_v,
                           bt_cfg, symbol=args.validate)
        print_result(r_v)
