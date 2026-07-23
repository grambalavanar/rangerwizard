"""
Run the Alpha Composite strategy through the genetic optimizer.

Usage
-----
  # Basket mode (multi-stock, robust fitness):
  python3.11 momentum/strategies/optimize_alpha.py
  python3.11 momentum/strategies/optimize_alpha.py --fast

  # Security-specific mode (optimise + walk-forward validate for one stock):
  python3.11 momentum/strategies/optimize_alpha.py --target AAPL
  python3.11 momentum/strategies/optimize_alpha.py --target NVDA --train-years 4 --val-years 2

  # Combined: basket optimise, then validate on a single out-of-sample stock:
  python3.11 momentum/strategies/optimize_alpha.py --validate AAPL
"""

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from momentum.strategies.alpha_composite import AlphaCompositeMomentumStrategy, TUNABLE_PARAM_SPACE
from momentum.test.genetic_optimizer import GeneticOptimizer, GAConfig, print_optimization_report, save_result
from momentum.test.strategy_tester import run_backtest, BacktestConfig, print_result
from momentum.test.run_backtest_example import load_price_data

parser = argparse.ArgumentParser(description="Genetic optimiser for AlphaCompositeMomentumStrategy")
parser.add_argument("--fast",        action="store_true", help="Quick smoke test (10 pop, 5 gen, 3 stocks)")
parser.add_argument("--target",      default=None,        help="Optimise specifically for this one symbol (e.g. AAPL)")
parser.add_argument("--train-years", default=4, type=int, help="Years of training data for --target mode (default 4)")
parser.add_argument("--val-years",   default=1, type=int, help="Years to hold out for walk-forward validation (default 1)")
parser.add_argument("--validate",    default=None,        help="After basket optimisation, validate on this symbol")
args = parser.parse_args()

bt_cfg = BacktestConfig(
    initial_capital      = 100_000,
    commission_per_share = 0.005,
    position_sizing      = "atr",
    atr_risk_pct         = 0.01,
    allow_short          = False,
)

# ── Security-specific mode ────────────────────────────────────────────────────
if args.target:
    total_years = args.train_years + args.val_years
    print(f"\n  Security-specific optimisation for {args.target}")
    print(f"  Train: {args.train_years}y  |  Validate (held-out): {args.val_years}y")

    print(f"\n  Loading {total_years}y of data for {args.target} ...")
    df_all, source = load_price_data(args.target, years=total_years)

    # Chronological split
    split_idx = int(len(df_all) * args.train_years / total_years)
    df_train = df_all.iloc[:split_idx].copy()
    df_val   = df_all.iloc[split_idx:].copy()
    print(f"  Train : {df_train.index[0].date()} → {df_train.index[-1].date()}  ({len(df_train)} bars)")
    print(f"  Val   : {df_val.index[0].date()}   → {df_val.index[-1].date()}  ({len(df_val)} bars)\n")

    cfg = GAConfig(
        population_size     = 10 if args.fast else 30,
        n_generations       = 5  if args.fast else 100,
        fitness_metric      = "sharpe",   # single stock → no consistency penalty
        consistency_penalty = 0.0,
        min_trades          = 3,
        n_workers           = 4,
        verbose             = True,
        years_of_data       = args.train_years,
    )

    opt = GeneticOptimizer(
        strategy_factory = AlphaCompositeMomentumStrategy,
        param_space      = TUNABLE_PARAM_SPACE,
        symbols          = [args.target],
        config           = cfg,
        price_data       = {args.target: df_train},
    )
    result = opt.run()
    print_optimization_report(result)

    # Walk-forward validation on held-out period
    print(f"\n{'═' * 60}")
    print(f"  WALK-FORWARD VALIDATION  ({args.target}, {args.val_years}y out-of-sample)")
    print(f"{'═' * 60}")

    strategy_val   = AlphaCompositeMomentumStrategy(params=result.best_params)
    result_train   = run_backtest(strategy_val, df_train, bt_cfg, symbol=f"{args.target}[train]")
    result_val     = run_backtest(AlphaCompositeMomentumStrategy(params=result.best_params),
                                  df_val, bt_cfg, symbol=f"{args.target}[val]")

    def _fmt(r):
        sh  = f"{r.sharpe:+.2f}" if not math.isnan(r.sharpe) else " n/a"
        cal = f"{r.calmar:+.2f}" if not math.isnan(r.calmar) else " n/a"
        return f"Sharpe={sh}  Calmar={cal}  CAGR={r.cagr_val:.1%}  MaxDD={r.max_dd:.1%}  Trades={r.total_trades}"

    print(f"  In-sample  (train) : {_fmt(result_train)}")
    print(f"  Out-of-sample (val): {_fmt(result_val)}")

    if not math.isnan(result_train.sharpe) and not math.isnan(result_val.sharpe):
        decay = result_val.sharpe / result_train.sharpe if result_train.sharpe != 0 else 0
        print(f"\n  Sharpe decay (val/train): {decay:.0%}  "
              f"({'⚠ possible overfit' if decay < 0.5 else '✓ robust'})")
    print()

    out_path = f"momentum/test/runs/{args.target}_opt.json"
    save_result(result, out_path)
    print(f"  Best params saved → {out_path}")

# ── Basket mode ───────────────────────────────────────────────────────────────
else:
    if args.fast:
        symbols = ["AAPL", "SPY", "JPM"]
        cfg = GAConfig(population_size=10, n_generations=5, fitness_metric="robust", verbose=True)
    else:
        # Diverse basket: mega-cap tech, broad index, financials, energy, consumer, semiconductor
        # Covers multiple regimes and sectors so optimised weights generalise well.
        symbols = ["AAPL", "MSFT", "NVDA", "SPY", "QQQ", "JPM", "XOM", "AMD", "AMZN", "V"]
        cfg = GAConfig(population_size=30, n_generations=100, fitness_metric="robust",
                       consistency_penalty=0.30, n_workers=4, verbose=True)

    opt = GeneticOptimizer(
        strategy_factory = AlphaCompositeMomentumStrategy,
        param_space      = TUNABLE_PARAM_SPACE,
        symbols          = symbols,
        config           = cfg,
    )
    result = opt.run()
    print_optimization_report(result)
    save_result(result, "momentum/test/runs/alpha_composite_opt.json")
    print(f"\n  Best params saved → momentum/test/runs/alpha_composite_opt.json")

    # Optional post-run single-stock validation
    if args.validate:
        print(f"\n{'═' * 60}")
        print(f"  POST-RUN VALIDATION on {args.validate}")
        print(f"{'═' * 60}")
        df_v, src_v = load_price_data(args.validate, years=3)
        r_v = run_backtest(AlphaCompositeMomentumStrategy(params=result.best_params),
                           df_v, bt_cfg, symbol=args.validate)
        print_result(r_v)

