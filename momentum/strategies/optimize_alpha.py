"""
Run the Alpha Composite strategy through the genetic optimizer.

Usage:
    python3.11 momentum/strategies/optimize_alpha.py           # full run
    python3.11 momentum/strategies/optimize_alpha.py --fast    # quick test
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from momentum.strategies.alpha_composite import AlphaCompositeMomentumStrategy, TUNABLE_PARAM_SPACE
from momentum.test.genetic_optimizer import GeneticOptimizer, GAConfig, print_optimization_report, save_result

parser = argparse.ArgumentParser()
parser.add_argument("--fast", action="store_true", help="Quick smoke test (10 pop, 5 gen, 3 stocks)")
args = parser.parse_args()

if args.fast:
    symbols = ["AAPL", "SPY", "JPM"]
    cfg = GAConfig(population_size=10, n_generations=5, fitness_metric="robust", verbose=True)
else:
    symbols = ["AAPL", "MSFT", "SPY", "QQQ", "JPM"]
    cfg = GAConfig(population_size=30, n_generations=25, fitness_metric="robust", n_workers=4, verbose=True)

opt = GeneticOptimizer(
    strategy_factory=AlphaCompositeMomentumStrategy,
    param_space=TUNABLE_PARAM_SPACE,
    symbols=symbols,
    config=cfg,
)

result = opt.run()
print_optimization_report(result)
save_result(result, "momentum/test/runs/alpha_composite_opt.json")
print(f"\n  Best params saved → momentum/test/runs/alpha_composite_opt.json")
