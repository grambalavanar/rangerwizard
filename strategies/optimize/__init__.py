"""
strategies/optimize — GA optimizer runners for all strategies.

Usage
-----
  # Unified strategy (recommended — optimises regime thresholds + both sub-strategies):
  python strategies/optimize/optimize_unified.py
  python strategies/optimize/optimize_unified.py --fast
  python strategies/optimize/optimize_unified.py --target AAPL --train-years 4 --val-years 2

  # Momentum only:
  python momentum/strategies/optimize_alpha.py

  # Mean reversion only:
  python mean_reversion/strategies/optimize_alpha_mr.py
"""
