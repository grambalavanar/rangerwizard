"""
strategies — unified trading strategy package

Combines momentum and mean-reversion strategies under one roof, gated by
an academically-backed regime classifier (Gaussian HMM + volatility + trend).

Quick-start
-----------
    from strategies import UnifiedAlphaStrategy, RegimeClassifier
    from strategies.test import run_backtest, BacktestConfig, load_price_data

    df, _ = load_price_data("AAPL", years=3)
    result = run_backtest(UnifiedAlphaStrategy(), df, symbol="AAPL")

Structure
---------
    strategies/
        tools/          momentum_tools, mean_reversion_tools, regime_tools
        test/           backtesting, GA optimizer, paper trader (re-exports)
        optimize/       GA optimizer runners for each strategy
        backtest/       backtest runners for each strategy
"""

import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── Sub-strategy imports ──────────────────────────────────────────────────────
from momentum.strategies.alpha_composite import (
    AlphaCompositeMomentumStrategy,
    DEFAULT_PARAMS as MOMENTUM_DEFAULT_PARAMS,
)
from mean_reversion.strategies.alpha_mean_reversion import (
    AlphaMeanReversionStrategy,
    DEFAULT_PARAMS as MR_DEFAULT_PARAMS,
)

# ── Regime tools ──────────────────────────────────────────────────────────────
from strategies.tools.regime_tools import (
    GaussianHMM, RegimeClassifier,
    volatility_regime, trend_regime, classify_regime,
    REGIME_MOMENTUM, REGIME_MEAN_REVERSION, REGIME_CASH,
)

# ── Unified strategy ─────────────────────────────────────────────────────────
from strategies.unified_alpha import (
    UnifiedAlphaStrategy,
    UNIFIED_DEFAULT_PARAMS,
    UNIFIED_PARAM_SPACE,
)

__all__ = [
    "AlphaCompositeMomentumStrategy",
    "AlphaMeanReversionStrategy",
    "GaussianHMM",
    "RegimeClassifier",
    "volatility_regime",
    "trend_regime",
    "classify_regime",
    "REGIME_MOMENTUM",
    "REGIME_MEAN_REVERSION",
    "REGIME_CASH",
    "UnifiedAlphaStrategy",
    "UNIFIED_DEFAULT_PARAMS",
    "UNIFIED_PARAM_SPACE",
    "MOMENTUM_DEFAULT_PARAMS",
    "MR_DEFAULT_PARAMS",
]
