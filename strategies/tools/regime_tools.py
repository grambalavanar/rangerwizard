"""
strategies/tools/regime_tools.py — thin re-export from alpha_composite.py
"""
import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from strategies.alpha_composite import (
    REGIME_MOMENTUM, REGIME_MEAN_REVERSION, REGIME_CASH,
    GaussianHMM, RegimeClassifier,
    volatility_regime, trend_regime, classify_regime,
)
__all__ = [
    "REGIME_MOMENTUM", "REGIME_MEAN_REVERSION", "REGIME_CASH",
    "GaussianHMM", "RegimeClassifier",
    "volatility_regime", "trend_regime", "classify_regime",
]
