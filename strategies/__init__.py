"""
strategies -- unified trading strategy package

All strategy logic lives in strategies/alpha_composite.py.
Import from there directly:

    from strategies.alpha_composite import (
        UnifiedAlphaStrategy, AlphaCompositeMomentumStrategy,
        AlphaMeanReversionStrategy, RegimeClassifier,
        REGIME_MOMENTUM, REGIME_MEAN_REVERSION, REGIME_CASH,
        UNIFIED_PARAM_SPACE,
    )

Lazy __getattr__ allows  without circular imports.
"""
import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_EXPORTED = {
    "AlphaCompositeMomentumStrategy", "DEFAULT_PARAMS", "TUNABLE_PARAM_SPACE",
    "AlphaMeanReversionStrategy", "MR_DEFAULT_PARAMS", "MR_TUNABLE_PARAM_SPACE",
    "GaussianHMM", "RegimeClassifier",
    "volatility_regime", "trend_regime", "classify_regime",
    "REGIME_MOMENTUM", "REGIME_MEAN_REVERSION", "REGIME_CASH",
    "UnifiedAlphaStrategy", "UNIFIED_DEFAULT_PARAMS", "UNIFIED_PARAM_SPACE",
}

def __getattr__(name):
    if name in _EXPORTED:
        from strategies import alpha_composite as _ac
        return getattr(_ac, name)
    raise AttributeError(f"module 'strategies' has no attribute '{name}'")
