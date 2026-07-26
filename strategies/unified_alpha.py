"""
strategies/unified_alpha.py — thin re-export from alpha_composite.py
"""
from strategies.alpha_composite import (
    UnifiedAlphaStrategy,
    UNIFIED_DEFAULT_PARAMS,
    UNIFIED_PARAM_SPACE,
)
__all__ = ["UnifiedAlphaStrategy", "UNIFIED_DEFAULT_PARAMS", "UNIFIED_PARAM_SPACE"]
