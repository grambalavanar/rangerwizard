"""
unified_alpha.py
================
UnifiedAlphaStrategy — the top-level strategy that automatically
deploys AlphaCompositeMomentumStrategy or AlphaMeanReversionStrategy
based on the current market regime as classified by the RegimeClassifier.

Architecture
------------
    Bar t
      │
      ▼
  RegimeClassifier.classify(df)
      │
      ├─ REGIME_MOMENTUM       → AlphaCompositeMomentumStrategy.generate_signals()
      ├─ REGIME_MEAN_REVERSION → AlphaMeanReversionStrategy.generate_signals()
      └─ REGIME_CASH           → signal = 0 (hold cash)
      │
      ▼
    Unified signal column (1=long, 0=flat)
    + _regime column for inspection

Genetic optimisation
--------------------
    UNIFIED_PARAM_SPACE tunes regime thresholds AND selected parameters
    from both sub-strategies simultaneously, finding the global optimum
    allocation of indicators across regimes.
"""

import os
import sys
from typing import Optional

import numpy as np
import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from momentum.test.strategy_tester import Strategy
from momentum.strategies.alpha_composite import (
    AlphaCompositeMomentumStrategy,
    DEFAULT_PARAMS as _MOM_DEFAULTS,
)
from mean_reversion.strategies.alpha_mean_reversion import (
    AlphaMeanReversionStrategy,
    DEFAULT_PARAMS as _MR_DEFAULTS,
)
from strategies.tools.regime_tools import (
    RegimeClassifier,
    REGIME_MOMENTUM, REGIME_MEAN_REVERSION, REGIME_CASH,
)


# ============================================================
# DEFAULT PARAMETERS
# ============================================================

UNIFIED_DEFAULT_PARAMS: dict = {
    # ── Regime Classifier ─────────────────────────────────────────────────
    "hmm_window":        252,   # HMM training window (Hamilton 1989: ~1 year)
    "hmm_threshold":     0.55,  # Bull-state P above this = momentum regime
    "vol_spike_mult":    1.5,   # vol_ratio > this → CASH (Barroso & SC 2015)
    "adx_trend_min":    25.0,   # ADX above this = trending (Wilder 1978)
    "adx_flat_max":     20.0,   # ADX below this = sideways
    "er_trend_min":      0.55,  # ER above this = trending (Kaufman 1995)
    "er_flat_max":       0.30,  # ER below this = sideways
    "hurst_window":      60,    # Hurst exponent window (Lo 1991)
    "min_regime_bars":    5,    # Minimum bars before regime change accepted

    # ── Key Momentum Params (override sub-strategy defaults) ──────────────
    "mom_entry_threshold": 0.60,
    "mom_exit_threshold":  0.35,
    "mom_atr_stop_mult":   2.5,

    # ── Key Mean Reversion Params ─────────────────────────────────────────
    "mr_entry_threshold":  0.60,
    "mr_exit_threshold":   0.35,
    "mr_atr_stop_mult":    1.5,
    "mr_time_stop_bars":   10,
    "mr_zscore_entry":    -1.5,
}


# ============================================================
# STRATEGY CLASS
# ============================================================

class UnifiedAlphaStrategy(Strategy):
    """
    Regime-switching composite strategy: momentum + mean reversion + cash.
    -----------------------------------------------------------------------
    What it is
        The top-level strategy that automatically deploys the right
        sub-strategy for the current market regime. Uses a Gaussian HMM
        (Hamilton 1989) combined with volatility and trend filters to
        classify each bar as MOMENTUM, MEAN_REVERSION, or CASH, then
        routes signals accordingly.

    Used by
        Use this as the single strategy class in all backtests, paper
        trading, and live trading. It is self-contained — you do not need
        to manually switch between momentum and MR strategies.

    When to use this vs individual strategies
        - UnifiedAlphaStrategy: live trading, full backtest spanning multiple
          market regimes (bull, bear, sideways). The regime classifier adapts
          to changing market conditions automatically.
        - AlphaCompositeMomentumStrategy alone: when you specifically want to
          test performance in trending regimes only.
        - AlphaMeanReversionStrategy alone: when you specifically want to test
          performance in choppy/ranging regimes only.

    Code example (backtest)
        >>> from strategies import UnifiedAlphaStrategy
        >>> from strategies.test import run_backtest, BacktestConfig, load_price_data
        >>> df, src = load_price_data("AAPL", years=5)
        >>> result  = run_backtest(UnifiedAlphaStrategy(), df, symbol="AAPL")
        >>> print_full_report(result, src)

    Code example (inspect regimes)
        >>> df_out = strategy.generate_signals(df.copy())
        >>> print(df_out["_regime"].value_counts())
        >>> import matplotlib.pyplot as plt
        >>> df_out["_regime"].value_counts().plot(kind="bar")

    Code example (genetic optimisation)
        >>> from strategies import UnifiedAlphaStrategy, UNIFIED_PARAM_SPACE
        >>> from strategies.test import GeneticOptimizer, GAConfig
        >>> opt = GeneticOptimizer(
        ...     strategy_factory = UnifiedAlphaStrategy,
        ...     param_space      = UNIFIED_PARAM_SPACE,
        ...     symbols          = ["AAPL", "MSFT", "SPY", "QQQ", "JPM",
        ...                         "AMD", "AMZN", "XOM", "V", "TSLA"],
        ...     config           = GAConfig(population_size=30, n_generations=100),
        ... )
        >>> result = opt.run()

    Overfitting note
        The regime thresholds and sub-strategy parameters together form a
        ~20-parameter search space. Always validate on a held-out period.
        The `regime_score` per stock (fraction of time in each regime)
        should be inspected — if CASH > 30%, the vol_spike_mult may be
        too aggressive for that stock's volatility characteristics.
    """

    name = "Unified Alpha (Momentum + Mean Reversion + Regime)"

    def __init__(self, params: Optional[dict] = None) -> None:
        p = {**UNIFIED_DEFAULT_PARAMS, **(params or {})}
        self._params = p

        # Build sub-strategy params by overriding the key unified params
        mom_override = {
            "entry_threshold": p["mom_entry_threshold"],
            "exit_threshold":  p["mom_exit_threshold"],
            "atr_stop_mult":   p["mom_atr_stop_mult"],
        }
        mr_override = {
            "entry_threshold": p["mr_entry_threshold"],
            "exit_threshold":  p["mr_exit_threshold"],
            "atr_stop_mult":   p["mr_atr_stop_mult"],
            "time_stop_bars":  p["mr_time_stop_bars"],
            "zscore_entry":    p["mr_zscore_entry"],
        }

        self._momentum_strategy = AlphaCompositeMomentumStrategy(
            {**_MOM_DEFAULTS, **mom_override}
        )
        self._mr_strategy = AlphaMeanReversionStrategy(
            {**_MR_DEFAULTS, **mr_override}
        )
        self._regime_clf = RegimeClassifier(
            hmm_window      = p["hmm_window"],
            hmm_threshold   = p["hmm_threshold"],
            vol_spike_mult  = p["vol_spike_mult"],
            adx_trend_min   = p["adx_trend_min"],
            adx_flat_max    = p["adx_flat_max"],
            er_trend_min    = p["er_trend_min"],
            er_flat_max     = p["er_flat_max"],
            hurst_window    = p["hurst_window"],
            min_regime_bars = p["min_regime_bars"],
        )

    @property
    def params(self) -> dict:
        return dict(self._params)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals by routing to the appropriate sub-strategy per bar.
        ----------------------------------------------------------------------
        What it does
            1. Classifies each bar with RegimeClassifier → pd.Series of labels.
            2. Runs BOTH sub-strategies on the full DataFrame (vectorised).
            3. Assembles the final signal column by selecting from the
               correct sub-strategy for each bar's regime.
            4. CASH regime → signal = 0 (flat).

        Returns
            df with "signal" (int), "_regime" (str), "_mom_composite" (float),
            "_mr_composite" (float) columns added.

        Code example
            >>> strategy = UnifiedAlphaStrategy()
            >>> df_out   = strategy.generate_signals(df.copy())
            >>> print(df_out[["Close", "_regime", "signal"]].tail(20))
        """
        # ── Regime classification ──────────────────────────────────────────
        regimes = self._regime_clf.classify(df)

        # ── Sub-strategy signals (run both, gate afterwards) ───────────────
        df_mom = self._momentum_strategy.generate_signals(df.copy())
        df_mr  = self._mr_strategy.generate_signals(df.copy())

        # ── Assemble gated signal ─────────────────────────────────────────
        signal = pd.Series(0, index=df.index, dtype=int)
        mom_mask = regimes == REGIME_MOMENTUM
        mr_mask  = regimes == REGIME_MEAN_REVERSION

        signal[mom_mask] = df_mom.loc[mom_mask, "signal"].values
        signal[mr_mask]  = df_mr.loc[mr_mask,  "signal"].values
        # CASH regime: signal stays 0

        df = df.copy()
        df["signal"]         = signal.values
        df["_regime"]        = regimes.values
        df["_mom_composite"] = df_mom.get("_composite",  pd.Series(np.nan, index=df.index)).values
        df["_mr_composite"]  = df_mr.get("_composite",   pd.Series(np.nan, index=df.index)).values
        return df


# ============================================================
# GENETIC OPTIMISER INTEGRATION
# ============================================================

try:
    from momentum.test.genetic_optimizer import ParameterSpace, IntParam, FloatParam

    UNIFIED_PARAM_SPACE = ParameterSpace(
        params={
            # Regime classifier thresholds
            "hmm_threshold":     FloatParam(0.45, 0.70),
            "vol_spike_mult":    FloatParam(1.2,  2.5),
            "adx_trend_min":     FloatParam(18.0, 32.0),
            "adx_flat_max":      FloatParam(14.0, 25.0),
            "er_trend_min":      FloatParam(0.40, 0.70),
            "er_flat_max":       FloatParam(0.15, 0.45),
            "min_regime_bars":   IntParam(2, 15),

            # Momentum sub-strategy
            "mom_entry_threshold": FloatParam(0.50, 0.75),
            "mom_exit_threshold":  FloatParam(0.20, 0.50),
            "mom_atr_stop_mult":   FloatParam(1.5, 4.0),

            # Mean reversion sub-strategy
            "mr_entry_threshold":  FloatParam(0.50, 0.75),
            "mr_exit_threshold":   FloatParam(0.20, 0.50),
            "mr_atr_stop_mult":    FloatParam(0.8,  2.5),
            "mr_time_stop_bars":   IntParam(3, 20),
            "mr_zscore_entry":     FloatParam(-2.5, -0.8),
        },
        constraints=[
            lambda p: p["adx_flat_max"]      < p["adx_trend_min"],
            lambda p: p["er_flat_max"]        < p["er_trend_min"],
            lambda p: p["mom_exit_threshold"] < p["mom_entry_threshold"],
            lambda p: p["mr_exit_threshold"]  < p["mr_entry_threshold"],
        ],
    )
    """
    Pre-built ParameterSpace for UnifiedAlphaStrategy.
    ---------------------------------------------------
    Tunes 15 parameters across three layers:
      1. Regime classifier thresholds (7 params)
      2. Momentum sub-strategy entry/exit/stop (3 params)
      3. Mean reversion sub-strategy entry/exit/stop/time/zscore (5 params)
    """

except ImportError:
    UNIFIED_PARAM_SPACE = None  # type: ignore


# ============================================================
# STANDALONE EXAMPLE
# ============================================================

def main() -> None:
    """Run UnifiedAlphaStrategy on AAPL and print ASCII report + regime breakdown."""
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    from momentum.test.strategy_tester import run_backtest, BacktestConfig
    from momentum.test.run_backtest_example import load_price_data, print_full_report

    print("\n  Loading AAPL data ...")
    df, source = load_price_data("AAPL", years=5)

    cfg = BacktestConfig(
        initial_capital      = 100_000,
        commission_per_share = 0.005,
        position_sizing      = "atr",
        atr_risk_pct         = 0.01,
        allow_short          = False,
    )

    strategy = UnifiedAlphaStrategy()
    print(f"  Running {strategy.name} ...")
    result   = run_backtest(strategy, df, cfg, symbol="AAPL")
    print_full_report(result, source)

    df_out = strategy.generate_signals(df.copy())
    print("  Regime distribution:")
    vc = pd.Series(df_out["_regime"].values).value_counts()
    total = len(df_out)
    for regime, count in vc.items():
        bar = "█" * int(count / total * 40)
        print(f"    {regime:<20} {bar}  {count} bars ({count/total:.0%})")
    print(f"\n  Current regime : {df_out['_regime'].iloc[-1]}")
    print(f"  Current signal : {'LONG' if df_out['signal'].iloc[-1] == 1 else 'FLAT'}\n")


if __name__ == "__main__":
    main()
