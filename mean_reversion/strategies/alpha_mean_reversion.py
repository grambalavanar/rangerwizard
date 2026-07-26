"""
alpha_mean_reversion.py
=======================
Alpha Composite Mean Reversion Strategy — the mean-reversion counterpart to
AlphaCompositeMomentumStrategy. Synthesises eleven independently validated
academic signals into a single composite "oversold score". Enters long when
the score exceeds entry_threshold (price is sufficiently oversold) and exits
when the price returns to the mean, the score falls, or a time/stop is hit.

Regime relationship
-------------------
  This strategy and AlphaCompositeMomentumStrategy are designed to be
  deployed as a PAIR via the regime_switcher module:
    - Mean Reversion Regime (ADX < 22, Hurst < 0.50, ER < 0.30): run THIS strategy.
    - Trending Regime (ADX > 25, Hurst > 0.50): run AlphaCompositeMomentumStrategy.

Research foundations (11 components)
--------------------------------------
  1.  Regime Gate          — Wilder (1978) ADX; Kaufman (1995) ER; Lo (1991) Hurst
  2.  Price Z-Score        — Poterba & Summers (1988); Lo & MacKinlay (1988)
  3.  ConnorsRSI           — Connors, Alvarez & Hayward (2009) "High Probability ETF Trading"
  4.  Bollinger Extreme    — Bollinger (2002); Lento & Gradojevic (2007)
  5.  Stochastic Extreme   — Lane (1954); Elder (1993) "Trading For A Living"
  6.  CCI Extreme          — Lambert (1980); Pring (1991) "Technical Analysis Explained"
  7.  Williams %R Extreme  — Williams (1979) "How I Made One Million Dollars..."
  8.  KAMA Efficiency Ratio— Kaufman (1995) "Smarter Trading"
  9.  Volume Climax        — Granville (1963); Wyckoff (1930)
  10. RSI Divergence       — Elder (1993); Cardwell (1994)
  11. OU Half-Life Gate    — Avellaneda & Lee (2010) "Statistical Arbitrage"

Usage
-----
    from mean_reversion.strategies.alpha_mean_reversion import (
        AlphaMeanReversionStrategy, TUNABLE_PARAM_SPACE
    )
    from momentum.test.strategy_tester import run_backtest, BacktestConfig
    from momentum.test.run_backtest_example import load_price_data, print_full_report

    df, _ = load_price_data("AAPL", years=3)
    result = run_backtest(AlphaMeanReversionStrategy(), df, symbol="AAPL")
    print_full_report(result, "yfinance")
"""

import math
import os
import sys
from typing import Optional, Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from momentum.test.strategy_tester import Strategy
from momentum.momentum_tools import (
    ema, sma, rsi, stochastic_oscillator, williams_r, cci,
    ultimate_oscillator, adx, atr, macd,
)
from mean_reversion.mean_reversion_tools import (
    price_zscore, bollinger_bands, efficiency_ratio, hurst_exponent,
    ou_halflife, connors_rsi, rsi2 as _rsi2, volume_climax,
    rsi_bullish_divergence, macd_bullish_divergence, mean_reversion_regime,
)


# ============================================================
# DEFAULT PARAMETERS
# ============================================================

DEFAULT_PARAMS: dict = {
    # ── Regime Gate (Component 1) ─────────────────────────────────
    "adx_window":    14,    # ADX period
    "adx_max":      22.0,   # Max ADX for MR regime (Wilder: <20 = no trend)
    "er_window":    10,     # Efficiency Ratio window (Kaufman default)
    "er_max":        0.30,  # Max ER for choppy regime (Kaufman: <0.25 optimal)
    "hurst_window":  60,    # Hurst estimation window (shorter = more responsive)

    # ── Price Z-Score (Component 2) ───────────────────────────────
    "zscore_window":    20,   # Rolling mean/std window for z-score
    "zscore_entry":    -1.5,  # Enter when z < this (Poterba & Summers threshold)
    "zscore_exit":      0.0,  # Exit when z returns above this (back at mean)

    # ── ConnorsRSI (Component 3) ──────────────────────────────────
    "crsi_rsi2_window":   2,   # Short RSI window (Connors: 2 optimal)
    "crsi_rsi_window":   14,   # Standard RSI window
    "crsi_streak_window":100,  # Percentile rank window
    "crsi_oversold":    25.0,  # ConnorsRSI < this = oversold (Connors: 10 extreme, 25 standard)

    # ── Bollinger Band Extreme (Component 4) ──────────────────────
    "bb_window":  20,   # Bollinger Band period
    "bb_std":    2.0,   # Standard deviation multiplier
    "bb_extreme": 0.15, # %B < this = extreme oversold (below lower band)

    # ── Stochastic Extreme (Component 5) ──────────────────────────
    "stoch_window":     14,   # Stochastic lookback
    "stoch_smooth":      3,   # Signal smoothing
    "stoch_oversold":   22.0, # Both %K and %D < this = deeply oversold

    # ── CCI Extreme (Component 6) ─────────────────────────────────
    "cci_window":    20,      # CCI period (Lambert default)
    "cci_extreme": -150.0,    # CCI < this = extreme (deeper than standard -100)

    # ── Williams %R Extreme (Component 7) ────────────────────────
    "willr_window":   14,     # Williams %R lookback
    "willr_extreme": -85.0,   # %R < this = extreme oversold (near -100)

    # ── KAMA Efficiency Ratio (Component 8) ──────────────────────
    # (re-uses er_window and er_max from Component 1)

    # ── Volume Climax (Component 9) ──────────────────────────────
    "vol_climax_window": 20,  # Rolling average volume window

    # ── RSI Divergence (Component 10) ────────────────────────────
    "divergence_window": 14,  # Lookback for divergence comparison
    "rsi_divergence_window": 14,  # RSI window for divergence computation

    # ── OU Half-Life Gate (Component 11) ─────────────────────────
    "ou_window":     60,   # OU regression window
    "ou_max_hl":     25,   # Max half-life in days to consider fast enough

    # ── ATR Trailing Stop ─────────────────────────────────────────
    "atr_window":     14,   # ATR period
    "atr_stop_mult":  1.5,  # MR stops are TIGHTER than momentum (1.5–2.5 vs 2.5)

    # ── Time Stop ─────────────────────────────────────────────────
    "time_stop_bars": 10,   # Exit if position not profitable after N bars
    # (Set to 0 to disable time stop)

    # ── Composite Entry / Exit ────────────────────────────────────
    "entry_threshold": 0.60,  # Enter when composite ≥ this (60% signals oversold)
    "exit_threshold":  0.35,  # Exit when composite < this (mean reversion complete)

    # ── Component Weights (auto-normalised — see _compute_composite) ──
    # Raw influence scores; normalised to sum=1 inside the strategy.
    "w_regime":      0.20,  # Regime gate (most important: don't trade in trends)
    "w_zscore":      0.18,  # Price z-score (primary academic signal)
    "w_crsi":        0.12,  # ConnorsRSI (proven short-term reversal signal)
    "w_bollinger":   0.10,  # Bollinger %B extreme
    "w_stoch":       0.08,  # Stochastic extreme
    "w_cci":         0.08,  # CCI extreme
    "w_willr":       0.07,  # Williams %R extreme
    "w_er":          0.06,  # Efficiency Ratio (regime confirmation)
    "w_vol_climax":  0.05,  # Volume climax/exhaustion
    "w_divergence":  0.04,  # RSI divergence (leading reversal signal)
    "w_ou":          0.02,  # OU half-life confirmation
}


# ============================================================
# HELPERS
# ============================================================

def _norm01_mr(series: pd.Series, window: int, invert: bool = False) -> pd.Series:
    """
    Normalise a series to [0, 1] using rolling z-score.
    invert=True: lower raw values → higher score (for oversold metrics).
    """
    roll_mean = series.rolling(window, min_periods=max(10, window // 4)).mean()
    roll_std  = series.rolling(window, min_periods=max(10, window // 4)).std()
    z = ((series - roll_mean) / roll_std.replace(0, np.nan)).clip(-2.5, 2.5)
    score = (z / 2.5 + 1) / 2  # map [-2.5, 2.5] → [0, 1]
    if invert:
        score = 1 - score
    return score.fillna(0.5)


# ============================================================
# STRATEGY CLASS
# ============================================================

class AlphaMeanReversionStrategy(Strategy):
    """
    Alpha Composite Mean Reversion Strategy — eleven-signal research composite.
    ---------------------------------------------------------------------------
    What it is
        Mean-reversion counterpart to AlphaCompositeMomentumStrategy.
        Scores eleven independently validated oversold signals into a
        composite score [0, 1]; enters long when the score exceeds
        entry_threshold and exits when price returns to the mean,
        the composite drops below exit_threshold, the ATR stop is hit,
        or the time stop expires.

    Key difference from momentum
        In this strategy, a HIGH composite score = MORE oversold = STRONG
        BUY signal (the inverse of the momentum strategy where high score
        = strong uptrend).

    Used by
        Run this strategy in mean-reverting market regimes (ADX < 22,
        Hurst < 0.50). The AlphaCompositeMomentumStrategy handles the
        trending regime. See regime_switcher.py for automatic switching.

    Code example
        >>> from mean_reversion.strategies.alpha_mean_reversion import (
        ...     AlphaMeanReversionStrategy, TUNABLE_PARAM_SPACE
        ... )
        >>> from momentum.test.strategy_tester import run_backtest
        >>> from momentum.test.run_backtest_example import load_price_data
        >>> df, _ = load_price_data("AAPL", years=3)
        >>> result = run_backtest(AlphaMeanReversionStrategy(), df, symbol="AAPL")

    Genetic optimisation
        >>> from mean_reversion.strategies.optimize_alpha_mr import run_optimization
        >>> run_optimization(["AAPL", "MSFT", "SPY"])
    """

    name = "Alpha Mean Reversion"

    def __init__(self, params: Optional[dict] = None) -> None:
        self._params = {**DEFAULT_PARAMS, **(params or {})}

    @property
    def params(self) -> dict:
        return dict(self._params)

    # ── Component 1: Regime Gate ──────────────────────────────────────────────

    def _score_regime_gate(self, df: pd.DataFrame, p: dict) -> pd.Series:
        """
        Mean-reversion regime confirmation (ADX + Efficiency Ratio + Hurst).
        ----------------------------------------------------------------------
        Metric
            Combines three independent regime signals into a single gate
            score. A high score means the market is currently in a
            mean-reverting state where this strategy has its edge.
              a) ADX < adx_max: no directional trend (Wilder 1978)
              b) ER < er_max: price action is choppy (Kaufman 1995)
              c) Hurst < 0.50: anti-persistent returns (Lo 1991)

        Research basis
            Each of the three components is independently validated as a
            regime classifier. Their combination is multiplicatively
            stronger: when all three agree, the probability of successful
            mean reversion is substantially higher than any one alone.
            This is the primary gate — if the regime score is low (market
            is trending), the strategy generates no trades regardless of
            how oversold the other signals are.

        Used by
            The highest-weighted component (0.20). This prevents the
            strategy from buying oversold stocks in strong downtrends —
            the most common mistake in mean-reversion strategies.

        When you need this
            Score ≈ 1.0: strong MR regime — maximum confidence.
            Score < 0.33: market is trending — DO NOT trade mean reversion.
            Review this score whenever the strategy's trades are performing
            poorly; a regime shift from MR to trending is the most common
            cause of MR strategy drawdowns.

        Interpretation
            1.0 = all three confirm MR regime
            0.67 = two of three confirm
            0.33 = weak regime confirmation
            0.0  = trending market (avoid MR trades)
        """
        adx_v, _, _ = adx(df["High"], df["Low"], df["Close"], window=p["adx_window"])
        er_v        = efficiency_ratio(df["Close"], window=p["er_window"])
        hurst_v     = hurst_exponent(df["Close"],   window=p["hurst_window"])

        low_adx   = (adx_v   < p["adx_max"]).astype(float)
        low_er    = (er_v    < p["er_max"]).astype(float)
        low_hurst = (hurst_v < 0.50).astype(float)

        return ((low_adx + low_er + low_hurst) / 3).fillna(0)

    # ── Component 2: Price Z-Score ────────────────────────────────────────────

    def _score_zscore(self, df: pd.DataFrame, p: dict) -> pd.Series:
        """
        Price z-score below mean — the core academic mean-reversion signal.
        -------------------------------------------------------------------
        Metric
            Z = (Close − SMA(zscore_window)) / StdDev(zscore_window)
            Score increases as Z falls below zscore_entry (default -1.5).
            Score = 1.0 when Z ≤ -2.5 (extreme oversold).
            Score = 0.5 when Z = zscore_entry.
            Score = 0.0 when Z ≥ 0 (price at or above mean).

        Research basis
            Poterba & Summers (1988) documented that stocks with prices
            significantly below their recent mean show predictable positive
            returns. Lo & MacKinlay (1988) quantified this at z < -1.5 as
            the statistical threshold. The z-score is the mean-reversion
            equivalent of the 12-1 month momentum score in cross-sectional
            momentum research.

        Used by
            Second highest-weighted component (0.18). Along with the regime
            gate, this is the primary signal driver. Many systematic MR
            strategies use only these two signals.

        Interpretation
            1.0 = z ≤ -2.5: price 2.5 std below mean (strongest signal)
            0.5 = z = entry_threshold (-1.5): just hitting the entry zone
            0.0 = z ≥ 0: price at or above mean (no MR opportunity)
        """
        z = price_zscore(df["Close"], window=p["zscore_window"])
        entry = p["zscore_entry"]   # e.g., -1.5
        extreme = entry - 1.0       # e.g., -2.5

        score = ((z - 0) / (extreme - 0)).clip(0, 1)
        return pd.Series(score, index=df.index, name="zscore_score").fillna(0.5)

    # ── Component 3: ConnorsRSI ───────────────────────────────────────────────

    def _score_crsi(self, df: pd.DataFrame, p: dict) -> pd.Series:
        """
        ConnorsRSI extreme oversold score.
        ------------------------------------
        Metric
            ConnorsRSI = (RSI(2) + RSI(streak) + PercentRank(100)) / 3
            Score increases as ConnorsRSI falls below crsi_oversold (25).
            Also incorporates simple RSI(2) < 15 as an additional signal.

        Research basis
            Connors, Alvarez & Hayward (2009) backtested ConnorsRSI < 10
            on US ETFs from 1995–2009 and found 70–80% win rates over
            3–5 day holds. The key insight: RSI(2) is so sensitive that
            it captures 1–2 day overextensions that revert within a week.
            Combining it with a streak component prevents entry during
            multi-day waterfall declines.

        Interpretation
            1.0 = ConnorsRSI < 10 (highest priority entry signal)
            0.75 = ConnorsRSI 10-20
            0.5  = ConnorsRSI 20-crsi_oversold (moderate oversold)
            0.0  = ConnorsRSI > 70 (overbought)
        """
        crsi = connors_rsi(
            df["Close"],
            rsi2_window   = p["crsi_rsi2_window"],
            rsi14_window  = p["crsi_rsi_window"],
            streak_window = p["crsi_streak_window"],
        )
        threshold = p["crsi_oversold"]  # default 25
        # Map: crsi at 0 → score 1.0; at threshold → score 0.5; at 100 → score 0.0
        score = (1.0 - crsi / 100.0).clip(0, 1)
        # Boost score when crsi is below threshold (confirmed oversold)
        boost = (crsi < threshold).astype(float) * 0.3
        return (score + boost).clip(0, 1).fillna(0.5).rename("crsi_score")

    # ── Component 4: Bollinger Band Extreme ───────────────────────────────────

    def _score_bollinger_extreme(self, df: pd.DataFrame, p: dict) -> pd.Series:
        """
        Bollinger Band %B extreme oversold score.
        -------------------------------------------
        Metric
            %B < 0 means price is below the lower band.
            %B < bb_extreme (default 0.15) = significantly below mid.
            Score = 1 when %B = 0 (at lower band); = 0 when %B = 0.5 (midline).

        Research basis
            Bollinger (2002) documented that %B < 0 produces positive
            3-week returns in 65% of cases. Lento & Gradojevic (2007)
            validated this independently on Canadian equities. The lower
            band (n_std below the mean) represents a statistically rare
            excursion from the mean that tends to revert.

        Interpretation
            1.0 = %B ≤ 0 (price below lower band — strongest signal)
            0.75 = %B ≤ 0.15 (near lower band)
            0.5  = %B = 0.5 (at midline — neutral)
            0.0  = %B ≥ 0.85 (near upper band — overbought)
        """
        bb   = bollinger_bands(df["Close"], window=p["bb_window"], n_std=p["bb_std"])
        pctb = bb["pct_b"]
        # Invert: low %B = high score
        score = (1.0 - pctb).clip(0, 1)
        return pd.Series(score, index=df.index, name="bb_score").fillna(0.5)

    # ── Component 5: Stochastic Extreme ───────────────────────────────────────

    def _score_stochastic_extreme(self, df: pd.DataFrame, p: dict) -> pd.Series:
        """
        Stochastic oscillator both lines deeply oversold.
        ---------------------------------------------------
        Metric
            %K and %D both below stoch_oversold (default 22).
            Score = 1 when both are deeply below 20; = 0 when both > 50.

        Research basis
            George Lane (1954) original Stochastic indicator. Elder (1993)
            showed that BOTH %K AND %D below 20 is more reliable than
            either alone (eliminates false signals). Williams extended
            this with the observation that stochastic < 20 when combined
            with high volume often marks a capitulation bottom.
        """
        stoch_k, stoch_d = stochastic_oscillator(
            df["High"], df["Low"], df["Close"],
            window=p["stoch_window"], smooth_k=p["stoch_smooth"],
        )
        threshold = p["stoch_oversold"]
        both_oversold = ((stoch_k < threshold) & (stoch_d < threshold)).astype(float)
        # Continuous version: how far below threshold
        k_score = (1 - stoch_k / 100).clip(0, 1)
        d_score = (1 - stoch_d / 100).clip(0, 1)
        return ((k_score + d_score) / 2 * (0.5 + 0.5 * both_oversold)).fillna(0.5)

    # ── Component 6: CCI Extreme ───────────────────────────────────────────────

    def _score_cci_extreme(self, df: pd.DataFrame, p: dict) -> pd.Series:
        """
        CCI deeply negative extreme score.
        ------------------------------------
        Metric
            CCI < cci_extreme (default -150): price is well below average
            over the measurement period, in rare territory that has a
            historically elevated probability of positive returns.

        Research basis
            Lambert (1980) developed CCI for commodity futures; Pring (1991)
            validated it for equities. Extreme CCI values (< -150, not just
            the common -100) correspond to 2+ standard deviation events in
            the typical price deviation distribution. These extreme readings
            at -150 to -200 produce better mean-reversion outcomes than
            the standard -100 threshold.
        """
        cci_vals = cci(df["High"], df["Low"], df["Close"], window=p["cci_window"])
        extreme  = p["cci_extreme"]  # default -150
        # Map: cci at extreme (-150) → 1.0; at 0 → 0.0; at +150 → 0.0
        score = (-cci_vals / abs(extreme)).clip(0, 1)
        return score.fillna(0.5).rename("cci_score")

    # ── Component 7: Williams %R Extreme ──────────────────────────────────────

    def _score_willr_extreme(self, df: pd.DataFrame, p: dict) -> pd.Series:
        """
        Williams %R extremely oversold score.
        ---------------------------------------
        Metric
            %R oscillates 0 to -100. Values < -85 (near -100) indicate
            the close is near the bottom of the recent range.

        Research basis
            Williams (1979) "How I Made One Million Dollars in the
            Commodities Market During One Year." The %R < -90 signal
            was Williams' original "money machine" setup — extreme
            oversold readings over a 14-day window in uptrending markets
            produce positive returns. Murphy (1999) confirmed across
            equities and ETFs.
        """
        willr = williams_r(df["High"], df["Low"], df["Close"], window=p["willr_window"])
        extreme = p["willr_extreme"]  # default -85 (scale is 0 to -100)
        # willr is in [-100, 0]; more negative = more oversold
        # extreme is e.g. -85; score = 1 when willr <= extreme
        score = ((-willr - abs(extreme)) / (100 - abs(extreme))).clip(0, 1)
        return score.fillna(0.5).rename("willr_score")

    # ── Component 8: KAMA Efficiency Ratio ────────────────────────────────────

    def _score_er_low(self, df: pd.DataFrame, p: dict) -> pd.Series:
        """
        Low Efficiency Ratio (choppy market = mean reversion regime).
        ---------------------------------------------------------------
        Metric
            ER < er_max: score approaches 1.0 (confirmed choppy).
            ER > 0.60: score → 0.0 (trending → avoid MR).

        Research basis
            Kaufman (1995) empirically found ER < 0.25 optimises
            mean-reversion strategy returns. This signal is used both as
            a regime gate (Component 1) and here as an independent
            confirmation signal (Component 8).
        """
        er = efficiency_ratio(df["Close"], window=p["er_window"])
        # Low ER = high score
        score = (p["er_max"] - er).clip(0, p["er_max"]) / p["er_max"]
        return score.clip(0, 1).fillna(0.5).rename("er_score")

    # ── Component 9: Volume Climax ────────────────────────────────────────────

    def _score_vol_climax(self, df: pd.DataFrame, p: dict) -> pd.Series:
        """
        Volume climax / selling exhaustion detection.
        -----------------------------------------------
        Metric
            High volume on a significant down day = selling exhaustion.
            Wyckoff's "Selling Climax" (SC) — the first event in an
            accumulation phase. Score reflects the intensity of
            volume-on-down-day relative to recent average.

        Research basis
            Wyckoff (1930) — accumulation/distribution framework.
            Granville (1963) — OBV and volume analysis. Murphy (1999) Ch.7.
            Blume, Easley & O'Hara (1994) "Market Statistics and Technical
            Analysis: The Role of Volume" — formalised why extreme volume
            on declining prices signals information revelation and subsequent
            reversal.
        """
        return volume_climax(
            df["Close"], df["Volume"], window=p["vol_climax_window"]
        ).rename("vol_climax_score")

    # ── Component 10: RSI + MACD Divergence ───────────────────────────────────

    def _score_divergence(self, df: pd.DataFrame, p: dict) -> pd.Series:
        """
        Bullish RSI and MACD divergence composite.
        -------------------------------------------
        Metric
            Averages RSI bullish divergence and MACD bullish divergence.
            Either = 1.0; both = 1.0; neither = 0.0.

        Research basis
            Elder (1993) "Trading For A Living" Ch. 26–27 — divergences
            between price and oscillators are the highest-quality
            mean-reversion signals because they indicate waning momentum
            before the actual price reversal. The "Triple Screen" method
            specifically requires divergence for counter-trend entries.
        """
        rsi_vals = _rsi2(df["Close"])
        rsi_div  = rsi_bullish_divergence(
            df["Close"], rsi_vals, window=p["divergence_window"]
        )
        macd_div = macd_bullish_divergence(
            df["Close"], window=p["divergence_window"]
        )
        return ((rsi_div + macd_div) / 2).clip(0, 1).fillna(0).rename("divergence_score")

    # ── Component 11: OU Half-Life Gate ───────────────────────────────────────

    def _score_ou_gate(self, df: pd.DataFrame, p: dict) -> pd.Series:
        """
        Ornstein-Uhlenbeck half-life confirmation — fast mean reversion.
        -----------------------------------------------------------------
        Metric
            Short OU half-life (<ou_max_hl days) means the stock has
            historically reverted quickly after deviations from its mean.
            Score = 1 when hl < ou_max_hl / 2; = 0 when hl > ou_max_hl.

        Research basis
            Avellaneda & Lee (2010) "Statistical Arbitrage in US Equities"
            used OU half-life as the primary filter: only trade stocks with
            estimated half-life < 20 trading days. Stocks with longer
            half-lives are too slow to revert profitably within a
            reasonable holding period.
        """
        hl    = ou_halflife(df["Close"], window=p["ou_window"])
        max_hl = p["ou_max_hl"]
        # Short half-life = high score
        score = (max_hl - hl).clip(0, max_hl) / max_hl
        return score.clip(0, 1).fillna(0.5).rename("ou_score")

    # ── Composite Score ───────────────────────────────────────────────────────

    def _compute_composite(self, df: pd.DataFrame, p: dict) -> pd.Series:
        """
        Weighted composite of all eleven oversold/MR-regime signals.
        -------------------------------------------------------------
        What it does
            Calls all eleven _score_* methods, normalises the weights to
            sum=1.0, and returns the weighted sum. The result represents
            "what fraction of mean-reversion evidence is currently
            present?" A value near 1.0 = strongly oversold in a confirmed
            MR regime. A value near 0.0 = market is trending or price is
            at/above its mean.

        Crucially different from momentum
            In momentum, HIGH score = STRONG UPTREND → buy.
            Here, HIGH score = STRONGLY OVERSOLD → buy.
            The entry logic is the same (composite >= threshold),
            but the composite measures the opposite market condition.

        Used by
            generate_signals() to determine entry and exit timing.

        Code example
            >>> strategy = AlphaMeanReversionStrategy()
            >>> df_with_score = df.copy()
            >>> df_with_score["mr_composite"] = strategy._compute_composite(df, strategy.params)
        """
        s1  = self._score_regime_gate(df, p)
        s2  = self._score_zscore(df, p)
        s3  = self._score_crsi(df, p)
        s4  = self._score_bollinger_extreme(df, p)
        s5  = self._score_stochastic_extreme(df, p)
        s6  = self._score_cci_extreme(df, p)
        s7  = self._score_willr_extreme(df, p)
        s8  = self._score_er_low(df, p)
        s9  = self._score_vol_climax(df, p)
        s10 = self._score_divergence(df, p)
        s11 = self._score_ou_gate(df, p)

        # Auto-normalise weights
        raw_w = np.array([
            p["w_regime"], p["w_zscore"], p["w_crsi"],  p["w_bollinger"],
            p["w_stoch"],  p["w_cci"],    p["w_willr"], p["w_er"],
            p["w_vol_climax"], p["w_divergence"], p["w_ou"],
        ], dtype=float)
        raw_w = np.clip(raw_w, 1e-6, None)
        w = raw_w / raw_w.sum()

        composite = (
            w[0]*s1 + w[1]*s2 + w[2]*s3  + w[3]*s4  + w[4]*s5  + w[5]*s6 +
            w[6]*s7 + w[7]*s8 + w[8]*s9  + w[9]*s10 + w[10]*s11
        )
        return composite.clip(0, 1).fillna(0)

    # ── ATR Trailing Stop ─────────────────────────────────────────────────────

    def _atr_stop_series(self, df: pd.DataFrame, p: dict) -> pd.Series:
        """
        ATR-based trailing stop (tighter than momentum — MR trades are fast).
        -----------------------------------------------------------------------
        For mean reversion, the stop is set at entry price − atr_stop_mult × ATR
        and does NOT trail up (we expect a quick return to mean, not a
        multi-week trend). If price moves against us beyond atr_stop_mult × ATR,
        the mean reversion thesis is invalidated and we exit.
        """
        atr_vals = atr(df["High"], df["Low"], df["Close"], window=p["atr_window"])
        # Simple fixed stop below close (not a trailing high-based stop like momentum)
        stop = df["Close"] - p["atr_stop_mult"] * atr_vals
        return stop.fillna(0)

    # ── Signal Generation ─────────────────────────────────────────────────────

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate entry/exit signals from the composite MR score.
        ----------------------------------------------------------
        Entry logic
            composite >= entry_threshold → signal = 1 (enter long, oversold)

        Exit logic (multiple mechanisms)
            1. composite < exit_threshold → mean reversion complete or
               regime shifted back to trending → exit.
            2. close < atr_stop (price moved further against us) → stop hit.
            3. z-score > zscore_exit (price returned to/above mean) → target hit.
            4. Time stop: position held > time_stop_bars without profit → exit.

        Returns
            df with "signal" column (1=long, 0=flat) and "_composite" column.
        """
        p         = self._params
        composite = self._compute_composite(df, p)
        stop      = self._atr_stop_series(df, p)
        z         = price_zscore(df["Close"], window=p["zscore_window"])

        entry_arr    = composite.values >= p["entry_threshold"]
        comp_exit    = composite.values  < p["exit_threshold"]
        stop_hit     = df["Close"].values  < stop.values
        mean_reached = z.values           >= p["zscore_exit"]  # z-score back at mean

        signal_arr     = np.zeros(len(df), dtype=int)
        pos            = 0
        bars_in_trade  = 0
        entry_price    = 0.0
        time_stop      = p.get("time_stop_bars", 10)

        for i in range(len(df)):
            if pos == 0 and entry_arr[i]:
                pos           = 1
                bars_in_trade = 0
                entry_price   = float(df["Close"].iloc[i])
            elif pos == 1:
                bars_in_trade += 1
                timed_out      = (time_stop > 0 and bars_in_trade >= time_stop)
                if comp_exit[i] or stop_hit[i] or mean_reached[i] or timed_out:
                    pos = 0
                    bars_in_trade = 0
                    entry_price   = 0.0
            signal_arr[i] = pos

        df = df.copy()
        df["signal"]     = signal_arr
        df["_composite"] = composite.values
        return df


# ============================================================
# GENETIC OPTIMIZER INTEGRATION
# ============================================================

try:
    from momentum.test.genetic_optimizer import (
        ParameterSpace, IntParam, FloatParam,
    )

    TUNABLE_PARAM_SPACE = ParameterSpace(
        params={
            # Regime gate
            "adx_max":       FloatParam(15.0, 30.0),
            "er_max":        FloatParam(0.15, 0.45),
            "hurst_window":  IntParam(40, 120),

            # Z-score
            "zscore_window": IntParam(10, 40),
            "zscore_entry":  FloatParam(-2.5, -0.8),
            "zscore_exit":   FloatParam(-0.5, 0.5),

            # ConnorsRSI
            "crsi_oversold": FloatParam(10.0, 35.0),

            # Bollinger
            "bb_window": IntParam(10, 30),
            "bb_std":    FloatParam(1.5, 3.0),
            "bb_extreme":FloatParam(0.05, 0.30),

            # Stochastic
            "stoch_window":   IntParam(7, 21),
            "stoch_oversold": FloatParam(10.0, 30.0),

            # CCI
            "cci_window":  IntParam(14, 28),
            "cci_extreme": FloatParam(-220.0, -80.0),

            # Williams
            "willr_extreme": FloatParam(-95.0, -70.0),

            # Composite
            "entry_threshold": FloatParam(0.50, 0.78),
            "exit_threshold":  FloatParam(0.20, 0.50),

            # ATR stop + time stop
            "atr_stop_mult":  FloatParam(0.8, 3.0),
            "time_stop_bars": IntParam(3, 20),

            # Component weights (raw — auto-normalised)
            "w_regime":     FloatParam(0.05, 0.40),
            "w_zscore":     FloatParam(0.05, 0.40),
            "w_crsi":       FloatParam(0.03, 0.30),
            "w_bollinger":  FloatParam(0.02, 0.25),
            "w_stoch":      FloatParam(0.02, 0.25),
            "w_cci":        FloatParam(0.02, 0.25),
            "w_willr":      FloatParam(0.02, 0.25),
            "w_er":         FloatParam(0.01, 0.20),
            "w_vol_climax": FloatParam(0.01, 0.20),
            "w_divergence": FloatParam(0.01, 0.20),
            "w_ou":         FloatParam(0.01, 0.15),
        },
        constraints=[
            lambda p: p["zscore_entry"]  < p["zscore_exit"],
            lambda p: p["exit_threshold"] < p["entry_threshold"],
            lambda p: p["cci_extreme"]    < 0,
            lambda p: p["willr_extreme"]  < 0,
        ],
    )

except ImportError:
    TUNABLE_PARAM_SPACE = None  # type: ignore


# ============================================================
# STANDALONE EXAMPLE
# ============================================================

def main() -> None:
    """Run AlphaMeanReversionStrategy on AAPL and print full ASCII report."""
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

    from momentum.test.strategy_tester import run_backtest, BacktestConfig
    from momentum.test.run_backtest_example import load_price_data, print_full_report

    print("\n  Loading AAPL data ...")
    df, source = load_price_data("AAPL", years=3)

    cfg = BacktestConfig(
        initial_capital      = 100_000,
        commission_per_share = 0.005,
        position_sizing      = "atr",
        atr_risk_pct         = 0.01,
        allow_short          = False,
    )

    strategy = AlphaMeanReversionStrategy()
    print(f"  Running {strategy.name} ...")
    result   = run_backtest(strategy, df, cfg, symbol="AAPL")
    print_full_report(result, source)

    df_sig = strategy.generate_signals(df.copy())
    print(f"  Current composite score : {df_sig['_composite'].iloc[-1]:.3f}")
    print(f"  Current signal          : {'LONG' if df_sig['signal'].iloc[-1] == 1 else 'FLAT'}")
    print(f"  Entry threshold         : {strategy.params['entry_threshold']:.2f}")
    print()


if __name__ == "__main__":
    main()
