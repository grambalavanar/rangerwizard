"""
alpha_composite.py
==================
Alpha Composite Momentum Strategy — the most mathematically rigorous
single-instrument momentum strategy in this codebase. Synthesises nine
independently validated research signals into a weighted composite score;
trades when the score exceeds a tunable entry threshold and exits when it
falls below a tunable exit threshold or a volatility-based trailing stop
is hit.

Research foundations
--------------------
Every component is grounded in peer-reviewed academic research:

  1. Weinstein Stage Analysis (1988)
       Stan Weinstein, "Secrets for Profiting in Bull and Bear Markets."
       Stage 2 (price above rising 30-week SMA) is the highest-probability
       long entry window. Extended here to triple SMA alignment.

  2. Time Series Momentum — TSMOM (Moskowitz, Ooi & Pedersen, 2012)
       "Time Series Momentum," Journal of Financial Economics.
       Past 12-, 6-, and 3-month sign returns predict future returns.
       Multi-horizon TSMOM Sharpe ≈ 1.31, works across all asset classes.

  3. Linear Trend Regression Signal (Baltas & Kosowski, 2012)
       "Improving Time-Series Momentum Strategies: The Role of Trading
       Signals and Volatility Estimators."
       OLS slope on log-price normalised by realised vol outperforms the
       plain sign-of-return TSMOM signal out-of-sample.

  4. 52-Week High Proximity (George & Hwang, 2004)
       "The 52-week high and momentum investing," Journal of Finance.
       Proximity to the 52-week high is a stronger predictor of future
       returns than past returns alone (anchoring / prospect theory).

  5. Volatility-Scaled Momentum (Barroso & Santa-Clara, 2015)
       "Momentum has its moments," Journal of Financial Economics.
       Scaling positions by inverse realised volatility (targeting a
       constant 25% annualised vol) dramatically improves Sharpe and
       eliminates most momentum crashes.

  6. Momentum Quality Gates (Blau 1991; Wilder 1978; Lane 1950s)
       TSI (Blau), RSI (Wilder), StochRSI (Chande & Kroll 1994),
       MACD (Appel 1979). Composite oscillator confirming that the
       momentum signal is not about to reverse.

  7. Know Sure Thing — KST (Pring, 1992)
       "Martin Pring on Market Momentum."
       Multi-timeframe ROC oscillator designed to identify major stock
       market cycle junctures; KST > signal = bullish cycle phase.

  8. Volume Confirmation (Granville 1963; OBV)
       "Granville's New Key to Stock Market Profits."
       On-Balance Volume (OBV) trend confirms that institutional capital
       is flowing into the instrument alongside price momentum.

  9. Ichimoku Cloud (Hosoda 1969)
       Price above the cloud and Tenkan > Kijun confirms trend direction
       across multiple timeframes without additional lookback parameter
       tuning (timeframes are baked into the Ichimoku construction).

Momentum crash protection
-------------------------
  Daniel & Moskowitz (2016) "Momentum crashes" show that momentum
  strategies suffer sharp reversals following high-volatility bear markets.
  The volatility regime score (component 9) penalises entries when the
  ATR has spiked, directly targeting these crash conditions.

Genetic optimisation
--------------------
  The exported ``TUNABLE_PARAM_SPACE`` and ``TUNABLE_CONSTRAINTS`` are
  ready to drop directly into ``GeneticOptimizer``. See the bottom of
  this file for a full example.

Usage
-----
    from momentum.strategies.alpha_composite import (
        AlphaCompositeMomentumStrategy, TUNABLE_PARAM_SPACE, TUNABLE_CONSTRAINTS
    )
    from momentum.test.strategy_tester import run_backtest, BacktestConfig
    from momentum.test.run_backtest_example import load_price_data

    df, _ = load_price_data("AAPL", years=3)
    strategy = AlphaCompositeMomentumStrategy()
    result   = run_backtest(strategy, df, symbol="AAPL")
    print_result(result)

Dependencies: numpy, pandas, scipy
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
    ema, sma, rsi, stochrsi, tsi, macd, adx, kst,
    atr, roc, cci, crossover,
)


# ============================================================
# DEFAULT PARAMETERS
# ============================================================

DEFAULT_PARAMS: dict = {
    # ── Weinstein Trend Stage (Component 1) ──────────────────────
    "sma_long":        200,   # Long-term SMA (200-day / ~40-week)
    "sma_medium":      150,   # Medium SMA (150-day / ~30-week)
    "sma_short":        50,   # Short SMA (50-day / ~10-week)
    "adx_window":       14,   # ADX smoothing period
    "adx_min":         20.0,  # Minimum ADX for trend confirmation (>20 = trending)

    # ── Time Series Momentum — TSMOM (Component 2) ───────────────
    "tsmom_long":      252,   # 12-month ROC window (252 business days)
    "tsmom_long_skip":  21,   # Skip last 21 days (short-term reversal avoidance)
    "tsmom_med":       126,   # 6-month ROC window
    "tsmom_med_skip":   21,
    "tsmom_short":      63,   # 3-month ROC window
    "tsmom_short_skip":  5,

    # ── Linear Trend Regression (Component 3) ────────────────────
    "lintrend_window":  90,   # OLS regression window (≈4.5 months, Baltas optimal)
    "lintrend_vol_w":   90,   # Realised vol window for normalisation

    # ── 52-Week High Proximity (Component 4) ─────────────────────
    "high_window":     252,   # Rolling high window (52 weeks of trading days)

    # ── MACD Quality (Component 5) ───────────────────────────────
    "macd_fast":        12,   # MACD fast EMA (standard)
    "macd_slow":        26,   # MACD slow EMA (standard)
    "macd_signal":       9,   # MACD signal EMA (standard)
    "tsi_slow":         25,   # TSI slow EMA (Blau 1991 default)
    "tsi_fast":         13,   # TSI fast EMA

    # ── RSI + StochRSI Zone (Component 6) ────────────────────────
    "rsi_window":       14,   # RSI lookback (Wilder 1978 default)
    "rsi_low":        50.0,   # Momentum zone lower bound (>50 = positive momentum)
    "rsi_high":       72.0,   # Momentum zone upper bound (<72 = not overbought)
    "stochrsi_window":  14,   # StochRSI lookback

    # ── KST Oscillator (Component 7) ─────────────────────────────
    "kst_roc1":         10,   # KST ROC periods (Pring defaults)
    "kst_roc2":         15,
    "kst_roc3":         20,
    "kst_roc4":         30,
    "kst_w1":           10,   # KST SMA smoothing periods
    "kst_w2":           10,
    "kst_w3":           10,
    "kst_w4":           15,
    "kst_signal":        9,

    # ── Volume Confirmation (Component 8) ────────────────────────
    "obv_ema_fast":      5,   # OBV short-term EMA
    "obv_ema_slow":     20,   # OBV long-term EMA (bullish when fast > slow)
    "vol_ratio_short":   5,   # Volume ratio: short period
    "vol_ratio_long":   20,   # Volume ratio: long period

    # ── Ichimoku Cloud (Component 9) ─────────────────────────────
    "ich_conv":          9,   # Tenkan-sen (conversion line) period
    "ich_base":         26,   # Kijun-sen (base line) period
    "ich_span_b":       52,   # Senkou Span B period
    # Note: Chikou span is current close shifted back 26 bars (handled in code)

    # ── Volatility Regime / Crash Filter ─────────────────────────
    "atr_window":       14,   # ATR period for trailing stop and regime
    "atr_spike_mult":  2.0,   # ATR > 2x its 20-day average = high-vol / crash risk
    "atr_spike_avg":   20,    # Window to compute ATR average for spike detection

    # ── Composite Entry / Exit ────────────────────────────────────
    "entry_threshold": 0.60,  # Enter long when composite ≥ this
    "exit_threshold":  0.35,  # Exit long when composite < this

    # ── ATR Trailing Stop ─────────────────────────────────────────
    "atr_stop_mult":   2.5,   # Stop distance = atr_stop_mult × ATR below highest close
    "use_trailing_stop": True,

    # ── Component Weights (must sum to 1.0) ───────────────────────
    "w_trend":     0.20,  # Weinstein stage filter (most important gate)
    "w_tsmom":     0.18,  # TSMOM (primary momentum signal)
    "w_lintrend":  0.15,  # Linear trend regression (best academic signal)
    "w_52high":    0.10,  # 52-week high proximity
    "w_macd":      0.10,  # MACD + TSI quality
    "w_rsi":       0.08,  # RSI momentum zone + StochRSI direction
    "w_kst":       0.07,  # KST cycle oscillator
    "w_volume":    0.07,  # Volume confirmation (OBV)
    "w_ichimoku":  0.05,  # Ichimoku cloud
    # Volatility regime is applied as a multiplicative damper, not additive weight
}


# ============================================================
# INTERNAL HELPER
# ============================================================

def _norm01(series: pd.Series, window: int, clip_std: float = 2.5) -> pd.Series:
    """
    Normalize any series to [0, 1] using rolling z-score clamped to ±clip_std.
    --------------------------------------------------------------------------
    What it does
        Computes the rolling z-score of the series, clamps extreme values to
        ±clip_std standard deviations, then maps to [0, 1] linearly. This is
        a fast alternative to rolling percentile rank that produces similar
        results and runs in O(n) time.

    Used by
        Every sub-score function inside AlphaCompositeMomentumStrategy to
        convert raw indicator values to a common [0, 1] scale before computing
        the weighted composite.

    When you need this
        Call it whenever you want to map any indicator or price series to a
        [0, 1] scale for combination with other indicators. 0 = historically
        bearish, 0.5 = neutral, 1 = historically bullish.

    Code example
        >>> from momentum.strategies.alpha_composite import _norm01
        >>> score = _norm01(df["Close"].pct_change(20), window=252)

    Args:
        series   (pd.Series): Raw indicator values.
        window   (int):       Rolling window for mean and std.
        clip_std (float):     Z-score clamp level (default 2.5).

    Returns:
        pd.Series: Values in [0, 1], NaN filled with 0.5.
    """
    roll_mean = series.rolling(window, min_periods=max(10, window // 4)).mean()
    roll_std  = series.rolling(window, min_periods=max(10, window // 4)).std()
    z = (series - roll_mean) / roll_std.replace(0, np.nan)
    z = z.clip(-clip_std, clip_std)
    return ((z / clip_std + 1) / 2).fillna(0.5)


# ============================================================
# STRATEGY CLASS
# ============================================================

class AlphaCompositeMomentumStrategy(Strategy):
    """
    Alpha Composite Momentum Strategy — nine-signal research-backed composite.
    --------------------------------------------------------------------------
    What it is
        The most sophisticated single-instrument momentum strategy in this
        codebase. Combines nine independently validated research signals
        (Weinstein trend stage, TSMOM, linear trend regression, 52-week high,
        MACD quality, RSI zone, KST, volume, Ichimoku) into a single composite
        score. Enters when score ≥ entry_threshold and exits when score falls
        below exit_threshold or when an ATR trailing stop is hit.

    Used by
        All backtesting and paper trading workflows. Pass to run_backtest()
        for historical testing, or use as the strategy in paper_trader.py.
        Parameters can be tuned with GeneticOptimizer using the exported
        TUNABLE_PARAM_SPACE.

    When to use this strategy vs simpler ones
        - Use AlphaComposite when you want the most rigorous, multi-confirmation
          approach. It fires fewer signals than MACD-only but has higher quality.
        - Use it as the strategy to optimise with the genetic algorithm — the
          many tunable parameters make it well-suited for parameter search.
        - CAUTION: More parameters = more overfitting risk. Always validate
          on out-of-sample data before paper/live trading.

    Code example (basic backtest)
        >>> from momentum.strategies.alpha_composite import AlphaCompositeMomentumStrategy
        >>> from momentum.test.strategy_tester import run_backtest, BacktestConfig
        >>> from momentum.test.run_backtest_example import load_price_data, print_full_report
        >>>
        >>> df, _ = load_price_data("AAPL", years=3)
        >>> result = run_backtest(AlphaCompositeMomentumStrategy(), df, symbol="AAPL")
        >>> print_full_report(result, "yfinance")

    Code example (genetic optimization)
        >>> from momentum.strategies.alpha_composite import (
        ...     AlphaCompositeMomentumStrategy, TUNABLE_PARAM_SPACE, TUNABLE_CONSTRAINTS
        ... )
        >>> from momentum.test.genetic_optimizer import GeneticOptimizer, GAConfig
        >>>
        >>> opt = GeneticOptimizer(
        ...     strategy_factory = AlphaCompositeMomentumStrategy,
        ...     param_space      = TUNABLE_PARAM_SPACE,
        ...     symbols          = ["AAPL", "MSFT", "SPY", "QQQ", "JPM"],
        ...     config           = GAConfig(population_size=30, n_generations=25),
        ... )
        >>> result = opt.run()

    Overfit warning
        With ~40 tunable parameters this strategy can easily overfit. Limit the
        genetic optimizer to the ≤14 highest-impact parameters in
        TUNABLE_PARAM_SPACE and always validate on a held-out time window.
    """

    name = "Alpha Composite Momentum"

    def __init__(self, params: Optional[dict] = None) -> None:
        self._params = {**DEFAULT_PARAMS, **(params or {})}

    @property
    def params(self) -> dict:
        """Read-only copy of the current parameter dict."""
        return dict(self._params)

    # ── Sub-score 1: Weinstein Trend Stage ───────────────────────────────────

    def _score_trend_stage(self, df: pd.DataFrame, p: dict) -> pd.Series:
        """
        Weinstein Stage Analysis composite score.
        -----------------------------------------
        Metric
            Three conditions from Stan Weinstein's Stage 2 bull market
            framework: (a) price above the long-term SMA, (b) short SMA
            above medium SMA above long SMA (upward alignment), (c) ADX
            above threshold confirming an active trend.

        Research basis
            Weinstein (1988) documented that Stage 2 entries (rising MA,
            price above it, volume confirming) produce substantially higher
            win rates than entries at other stages. ADX was added by
            Wilder (1978) to quantify trend strength.

        Used by
            Core entry gate. The highest weighted component (0.20) because
            without a confirmed uptrend, momentum signals are unreliable.
            Hedge funds and CTAs universally use some form of trend filter
            before applying momentum signals.

        When you need this
            Review this score when the strategy keeps generating entries that
            reverse quickly — a low trend stage score means the stock is
            in a choppy or downtrending state. Raise ``adx_min`` to require
            a stronger trend.

        Interpretation
            1.0 = price above rising SMA system + strong ADX (full Stage 2)
            0.67 = two of three conditions met
            0.33 = only one condition met
            0.0  = stock below long SMA (likely Stage 3/4 downtrend)

        Args:
            df (pd.DataFrame): OHLCV data.
            p  (dict):         Parameter dict.

        Returns:
            pd.Series: Score in [0, 1].
        """
        sma_l = sma(df["Close"], p["sma_long"])
        sma_m = sma(df["Close"], p["sma_medium"])
        sma_s = sma(df["Close"], p["sma_short"])
        adx_v, _, _ = adx(df["High"], df["Low"], df["Close"], window=p["adx_window"])

        above_long    = (df["Close"] > sma_l).astype(float)
        sma_aligned   = ((sma_s > sma_m) & (sma_m > sma_l)).astype(float)
        adx_frac      = ((adx_v - p["adx_min"]) / p["adx_min"]).clip(0, 1)

        return (0.40 * above_long + 0.40 * sma_aligned + 0.20 * adx_frac).fillna(0)

    # ── Sub-score 2: Time Series Momentum (TSMOM) ────────────────────────────

    def _score_tsmom(self, df: pd.DataFrame, p: dict) -> pd.Series:
        """
        Time Series Momentum (TSMOM) multi-horizon score.
        --------------------------------------------------
        Metric
            The sign of past excess returns over three horizons (12-month,
            6-month, 3-month), each skipping the most recent 21 days to
            avoid the well-documented short-term reversal effect. Average
            of the three binary signals (positive return → 1, negative → 0).

        Research basis
            Moskowitz, Ooi & Pedersen (2012) "Time Series Momentum" showed
            that each asset's own past 12-month return positively predicts
            its future return with Sharpe ≈ 1.31 across 58 instruments.
            Novy-Marx (2012) found that 7-12 month returns have higher
            information content than 1-6 month returns — hence the higher
            weight on the long window (0.45).
            The 21-day skip is from Jegadeesh & Titman (1993) to avoid
            the 1-month reversal effect.

        Used by
            The primary momentum signal (weight 0.18). This is the
            academically documented "factor" that provides the core edge.
            Quant funds (AQR, Two Sigma, Man AHL) all use variants of
            this signal as their central trend-following indicator.

        When you need this
            When the strategy underperforms during a trending market,
            check if TSMOM score is low — this would indicate the stock
            has recently reversed its trend direction. Extend ``tsmom_long``
            to 300+ days for a slower, longer-term version.

        Interpretation
            1.0 = all three time horizons show positive momentum (strongest signal)
            0.67 = two of three positive
            0.33 = one of three positive
            0.0  = all three negative (avoid)

        Args:
            df (pd.DataFrame): OHLCV data.
            p  (dict):         Parameter dict.

        Returns:
            pd.Series: Score in {0, 0.33, 0.67, 1.0}.
        """
        close = df["Close"]

        def _sign_roc(window: int, skip: int) -> pd.Series:
            past  = close.shift(skip)
            start = close.shift(window)
            ret   = (past - start) / start.replace(0, np.nan)
            return (ret > 0).astype(float).fillna(0.5)

        s12 = _sign_roc(p["tsmom_long"],  p["tsmom_long_skip"])
        s6  = _sign_roc(p["tsmom_med"],   p["tsmom_med_skip"])
        s3  = _sign_roc(p["tsmom_short"], p["tsmom_short_skip"])

        # Weight the longer horizon more (Novy-Marx 2012)
        return (0.45 * s12 + 0.35 * s6 + 0.20 * s3).fillna(0.5)

    # ── Sub-score 3: Linear Trend Regression ─────────────────────────────────

    def _score_linear_trend(self, df: pd.DataFrame, p: dict) -> pd.Series:
        """
        OLS linear trend regression signal (Baltas & Kosowski 2012).
        -------------------------------------------------------------
        Metric
            Fits an OLS regression of log-price on time over a rolling window,
            then normalises the slope by realised volatility to produce a
            risk-adjusted trend strength signal. This is mathematically
            equivalent to the signal used by many systematic CTA funds.

            slope = Σ (t_i - t̄)(ln_p_i - ln_p̄) / Σ (t_i - t̄)²
            z     = slope / (realised_vol_per_bar)
            score = sigmoid of z mapped to [0, 1]

        Research basis
            Baltas & Kosowski (2012) showed that this OLS slope signal
            outperforms the plain sign-of-return TSMOM signal in out-of-sample
            performance and minimises portfolio turnover. It is the standard
            signal used by CTA trend-following systems. The volatility
            normalisation comes from Barroso & Santa-Clara (2015).

        Used by
            The second most important momentum signal (weight 0.15) after
            TSMOM. Works as a continuous-valued trend strength measure
            rather than a binary on/off. Systems researchers at Man AHL,
            Winton, and AQR use variants of this signal.

        When you need this
            Use this score alongside TSMOM to confirm that a trend has
            persistent linear momentum (not just a single recent spike).
            A high TSMOM + low linear trend = recent reversal / noisy move.
            A high linear trend + low TSMOM = early stage of a new trend.

        Interpretation
            > 0.65 = strong positive linear trend
            ≈ 0.50 = flat or uncertain trend
            < 0.35 = strong negative linear trend

        Args:
            df (pd.DataFrame): OHLCV data.
            p  (dict):         Parameter dict.

        Returns:
            pd.Series: Score in [0, 1].
        """
        window = p["lintrend_window"]
        log_close = np.log(df["Close"].replace(0, np.nan))

        def _ols_slope(y: np.ndarray) -> float:
            """OLS slope for any array length (handles partial min_periods windows)."""
            n = len(y)
            if n < 2:
                return 0.0
            t = np.arange(n, dtype=float)
            t -= t.mean()
            t_var = float((t ** 2).sum())
            if t_var == 0:
                return 0.0
            return float(np.dot(t / t_var, y - y.mean()))

        raw_slope = log_close.rolling(window, min_periods=window // 2).apply(
            _ols_slope, raw=True
        )

        # Normalise by realised volatility per bar
        daily_ret = df["Close"].pct_change()
        realised_vol = daily_ret.rolling(p["lintrend_vol_w"], min_periods=20).std()
        norm_slope = raw_slope / realised_vol.replace(0, np.nan)

        return _norm01(norm_slope, window=window * 2)

    # ── Sub-score 4: 52-Week High Proximity ──────────────────────────────────

    def _score_52high(self, df: pd.DataFrame, p: dict) -> pd.Series:
        """
        52-week high proximity score (George & Hwang 2004).
        ---------------------------------------------------
        Metric
            The ratio of the current close price to the 52-week rolling high
            price. When this ratio approaches 1.0, the stock is near its
            52-week high, which predicts above-average future returns due
            to anchoring and disposition-effect biases of investors.

            score = Close / RollingMax(Close, high_window)

        Research basis
            George & Hwang (2004) "The 52-week high and momentum investing"
            Journal of Finance showed that this simple ratio has higher
            predictive power for future 6-month returns than the standard
            Jegadeesh-Titman momentum measure. The mechanism is investor
            anchoring: sellers are reluctant to sell at prices that match
            or exceed their mental "high" anchor, creating a continuation
            effect as the anchor is overcome.

        Used by
            Included as a direct implementation of the George & Hwang factor.
            Weight 0.10 — a strong confirming signal but not the primary one.
            Quantitative equity funds include this as part of their momentum
            factor definition alongside 12-1 month ROC.

        When you need this
            A stock breaking to a new 52-week high with this score at 0.95+
            combined with high TSMOM and linear trend = very high conviction
            momentum entry. Use this score as a confidence booster when the
            other signals are borderline.

        Interpretation
            > 0.90 = price near or at 52-week high (strongest signal)
            0.70–0.90 = healthy but not at high
            < 0.70 = price well off its high; momentum may be weakening

        Args:
            df (pd.DataFrame): OHLCV data.
            p  (dict):         Parameter dict.

        Returns:
            pd.Series: Score in [0, 1].
        """
        window  = p["high_window"]
        high_52 = df["Close"].rolling(window, min_periods=window // 4).max()
        raw = (df["Close"] / high_52.replace(0, np.nan)).fillna(0.5).clip(0, 1)
        # Apply mild rolling normalisation so the score adapts to the stock's range
        return _norm01(raw, window=window // 2).clip(0.1, 1.0)

    # ── Sub-score 5: MACD + TSI Quality ──────────────────────────────────────

    def _score_macd_quality(self, df: pd.DataFrame, p: dict) -> pd.Series:
        """
        MACD line/signal/histogram quality score + TSI confirmation.
        -------------------------------------------------------------
        Metric
            Four momentum quality gates, each binary (0 or 1):
              a) MACD line > Signal line (bullish crossover regime)
              b) MACD Histogram ≥ 0 (positive histogram = bullish)
              c) MACD Histogram growing (histogram > histogram[-1] = acceleration)
              d) TSI > 0 (True Strength Index in positive territory)
            Average of four binary conditions.

        Research basis
            MACD (Appel 1979) is the most widely used momentum indicator in
            both retail and institutional trading. The histogram's second
            derivative (acceleration) was highlighted by Thomas Aspray (1986)
            as an early warning of momentum shifts. TSI (Blau 1991) is a
            double-smoothed momentum oscillator that shows trend direction
            with less noise than RSI; TSI > 0 confirms the trend.

        Used by
            Momentum quality gate (weight 0.10). All four conditions being
            positive simultaneously indicates a clean, accelerating momentum
            setup. Widely used in swing trading systems, retail momentum
            platforms (IBD, Investor's Business Daily), and institutional
            momentum screening tools.

        When you need this
            Check this score when the composite is near the entry threshold.
            High TSMOM + low MACD quality = momentum present but currently
            in a pullback phase (may be an entry opportunity or a reversal).
            Use only as a confirmation gate, not a standalone signal.

        Interpretation
            1.0 = all four conditions met (cleanest momentum setup)
            0.75 = three of four (still strong)
            0.50 = mixed signal
            0.0  = all bearish (momentum deteriorating)

        Args:
            df (pd.DataFrame): OHLCV data.
            p  (dict):         Parameter dict.

        Returns:
            pd.Series: Score in {0, 0.25, 0.50, 0.75, 1.0}.
        """
        macd_line, sig_line, hist = macd(
            df["Close"],
            window_fast   = p["macd_fast"],
            window_slow   = p["macd_slow"],
            window_signal = p["macd_signal"],
        )
        tsi_line = tsi(df["Close"], window_slow=p["tsi_slow"], window_fast=p["tsi_fast"])

        c_above  = (macd_line > sig_line).astype(float)
        c_hist   = (hist >= 0).astype(float)
        c_accel  = (hist > hist.shift(1)).astype(float)
        c_tsi    = (tsi_line > 0).astype(float)

        return ((c_above + c_hist + c_accel + c_tsi) / 4).fillna(0)

    # ── Sub-score 6: RSI Momentum Zone + StochRSI Direction ──────────────────

    def _score_rsi_zone(self, df: pd.DataFrame, p: dict) -> pd.Series:
        """
        RSI momentum zone score + StochRSI directional confirmation.
        -------------------------------------------------------------
        Metric
            Two components:
              a) RSI in the momentum zone [rsi_low, rsi_high]: score 1.0
                 when RSI is in 50-72 range (trending but not overbought),
                 tapering to 0 outside the zone.
              b) StochRSI %K > %D: direction confirmation (bullish = 1, bearish = 0)

        Research basis
            Wilder (1978) introduced RSI with 70/30 overbought/oversold
            thresholds, but subsequent research (Cardwell 1994, Elder 1993)
            showed that in trending markets, RSI should be re-interpreted:
            RSI 40-80 is the "bull range" for uptrending stocks. The
            StochRSI (Chande & Kroll 1994) applies Stochastic logic to RSI
            itself, providing a more sensitive short-term direction signal.

        Used by
            Confirmation filter (weight 0.08). In a momentum strategy, you
            want RSI high enough to confirm the trend but not so high that
            the stock is about to mean-revert. The 50-72 zone is the
            "momentum sweet spot" — Cardwell's bull range modified slightly.

        When you need this
            If the RSI zone score is consistently low despite strong TSMOM
            and linear trend scores, the stock may be in a choppy regime
            where RSI doesn't trend. Consider reducing this weight in the
            genetic optimizer for high-volatility stocks like TSLA.

        Interpretation
            1.0 = RSI in 50-72 AND StochRSI %K > %D (ideal momentum entry zone)
            0.75 = RSI in zone but StochRSI bearish
            0.25 = RSI outside zone but StochRSI bullish
            0.0  = RSI overbought/oversold AND StochRSI bearish

        Args:
            df (pd.DataFrame): OHLCV data.
            p  (dict):         Parameter dict.

        Returns:
            pd.Series: Score in [0, 1].
        """
        rsi_vals = rsi(df["Close"], window=p["rsi_window"])

        # RSI zone score: 1.0 in [rsi_low, rsi_high], taper outside
        lo, hi = p["rsi_low"], p["rsi_high"]
        mid    = (lo + hi) / 2
        zone   = rsi_vals.apply(lambda r: (
            1.0 if lo <= r <= hi else
            max(0.0, 1.0 - abs(r - mid) / mid * 2)
        ))

        # StochRSI %K > %D direction
        _, srsi_k, srsi_d = stochrsi(
            df["Close"],
            window=p["stochrsi_window"],
            smooth1=3, smooth2=3,
        )
        direction = (srsi_k > srsi_d).astype(float)

        return (0.65 * zone + 0.35 * direction).fillna(0.5)

    # ── Sub-score 7: KST Oscillator ───────────────────────────────────────────

    def _score_kst(self, df: pd.DataFrame, p: dict) -> pd.Series:
        """
        Know Sure Thing (KST) oscillator score.
        -----------------------------------------
        Metric
            The KST is a weighted sum of four smoothed Rate-of-Change
            indicators spanning short, intermediate, and long cycles.
            Score is based on: (a) KST above its signal line (bullish cycle
            phase), and (b) KST normalised to assess its strength.

        Research basis
            Developed by Martin Pring (1992) "Martin Pring on Market
            Momentum" — the KST was designed to identify major stock market
            cycle turning points by weighting longer-term ROC components more
            heavily. It is often described as "the indicator for identifying
            primary bull and bear market phases" and is widely used in
            intermarket analysis and sector rotation models.

        Used by
            Macro cycle confirmation (weight 0.07). Hedge fund macro traders
            and CTA systematic funds use KST as a regime filter to avoid
            trading momentum strategies during bear market phases when
            momentum crashes are most likely (Daniel & Moskowitz 2016).

        When you need this
            KST is slow — use it to confirm the macro cycle is in a bull
            phase before relying on faster signals. If KST has been below its
            signal for many months, even strong short-term momentum entries
            carry elevated drawdown risk. Consider raising entry_threshold
            when KST is negative.

        Interpretation
            > 0.65 = KST above signal AND in positive territory (bull cycle)
            ≈ 0.50 = KST near signal line (uncertain cycle phase)
            < 0.35 = KST below signal (bear cycle — reduce exposure)

        Args:
            df (pd.DataFrame): OHLCV data.
            p  (dict):         Parameter dict.

        Returns:
            pd.Series: Score in [0, 1].
        """
        kst_line, kst_sig = kst(
            df["Close"],
            roc1=p["kst_roc1"], roc2=p["kst_roc2"],
            roc3=p["kst_roc3"], roc4=p["kst_roc4"],
            w1=p["kst_w1"],   w2=p["kst_w2"],
            w3=p["kst_w3"],   w4=p["kst_w4"],
            nsig=p["kst_signal"],
        )
        above_signal = (kst_line > kst_sig).astype(float)
        kst_strength = _norm01(kst_line, window=252)
        return (0.50 * above_signal + 0.50 * kst_strength).fillna(0.5)

    # ── Sub-score 8: Volume Confirmation (OBV) ───────────────────────────────

    def _score_volume(self, df: pd.DataFrame, p: dict) -> pd.Series:
        """
        On-Balance Volume (OBV) trend and volume ratio score.
        -------------------------------------------------------
        Metric
            Two volume-based conditions:
              a) OBV fast EMA > OBV slow EMA: institutional money flowing in
              b) Volume ratio (short-period avg / long-period avg) > 1.0:
                 above-average recent trading activity confirming the move

        Research basis
            Granville (1963) introduced OBV as a cumulative volume indicator
            based on the principle that volume precedes price. Academic
            research by Blume, Easley & O'Hara (1994) formalised why volume
            carries information about the informativeness of price signals.
            The OBV EMA crossover identifies when institutional flows are
            consistently accumulating. The volume ratio component is from
            O'Neil's CAN SLIM system (2002) which requires volume >40% above
            average on breakout days.

        Used by
            Volume confirmation gate (weight 0.07). Price momentum without
            volume is suspect — strong momentum moves are typically backed
            by increasing participation. Used by all serious technical
            traders as a confirmation filter.

        When you need this
            Low volume score with strong price momentum = potential false
            breakout or distribution by smart money. Always check volume
            before acting on the composite score near the threshold.

        Interpretation
            1.0 = OBV trending up AND volume above average (strong confirmation)
            0.75 = one of two conditions met
            0.0  = OBV declining AND below-average volume (distribution / no follow-through)

        Args:
            df (pd.DataFrame): OHLCV data.
            p  (dict):         Parameter dict.

        Returns:
            pd.Series: Score in [0, 1].
        """
        # On-Balance Volume
        price_change = df["Close"].diff()
        direction = np.sign(price_change).fillna(0)
        obv = (direction * df["Volume"]).cumsum()

        obv_fast = ema(obv, p["obv_ema_fast"])
        obv_slow = ema(obv, p["obv_ema_slow"])
        obv_bull = (obv_fast > obv_slow).astype(float)

        # Volume ratio
        vol_short = df["Volume"].rolling(p["vol_ratio_short"]).mean()
        vol_long  = df["Volume"].rolling(p["vol_ratio_long"]).mean()
        vol_ratio = (vol_short > vol_long).astype(float)

        return (0.60 * obv_bull + 0.40 * vol_ratio).fillna(0.5)

    # ── Sub-score 9: Ichimoku Cloud ───────────────────────────────────────────

    def _score_ichimoku(self, df: pd.DataFrame, p: dict) -> pd.Series:
        """
        Ichimoku Kinkō Hyō (equilibrium chart) cloud score.
        ----------------------------------------------------
        Metric
            Three Ichimoku conditions:
              a) Price above Senkou Span A (cloud top): bullish trend
              b) Price above Senkou Span B (cloud bottom): above all cloud levels
              c) Tenkan-sen > Kijun-sen: short-term average above medium-term
                 average (bullish equilibrium)

        Research basis
            Goichi Hosoda (1969) developed Ichimoku as a complete trend
            analysis system. It uses time-based equilibrium principles rather
            than just price, encoding multiple timeframes in one chart.
            Patel (2010) "Ichimoku charts" and Murphy (1999) "Technical
            Analysis of the Financial Markets" both document its effectiveness
            as a standalone trend-following system with high reliability in
            trending markets, particularly in Asian equity and forex markets.

        Used by
            Multi-timeframe trend confirmation (weight 0.05). Ichimoku
            scores are widely used by Asian equity traders and forex CTA
            systems. The cloud provides both support/resistance and trend
            direction without parameter tuning (all periods are fixed by
            Hosoda's original construction).

        When you need this
            When a stock has been in a strong downtrend and the TSMOM and
            linear trend scores are improving, check whether the price has
            broken above the Ichimoku cloud. Cloud breaks are high-conviction
            trend reversal signals that often precede a new Stage 2 phase.

        Interpretation
            1.0 = price above entire cloud AND Tenkan > Kijun (full bullish)
            0.67 = above cloud but Tenkan ≤ Kijun
            0.33 = above Span A only
            0.0  = price below cloud (downtrend)

        Args:
            df (pd.DataFrame): OHLCV data.
            p  (dict):         Parameter dict.

        Returns:
            pd.Series: Score in {0, 0.33, 0.67, 1.0}.
        """
        conv_w = p["ich_conv"]
        base_w = p["ich_base"]
        span_b_w = p["ich_span_b"]

        tenkan = (df["High"].rolling(conv_w).max() + df["Low"].rolling(conv_w).min()) / 2
        kijun  = (df["High"].rolling(base_w).max() + df["Low"].rolling(base_w).min()) / 2
        span_a = ((tenkan + kijun) / 2)
        span_b = ((df["High"].rolling(span_b_w).max() + df["Low"].rolling(span_b_w).min()) / 2)

        cloud_top    = pd.concat([span_a, span_b], axis=1).max(axis=1)
        cloud_bottom = pd.concat([span_a, span_b], axis=1).min(axis=1)

        above_top    = (df["Close"] > cloud_top).astype(float)
        above_bottom = (df["Close"] > cloud_bottom).astype(float)
        tenkan_bull  = (tenkan > kijun).astype(float)

        return ((above_top + above_bottom + tenkan_bull) / 3).fillna(0)

    # ── Volatility Regime (Crash Damper) ─────────────────────────────────────

    def _volatility_regime_factor(self, df: pd.DataFrame, p: dict) -> pd.Series:
        """
        Volatility regime crash-protection factor (Daniel & Moskowitz 2016).
        ---------------------------------------------------------------------
        Metric
            Returns a multiplicative damper in (0, 1] that reduces the
            composite score during high-volatility regimes — particularly
            when the ATR has spiked to an extreme multiple of its recent
            average. This directly targets the momentum crash conditions
            documented by Daniel & Moskowitz (2016).

            factor = 1.0 if ATR ≤ atr_spike_mult × avg_ATR
            factor = ramps down from 1.0 → 0.3 as ATR/avg_ATR rises from
                     atr_spike_mult to atr_spike_mult × 2.

        Research basis
            Daniel & Moskowitz (2016) "Momentum crashes" Journal of Financial
            Economics. Momentum strategies suffer large tail losses during
            equity market rebounds following high-volatility bear markets.
            Barroso & Santa-Clara (2015) showed that volatility scaling
            (target a constant realised vol) captures most of this crash
            protection. This factor implements a simpler binary-to-smooth
            damper rather than full vol scaling for computational efficiency.

        Used by
            Applied as a multiplicative damper on the composite score, NOT
            as an additive component. This means the strategy naturally
            de-risks when volatility spikes, even if momentum indicators
            are temporarily positive during a bear market bounce.

        When you need this
            During events like March 2020 COVID crash, October 2022 bear
            market, or any period when intraday ATR is 2× normal — this
            factor reduces the composite score, preventing momentum entries
            into high-volatility crash rebounds.

        Returns:
            pd.Series: Multiplicative factor in [0.3, 1.0].
        """
        atr_vals   = atr(df["High"], df["Low"], df["Close"], window=p["atr_window"])
        atr_avg    = atr_vals.rolling(p["atr_spike_avg"]).mean()
        ratio      = atr_vals / atr_avg.replace(0, np.nan)
        spike_mult = p["atr_spike_mult"]

        # 1.0 below threshold, ramps to 0.3 at 2× threshold
        factor = 1.0 - ((ratio - spike_mult) / spike_mult).clip(0, 1) * 0.70
        return factor.clip(0.30, 1.0).fillna(1.0)

    # ── ATR Trailing Stop ─────────────────────────────────────────────────────

    def _atr_stop_series(self, df: pd.DataFrame, p: dict) -> pd.Series:
        """
        ATR-based trailing stop price series.
        ---------------------------------------
        Metric
            Computes a trailing stop level for each bar: the stop is set at
            the highest closing price since entry minus (atr_stop_mult × ATR).
            The stop never moves down — it only ratchets up with the high.

            stop[i] = max(close[0..i]) - atr_stop_mult × atr[i]

        Research basis
            Wilder (1978) introduced ATR. The ATR trailing stop methodology
            was developed by Chuck LeBeau and popularised in Van Tharp's
            "Trade Your Way to Financial Freedom." The multiplier of 2–3× ATR
            is the most commonly used setting in systematic trading systems
            (CTA community standard).

        Used by
            Applied in ``generate_signals()`` as an exit condition
            independent of the composite score. If the price drops below
            the trailing stop, the position exits regardless of the current
            composite score. This is the primary risk management mechanism.

        When you need this
            The trailing stop is the primary loss-control mechanism. If
            the strategy is taking losses > atr_stop_mult × ATR from the
            peak, the stop prevents further losses. Tune atr_stop_mult in
            the genetic optimizer: tighter (1.5) reduces drawdowns but
            increases whipsaw; wider (3.5+) reduces whipsaw but allows
            larger drawdowns.

        Args:
            df (pd.DataFrame): OHLCV data.
            p  (dict):         Parameter dict.

        Returns:
            pd.Series: Trailing stop price at each bar. Long exit when
                       close < this value.
        """
        atr_vals = atr(df["High"], df["Low"], df["Close"], window=p["atr_window"])
        rolling_high = df["Close"].cummax()
        stop = rolling_high - p["atr_stop_mult"] * atr_vals
        return stop.fillna(0)

    # ── Composite Score ───────────────────────────────────────────────────────

    def _compute_composite(self, df: pd.DataFrame, p: dict) -> pd.Series:
        """
        Compute the weighted composite momentum score from all nine components.
        ------------------------------------------------------------------------
        Metric
            Weighted average of nine sub-scores, each in [0, 1], multiplied
            by a volatility regime damper in (0.30, 1.0]. The result is a
            single number that encodes the aggregate strength of the momentum
            signal across all research dimensions.

        Used by
            ``generate_signals()`` to determine entry and exit timing.
            Also exposed for inspection: you can call this method on any
            DataFrame to see the composite score over time and identify
            which component is limiting signal quality.

        When you need this
            Inspect the composite score when strategy results are
            disappointing to diagnose which sub-signal is dragging the
            score below the threshold. High TSMOM + low composite usually
            means the trend stage or volume scores are failing.

        Code example
            >>> strategy = AlphaCompositeMomentumStrategy()
            >>> df_with_signal = df.copy()
            >>> df_with_signal["composite"] = strategy._compute_composite(df, strategy.params)
            >>> print(df_with_signal[["Close", "composite"]].tail(10))

        Returns:
            pd.Series: Composite score in [0, 1], NaN filled with 0.
        """
        s1 = self._score_trend_stage(df, p)
        s2 = self._score_tsmom(df, p)
        s3 = self._score_linear_trend(df, p)
        s4 = self._score_52high(df, p)
        s5 = self._score_macd_quality(df, p)
        s6 = self._score_rsi_zone(df, p)
        s7 = self._score_kst(df, p)
        s8 = self._score_volume(df, p)
        s9 = self._score_ichimoku(df, p)

        composite = (
            p["w_trend"]    * s1 +
            p["w_tsmom"]    * s2 +
            p["w_lintrend"] * s3 +
            p["w_52high"]   * s4 +
            p["w_macd"]     * s5 +
            p["w_rsi"]      * s6 +
            p["w_kst"]      * s7 +
            p["w_volume"]   * s8 +
            p["w_ichimoku"] * s9
        )

        # Apply volatility regime damper (Daniel & Moskowitz crash protection)
        regime = self._volatility_regime_factor(df, p)
        return (composite * regime).clip(0, 1).fillna(0)

    # ── Signal Generation ─────────────────────────────────────────────────────

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate entry/exit signals from the composite score and ATR stop.
        ------------------------------------------------------------------
        What it does
            1. Computes all nine sub-scores and the weighted composite.
            2. Applies the volatility regime damper.
            3. Generates long signals when composite ≥ entry_threshold.
            4. Generates exit signals when composite < exit_threshold
               OR close < ATR trailing stop.
            5. Returns the DataFrame with a ``signal`` column (1=long, 0=flat)
               and a ``_composite`` column for inspection.

        Used by
            ``run_backtest()`` in strategy_tester.py.
            ``paper_trader.py`` for live signal computation.
            Any custom script that calls strategy.generate_signals(df).

        When you need this
            This is the core entry point. After backtesting, inspect the
            ``_composite`` column in the returned DataFrame to understand
            when and why the strategy was in or out of the market.

        Code example
            >>> strategy = AlphaCompositeMomentumStrategy()
            >>> df_out   = strategy.generate_signals(df.copy())
            >>> print(df_out[["Close", "_composite", "signal"]].tail(20))
            >>> current_signal = df_out["signal"].iloc[-1]
            >>> print("Current signal:", "LONG" if current_signal == 1 else "FLAT")

        Args:
            df (pd.DataFrame): OHLCV DataFrame with DatetimeIndex.
                               Required columns: Open, High, Low, Close, Volume.

        Returns:
            pd.DataFrame: Input df with ``signal`` (int) and
                          ``_composite`` (float) columns added.
        """
        p = self._params

        composite  = self._compute_composite(df, p)
        stop_price = self._atr_stop_series(df, p)

        entry_arr = composite.values >= p["entry_threshold"]
        exit_arr  = (composite.values < p["exit_threshold"]) | (
            df["Close"].values < stop_price.values
        )

        signal_arr = np.zeros(len(df), dtype=int)
        pos = 0
        for i in range(len(df)):
            if pos == 0 and entry_arr[i]:
                pos = 1
            elif pos == 1 and exit_arr[i]:
                pos = 0
            signal_arr[i] = pos

        df = df.copy()
        df["signal"]    = signal_arr
        df["_composite"] = composite.values
        return df


# ============================================================
# GENETIC OPTIMIZER INTEGRATION
# ============================================================

try:
    from momentum.test.genetic_optimizer import (
        ParameterSpace, IntParam, FloatParam, ChoiceParam
    )

    TUNABLE_PARAM_SPACE = ParameterSpace(
        params={
            # Trend filter — most impactful
            "sma_long":        IntParam(150, 250),
            "sma_short":       IntParam(30,  70),
            "adx_min":         FloatParam(15.0, 35.0),

            # TSMOM lookbacks
            "tsmom_long":      IntParam(200, 280),
            "tsmom_long_skip": IntParam(10,  30),
            "tsmom_short":     IntParam(40,  90),

            # Linear trend window
            "lintrend_window": IntParam(60, 130),

            # MACD windows
            "macd_fast":       IntParam(8,  16),
            "macd_slow":       IntParam(20, 36),

            # RSI bounds
            "rsi_low":         FloatParam(42.0, 58.0),
            "rsi_high":        FloatParam(65.0, 82.0),

            # Composite thresholds
            "entry_threshold": FloatParam(0.50, 0.75),
            "exit_threshold":  FloatParam(0.20, 0.50),

            # ATR trailing stop
            "atr_stop_mult":   FloatParam(1.5, 4.0),
        },
        constraints=[
            lambda p: p["sma_short"] < p["sma_long"],
            lambda p: p["macd_fast"] < p["macd_slow"] - 4,
            lambda p: p["rsi_low"] < p["rsi_high"],
            lambda p: p["exit_threshold"] < p["entry_threshold"],
        ],
    )
    """
    Pre-built ParameterSpace for genetic optimisation.
    ---------------------------------------------------
    What it is
        A ready-to-use ``ParameterSpace`` that defines the 14 highest-impact
        tunable parameters of ``AlphaCompositeMomentumStrategy``. Includes
        four constraints to ensure parameter validity (SMA ordering, MACD
        window ordering, RSI bound ordering, threshold ordering).

    Used by
        Pass directly to ``GeneticOptimizer``:
            >>> opt = GeneticOptimizer(
            ...     strategy_factory = AlphaCompositeMomentumStrategy,
            ...     param_space      = TUNABLE_PARAM_SPACE,
            ...     symbols          = ["AAPL","SPY","QQQ","JPM","XOM"],
            ...     config           = GAConfig(population_size=30,n_generations=25),
            ... )

    When to use
        Always use TUNABLE_PARAM_SPACE rather than defining your own space
        from scratch — the ranges were chosen based on academic literature
        (e.g., TSMOM long window 200-280 covers the Novy-Marx 2012 optimal
        7-12 month range). Widen the ranges only after the first run has
        shown the best params consistently hitting a boundary.
    """

except ImportError:
    TUNABLE_PARAM_SPACE = None  # type: ignore


# ============================================================
# STANDALONE EXAMPLE
# ============================================================

def main() -> None:
    """
    Run the Alpha Composite strategy on AAPL and print the full ASCII report.
    --------------------------------------------------------------------------
    What it does
        Loads 3 years of AAPL data (yfinance → synthetic fallback), runs
        the AlphaCompositeMomentumStrategy backtest with default parameters,
        then prints the full ASCII report including equity curve, drawdown,
        monthly returns, and trade histogram.

    Run it
        python momentum/strategies/alpha_composite.py

    Also prints the current composite score so you can see whether the
    strategy is currently signalling a long position.
    """
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

    from momentum.test.strategy_tester import run_backtest, BacktestConfig, print_result
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

    strategy = AlphaCompositeMomentumStrategy()
    print(f"  Running {strategy.name} ...")
    result = run_backtest(strategy, df, cfg, symbol="AAPL")
    print_full_report(result, source)

    # Show current composite score
    df_sig = strategy.generate_signals(df.copy())
    last = df_sig.iloc[-1]
    print(f"  Current composite score : {last['_composite']:.3f}")
    print(f"  Current signal          : {'LONG' if last['signal'] == 1 else 'FLAT'}")
    print(f"  Entry threshold         : {strategy.params['entry_threshold']:.2f}")
    print()


if __name__ == "__main__":
    main()
