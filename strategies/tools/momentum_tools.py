"""
momentum_tools.py
=================
A comprehensive library of momentum trading indicators, trend metrics, risk
measures, and portfolio utilities sourced from top open-source quantitative
trading repositories:

  - bukosabino/ta        https://github.com/bukosabino/ta
  - ranaroussi/quantstats https://github.com/ranaroussi/quantstats
  - twopirllc/pandas-ta  https://github.com/twopirllc/pandas-ta
  - jmrichardson/tuneta  https://github.com/jmrichardson/tuneta
  - kernc/backtesting.py https://github.com/kernc/backtesting.py

All functions accept pandas Series / DataFrames of OHLCV price data and return
pandas Series of indicator values, making them drop-in compatible with any
DataFrame-based backtesting or live-trading workflow.

Dependencies: numpy, pandas  (pip install numpy pandas)

Usage quick-start
-----------------
    import pandas as pd
    from momentum.momentum_tools import rsi, macd, sharpe_ratio

    df = pd.read_csv("AAPL.csv", parse_dates=["Date"], index_col="Date")
    df["rsi"] = rsi(df["Close"])
    macd_line, signal, hist = macd(df["Close"])

Sections
--------
  1. Helpers & Moving Averages
  2. Momentum Oscillators
  3. Trend Strength Indicators
  4. Risk & Performance Metrics
  5. Cross-Sectional & Portfolio Momentum
  6. Signal Generation Utilities
"""

import numpy as np
import pandas as pd
from typing import Tuple


# ============================================================
# 1. HELPERS & MOVING AVERAGES
# ============================================================

def ema(series: pd.Series, window: int) -> pd.Series:
    """
    Exponential Moving Average (EMA)
    --------------------------------
    Metric
        A weighted moving average that applies exponentially decreasing
        weights to older data points, making it more responsive to recent
        price changes than a simple moving average.

    Used by
        Virtually all momentum and trend-following traders. EMA is the
        building block for MACD, PPO, KAMA, and many other indicators.
        Hedge funds and quant desks commonly use the 9, 12, 26, 50, and 200
        period EMAs.

    When to use
        Use EMA to smooth noisy price data and identify the current trend
        direction. A price above EMA suggests an uptrend; below suggests
        a downtrend. EMA crossovers (fast crosses slow) are common entry
        and exit signals in momentum strategies.

    Code example
        >>> from momentum.momentum_tools import ema
        >>> fast = ema(df["Close"], window=12)
        >>> slow = ema(df["Close"], window=26)
        >>> signal = fast > slow   # True = uptrend

    Args:
        series  (pd.Series): Closing price series.
        window  (int):       Number of periods for the EMA.

    Returns:
        pd.Series: EMA values indexed to the input series.
    """
    return series.ewm(span=window, adjust=False).mean()


def sma(series: pd.Series, window: int) -> pd.Series:
    """
    Simple Moving Average (SMA)
    ---------------------------
    Metric
        The arithmetic mean of prices over the past `window` periods.
        Equal weight is given to every data point in the window.

    Used by
        The most widely used baseline for trend identification. Retail
        traders, institutional desks, and systematic funds all reference
        the 20-, 50-, and 200-day SMA as key support/resistance levels.

    When to use
        Use SMA to establish a price baseline and identify trend direction.
        The Golden Cross (50-day SMA crosses above 200-day SMA) is one of
        the most cited momentum buy signals in equities. Use shorter SMAs
        (5–20) for faster, intraday momentum strategies.

    Code example
        >>> from momentum.momentum_tools import sma
        >>> trend = sma(df["Close"], window=50)
        >>> above_trend = df["Close"] > trend   # momentum filter

    Args:
        series  (pd.Series): Closing price series.
        window  (int):       Number of periods.

    Returns:
        pd.Series: SMA values.
    """
    return series.rolling(window=window).mean()


def wma(series: pd.Series, window: int) -> pd.Series:
    """
    Weighted Moving Average (WMA)
    -----------------------------
    Metric
        A moving average where each period receives a linearly increasing
        weight, so the most recent price has the highest weight.

    Used by
        Traders who want more responsiveness than SMA but with a smoother
        curve than EMA. Common in short-term momentum systems and
        intraday trading. Used by the Awesome Oscillator internally.

    When to use
        Use WMA when you need a moving average that reacts more quickly
        to recent price changes than SMA but less erratically than raw
        price. Particularly useful in fast-moving momentum environments
        (e.g., pre-earnings moves, opening range breakouts).

    Code example
        >>> from momentum.momentum_tools import wma
        >>> w = wma(df["Close"], window=9)

    Args:
        series  (pd.Series): Closing price series.
        window  (int):       Number of periods.

    Returns:
        pd.Series: WMA values.
    """
    weights = np.arange(1, window + 1, dtype=float)
    weights /= weights.sum()

    def _wma(x: np.ndarray) -> float:
        return float(np.dot(weights, x))

    return series.rolling(window=window).apply(_wma, raw=True)


# ============================================================
# 2. MOMENTUM OSCILLATORS
# ============================================================

def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """
    Relative Strength Index (RSI)
    -----------------------------
    Metric
        Compares the average size of recent up-moves to down-moves over
        `window` periods, producing a value between 0 and 100. RSI = 100
        when all moves are positive; RSI = 0 when all moves are negative.
        Formula: RSI = 100 - 100 / (1 + avg_gain / avg_loss)

    Used by
        One of the most widely used indicators across all asset classes.
        Retail traders use it to spot overbought/oversold conditions;
        institutional quants use it in mean-reversion and momentum factor
        models. Welles Wilder introduced it in 1978.

    When to use
        - RSI > 70: asset may be overbought — consider taking profits or
          waiting for a pullback before entering a long.
        - RSI < 30: asset may be oversold — potential long entry if the
          trend is still intact.
        - RSI divergence (price makes new high but RSI does not) warns
          of weakening momentum before a reversal.
        - In a strong uptrend, RSI often oscillates between 40–80.
          Use 40 as the support level (buy the dip) instead of 30.

    Code example
        >>> from momentum.momentum_tools import rsi
        >>> df["rsi"] = rsi(df["Close"], window=14)
        >>> buy_signal  = df["rsi"] < 30
        >>> sell_signal = df["rsi"] > 70

    Args:
        close  (pd.Series): Closing price series.
        window (int):       Lookback period (default 14).

    Returns:
        pd.Series: RSI values (0–100).

    Reference: https://www.investopedia.com/terms/r/rsi.asp
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return pd.Series(100 - (100 / (1 + rs)), index=close.index, name="rsi")


def stochastic_oscillator(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
    smooth_k: int = 3,
) -> Tuple[pd.Series, pd.Series]:
    """
    Stochastic Oscillator (%K and %D)
    ----------------------------------
    Metric
        Measures where the closing price sits relative to the high-low
        range over the past `window` periods:
            %K = 100 * (Close - Lowest Low) / (Highest High - Lowest Low)
            %D = SMA(%K, smooth_k)   ← signal line

    Used by
        George Lane introduced it in the 1950s. Widely used by swing
        traders and day traders to find momentum reversals. Common in
        equity, forex, and futures markets.

    When to use
        - %K > 80: overbought; %K < 20: oversold.
        - Buy signal: %K crosses above %D while both are below 20.
        - Sell signal: %K crosses below %D while both are above 80.
        - Use in combination with ADX: only take signals when ADX > 25
          (trend present) to avoid whipsaws in ranging markets.

    Code example
        >>> from momentum.momentum_tools import stochastic_oscillator
        >>> stoch_k, stoch_d = stochastic_oscillator(df["High"], df["Low"], df["Close"])
        >>> buy = (stoch_k < 20) & (stoch_k > stoch_d)

    Args:
        high     (pd.Series): High price series.
        low      (pd.Series): Low price series.
        close    (pd.Series): Closing price series.
        window   (int):       Lookback period for the range (default 14).
        smooth_k (int):       SMA period for %D signal line (default 3).

    Returns:
        Tuple[pd.Series, pd.Series]: (%K, %D) both 0–100.

    Reference: https://www.investopedia.com/terms/s/stochasticoscillator.asp
    """
    lowest_low = low.rolling(window=window).min()
    highest_high = high.rolling(window=window).max()
    stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    stoch_d = stoch_k.rolling(window=smooth_k).mean()
    return (
        pd.Series(stoch_k, index=close.index, name="stoch_k"),
        pd.Series(stoch_d, index=close.index, name="stoch_d"),
    )


def stochrsi(
    close: pd.Series,
    window: int = 14,
    smooth1: int = 3,
    smooth2: int = 3,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Stochastic RSI (StochRSI, %K, %D)
    ------------------------------------
    Metric
        Applies the Stochastic Oscillator formula to the RSI values
        instead of raw price, producing a more sensitive oscillator
        (0–1 scale) that responds faster to momentum shifts.
            StochRSI = (RSI - min(RSI,n)) / (max(RSI,n) - min(RSI,n))

    Used by
        Developed by Tushar Chande and Stanley Kroll (1994). Used heavily
        by crypto traders and short-term equity momentum traders who
        need faster signals than plain RSI. Popular in algorithmic
        systems that trade volatile assets.

    When to use
        - StochRSI > 0.8: extremely overbought (more extreme than RSI > 70).
        - StochRSI < 0.2: extremely oversold.
        - Use %K/%D crossovers as finer-grained entry timing within a
          broader RSI or trend-based setup.
        - Best on higher-volatility stocks or during high-volume sessions
          where RSI signals alone are too slow.

    Code example
        >>> from momentum.momentum_tools import stochrsi
        >>> srsi, srsi_k, srsi_d = stochrsi(df["Close"])
        >>> buy = (srsi_k < 0.2) & (srsi_k > srsi_d)

    Args:
        close   (pd.Series): Closing price series.
        window  (int):       RSI and stoch window (default 14).
        smooth1 (int):       EMA period for %K (default 3).
        smooth2 (int):       EMA period for %D (default 3).

    Returns:
        Tuple[pd.Series, pd.Series, pd.Series]: (stochrsi, %K, %D).

    Reference: https://www.investopedia.com/terms/s/stochrsi.asp
    """
    rsi_vals = rsi(close, window=window)
    rsi_min = rsi_vals.rolling(window=window).min()
    rsi_max = rsi_vals.rolling(window=window).max()
    raw = (rsi_vals - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan)
    k = raw.ewm(span=smooth1, adjust=False).mean()
    d = k.ewm(span=smooth2, adjust=False).mean()
    return (
        pd.Series(raw, index=close.index, name="stochrsi"),
        pd.Series(k,   index=close.index, name="stochrsi_k"),
        pd.Series(d,   index=close.index, name="stochrsi_d"),
    )


def williams_r(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
) -> pd.Series:
    """
    Williams %R
    -----------
    Metric
        Inverse of the Fast Stochastic Oscillator. Measures how close the
        current close is to the highest high over the lookback period,
        oscillating from 0 (at the top of the range) to -100 (at the bottom).
            %R = -100 * (Highest High - Close) / (Highest High - Lowest Low)

    Used by
        Developed by Larry Williams. Favored by short-term swing traders
        in equities, futures, and forex. Used by algorithmic systems as a
        fast momentum filter, especially for intraday or daily timeframes.

    When to use
        - %R above -20: overbought region — watch for reversal.
        - %R below -80: oversold region — watch for bounce/entry.
        - %R moving from below -80 to above -50 confirms bullish momentum.
        - Particularly useful as a confirmation tool for RSI signals.

    Code example
        >>> from momentum.momentum_tools import williams_r
        >>> df["willr"] = williams_r(df["High"], df["Low"], df["Close"])
        >>> buy = df["willr"] < -80

    Args:
        high   (pd.Series): High price series.
        low    (pd.Series): Low price series.
        close  (pd.Series): Closing price series.
        window (int):       Lookback period (default 14).

    Returns:
        pd.Series: Williams %R values (0 to -100).

    Reference: https://www.investopedia.com/terms/w/williamsr.asp
    """
    highest_high = high.rolling(window=window).max()
    lowest_low = low.rolling(window=window).min()
    wr = -100 * (highest_high - close) / (highest_high - lowest_low).replace(0, np.nan)
    return pd.Series(wr, index=close.index, name="williams_r")


def roc(close: pd.Series, window: int = 12) -> pd.Series:
    """
    Rate of Change (ROC)
    --------------------
    Metric
        Measures the percentage change in price from `window` periods ago
        to today. Also called the "momentum oscillator."
            ROC = 100 * (Close - Close[n]) / Close[n]

    Used by
        The core building block of most momentum factor models in academic
        finance (Jegadeesh & Titman 1993, Fama-French). Quant funds use
        ROC to rank stocks in cross-sectional momentum screens. Also used
        by retail traders for simple trend-following systems.

    When to use
        - ROC > 0: price is higher than it was n periods ago → upward
          momentum.
        - ROC < 0: negative momentum, consider staying in cash or shorting.
        - Use 12-month ROC (skip last month) for classical monthly
          cross-sectional momentum strategies.
        - Use 3–5 day ROC for short-term intraday momentum trades.
        - Combine with RSI: high ROC + RSI not yet overbought = strong
          momentum with room to run.

    Code example
        >>> from momentum.momentum_tools import roc
        >>> df["roc_12"] = roc(df["Close"], window=12)
        >>> strong_momentum = df["roc_12"] > 10   # +10% over 12 periods

    Args:
        close  (pd.Series): Closing price series.
        window (int):       Lookback period (default 12).

    Returns:
        pd.Series: ROC values (percent).

    Reference: https://school.stockcharts.com/doku.php?id=technical_indicators:rate_of_change_roc_and_momentum
    """
    roc_vals = 100 * (close - close.shift(window)) / close.shift(window).replace(0, np.nan)
    return pd.Series(roc_vals, index=close.index, name=f"roc_{window}")


def awesome_oscillator(
    high: pd.Series,
    low: pd.Series,
    fast: int = 5,
    slow: int = 34,
) -> pd.Series:
    """
    Awesome Oscillator (AO)
    -----------------------
    Metric
        Measures market momentum by computing the difference between a fast
        and slow SMA of the bar's midpoint price (H+L)/2. Developed by
        Bill Williams.
            AO = SMA(midpoint, 5) - SMA(midpoint, 34)

    Used by
        Popularised by Bill Williams in his "Trading Chaos" books. Widely
        used by retail traders and system developers. The zero-line
        crossover and twin-peaks patterns are common in retail trading
        communities. Institutional desks use it as a noise filter.

    When to use
        - AO crosses above zero → bullish momentum shift, potential buy.
        - AO crosses below zero → bearish momentum shift, potential sell.
        - "Saucer" pattern (three consecutive AO bars, middle is lowest,
          all above zero) = buy signal.
        - "Twin Peaks" below zero (second peak is higher than first) = buy.
        - Use together with Bill Williams' Alligator (3-EMA trend system)
          for context.

    Code example
        >>> from momentum.momentum_tools import awesome_oscillator
        >>> df["ao"] = awesome_oscillator(df["High"], df["Low"])
        >>> buy = df["ao"] > 0

    Args:
        high (pd.Series): High price series.
        low  (pd.Series): Low price series.
        fast (int):       Fast SMA period (default 5).
        slow (int):       Slow SMA period (default 34).

    Returns:
        pd.Series: Awesome Oscillator values.

    Reference: https://www.tradingview.com/wiki/Awesome_Oscillator_(AO)
    """
    mid = (high + low) / 2
    ao = mid.rolling(fast).mean() - mid.rolling(slow).mean()
    return pd.Series(ao, index=high.index, name="ao")


def tsi(close: pd.Series, window_slow: int = 25, window_fast: int = 13) -> pd.Series:
    """
    True Strength Index (TSI)
    -------------------------
    Metric
        Double-smoothed momentum indicator that shows both trend direction
        and overbought/oversold conditions. It divides a double-EMA of
        price changes by a double-EMA of absolute price changes, scaling
        to approximately -100 to +100.

    Used by
        Introduced by William Blau (1991). Used by systematic traders and
        quants who want a smoother, less noisy version of RSI. Favored in
        equity futures and ETF momentum models where noise reduction is
        important.

    When to use
        - TSI > 0: positive momentum (bullish).
        - TSI < 0: negative momentum (bearish).
        - TSI crosses above its EMA signal line → buy.
        - TSI crosses below its EMA signal line → sell.
        - Divergence between price and TSI is a leading reversal signal.

    Code example
        >>> from momentum.momentum_tools import tsi
        >>> df["tsi"] = tsi(df["Close"])
        >>> buy = df["tsi"] > 0

    Args:
        close       (pd.Series): Closing price series.
        window_slow (int):       Slow EMA period (default 25).
        window_fast (int):       Fast EMA period (default 13).

    Returns:
        pd.Series: TSI values.

    Reference: https://en.wikipedia.org/wiki/True_strength_index
    """
    diff = close.diff(1)
    smoothed = diff.ewm(span=window_slow, adjust=False).mean().ewm(
        span=window_fast, adjust=False
    ).mean()
    smoothed_abs = diff.abs().ewm(span=window_slow, adjust=False).mean().ewm(
        span=window_fast, adjust=False
    ).mean()
    tsi_vals = 100 * smoothed / smoothed_abs.replace(0, np.nan)
    return pd.Series(tsi_vals, index=close.index, name="tsi")


def cci(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 20,
    constant: float = 0.015,
) -> pd.Series:
    """
    Commodity Channel Index (CCI)
    ------------------------------
    Metric
        Measures the deviation of typical price (H+L+C)/3 from its simple
        moving average, normalized by the mean absolute deviation.
        Originally designed for commodities but broadly applied.
            CCI = (Typical Price - SMA) / (constant * Mean Absolute Dev)

    Used by
        Developed by Donald Lambert (1980). Used by commodity and equity
        traders to identify cyclical turns. Quant systematic funds use CCI
        as a momentum signal in multi-factor models. Popular in the
        "trend trading" community.

    When to use
        - CCI > +100: strong upward momentum, asset above its average —
          breakout buy signal in a trending environment.
        - CCI < -100: strong downward momentum — potential short or
          exit long.
        - CCI returning back inside ±100 from an extreme = reversal signal.
        - Use 20-period CCI for daily charts, 14-period for intraday.

    Code example
        >>> from momentum.momentum_tools import cci
        >>> df["cci"] = cci(df["High"], df["Low"], df["Close"])
        >>> buy = (df["cci"] > 100) & (df["cci"].shift(1) < 100)

    Args:
        high     (pd.Series): High price series.
        low      (pd.Series): Low price series.
        close    (pd.Series): Closing price series.
        window   (int):       Rolling window (default 20).
        constant (float):     Scaling constant (default 0.015).

    Returns:
        pd.Series: CCI values.

    Reference: http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:commodity_channel_index_cci
    """
    typical = (high + low + close) / 3
    sma_tp = typical.rolling(window=window).mean()
    mad = typical.rolling(window=window).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    cci_vals = (typical - sma_tp) / (constant * mad.replace(0, np.nan))
    return pd.Series(cci_vals, index=close.index, name="cci")


def ultimate_oscillator(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window1: int = 7,
    window2: int = 14,
    window3: int = 28,
) -> pd.Series:
    """
    Ultimate Oscillator (UO)
    ------------------------
    Metric
        Larry Williams' (1976) oscillator capturing momentum across three
        timeframes simultaneously, reducing false signals from single-period
        approaches.
            BP = Close - min(Low, Prior Close)
            TR = max(High, Prior Close) - min(Low, Prior Close)
            UO = 100 * (4*Avg7 + 2*Avg14 + Avg28) / 7

    Used by
        Developed by Larry Williams. Used by systematic traders who want
        a multi-timeframe view of buying pressure. Popular in multi-factor
        momentum models and as a confirmation filter for RSI divergence
        signals.

    When to use
        - UO above 70: overbought — watch for sell/reversal.
        - UO below 30: oversold — watch for buy opportunity.
        - Williams' original strategy: buy when UO < 30 and a bullish
          divergence forms, then sell when UO rises above 70.
        - Useful when RSI and Stochastic give conflicting signals; UO
          acts as a tie-breaker.

    Code example
        >>> from momentum.momentum_tools import ultimate_oscillator
        >>> df["uo"] = ultimate_oscillator(df["High"], df["Low"], df["Close"])
        >>> buy = df["uo"] < 30

    Args:
        high    (pd.Series): High price series.
        low     (pd.Series): Low price series.
        close   (pd.Series): Closing price series.
        window1 (int):       Short period (default 7).
        window2 (int):       Medium period (default 14).
        window3 (int):       Long period (default 28).

    Returns:
        pd.Series: Ultimate Oscillator values (0–100).

    Reference: http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:ultimate_oscillator
    """
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high, prev_close], axis=1
    ).max(axis=1) - pd.concat([low, prev_close], axis=1).min(axis=1)
    buying_pressure = close - pd.concat([low, prev_close], axis=1).min(axis=1)

    def _avg(bp: pd.Series, tr: pd.Series, w: int) -> pd.Series:
        return bp.rolling(w).sum() / tr.rolling(w).sum().replace(0, np.nan)

    avg1 = _avg(buying_pressure, true_range, window1)
    avg2 = _avg(buying_pressure, true_range, window2)
    avg3 = _avg(buying_pressure, true_range, window3)
    uo_vals = 100 * (4 * avg1 + 2 * avg2 + avg3) / 7
    return pd.Series(uo_vals, index=close.index, name="uo")


def ppo(
    close: pd.Series,
    window_fast: int = 12,
    window_slow: int = 26,
    window_signal: int = 9,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Percentage Price Oscillator (PPO)
    ----------------------------------
    Metric
        MACD expressed as a percentage of the slower EMA, making it
        comparable across different price levels and instruments.
            PPO = 100 * (EMA_fast - EMA_slow) / EMA_slow
            Signal = EMA(PPO, signal)
            Histogram = PPO - Signal

    Used by
        Used by quant analysts comparing momentum signals across different
        securities at different price levels (e.g., comparing a $5 stock
        to a $500 stock). Common in cross-sectional factor models. The
        percentage-based output makes it easier to set universal
        thresholds.

    When to use
        - PPO > 0: fast EMA above slow EMA → bullish momentum.
        - PPO crosses Signal from below → buy signal.
        - PPO Histogram turning positive → early momentum shift signal.
        - Use PPO instead of MACD when comparing momentum across a
          portfolio of stocks with very different price levels.

    Code example
        >>> from momentum.momentum_tools import ppo
        >>> ppo_line, signal, hist = ppo(df["Close"])
        >>> buy = (ppo_line > signal) & (ppo_line.shift(1) < signal.shift(1))

    Args:
        close         (pd.Series): Closing price series.
        window_fast   (int):       Fast EMA period (default 12).
        window_slow   (int):       Slow EMA period (default 26).
        window_signal (int):       Signal EMA period (default 9).

    Returns:
        Tuple[pd.Series, pd.Series, pd.Series]: (PPO line, signal line, histogram).

    Reference: https://school.stockcharts.com/doku.php?id=technical_indicators:price_oscillators_ppo
    """
    ema_fast = ema(close, window_fast)
    ema_slow = ema(close, window_slow)
    ppo_line = 100 * (ema_fast - ema_slow) / ema_slow.replace(0, np.nan)
    ppo_signal = ema(ppo_line, window_signal)
    ppo_hist = ppo_line - ppo_signal
    return (
        pd.Series(ppo_line,   index=close.index, name="ppo"),
        pd.Series(ppo_signal, index=close.index, name="ppo_signal"),
        pd.Series(ppo_hist,   index=close.index, name="ppo_hist"),
    )


def kama(
    close: pd.Series,
    window: int = 10,
    fast_period: int = 2,
    slow_period: int = 30,
) -> pd.Series:
    """
    Kaufman's Adaptive Moving Average (KAMA)
    -----------------------------------------
    Metric
        An adaptive moving average that self-adjusts its smoothing speed
        based on market noise. It moves quickly when price trends strongly
        (high Efficiency Ratio) and slows down in choppy/sideways markets
        (low Efficiency Ratio).

    Used by
        Developed by Perry Kaufman (1995). Used by systematic CTA funds
        and quantitative momentum traders who want a trend-following
        indicator that automatically adjusts to different market regimes.
        Avoids many false signals during consolidation periods.

    When to use
        - Price crosses above KAMA → momentum building, potential buy.
        - Price crosses below KAMA → momentum deteriorating, exit long.
        - KAMA slope flattening = market is ranging; avoid new entries.
        - KAMA is steepening = trending market; add to winners.
        - Particularly useful for breakout strategies — wait for KAMA
          to confirm the breakout before entering.

    Code example
        >>> from momentum.momentum_tools import kama
        >>> df["kama"] = kama(df["Close"])
        >>> buy = df["Close"] > df["kama"]

    Args:
        close       (pd.Series): Closing price series.
        window      (int):       Efficiency Ratio window (default 10).
        fast_period (int):       Fastest EMA constant period (default 2).
        slow_period (int):       Slowest EMA constant period (default 30).

    Returns:
        pd.Series: KAMA values.

    Reference: https://www.tradingview.com/ideas/kama/
    """
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)

    close_arr = close.values.astype(float)
    noise = pd.Series(np.abs(close - close.shift(1)))
    direction = np.abs(close_arr - np.roll(close_arr, window))
    volatility = noise.rolling(window).sum().values
    with np.errstate(divide="ignore", invalid="ignore"):
        er = np.where(volatility != 0, direction / volatility, 0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

    kama_arr = np.full(len(close), np.nan)
    first = window - 1
    kama_arr[first] = close_arr[first]
    for i in range(first + 1, len(close)):
        if not np.isnan(sc[i]):
            kama_arr[i] = kama_arr[i - 1] + sc[i] * (close_arr[i] - kama_arr[i - 1])
        else:
            kama_arr[i] = kama_arr[i - 1]
    return pd.Series(kama_arr, index=close.index, name="kama")


# ============================================================
# 3. TREND STRENGTH INDICATORS
# ============================================================

def macd(
    close: pd.Series,
    window_fast: int = 12,
    window_slow: int = 26,
    window_signal: int = 9,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    MACD — Moving Average Convergence / Divergence
    -----------------------------------------------
    Metric
        Trend-following momentum indicator showing the relationship between
        two EMAs of price. Three components:
            MACD line  = EMA(12) - EMA(26)
            Signal line = EMA(MACD, 9)
            Histogram  = MACD - Signal

    Used by
        One of the most universally followed indicators. Retail traders
        use crossovers for entry/exit signals; institutional traders use
        the histogram for momentum strength; quant funds incorporate MACD
        in momentum factor models. Applies to equities, forex, crypto,
        and futures.

    When to use
        - MACD line crosses above Signal → bullish crossover, buy signal.
        - MACD line crosses below Signal → bearish crossover, sell signal.
        - MACD Histogram turns positive (from negative) → early buy signal.
        - MACD above zero line = underlying trend is bullish; below = bearish.
        - Divergence: price makes new high but MACD does not → weakening
          momentum, consider reducing position.
        - Combine with ADX > 25 to filter crossover signals in trending
          markets only.

    Code example
        >>> from momentum.momentum_tools import macd
        >>> macd_line, signal, hist = macd(df["Close"])
        >>> buy  = (macd_line > signal) & (macd_line.shift(1) <= signal.shift(1))
        >>> sell = (macd_line < signal) & (macd_line.shift(1) >= signal.shift(1))

    Args:
        close         (pd.Series): Closing price series.
        window_fast   (int):       Fast EMA period (default 12).
        window_slow   (int):       Slow EMA period (default 26).
        window_signal (int):       Signal EMA period (default 9).

    Returns:
        Tuple[pd.Series, pd.Series, pd.Series]:
            (macd_line, signal_line, histogram).

    Reference: https://en.wikipedia.org/wiki/MACD
    """
    ema_fast = ema(close, window_fast)
    ema_slow = ema(close, window_slow)
    macd_line = ema_fast - ema_slow
    sig_line = ema(macd_line, window_signal)
    histogram = macd_line - sig_line
    return (
        pd.Series(macd_line, index=close.index, name="macd"),
        pd.Series(sig_line,  index=close.index, name="macd_signal"),
        pd.Series(histogram, index=close.index, name="macd_hist"),
    )


def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Average Directional Index (ADX) with +DI and -DI
    -------------------------------------------------
    Metric
        Measures trend strength (not direction) on a 0–100 scale.
        Derived from the Directional Movement system developed by
        J. Welles Wilder (1978):
            +DI: smoothed positive directional movement
            -DI: smoothed negative directional movement
            ADX: smoothed average of the DX ratio

    Used by
        Widely used by professional trend-following CTAs to assess whether
        a market is trending or ranging. Quant systematic funds use ADX as
        a regime filter — only running momentum signals when ADX is high.
        Essential in any momentum system to avoid trading in choppy markets.

    When to use
        - ADX > 25: market is trending — momentum signals are reliable.
        - ADX < 20: market is ranging — momentum signals produce many
          false positives; consider pausing trend strategies.
        - ADX > 40: extremely strong trend — consider trailing stops but
          do not fade the trend.
        - +DI crosses above -DI with ADX > 25 → strong bullish entry.
        - -DI crosses above +DI with ADX > 25 → strong bearish entry.

    Code example
        >>> from momentum.momentum_tools import adx
        >>> adx_vals, pdi, mdi = adx(df["High"], df["Low"], df["Close"])
        >>> trending = adx_vals > 25
        >>> bull_trend = trending & (pdi > mdi)

    Args:
        high   (pd.Series): High price series.
        low    (pd.Series): Low price series.
        close  (pd.Series): Closing price series.
        window (int):       Smoothing period (default 14).

    Returns:
        Tuple[pd.Series, pd.Series, pd.Series]: (ADX, +DI, -DI).

    Reference: http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:average_directional_index_adx
    """
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low,
         (high - prev_close).abs(),
         (low - prev_close).abs()], axis=1
    ).max(axis=1)

    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    atr = pd.Series(plus_dm, index=close.index).ewm(alpha=1 / window, adjust=False).mean()
    pdi_s = pd.Series(plus_dm, index=close.index).ewm(alpha=1 / window, adjust=False).mean()
    mdi_s = pd.Series(minus_dm, index=close.index).ewm(alpha=1 / window, adjust=False).mean()
    tr_s = tr.ewm(alpha=1 / window, adjust=False).mean()

    pdi_vals = 100 * pdi_s / tr_s.replace(0, np.nan)
    mdi_vals = 100 * mdi_s / tr_s.replace(0, np.nan)
    dx = 100 * (pdi_vals - mdi_vals).abs() / (pdi_vals + mdi_vals).replace(0, np.nan)
    adx_vals = dx.ewm(alpha=1 / window, adjust=False).mean()

    return (
        pd.Series(adx_vals, index=close.index, name="adx"),
        pd.Series(pdi_vals, index=close.index, name="adx_pos"),
        pd.Series(mdi_vals, index=close.index, name="adx_neg"),
    )


def aroon(
    high: pd.Series,
    low: pd.Series,
    window: int = 25,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Aroon Indicator (Up, Down, Oscillator)
    ----------------------------------------
    Metric
        Identifies when trends are likely to change direction by measuring
        the number of periods since the last highest high and lowest low:
            Aroon Up   = 100 * (n - periods since n-period high) / n
            Aroon Down = 100 * (n - periods since n-period low) / n
            Oscillator = Aroon Up - Aroon Down

    Used by
        Developed by Tushar Chande (1995). Used by trend traders to
        identify emerging and weakening trends earlier than ADX. Popular
        in systematic equity momentum models. Particularly useful for
        identifying when a consolidation is about to turn into a trend.

    When to use
        - Aroon Up near 100 and Aroon Down near 0 → strong uptrend,
          buy and hold.
        - Aroon Down near 100 and Aroon Up near 0 → strong downtrend.
        - Aroon Up crosses above Aroon Down → new bullish trend beginning.
        - Oscillator > 50: bullish; Oscillator < -50: bearish.
        - Use as an early entry signal before ADX confirms the trend.

    Code example
        >>> from momentum.momentum_tools import aroon
        >>> aroon_up, aroon_down, aroon_osc = aroon(df["High"], df["Low"])
        >>> emerging_bull = (aroon_up > 70) & (aroon_down < 30)

    Args:
        high   (pd.Series): High price series.
        low    (pd.Series): Low price series.
        window (int):       Lookback period (default 25).

    Returns:
        Tuple[pd.Series, pd.Series, pd.Series]:
            (Aroon Up, Aroon Down, Oscillator), all as pd.Series.

    Reference: https://www.investopedia.com/terms/a/aroon.asp
    """
    aroon_up = high.rolling(window + 1).apply(
        lambda x: float(np.argmax(x)) / window * 100, raw=True
    )
    aroon_down = low.rolling(window + 1).apply(
        lambda x: float(np.argmin(x)) / window * 100, raw=True
    )
    aroon_osc = aroon_up - aroon_down
    return (
        pd.Series(aroon_up,   index=high.index, name="aroon_up"),
        pd.Series(aroon_down, index=high.index, name="aroon_down"),
        pd.Series(aroon_osc,  index=high.index, name="aroon_osc"),
    )


def parabolic_sar(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    step: float = 0.02,
    max_step: float = 0.20,
) -> pd.Series:
    """
    Parabolic SAR (Stop and Reverse)
    ---------------------------------
    Metric
        J. Welles Wilder's trailing stop indicator that places dots below
        rising prices (uptrend) and above falling prices (downtrend). The
        SAR accelerates as the trend continues, tightening the trailing stop.

    Used by
        Widely used by professional trend traders as a dynamic trailing
        stop-loss and trend direction indicator. Common in systematic
        futures and equity trading. Used by many IBKR algo traders to
        manage open positions.

    When to use
        - Price crosses above SAR → trend reverses to uptrend, buy.
        - Price crosses below SAR → trend reverses to downtrend, sell.
        - Use SAR as a trailing stop level for open positions.
        - Combine with ADX: only act on SAR reversals when ADX > 25.
        - In choppy markets (ADX < 20), SAR produces many whipsaws —
          avoid or widen step parameter.
        - Typical step = 0.02 (accelerates 0.02 per new extreme);
          max_step = 0.20. Increase step for faster, tighter stops.

    Code example
        >>> from momentum.momentum_tools import parabolic_sar
        >>> df["psar"] = parabolic_sar(df["High"], df["Low"], df["Close"])
        >>> buy  = df["Close"] > df["psar"]
        >>> sell = df["Close"] < df["psar"]

    Args:
        high     (pd.Series): High price series.
        low      (pd.Series): Low price series.
        close    (pd.Series): Closing price series.
        step     (float):     Acceleration factor increment (default 0.02).
        max_step (float):     Maximum acceleration factor (default 0.20).

    Returns:
        pd.Series: Parabolic SAR values.

    Reference: https://school.stockcharts.com/doku.php?id=technical_indicators:parabolic_sar
    """
    high_arr = high.values.astype(float)
    low_arr = low.values.astype(float)
    n = len(high_arr)
    psar_arr = np.zeros(n)
    bull = True
    af = step
    ep = high_arr[0]
    psar_arr[0] = low_arr[0]

    for i in range(1, n):
        prev_psar = psar_arr[i - 1]
        if bull:
            psar_arr[i] = prev_psar + af * (ep - prev_psar)
            psar_arr[i] = min(psar_arr[i], low_arr[i - 1])
            if i > 1:
                psar_arr[i] = min(psar_arr[i], low_arr[i - 2])
            if low_arr[i] < psar_arr[i]:
                bull = False
                psar_arr[i] = ep
                ep = low_arr[i]
                af = step
            else:
                if high_arr[i] > ep:
                    ep = high_arr[i]
                    af = min(af + step, max_step)
        else:
            psar_arr[i] = prev_psar - af * (prev_psar - ep)
            psar_arr[i] = max(psar_arr[i], high_arr[i - 1])
            if i > 1:
                psar_arr[i] = max(psar_arr[i], high_arr[i - 2])
            if high_arr[i] > psar_arr[i]:
                bull = True
                psar_arr[i] = ep
                ep = high_arr[i]
                af = step
            else:
                if low_arr[i] < ep:
                    ep = low_arr[i]
                    af = min(af + step, max_step)
    return pd.Series(psar_arr, index=close.index, name="psar")


def kst(
    close: pd.Series,
    roc1: int = 10, roc2: int = 15, roc3: int = 20, roc4: int = 30,
    w1: int = 10,   w2: int = 10,   w3: int = 10,   w4: int = 15,
    nsig: int = 9,
) -> Tuple[pd.Series, pd.Series]:
    """
    Know Sure Thing (KST) Oscillator
    ----------------------------------
    Metric
        Martin Pring's multi-timeframe momentum indicator that combines
        four different Rate of Change (ROC) measures, each smoothed with
        an SMA and weighted to emphasize longer cycles:
            KST = 1*RCMA1 + 2*RCMA2 + 3*RCMA3 + 4*RCMA4

    Used by
        Developed by Martin Pring. Used by cycle analysts and macro
        momentum traders who want to capture long-term momentum cycles.
        Common in ETF rotation strategies and sector momentum models.
        Popular among practitioners of intermarket analysis.

    When to use
        - KST crosses above its signal line → bullish; buy signal.
        - KST crosses below its signal line → bearish; sell signal.
        - Best on weekly or monthly data for intermediate-term momentum.
        - Use to identify major stock market cycle turning points.
        - Daily KST is noisier; best for swing trading setups.

    Code example
        >>> from momentum.momentum_tools import kst
        >>> kst_line, kst_signal = kst(df["Close"])
        >>> buy = (kst_line > kst_signal) & (kst_line.shift(1) <= kst_signal.shift(1))

    Args:
        close  (pd.Series): Closing price series.
        roc1-4 (int):       ROC periods for four components.
        w1-w4  (int):       SMA smoothing periods for each ROC.
        nsig   (int):       Signal line SMA period (default 9).

    Returns:
        Tuple[pd.Series, pd.Series]: (KST line, Signal line).

    Reference: https://en.wikipedia.org/wiki/KST_oscillator
    """
    def _rcma(c: pd.Series, r: int, w: int) -> pd.Series:
        roc_v = roc(c, window=r)
        return roc_v.rolling(w).mean()

    kst_line = (
        1 * _rcma(close, roc1, w1) +
        2 * _rcma(close, roc2, w2) +
        3 * _rcma(close, roc3, w3) +
        4 * _rcma(close, roc4, w4)
    )
    signal_line = kst_line.rolling(nsig).mean()
    return (
        pd.Series(kst_line,    index=close.index, name="kst"),
        pd.Series(signal_line, index=close.index, name="kst_signal"),
    )


# ============================================================
# 4. RISK & PERFORMANCE METRICS
# ============================================================

def volatility(returns: pd.Series, periods: int = 252) -> float:
    """
    Annualized Volatility (Standard Deviation of Returns)
    ------------------------------------------------------
    Metric
        Annualized standard deviation of daily/periodic returns.
        Measures the dispersion of returns around the mean.
            Vol = std(returns) * sqrt(periods)

    Used by
        A fundamental risk metric used by every type of trader and risk
        manager. Options traders use it to compare to implied volatility.
        Portfolio managers use it for position sizing (e.g., risk parity).
        Quant funds use it in Sharpe ratio and volatility-adjusted momentum.

    When to use
        - Use to size positions: smaller position in high-volatility stocks,
          larger in low-volatility stocks (volatility targeting).
        - Compare against VIX or implied vol to see if a stock is cheap
          or expensive on a vol-adjusted basis.
        - Use as a regime indicator: rising vol often precedes corrections.

    Code example
        >>> from momentum.momentum_tools import volatility
        >>> returns = df["Close"].pct_change().dropna()
        >>> annual_vol = volatility(returns)
        >>> print(f"Annual vol: {annual_vol:.1%}")

    Args:
        returns (pd.Series): Daily return series (decimal, e.g. 0.01 = 1%).
        periods (int):       Trading periods per year (252 = daily).

    Returns:
        float: Annualized volatility.
    """
    return float(returns.std() * np.sqrt(periods))


def sharpe_ratio(
    returns: pd.Series, risk_free_rate: float = 0.0, periods: int = 252
) -> float:
    """
    Sharpe Ratio
    ------------
    Metric
        Measures risk-adjusted return by dividing excess return (return
        above the risk-free rate) by the standard deviation of returns.
            Sharpe = (mean_return - rf_per_period) / std(returns) * sqrt(periods)

    Used by
        The most widely used performance metric in finance. Portfolio
        managers, hedge funds, and retail traders use it to compare
        strategies on a risk-adjusted basis. Required reporting metric for
        most institutional funds. IBKR TWS risk reports include Sharpe.

    When to use
        - Use to compare two strategies: the one with the higher Sharpe
          is more efficient per unit of risk.
        - Sharpe > 1.0 is generally considered acceptable.
        - Sharpe > 2.0 is excellent for a systematic strategy.
        - Sharpe > 3.0 is exceptional (likely on in-sample data — beware).
        - Run this on your backtest to evaluate strategy quality before
          going live with real capital.

    Code example
        >>> from momentum.momentum_tools import sharpe_ratio
        >>> returns = df["Close"].pct_change().dropna()
        >>> sharpe = sharpe_ratio(returns, risk_free_rate=0.05)
        >>> print(f"Sharpe: {sharpe:.2f}")

    Args:
        returns        (pd.Series): Daily return series.
        risk_free_rate (float):     Annual risk-free rate (e.g. 0.05 = 5%).
        periods        (int):       Trading periods per year (default 252).

    Returns:
        float: Annualized Sharpe ratio.

    Reference: https://www.investopedia.com/terms/s/sharperatio.asp
    """
    rf_per_period = (1 + risk_free_rate) ** (1 / periods) - 1
    excess = returns - rf_per_period
    std = returns.std(ddof=1)
    if std == 0:
        return 0.0
    return float((excess.mean() / std) * np.sqrt(periods))


def sortino_ratio(
    returns: pd.Series, risk_free_rate: float = 0.0, periods: int = 252
) -> float:
    """
    Sortino Ratio
    -------------
    Metric
        Like the Sharpe ratio but uses downside deviation (only negative
        return volatility) instead of total volatility. This avoids
        penalising upside volatility, making it a more appropriate
        risk-adjusted metric for asymmetric return profiles.

    Used by
        Preferred over Sharpe by momentum traders whose strategies exhibit
        positive skewness (large wins, small losses). Used by CTAs,
        systematic equity funds, and options traders. Most backtesting
        platforms (including QuantConnect) report this by default.

    When to use
        - Use when your strategy has asymmetric returns (momentum systems
          often have large winners but frequent small losers).
        - A Sortino > 2.0 is excellent for a momentum strategy.
        - Compare Sortino vs Sharpe: if Sortino >> Sharpe, your strategy
          has positive skewness (which is desirable).
        - Run alongside Sharpe to get a complete picture of the
          risk/reward profile of your algorithm.

    Code example
        >>> from momentum.momentum_tools import sortino_ratio
        >>> returns = df["Close"].pct_change().dropna()
        >>> sortino = sortino_ratio(returns)
        >>> print(f"Sortino: {sortino:.2f}")

    Args:
        returns        (pd.Series): Daily return series.
        risk_free_rate (float):     Annual risk-free rate.
        periods        (int):       Trading periods per year.

    Returns:
        float: Annualized Sortino ratio.

    Reference: http://www.redrockcapital.com/Sortino__A__Sharper__Ratio_Red_Rock_Capital.pdf
    """
    rf_per_period = (1 + risk_free_rate) ** (1 / periods) - 1
    excess = returns - rf_per_period
    downside = np.sqrt((excess[excess < 0] ** 2).sum() / len(excess))
    if downside == 0:
        return 0.0
    return float((excess.mean() / downside) * np.sqrt(periods))


def max_drawdown(returns: pd.Series) -> float:
    """
    Maximum Drawdown (MDD)
    ----------------------
    Metric
        The largest peak-to-trough decline in the cumulative return series,
        expressed as a percentage. Represents the worst-case loss a
        strategy inflicted during the backtest period.
            MDD = min[(equity - rolling_max_equity) / rolling_max_equity]

    Used by
        Used by every serious trader and risk manager to evaluate downside
        risk. Required metric for hedge fund due diligence. Many traders
        use MDD as a circuit-breaker: if live drawdown exceeds backtest MDD,
        stop the strategy until it is reviewed.

    When to use
        - A key metric to decide whether you can psychologically and
          financially sustain a strategy's losses.
        - Compare MDD to expected annual return: if MDD > annual return,
          the strategy may not recover within a year.
        - Use MDD to set automatic risk-off rules: "If daily drawdown
          exceeds 5%, halt trading for the day."
        - Track live MDD vs backtest MDD as a regime-change signal.

    Code example
        >>> from momentum.momentum_tools import max_drawdown
        >>> returns = df["Close"].pct_change().dropna()
        >>> mdd = max_drawdown(returns)
        >>> print(f"Max Drawdown: {mdd:.1%}")

    Args:
        returns (pd.Series): Daily return series.

    Returns:
        float: Maximum drawdown as a negative decimal (e.g. -0.25 = -25%).
    """
    cum_returns = (1 + returns).cumprod()
    rolling_max = cum_returns.cummax()
    drawdown = (cum_returns - rolling_max) / rolling_max
    return float(drawdown.min())


def calmar_ratio(returns: pd.Series, periods: int = 252) -> float:
    """
    Calmar Ratio
    ------------
    Metric
        Ratio of CAGR (compound annual growth rate) to the absolute value
        of the maximum drawdown. Measures how much annual return you earn
        per unit of worst-case loss.

    Used by
        Widely used by CTAs and hedge funds for evaluating trend-following
        strategies. Better than Sharpe for strategies where volatility is
        asymmetric (e.g., trend-following systems have occasional large
        drawdowns). Industry standard for CTA performance evaluation.

    When to use
        - Use to compare strategies with similar CAGR but different risk.
        - Calmar > 1.0 means you earn more annually than your worst loss.
        - Calmar > 3.0 is considered excellent for a momentum strategy.
        - Use alongside MDD and Sharpe to get a full risk picture.

    Code example
        >>> from momentum.momentum_tools import calmar_ratio
        >>> returns = df["Close"].pct_change().dropna()
        >>> calmar = calmar_ratio(returns)
        >>> print(f"Calmar: {calmar:.2f}")

    Args:
        returns (pd.Series): Daily return series.
        periods (int):       Trading periods per year (default 252).

    Returns:
        float: Calmar ratio (positive).
    """
    n_years = len(returns) / periods
    if n_years == 0:
        return 0.0
    total = (1 + returns).prod()
    annual = total ** (1 / n_years) - 1
    mdd = abs(max_drawdown(returns))
    if mdd == 0:
        return 0.0
    return float(annual / mdd)


def cagr(returns: pd.Series, periods: int = 252) -> float:
    """
    Compound Annual Growth Rate (CAGR)
    -----------------------------------
    Metric
        The geometric mean annual return, representing the steady rate
        of return that would produce the same total return over the period.
            CAGR = (total_return + 1)^(periods/n) - 1

    Used by
        The primary return metric used by investors to compare strategies
        of different lengths. Reported by every fund, backtester, and
        broker (including IBKR). Essential for comparing strategies with
        different time horizons.

    When to use
        - Always report CAGR alongside MDD and Sharpe in your backtest
          summary. CAGR alone can be misleading without MDD context.
        - Compare CAGR against benchmark (e.g., SPY CAGR) to see if
          strategy is alpha-generating.
        - Use CAGR to set realistic daily P&L expectations.

    Code example
        >>> from momentum.momentum_tools import cagr
        >>> returns = df["Close"].pct_change().dropna()
        >>> annual_return = cagr(returns)
        >>> print(f"CAGR: {annual_return:.1%}")

    Args:
        returns (pd.Series): Daily return series.
        periods (int):       Trading periods per year (default 252).

    Returns:
        float: CAGR as a decimal (e.g. 0.15 = 15%).
    """
    n_years = len(returns) / periods
    if n_years == 0:
        return 0.0
    total = float((1 + returns).prod())
    return float(abs(total) ** (1 / n_years) - 1)


def win_rate(returns: pd.Series) -> float:
    """
    Win Rate
    --------
    Metric
        The fraction of periods (or trades) with a positive return.
            Win Rate = count(returns > 0) / count(returns != 0)

    Used by
        Used by discretionary and systematic traders to characterize
        strategy behavior. High win rate strategies feel psychologically
        comfortable but can still have poor risk/reward. Important for
        understanding the psychological demands of a strategy.

    When to use
        - Use alongside average win / average loss to assess overall
          expectancy.
        - Momentum strategies typically have win rates of 40–60%.
          Lower win rates are fine if wins are much larger than losses.
        - If win rate drops significantly in live trading vs backtest,
          it may signal overfitting or regime change.

    Code example
        >>> from momentum.momentum_tools import win_rate
        >>> returns = df["Close"].pct_change().dropna()
        >>> wr = win_rate(returns)
        >>> print(f"Win rate: {wr:.1%}")

    Args:
        returns (pd.Series): Return series (daily, trade-level, etc.).

    Returns:
        float: Win rate (0–1 scale).
    """
    non_zero = returns[returns != 0].dropna()
    if len(non_zero) == 0:
        return 0.0
    return float((non_zero > 0).sum() / len(non_zero))


def kelly_criterion(returns: pd.Series) -> float:
    """
    Kelly Criterion (Full Kelly)
    ----------------------------
    Metric
        The theoretically optimal fraction of capital to wager on each
        trade to maximise long-term geometric growth:
            Kelly = (W * R - L) / R
        where W = win probability, L = loss probability,
        R = avg win / avg loss.

    Used by
        Introduced by John Kelly (1956). Used by quantitative traders,
        card counters, and sports bettors for position sizing. Ed Thorp
        popularised it in financial markets. Many practitioners use
        "Half Kelly" or "Quarter Kelly" to reduce ruin risk.

    When to use
        - Use the output as the *maximum* fraction of your account to
          risk per trade. Never exceed full Kelly — it maximises ruin.
        - In practice, use 25–50% of the Kelly fraction for more
          conservative position sizing.
        - If Kelly returns a negative number, the strategy has a negative
          expected value — do not trade it.
        - Recalculate monthly as win rates and R-multiples evolve.

    Code example
        >>> from momentum.momentum_tools import kelly_criterion
        >>> returns = strategy_returns_series
        >>> kelly_f = kelly_criterion(returns)
        >>> position_size = 0.5 * kelly_f   # Half Kelly

    Args:
        returns (pd.Series): Return series.

    Returns:
        float: Kelly fraction (0–1 scale; negative = negative edge).

    Reference: http://en.wikipedia.org/wiki/Kelly_criterion
    """
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    if len(wins) == 0 or len(losses) == 0:
        return 0.0
    win_prob = len(wins) / (len(wins) + len(losses))
    loss_prob = 1 - win_prob
    avg_win = wins.mean()
    avg_loss = abs(losses.mean())
    if avg_loss == 0:
        return 0.0
    r = avg_win / avg_loss
    return float((win_prob * r - loss_prob) / r)


def value_at_risk(
    returns: pd.Series, confidence: float = 0.95
) -> float:
    """
    Value at Risk (VaR) — Parametric Method
    ----------------------------------------
    Metric
        Estimates the maximum expected daily loss at a given confidence
        level, assuming normally distributed returns (parametric/Gaussian
        method). E.g., 95% VaR = worst expected daily loss 95% of the time.

    Used by
        Basel III regulatory requirement for banks. Used by risk managers
        to set daily loss limits and margin requirements. Every major
        brokerage (including IBKR) calculates VaR for portfolio risk.
        Systematic traders use it to set position limits.

    When to use
        - Use to set daily stop-loss levels: if your 95% VaR is -2%,
          consider exiting positions if the portfolio drops >2% intraday.
        - Compare live daily losses against VaR as an alert system.
        - Increase confidence to 99% for more conservative risk limits.
        - VaR alone doesn't capture tail risk — always pair with CVaR.

    Code example
        >>> from momentum.momentum_tools import value_at_risk
        >>> returns = df["Close"].pct_change().dropna()
        >>> var_95 = value_at_risk(returns, confidence=0.95)
        >>> print(f"95% VaR: {var_95:.2%}")

    Args:
        returns    (pd.Series): Daily return series.
        confidence (float):     Confidence level (default 0.95 = 95%).

    Returns:
        float: VaR as a negative decimal (e.g. -0.02 = -2% max daily loss).
    """
    from scipy.stats import norm
    mu = returns.mean()
    sigma = returns.std()
    return float(norm.ppf(1 - confidence, mu, sigma))


def conditional_value_at_risk(
    returns: pd.Series, confidence: float = 0.95
) -> float:
    """
    Conditional Value at Risk (CVaR) / Expected Shortfall
    -------------------------------------------------------
    Metric
        The expected return *given* that the loss exceeds the VaR threshold.
        Also called Expected Shortfall (ES). CVaR captures tail risk that
        VaR misses by averaging losses beyond the VaR cutoff.

    Used by
        Preferred over VaR by academic researchers and sophisticated risk
        managers because it is coherent (VaR is not). Required by FRTB
        (Fundamental Review of the Trading Book) regulations. Used by
        quant funds to size positions in tail-risky instruments.

    When to use
        - Use alongside VaR: VaR tells you the threshold; CVaR tells you
          how bad it gets beyond that threshold.
        - High CVaR relative to VaR (large CVaR/VaR ratio) signals fat
          tails — consider reducing leverage.
        - Use 95% CVaR as your "expected loss in a bad market day"
          for realistic drawdown scenario planning.

    Code example
        >>> from momentum.momentum_tools import conditional_value_at_risk
        >>> returns = df["Close"].pct_change().dropna()
        >>> cvar_95 = conditional_value_at_risk(returns, confidence=0.95)
        >>> print(f"95% CVaR: {cvar_95:.2%}")

    Args:
        returns    (pd.Series): Daily return series.
        confidence (float):     Confidence level (default 0.95).

    Returns:
        float: CVaR as a negative decimal.
    """
    var = value_at_risk(returns, confidence)
    tail = returns[returns < var]
    if len(tail) == 0:
        return var
    return float(tail.mean())


# ============================================================
# 5. CROSS-SECTIONAL & PORTFOLIO MOMENTUM
# ============================================================

def momentum_score(
    close: pd.Series,
    lookback: int = 252,
    skip: int = 21,
) -> float:
    """
    12-1 Momentum Score (Cross-Sectional Momentum Factor)
    ------------------------------------------------------
    Metric
        The classic Jegadeesh & Titman (1993) momentum factor: total return
        over the past `lookback` periods, excluding the most recent `skip`
        periods (to avoid short-term reversal contamination).
            Score = (Close[-skip] - Close[-lookback]) / Close[-lookback]

    Used by
        The cornerstone of quantitative equity momentum investing. Used by
        factor-based ETFs (e.g., MTUM), hedge funds (AQR, Cliff Asness),
        and academic researchers. Also used by systematic traders to rank
        a universe of stocks for long/short selection.

    When to use
        - Rank a universe of stocks by momentum_score and long the top
          decile, short the bottom decile (standard Jegadeesh-Titman).
        - Use 252-day lookback (1 year) with 21-day skip for daily data.
        - Rebalance monthly for a long-only momentum portfolio.
        - Use in combination with trend filter (e.g., price > 200-day SMA)
          to avoid holding stocks in a downtrend.

    Code example
        >>> from momentum.momentum_tools import momentum_score
        >>> scores = {
        ...     ticker: momentum_score(price_series)
        ...     for ticker, price_series in price_dict.items()
        ... }
        >>> top_picks = sorted(scores, key=scores.get, reverse=True)[:10]

    Args:
        close    (pd.Series): Closing price series.
        lookback (int):       Total lookback window in periods (default 252).
        skip     (int):       Periods to skip at the recent end (default 21).

    Returns:
        float: Momentum score as a decimal return (NaN if insufficient data).

    Reference: Jegadeesh & Titman, "Returns to Buying Winners and Selling
               Losers", Journal of Finance, 1993.
    """
    if len(close) < lookback:
        return float("nan")
    end_price = close.iloc[-(skip + 1)]
    start_price = close.iloc[-lookback]
    if start_price == 0:
        return float("nan")
    return float((end_price - start_price) / start_price)


def rank_momentum(
    prices_dict: dict,
    lookback: int = 252,
    skip: int = 21,
    top_n: int = 10,
) -> list:
    """
    Cross-Sectional Momentum Ranking
    ----------------------------------
    Metric
        Ranks a universe of assets by their 12-1 momentum score and
        returns the top N assets. The foundation of most cross-sectional
        momentum (relative strength) strategies.

    Used by
        Used by quant ETF managers, factor investors, and systematic hedge
        funds. Also used by retail momentum traders to select the strongest
        stocks from a watchlist or sector.

    When to use
        - Use monthly to rebalance a momentum portfolio of stocks.
        - Feed a watchlist of 50–500 stocks to identify the top 10%
          for long entries.
        - Combine with quality/fundamental filters to reduce the number
          of value traps in the long portfolio.
        - Use as a stock selection engine before applying your
          technical entry criteria (RSI, MACD, etc.).

    Code example
        >>> from momentum.momentum_tools import rank_momentum
        >>> # prices_dict = {"AAPL": pd.Series(...), "MSFT": pd.Series(...), ...}
        >>> top_stocks = rank_momentum(prices_dict, top_n=10)
        >>> print("Top momentum stocks:", top_stocks)

    Args:
        prices_dict (dict):   {ticker: pd.Series of closing prices}.
        lookback    (int):    Momentum lookback in periods (default 252).
        skip        (int):    Skip window in periods (default 21).
        top_n       (int):    Number of top assets to return (default 10).

    Returns:
        list: Tickers sorted by descending momentum score (top N).
    """
    scores = {}
    for ticker, price_series in prices_dict.items():
        score = momentum_score(price_series, lookback=lookback, skip=skip)
        if not np.isnan(score):
            scores[ticker] = score
    sorted_tickers = sorted(scores, key=scores.get, reverse=True)
    return sorted_tickers[:top_n]


def dual_momentum(
    asset_returns: pd.Series,
    benchmark_returns: pd.Series,
    lookback: int = 252,
) -> str:
    """
    Gary Antonacci's Dual Momentum Signal
    ----------------------------------------
    Metric
        Combines absolute momentum (time-series momentum: is the asset
        beating cash?) with relative momentum (is the asset beating a
        benchmark?). Both conditions must be satisfied to hold the asset.

    Used by
        Developed by Gary Antonacci ("Dual Momentum Investing", 2014).
        Used by individual investors and systematic fund managers for
        simple but highly effective asset allocation. One of the
        best-documented momentum strategies with decades of live track
        record.

    When to use
        - Use monthly on a small set of assets (e.g., SPY vs AGG vs SHY).
        - "BUY": asset has positive absolute momentum AND beats benchmark.
        - "BENCHMARK": asset beats benchmark but has negative absolute
          momentum — hold benchmark.
        - "CASH": neither condition met — move to cash/T-bills.
        - Best for long-term systematic investors managing a few ETFs
          rather than individual stock traders.

    Code example
        >>> from momentum.momentum_tools import dual_momentum
        >>> spy_ret  = spy_prices.pct_change().dropna()
        >>> agg_ret  = agg_prices.pct_change().dropna()
        >>> signal = dual_momentum(spy_ret, agg_ret)
        >>> print(f"Dual Momentum Signal: {signal}")

    Args:
        asset_returns     (pd.Series): Return series of the primary asset.
        benchmark_returns (pd.Series): Return series of the benchmark.
        lookback          (int):       Lookback in periods (default 252).

    Returns:
        str: "BUY", "BENCHMARK", or "CASH".

    Reference: Gary Antonacci, "Dual Momentum Investing", McGraw-Hill, 2014.
               https://www.optimalmomentum.com/
    """
    if len(asset_returns) < lookback or len(benchmark_returns) < lookback:
        return "CASH"
    asset_ret = float((1 + asset_returns.iloc[-lookback:]).prod() - 1)
    bench_ret = float((1 + benchmark_returns.iloc[-lookback:]).prod() - 1)
    absolute_mom = asset_ret > 0           # beats cash (positive return)
    relative_mom = asset_ret > bench_ret   # beats benchmark
    if absolute_mom and relative_mom:
        return "BUY"
    elif relative_mom:
        return "BENCHMARK"
    else:
        return "CASH"


# ============================================================
# 6. SIGNAL GENERATION UTILITIES
# ============================================================

def crossover(series_fast: pd.Series, series_slow: pd.Series) -> pd.Series:
    """
    Crossover Signal (Fast crosses above Slow)
    -------------------------------------------
    Metric
        Returns a boolean series that is True on the exact bar where the
        fast series crosses above the slow series. Used to generate precise
        entry signals from any two indicator lines (EMA crossovers, MACD
        crossovers, RSI threshold crossings, etc.).

    Used by
        Used universally in systematic trading. The backbone of crossover-
        based entry logic (EMA crossovers, Golden Cross, MACD signal
        crossings). Simple but forms the basis of many profitable
        systematic strategies.

    When to use
        - EMA(fast) crosses EMA(slow): buy signal.
        - MACD crosses above Signal line: buy signal.
        - RSI crosses above 30 (from below): oversold exit / buy signal.
        - Stochastic %K crosses above %D: buy signal.
        - Aroon Up crosses above Aroon Down: new uptrend signal.

    Code example
        >>> from momentum.momentum_tools import crossover, ema
        >>> fast = ema(df["Close"], 12)
        >>> slow = ema(df["Close"], 26)
        >>> buy_signal = crossover(fast, slow)
        >>> entry_dates = df.index[buy_signal]

    Args:
        series_fast (pd.Series): The faster / higher series.
        series_slow (pd.Series): The slower / lower series.

    Returns:
        pd.Series (bool): True where fast crosses above slow.
    """
    above_now = series_fast > series_slow
    above_prev = series_fast.shift(1) <= series_slow.shift(1)
    return pd.Series(above_now & above_prev, index=series_fast.index, name="crossover")


def crossunder(series_fast: pd.Series, series_slow: pd.Series) -> pd.Series:
    """
    Crossunder Signal (Fast crosses below Slow)
    --------------------------------------------
    Metric
        Returns a boolean series that is True on the exact bar where the
        fast series crosses below the slow series. The mirror of crossover(),
        used to generate sell / short signals.

    Used by
        Same community as crossover(). Used for sell signals in systematic
        strategies. Death Cross (50-day SMA crosses below 200-day SMA) is
        one of the most watched crossunder signals in equity markets.

    When to use
        - EMA(fast) crosses below EMA(slow): sell / exit long signal.
        - MACD crosses below Signal line: sell / short signal.
        - RSI crosses below 70 (from above): overbought exit signal.
        - Price drops below Parabolic SAR: exit long, consider short.

    Code example
        >>> from momentum.momentum_tools import crossunder, ema
        >>> fast = ema(df["Close"], 12)
        >>> slow = ema(df["Close"], 26)
        >>> sell_signal = crossunder(fast, slow)

    Args:
        series_fast (pd.Series): The faster series.
        series_slow (pd.Series): The slower / reference series.

    Returns:
        pd.Series (bool): True where fast crosses below slow.
    """
    below_now = series_fast < series_slow
    below_prev = series_fast.shift(1) >= series_slow.shift(1)
    return pd.Series(below_now & below_prev, index=series_fast.index, name="crossunder")


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
) -> pd.Series:
    """
    Average True Range (ATR)
    ------------------------
    Metric
        Measures market volatility by averaging the True Range over
        `window` periods. True Range = max(H-L, |H-prev_C|, |L-prev_C|).
        ATR does not indicate direction — only volatility magnitude.

    Used by
        Developed by J. Welles Wilder (1978). Universally used for:
        - Stop-loss placement (e.g., 2×ATR below entry)
        - Position sizing (risk a fixed dollar amount per ATR unit)
        - Volatility filtering (avoid trading when ATR is abnormally high)
        Essential tool for any systematic momentum trader managing risk.

    When to use
        - Set stop-loss at entry_price - 2 * ATR (long) to avoid being
          stopped out by normal daily noise.
        - Size positions so that 1 ATR move equals a fixed % of capital:
            shares = (risk_per_trade_$) / ATR
        - If ATR expands significantly, reduce position size proportionally.
        - Use ATR as a trend filter: breakout entries are more reliable
          when ATR is rising (expanding volatility = trend developing).

    Code example
        >>> from momentum.momentum_tools import atr
        >>> df["atr"] = atr(df["High"], df["Low"], df["Close"])
        >>> stop_loss = df["Close"] - 2 * df["atr"]

    Args:
        high   (pd.Series): High price series.
        low    (pd.Series): Low price series.
        close  (pd.Series): Closing price series.
        window (int):       Smoothing window (default 14).

    Returns:
        pd.Series: ATR values in price units.

    Reference: https://www.investopedia.com/terms/a/atr.asp
    """
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low,
         (high - prev_close).abs(),
         (low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr_vals = tr.ewm(span=window, adjust=False).mean()
    return pd.Series(atr_vals, index=close.index, name="atr")


def zscore(series: pd.Series, window: int = 20) -> pd.Series:
    """
    Rolling Z-Score
    ---------------
    Metric
        Measures how many standard deviations the current value is from
        the rolling mean. Useful for normalising any indicator to a
        comparable scale for comparison or signal generation.
            Z = (value - rolling_mean) / rolling_std

    Used by
        Used by quant analysts for mean-reversion strategies and for
        normalising momentum signals in multi-factor models. Also used
        to detect anomalies (e.g., volume spikes, unusual price moves)
        in systematic entry filters.

    When to use
        - Z-score > 2: asset is far above its recent mean → potential
          overbought condition in a mean-reversion context.
        - Z-score < -2: asset is far below its recent mean → potential
          oversold condition or mean-reversion entry.
        - Use Z-score to normalise RSI or ROC values across a stock
          universe so all indicators are on the same scale.
        - Combine with momentum score: high momentum + low Z-score means
          the stock is in an uptrend but not currently overextended.

    Code example
        >>> from momentum.momentum_tools import zscore
        >>> df["rsi_z"] = zscore(df["rsi"], window=20)
        >>> mean_rev_entry = df["rsi_z"] < -2.0

    Args:
        series (pd.Series): Input indicator or price series.
        window (int):       Rolling window for mean and std (default 20).

    Returns:
        pd.Series: Z-score values.
    """
    roll_mean = series.rolling(window=window).mean()
    roll_std = series.rolling(window=window).std()
    z = (series - roll_mean) / roll_std.replace(0, np.nan)
    return pd.Series(z, index=series.index, name=f"zscore_{window}")
