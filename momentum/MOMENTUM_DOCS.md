# Momentum Tools — Reference Documentation

A comprehensive library of momentum trading functions for use with IBKR Trader Workstation.
All functions live in [`momentum_tools.py`](momentum_tools.py).

**Sourced from:**
- [bukosabino/ta](https://github.com/bukosabino/ta) — Technical Analysis library
- [ranaroussi/quantstats](https://github.com/ranaroussi/quantstats) — Portfolio analytics
- [twopirllc/pandas-ta](https://github.com/twopirllc/pandas-ta) — Pandas TA
- [jmrichardson/tuneta](https://github.com/jmrichardson/tuneta) — Tuned TA
- [kernc/backtesting.py](https://github.com/kernc/backtesting.py) — Backtesting framework

**Dependencies:** `numpy`, `pandas`, `scipy`

---

## Table of Contents

### 1. [Helpers & Moving Averages](#1-helpers--moving-averages)
| Function | Description |
|---|---|
| [`ema()`](#ema--exponential-moving-average) | Exponential Moving Average |
| [`sma()`](#sma--simple-moving-average) | Simple Moving Average |
| [`wma()`](#wma--weighted-moving-average) | Weighted Moving Average |

### 2. [Momentum Oscillators](#2-momentum-oscillators)
| Function | Description |
|---|---|
| [`rsi()`](#rsi--relative-strength-index) | Relative Strength Index |
| [`stochastic_oscillator()`](#stochastic_oscillator--stochastic-k-and-d) | Stochastic %K and %D |
| [`stochrsi()`](#stochrsi--stochastic-rsi) | Stochastic RSI |
| [`williams_r()`](#williams_r--williams-r) | Williams %R |
| [`roc()`](#roc--rate-of-change) | Rate of Change |
| [`awesome_oscillator()`](#awesome_oscillator--awesome-oscillator) | Awesome Oscillator |
| [`tsi()`](#tsi--true-strength-index) | True Strength Index |
| [`cci()`](#cci--commodity-channel-index) | Commodity Channel Index |
| [`ultimate_oscillator()`](#ultimate_oscillator--ultimate-oscillator) | Ultimate Oscillator |
| [`ppo()`](#ppo--percentage-price-oscillator) | Percentage Price Oscillator |
| [`kama()`](#kama--kaufmans-adaptive-moving-average) | Kaufman's Adaptive Moving Average |

### 3. [Trend Strength Indicators](#3-trend-strength-indicators)
| Function | Description |
|---|---|
| [`macd()`](#macd--moving-average-convergence--divergence) | MACD + Signal + Histogram |
| [`adx()`](#adx--average-directional-index) | Average Directional Index |
| [`aroon()`](#aroon--aroon-indicator) | Aroon Up / Down / Oscillator |
| [`parabolic_sar()`](#parabolic_sar--parabolic-sar) | Parabolic SAR |
| [`kst()`](#kst--know-sure-thing-oscillator) | Know Sure Thing Oscillator |

### 4. [Risk & Performance Metrics](#4-risk--performance-metrics)
| Function | Description |
|---|---|
| [`volatility()`](#volatility--annualized-volatility) | Annualized Volatility |
| [`sharpe_ratio()`](#sharpe_ratio--sharpe-ratio) | Sharpe Ratio |
| [`sortino_ratio()`](#sortino_ratio--sortino-ratio) | Sortino Ratio |
| [`max_drawdown()`](#max_drawdown--maximum-drawdown) | Maximum Drawdown |
| [`calmar_ratio()`](#calmar_ratio--calmar-ratio) | Calmar Ratio |
| [`cagr()`](#cagr--compound-annual-growth-rate) | Compound Annual Growth Rate |
| [`win_rate()`](#win_rate--win-rate) | Win Rate |
| [`kelly_criterion()`](#kelly_criterion--kelly-criterion) | Kelly Criterion |
| [`value_at_risk()`](#value_at_risk--value-at-risk) | Value at Risk (VaR) |
| [`conditional_value_at_risk()`](#conditional_value_at_risk--conditional-value-at-risk) | CVaR / Expected Shortfall |

### 5. [Cross-Sectional & Portfolio Momentum](#5-cross-sectional--portfolio-momentum)
| Function | Description |
|---|---|
| [`momentum_score()`](#momentum_score--12-1-momentum-score) | 12-1 Momentum Score |
| [`rank_momentum()`](#rank_momentum--cross-sectional-ranking) | Cross-Sectional Ranking |
| [`dual_momentum()`](#dual_momentum--dual-momentum-signal) | Gary Antonacci's Dual Momentum |

### 6. [Signal Generation Utilities](#6-signal-generation-utilities)
| Function | Description |
|---|---|
| [`crossover()`](#crossover--crossover-signal) | Fast crosses above Slow |
| [`crossunder()`](#crossunder--crossunder-signal) | Fast crosses below Slow |
| [`atr()`](#atr--average-true-range) | Average True Range |
| [`zscore()`](#zscore--rolling-z-score) | Rolling Z-Score |

---

## 1. Helpers & Moving Averages

### `ema` — Exponential Moving Average

[→ Source](momentum_tools.py#L55)

**What it measures:** A weighted moving average applying exponentially decreasing weights to older data, making it more responsive to recent prices than SMA.

**Used by:** Virtually all momentum and trend-following traders. EMA is the building block for MACD, PPO, KAMA, and many other indicators. Hedge funds and quant desks reference the 9, 12, 26, 50, and 200 period EMAs.

**When you need it:** Use EMA to smooth noisy price data and identify the current trend direction. A price above EMA suggests an uptrend; below suggests a downtrend. EMA crossovers (fast crosses slow) are common entry and exit signals in momentum strategies.

```python
from momentum.momentum_tools import ema, crossover
fast = ema(df["Close"], window=12)
slow = ema(df["Close"], window=26)
buy_signal = crossover(fast, slow)   # True on the crossover bar
```

---

### `sma` — Simple Moving Average

[→ Source](momentum_tools.py#L96)

**What it measures:** The arithmetic mean of prices over the past `window` periods. Equal weight is given to every data point.

**Used by:** The most widely used baseline for trend identification. Retail traders, institutional desks, and systematic funds all reference the 20-, 50-, and 200-day SMA as key support/resistance levels.

**When you need it:** Use SMA to establish a price baseline. The Golden Cross (50-day SMA crosses above 200-day SMA) is one of the most cited momentum buy signals. Use shorter SMAs (5–20) for faster intraday momentum strategies.

```python
from momentum.momentum_tools import sma
trend_filter = df["Close"] > sma(df["Close"], window=200)   # price above 200-day SMA
```

---

### `wma` — Weighted Moving Average

[→ Source](momentum_tools.py#L133)

**What it measures:** A moving average where each period receives a linearly increasing weight, so the most recent price has the highest weight.

**Used by:** Traders who want more responsiveness than SMA but a smoother curve than EMA. Common in short-term momentum systems and intraday trading.

**When you need it:** Use WMA when you need a moving average that reacts more quickly to recent price changes than SMA but less erratically than raw price. Particularly useful for fast-moving momentum environments (e.g., pre-earnings moves, opening range breakouts).

```python
from momentum.momentum_tools import wma
df["wma9"] = wma(df["Close"], window=9)
```

---

## 2. Momentum Oscillators

### `rsi` — Relative Strength Index

[→ Source](momentum_tools.py#L186)

**What it measures:** Compares the average size of recent up-moves to down-moves over `window` periods, producing a value between 0 and 100. RSI = 100 when all moves are positive.

**Used by:** One of the most widely used indicators across all asset classes. Retail traders use it to spot overbought/oversold conditions; institutional quants use it in mean-reversion and momentum factor models. Introduced by Welles Wilder in 1978.

**When you need it:**
- RSI > 70 → asset may be overbought, consider taking profits
- RSI < 30 → asset may be oversold, potential long entry if trend is intact
- RSI divergence (price makes new high, RSI does not) → weakening momentum before a reversal
- In a strong uptrend, use 40 as the support level (buy the dip) instead of 30

```python
from momentum.momentum_tools import rsi
df["rsi"] = rsi(df["Close"], window=14)
buy_signal  = df["rsi"] < 30
sell_signal = df["rsi"] > 70
```

**Reference:** https://www.investopedia.com/terms/r/rsi.asp

---

### `stochastic_oscillator` — Stochastic %K and %D

[→ Source](momentum_tools.py#L248)

**What it measures:** Measures where the closing price sits relative to the high-low range over the past `window` periods. Returns %K (raw) and %D (signal = SMA of %K).

**Used by:** George Lane introduced it in the 1950s. Widely used by swing traders and day traders to find momentum reversals. Common in equity, forex, and futures markets.

**When you need it:**
- %K > 80 → overbought; %K < 20 → oversold
- Buy signal: %K crosses above %D while both are below 20
- Sell signal: %K crosses below %D while both are above 80
- Combine with ADX > 25 to avoid whipsaws in ranging markets

```python
from momentum.momentum_tools import stochastic_oscillator
stoch_k, stoch_d = stochastic_oscillator(df["High"], df["Low"], df["Close"])
buy = (stoch_k < 20) & (stoch_k > stoch_d)
```

**Reference:** https://www.investopedia.com/terms/s/stochasticoscillator.asp

---

### `stochrsi` — Stochastic RSI

[→ Source](momentum_tools.py#L312)

**What it measures:** Applies the Stochastic Oscillator formula to RSI values instead of raw price, producing a more sensitive oscillator (0–1 scale) that responds faster to momentum shifts.

**Used by:** Developed by Tushar Chande and Stanley Kroll (1994). Used heavily by crypto traders and short-term equity momentum traders. Popular in algorithmic systems trading volatile assets.

**When you need it:**
- StochRSI > 0.8 → extremely overbought (more extreme than RSI > 70)
- StochRSI < 0.2 → extremely oversold
- Use %K/%D crossovers as finer-grained entry timing within a broader RSI setup
- Best on higher-volatility stocks during high-volume sessions

```python
from momentum.momentum_tools import stochrsi
srsi, srsi_k, srsi_d = stochrsi(df["Close"])
buy = (srsi_k < 0.2) & (srsi_k > srsi_d)
```

**Reference:** https://www.investopedia.com/terms/s/stochrsi.asp

---

### `williams_r` — Williams %R

[→ Source](momentum_tools.py#L380)

**What it measures:** Inverse of the Fast Stochastic Oscillator. Measures how close the current close is to the highest high over the lookback period (0 to -100 scale).

**Used by:** Developed by Larry Williams. Favored by short-term swing traders in equities, futures, and forex. Used by algorithmic systems as a fast momentum filter.

**When you need it:**
- %R above -20 → overbought; watch for reversal
- %R below -80 → oversold; watch for bounce/entry
- %R moving from below -80 to above -50 confirms bullish momentum
- Particularly useful as a confirmation tool for RSI signals

```python
from momentum.momentum_tools import williams_r
df["willr"] = williams_r(df["High"], df["Low"], df["Close"])
buy = df["willr"] < -80
```

**Reference:** https://www.investopedia.com/terms/w/williamsr.asp

---

### `roc` — Rate of Change

[→ Source](momentum_tools.py#L432)

**What it measures:** Percentage change in price from `window` periods ago to today. The simplest pure momentum oscillator.
`ROC = 100 * (Close - Close[n]) / Close[n]`

**Used by:** The core building block of most momentum factor models in academic finance (Jegadeesh & Titman 1993, Fama-French). Quant funds use ROC to rank stocks in cross-sectional momentum screens.

**When you need it:**
- ROC > 0 → price is higher than n periods ago → upward momentum
- Use 12-month ROC (skip last month) for classical monthly cross-sectional momentum
- Use 3–5 day ROC for short-term intraday momentum trades
- Combine with RSI: high ROC + RSI not yet overbought = strong momentum with room to run

```python
from momentum.momentum_tools import roc
df["roc_12"] = roc(df["Close"], window=12)
strong_momentum = df["roc_12"] > 10   # +10% over 12 periods
```

**Reference:** https://school.stockcharts.com/doku.php?id=technical_indicators:rate_of_change_roc_and_momentum

---

### `awesome_oscillator` — Awesome Oscillator

[→ Source](momentum_tools.py#L492)

**What it measures:** Measures market momentum by computing the difference between a 5-period and 34-period SMA of the bar's midpoint price (H+L)/2. Developed by Bill Williams.

**Used by:** Popularised by Bill Williams in "Trading Chaos." Widely used by retail traders and system developers. Institutional desks use it as a noise filter.

**When you need it:**
- AO crosses above zero → bullish momentum shift, potential buy
- AO crosses below zero → bearish momentum shift, potential sell
- "Saucer" pattern (three consecutive AO bars, middle is lowest, all above zero) = buy
- Combine with Bill Williams' Alligator (3-EMA trend system)

```python
from momentum.momentum_tools import awesome_oscillator
df["ao"] = awesome_oscillator(df["High"], df["Low"])
buy = df["ao"] > 0
```

**Reference:** https://www.tradingview.com/wiki/Awesome_Oscillator_(AO)

---

### `tsi` — True Strength Index

[→ Source](momentum_tools.py#L551)

**What it measures:** Double-smoothed momentum indicator showing both trend direction and overbought/oversold conditions. Scales to approximately -100 to +100.

**Used by:** Introduced by William Blau (1991). Used by systematic traders and quants who want a smoother, less noisy version of RSI. Favored in equity futures and ETF momentum models.

**When you need it:**
- TSI > 0 → positive momentum (bullish)
- TSI < 0 → negative momentum (bearish)
- TSI crosses above its EMA signal line → buy
- Divergence between price and TSI is a leading reversal signal

```python
from momentum.momentum_tools import tsi
df["tsi"] = tsi(df["Close"])
buy = df["tsi"] > 0
```

**Reference:** https://en.wikipedia.org/wiki/True_strength_index

---

### `cci` — Commodity Channel Index

[→ Source](momentum_tools.py#L601)

**What it measures:** Measures the deviation of typical price from its SMA, normalized by the mean absolute deviation. Originally designed for commodities, broadly applied.

**Used by:** Developed by Donald Lambert (1980). Used by commodity and equity traders to identify cyclical turns. Quant systematic funds use CCI as a momentum signal in multi-factor models.

**When you need it:**
- CCI > +100 → strong upward momentum above average; breakout buy signal in a trending environment
- CCI < -100 → strong downward momentum; potential short or exit long
- CCI returning inside ±100 from an extreme = reversal signal

```python
from momentum.momentum_tools import cci
df["cci"] = cci(df["High"], df["Low"], df["Close"])
buy = (df["cci"] > 100) & (df["cci"].shift(1) < 100)   # breakout above +100
```

**Reference:** http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:commodity_channel_index_cci

---

### `ultimate_oscillator` — Ultimate Oscillator

[→ Source](momentum_tools.py#L657)

**What it measures:** Larry Williams' (1976) oscillator capturing momentum across three timeframes (7, 14, 28 periods) simultaneously, reducing false signals from single-period approaches.

**Used by:** Developed by Larry Williams. Used by systematic traders who want a multi-timeframe view of buying pressure. Popular in multi-factor momentum models.

**When you need it:**
- UO above 70 → overbought; watch for sell/reversal
- UO below 30 → oversold; watch for buy opportunity
- Williams' original strategy: buy when UO < 30 and a bullish divergence forms
- Useful when RSI and Stochastic give conflicting signals; UO acts as a tie-breaker

```python
from momentum.momentum_tools import ultimate_oscillator
df["uo"] = ultimate_oscillator(df["High"], df["Low"], df["Close"])
buy = df["uo"] < 30
```

**Reference:** http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:ultimate_oscillator

---

### `ppo` — Percentage Price Oscillator

[→ Source](momentum_tools.py#L723)

**What it measures:** MACD expressed as a percentage of the slower EMA, making it comparable across different price levels and instruments.

**Used by:** Used by quant analysts comparing momentum signals across different securities at different price levels. Common in cross-sectional factor models.

**When you need it:**
- PPO > 0 → fast EMA above slow EMA → bullish momentum
- PPO crosses Signal from below → buy signal
- PPO Histogram turning positive → early momentum shift
- Use PPO instead of MACD when comparing momentum across a portfolio of stocks with very different price levels

```python
from momentum.momentum_tools import ppo
ppo_line, signal, hist = ppo(df["Close"])
buy = (ppo_line > signal) & (ppo_line.shift(1) < signal.shift(1))
```

**Reference:** https://school.stockcharts.com/doku.php?id=technical_indicators:price_oscillators_ppo

---

### `kama` — Kaufman's Adaptive Moving Average

[→ Source](momentum_tools.py#L784)

**What it measures:** An adaptive moving average that self-adjusts its smoothing speed based on market noise (Efficiency Ratio). Moves quickly in strong trends, slowly in choppy markets.

**Used by:** Developed by Perry Kaufman (1995). Used by systematic CTA funds and quantitative momentum traders. Avoids many false signals during consolidation.

**When you need it:**
- Price crosses above KAMA → momentum building, potential buy
- Price crosses below KAMA → momentum deteriorating, exit long
- KAMA slope flattening → market is ranging, avoid new entries
- KAMA steepening → trending market, add to winners
- Use for breakout strategies — wait for KAMA to confirm the breakout

```python
from momentum.momentum_tools import kama
df["kama"] = kama(df["Close"])
buy = df["Close"] > df["kama"]
```

**Reference:** https://www.tradingview.com/ideas/kama/

---

## 3. Trend Strength Indicators

### `macd` — Moving Average Convergence / Divergence

[→ Source](momentum_tools.py#L871)

**What it measures:** Trend-following momentum indicator showing the relationship between two EMAs of price (12 and 26 period). Returns MACD line, Signal line (9-period EMA of MACD), and Histogram.

**Used by:** One of the most universally followed indicators. Retail traders use crossovers for entry/exit signals; institutional traders use the histogram for momentum strength; quant funds incorporate MACD in momentum factor models. Applies to equities, forex, crypto, and futures.

**When you need it:**
- MACD line crosses above Signal → bullish crossover, buy signal
- MACD crosses below Signal → bearish crossover, sell signal
- Histogram turns positive (from negative) → early buy signal
- MACD above zero line → underlying trend is bullish
- Divergence: price makes new high but MACD does not → weakening momentum

```python
from momentum.momentum_tools import macd
macd_line, signal, hist = macd(df["Close"])
buy  = (macd_line > signal) & (macd_line.shift(1) <= signal.shift(1))
sell = (macd_line < signal) & (macd_line.shift(1) >= signal.shift(1))
```

**Reference:** https://en.wikipedia.org/wiki/MACD

---

### `adx` — Average Directional Index

[→ Source](momentum_tools.py#L933)

**What it measures:** Measures trend strength (not direction) on a 0–100 scale. Returns ADX, +DI (bullish directional movement), and -DI (bearish directional movement). Developed by J. Welles Wilder (1978).

**Used by:** Widely used by professional trend-following CTAs to assess whether a market is trending or ranging. **Essential in any momentum system to avoid trading in choppy markets.**

**When you need it:**
- ADX > 25 → market is trending; momentum signals are reliable
- ADX < 20 → market is ranging; momentum signals produce many false positives
- ADX > 40 → extremely strong trend; consider trailing stops
- +DI crosses above -DI with ADX > 25 → strong bullish entry
- -DI crosses above +DI with ADX > 25 → strong bearish entry

```python
from momentum.momentum_tools import adx
adx_vals, pdi, mdi = adx(df["High"], df["Low"], df["Close"])
trending   = adx_vals > 25
bull_trend = trending & (pdi > mdi)
```

**Reference:** http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:average_directional_index_adx

---

### `aroon` — Aroon Indicator

[→ Source](momentum_tools.py#L1006)

**What it measures:** Identifies when trends are likely to change direction by measuring periods since the last highest high and lowest low. Returns Aroon Up, Aroon Down, and the Oscillator (Up - Down).

**Used by:** Developed by Tushar Chande (1995). Used by trend traders to identify emerging and weakening trends earlier than ADX. Popular in systematic equity momentum models.

**When you need it:**
- Aroon Up near 100 and Aroon Down near 0 → strong uptrend
- Aroon Up crosses above Aroon Down → new bullish trend beginning
- Oscillator > 50 → bullish; Oscillator < -50 → bearish
- Use as an early entry signal before ADX confirms the trend

```python
from momentum.momentum_tools import aroon
aroon_up, aroon_down, aroon_osc = aroon(df["High"], df["Low"])
emerging_bull = (aroon_up > 70) & (aroon_down < 30)
```

**Reference:** https://www.investopedia.com/terms/a/aroon.asp

---

### `parabolic_sar` — Parabolic SAR

[→ Source](momentum_tools.py#L1062)

**What it measures:** J. Welles Wilder's trailing stop indicator that places dots below rising prices (uptrend) and above falling prices (downtrend). The SAR accelerates as the trend continues.

**Used by:** Widely used by professional trend traders as a dynamic trailing stop-loss and trend direction indicator. Common in systematic futures and equity trading. Used by many IBKR algo traders to manage open positions.

**When you need it:**
- Price crosses above SAR → trend reverses to uptrend, buy
- Price crosses below SAR → trend reverses to downtrend, sell
- **Use SAR as a trailing stop level for open positions**
- Combine with ADX: only act on SAR reversals when ADX > 25
- In choppy markets (ADX < 20), SAR produces many whipsaws — avoid

```python
from momentum.momentum_tools import parabolic_sar
df["psar"] = parabolic_sar(df["High"], df["Low"], df["Close"])
trailing_stop = df["psar"]   # use as your stop-loss level
```

**Reference:** https://school.stockcharts.com/doku.php?id=technical_indicators:parabolic_sar

---

### `kst` — Know Sure Thing Oscillator

[→ Source](momentum_tools.py#L1143)

**What it measures:** Martin Pring's multi-timeframe momentum indicator combining four different Rate of Change measures, each smoothed and weighted to emphasize longer cycles.

**Used by:** Developed by Martin Pring. Used by cycle analysts and macro momentum traders. Common in ETF rotation strategies and sector momentum models. Popular among practitioners of intermarket analysis.

**When you need it:**
- KST crosses above its signal line → bullish; buy signal
- KST crosses below its signal line → bearish; sell signal
- Best on weekly or monthly data for intermediate-term momentum
- Use to identify major stock market cycle turning points

```python
from momentum.momentum_tools import kst
kst_line, kst_signal = kst(df["Close"])
buy = (kst_line > kst_signal) & (kst_line.shift(1) <= kst_signal.shift(1))
```

**Reference:** https://en.wikipedia.org/wiki/KST_oscillator

---

## 4. Risk & Performance Metrics

### `volatility` — Annualized Volatility

[→ Source](momentum_tools.py#L1211)

**What it measures:** Annualized standard deviation of daily returns. Measures the dispersion of returns around the mean.

**Used by:** A fundamental risk metric used by every type of trader and risk manager. Options traders compare it to implied volatility. Portfolio managers use it for position sizing (risk parity). Quant funds use it in volatility-adjusted momentum.

**When you need it:**
- Use to size positions: smaller in high-volatility stocks, larger in low-volatility stocks
- Compare against VIX or implied vol to identify cheap/expensive vol
- Rising volatility often precedes corrections — use as a regime indicator

```python
from momentum.momentum_tools import volatility
returns = df["Close"].pct_change().dropna()
annual_vol = volatility(returns)
print(f"Annual vol: {annual_vol:.1%}")
```

---

### `sharpe_ratio` — Sharpe Ratio

[→ Source](momentum_tools.py#L1258)

**What it measures:** Measures risk-adjusted return by dividing excess return (above the risk-free rate) by the standard deviation of returns.

**Used by:** The most widely used performance metric in finance. Required reporting metric for most institutional funds. IBKR TWS risk reports include Sharpe.

**When you need it:**
- Use to compare two strategies: higher Sharpe = more efficient per unit of risk
- Sharpe > 1.0 → generally acceptable
- Sharpe > 2.0 → excellent for a systematic strategy
- **Run this on every backtest to evaluate strategy quality before going live**

```python
from momentum.momentum_tools import sharpe_ratio
returns = df["Close"].pct_change().dropna()
sharpe = sharpe_ratio(returns, risk_free_rate=0.05)
print(f"Sharpe: {sharpe:.2f}")
```

**Reference:** https://www.investopedia.com/terms/s/sharperatio.asp

---

### `sortino_ratio` — Sortino Ratio

[→ Source](momentum_tools.py#L1313)

**What it measures:** Like Sharpe ratio but uses downside deviation only (negative return volatility), avoiding penalising upside volatility.

**Used by:** Preferred over Sharpe by momentum traders whose strategies exhibit positive skewness. Used by CTAs, systematic equity funds, and options traders.

**When you need it:**
- Use when your strategy has asymmetric returns (momentum systems often have large winners/frequent small losers)
- Sortino > 2.0 → excellent for a momentum strategy
- If Sortino >> Sharpe, your strategy has positive skewness (desirable)
- Run alongside Sharpe for a complete risk/reward picture

```python
from momentum.momentum_tools import sortino_ratio
returns = df["Close"].pct_change().dropna()
sortino = sortino_ratio(returns)
print(f"Sortino: {sortino:.2f}")
```

**Reference:** http://www.redrockcapital.com/Sortino__A__Sharper__Ratio_Red_Rock_Capital.pdf

---

### `max_drawdown` — Maximum Drawdown

[→ Source](momentum_tools.py#L1368)

**What it measures:** The largest peak-to-trough decline in cumulative returns, expressed as a percentage. Represents the worst-case loss a strategy inflicted.

**Used by:** Used by every serious trader and risk manager to evaluate downside risk. Required metric for hedge fund due diligence. **Many traders use MDD as a circuit-breaker: if live drawdown exceeds backtest MDD, stop the strategy.**

**When you need it:**
- Key metric to decide whether you can sustain a strategy's losses
- Compare MDD to expected annual return: if MDD > annual return, may not recover within a year
- Use to set automatic risk-off rules: "If drawdown exceeds 5%, halt trading"
- Track live MDD vs backtest MDD as a regime-change signal

```python
from momentum.momentum_tools import max_drawdown
returns = df["Close"].pct_change().dropna()
mdd = max_drawdown(returns)
print(f"Max Drawdown: {mdd:.1%}")
```

---

### `calmar_ratio` — Calmar Ratio

[→ Source](momentum_tools.py#L1413)

**What it measures:** Ratio of CAGR to the absolute value of maximum drawdown.

**Used by:** Widely used by CTAs and hedge funds for trend-following strategies. Better than Sharpe for strategies with asymmetric returns. Industry standard for CTA performance evaluation.

**When you need it:**
- Calmar > 1.0 → you earn more annually than your worst loss
- Calmar > 3.0 → excellent for a momentum strategy
- Use alongside MDD and Sharpe for a full risk picture

```python
from momentum.momentum_tools import calmar_ratio
returns = df["Close"].pct_change().dropna()
calmar = calmar_ratio(returns)
print(f"Calmar: {calmar:.2f}")
```

---

### `cagr` — Compound Annual Growth Rate

[→ Source](momentum_tools.py#L1455)

**What it measures:** The geometric mean annual return, representing the steady rate of return that would produce the same total return.

**Used by:** The primary return metric for comparing strategies of different lengths. Reported by every fund, backtester, and broker (including IBKR).

**When you need it:**
- Always report CAGR alongside MDD and Sharpe in your backtest summary
- Compare CAGR against benchmark (e.g., SPY CAGR) to see if strategy generates alpha
- Use CAGR to set realistic daily P&L expectations

```python
from momentum.momentum_tools import cagr
returns = df["Close"].pct_change().dropna()
annual_return = cagr(returns)
print(f"CAGR: {annual_return:.1%}")
```

---

### `win_rate` — Win Rate

[→ Source](momentum_tools.py#L1494)

**What it measures:** The fraction of periods (or trades) with a positive return.

**Used by:** Used by discretionary and systematic traders to characterize strategy behavior. Important for understanding the psychological demands of a strategy.

**When you need it:**
- Use alongside average win/loss to assess overall expectancy
- Momentum strategies typically have win rates of 40–60%
- If win rate drops significantly in live trading vs backtest, it may signal overfitting or regime change

```python
from momentum.momentum_tools import win_rate
returns = df["Close"].pct_change().dropna()
wr = win_rate(returns)
print(f"Win rate: {wr:.1%}")
```

---

### `kelly_criterion` — Kelly Criterion

[→ Source](momentum_tools.py#L1535)

**What it measures:** The theoretically optimal fraction of capital to allocate per trade to maximise long-term geometric growth.

**Used by:** Introduced by John Kelly (1956). Used by quantitative traders and portfolio managers for position sizing. Ed Thorp popularised it in financial markets.

**When you need it:**
- Use output as the **maximum** fraction of your account to risk per trade
- In practice, use 25–50% of the Kelly fraction (Half/Quarter Kelly)
- Negative Kelly → strategy has negative expected value — do not trade it
- Recalculate monthly as win rates and R-multiples evolve

```python
from momentum.momentum_tools import kelly_criterion
strategy_returns = df["strategy_ret"]
kelly_f = kelly_criterion(strategy_returns)
position_size = 0.5 * kelly_f   # Half Kelly — safer in practice
print(f"Kelly Fraction: {kelly_f:.1%}  |  Half Kelly: {position_size:.1%}")
```

**Reference:** http://en.wikipedia.org/wiki/Kelly_criterion

---

### `value_at_risk` — Value at Risk

[→ Source](momentum_tools.py#L1594)

**What it measures:** Estimates the maximum expected daily loss at a given confidence level, assuming normally distributed returns (parametric method). E.g., 95% VaR = worst expected daily loss 95% of the time.

**Used by:** Basel III regulatory requirement. Used by risk managers to set daily loss limits. Every major brokerage including IBKR calculates VaR for portfolio risk.

**When you need it:**
- Use to set daily stop-loss levels: if 95% VaR is -2%, exit positions if portfolio drops >2% intraday
- Compare live daily losses against VaR as an alert system
- Always pair with CVaR — VaR alone doesn't capture tail risk

```python
from momentum.momentum_tools import value_at_risk
returns = df["Close"].pct_change().dropna()
var_95 = value_at_risk(returns, confidence=0.95)
print(f"95% VaR: {var_95:.2%}")
```

---

### `conditional_value_at_risk` — Conditional Value at Risk

[→ Source](momentum_tools.py#L1646)

**What it measures:** The expected return *given* that the loss exceeds the VaR threshold. Also called Expected Shortfall. Captures tail risk that VaR misses.

**Used by:** Preferred over VaR by academic researchers and sophisticated risk managers because it is a coherent risk measure. Required by FRTB regulations. Used by quant funds to size positions in tail-risky instruments.

**When you need it:**
- High CVaR relative to VaR signals fat tails — consider reducing leverage
- Use 95% CVaR as your "expected loss in a bad market day" for scenario planning
- Always report alongside VaR for a complete tail-risk picture

```python
from momentum.momentum_tools import conditional_value_at_risk
returns = df["Close"].pct_change().dropna()
cvar_95 = conditional_value_at_risk(returns, confidence=0.95)
print(f"95% CVaR: {cvar_95:.2%}")
```

---

## 5. Cross-Sectional & Portfolio Momentum

### `momentum_score` — 12-1 Momentum Score

[→ Source](momentum_tools.py#L1716)

**What it measures:** The classic Jegadeesh & Titman (1993) momentum factor: total return over the past `lookback` periods, excluding the most recent `skip` periods (to avoid short-term reversal contamination).

**Used by:** The cornerstone of quantitative equity momentum investing. Used by factor-based ETFs (e.g., MTUM), hedge funds (AQR, Cliff Asness), and academic researchers. The single most documented anomaly in finance.

**When you need it:**
- Rank a universe of stocks by `momentum_score` and long the top decile
- Use 252-day lookback with 21-day skip for daily data
- Rebalance monthly for a long-only momentum portfolio
- Combine with trend filter (price > 200-day SMA) to avoid stocks in downtrends

```python
from momentum.momentum_tools import momentum_score
scores = {
    ticker: momentum_score(price_series)
    for ticker, price_series in price_dict.items()
}
top_picks = sorted(scores, key=scores.get, reverse=True)[:10]
```

**Reference:** Jegadeesh & Titman, "Returns to Buying Winners and Selling Losers", *Journal of Finance*, 1993.

---

### `rank_momentum` — Cross-Sectional Ranking

[→ Source](momentum_tools.py#L1762)

**What it measures:** Ranks a universe of assets by their 12-1 momentum score and returns the top N assets. The foundation of most cross-sectional momentum (relative strength) strategies.

**Used by:** Used by quant ETF managers, factor investors, and systematic hedge funds. Also used by retail momentum traders to select the strongest stocks from a watchlist.

**When you need it:**
- Use monthly to rebalance a momentum portfolio
- Feed a watchlist of 50–500 stocks to identify the top 10%
- Combine with quality/fundamental filters to reduce value traps
- Use as a stock selection engine before applying technical entry criteria

```python
from momentum.momentum_tools import rank_momentum
# prices_dict = {"AAPL": pd.Series(...), "MSFT": pd.Series(...), ...}
top_stocks = rank_momentum(prices_dict, top_n=10)
print("Top momentum stocks:", top_stocks)
```

---

### `dual_momentum` — Dual Momentum Signal

[→ Source](momentum_tools.py#L1813)

**What it measures:** Gary Antonacci's system combining absolute momentum (does the asset beat cash?) with relative momentum (does the asset beat a benchmark?). Returns "BUY", "BENCHMARK", or "CASH".

**Used by:** Developed by Gary Antonacci ("Dual Momentum Investing", 2014). Used by individual investors and systematic fund managers for simple but highly effective asset allocation.

**When you need it:**
- "BUY": asset has positive absolute momentum AND beats benchmark
- "BENCHMARK": asset beats benchmark but has negative absolute momentum
- "CASH": neither condition met — move to T-bills
- Best for long-term systematic investors managing ETF rotation (SPY/AGG/SHY)
- Use monthly on a small set of assets

```python
from momentum.momentum_tools import dual_momentum
spy_ret  = spy_prices.pct_change().dropna()
agg_ret  = agg_prices.pct_change().dropna()
signal = dual_momentum(spy_ret, agg_ret)
print(f"Dual Momentum Signal: {signal}")
```

**Reference:** Gary Antonacci, *Dual Momentum Investing*, McGraw-Hill, 2014. https://www.optimalmomentum.com/

---

## 6. Signal Generation Utilities

### `crossover` — Crossover Signal

[→ Source](momentum_tools.py#L1893)

**What it measures:** Returns a boolean series that is True on the exact bar where the fast series crosses above the slow series.

**Used by:** Universal in systematic trading. The backbone of crossover-based entry logic (EMA crossovers, MACD signal crossings, RSI threshold crossings).

**When you need it:**
- EMA(fast) crosses EMA(slow): buy signal
- MACD crosses above Signal line: buy signal
- RSI crosses above 30 from below: oversold exit/buy signal
- Stochastic %K crosses above %D: buy signal

```python
from momentum.momentum_tools import crossover, ema
fast = ema(df["Close"], 12)
slow = ema(df["Close"], 26)
buy_signal  = crossover(fast, slow)
entry_dates = df.index[buy_signal]
```

---

### `crossunder` — Crossunder Signal

[→ Source](momentum_tools.py#L1938)

**What it measures:** Returns a boolean series that is True on the exact bar where the fast series crosses below the slow series. Mirror of `crossover()`.

**Used by:** Used for sell/short signals in systematic strategies. The Death Cross (50-day SMA crosses below 200-day SMA) is one of the most watched crossunder signals.

**When you need it:**
- EMA(fast) crosses below EMA(slow): sell/exit long signal
- MACD crosses below Signal line: sell/short signal
- RSI crosses below 70 from above: overbought exit signal
- Price drops below Parabolic SAR: exit long, consider short

```python
from momentum.momentum_tools import crossunder, ema
fast = ema(df["Close"], 12)
slow = ema(df["Close"], 26)
sell_signal = crossunder(fast, slow)
```

---

### `atr` — Average True Range

[→ Source](momentum_tools.py#L1981)

**What it measures:** Measures market volatility by averaging the True Range over `window` periods. Does not indicate direction — only volatility magnitude.

**Used by:** Developed by J. Welles Wilder (1978). **Universally used for stop-loss placement and position sizing.** Essential tool for any systematic momentum trader managing risk.

**When you need it:**
- Set stop-loss at `entry_price - 2 * ATR` (long) to avoid being stopped out by daily noise
- Size positions so that 1 ATR move equals a fixed % of capital: `shares = risk_$ / ATR`
- If ATR expands significantly, reduce position size proportionally
- Rising ATR on a breakout confirms the move is backed by volatility expansion

```python
from momentum.momentum_tools import atr
df["atr"] = atr(df["High"], df["Low"], df["Close"])
stop_loss   = df["Close"] - 2 * df["atr"]
# Position sizing: risk $500 per trade
shares = 500 / df["atr"].iloc[-1]
```

**Reference:** https://www.investopedia.com/terms/a/atr.asp

---

### `zscore` — Rolling Z-Score

[→ Source](momentum_tools.py#L2044)

**What it measures:** How many standard deviations the current value is from the rolling mean. Normalizes any indicator to a comparable scale.

**Used by:** Used by quant analysts for mean-reversion strategies and for normalising momentum signals in multi-factor models. Also used to detect anomalies (volume spikes, unusual price moves) in systematic entry filters.

**When you need it:**
- Z-score > 2 → far above recent mean → overbought in a mean-reversion context
- Z-score < -2 → far below recent mean → mean-reversion entry
- Use to normalise RSI or ROC values across a stock universe for cross-sectional comparison
- Combine with momentum score: high momentum + low Z-score = uptrend with no current extension

```python
from momentum.momentum_tools import zscore, rsi
df["rsi"] = rsi(df["Close"])
df["rsi_z"] = zscore(df["rsi"], window=20)
mean_rev_entry = df["rsi_z"] < -2.0
```

---

## Quick-Start Integration with IBKR / myIBApp.py

```python
import pandas as pd
from myIBApp import connect_to_tws
from momentum.momentum_tools import (
    rsi, macd, adx, atr, parabolic_sar,
    sharpe_ratio, max_drawdown, cagr,
    crossover, crossunder
)

# Connect to TWS
app = connect_to_tws()

# Build your indicator suite from OHLCV data
def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the momentum indicator suite to an OHLCV DataFrame."""
    df["rsi"]        = rsi(df["Close"])
    macd_l, sig, _   = macd(df["Close"])
    df["macd"]       = macd_l
    df["macd_sig"]   = sig
    adx_v, pdi, mdi  = adx(df["High"], df["Low"], df["Close"])
    df["adx"]        = adx_v
    df["pdi"]        = pdi
    df["mdi"]        = mdi
    df["atr"]        = atr(df["High"], df["Low"], df["Close"])
    df["psar"]       = parabolic_sar(df["High"], df["Low"], df["Close"])
    df["buy"]        = crossover(macd_l, sig) & (adx_v > 25)
    df["sell"]       = crossunder(macd_l, sig) & (adx_v > 25)
    return df
```
