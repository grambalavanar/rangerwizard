# Alpha Mean Reversion Strategy

The mean-reversion counterpart to `AlphaCompositeMomentumStrategy`. Combines **eleven independently peer-reviewed signals** into a single "oversold score" and enters long when the score exceeds a threshold, exiting when price returns to the mean, a stop is hit, or a time stop expires.

**Files:** [`mean_reversion/`](../mean_reversion/)

---

## Research Foundations

| # | Component | Source | Key Finding |
|---|---|---|---|
| 1 | Regime Gate (ADX + ER + Hurst) | Wilder (1978); Kaufman (1995); Lo (1991) | ADX < 20 + ER < 0.30 + H < 0.50 = confirmed MR regime |
| 2 | Price Z-Score | Poterba & Summers (1988); Lo & MacKinlay (1988) | z < -1.5 predicts positive returns within 3–10 days |
| 3 | ConnorsRSI | Connors, Alvarez & Hayward (2009) | ConnorsRSI < 10 → 70–80% win rate, 3–5 day hold |
| 4 | Bollinger %B Extreme | Bollinger (2002); Lento & Gradojevic (2007) | %B < 0 → positive 3-week return 65% of the time |
| 5 | Stochastic Extreme | Lane (1954); Elder (1993) | Both %K and %D < 20 = confirmed oversold |
| 6 | CCI Extreme | Lambert (1980); Pring (1991) | CCI < -150 (deeper than -100) = higher-probability reversal |
| 7 | Williams %R Extreme | Williams (1979) | %R < -85 = extreme oversold in recent range |
| 8 | KAMA Efficiency Ratio | Kaufman (1995) | ER < 0.25 optimises mean-reversion returns |
| 9 | Volume Climax | Granville (1963); Wyckoff (1930) | High vol on down day = Selling Climax → accumulation |
| 10 | RSI + MACD Divergence | Elder (1993); Cardwell (1994) | Divergence = leading reversal signal before price turn |
| 11 | OU Half-Life | Avellaneda & Lee (2010) | HL < 20 days = fast enough to trade profitably |

---

## Key Difference from Momentum

| | Momentum Strategy | Mean Reversion Strategy |
|---|---|---|
| **High composite score means** | Strong uptrend — buy | Strongly oversold — buy |
| **Exit trigger** | Trend weakening | Price returns to mean |
| **Stop type** | Trailing ATR (wide) | Fixed ATR (tight) |
| **Extra exit** | — | Time stop (default 10 bars) |
| **Best regime** | ADX > 25, Hurst > 0.5 | ADX < 22, Hurst < 0.5 |

---

## Quick Start

```bash
# Backtest demo (AAPL 3 years):
python3.11 mean_reversion/strategies/alpha_mean_reversion.py

# Backtest example runner:
python3.11 mean_reversion/test/run_backtest_mr.py --symbol AMD --years 5

# Genetic optimization (basket):
python3.11 mean_reversion/strategies/optimize_alpha_mr.py --fast

# Security-specific optimization with walk-forward:
python3.11 mean_reversion/strategies/optimize_alpha_mr.py --target AAPL --train-years 5 --val-years 2

# Backtest with optimized params (single):
python3.11 mean_reversion/strategies/backtest_optimized_mr.py --symbol AAPL --years 5

# Backtest all universes:
python3.11 mean_reversion/strategies/backtest_optimized_mr.py --universe all --years 3
```

---

## Directory Structure

```
mean_reversion/
    __init__.py
    mean_reversion_tools.py          ← MR indicator library
    strategies/
        __init__.py
        alpha_mean_reversion.py      ← Main composite strategy
        optimize_alpha_mr.py         ← GA optimizer runner
        backtest_optimized_mr.py     ← Historical test runner
        ALPHA_MEAN_REVERSION_DOCS.md ← This file
    test/
        __init__.py                  ← Re-exports from momentum/test
        run_backtest_mr.py           ← ASCII report runner
        paper_config_mr.json         ← Paper trading config
```

---

## `mean_reversion_tools.py` — Indicator Library

[→ Source](../mean_reversion_tools.py)

### `price_zscore()` — Rolling Z-Score

**What it measures:** `(Close − SMA(n)) / StdDev(n)`. The single most academically validated mean-reversion signal. Negative values = price below its mean = oversold.

**Research:** Poterba & Summers (1988); Lo & MacKinlay (1988).

**When:** z < -1.5 → moderate entry; z < -2.0 → high-conviction entry.

```python
from mean_reversion.mean_reversion_tools import price_zscore
z = price_zscore(df["Close"], window=20)
oversold = z < -1.5
```

---

### `bollinger_bands()` — Complete Bollinger Band Suite

**What it returns:** `upper`, `lower`, `mid`, `pct_b`, `bandwidth` as a dict.

**Research:** Bollinger (2002); %B < 0 predicts positive 3-week returns 65% of the time (Lento & Gradojevic 2007).

```python
from mean_reversion.mean_reversion_tools import bollinger_bands
bb = bollinger_bands(df["Close"], window=20, n_std=2.0)
below_band = bb["pct_b"] < 0.05   # below lower band
squeeze = bb["bandwidth"] < bb["bandwidth"].quantile(0.20)  # low volatility
```

---

### `efficiency_ratio()` — Kaufman's ER

**What it measures:** 0 = perfectly choppy (mean reversion works), 1 = perfectly trending (momentum works).

**Research:** Kaufman (1995) — ER < 0.25 is the operational threshold for switching to mean reversion.

```python
from mean_reversion.mean_reversion_tools import efficiency_ratio
er = efficiency_ratio(df["Close"], window=10)
mr_regime = er < 0.25
```

---

### `hurst_exponent()` — Trend vs Mean-Reversion Classifier

**What it measures:** H < 0.5 = anti-persistent (mean-reverting); H > 0.5 = persistent (trending).

**Research:** Hurst (1951); Lo (1991) "Long-Term Memory in Stock Market Prices."

```python
from mean_reversion.mean_reversion_tools import hurst_exponent
h = hurst_exponent(df["Close"], window=100)
mr_confirmed = h < 0.48
```

---

### `ou_halflife()` — Ornstein-Uhlenbeck Half-Life

**What it measures:** Expected days for price to revert halfway to its mean after a deviation. Short HL = fast mean reversion = better trade candidates.

**Research:** Avellaneda & Lee (2010) — only trade stocks with HL < 20 days.

```python
from mean_reversion.mean_reversion_tools import ou_halflife
hl = ou_halflife(df["Close"], window=60)
fast_reverter = hl < 15   # reverts within ~2 trading weeks
```

---

### `variance_ratio()` — Lo-MacKinlay VR Test

**What it measures:** VR < 1 = negative autocorrelation = mean-reverting. VR = 1 = random walk. VR > 1 = trending.

**Research:** Lo & MacKinlay (1988) — gold-standard statistical test for mean reversion.

```python
from mean_reversion.mean_reversion_tools import variance_ratio
vr = variance_ratio(df["Close"], q=5)
mr_evidence = vr < 0.90
```

---

### `connors_rsi()` — ConnorsRSI

**What it measures:** `(RSI(2) + RSI(streak) + PercentRank(100)) / 3`. Values < 10 = extreme oversold → 70–80% win rate over 3–5 days.

**Research:** Connors, Alvarez & Hayward (2009) "High Probability ETF Trading."

```python
from mean_reversion.mean_reversion_tools import connors_rsi
crsi = connors_rsi(df["Close"])
strong_buy = crsi < 10
```

---

### `volume_climax()` — Selling Exhaustion Detector

**What it measures:** High volume on a significant down day = institutional capitulation → often marks a bottom.

**Research:** Wyckoff (1930) Selling Climax; Granville (1963) OBV.

---

### `rsi_bullish_divergence()` — Leading Reversal Signal

**What it measures:** Price makes new low but RSI does not = weakening downside momentum, precedes reversal.

**Research:** Elder (1993) "Trading For A Living" Ch. 26–27.

---

### `mean_reversion_regime()` — Composite Regime Score

**What it measures:** Combined ADX + ER + Hurst signal. 1.0 = all three confirm MR regime. Use as a primary filter before running any MR strategy.

```python
from mean_reversion.mean_reversion_tools import mean_reversion_regime
regime = mean_reversion_regime(df["Close"], df["High"], df["Low"])
trade_today = regime.iloc[-1] >= 0.67   # at least 2/3 confirm
```

---

## `AlphaMeanReversionStrategy` — The Strategy Class

[→ Source](alpha_mean_reversion.py)

```python
from mean_reversion.strategies.alpha_mean_reversion import AlphaMeanReversionStrategy

# Default parameters
strategy = AlphaMeanReversionStrategy()

# Custom parameters
strategy = AlphaMeanReversionStrategy(params={
    "entry_threshold": 0.65,
    "time_stop_bars":  8,
    "atr_stop_mult":   1.5,
    "zscore_entry":    -2.0,
})

# With optimised parameters
strategy = AlphaMeanReversionStrategy(params=opt_result.best_params)
```

**Exit mechanisms (all active simultaneously):**

| Exit | Trigger | Purpose |
|---|---|---|
| Composite exit | `composite < exit_threshold` | Mean reversion complete / regime shift |
| Mean target | `z-score >= zscore_exit` | Price returned to mean |
| ATR stop | `close < entry − atr_mult × ATR` | Loss control |
| Time stop | `bars_held >= time_stop_bars` | Prevents holding stale MR trades |

---

## Genetic Optimization

```python
from mean_reversion.strategies.alpha_mean_reversion import (
    AlphaMeanReversionStrategy, TUNABLE_PARAM_SPACE,
)
from mean_reversion.test import GeneticOptimizer, GAConfig, print_optimization_report

opt = GeneticOptimizer(
    strategy_factory = AlphaMeanReversionStrategy,
    param_space      = TUNABLE_PARAM_SPACE,
    symbols          = ["AAPL", "MSFT", "AMD", "SPY", "QQQ", "V", "TSLA", "JPM", "XOM", "AMZN"],
    config           = GAConfig(
        population_size     = 30,
        n_generations       = 100,
        fitness_metric      = "robust",
        consistency_penalty = 0.30,
        n_workers           = 4,
    ),
)
result = opt.run()
print_optimization_report(result)
```

Results saved to `momentum/test/runs/alpha_mr_opt.json`.

**25 tunable parameters** including all 11 component weights, regime thresholds, oscillator windows, entry/exit levels, stop distances, and time stop length.

---

## Regime Switching with Momentum

The two strategies are designed to complement each other:

```python
from mean_reversion.mean_reversion_tools import mean_reversion_regime
from momentum.strategies.alpha_composite import AlphaCompositeMomentumStrategy
from mean_reversion.strategies.alpha_mean_reversion import AlphaMeanReversionStrategy

# Compute regime
regime_score = mean_reversion_regime(df["Close"], df["High"], df["Low"]).iloc[-1]

if regime_score >= 0.67:
    # Mean-reverting market — use MR strategy
    strategy = AlphaMeanReversionStrategy(params=mr_params)
    print("Regime: MEAN REVERSION")
else:
    # Trending market — use momentum strategy
    strategy = AlphaCompositeMomentumStrategy(params=mom_params)
    print("Regime: MOMENTUM")

df_out = strategy.generate_signals(df.copy())
signal  = df_out["signal"].iloc[-1]
```

---

## MR-Specific Diagnostic Metrics

```python
from mean_reversion.mean_reversion_tools import avg_holding_period, mean_reversion_speed

# After backtesting
avg_hold = avg_holding_period(result.trades)
autocorr = mean_reversion_speed(result.daily_returns)

print(f"Avg holding period  : {avg_hold:.1f} days  (target < 10)")
print(f"Return autocorr lag1: {autocorr:+.3f}       (target < 0)")
```

| Metric | Target | Meaning if wrong |
|---|---|---|
| Avg holding period | < 10 days | If >20 days: you're holding too long, consider tightening time stop |
| Return autocorr lag1 | < 0 | If > 0: you may be in a trending regime, check regime gate |
| Win rate | 50–70% | MR has higher win rate but smaller wins than momentum |
| Sharpe | > 0.8 | Acceptable for a MR-only strategy |
