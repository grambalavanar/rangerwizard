# Alpha Composite — Complete Strategy Reference

**File:** [`strategies/alpha_composite.py`](alpha_composite.py)  
**3,062 lines · self-contained · no external strategy imports**

---

## What is this file?

`alpha_composite.py` is the single source of truth for all trading strategy logic. It contains:

| Section | Contents |
|---|---|
| **Regime Classifier** | `GaussianHMM`, `RegimeClassifier`, `volatility_regime()`, `trend_regime()`, `classify_regime()` |
| **Momentum Strategy** | `AlphaCompositeMomentumStrategy` — 11 signals |
| **Mean Reversion Strategy** | `AlphaMeanReversionStrategy` — 11 signals |
| **Unified Strategy** | `UnifiedAlphaStrategy` — routes signals via regime |
| **Parameter Spaces** | `TUNABLE_PARAM_SPACE`, `MR_TUNABLE_PARAM_SPACE`, `UNIFIED_PARAM_SPACE` |

---

## Architecture

```
strategies/alpha_composite.py
│
├── REGIME CLASSIFIER (Hamilton 1989 HMM + Bollerslev vol + Wilder/Kaufman/Lo trend)
│     GaussianHMM ──────────────────────────────────────────────── P(bull | returns)
│     volatility_regime ─────────────────────────────────────── elevated / normal
│     trend_regime ──────────────────────────────────────── trending / sideways
│     RegimeClassifier ────── combines all three ──► MOMENTUM / MEAN_REVERSION / CASH
│
├── AlphaCompositeMomentumStrategy   (11 signals, used in MOMENTUM regime)
│     1. Weinstein Stage (SMA alignment + ADX)        w ≈ 0.20
│     2. TSMOM multi-horizon (12-1, 6-1, 3mo)         w ≈ 0.18
│     3. Linear Trend Regression (OLS/vol-adj)        w ≈ 0.15
│     4. 52-Week High Proximity                       w ≈ 0.10
│     5. MACD + TSI Quality                           w ≈ 0.10
│     6. RSI Zone + StochRSI                          w ≈ 0.08
│     7. KST Oscillator                               w ≈ 0.07
│     8. Volume Confirmation (OBV)                    w ≈ 0.07
│     9. Ichimoku Cloud                               w ≈ 0.05
│    10. Bollinger Band %B                            w ≈ 0.07
│    11. Money Flow Index (MFI)                       w ≈ 0.07
│        [vol crash damper × composite]
│
├── AlphaMeanReversionStrategy       (11 signals, used in MEAN_REVERSION regime)
│     1. Regime Gate (ADX + ER + Hurst)               w ≈ 0.20
│     2. Price Z-Score                                w ≈ 0.18
│     3. ConnorsRSI                                   w ≈ 0.12
│     4. Bollinger Band extreme (%B < 0.15)           w ≈ 0.10
│     5. Stochastic Extreme                           w ≈ 0.08
│     6. CCI Extreme (< -150)                         w ≈ 0.08
│     7. Williams %R Extreme (< -85)                  w ≈ 0.07
│     8. KAMA Efficiency Ratio                        w ≈ 0.06
│     9. Volume Climax (Wyckoff selling exhaustion)   w ≈ 0.05
│    10. RSI + MACD Bullish Divergence                w ≈ 0.04
│    11. OU Half-Life Gate                            w ≈ 0.02
│
└── UnifiedAlphaStrategy             (regime-gated, routes to sub-strategies)
      │
      ├─ MOMENTUM       → AlphaCompositeMomentumStrategy.generate_signals()
      ├─ MEAN_REVERSION → AlphaMeanReversionStrategy.generate_signals()
      └─ CASH           → signal = 0
```

---

## Quick Start

```bash
# Run the standalone demo (UnifiedAlphaStrategy on AAPL):
python3.11 strategies/alpha_composite.py

# Different symbol:
python3.11 strategies/alpha_composite.py --symbol MSFT --years 5

# Backtest all universes:
python3.11 strategies/backtest/backtest_unified.py --universe all

# Optimize:
python3.11 strategies/optimize/optimize_unified.py --fast
```

---

## Imports (single-file, all self-contained)

```python
from strategies.alpha_composite import (
    # Regime
    REGIME_MOMENTUM, REGIME_MEAN_REVERSION, REGIME_CASH,
    GaussianHMM, RegimeClassifier,
    volatility_regime, trend_regime, classify_regime,
    # Strategies
    AlphaCompositeMomentumStrategy,
    AlphaMeanReversionStrategy,
    UnifiedAlphaStrategy,
    # Parameters
    DEFAULT_PARAMS, MR_DEFAULT_PARAMS, UNIFIED_DEFAULT_PARAMS,
    # GA param spaces
    TUNABLE_PARAM_SPACE, MR_TUNABLE_PARAM_SPACE, UNIFIED_PARAM_SPACE,
)
```

---

## Section 1: Regime Classifier

### `REGIME_MOMENTUM / REGIME_MEAN_REVERSION / REGIME_CASH`

String constants used as labels. Compare `df["_regime"].iloc[-1]` against these.

---

### `GaussianHMM` — 2-State Hidden Markov Model

**Research:** Hamilton (1989) "A New Approach to the Economic Analysis of Nonstationary Time Series"; Ang & Bekaert (2002); Turner, Startz & Nelson (1989).

**What it does:** Fits a 2-state Gaussian HMM to daily log-returns using the Baum-Welch EM algorithm. State 0 = low-volatility/positive-drift (bull); State 1 = high-volatility/negative-drift (bear/choppy). The bull-state posterior probability is the primary HMM regime signal.

**After fitting, check:**
- `hmm.state_means[0]` > 0 (bull state has positive mean return)
- `hmm.state_vols[0]` < `hmm.state_vols[1]` (bull state has lower vol)
- `hmm.transition_matrix[0,0]` > 0.92 (regimes are persistent)

```python
from strategies.alpha_composite import GaussianHMM
import numpy as np

returns = np.log(df["Close"]).diff().dropna().values
hmm = GaussianHMM(n_states=2, n_iter=200).fit(returns.reshape(-1, 1))
proba = hmm.predict_proba(returns.reshape(-1, 1))
bull_prob = proba[:, hmm.bull_state]
print(f"Bull drift: {hmm.state_means[0]*252:.1%}/yr  Bull vol: {hmm.state_vols[0]:.1%}")
```

**Key methods:**
- `fit(X)` — Baum-Welch EM on shape (T,1) returns array
- `predict_proba(X)` — smoothed P(q_t=state | all observations), shape (T,K)
- `predict(X)` — Viterbi most-likely state sequence, shape (T,)
- `.state_means`, `.state_vols`, `.transition_matrix` — fitted parameters

---

### `volatility_regime()` — Vol Spike Detection

**Research:** Bollerslev (1986) GARCH; Barroso & Santa-Clara (2015) "Momentum has its moments" — vol_ratio > 1.5 is the empirical crash threshold.

**What it returns:** `"elevated"` / `"normal"` / `"compressed"` per bar.

```python
from strategies.alpha_composite import volatility_regime
vol_reg = volatility_regime(df["Close"], spike_mult=1.5, compress_mult=0.7)
```

| Value | Condition | Strategy action |
|---|---|---|
| `"elevated"` | 20d vol > 1.5× 252d vol | → CASH |
| `"normal"` | Between thresholds | Normal operation |
| `"compressed"` | 20d vol < 0.7× 252d vol | MR-friendly |

---

### `trend_regime()` — Multi-Source Trend Classifier

**Research:** Wilder (1978) ADX; Kaufman (1995) Efficiency Ratio; Lo (1991) Hurst exponent.

**What it returns:** `"trending"` / `"sideways"` / `"uncertain"` — requires 2 of 3 conditions.

```python
from strategies.alpha_composite import trend_regime
t_reg = trend_regime(df["Close"], df["High"], df["Low"],
                     adx_trend=25.0, adx_flat=20.0,
                     er_trend=0.55, er_flat=0.30, hurst_window=60)
```

---

### `RegimeClassifier` — Combined Regime Decision

**What it produces:** One of `REGIME_MOMENTUM`, `REGIME_MEAN_REVERSION`, `REGIME_CASH` per bar.

**Decision logic:**
- `vol = "elevated"` → **CASH** (override everything)
- `trend = "trending"` AND `HMM P(bull) > hmm_threshold` → **MOMENTUM**
- `trend = "sideways"` OR `vol = "compressed"` → **MEAN_REVERSION**
- Otherwise: use HMM alone to decide

**Markov smoothing:** `min_regime_bars` prevents rapid switching.

```python
from strategies.alpha_composite import RegimeClassifier

clf = RegimeClassifier(
    hmm_window      = 252,   # HMM training window
    hmm_threshold   = 0.55,  # bull-state probability threshold
    vol_spike_mult  = 1.5,   # vol ratio above this → CASH
    adx_trend_min   = 25.0,
    min_regime_bars = 5,     # smoothing to reduce whipsaw
)
regimes = clf.classify(df)
print(regimes.value_counts())
```

---

### `classify_regime()` — One-call Convenience

```python
from strategies.alpha_composite import classify_regime
regimes = classify_regime(df)   # uses all defaults
print(regimes.tail(5))
```

---

## Section 2: AlphaCompositeMomentumStrategy

**11 signals, all auto-weight-normalised. HIGH score = strong uptrend = BUY.**

### Default parameters

```python
from strategies.alpha_composite import DEFAULT_PARAMS
```

Key params: `entry_threshold=0.60`, `exit_threshold=0.35`, `atr_stop_mult=2.5`.  
All 11 `w_*` weight params auto-normalise — the GA tunes them as raw influence scores.

### Signal components

| # | Method | Research | Default weight |
|---|---|---|---|
| 1 | `_score_trend_stage()` | Weinstein (1988); Wilder (1978) ADX | 0.20 |
| 2 | `_score_tsmom()` | Moskowitz, Ooi & Pedersen (2012); Novy-Marx (2012) | 0.18 |
| 3 | `_score_linear_trend()` | Baltas & Kosowski (2012); Barroso & SC (2015) | 0.15 |
| 4 | `_score_52high()` | George & Hwang (2004) | 0.10 |
| 5 | `_score_macd_quality()` | Appel (1979); Blau (1991) TSI; Aspray (1986) | 0.10 |
| 6 | `_score_rsi_zone()` | Wilder (1978); Cardwell (1994); Chande & Kroll (1994) | 0.08 |
| 7 | `_score_kst()` | Pring (1992) | 0.07 |
| 8 | `_score_volume()` | Granville (1963); Blume, Easley & O'Hara (1994) | 0.07 |
| 9 | `_score_ichimoku()` | Hosoda (1969); Murphy (1999) | 0.05 |
| 10 | `_score_bollinger()` | Bollinger (1983/2002) | 0.07 |
| 11 | `_score_mfi()` | Achelis (2001); Cardwell (1994) | 0.07 |
| — | `_volatility_regime_factor()` | Daniel & Moskowitz (2016) | multiplicative damper |

### Signal interpretation guide

| Signal | Score ≈ 1.0 | Score ≈ 0.5 | Score ≈ 0.0 |
|---|---|---|---|
| Trend Stage | Full Stage 2 (ADX>20, SMAs aligned) | 1-2 conditions met | Below 200 SMA |
| TSMOM | All 3 horizons positive | 1-2 horizons positive | All negative |
| Linear Trend | Strong upward OLS slope | Flat | Strong downward slope |
| 52-Week High | Near new high | Mid-range | Well below high |
| MACD Quality | All 4 conditions met | 2 of 4 | All bearish |
| RSI Zone | RSI 50–72, StochRSI K>D | Near zone edge | Overbought/oversold |
| KST | Above signal, positive | Near signal | Below signal |
| Volume | OBV trending + vol above avg | One condition | Both failing |
| Ichimoku | Price above cloud, Tenkan>Kijun | Above cloud only | Below cloud |
| Bollinger | %B ≈ 0.70 (momentum zone) | At midline | Below midline |
| MFI | 50–75 zone | Zone edge | <30 or >80 |

### Usage

```python
from strategies.alpha_composite import AlphaCompositeMomentumStrategy, TUNABLE_PARAM_SPACE
from strategies.test import run_backtest, BacktestConfig, load_price_data, print_full_report

df, src = load_price_data("AAPL", years=3)
result  = run_backtest(AlphaCompositeMomentumStrategy(), df, symbol="AAPL")
print_full_report(result, src)

# Inspect composite
df_out = AlphaCompositeMomentumStrategy().generate_signals(df.copy())
print(df_out[["Close", "_composite", "signal"]].tail(10))
```

### Genetic optimisation

```python
from strategies.alpha_composite import AlphaCompositeMomentumStrategy, TUNABLE_PARAM_SPACE
from strategies.test import GeneticOptimizer, GAConfig, print_optimization_report

opt = GeneticOptimizer(
    strategy_factory = AlphaCompositeMomentumStrategy,
    param_space      = TUNABLE_PARAM_SPACE,
    symbols          = ["AAPL", "MSFT", "SPY", "QQQ", "JPM", "NVDA", "AMD", "V", "XOM", "AMZN"],
    config           = GAConfig(population_size=30, n_generations=100, fitness_metric="robust"),
)
result = opt.run()
print_optimization_report(result)
```

`TUNABLE_PARAM_SPACE` tunes: SMA windows, ADX threshold, TSMOM lookbacks, linear trend window, MACD windows, RSI bounds, entry/exit thresholds, ATR stop, **all 11 component weights**.

---

## Section 3: AlphaMeanReversionStrategy

**11 signals, all auto-weight-normalised. HIGH score = strongly OVERSOLD = BUY.**

### Default parameters

```python
from strategies.alpha_composite import MR_DEFAULT_PARAMS
```

Key params: `entry_threshold=0.60`, `exit_threshold=0.35`, `atr_stop_mult=1.5`, `time_stop_bars=10`, `zscore_entry=-1.5`.

### Signal components

| # | Method | Research | Default weight |
|---|---|---|---|
| 1 | `_score_regime_gate()` | Wilder (1978) ADX; Kaufman (1995) ER; Lo (1991) Hurst | 0.20 |
| 2 | `_score_zscore()` | Poterba & Summers (1988); Lo & MacKinlay (1988) | 0.18 |
| 3 | `_score_crsi()` | Connors, Alvarez & Hayward (2009) | 0.12 |
| 4 | `_score_bollinger_extreme()` | Bollinger (2002); Lento & Gradojevic (2007) | 0.10 |
| 5 | `_score_stochastic_extreme()` | Lane (1954); Elder (1993) | 0.08 |
| 6 | `_score_cci_extreme()` | Lambert (1980); Pring (1991) | 0.08 |
| 7 | `_score_willr_extreme()` | Williams (1979) | 0.07 |
| 8 | `_score_er_low()` | Kaufman (1995) | 0.06 |
| 9 | `_score_vol_climax()` | Granville (1963); Wyckoff (1930); Blume et al. (1994) | 0.05 |
| 10 | `_score_divergence()` | Elder (1993); Cardwell (1994) | 0.04 |
| 11 | `_score_ou_gate()` | Avellaneda & Lee (2010) | 0.02 |

### Exit mechanisms (all active simultaneously)

| Exit | Trigger |
|---|---|
| Composite exit | `composite < exit_threshold` |
| Mean return | `z-score >= zscore_exit` (price back at mean) |
| ATR stop | `close < entry − atr_mult × ATR` |
| Time stop | `bars_held >= time_stop_bars` |

### Usage

```python
from strategies.alpha_composite import AlphaMeanReversionStrategy
from strategies.test import run_backtest

result = run_backtest(AlphaMeanReversionStrategy(), df, symbol="AAPL")
```

### MR diagnostic metrics

```python
from strategies.tools.mean_reversion_tools import avg_holding_period, mean_reversion_speed

avg_hold = avg_holding_period(result.trades)        # target < 10 days
autocorr = mean_reversion_speed(result.daily_returns)  # target < 0 (confirms MR)
```

---

## Section 4: UnifiedAlphaStrategy

**The top-level strategy. Use this for all live trading and comprehensive backtests.**

### Default parameters

```python
from strategies.alpha_composite import UNIFIED_DEFAULT_PARAMS
```

22 tunable parameters: 7 regime thresholds + 3 momentum sub-params + 5 MR sub-params + 7 others.

### What it adds vs individual strategies

| Feature | Momentum alone | MR alone | Unified |
|---|---|---|---|
| Trending markets | ✓ | ✗ (false signals) | ✓ via regime gate |
| Choppy markets | ✗ (whipsaw) | ✓ | ✓ via regime gate |
| Vol crashes | ✗ (momentum crashes) | ✗ | ✓ → CASH |
| Auto-adapts | ✗ | ✗ | ✓ |

### Usage

```python
from strategies.alpha_composite import UnifiedAlphaStrategy, UNIFIED_PARAM_SPACE
from strategies.test import run_backtest, BacktestConfig, load_price_data, print_full_report

df, src = load_price_data("AAPL", years=5)
result  = run_backtest(UnifiedAlphaStrategy(), df, BacktestConfig(), symbol="AAPL")
print_full_report(result, src)

# Inspect regimes + composites
df_out = UnifiedAlphaStrategy().generate_signals(df.copy())
print(df_out[["Close", "_regime", "_mom_composite", "_mr_composite", "signal"]].tail(10))
print(df_out["_regime"].value_counts())
```

### Output columns

| Column | Description |
|---|---|
| `signal` | 1=long, 0=flat |
| `_regime` | `MOMENTUM` / `MEAN_REVERSION` / `CASH` |
| `_mom_composite` | Momentum sub-strategy composite score [0,1] |
| `_mr_composite` | MR sub-strategy composite score [0,1] |

### Genetic optimisation

```python
from strategies.alpha_composite import UnifiedAlphaStrategy, UNIFIED_PARAM_SPACE
from strategies.test import GeneticOptimizer, GAConfig, print_optimization_report, save_result

opt = GeneticOptimizer(
    strategy_factory = UnifiedAlphaStrategy,
    param_space      = UNIFIED_PARAM_SPACE,
    symbols          = ["AAPL","MSFT","NVDA","SPY","QQQ","JPM","XOM","AMD","AMZN","V"],
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
save_result(result, "momentum/test/runs/unified_opt.json")
```

---

## Complete Command Reference

```bash
# ── Run strategies ───────────────────────────────────────────────────────────
python3.11 strategies/alpha_composite.py                    # Unified on AAPL 5y
python3.11 strategies/alpha_composite.py --symbol NVDA      # Different symbol
python3.11 strategies/backtest/backtest_unified.py --universe all --years 3
python3.11 strategies/backtest/backtest_unified.py --symbol AMD --years 7

# ── Optimise ─────────────────────────────────────────────────────────────────
python3.11 strategies/optimize/optimize_unified.py --fast    # ~5 min
python3.11 strategies/optimize/optimize_unified.py           # ~60-90 min
python3.11 strategies/optimize/optimize_unified.py --target AAPL --train-years 5 --val-years 2

# ── Daily live trading ────────────────────────────────────────────────────────
python3.11 daily_signal.py --universe all --dry-run          # test signals
python3.11 daily_signal.py --universe all                    # paper (port 7497)
python3.11 daily_signal.py --universe all --live             # live (port 7496)
```

---

## Research Bibliography

| Paper | Used in |
|---|---|
| Hamilton (1989) "A New Approach to the Economic Analysis..." | GaussianHMM |
| Ang & Bekaert (2002) "Regime Switches in Interest Rates" | HMM validation |
| Nystrup, Madsen & Lindström (2017) "Long Horizon Forecasting..." | Rolling HMM |
| Bollerslev (1986) GARCH | volatility_regime |
| Barroso & Santa-Clara (2015) "Momentum has its moments" | vol crash gate |
| Daniel & Moskowitz (2016) "Momentum crashes" | vol damper, CASH gate |
| Wilder (1978) "New Concepts in Technical Trading Systems" | ADX, ATR, RSI |
| Kaufman (1995) "Smarter Trading" | Efficiency Ratio, KAMA |
| Lo (1991) "Long-Term Memory in Stock Market Prices" | Hurst exponent |
| Weinstein (1988) "Secrets for Profiting in Bull and Bear Markets" | Stage Analysis |
| Moskowitz, Ooi & Pedersen (2012) "Time Series Momentum" | TSMOM (Sharpe 1.31) |
| Novy-Marx (2012) "Is Momentum Really Momentum?" | Intermediate momentum weighting |
| Baltas & Kosowski (2012) "Improving TSMOM Strategies" | Linear trend regression |
| George & Hwang (2004) "The 52-week high and momentum investing" | 52-week high proximity |
| Pring (1992) "Martin Pring on Market Momentum" | KST oscillator |
| Granville (1963) "Granville's New Key to Stock Market Profits" | OBV |
| Hosoda (1969) Ichimoku Kinkō Hyō | Ichimoku cloud |
| Blau (1991) TSI | True Strength Index |
| Appel (1979) MACD | MACD quality gate |
| Poterba & Summers (1988) "Mean Reversion in Stock Prices" | Price z-score |
| Lo & MacKinlay (1988) "Stock Market Prices Do Not Follow Random Walks" | Z-score threshold |
| Connors, Alvarez & Hayward (2009) "High Probability ETF Trading" | ConnorsRSI |
| Bollinger (2002) "Bollinger on Bollinger Bands" | Bollinger extreme |
| Lane (1954) Stochastic Oscillator | Stochastic extreme |
| Lambert (1980) Commodity Channel Index | CCI extreme |
| Williams (1979) "How I Made One Million Dollars..." | Williams %R |
| Wyckoff (1930) accumulation/distribution | Volume climax |
| Elder (1993) "Trading For A Living" | RSI/MACD divergence |
| Avellaneda & Lee (2010) "Statistical Arbitrage in US Equities" | OU half-life |
