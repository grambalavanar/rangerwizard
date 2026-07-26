# Unified Alpha Strategy — Complete Reference

The `strategies/` package combines momentum and mean reversion into a single regime-aware system. A **Gaussian Hidden Markov Model** (Hamilton 1989) combined with volatility and trend filters automatically deploys the right sub-strategy for each market condition.

---

## Architecture

```
                    market data (OHLCV)
                           │
              ┌────────────▼────────────┐
              │     RegimeClassifier     │
              │                          │
              │  1. Gaussian HMM (2-state)│  Hamilton (1989)
              │  2. Volatility ratio      │  Bollerslev (1986)
              │  3. ADX + ER + Hurst gate │  Wilder / Kaufman / Lo
              └──────────┬───────────────┘
                         │
           ┌─────────────┼─────────────┐
           ▼             ▼             ▼
       MOMENTUM    MEAN_REVERSION    CASH
           │             │             │
  AlphaComposite   AlphaMeanRev    signal=0
  Momentum(11)     Strategy(11)
  signals          signals
           │             │
           └──────┬──────┘
                  ▼
         unified signal column
         + _regime inspection column
```

---

## Directory Structure

```
strategies/
    __init__.py                  ← unified public API
    unified_alpha.py             ← UnifiedAlphaStrategy
    tools/
        __init__.py              ← re-exports all indicators
        regime_tools.py          ← GaussianHMM + RegimeClassifier
    test/
        __init__.py              ← re-exports strategy_tester, GA, etc.
    optimize/
        optimize_unified.py      ← GA optimizer for UnifiedAlphaStrategy
    backtest/
        backtest_unified.py      ← backtest + regime breakdown report
```

Existing packages remain accessible and unchanged:
- `momentum/` — AlphaCompositeMomentumStrategy (11 signals)
- `mean_reversion/` — AlphaMeanReversionStrategy (11 signals)

---

## Quick Start

```bash
# Backtest with default params (no optimization needed):
python3.11 strategies/unified_alpha.py

# Backtest all universes with optimised params:
python3.11 strategies/backtest/backtest_unified.py --universe all --years 3

# Optimize (fast smoke test ~2 min):
python3.11 strategies/optimize/optimize_unified.py --fast

# Optimize (full run ~60–90 min, 10-stock diverse basket):
python3.11 strategies/optimize/optimize_unified.py

# Security-specific optimization with walk-forward:
python3.11 strategies/optimize/optimize_unified.py --target AAPL --train-years 5 --val-years 2
```

---

## Regime Classifier

### `GaussianHMM` — 2-State Hidden Markov Model

[→ Source](tools/regime_tools.py#L60)

**Research:** Hamilton (1989); Ang & Bekaert (2002); Turner, Startz & Nelson (1989)

**What it does:** Fits a 2-state Gaussian HMM to daily log-returns using the Baum-Welch EM algorithm. State 0 = low-volatility/positive-drift (bull/momentum); State 1 = high-volatility/negative-drift (bear/choppy). The bull-state posterior probability is used as the primary HMM regime signal.

**Key properties after fitting:**

| Property | What to check |
|---|---|
| `hmm.state_means[0]` | Should be positive (bull mean return) |
| `hmm.state_vols[0]` | Should be lower (bull = less volatile) |
| `hmm.transition_matrix[0,0]` | Should be > 0.92 (regimes are persistent) |

```python
from strategies.tools.regime_tools import GaussianHMM
import numpy as np

returns = np.log(df["Close"]).diff().dropna().values
hmm = GaussianHMM(n_states=2).fit(returns.reshape(-1, 1))
proba = hmm.predict_proba(returns.reshape(-1, 1))
bull_prob = proba[:, hmm.bull_state]

print(f"Bull state mean daily return : {hmm.state_means[0]*252:.1%}/yr")
print(f"Bull state annualised vol    : {hmm.state_vols[0]:.1%}")
print(f"Regime persistence (p00)     : {hmm.transition_matrix[0,0]:.3f}")
```

---

### `volatility_regime()` — Volatility Spike Detection

[→ Source](tools/regime_tools.py#L245)

**Research:** Bollerslev (1986) GARCH; Barroso & Santa-Clara (2015) "Momentum has its moments"; Daniel & Moskowitz (2016) "Momentum Crashes"

**What it measures:** `vol_ratio = realized_vol(20d) / realized_vol(252d)`. Above `spike_mult` (default 1.5) → "elevated" → CASH signal. The 1.5× threshold is from Barroso & Santa-Clara (2015) — the single most impactful crash-protection finding in the momentum literature.

```python
from strategies.tools.regime_tools import volatility_regime
vol_reg = volatility_regime(df["Close"], spike_mult=1.5)
# "elevated" | "normal" | "compressed"
```

---

### `trend_regime()` — Multi-Source Trend Classifier

[→ Source](tools/regime_tools.py#L316)

**Research:** Wilder (1978) ADX; Kaufman (1995) ER; Lo (1991) Hurst exponent

**What it measures:** Requires 2 of 3 conditions to classify a regime:
- "trending": ADX > 25, ER > 0.55, Hurst > 0.50
- "sideways": ADX < 20, ER < 0.30, Hurst < 0.50
- "uncertain": mixed signals

```python
from strategies.tools.regime_tools import trend_regime
t_reg = trend_regime(df["Close"], df["High"], df["Low"])
# "trending" | "sideways" | "uncertain"
```

---

### `RegimeClassifier` — Combined Regime Decision

[→ Source](tools/regime_tools.py#L395)

**What it produces:** One of three labels per bar:

| Label | Condition | Action |
|---|---|---|
| `REGIME_MOMENTUM` | trend="trending" AND HMM P(bull) > 0.55 AND vol="normal" | Run AlphaCompositeMomentumStrategy |
| `REGIME_MEAN_REVERSION` | trend="sideways" OR vol="compressed" | Run AlphaMeanReversionStrategy |
| `REGIME_CASH` | vol="elevated" (spike_mult exceeded) | Hold cash |

**Markov smoothing:** `min_regime_bars` (default 5) prevents rapid regime switching — a regime must persist for N bars before it's accepted. This reduces turnover without sacrificing regime detection quality.

```python
from strategies.tools.regime_tools import RegimeClassifier

clf = RegimeClassifier(
    hmm_window      = 252,   # training window
    hmm_threshold   = 0.55,  # bull-state probability threshold
    vol_spike_mult  = 1.5,   # vol ratio above this → CASH
    adx_trend_min   = 25.0,  # ADX for trending regime
    min_regime_bars = 5,     # smoothing
)
regimes = clf.classify(df)
print(regimes.value_counts())
```

---

## `UnifiedAlphaStrategy` — The Strategy

[→ Source](unified_alpha.py)

**22 tunable parameters** across three layers:
1. **Regime thresholds** (7 params) — when to switch between regimes
2. **Momentum sub-strategy** (3 params) — entry/exit/stop in trending markets  
3. **Mean reversion sub-strategy** (5 params) — entry/exit/stop/time/zscore in ranging markets

```python
from strategies import UnifiedAlphaStrategy

# Default parameters
strategy = UnifiedAlphaStrategy()

# With optimised parameters
import json
with open("momentum/test/runs/unified_opt.json") as f:
    opt_params = json.load(f)["best_params"]
strategy = UnifiedAlphaStrategy(params=opt_params)

# Backtest + inspect regimes
from strategies.test import run_backtest, BacktestConfig
result  = run_backtest(strategy, df, BacktestConfig(), symbol="AAPL")
df_out  = strategy.generate_signals(df.copy())
print(df_out["_regime"].value_counts())
```

**Columns added to DataFrame:**

| Column | Description |
|---|---|
| `signal` | 1=long, 0=flat |
| `_regime` | Current regime label |
| `_mom_composite` | Momentum sub-strategy composite score [0,1] |
| `_mr_composite` | Mean reversion sub-strategy composite score [0,1] |

---

## Genetic Optimisation

```bash
# Full basket optimisation (10 stocks, 100 generations):
python3.11 strategies/optimize/optimize_unified.py

# Fast test:
python3.11 strategies/optimize/optimize_unified.py --fast

# Security-specific with walk-forward validation:
python3.11 strategies/optimize/optimize_unified.py --target NVDA --train-years 5 --val-years 2

# After optimisation, validate on a different symbol:
python3.11 strategies/optimize/optimize_unified.py --validate SPY
```

Results saved to `momentum/test/runs/unified_opt.json`. Then run:

```bash
python3.11 strategies/backtest/backtest_unified.py --universe all
```

---

## Full Workflow

```python
# Complete workflow from scratch
from strategies import UnifiedAlphaStrategy, UNIFIED_PARAM_SPACE
from strategies.test import (
    GeneticOptimizer, GAConfig, print_optimization_report, save_result,
    run_backtest, BacktestConfig, load_price_data, print_full_report,
)

# 1. Load data
df, src = load_price_data("AAPL", years=5)

# 2. Quick backtest with defaults (no GA needed)
result = run_backtest(UnifiedAlphaStrategy(), df, symbol="AAPL")
print_full_report(result, src)

# 3. Optimize
opt = GeneticOptimizer(
    strategy_factory = UnifiedAlphaStrategy,
    param_space      = UNIFIED_PARAM_SPACE,
    symbols          = ["AAPL", "MSFT", "SPY", "QQQ", "JPM", "AMD", "AMZN", "V", "XOM", "NVDA"],
    config           = GAConfig(population_size=30, n_generations=100, fitness_metric="robust"),
)
opt_result = opt.run()
print_optimization_report(opt_result)
save_result(opt_result, "momentum/test/runs/unified_opt.json")

# 4. Backtest with optimised params
strategy_opt = UnifiedAlphaStrategy(params=opt_result.best_params)
result_opt   = run_backtest(strategy_opt, df, BacktestConfig(), symbol="AAPL")
print_full_report(result_opt, src)

# 5. Inspect regime distribution
df_out = strategy_opt.generate_signals(df.copy())
print(df_out["_regime"].value_counts())

# 6. Update daily_signal.py to use UnifiedAlphaStrategy:
# In paper_config.json or daily_signal.py:
#   strategy = UnifiedAlphaStrategy(params=opt_result.best_params)
```

---

## Research Summary

| Component | Paper | Key Contribution |
|---|---|---|
| HMM regime model | Hamilton (1989) | Markov regime-switching framework |
| HMM validation | Ang & Bekaert (2002) | 2-state Gaussian HMM for equities |
| HMM rolling | Nystrup et al. (2017) | Adaptive rolling HMM |
| Volatility crash gate | Barroso & Santa-Clara (2015) | vol_ratio > 1.5 → avoid momentum |
| Momentum crashes | Daniel & Moskowitz (2016) | Tail risk of momentum strategies |
| Trend strength | Wilder (1978) | ADX as regime classifier |
| Efficiency Ratio | Kaufman (1995) | ER < 0.25 = mean reversion regime |
| Anti-persistence | Lo (1991) | Hurst < 0.5 = mean-reverting |
| Core MR signal | Poterba & Summers (1988) | Z-score mean reversion |
| Short-term MR | Connors et al. (2009) | ConnorsRSI < 10 edge |
| Core momentum | Moskowitz et al. (2012) | TSMOM Sharpe 1.31 |
| Trend signal | Baltas & Kosowski (2012) | Linear trend regression |
