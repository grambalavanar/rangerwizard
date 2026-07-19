# Alpha Composite Momentum Strategy

The most rigorous single-instrument momentum strategy in this codebase.
Combines **nine independently peer-reviewed signals** from academic finance and
practitioner research into a single weighted composite score. Trades when the
score exceeds an entry threshold and exits when it falls below an exit threshold
or when an ATR trailing stop is hit.

**File:** [`momentum/strategies/alpha_composite.py`](alpha_composite.py)

---

## Research Foundations

Every component is directly grounded in published academic or practitioner research:

| # | Component | Source | Sharpe (reported) |
|---|---|---|---|
| 1 | Weinstein Stage Analysis | Weinstein (1988) | n/a (practitioner) |
| 2 | Time Series Momentum (TSMOM) | Moskowitz, Ooi & Pedersen (2012) | **1.31** |
| 3 | Linear Trend Regression | Baltas & Kosowski (2012) | > TSMOM |
| 4 | 52-Week High Proximity | George & Hwang (2004) | > standard momentum |
| 5 | Volatility Scaling (crash protection) | Barroso & Santa-Clara (2015) | +40% improvement |
| 6 | RSI + StochRSI Zone | Wilder (1978); Chande & Kroll (1994) | confirmation |
| 7 | MACD + TSI Quality | Appel (1979); Blau (1991) | confirmation |
| 8 | KST Oscillator | Pring (1992) | cycle identification |
| 9 | Volume Confirmation (OBV) | Granville (1963) | confirmation |
| 10 | Ichimoku Cloud | Hosoda (1969) | multi-timeframe |
| — | Momentum Crash Protection | Daniel & Moskowitz (2016) | damper |

**Key insight from the research:** Volatility scaling (Barroso & Santa-Clara 2015) and
the linear trend regression signal (Baltas & Kosowski 2012) are the two most
impactful improvements over plain 12-month momentum. Both are implemented here.

---

## Quick Start

```bash
# Run the built-in AAPL backtest demo:
python3.11 momentum/strategies/alpha_composite.py

# Backtest from your own script:
python3.11 -c "
from momentum.strategies.alpha_composite import AlphaCompositeMomentumStrategy
from momentum.test.strategy_tester import run_backtest, BacktestConfig
from momentum.test.run_backtest_example import load_price_data, print_full_report

df, src = load_price_data('AAPL', years=3)
result  = run_backtest(AlphaCompositeMomentumStrategy(), df, symbol='AAPL')
print_full_report(result, src)
"
```

---

## Table of Contents

| Symbol | Type | Description |
|---|---|---|
| [`DEFAULT_PARAMS`](#default_params--all-strategy-parameters) | `dict` | All 40+ default parameter values |
| [`_norm01()`](#_norm01--rolling-normalizer) | Function | Normalize any series to [0, 1] |
| [`AlphaCompositeMomentumStrategy`](#alphacompositemomentumsrategy--the-strategy-class) | Strategy class | Main strategy |
| [`_score_trend_stage()`](#_score_trend_stage--weinstein-stage-filter) | Method | Weinstein trend stage score |
| [`_score_tsmom()`](#_score_tsmom--time-series-momentum) | Method | Multi-horizon TSMOM score |
| [`_score_linear_trend()`](#_score_linear_trend--linear-trend-regression) | Method | OLS slope regression score |
| [`_score_52high()`](#_score_52high--52-week-high-proximity) | Method | 52-week high proximity |
| [`_score_macd_quality()`](#_score_macd_quality--macd--tsi-quality) | Method | MACD + TSI quality gates |
| [`_score_rsi_zone()`](#_score_rsi_zone--rsi-momentum-zone) | Method | RSI zone + StochRSI direction |
| [`_score_kst()`](#_score_kst--kst-oscillator) | Method | KST cycle oscillator |
| [`_score_volume()`](#_score_volume--volume-confirmation) | Method | OBV + volume ratio |
| [`_score_ichimoku()`](#_score_ichimoku--ichimoku-cloud) | Method | Ichimoku cloud score |
| [`_volatility_regime_factor()`](#_volatility_regime_factor--crash-protection) | Method | Momentum crash damper |
| [`_atr_stop_series()`](#_atr_stop_series--atr-trailing-stop) | Method | ATR trailing stop prices |
| [`_compute_composite()`](#_compute_composite--weighted-composite-score) | Method | Weighted composite score |
| [`generate_signals()`](#generate_signals--entry--exit-signals) | Method | Main signal column |
| [`TUNABLE_PARAM_SPACE`](#tunable_param_space--genetic-optimizer-integration) | `ParameterSpace` | Ready for GeneticOptimizer |

---

## Component Weights

Default weighting of the nine signals (sums to 1.0):

```
Weinstein Trend Stage   ████████████████████  0.20   ← most important gate
TSMOM Multi-horizon     ██████████████████    0.18   ← primary momentum signal
Linear Trend Regression ███████████████       0.15   ← best academic signal
52-Week High Proximity  ██████████            0.10
MACD + TSI Quality      ██████████            0.10
RSI Momentum Zone       ████████              0.08
KST Cycle Oscillator    ███████               0.07
Volume Confirmation     ███████               0.07
Ichimoku Cloud          █████                 0.05
```

The volatility regime factor is a **multiplicative damper** (not an additive weight), applied after the composite is computed. It reduces the score during high-volatility crash regimes without being counted in the weight sum.

---

## `DEFAULT_PARAMS` — All Strategy Parameters

[→ Source](alpha_composite.py#L58)

All 40+ default parameter values. Pass a partial dict to `AlphaCompositeMomentumStrategy(params={...})` to override any subset.

```python
from momentum.strategies.alpha_composite import DEFAULT_PARAMS
print(DEFAULT_PARAMS)
```

**Key parameters to know:**

| Parameter | Default | What it controls |
|---|---|---|
| `sma_long` | 200 | Long-term trend SMA window |
| `entry_threshold` | 0.60 | Composite score to enter long |
| `exit_threshold` | 0.35 | Composite score to exit long |
| `atr_stop_mult` | 2.5 | ATR trailing stop distance |
| `tsmom_long` | 252 | 12-month TSMOM lookback |
| `tsmom_long_skip` | 21 | Days to skip (reversal avoidance) |

---

## `_norm01()` — Rolling Normalizer

[→ Source](alpha_composite.py#L131)

**What it does:** Converts any raw indicator series to a [0, 1] scale using rolling z-score normalization. 0 = historically bearish extreme, 0.5 = neutral, 1 = historically bullish extreme.

**Used by:** Every sub-score method internally. Also useful externally when you want to normalize custom indicators before combining them with the composite.

```python
from momentum.strategies.alpha_composite import _norm01

# Normalize a 20-day return series
ret20 = df["Close"].pct_change(20)
score = _norm01(ret20, window=252)
print(score.tail())  # values between 0 and 1
```

---

## `AlphaCompositeMomentumStrategy` — The Strategy Class

[→ Source](alpha_composite.py#L168)

Subclass of `Strategy` from `strategy_tester.py`. Works in all backtesting, paper trading, and genetic optimization workflows.

```python
from momentum.strategies.alpha_composite import AlphaCompositeMomentumStrategy

# Default parameters
strategy = AlphaCompositeMomentumStrategy()

# Custom parameters (override any subset)
strategy = AlphaCompositeMomentumStrategy(params={
    "entry_threshold": 0.65,
    "exit_threshold":  0.30,
    "atr_stop_mult":   2.0,
    "sma_long":        200,
})

# With optimised parameters from genetic optimizer
strategy = AlphaCompositeMomentumStrategy(params=opt_result.best_params)
```

---

## `_score_trend_stage()` — Weinstein Stage Filter

[→ Source](alpha_composite.py#L247)

**What it measures:** Three Weinstein Stage 2 conditions: price above SMA(200), upward SMA alignment (50 > 150 > 200), and ADX above threshold. Weight: **0.20** (highest).

**Research:** Stan Weinstein (1988) "Secrets for Profiting in Bull and Bear Markets." Stage 2 = rising 30-week SMA with price above it — the highest-probability long entry phase.

**When you need this:** If the strategy rarely enters during what looks like a strong uptrend, check if the SMA alignment condition is failing. The 50 > 150 > 200 requirement is strict — a stock that recently exited a downtrend may have the price above SMA(200) but the SMAs not yet aligned.

| Score | Meaning |
|---|---|
| 1.0 | Full Stage 2: price above SMA, aligned SMAs, strong ADX |
| 0.67 | Two of three conditions met |
| 0.33 | One condition met |
| 0.0 | Stock in downtrend (Stage 3/4) |

---

## `_score_tsmom()` — Time Series Momentum

[→ Source](alpha_composite.py#L307)

**What it measures:** Sign of excess returns over three lookback windows (12-month, 6-month, 3-month), each skipping the most recent 21 days (reversal avoidance). Weighted average with more weight on longer horizons (Novy-Marx 2012). Weight: **0.18**.

**Research:**
- Moskowitz, Ooi & Pedersen (2012) "Time Series Momentum" — Sharpe 1.31, works across all asset classes
- Novy-Marx (2012) "Is Momentum Really Momentum?" — 7-12 month returns are stronger predictors
- Jegadeesh & Titman (1993) — 21-day skip avoids short-term reversal

**When you need this:** This is the core momentum signal. If TSMOM is high but the composite is below the entry threshold, one of the other gates is blocking entry. This is usually the trend stage or volume score. TSMOM rarely needs tuning beyond adjusting the window lengths.

| Score | Meaning |
|---|---|
| 1.0 | All three time horizons show positive returns |
| 0.67 | Two of three positive |
| 0.33 | One of three positive |
| 0.0 | All three negative |

---

## `_score_linear_trend()` — Linear Trend Regression

[→ Source](alpha_composite.py#L367)

**What it measures:** OLS regression slope of log-price over a rolling window, normalised by realised volatility. This is the academically superior version of the momentum signal. Weight: **0.15**.

**Research:**
- Baltas & Kosowski (2012) showed this signal outperforms the plain sign-of-return signal used in standard TSMOM
- It is mathematically equivalent to the "trend-following signal" used by Man AHL, Winton, and other major CTAs
- Barroso & Santa-Clara (2015) supply the volatility normalisation

**When you need this:** Use alongside TSMOM. A high TSMOM + low linear trend = recent price surge (not a sustained linear trend). A low TSMOM + rising linear trend = early development of a new trend. The linear trend is more sensitive to the current slope while TSMOM is more sensitive to total past return.

```python
# Inspect the linear trend score for AAPL
strategy = AlphaCompositeMomentumStrategy()
s = strategy._score_linear_trend(df, strategy.params)
print(s.tail())
```

---

## `_score_52high()` — 52-Week High Proximity

[→ Source](alpha_composite.py#L432)

**What it measures:** Current close as a fraction of the 52-week rolling high. Values near 1.0 indicate the stock is at or near its 52-week high, which statistically predicts continuation. Weight: **0.10**.

**Research:** George & Hwang (2004) "The 52-week high and momentum investing" *Journal of Finance* — The proximity ratio has higher predictive power for future 6-month returns than the standard Jegadeesh-Titman 12-1 month return measure. Mechanism: investor anchoring (sellers are reluctant to sell at prices exceeding their mental "high" anchor).

**When you need this:** A stock making new 52-week highs with score near 1.0, combined with strong TSMOM and linear trend = maximum conviction entry. Think of it as a breakout confirmation signal.

---

## `_score_macd_quality()` — MACD + TSI Quality

[→ Source](alpha_composite.py#L482)

**What it measures:** Four binary momentum quality gates: MACD line > signal, histogram ≥ 0, histogram accelerating (growing), TSI > 0. Fraction of conditions met. Weight: **0.10**.

**Research:**
- MACD: Appel (1979)
- Histogram acceleration: Aspray (1986) as early warning of momentum shifts
- TSI: Blau (1991) — double-smoothed momentum, less noisy than RSI

**When you need this:** Use as a confirmation gate for entry. High composite + low MACD quality = momentum may be in a temporary pullback within the larger uptrend (potential high-quality entry) OR early stages of a reversal. Check MACD histogram direction to distinguish.

---

## `_score_rsi_zone()` — RSI Momentum Zone

[→ Source](alpha_composite.py#L542)

**What it measures:** RSI in the "momentum zone" [50–72] scores 1.0; RSI outside the zone scores decrease. Combined with StochRSI %K > %D directional confirmation. Weight: **0.08**.

**Research:**
- Wilder (1978) original RSI, 70/30 thresholds
- Cardwell (1994): In trending markets, RSI 40–80 is the bull range; 70 is not overbought in uptrends
- Chande & Kroll (1994) — StochRSI for faster directional sensitivity

**When you need this:** RSI zone score being low (RSI > 72) combined with other strong signals = possible overbought condition. Consider waiting for a pullback before entering. RSI below 50 in a supposed uptrend = trend may be failing.

---

## `_score_kst()` — KST Oscillator

[→ Source](alpha_composite.py#L601)

**What it measures:** KST above its signal line (50% weight) + KST normalised strength (50% weight). KST positive = bull market cycle phase. Weight: **0.07**.

**Research:** Pring (1992) "Martin Pring on Market Momentum." The KST was designed specifically to identify primary bull/bear market cycle phases by weighting longer ROC components more heavily.

**When you need this:** KST is the slowest component — use it to check the macro cycle. If KST has been below its signal line for many months, momentum entries carry elevated drawdown risk (you may be fighting a bear market). The KST crossing above its signal line is a powerful confirmation of a new bull cycle beginning.

---

## `_score_volume()` — Volume Confirmation

[→ Source](alpha_composite.py#L648)

**What it measures:** OBV fast EMA > OBV slow EMA (institutional accumulation) and volume ratio above 1.0 (above-average recent activity). Weight: **0.07**.

**Research:**
- Granville (1963) On-Balance Volume
- Blume, Easley & O'Hara (1994) "Market statistics and technical analysis: The role of volume" — volume carries information about the informativeness of price signals
- O'Neil (2002) CAN SLIM system: volume >40% above average on breakout days is required for confirmation

**When you need this:** Low volume score during a price momentum entry signal = potential false breakout. Institutional money is not participating. Use as a critical filter especially for breakout entries near 52-week highs.

---

## `_score_ichimoku()` — Ichimoku Cloud

[→ Source](alpha_composite.py#L700)

**What it measures:** Three Ichimoku conditions: price above Senkou Span A, price above Senkou Span B, Tenkan-sen > Kijun-sen. Weight: **0.05**.

**Research:** Hosoda (1969) / Goichi Hosoda's original 1969 work "一目均衡表" (Ichimoku Kinko Hyo). Murphy (1999) "Technical Analysis of the Financial Markets" — Chapter 14 extensively covers Ichimoku's multi-timeframe effectiveness.

**When you need this:** A price breaking above the full Ichimoku cloud (score → 1.0) after a period below it is one of the most reliable trend reversal signals in technical analysis. Particularly useful as a leading indicator before a new Stage 2 begins.

---

## `_volatility_regime_factor()` — Crash Protection

[→ Source](alpha_composite.py#L750)

**What it measures:** A multiplicative damper in [0.30, 1.0] that reduces the composite score when ATR has spiked to more than `atr_spike_mult` × its 20-day average. This targets momentum crash conditions directly.

**Research:** Daniel & Moskowitz (2016) "Momentum crashes" *Journal of Financial Economics* — momentum strategies suffer large tail losses during equity market rebounds following high-volatility bear markets. Barroso & Santa-Clara (2015) showed constant volatility scaling captures most of this protection.

**When you need this:** During events like the March 2020 COVID crash, October 2022 bear market, or any period when intraday ranges spike dramatically. This factor automatically reduces the effective composite score, preventing entries into dangerous high-volatility momentum chases.

> **This is not in the component weights** — it is a multiplicative damper applied after all nine scores are combined. You cannot tune it away with lower weights; it is always active.

---

## `_atr_stop_series()` — ATR Trailing Stop

[→ Source](alpha_composite.py#L797)

**What it measures:** For each bar, computes a trailing stop price: `highest_close_since_entry - atr_stop_mult × ATR`. The stop only moves up, never down.

**Research:** Wilder (1978) ATR; LeBeau & Lucas (1992) "Computer Analysis of the Futures Market" — ATR trailing stop is the standard exit method for systematic trend-following CTAs.

**When you need this:** The trailing stop is the primary risk control. If a position moves in your favour and then reverses, the stop captures most of the profit. Tune `atr_stop_mult` in the genetic optimizer: lower (1.5–2.0) = tighter stops, lower drawdown but more whipsaw; higher (3.0+) = wider stops, larger drawdown but more trend-following profits.

```python
# Inspect where the stop would be right now
strategy = AlphaCompositeMomentumStrategy()
stops = strategy._atr_stop_series(df, strategy.params)
print(f"Current price: ${df['Close'].iloc[-1]:.2f}")
print(f"Current stop : ${stops.iloc[-1]:.2f}")
```

---

## `_compute_composite()` — Weighted Composite Score

[→ Source](alpha_composite.py#L846)

**What it does:** Computes the weighted average of all nine sub-scores, then multiplies by the volatility regime factor. Returns a single [0, 1] score per bar.

**When you need this:** Inspect the composite directly to diagnose strategy behaviour:

```python
strategy = AlphaCompositeMomentumStrategy()
df_out = strategy.generate_signals(df.copy())
print(df_out[["Close", "_composite", "signal"]].tail(10))

# Current composite vs thresholds
latest = df_out["_composite"].iloc[-1]
p = strategy.params
print(f"Composite: {latest:.3f}  Entry: {p['entry_threshold']}  Exit: {p['exit_threshold']}")
```

---

## `generate_signals()` — Entry / Exit Signals

[→ Source](alpha_composite.py#L893)

**What it does:** The main entry point. Runs all sub-scores, computes the composite, applies the trailing stop, and generates a `signal` column (1 = long, 0 = flat).

**Entry logic:**
- `composite >= entry_threshold` → signal = 1 (enter long)

**Exit logic:**
- `composite < exit_threshold` → signal = 0 (exit)
- `close < atr_trailing_stop` → signal = 0 (stop hit)

```python
strategy = AlphaCompositeMomentumStrategy()
df_out   = strategy.generate_signals(df.copy())

# Current signal
signal = df_out["signal"].iloc[-1]
comp   = df_out["_composite"].iloc[-1]
print(f"Signal: {'LONG' if signal == 1 else 'FLAT'} | Composite: {comp:.3f}")
```

---

## `TUNABLE_PARAM_SPACE` — Genetic Optimizer Integration

[→ Source](alpha_composite.py#L942)

A ready-built `ParameterSpace` with 14 high-impact parameters and 4 validity constraints. Drop directly into `GeneticOptimizer`:

```python
from momentum.strategies.alpha_composite import (
    AlphaCompositeMomentumStrategy,
    TUNABLE_PARAM_SPACE,
)
from momentum.test.genetic_optimizer import GeneticOptimizer, GAConfig, print_optimization_report

opt = GeneticOptimizer(
    strategy_factory = AlphaCompositeMomentumStrategy,
    param_space      = TUNABLE_PARAM_SPACE,
    symbols          = ["AAPL", "MSFT", "SPY", "QQQ", "JPM", "XOM"],
    config           = GAConfig(
        population_size     = 30,
        n_generations       = 25,
        fitness_metric      = "robust",
        consistency_penalty = 0.30,
        min_trades          = 5,
        n_workers           = 4,
    ),
)
result = opt.run()
print_optimization_report(result)

# Save result
from momentum.test.genetic_optimizer import save_result
save_result(result, "momentum/test/runs/alpha_composite_opt.json")

# Use best params
best_strategy = AlphaCompositeMomentumStrategy(result.best_params)
```

**Tunable parameters:**

| Parameter | Range | Why tuneable |
|---|---|---|
| `sma_long` | 150–250 | Trend lookback sensitivity |
| `sma_short` | 30–70 | Short-term trend sensitivity |
| `adx_min` | 15–35 | Trend strength requirement |
| `tsmom_long` | 200–280 | Primary momentum lookback |
| `tsmom_long_skip` | 10–30 | Short-term reversal skip period |
| `tsmom_short` | 40–90 | Short-term momentum lookback |
| `lintrend_window` | 60–130 | Regression window |
| `macd_fast` | 8–16 | MACD responsiveness |
| `macd_slow` | 20–36 | MACD trend window |
| `rsi_low` | 42–58 | Momentum zone lower bound |
| `rsi_high` | 65–82 | Overbought threshold |
| `entry_threshold` | 0.50–0.75 | Composite entry bar |
| `exit_threshold` | 0.20–0.50 | Composite exit bar |
| `atr_stop_mult` | 1.5–4.0 | Stop-loss tightness |

---

## Full Workflow Example

```python
# 1. Backtest with defaults
from momentum.strategies.alpha_composite import AlphaCompositeMomentumStrategy
from momentum.test.strategy_tester import run_backtest, BacktestConfig
from momentum.test.run_backtest_example import load_price_data, print_full_report

df, src = load_price_data("AAPL", years=5)
cfg     = BacktestConfig(initial_capital=100_000, position_sizing="atr")
result  = run_backtest(AlphaCompositeMomentumStrategy(), df, cfg, symbol="AAPL")
print_full_report(result, src)

# 2. Compare against simpler strategy
from momentum.test.strategy_tester import MACDMomentumStrategy, run_comparison, compare_results
results = run_comparison(
    {"Alpha Composite": AlphaCompositeMomentumStrategy(), "MACD Baseline": MACDMomentumStrategy()},
    df, cfg, symbol="AAPL"
)
print(compare_results(results).to_string())

# 3. Optimise with genetic algorithm
from momentum.strategies.alpha_composite import TUNABLE_PARAM_SPACE
from momentum.test.genetic_optimizer import GeneticOptimizer, GAConfig, print_optimization_report

opt = GeneticOptimizer(
    strategy_factory = AlphaCompositeMomentumStrategy,
    param_space      = TUNABLE_PARAM_SPACE,
    symbols          = ["AAPL", "MSFT", "SPY", "QQQ", "JPM"],
    config           = GAConfig(population_size=30, n_generations=25, fitness_metric="robust"),
)
opt_result = opt.run()
print_optimization_report(opt_result)

# 4. Backtest with optimised params
best = AlphaCompositeMomentumStrategy(opt_result.best_params)
result_opt = run_backtest(best, df, cfg, symbol="AAPL")
print_full_report(result_opt, src)

# 5. Deploy to paper trader
# In paper_config.json:
# {
#   "strategy_module": "momentum.strategies.alpha_composite.AlphaCompositeMomentumStrategy",
#   "symbols": ["AAPL"],
#   "tws_port": 7497,
#   ...
# }
```

---

## Overfitting Warning

> With 14 tunable parameters, the genetic optimizer can overfit to the training window.
> **Always validate on a held-out period** before going live.

```python
# Train on years 1-3
df_train = df[df.index < "2025-01-01"]
opt = GeneticOptimizer(..., price_data={sym: df_train[sym] for sym in symbols})
result_opt = opt.run()

# Validate on year 4 (out-of-sample)
df_val = df[df.index >= "2025-01-01"]
strategy = AlphaCompositeMomentumStrategy(result_opt.best_params)
result_val = run_backtest(strategy, df_val, cfg, symbol="AAPL")
print_full_report(result_val, "validation")

# If val Sharpe << train Sharpe → overfitting
# Increase consistency_penalty in GAConfig or add more diverse symbols
```

---

## Run Commands

```bash
# Standalone backtest demo (AAPL):
python3.11 momentum/strategies/alpha_composite.py

# Backtest any symbol:
python3.11 -c "
import sys; sys.path.insert(0, '.')
from momentum.strategies.alpha_composite import AlphaCompositeMomentumStrategy
from momentum.test.strategy_tester import run_backtest, print_result
from momentum.test.run_backtest_example import load_price_data
df, src = load_price_data('SPY', years=5)
r = run_backtest(AlphaCompositeMomentumStrategy(), df, symbol='SPY')
print_result(r)
"

# Run genetic optimization (fast test):
python3.11 -c "
import sys; sys.path.insert(0, '.')
from momentum.strategies.alpha_composite import AlphaCompositeMomentumStrategy, TUNABLE_PARAM_SPACE
from momentum.test.genetic_optimizer import GeneticOptimizer, GAConfig, print_optimization_report
opt = GeneticOptimizer(
    AlphaCompositeMomentumStrategy, TUNABLE_PARAM_SPACE,
    ['AAPL','SPY'],
    GAConfig(population_size=10, n_generations=5, verbose=True)
)
r = opt.run()
print_optimization_report(r)
"
```
