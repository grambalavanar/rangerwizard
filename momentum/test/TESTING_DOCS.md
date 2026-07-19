# Testing Infrastructure — Reference Documentation

Testing library and paper trading runner for momentum strategies.

| File | Purpose |
|---|---|
| [`strategy_tester.py`](strategy_tester.py) | Backtesting library — historical simulation + comparison |
| [`paper_trader.py`](paper_trader.py) | Schedulable paper-trading execution script |
| [`paper_config.json`](paper_config.json) | Example configuration file for paper_trader.py |
| [`run_backtest_example.py`](run_backtest_example.py) | Standalone script: run a backtest and print an ASCII report |
| [`genetic_optimizer.py`](genetic_optimizer.py) | Genetic algorithm: tune strategy parameters across a stock basket |

**Dependencies:** `numpy`, `pandas`, `scipy`, `ibapi`  
Optional: `yfinance` (for data), `matplotlib` (for `plot_equity_curve`)

```bash
pip install yfinance      # recommended data source
```

---

## Table of Contents

### strategy_tester.py
| Symbol | Type | Description |
|---|---|---|
| [`Strategy`](#strategy--abstract-base-class) | Abstract class | Base class for all strategies |
| [`BacktestConfig`](#backtestconfig--backtest-configuration) | Dataclass | Simulation parameters |
| [`BacktestResult`](#backtestresult--result-container) | Dataclass | All metrics + trades + equity curve |
| [`Trade`](#trade--single-trade-record) | Dataclass | One round-trip trade record |
| [`run_backtest()`](#run_backtest--run-a-historical-backtest) | Function | Backtest a single strategy |
| [`run_comparison()`](#run_comparison--compare-multiple-strategies) | Function | Backtest multiple strategies side by side |
| [`fetch_ibkr_history()`](#fetch_ibkr_history--fetch-historical-data-from-ibkr) | Function | Pull OHLCV data from IBKR TWS |
| [`print_result()`](#print_result--print-a-formatted-summary) | Function | Print a backtest result to terminal |
| [`compare_results()`](#compare_results--build-a-comparison-table) | Function | DataFrame comparing multiple results |
| [`plot_equity_curve()`](#plot_equity_curve--plot-the-equity-curve) | Function | Plot equity curve with drawdown |
| [`MACDMomentumStrategy`](#macdsmomentumstrategy--example-strategy) | Strategy | Bundled example strategy (MACD + ADX) |

### run_backtest_example.py
| Symbol | Type | Description |
|---|---|---|
| [`load_price_data()`](#load_price_data--load-historical-price-data) | Function | Load OHLCV data (yfinance / IBKR / CSV / synthetic) |
| [`ascii_line_chart()`](#ascii_line_chart--ascii-line-chart) | Function | Time-series line chart in ASCII |
| [`ascii_area_chart()`](#ascii_area_chart--ascii-area-chart) | Function | Downward-filled area chart (for drawdown) |
| [`ascii_histogram()`](#ascii_histogram--ascii-histogram) | Function | Horizontal bar histogram of trade returns |
| [`ascii_monthly_table()`](#ascii_monthly_table--monthly-returns-table) | Function | Year × month returns grid |
| [`ascii_trade_table()`](#ascii_trade_table--recent-trades-table) | Function | Last N trades as formatted table |
| [`print_full_report()`](#print_full_report--full-ascii-report) | Function | Print the complete report with all charts |
| [`main()`](#main--cli-entry-point-1) | Function | CLI entry point — run and report |

### genetic_optimizer.py
| Symbol | Type | Description |
|---|---|---|
| [`IntParam`](#intparam--integer-parameter) | Dataclass | Integer-valued parameter with bounds |
| [`FloatParam`](#floatparam--float-parameter) | Dataclass | Continuous float parameter with bounds |
| [`ChoiceParam`](#choiceparam--discrete-choice-parameter) | Dataclass | Discrete-choice parameter from a list |
| [`ParameterSpace`](#parameterspace--search-space) | Class | Holds param definitions + all GA operations |
| [`Individual`](#individual--population-member) | Dataclass | One chromosome + its fitness score |
| [`GAConfig`](#gaconfig--ga-hyperparameters) | Dataclass | Genetic algorithm settings |
| [`OptimizationResult`](#optimizationresult--optimization-output) | Dataclass | All outputs from a completed run |
| [`GeneticOptimizer`](#geneticoptimizer--the-main-optimizer) | Class | Runs the GA and returns best params |
| [`save_result()`](#save_result--save-result-to-json) | Function | Persist an OptimizationResult to JSON |
| [`load_result()`](#load_result--load-result-from-json) | Function | Reload a saved result |
| [`print_optimization_report()`](#print_optimization_report--print-full-ascii-report) | Function | Full ASCII report with charts |
| [`main()`](#main--cli-entry-point-2) | Function | CLI demo entry point |

### paper_trader.py
| Symbol | Type | Description |
|---|---|---|
| [`PaperConfig`](#paperconfig--paper-trader-configuration) | Dataclass | All paper trader settings |
| [`SymbolState`](#symbolstate--per-symbol-position-state) | Dataclass | Position/signal state for one ticker |
| [`TradingState`](#tradingstate--full-account-state) | Dataclass | All-symbol persistent state |
| [`PaperTrader`](#papertrader--automated-paper-trading-engine) | Class | The main paper-trading engine |
| [`PaperTrader.run()`](#paphertraderrun--execute-one-trading-cycle) | Method | Run one complete signal-to-order cycle |
| [`load_strategy()`](#load_strategy--dynamically-load-a-strategy) | Function | Import a Strategy by dotted path |
| [`review_paper_performance()`](#review_paper_performance--review-paper-trading-results) | Function | Print P&L summary from logs |
| [`main()`](#main--cli-entry-point) | Function | CLI / cron entry point |

---

## strategy_tester.py

### `Strategy` — Abstract Base Class

[→ Source](strategy_tester.py#L51)

**What it is:** The contract all strategies must satisfy. Subclass this and implement `generate_signals(df)` to create your strategy. Both the backtester and paper trader accept any `Strategy` subclass.

**Used by:** All strategies in this framework. You write the signal logic once; the tester and paper trader consume it automatically.

**Signal convention:**
- `1` = enter long
- `-1` = enter short
- `0` = flat / cash

```python
from momentum.test.strategy_tester import Strategy
from momentum.momentum_tools import rsi, crossover

class RSIStrategy(Strategy):
    name    = "RSI Bounce"
    symbols = ["AAPL", "MSFT"]

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df["rsi"]    = rsi(df["Close"], window=14)
        df["signal"] = 0
        df.loc[df["rsi"] < 30, "signal"] = 1   # oversold → long
        df.loc[df["rsi"] > 70, "signal"] = 0   # overbought → exit
        df["signal"] = df["signal"].ffill().fillna(0)
        return df
```

> **When you need this:** Every strategy starts here. Any time you want to test a new trading idea, create a subclass of `Strategy`.

---

### `BacktestConfig` — Backtest Configuration

[→ Source](strategy_tester.py#L123)

**What it is:** A dataclass holding all parameters that control how the backtest engine simulates trades: account size, commissions, position sizing, and risk circuit-breakers.

**Used by:** Passed to `run_backtest()` and `run_comparison()`. Share one config across many tests for fair comparisons.

```python
from momentum.test.strategy_tester import BacktestConfig

cfg = BacktestConfig(
    initial_capital      = 100_000,
    commission_per_share = 0.005,      # IBKR Fixed plan
    slippage_pct         = 0.001,      # 0.1% per trade
    position_sizing      = "atr",      # volatility-adjusted sizing
    atr_risk_pct         = 0.01,       # risk 1% of capital per ATR
    allow_short          = False,      # long-only (cash account)
    max_drawdown_pct     = 0.20,       # halt if equity drops 20%
)
```

**Position sizing options:**

| Value | Behaviour |
|---|---|
| `"fixed"` | `shares = (capital × position_pct) / price` |
| `"atr"` | `shares = (capital × atr_risk_pct / ATR)` — volatility-adjusted |
| `"kelly"` | Uses Kelly Criterion from trade history (updates each trade) |

> **When you need this:** Set `initial_capital` to your actual account size. Use `"atr"` sizing when testing across assets with different volatilities. Always set `max_drawdown_pct` to simulate your real-world circuit-breaker.

---

### `BacktestResult` — Result Container

[→ Source](strategy_tester.py#L186)

**What it is:** A dataclass containing all performance metrics, the full equity curve, daily returns, and every individual trade from a backtest. The output of `run_backtest()`.

**Used by:** `print_result()`, `compare_results()`, `plot_equity_curve()`. Inspect this object to decide if a strategy is ready for paper trading.

**Key fields:**

| Field | Description |
|---|---|
| `total_return` | (final_equity / initial_capital) - 1 |
| `cagr_val` | Compound Annual Growth Rate |
| `sharpe` | Annualised Sharpe ratio |
| `sortino` | Annualised Sortino ratio |
| `calmar` | CAGR / |max_drawdown| |
| `max_dd` | Worst peak-to-trough equity decline (negative number) |
| `win_rate_val` | Fraction of trades with positive P&L |
| `kelly_fraction` | Suggested max position size fraction |
| `equity_curve` | `pd.Series` of daily portfolio values |
| `trades` | `List[Trade]` — all completed round trips |

> **When you need this:** After any `run_backtest()` call. Use `result.trades` to dig into individual trade details. Use `result.equity_curve` for custom charting.

---

### `Trade` — Single Trade Record

[→ Source](strategy_tester.py#L165)

**What it is:** A dataclass capturing every detail of one round-trip trade (entry + exit): dates, prices, shares, and P&L.

**Used by:** Stored in `BacktestResult.trades`. Useful for analysing trade distributions and identifying systematic entry/exit issues.

```python
result = run_backtest(MyStrategy(), df, cfg)
for t in result.trades:
    print(f"{t.entry_date} → {t.exit_date}  PnL: ${t.net_pnl:+.2f}  ({t.return_pct:.2%})")
```

---

### `run_backtest()` — Run a Historical Backtest

[→ Source](strategy_tester.py#L246)

**What it does:** Simulates trading a strategy on historical OHLCV data bar by bar. Signals generated on bar N execute at the Open of bar N+1 (realistic fill assumption). Tracks capital, positions, trades, and equity daily.

**Used by:** The primary entry point for any strategy validation. Use before paper trading or live trading.

**When you need this:**
- Every time you create or modify a strategy
- When tuning indicator parameters (e.g., trying RSI windows 7, 10, 14, 21)
- After any market regime change to see if the strategy still works

```python
from momentum.test.strategy_tester import run_backtest, BacktestConfig, print_result
import pandas as pd

df  = pd.read_csv("AAPL.csv", parse_dates=["Date"], index_col="Date")
cfg = BacktestConfig(initial_capital=100_000)
result = run_backtest(MACDMomentumStrategy(), df, cfg, symbol="AAPL")
print_result(result)
```

**Or with IBKR data:**
```python
from myIBApp import connect_to_tws
from momentum.test.strategy_tester import fetch_ibkr_history, run_backtest

app = connect_to_tws()
df  = fetch_ibkr_history(app, "AAPL", duration="2 Y", bar_size="1 day")
result = run_backtest(MACDMomentumStrategy(), df, symbol="AAPL")
```

**Args:**

| Arg | Type | Description |
|---|---|---|
| `strategy` | `Strategy` | Instantiated strategy |
| `df` | `pd.DataFrame` | OHLCV data with `DatetimeIndex` |
| `config` | `BacktestConfig` | Simulation params (uses defaults if None) |
| `symbol` | `str` | Ticker label for reporting |

**Returns:** `BacktestResult`

---

### `run_comparison()` — Compare Multiple Strategies

[→ Source](strategy_tester.py#L407)

**What it does:** Runs `run_backtest()` for each strategy in the dictionary on the same dataset, then returns all results for side-by-side comparison. Essential for strategy selection and parameter sweeps.

**Used by:** Quant analysts ranking candidate strategies or comparing parameter variations before picking one for paper trading.

**When you need this:**
- When you have 2+ strategies and want to pick the best one
- When parameter sweeping (e.g., "which RSI window gives the best Sharpe?")
- Before deciding which strategy to configure in `paper_config.json`

```python
from momentum.test.strategy_tester import run_comparison, compare_results

results = run_comparison(
    {
        "MACD":       MACDMomentumStrategy(),
        "RSI Bounce": RSIStrategy(),
        "ADX Trend":  ADXTrendStrategy(),
    },
    df,
    config=cfg,
    symbol="AAPL"
)
table = compare_results(results)
print(table.to_string())
table.to_csv("strategy_comparison.csv")
```

---

### `fetch_ibkr_history()` — Fetch Historical Data from IBKR

[→ Source](strategy_tester.py#L456)

**What it does:** Requests historical OHLCV bar data from a connected IBKR TWS instance. Returns a clean `DataFrame` with `DatetimeIndex` and `Open, High, Low, Close, Volume` columns, ready for backtesting.

**Used by:** When you want backtest data that exactly matches IBKR's data feed — ensuring no discrepancy between backtest and live execution.

**When you need this:**
- When you don't have a separate data source (Bloomberg, Refinitiv, yfinance)
- When testing instruments IBKR has that free data sources may not
- When verifying your backtest data matches what TWS would show live

```python
from myIBApp import connect_to_tws
from momentum.test.strategy_tester import fetch_ibkr_history

app = connect_to_tws()
df  = fetch_ibkr_history(
    app,
    symbol    = "AAPL",
    duration  = "2 Y",       # 2 years of data
    bar_size  = "1 day",     # daily bars
)
print(df.tail())
app.disconnect()
```

**IBKR duration strings:** `"1 D"`, `"5 D"`, `"1 W"`, `"1 M"`, `"3 M"`, `"6 M"`, `"1 Y"`, `"2 Y"`  
**IBKR bar size strings:** `"1 min"`, `"5 mins"`, `"15 mins"`, `"1 hour"`, `"1 day"`

> **Note:** IBKR rate-limits historical data requests. For large histories (> 1 year of intraday), spread requests with `time.sleep(10)` between them.

---

### `print_result()` — Print a Formatted Summary

[→ Source](strategy_tester.py#L541)

**What it does:** Prints a human-readable performance table to stdout from a `BacktestResult`.

```python
print_result(result)
# ====================================================
#   Strategy : MACD + ADX Filter
#   Symbol   : AAPL
#   Period   : 2022-01-03  →  2025-12-31
# ====================================================
#   Initial Capital                   $100,000.00
#   Final Equity                      $147,320.18
#   Total Return                           47.32%
#   CAGR                                   12.85%
#   ...
```

---

### `compare_results()` — Build a Comparison Table

[→ Source](strategy_tester.py#L589)

**What it does:** Takes the output of `run_comparison()` and returns a `pd.DataFrame` with one row per strategy and all key metrics as columns, sorted by Sharpe ratio.

```python
table = compare_results(results)
print(table.to_string())
table.to_csv("comparison.csv", index=False)
```

---

### `plot_equity_curve()` — Plot the Equity Curve

[→ Source](strategy_tester.py#L626)

**What it does:** Uses `matplotlib` to draw the strategy's equity curve over time with a drawdown panel below. Optionally overlays a buy-and-hold benchmark.

**When you need this:**  
Always view the equity curve before trusting Sharpe/Calmar numbers. The shape of the curve reveals whether gains are consistent or concentrated in a few lucky periods.

```python
from momentum.test.strategy_tester import plot_equity_curve

# Buy-and-hold benchmark normalised to same starting capital
bm = df["Close"] / df["Close"].iloc[0] * cfg.initial_capital
plot_equity_curve(result, benchmark=bm)
```

> **Requires:** `pip install matplotlib`

---

### `MACDMomentumStrategy` — Example Strategy

[→ Source](strategy_tester.py#L677)

**What it is:** A bundled, working example strategy (MACD line crosses above signal line while ADX > 25). Use it as a baseline or copy it as a starting template.

```python
from momentum.test.strategy_tester import (
    MACDMomentumStrategy, BacktestConfig, run_backtest, print_result
)
cfg    = BacktestConfig(initial_capital=100_000)
result = run_backtest(MACDMomentumStrategy(), df, cfg, symbol="AAPL")
print_result(result)
```

---

## paper_trader.py

### `PaperConfig` — Paper Trader Configuration

[→ Source](paper_trader.py#L95)

**What it is:** All settings the paper trader needs, loaded from a JSON file. Edit `paper_config.json` to change settings without touching the code.

```json
{
  "strategy_module":  "my_strategies.MACDMomentumStrategy",
  "symbols":          ["AAPL", "MSFT"],
  "bar_size":         "1 day",
  "duration":         "6 M",
  "position_pct":     0.10,
  "max_shares":       500,
  "tws_port":         7497,
  "state_file":       "paper_state.json",
  "trade_log":        "paper_trades.csv",
  "daily_loss_limit": 0.02,
  "dry_run":          false
}
```

**TWS Port reference:**

| Port | What it connects to |
|---|---|
| `7497` | TWS paper trading account **(default — use this)** |
| `7496` | TWS live account (real money — be careful) |
| `4002` | IB Gateway paper account |
| `4001` | IB Gateway live account |

> **When to adjust:** Change `tws_port` to `7496` **only** after your paper trading performance is satisfactory for at least 2–3 months. Reduce `position_pct` to manage risk. Set `daily_loss_limit` to 0.02 (2%) for a firm daily stop.

---

### `SymbolState` — Per-Symbol Position State

[→ Source](paper_trader.py#L186)

**What it is:** Everything the paper trader needs to remember between runs for one symbol: current position, fill price, and the last signal value.

**Used by:** Stored in `TradingState.symbols`. Persisted to `paper_state.json` between cron runs so the trader knows what it holds when it wakes up next time.

**Fields:**

| Field | Description |
|---|---|
| `position` | 1=long, -1=short, 0=flat |
| `shares` | Number of shares currently held |
| `entry_price` | Average fill price of current position |
| `last_signal` | Signal value on last run |
| `ibkr_order_id` | Order ID of last submitted order |

---

### `TradingState` — Full Account State

[→ Source](paper_trader.py#L226)

**What it is:** The complete persistent state: one `SymbolState` per symbol plus account-level cash, realised P&L, and today's P&L (for the daily loss limit check).

**Used by:** Loaded at the start of every run by `PaperTrader.load_state()` and saved at the end by `PaperTrader.save_state()`.

**State file location:** `paper_state.json` (configured in `paper_config.json`)

> **Editing the state file:** If you manually close a position in TWS (e.g., during a connectivity issue), edit `paper_state.json` directly to set that symbol's `position` and `shares` to 0. Otherwise the paper trader will think it still holds the position.

---

### `PaperTrader` — Automated Paper Trading Engine

[→ Source](paper_trader.py#L260)

**What it is:** Orchestrates the full signal-to-order cycle for one scheduled execution.

**Lifecycle of one run:**
```
connect_to_tws()
  → load_state()
  → for each symbol:
      fetch_market_data()
      compute_signal()
      execute_signal()
      log_trade()
  → save_state()
  → disconnect()
```

---

### `PaperTrader.run()` — Execute One Trading Cycle

[→ Source](paper_trader.py#L390)

**What it does:** The single method you call (or the scheduler calls) to run one complete paper-trading cycle.

**When you need this:**
- The cron job calls this automatically on schedule
- Call manually from a REPL to run a one-off cycle for testing
- Use `dry_run=True` in the config the first few times to verify signals before enabling orders

```python
from momentum.test.paper_trader import PaperTrader, PaperConfig
from momentum.test.strategy_tester import MACDMomentumStrategy

cfg    = PaperConfig.from_json("paper_config.json")
trader = PaperTrader(MACDMomentumStrategy(), cfg)
trader.run()
```

---

### `load_strategy()` — Dynamically Load a Strategy

[→ Source](paper_trader.py#L471)

**What it does:** Imports a `Strategy` subclass from a dotted module path string. Lets you configure the strategy in JSON without hard-coding it in the script.

```python
from momentum.test.paper_trader import load_strategy
strategy = load_strategy("my_strategies.MACDMomentumStrategy")
```

> **When you need this:** When adding a new strategy, create the class in your own module (e.g., `my_strategies.py` in the repo root), then update `strategy_module` in `paper_config.json`.

---

### `review_paper_performance()` — Review Paper Trading Results

[→ Source](paper_trader.py#L527)

**What it does:** Reads the trade log CSV and state file and prints a formatted summary of paper trading performance: current positions, total P&L, win rate.

```python
from momentum.test.paper_trader import review_paper_performance
review_paper_performance("paper_trades.csv", "paper_state.json")
```

**Or from the CLI:**
```bash
python momentum/test/paper_trader.py --config paper_config.json --review
```

---

### `main()` — CLI Entry Point

[→ Source](paper_trader.py#L582)

**What it does:** Parses CLI arguments, loads config and strategy, and calls `PaperTrader.run()` once.

**CLI arguments:**

| Argument | Description |
|---|---|
| `--config FILE` | Path to JSON config file **(required)** |
| `--dry-run` | Compute signals but do NOT submit any orders |
| `--review` | Print P&L summary and exit |

---

## run_backtest_example.py

A self-contained script that runs the bundled `MACDMomentumStrategy` backtest and prints a complete ASCII report. The quickest way to see the framework in action.

### Quick start

```bash
# No setup required — uses synthetic data if yfinance not installed:
python momentum/test/run_backtest_example.py

# Real data from Yahoo Finance:
pip install yfinance
python momentum/test/run_backtest_example.py --symbol AAPL --years 3

# More options:
python momentum/test/run_backtest_example.py --symbol SPY --years 5 --sizing atr
python momentum/test/run_backtest_example.py --capital 50000
python momentum/test/run_backtest_example.py --source ibkr --symbol MSFT
python momentum/test/run_backtest_example.py --source csv --csv-path data/AAPL.csv
```

### Sample output

```
  RangerwizardBacktest  |  2026-07-19 10:30:00
  Symbol: AAPL  |  Source: auto  |  Sizing: fixed

  Loading data ...
  Downloading AAPL (3y) from Yahoo Finance ... done. (756 bars)

  Running backtest: MACD + ADX Filter on AAPL ...

════════════════════════════════════════════════════════════════════════════════
  BACKTEST REPORT: MACD + ADX Filter on AAPL
  Period: 2023-07-19  →  2026-07-18  |  Data: yfinance
════════════════════════════════════════════════════════════════════════════════

┌──────────────────────────────────────────────────────────────────────────────┐
│                           PERFORMANCE METRICS                                │
├──────────────────────────────────────────────────────────────────────────────┤
│  Initial Capital              $100,000.00   Final Equity         $138,205.42 │
│  Total Return                      38.21%   CAGR                     11.46%  │
│  Max Drawdown                     -14.73%   Annualised Vol            10.42% │
│  Sharpe Ratio                       1.32    Sortino Ratio              1.98  │
│  Calmar Ratio                       0.78    Win Rate                  58.33% │
│  Total Trades                          12   Kelly Fraction            14.20% │
└──────────────────────────────────────────────────────────────────────────────┘

────────────────────────────────────────────────────────────────────────────────
  Equity Curve
   $138,205 │                                                    ▪▪▪▪▪▪▪▪▪▪
            │                                         ▪▪▪▪▪▪▪▪▪│
   $119,102 │                               ▪▪▪▪▪▪▪▪▪
            │                      ▪▪▪▪▪▪▪▪▪
   $100,000 │▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪▪
            └────────────────────────────────────────────────────────────────
            2023-07-19              2025-01-04                    2026-07-18

────────────────────────────────────────────────────────────────────────────────
  Drawdown
    0.0% │
         │█████                    ████                 ██
   -7.4% │█████████         ███████████          ███████████
  -14.7% │███████████████████████████████████████████████████
         └────────────────────────────────────────────────────────────────

────────────────────────────────────────────────────────────────────────────────
  Monthly Returns (%)
  ──────────────────────────────────────────────────────────────────────────────
    Year      Jan      Feb      Mar      Apr      May      Jun   ...    EOY
  ──────────────────────────────────────────────────────────────────────────────
    2023    +1.2%   -0.8%   +3.1%   +0.4%   +2.7%   -1.1%  ...  +14.2%
    2024    -2.1%   +4.3%   +1.8%   -0.3%   +3.2%   +0.7%  ...  +16.8%
    2025    +0.9%   +1.4%   -3.2%   +2.1%   +1.8%   -0.4%  ...   +7.1%

────────────────────────────────────────────────────────────────────────────────
  Trade Return Distribution
  ──────────────────────────────────────────────────────────────────────────────
  -12.0% →  -9.0%  │░                                             1
   -9.0% →  -6.0%  │░░                                            2
   -6.0% →  -3.0%  │░░░░░░░░░░░░░░░░░                             5
   -3.0% →   0.0%  │░░░░░░░░░░░░░░░░░░░░░░░░░░░░                  7
    0.0% →  +3.0%  │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓      10
   +3.0% →  +6.0%  │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓                          5
   +6.0% →  +9.0%  │▓▓▓▓▓▓▓▓▓▓▓                                   3
  ──────────────────────────────────────────────────────────────────────────────
  ░ = losses  ▓ = gains   n = 33 trades
```

---

### `load_price_data()` — Load Historical Price Data

[→ Source](run_backtest_example.py#L84)

**What it does:** Tries data sources in priority order until one succeeds: yfinance → IBKR → CSV → synthetic GBM. Returns an OHLCV DataFrame and a label string for the report.

**Used by:** Called at the top of `main()` in `run_backtest_example.py`. Also useful in your own backtest scripts when you want a robust data loader.

**When you need this:**
- Use `source="auto"` (default) for the most convenient setup
- Use `source="synthetic"` to test strategy code without any network access
- Use `source="csv"` when you have a specific dataset you want to reproduce results against

```python
from momentum.test.run_backtest_example import load_price_data

df, label = load_price_data("AAPL", years=3, source="yfinance")
print(f"Loaded {len(df)} bars from {label}")
```

---

### `ascii_line_chart()` — ASCII Line Chart

[→ Source](run_backtest_example.py#L196)

**What it renders:** A grid of `width × height` characters where each column represents a time period and `▪` marks the scaled value. Y-axis labels on the left; date labels at the bottom.

**Used by:** `print_equity_chart()` for the portfolio equity curve. Also usable with any `pd.Series` that has a `DatetimeIndex`.

**When you need this:**
- Terminal-only environments where matplotlib is unavailable (cron logs, SSH sessions, Slack)
- Quick visual checks without opening a notebook

```python
from momentum.test.run_backtest_example import ascii_line_chart
chart = ascii_line_chart(result.equity_curve, "Portfolio Value", width=70, height=10)
print(chart)
```

---

### `ascii_area_chart()` — ASCII Area Chart

[→ Source](run_backtest_example.py#L270)

**What it renders:** A downward-filled area chart where `█` characters fill from the zero line down to each value. Deeper fills = larger drawdowns. Zero is always at the top.

**Used by:** `print_drawdown_chart()` to render the drawdown series.

**When you need this:**
- Visually identifying the duration and severity of drawdown periods in terminal output
- Including in cron logs to track live paper trading drawdown over time

```python
from momentum.test.run_backtest_example import ascii_area_chart
dd = (equity / equity.cummax()) - 1
print(ascii_area_chart(dd, "Drawdown", width=70, height=8))
```

---

### `ascii_histogram()` — ASCII Histogram

[→ Source](run_backtest_example.py#L346)

**What it renders:** A horizontal bar chart where each row is a return bin. `░` = losses, `▓` = gains. Bar length represents the count of trades in that bin.

**Used by:** `print_trade_histogram()` to show the distribution of individual trade returns.

**When you need this:**
- Checking if your strategy has a right-skewed distribution (few large wins, many small losses) — ideal for trend-following
- Spotting if one or two outlier wins are responsible for all the CAGR

```python
from momentum.test.run_backtest_example import ascii_histogram
rets = [t.return_pct for t in result.trades]
print(ascii_histogram(rets, "Trade Returns", bins=12, width=40))
```

---

### `ascii_monthly_table()` — Monthly Returns Table

[→ Source](run_backtest_example.py#L403)

**What it renders:** A year × month grid of compounded monthly returns. Each cell shows `+X.X%` or `-X.X%`. The `EOY` column shows the full-year compounded return.

**Used by:** `print_monthly_table()` to show seasonality and year-over-year consistency.

**When you need this:**
- Identifying seasonal patterns (months where the strategy consistently wins or loses)
- Checking year-over-year consistency — a good strategy should have mostly positive years
- Spotting regime changes: if an entire year is negative, the strategy likely hit a hostile regime

```python
from momentum.test.run_backtest_example import ascii_monthly_table
print(ascii_monthly_table(result.daily_returns))
```

---

### `ascii_trade_table()` — Recent Trades Table

[→ Source](run_backtest_example.py#L459)

**What it renders:** A formatted table showing the last N trades with columns: entry date, exit date, direction (+LONG/-SHORT), shares, entry price, exit price, net P&L, and return %.

**Used by:** `print_recent_trades()` as the final section of the report.

**When you need this:**
- Verifying the backtest is entering and exiting at correct prices/dates
- Finding which specific trade caused a large drawdown
- Cross-referencing against your paper trade log

```python
from momentum.test.run_backtest_example import ascii_trade_table
print(ascii_trade_table(result.trades, n=20))
```

---

### `print_full_report()` — Full ASCII Report

[→ Source](run_backtest_example.py#L529)

**What it does:** Orchestrates the complete report by calling all section printers in order: header → metrics box → equity curve → drawdown → monthly returns → trade histogram → recent trades → interpretation guide.

**Used by:** `main()` after the backtest completes. Also call this directly in your own scripts after any `run_backtest()`.

```python
from momentum.test.run_backtest_example import print_full_report
result = run_backtest(MyStrategy(), df, cfg, symbol="AAPL")
print_full_report(result, data_source="yfinance")
```

---

### `main()` — CLI Entry Point

[→ Source](run_backtest_example.py#L595)

**What it does:** Parses CLI arguments, loads data, runs the `MACDMomentumStrategy` backtest, and prints the full ASCII report.

**CLI arguments:**

| Argument | Default | Description |
|---|---|---|
| `--symbol TICKER` | `AAPL` | Stock ticker to test |
| `--years N` | `3` | Years of history |
| `--source SOURCE` | `auto` | `auto` \| `yfinance` \| `ibkr` \| `csv` \| `synthetic` |
| `--csv-path PATH` | — | CSV file path (required if `--source csv`) |
| `--capital AMOUNT` | `100000` | Starting capital in dollars |
| `--sizing METHOD` | `fixed` | `fixed` \| `atr` \| `kelly` |
| `--allow-short` | off | Allow short positions (`signal = -1`) |

---

## End-to-End Workflow

### Step 1 — Write your strategy

```python
# my_strategies.py  (in the repo root)
import pandas as pd
from momentum.test.strategy_tester import Strategy
from momentum.momentum_tools import rsi, macd, adx, crossover

class MyStrategy(Strategy):
    name    = "RSI + MACD Combo"
    symbols = ["AAPL", "MSFT"]

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df["rsi"]          = rsi(df["Close"], window=14)
        macd_l, sig, _     = macd(df["Close"])
        adx_v, _, _        = adx(df["High"], df["Low"], df["Close"])
        df["signal"] = 0
        buy  = (df["rsi"] < 50) & crossover(macd_l, sig) & (adx_v > 20)
        sell = crossover(sig, macd_l)   # macd crosses back below signal
        df.loc[buy,  "signal"] = 1
        df.loc[sell, "signal"] = 0
        df["signal"] = df["signal"].ffill().fillna(0)
        return df
```

### Step 2 — Backtest on historical data

```python
# backtest_runner.py
from myIBApp import connect_to_tws
from momentum.test.strategy_tester import (
    fetch_ibkr_history, BacktestConfig,
    run_backtest, run_comparison,
    print_result, compare_results, plot_equity_curve,
)
from my_strategies import MyStrategy
from momentum.test.strategy_tester import MACDMomentumStrategy

app = connect_to_tws()
df  = fetch_ibkr_history(app, "AAPL", duration="3 Y", bar_size="1 day")
app.disconnect()

cfg = BacktestConfig(
    initial_capital      = 100_000,
    commission_per_share = 0.005,
    position_sizing      = "atr",
    allow_short          = False,
    max_drawdown_pct     = 0.25,
)

# Single strategy
result = run_backtest(MyStrategy(), df, cfg, symbol="AAPL")
print_result(result)
plot_equity_curve(result)

# Compare vs baseline
results = run_comparison(
    {"My Strategy": MyStrategy(), "MACD Baseline": MACDMomentumStrategy()},
    df, cfg, symbol="AAPL"
)
print(compare_results(results).to_string())
```

### Step 3 — Configure the paper trader

Edit [`paper_config.json`](paper_config.json):
```json
{
  "strategy_module": "my_strategies.MyStrategy",
  "symbols":         ["AAPL"],
  "tws_port":        7497,
  "dry_run":         true
}
```

### Step 4 — Test with dry-run

```bash
# From the repo root:
python momentum/test/paper_trader.py --config momentum/test/paper_config.json --dry-run
```

Verify the log shows the expected signals without any orders being submitted.

### Step 5 — Schedule for live paper trading

```bash
# Edit crontab:
crontab -e

# Add this line (runs every weekday at 9:35 AM local time):
35 9 * * 1-5 cd /Users/ram/Documents/GH/rangerwizard && \
    python momentum/test/paper_trader.py \
    --config momentum/test/paper_config.json \
    >> momentum/test/logs/paper_trader.log 2>&1
```

Create the logs directory first:
```bash
mkdir -p /Users/ram/Documents/GH/rangerwizard/momentum/test/logs
```

### Step 6 — Review paper trading performance

```bash
python momentum/test/paper_trader.py \
    --config momentum/test/paper_config.json \
    --review
```

### Step 7 — Go live (when ready)

After consistent paper performance, change `tws_port` from `7497` to `7496` in `paper_config.json`. Everything else stays the same.

---

## File Outputs

| File | Created by | Contents |
|---|---|---|
| `paper_state.json` | `PaperTrader.save_state()` | Current positions and last signals |
| `paper_trades.csv` | `PaperTrader.log_trade()` | Full trade history (timestamp, symbol, action, price, P&L) |
| `logs/paper_trader.log` | cron redirect | Full run logs with timestamps |
| `runs/macd_opt_latest.json` | `save_result()` | Best params + fitness history from last GA run |

---

## genetic_optimizer.py

Tunes any strategy's parameters across a basket of stocks using a genetic algorithm.
Finds settings that generalise across different market regimes rather than overfitting to one instrument.

### Quick start

```bash
# Full demo run (~3–6 min, 5 stocks, 20 generations):
python3.11 momentum/test/genetic_optimizer.py

# Fast smoke test (~30 sec, 4 stocks, 5 generations):
python3.11 momentum/test/genetic_optimizer.py --fast
```

### How it works

```
  Random population of param dicts
           │
  ┌────────▼────────────────────────────────────────────────────┐
  │  For each generation:                                        │
  │    Evaluate every individual on all symbols (threaded)       │
  │    fitness = mean_sharpe − 0.3 × std_sharpe  (robust mode)  │
  │    Tournament selection → crossover → mutation               │
  │    Elites copied unchanged → fill rest with offspring        │
  └────────▼────────────────────────────────────────────────────┘
  Best individual → OptimizationResult
```

### Sample output

```
  ══════════════════════════════════════════════════════════════════
  GENETIC OPTIMIZER
  Strategy factory : TunableMACDStrategy
  Symbols          : ['AAPL', 'SPY', 'QQQ', 'JPM', 'XOM']
  Population       : 25  Generations: 20
  Fitness metric   : robust  Workers: 4
  ══════════════════════════════════════════════════════════════════

  Gen   1/20  │░░░░░░░░░░░░░░░░░░░░│  best=-0.124  avg=-0.381  │  fast=8  slow=41  adx_min=28.50
  Gen   2/20  │▓▓░░░░░░░░░░░░░░░░░░│  best=+0.213  avg=-0.102  │  fast=9  slow=38  adx_min=24.10
  Gen   5/20  │▓▓▓▓▓▓░░░░░░░░░░░░░░│  best=+0.587  avg=+0.314  │  fast=10 slow=32  adx_min=21.40
  Gen  10/20  │▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░│  best=+0.842  avg=+0.621  │  fast=11 slow=28  adx_min=20.10
  Gen  15/20  │▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░│  best=+0.941  avg=+0.783  │  fast=11 slow=27  adx_min=19.80
  Gen  20/20  │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░│  best=+0.978  avg=+0.851  │  fast=11 slow=27  adx_min=20.10

  ══════════════════════════════════════════════════════════════════
  GENETIC OPTIMIZATION COMPLETE
  Symbols     : ['AAPL', 'SPY', 'QQQ', 'JPM', 'XOM']
  Total tests : 2,500   Runtime: 214.3s

  ┌──────────────────────────────────────────────────────────────────┐
  │                    BEST PARAMETERS FOUND                         │
  ├──────────────────────────────────────────────────────────────────┤
  │  fast                         11                                 │
  │  slow                         27                                 │
  │  adx_period                   14                                 │
  │  adx_min                      20.10                              │
  │  Best Fitness Score           +0.9780                            │
  └──────────────────────────────────────────────────────────────────┘

  Best Individual — Per-Stock Results
  ────────────────────────────────────────────────────────────────
  Symbol    Sharpe   Calmar    MaxDD     CAGR   Trades  WinRate
  ────────────────────────────────────────────────────────────────
    AAPL    +1.24    +0.91   -12.3%   +14.2%       18   61.1%
    SPY     +0.98    +0.72    -9.8%   +11.1%       22   54.5%
    QQQ     +1.15    +0.84   -10.9%   +13.4%       19   57.9%
    JPM     +0.81    +0.54   -14.1%    +8.9%       16   50.0%
    XOM     +0.72    +0.48   -15.2%    +7.8%       14   50.0%
  ────────────────────────────────────────────────────────────────
  MEAN      +0.98
  STD        0.19   (lower std = more consistent across stocks)
```

---

### `IntParam` — Integer Parameter

[→ Source](genetic_optimizer.py#L63)

**What it is:** Defines a single integer hyperparameter with inclusive min/max bounds.

**Used by:** Passed as values in the `params` dict to `ParameterSpace`. Use for indicator windows, bar counts, lookback periods.

```python
from momentum.test.genetic_optimizer import IntParam, ParameterSpace
space = ParameterSpace({"rsi_window": IntParam(5, 50)})
```

---

### `FloatParam` — Float Parameter

[→ Source](genetic_optimizer.py#L104)

**What it is:** Defines a continuous float parameter with inclusive min/max bounds and optional decimal precision.

**Used by:** Passed as values in the `params` dict to `ParameterSpace`. Use for thresholds (ADX cutoff, RSI level), multipliers (ATR stop-loss distance), or any decimal quantity.

```python
space = ParameterSpace({"adx_min": FloatParam(10.0, 45.0, decimals=2)})
```

---

### `ChoiceParam` — Discrete Choice Parameter

[→ Source](genetic_optimizer.py#L145)

**What it is:** A parameter that can only take one of a fixed set of values. Mutation randomly picks one of the choices.

**Used by:** For categorical strategy settings: which indicator type to use, bar size, position direction mode.

```python
space = ParameterSpace({"ma_type": ChoiceParam(["sma", "ema", "wma"])})
```

---

### `ParameterSpace` — Search Space

[→ Source](genetic_optimizer.py#L184)

**What it is:** Container for all parameter definitions plus the genetic operations (random sampling, crossover, mutation, constraint enforcement). Passed to `GeneticOptimizer`.

**When you need this:** Once per strategy type. Define bounds for every tunable parameter and any constraints between them.

```python
from momentum.test.genetic_optimizer import ParameterSpace, IntParam, FloatParam

space = ParameterSpace(
    params={
        "fast":    IntParam(5,  20),
        "slow":    IntParam(15, 60),
        "adx_min": FloatParam(15.0, 40.0),
    },
    constraints=[
        lambda p: p["fast"] < p["slow"] - 4,   # MACD: fast must be < slow - 4
    ]
)
```

> **Constraint tip:** If a parameter consistently hits its boundary (min or max) after optimization, expand the `IntParam`/`FloatParam` range and re-run — the optimizer is trying to go further but can't.

---

### `Individual` — Population Member

[→ Source](genetic_optimizer.py#L305)

**What it is:** One chromosome (parameter dict) + its evaluated fitness score + per-stock `BacktestResult` objects.

**When you need this:** Inspect `result.best_individual.per_stock_results` to see the full backtest results for the champion parameter set on each stock.

```python
best = result.best_individual
for sym, r in best.per_stock_results.items():
    print(f"{sym}: Sharpe={r.sharpe:.2f}  Trades={r.total_trades}")
```

---

### `GAConfig` — GA Hyperparameters

[→ Source](genetic_optimizer.py#L342)

**What it is:** All settings controlling the genetic algorithm: population size, generations, mutation rate, fitness metric, and parallelism.

**Key settings:**

| Field | Default | Notes |
|---|---|---|
| `population_size` | 40 | More = better exploration, slower. 25–60 typical. |
| `n_generations` | 30 | More = finer convergence. 15–40 typical. |
| `mutation_rate` | 0.15 | 0.05 = fine-tune; 0.25 = explore new space |
| `fitness_metric` | `"robust"` | `"robust"` penalises inconsistency across stocks |
| `consistency_penalty` | 0.30 | Higher = stronger penalty for stock-to-stock variance |
| `min_trades` | 5 | Discard chromosomes that barely trade |
| `n_workers` | 4 | Parallel threads — set to your CPU core count |

```python
cfg = GAConfig(
    population_size     = 40,
    n_generations       = 30,
    fitness_metric      = "robust",
    consistency_penalty = 0.30,
    min_trades          = 8,
    n_workers           = 4,
    random_seed         = 42,     # set for reproducible runs
)
```

**Fitness metric guide:**

| Metric | Best for |
|---|---|
| `"sharpe"` | Maximise risk-adjusted return, ignoring cross-stock consistency |
| `"calmar"` | Minimise drawdown relative to return |
| `"composite"` | Balanced: 40% Sharpe + 30% Calmar + 30% Win Rate |
| `"robust"` | **Recommended** — Sharpe minus a penalty for inconsistency across stocks |

---

### `OptimizationResult` — Optimization Output

[→ Source](genetic_optimizer.py#L436)

**What it is:** Complete output from a GA run: best parameters, fitness trajectory, parameter convergence history, per-stock results.

**Key fields to inspect:**

```python
result = opt.run()

# Use immediately:
best_strategy = TunableMACD(result.best_params)

# Check convergence:
print(result.fitness_history)       # best fitness per generation
print(result.avg_fitness_history)   # average fitness per generation

# Check parameter evolution:
print(result.param_convergence)     # {param: [value_at_gen_0, ..., value_at_gen_N]}

# Check cross-stock consistency:
for sym, r in result.best_individual.per_stock_results.items():
    print(sym, r.sharpe)
```

> **Overfitting check:** If `fitness_history[-1] >> avg_fitness_history[-1]`, the population has converged to a single local optimum. Try increasing `mutation_rate` or `population_size`.

---

### `GeneticOptimizer` — The Main Optimizer

[→ Source](genetic_optimizer.py#L493)

**What it is:** The engine that runs the full GA loop. Takes a strategy factory, parameter space, stock symbols, and config — returns an `OptimizationResult`.

**When you need this:**
- When indicator parameters (RSI window, MACD fast/slow) are clearly sub-optimal
- When you have multiple free parameters and want a principled way to set them
- After a market regime change — re-optimize on recent data

**Writing a tunable strategy (the factory pattern):**

```python
from momentum.test.strategy_tester import Strategy
from momentum.momentum_tools import rsi, crossover

class TunableRSI(Strategy):
    def __init__(self, params: dict):
        self._params = params

    @property
    def name(self):
        return f"RSI(w={self._params['window']},ob={self._params['overbought']:.0f})"

    def generate_signals(self, df):
        r = rsi(df["Close"], window=int(self._params["window"]))
        df["signal"] = 0
        df.loc[r < self._params["oversold"],    "signal"] = 1
        df.loc[r > self._params["overbought"],  "signal"] = 0
        df["signal"] = df["signal"].ffill().fillna(0)
        return df

space = ParameterSpace(
    params={
        "window":     IntParam(5, 30),
        "oversold":   FloatParam(20.0, 40.0),
        "overbought": FloatParam(60.0, 85.0),
    },
    constraints=[lambda p: p["oversold"] < p["overbought"]],
)

opt = GeneticOptimizer(
    strategy_factory = TunableRSI,
    param_space      = space,
    symbols          = ["AAPL", "MSFT", "SPY", "QQQ", "JPM"],
    config           = GAConfig(population_size=30, n_generations=20),
)
result = opt.run()
print_optimization_report(result)
```

---

### `save_result()` — Save Result to JSON

[→ Source](genetic_optimizer.py#L668)

**What it does:** Serialises key optimization fields to JSON for later review or reuse of the best parameters.

```python
save_result(result, "runs/rsi_opt_v1.json")
```

---

### `load_result()` — Load Result from JSON

[→ Source](genetic_optimizer.py#L722)

**What it does:** Reads a saved JSON file and returns it as a dict. Use `data["best_params"]` to get the champion parameters.

```python
data = load_result("runs/rsi_opt_v1.json")
strategy = TunableRSI(data["best_params"])
```

---

### `print_optimization_report()` — Print Full ASCII Report

[→ Source](genetic_optimizer.py#L813)

**What it does:** Renders the complete optimization report: best params box, per-stock results table, fitness convergence chart, average fitness chart, and parameter evolution table.

```python
result = opt.run()
print_optimization_report(result)
```

**How to interpret:**
- **Fitness chart flattening** → convergence achieved; further generations won't help much
- **Per-stock Sharpe STD < 0.4** → parameters are robust across instruments
- **Parameter at boundary** → expand the `IntParam`/`FloatParam` range
- **avg_fitness << best_fitness** → population stuck in local optima; increase `mutation_rate`

---

### `main()` — CLI Entry Point

[→ Source](genetic_optimizer.py#L875)

**What it does:** Runs the bundled `TunableMACDStrategy` demo on a 5-stock basket.

```bash
# Full run (~3–6 min):
python3.11 momentum/test/genetic_optimizer.py

# Quick smoke test (~30 sec):
python3.11 momentum/test/genetic_optimizer.py --fast
```

---

### Overfitting Warning

The genetic optimizer will find parameters that maximise fitness **on the training data**. Always validate on a held-out period:

```python
# Train on older data
df_train = df[df.index < "2025-01-01"]
opt = GeneticOptimizer(..., price_data={sym: df_train for sym in symbols})
result = opt.run()

# Validate on newer data (out-of-sample)
df_val = df[df.index >= "2025-01-01"]
strategy = TunableMACD(result.best_params)
val_result = run_backtest(strategy, df_val, cfg, symbol="AAPL")
print_result(val_result)
# If val Sharpe << train Sharpe → overfitting
```

