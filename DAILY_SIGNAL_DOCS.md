# daily_signal.py — Documentation

Runs the AlphaComposite momentum strategy daily via IBKR, places trades, and
logs everything for later strategy refinement.

**File:** [`daily_signal.py`](daily_signal.py) (repo root)

---

## Quick Start

```bash
# Paper trading, default symbols, dry-run (no orders):
python3.11 daily_signal.py --dry-run

# Paper trading, real orders:
python3.11 daily_signal.py

# Custom symbols:
python3.11 daily_signal.py --symbols AAPL MSFT SPY NVDA

# Pre-defined universe:
python3.11 daily_signal.py --universe megacap

# Live account (real money — use with caution):
python3.11 daily_signal.py --live

# Smaller risk per trade (0.5% instead of 1%):
python3.11 daily_signal.py --risk-pct 0.005
```

**Cron (runs at 9:45 AM ET on weekdays):**
```bash
# crontab -e
45 9 * * 1-5 cd /Users/ram/Documents/GH/rangerwizard && \
    python3.11 daily_signal.py >> logs/cron.log 2>&1
```

---

## How It Works

```
Connect to IBKR (paper port 7497)
         │
Fetch account NLV + open positions
         │
For each symbol:
    Fetch 1Y daily OHLCV from IBKR
    Run AlphaCompositeMomentumStrategy (optimised params)
    ┌── Signal = 1 (LONG) ──────────────────────────────┐
    │   Position = 0 → BUY  (1% NLV / price = shares)   │
    │   Position > 0 → HOLD (already long)               │
    └───────────────────────────────────────────────────┘
    ┌── Signal = 0 (FLAT) ──────────────────────────────┐
    │   Position > 0 → SELL (close all shares)           │
    │   Position = 0 → HOLD (already flat)               │
    └───────────────────────────────────────────────────┘
    Log decision to CSV (whether or not a trade was placed)

Disconnect
```

---

## Log Files (`logs/` — excluded from git)

| File | Updated by | Contents |
|---|---|---|
| `logs/daily_signal_YYYY-MM-DD.log` | `setup_logging()` | Full text run log |
| `logs/decisions_YYYY-MM-DD.csv` | `log_decision()` | Every signal evaluation (BUY/SELL/HOLD) |
| `logs/trades_YYYY-MM-DD.csv` | `log_trade()` | Orders actually submitted |
| `logs/cron.log` | cron redirect | stdout/stderr from scheduled runs |

### `decisions_YYYY-MM-DD.csv` columns

| Column | Description |
|---|---|
| `timestamp` | ISO datetime of the evaluation |
| `symbol` | Ticker |
| `signal` | 1=long, 0=flat |
| `composite` | Composite score [0, 1] |
| `action` | BUY / SELL / HOLD |
| `qty` | Shares in order (0 for HOLD) |
| `price_est` | Estimated fill price |
| `dollar_risk` | qty × price |
| `nlv` | Account NLV at time of decision |
| `risk_pct` | Configured risk fraction |
| `order_id` | IBKR order ID (-1=dry-run) |
| `params_hash` | SHA-256 of params (12 chars) |

---

## CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--symbols TICK ...` | `AAPL MSFT SPY QQQ NVDA` | Custom symbol list |
| `--universe NAME` | — | `megacap` \| `movers` \| `volume` \| `all` |
| `--live` | off | Use live account (port 7496). Default: paper (7497) |
| `--dry-run` | off | Compute signals, do NOT place orders |
| `--risk-pct FLOAT` | `0.01` | Fraction of NLV per trade (1% = 0.01) |
| `--years INT` | `1` | Years of OHLCV history to fetch |
| `--params-file PATH` | `momentum/test/runs/alpha_composite_opt.json` | Optimizer output |

**Port reference:**

| Port | Connects to |
|---|---|
| `7497` | TWS paper account **(default)** |
| `7496` | TWS live account |
| `4002` | IB Gateway paper |
| `4001` | IB Gateway live |

---

## Function Reference

### `setup_logging()` — Configure logging

[→ Source](daily_signal.py#L80)

**What it does:** Creates `logs/` dir, attaches a file handler (`logs/daily_signal_YYYY-MM-DD.log`) and a stream handler for terminal output. Called once at the start of `main()`.

**When you need this:** Modify this function to add email alerts on errors (e.g., using `logging.handlers.SMTPHandler`).

---

### `load_params()` — Load optimised strategy parameters

[→ Source](daily_signal.py#L125)

**What it does:** Reads `momentum/test/runs/alpha_composite_opt.json` produced by the genetic optimizer. Falls back to `DEFAULT_PARAMS` if the file is missing.

**When you need this:** After re-running `optimize_alpha.py`, new parameters are automatically picked up on the next daily run — no code changes needed.

```python
params = load_params(PARAMS_FILE)
strategy = AlphaCompositeMomentumStrategy(params=params)
```

---

### `connect_ibkr()` — Connect to IBKR TWS

[→ Source](daily_signal.py#L172)

**What it does:** Opens a fresh `myIBApp` connection on the specified port, starts the socket thread, and waits for `nextValidId`.

**Ports:** 7497 paper (default), 7496 live. TWS or IB Gateway must be running.

```python
app = connect_ibkr(PORT_PAPER)
```

---

### `fetch_account_nlv()` — Fetch Net Liquidation Value

[→ Source](daily_signal.py#L224)

**What it fetches:** Account NLV = total account value if all positions closed at current prices. Used to size every trade at exactly `risk_pct × NLV` dollars.

**When you need this:** NLV changes daily. Always fetching fresh ensures position sizing is consistent regardless of account growth or losses.

```python
nlv = fetch_account_nlv(app)
max_per_trade = nlv * 0.01   # 1% of account
```

---

### `fetch_positions()` — Fetch open positions

[→ Source](daily_signal.py#L284)

**What it fetches:** All currently held positions: `{symbol: (quantity, avg_cost)}`. Used to compare against strategy signals to determine BUY/SELL/HOLD.

```python
positions = fetch_positions(app)
aapl_qty = positions.get("AAPL", (0, 0.0))[0]
```

---

### `fetch_ohlcv()` — Download historical OHLCV from IBKR

[→ Source](daily_signal.py#L349)

**What it fetches:** Daily OHLCV bars from IBKR's historical data service — the same data shown in TWS charts, fully adjusted for splits and dividends. The strategy needs ≥252 bars to compute all indicators.

**Duration strings:** `"1 D"`, `"1 W"`, `"1 M"`, `"1 Y"`  
**Bar sizes:** `"1 min"`, `"5 mins"`, `"1 day"`

```python
df = fetch_ohlcv(app, "AAPL", duration="1 Y", bar_size="1 day")
```

---

### `fetch_current_price()` — Get current market price

[→ Source](daily_signal.py#L428)

**What it fetches:** Current ask/last price via IBKR snapshot quote. Used to convert the dollar risk limit into a share count.

```python
price = fetch_current_price(app, "AAPL")
```

---

### `compute_signal()` — Run strategy, return signal + score

[→ Source](daily_signal.py#L492)

**What it computes:** Runs `AlphaCompositeMomentumStrategy.generate_signals()` on the OHLCV DataFrame and returns the latest bar's signal (1=long, 0=flat) and the raw composite score (0–1).

```python
signal, composite = compute_signal(df, params)
print(f"Signal: {signal}  Composite: {composite:.3f}")
```

---

### `determine_action()` — BUY / SELL / HOLD logic

[→ Source](daily_signal.py#L526)

**What it does:** Compares desired signal vs current position:

| Desired | Held | Action |
|---|---|---|
| 1 (long) | 0 shares | **BUY** |
| 0 (flat) | > 0 shares | **SELL** |
| 1 (long) | > 0 shares | HOLD |
| 0 (flat) | 0 shares | HOLD |

```python
action = determine_action("AAPL", desired_signal=1, current_position=0)
# → "BUY"
```

---

### `calculate_position_size()` — 1% of NLV in shares

[→ Source](daily_signal.py#L582)

**What it computes:** `shares = floor(NLV × risk_pct / price)`, capped at `MAX_SHARES=1000`.

**Research:** Fixed-fractional sizing (Van Tharp). At 1% risk per trade, a 100% loss on one position costs only 1% of the account.

```python
qty = calculate_position_size(nlv=100_000, price=185.0, risk_pct=0.01)
# → 54 shares  ($10,000 / $185 = 54)
```

---

### `place_order()` — Submit market order to IBKR

[→ Source](daily_signal.py#L626)

**What it does:** Builds a SMART-routed market order via `myIBApp.make_stock_contract()` + `make_market_order()` and calls `placeOrder`. In `--dry-run` mode, logs the intended order without submitting.

> **IMPORTANT:** This places real orders when `dry_run=False` on a live account. Always test with `--dry-run` first.

```python
order_id = place_order(app, "AAPL", "BUY", 54, dry_run=False)
```

---

### `log_decision()` — Append to decisions CSV

[→ Source](daily_signal.py#L687)

**What it logs:** Every signal evaluation (BUY, SELL, or HOLD) with timestamp, composite score, price, NLV, and a hash of the parameters. HOLDs are logged too — they are as valuable as trades for ML training.

**For ML refinement:** Load the CSV, join with actual subsequent price data, and label each row with trade outcome. Train a classifier to predict when the composite score is most reliable.

---

### `log_trade()` — Append to trades CSV

[→ Source](daily_signal.py#L754)

**What it logs:** Only rows where an order was actually submitted (non-None `order_id`). Use this file to verify IBKR fills and compute realised P&L.

---

### `main()` — Entry point

[→ Source](daily_signal.py#L795)

**Cron setup:**
```bash
# crontab -e
45 9 * * 1-5 cd /Users/ram/Documents/GH/rangerwizard && \
    python3.11 daily_signal.py >> logs/cron.log 2>&1
```

**launchd (macOS) — create `~/Library/LaunchAgents/com.rangerwizard.daily.plist`:**
```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.rangerwizard.daily</string>
  <key>ProgramArguments</key><array>
    <string>/opt/homebrew/bin/python3.11</string>
    <string>/Users/ram/Documents/GH/rangerwizard/daily_signal.py</string>
  </array>
  <key>WorkingDirectory</key>
    <string>/Users/ram/Documents/GH/rangerwizard</string>
  <key>StartCalendarInterval</key><dict>
    <key>Hour</key><integer>9</integer>
    <key>Minute</key><integer>45</integer>
    <key>Weekday</key><integer>1</integer>
  </dict>
  <key>StandardOutPath</key>
    <string>/Users/ram/Documents/GH/rangerwizard/logs/launchd.log</string>
  <key>StandardErrorPath</key>
    <string>/Users/ram/Documents/GH/rangerwizard/logs/launchd.log</string>
</dict></plist>
```
Then: `launchctl load ~/Library/LaunchAgents/com.rangerwizard.daily.plist`

---

## Using Log Data to Refine the Strategy

```python
import pandas as pd

# Load decisions log
df = pd.read_csv("logs/decisions_2026-07-19.csv", parse_dates=["timestamp"])

# All signals that triggered a BUY
buys = df[df["action"] == "BUY"]

# Composite score distribution by action
print(df.groupby("action")["composite"].describe())

# Symbols where composite was high but action was HOLD (already long)
borderline = df[(df["composite"].astype(float) > 0.65) & (df["action"] == "HOLD")]

# Load trades to see what was actually filled
trades = pd.read_csv("logs/trades_2026-07-19.csv")
```
