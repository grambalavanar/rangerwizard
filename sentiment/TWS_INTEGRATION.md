# TWS Integration Guide

## Overview

The `strategy_orchestrator.py` has been updated to integrate with Interactive Brokers' TWS API via `myIBApp`, enabling:

- **Live market data streaming** from TWS
- **Real-time order execution** to your account
- **Bid/ask spread awareness** for accurate fills
- **Paper trading fallback** if TWS is unavailable
- **Portfolio statistics** and position tracking

## Connection Setup

### Prerequisites

1. **Interactive Brokers Account**: Active TWS connection
2. **TWS Running**: Paper or Live
3. **API Enabled**: In TWS, go to `Edit → Properties → API → Enable ActiveX and Socket Clients`
4. **Port Configuration**: Default is `7496` (paper trading) or `7497` (live)
5. **Firewall**: Allow localhost connections

### Starting the Strategy

#### Paper Trading Mode (Default)
```bash
cd /Users/ram/Documents/rangerwizard/sentiment

# Run hourly evaluation with mock data (no TWS required)
python3 strategy_orchestrator.py --max-iterations 5

# With debug output
python3 strategy_orchestrator.py --debug --max-iterations 5

# Faster testing: every 10 seconds
python3 strategy_orchestrator.py --interval 10 --max-iterations 5
```

#### Live Trading Mode with TWS
```bash
# Connect to TWS with hourly evaluation (default: 127.0.0.1:7496)
python3 strategy_orchestrator.py --live

# Custom host/port
python3 strategy_orchestrator.py --live --host 127.0.0.1 --port 7496

# With debug output
python3 strategy_orchestrator.py --live --debug

# More aggressive: check every 15 minutes
python3 strategy_orchestrator.py --live --interval 900
```

## Command-Line Options

```bash
--live                    # Enable live TWS execution (default: paper trading)
--host HOST              # TWS hostname (default: 127.0.0.1)
--port PORT              # TWS port (default: 7496 for paper, 7497 for live)
--client-id CLIENT_ID    # TWS client ID (default: 1)
--interval INTERVAL      # Loop interval in seconds (default: 3600 = 1 hour)
--max-iterations N       # Stop after N iterations
--debug                  # Enable verbose debug output
```

**Example usage:**
```bash
# Default: hourly evaluation in paper mode
python3 strategy_orchestrator.py

# Hourly live trading with debug
python3 strategy_orchestrator.py --live --debug

# Every 15 minutes (900s)
python3 strategy_orchestrator.py --live --interval 900

# Testing: every 5 seconds, 10 iterations
python3 strategy_orchestrator.py --interval 5 --max-iterations 10
```

## Features

### 1. Real-Time Market Data
```python
orchestrator = SentimentStrategyOrchestrator(paper_trading=False)
# Automatically fetches bid/ask/price/volume from TWS for monitored tickers

# Monitored tickers (line 305):
monitored_tickers = ['AAPL', 'TSLA', 'MSFT', 'AMZN']
```

**Customization**: Edit line 305 to monitor different tickers
```python
monitored_tickers = ['YOUR_TICKER_1', 'YOUR_TICKER_2', ...]
```

### 2. Live Order Execution
When `--live` flag is used:
- **BUY/SELL orders** placed at market or limit
- **Execution price** uses TWS bid/ask (accurate fills)
- **Order status** tracked (LIVE_FILLED, LIVE_PENDING, FAILED)
- **Position tracking** maintained in real-time
- **Paper fallback** if execution fails

### 3. Statistics & Monitoring
Automatic display every iteration:
```
============================================================
STRATEGY STATISTICS
============================================================
Connection Status: ✓ Live (TWS)
Total Trades: 42
Win Rate: 61.5%
Unrealized P&L: $3,847.50
Total P&L: $12,450.00
Daily P&L: $1,050.00
Portfolio Value: $1,034,500.00
Active Positions: 3

------------------------------------------------------------
POSITIONS
------------------------------------------------------------
  AAPL: LONG   100 @ $180.00 → $182.50 | P&L: $250 (+1.4%)
  TSLA: SHORT  50 @ $250.00 → $248.00 | P&L: $100 (+0.8%)
  MSFT: LONG   75 @ $300.00 → $302.00 | P&L: $150 (+0.7%)
============================================================
```

### 4. Graceful Fallback
If TWS is unavailable:
```
Warning: myIBApp not found. Run in paper trading mode only.
```
Strategy automatically falls back to paper trading with simulated data.

## Code Integration

### Access TWS App
```python
orchestrator = SentimentStrategyOrchestrator(paper_trading=False)

# Direct access to myIBApp instance
if orchestrator.connected and orchestrator.app:
    # Get current price
    price = orchestrator.app.get_stock_price('AAPL')
    
    # Place order
    order = orchestrator.app.place_order('AAPL', 100, 180.00, 'BUY')
```

### Market Data Updates
```python
# Fetch fresh data from TWS for specific tickers
orchestrator.fetch_market_data_from_tws(['AAPL', 'TSLA', 'MSFT'])

# Check current price
print(orchestrator.market_data['AAPL']['price'])
print(orchestrator.market_data['AAPL']['bid'])
print(orchestrator.market_data['AAPL']['ask'])
```

### Statistics Calculation
```python
# Get full stats dict
stats = orchestrator.calculate_portfolio_stats()

# Display formatted stats
orchestrator.display_stats()

# Access individual stats
print(f"Win Rate: {stats['win_rate']:.1f}%")
print(f"Portfolio Value: ${stats['portfolio_value']:,.2f}")
print(f"Unrealized P&L: ${stats['unrealized_pnl']:,.2f}")
```

## Execution Flow

```
Start Strategy
    ↓
Connect to TWS (if --live)
    ↓
[Loop] ─────────────────────────────
│                                  │
├─ Fetch Market Data from TWS      │
│  ├─ Bid/Ask prices               │
│  ├─ Recent volumes               │
│  └─ Update price history         │
│                                  │
├─ Ingest News Articles            │
│  ├─ Sentiment scoring            │
│  └─ Add to rolling windows       │
│                                  │
├─ Generate Trading Signals        │
│  ├─ 15min window                 │
│  ├─ 1hour window                 │
│  └─ 1day window                  │
│                                  │
├─ Check Market Confirmation       │
│  ├─ Volume spike check           │
│  ├─ Price direction check        │
│  └─ Breakout validation          │
│                                  │
├─ Size Positions                  │
│  ├─ Calculate based on strength  │
│  ├─ Respect risk limits          │
│  └─ Determine entry price        │
│                                  │
├─ Execute Trades                  │
│  ├─ Live: TWS execution          │
│  ├─ Paper: Log only              │
│  └─ Bid/ask fill determination   │
│                                  │
├─ Calculate Stats                 │
│  ├─ Portfolio value              │
│  ├─ Realized/Unrealized P&L      │
│  ├─ Win rate                     │
│  └─ Position tracking            │
│                                  │
├─ Display Statistics              │
│  ├─ Connection status            │
│  ├─ Active positions             │
│  ├─ P&L breakdown                │
│  └─ Trade count                  │
│                                  │
└─ Sleep & Loop ────────────────────
       ↓
   [--max-iterations check]
       │
       └─→ Stop or Continue
```

## Trading Logic

### Order Execution Decision
```python
# Step 1: Generate sentiment signal
signal = 1  # LONG, -1 for SHORT, 0 for NEUTRAL

# Step 2: Validate with market confirmation
confirmed = confirmation_filter.check_confirmation(
    ticker='AAPL',
    sentiment_signal=1,
    current_price=180.00,
    current_volume=80_000_000,
    recent_prices=[...],
    recent_volumes=[...]
)

# Step 3: Determine execution
if confirmed == ConfirmationStatus.CONFIRMED:
    # Execute BUY order
    if live_trading:
        ask_price = market_data['AAPL']['ask']  # Use bid/ask from TWS
        order = app.place_order('AAPL', qty, ask_price, 'BUY')
    else:
        print(f"[PAPER] BUY AAPL @ {market_data['AAPL']['price']}")
```

## Performance Monitoring

### Key Metrics
- **Win Rate**: Percentage of profitable trades
- **Total P&L**: Cumulative profit/loss
- **Daily P&L**: Today's profit/loss
- **Unrealized P&L**: Current open position gain/loss
- **Total Trades**: Number of executed trades
- **Portfolio Value**: Current account equity

### Position Details
Each position shows:
- **Side**: LONG or SHORT
- **Size**: Number of shares
- **Entry Price**: Original purchase/short price
- **Current Price**: Latest market price from TWS
- **P&L**: Dollar profit/loss
- **P&L %**: Percentage return

## Troubleshooting

### "Warning: myIBApp not found"
- Make sure `myIBApp.py` is in parent directory
- Ensure TWS is running if you want live trading
- Strategy falls back to paper trading

### "Failed to connect to TWS"
```bash
# Check TWS is running at correct port
# Default paper: 127.0.0.1:7496
# Default live: 127.0.0.1:7497

# Try explicit port
python3 strategy_orchestrator.py --live --port 7496 --debug
```

### "No market data received"
- Verify TWS connection active
- Check app.py has market data subscriptions
- Monitor with `--debug` flag for details

### Trades not executing
- Verify `--live` flag is set
- Check sufficient buying power in account
- Review order rejection reasons in TWS

## Examples

### Example 1: Default Hourly Paper Trading
```bash
python3 strategy_orchestrator.py --max-iterations 5
```
Output: 5 iterations of strategy, 1 hour apart, paper trading mode

### Example 2: Live Hourly Trading
```bash
python3 strategy_orchestrator.py --live
```
Output: Continuous hourly strategy with live TWS orders

### Example 3: Aggressive Testing (Every 10 Seconds)
```bash
python3 strategy_orchestrator.py --interval 10 --max-iterations 3 --debug
```
Output: 3 quick iterations for testing, verbose output

### Example 4: Mid-Frequency Trading (Every 15 Minutes)
```bash
python3 strategy_orchestrator.py --live --interval 900 --debug
```
Output: Live trading every 15 minutes with debug output

## Integration with Other Components

### Combined with myBreakoutSignal
```python
from sentiment.sentiment_signal import SentimentSignal
from myBreakoutSignal import breakout_signal

# Use sentiment as additional confirmation for breakout
if breakout_signal(prices, lookback) and sentiment_direction == 1:
    # Strong signal: execute full position
    execute_trade(full_size)
else:
    # Reduced confidence: half position
    execute_trade(half_size)
```

### Combined with geneticOptimize
```python
# Optimize sentiment thresholds alongside breakout parameters
params = {
    'duration_str': '1 W',
    'bar_size': '15 mins',
    'sentiment_long_threshold': 0.65,  # Vary this
    'sentiment_short_threshold': -0.65  # And this
}
# Use geneticOptimize.py to find best values
```

## Next Steps

1. **Paper Trade**: Run 1-2 weeks of paper trading to verify
2. **Monitor Stats**: Watch win rate, daily P&L, portfolio growth
3. **Optimize**: Adjust LONG/SHORT thresholds, timeframes
4. **Go Live**: Switch to `--live` with small position sizes
5. **Scale Up**: Gradually increase position sizes as performance validates

## Security Notes

- Never hardcode credentials
- Use localhost connection only
- Store API credentials in TWS configuration
- Test with paper trading first
- Start with small position sizes
- Monitor positions constantly
- Use daily/weekly loss limits
