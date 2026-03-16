# Sentiment-Based Trading Strategy

This module implements a comprehensive news sentiment trading system that combines financial news analysis with market confirmation filters and portfolio management.

## Architecture

```
News Events
    ↓
[news_feed.py] ← Ingests articles, deduplicates, tags by ticker
    ↓
[sentiment_scorer.py] ← Finance-aware sentiment scoring (-1 to +1)
    ↓
[sentiment_signal.py] ← Aggregates into LONG/SHORT/NEUTRAL signals
    ↓
[confirmation_filter.py] ← Validates with price/volume confirmation
    ↓
[portfolio_manager.py] ← Position sizing, sector balance, risk limits
    ↓
[strategy_orchestrator.py] ← Main loop, order execution, tracking
```

## Components

### 1. **sentiment_scorer.py** - Finance-Aware Sentiment Analysis
- Keyword-based scoring with finance-specific lexicon
- Outputs: polarity (-1 to 1), strength (0-1), relevance, novelty
- Combines multiple scoring dimensions into weighted score
- Handles emphasis modifiers (significantly, dramatically, etc.)
- High-impact event detection (earnings, M&A, regulatory)

**Usage:**
```python
scorer = SentimentScorer()
result = scorer.score("Breaking: Company beats earnings", "AAPL Earnings Beat")
# Returns: {polarity, strength, relevance, novelty, raw_score}
```

### 2. **news_feed.py** - Article Management
- In-memory news feed with deduplication (prevents duplicate trades)
- Event tagging: ticker, source, event_type
- Time-windowed filtering
- Mock API for testing (replace with newsapi.org or finnhub.io)

**Features:**
- Threadsafe article ingestion
- Rolling dedup cache (3600s default)
- Sorted by recency
- Tag-based filtering

### 3. **sentiment_signal.py** - Signal Generation
- Aggregates articles into rolling windows (15min, 1hour, 1day)
- Generates LONG/SHORT/NEUTRAL signals based on thresholds
- Hysteresis to prevent signal flipping on noise
- Ranks signals by strength

**Thresholds:**
- LONG: sentiment_score > 0.65
- SHORT: sentiment_score < -0.65
- NEUTRAL: within ±0.15 range

### 4. **confirmation_filter.py** - Market Validation
Confirms sentiment signals with market data:

**Volume Check:**
- Requires abnormal volume (>1.5x average)
- Prevents trading on low-volume sentiment spikes

**Price Direction Check:**
- LONG signals require upward price momentum
- SHORT signals require downward price momentum
- 5-bar lookback for trend confirmation

**Breakout Check:**
- LONG requires new 20-bar high
- SHORT requires new 20-bar low
- Validates price is actually moving on sentiment

### 5. **portfolio_manager.py** - Risk Management
Position sizing and portfolio construction:

**Risk Limits:**
- $100,000 max per position
- $300,000 max per sector
- $1,000,000 max total portfolio
- $50,000 daily loss limit
- 5% hard stop per position
- 7-day max holding period

**Position Sizing:**
```python
size = calculate_position_size(
    ticker='AAPL',
    entry_price=150.0,
    signal_strength=0.8  # Confidence 0-1
)
# Scales position by signal strength and applies all limits
```

**Rebalancing:**
- Rank positions by signal strength
- Sector exposure tracking
- Automatic position reduction to meet limits

### 6. **strategy_orchestrator.py** - Main Engine
Event loop orchestrating the full strategy:

**Every Iteration:**
1. Load mock or real news articles
2. Generate sentiment signals
3. Check market confirmation
4. Evaluate opportunities
5. Generate trade orders
6. Execute (paper or live)
7. Track positions and P&L

**Interfaces:**
```python
orchestrator = SentimentStrategyOrchestrator(
    paper_trading=True,
    debug=False
)

# Ingest news
orchestrator.ingest_news(
    headline="Apple Beats Earnings",
    text="...",
    ticker="AAPL",
    event_type="earnings"
)

# Update market data
orchestrator.update_market_data("AAPL", price=150.0, volume=80_000_000)

# Generate trade opportunities
opportunities = orchestrator.evaluate_opportunities()
trades = orchestrator.generate_trades(opportunities)
executed = orchestrator.execute_trades(trades)

# Run continuous loop
orchestrator.run_loop(interval=60, max_iterations=100)
```

## Running the Strategy

### Test Individual Components
```bash
# Test sentiment scoring
python3 sentiment_scorer.py --test

# Test news feed
python3 news_feed.py --test

# Test signal generation
python3 sentiment_signal.py --test

# Test confirmation filter
python3 confirmation_filter.py --test

# Test portfolio manager
python3 portfolio_manager.py --test
```

### Run Full Strategy
```bash
# Paper trading (default)
python3 strategy_orchestrator.py --interval 10 --max-iterations 5

# Debug mode
python3 strategy_orchestrator.py --debug --interval 5 --max-iterations 2

# Live trading (requires TWS connected)
python3 strategy_orchestrator.py --live --interval 60
```

## Signal Flow Example

**Input:** News article "Apple Beats Q3 Earnings With Strong iPhone Sales"

**Processing:**
1. **Sentiment Score:** +0.75 (beats, strong = positive)
2. **Window Aggregation:** 1-hour rolling = +0.68 (multiple articles)
3. **Signal Generation:** LONG (> 0.65 threshold)
4. **Market Validation:**
   - Volume: 85M vs 50M avg ✓
   - Price: Up 2% last 5 bars ✓
   - Breakout: New 20-bar high ✓
5. **Confirmation:** CONFIRMED
6. **Action:** BUY AAPL at current price
7. **Position Size:** 67 shares (based on 0.75 signal strength)

## Integration with myIBApp

The strategy is designed to work with the rangerwizard trading system:

**Data Connection:**
```python
# In strategy_orchestrator.py (future enhancement)
from myIBApp import MyIBApp

app = MyIBApp()
app.connect('127.0.0.1', 7496)

# Subscribe to real-time news (via newsapi or similar)
# Update market data from TWS
orchestrator.update_market_data(ticker, price, volume)

# Execute orders via myIBApp
order = app.place_order(ticker, quantity, price, 'BUY')
```

## Customization

**Adjusting Thresholds:**
```python
# In sentiment_signal.py
SentimentSignal.LONG_THRESHOLD = 0.70      # Stricter
SentimentSignal.SHORT_THRESHOLD = -0.70
SentimentSignal.NEUTRAL_RANGE = 0.10       # Narrower
```

**Changing Risk Limits:**
```python
# In portfolio_manager.py
PortfolioManager.MAX_POSITION_SIZE = 150_000  # Larger per position
PortfolioManager.MAX_DAILY_LOSS = 100_000     # Higher daily stop
PortfolioManager.HARD_STOP_PCT = 0.03         # 3% stops
```

**Adding Sectors:**
```python
# In portfolio_manager.py
PortfolioManager.SECTOR_MAP = {
    'YOUR_TICKER': 'Your Sector',
    ...
}
```

## Data Requirements

**Market Data (per stock):**
- Current price
- Current volume
- Last 50 bars of prices and volumes (for confirmation)

**News Data:**
- Headline
- Full text
- Ticker reference
- Timestamp
- Source (for filtering)

## Testing with Mock Data

All components can run with MockNewsAPI without real news feed integration:

```python
from news_feed import MockNewsAPI

# Get sample articles
articles = MockNewsAPI.get_articles()

# Process with sentiment
signal_gen = SentimentSignal()
signals = signal_gen.process_articles(articles)

# Check confirmations
for ticker, market_data in market_prices.items():
    filter_obj.check_confirmation(ticker, 1, price, volume, prices, volumes)
```

## Performance Metrics

The strategy tracks:
- Total trades executed
- P&L per position and per sector
- Win rate (estimated from sentiment confirmation)
- Position holding periods
- Sector exposure distribution
- Daily/cumulative losses vs limits

## Future Enhancements

1. **Real News Integration**
   - newsapi.org for headline feeds
   - finnhub.io for real-time events
   - Web scraping for company-specific news

2. **Advanced NLP**
   - FinBERT transformer models
   - Context-aware sentiment
   - Entity extraction (people, contracts, etc.)

3. **TWS Integration**
   - Live order placement via myIBApp
   - Real-time market data subscription
   - Position reconciliation

4. **Portfolio Construction**
   - Long/short market-neutral pairs
   - Sector decile split
   - Correlation analysis

5. **Risk Management**
   - Dynamic position sizing based on volatility
   - Correlation-adjusted limits
   - Monte Carlo scenario testing

## Architecture Notes

- All components are modular and independently testable
- Thread-safe (uses locks for shared state)
- Paper trading mode for backtesting
- Extensible to real-time news feeds
- Compatible with Interactive Brokers TWS API (myIBApp)
