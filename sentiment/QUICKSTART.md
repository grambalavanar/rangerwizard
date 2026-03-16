# Sentiment Strategy Quick Start Guide

## 1. Basic Setup

```python
from sentiment_scorer import SentimentScorer
from news_feed import NewsFeed, MockNewsAPI
from sentiment_signal import SentimentSignal
from confirmation_filter import ConfirmationFilter
from portfolio_manager import PortfolioManager
from strategy_orchestrator import SentimentStrategyOrchestrator
```

## 2. Score a Single Article

```python
from sentiment_scorer import SentimentScorer

scorer = SentimentScorer()

# Score an article
article_text = "Apple beat earnings expectations with strong iPhone sales"
headline = "Apple Reports Record Q3 Earnings"

result = scorer.score(article_text, headline)

print(f"Polarity: {result['polarity']:.2f}")      # -1 to +1
print(f"Strength: {result['strength']:.2f}")      # 0 to 1
print(f"Relevance: {result['relevance']:.2f}")    # 0 to 1
print(f"Novelty: {result['novelty']:.2f}")        # 0 to 1
print(f"Weighted Score: {result['raw_score']:.2f}") # Combined

# Output might be:
# Polarity: 0.85
# Strength: 0.60
# Relevance: 0.80
# Novelty: 0.60
# Weighted Score: 0.72
```

## 3. Manage Articles from News Feed

```python
from news_feed import NewsFeed, MockNewsAPI

feed = NewsFeed()

# Get mock articles
articles = MockNewsAPI.get_articles()

# Add articles to feed (deduplicates automatically)
for article in articles:
    added = feed.add_article(
        headline=article['headline'],
        text=article['text'],
        ticker=article['ticker'],
        source='news_api',
        event_type=article.get('event_type', 'general')
    )
    if added:
        print(f"Added: {article['headline'][:50]}...")

# Retrieve articles
recent = feed.get_articles(ticker='AAPL', since_seconds=3600)
print(f"Found {len(recent)} AAPL articles in last hour")

# Get most recent articles
latest = feed.get_latest(count=10, ticker='AAPL')
for article in latest:
    print(f"  - {article['headline']}")
```

## 4. Generate Trading Signals

```python
from sentiment_signal import SentimentSignal, TradeSignal
from news_feed import MockNewsAPI

signal_gen = SentimentSignal()

# Get articles
articles = MockNewsAPI.get_articles()

# Process articles and generate signals
signals = signal_gen.process_articles(articles)

# Examine signals by ticker
for ticker, windows_data in signals.items():
    print(f"\n{ticker}:")
    for window_name, data in windows_data.items():
        signal = data['signal']
        score = data['sentiment_score']
        
        if signal == TradeSignal.LONG:
            print(f"  {window_name}: BUY (score: {score:+.2f})")
        elif signal == TradeSignal.SHORT:
            print(f"  {window_name}: SELL (score: {score:+.2f})")
        else:
            print(f"  {window_name}: HOLD (score: {score:+.2f})")

# Get only actionable signals (high confidence)
actionable = signal_gen.get_actionable_signals(signals, min_strength=0.7)
print(f"\nActionable signals: {list(actionable.keys())}")

# Rank by strength
strongest = signal_gen.get_strongest_signals(signals, window='1hour')
for ticker, score, signal in strongest:
    print(f"{ticker}: {signal.name} ({score:+.2f})")
```

## 5. Validate with Market Confirmation

```python
from confirmation_filter import ConfirmationFilter, ConfirmationStatus

filter_obj = ConfirmationFilter()

# Market data for AAPL
current_price = 180.0
current_volume = 75_000_000
recent_prices = [170, 172, 175, 177, 180]  # Uptrend
recent_volumes = [40_000_000, 42_000_000, 45_000_000, 50_000_000, 75_000_000]  # Rising

# Check if LONG signal is confirmed
result = filter_obj.check_confirmation(
    ticker='AAPL',
    sentiment_signal=1,  # LONG
    current_price=current_price,
    current_volume=current_volume,
    recent_prices=recent_prices,
    recent_volumes=recent_volumes
)

if result['confirmed'] == ConfirmationStatus.CONFIRMED:
    print("✓ LONG signal is confirmed - trade conditions met")
else:
    print(f"✗ Signal rejected: {result['details']}")
    print(f"  Volume: {result['volume_check']}")
    print(f"  Price: {result['price_direction_check']}")
    print(f"  Breakout: {result['breakout_check']}")
```

## 6. Size Positions with Risk Management

```python
from portfolio_manager import PortfolioManager

pm = PortfolioManager(account_value=1_000_000)

# Calculate appropriate position size
entry_price = 150.0
signal_strength = 0.8  # 80% confidence

qty = pm.calculate_position_size('AAPL', entry_price, signal_strength)
print(f"Position size: {qty} shares @ ${entry_price} = ${qty * entry_price:,.0f}")

# Enter position
trade = pm.enter_position('AAPL', 'LONG', entry_price, qty, signal_strength=0.8)
print(f"Entered: {trade['action']} {trade['quantity']} @ ${trade['price']}")

# Check portfolio health
limits = pm.check_risk_limits()
print(f"Risk limits OK: {all(limits.values())}")

# Monitor positions
stats = pm.get_portfolio_stats({'AAPL': 165.0})
print(f"P&L: ${stats['total_pnl']:,.0f}")
print(f"Return: {stats['return_pct']:.2f}%")
```

## 7. Run Full Strategy Loop

```python
from strategy_orchestrator import SentimentStrategyOrchestrator

# Create orchestrator
orchestrator = SentimentStrategyOrchestrator(
    paper_trading=True,  # Paper trading mode
    debug=True
)

# Option 1: Run for fixed iterations
orchestrator.run_loop(interval=10, max_iterations=5)

# Option 2: Run until keyboard interrupt
# orchestrator.run_loop(interval=60)
```

## 8. Command Line Usage

```bash
# Run with defaults (paper trading, 10-second intervals, 5 iterations)
python3 strategy_orchestrator.py --interval 10 --max-iterations 5

# Debug mode with verbose output
python3 strategy_orchestrator.py --debug --interval 5 --max-iterations 2

# Live trading (requires TWS connection at 127.0.0.1:7496)
python3 strategy_orchestrator.py --live --interval 60

# Test individual components
python3 sentiment_scorer.py --test
python3 news_feed.py --test
python3 sentiment_signal.py --test
python3 confirmation_filter.py --test
python3 portfolio_manager.py --test
```

## 9. Integration with myIBApp

To integrate with your Interactive Brokers connection:

```python
import sys
sys.path.insert(0, '..')  # Access parent directory
from myIBApp import MyIBApp
from strategy_orchestrator import SentimentStrategyOrchestrator

# Create connections
app = MyIBApp()
app.connect('127.0.0.1', 7496)  # TWS connection

orchestrator = SentimentStrategyOrchestrator(paper_trading=False)

# In a real setup, you would:
# 1. Subscribe to real news feeds (newsapi.org, finnhub.io)
# 2. Subscribe to market data via app.subscribeMarketData()
# 3. Execute orders via app.placeOrder()
# 4. Run orchestrator.run_loop() continuously
```

## 10. Backtesting with Historical Data

```python
from news_feed import MockNewsAPI, NewsFeed
from sentiment_signal import SentimentSignal
from datetime import datetime

# Simulate historical news flow
feed = NewsFeed()
signal_gen = SentimentSignal()

for i, article in enumerate(MockNewsAPI.get_articles()):
    # Add article
    feed.add_article(
        article['headline'],
        article['text'],
        article['ticker'],
        source='historical'
    )
    
    # Generate signal at this point in time
    articles = feed.get_articles()
    signals = signal_gen.process_articles(articles)
    
    # Log signals
    print(f"[{datetime.now()}] Article {i+1}: {article['headline'][:40]}...")
    for ticker, data in signals.items():
        print(f"  {ticker}: {data['1hour']['signal'].name}")
```

## Key Insights

1. **Sentiment Alone is Not Enough**: The strategy requires market confirmation (volume, price, breakout)

2. **Hysteresis Prevents Whipsaws**: Signals don't flip on every slight sentiment change

3. **Position Sizing Matters**: Larger positions only on high-confidence signals

4. **Risk Management First**: Hard stops, daily loss limits, sector caps prevent catastrophic losses

5. **Rolling Windows**: Aggregating news over time (15min, 1hr, 1day) reduces noise

6. **Action Confirmation**: New highs/lows + volume spikes validate sentiment

## Troubleshooting

**No signals generated:**
- Check: Are articles being added to feed? (`feed.get_articles()`)
- Check: Sentiment scores extreme? (< -0.65 or > 0.65)
- Check: Have enough articles? (need 2+ per window)

**Signals not confirmed:**
- Check: Volume high enough? (need 1.5x average)
- Check: Price moving in signal direction? (5-bar trend)
- Check: Breakout happening? (new high/low)

**Positions not sized:**
- Check: Signal strength is low (scales position down)
- Check: Sector already full (capped at $300k)
- Check: Total portfolio full (capped at $1M)

## Next Steps

1. Connect real news API (replace MockNewsAPI)
2. Integrate TWS market data (via myIBApp)
3. Paper trade for 1-2 weeks to validate
4. Add position tracking and reporting
5. Backtest against historical data
6. Deploy live with position limits
