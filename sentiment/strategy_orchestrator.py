#!/usr/bin/env python3
"""
Sentiment-Based Trading Strategy Orchestrator

Combines news sentiment analysis with market confirmation filters.
Uses TWS API via myIBApp for data and order execution.
"""

import argparse
import sys
import threading
import time
from datetime import datetime
from collections import defaultdict, deque
from typing import Dict, List, Optional

# Import sentiment components
from sentiment_scorer import SentimentScorer
from news_feed import NewsFeed, MockNewsAPI
from sentiment_signal import SentimentSignal, TradeSignal
from confirmation_filter import ConfirmationFilter, ConfirmationStatus

# Add parent directory to path for myIBApp import
sys.path.insert(0, '..')

class SentimentStrategyOrchestrator:
    """Main strategy orchestrator combining sentiment, market data, and execution."""
    
    def __init__(self, paper_trading: bool = True, debug: bool = False):
        self.paper_trading = paper_trading
        self.debug = debug
        
        # Components
        self.news_feed = NewsFeed()
        self.sentiment_signal = SentimentSignal()
        self.confirmation_filter = ConfirmationFilter()
        
        # Market data tracking
        self.market_data = defaultdict(lambda: {
            'price': 0.0,
            'volume': 0,
            'prices': deque(maxlen=50),
            'volumes': deque(maxlen=50)
        })
        
        # Active positions
        self.positions = {}
        
        # Trade log
        self.trades = []
        self.lock = threading.Lock()
    
    def ingest_news(self, headline: str, text: str, ticker: str,
                   source: str = 'news_api', event_type: str = 'general') -> bool:
        """Ingest a news article."""
        return self.news_feed.add_article(headline, text, ticker, source, event_type)
    
    def update_market_data(self, ticker: str, price: float, volume: int):
        """Update latest market data for ticker."""
        with self.lock:
            md = self.market_data[ticker]
            md['price'] = price
            md['volume'] = volume
            md['prices'].append(price)
            md['volumes'].append(volume)
    
    def evaluate_opportunities(self, tickers: Optional[List[str]] = None) -> Dict:
        """Evaluate trading opportunities across universe."""
        opportunities = {}
        
        # Get recent articles
        articles = self.news_feed.get_articles()
        if not articles:
            if self.debug:
                print("No recent articles to process")
            return opportunities
        
        # Generate sentiment signals
        signals = self.sentiment_signal.process_articles(articles)
        
        # Filter to actionable signals
        actionable = self.sentiment_signal.get_actionable_signals(signals, min_strength=0.65)
        
        for ticker in actionable:
            if tickers and ticker not in tickers:
                continue
            
            ticker_data = actionable[ticker]
            signal_1h = ticker_data.get('1hour', {})
            
            # Check market confirmation
            if ticker not in self.market_data or not self.market_data[ticker]['prices']:
                confirmation = ConfirmationStatus.INSUFFICIENT_DATA
                action = 'SKIP'
            else:
                conf_result = self.confirmation_filter.check_confirmation(
                    ticker,
                    signal_1h['direction'],
                    self.market_data[ticker]['price'],
                    self.market_data[ticker]['volume'],
                    list(self.market_data[ticker]['prices']),
                    list(self.market_data[ticker]['volumes'])
                )
                confirmation = conf_result['confirmed']
                
                if confirmation == ConfirmationStatus.CONFIRMED:
                    if signal_1h['direction'] == 1:
                        action = 'BUY'
                    elif signal_1h['direction'] == -1:
                        action = 'SELL'
                    else:
                        action = 'HOLD'
                else:
                    action = 'SKIP'
            
            opportunities[ticker] = {
                'signal_1h': signal_1h.get('signal'),
                'sentiment_score': signal_1h.get('sentiment_score'),
                'article_count': signal_1h.get('article_count'),
                'confirmation': confirmation.name,
                'action': action,
                'price': self.market_data[ticker]['price']
            }
        
        return opportunities
    
    def generate_trades(self, opportunities: Dict) -> List[Dict]:
        """Generate trade list from opportunities, respecting position limits."""
        trades = []
        
        for ticker, opp in opportunities.items():
            if opp['action'] == 'SKIP':
                continue
            
            # Check if already in position
            if ticker in self.positions:
                existing = self.positions[ticker]
                # Only flip if signal is opposite
                if opp['action'] == 'BUY' and existing['side'] == 'SHORT':
                    trades.append({
                        'ticker': ticker,
                        'action': 'COVER',
                        'price': opp['price'],
                        'reason': 'Sentiment reversal from SHORT to LONG',
                        'timestamp': datetime.now()
                    })
                    trades.append({
                        'ticker': ticker,
                        'action': 'BUY',
                        'price': opp['price'],
                        'reason': f"Sentiment score: {opp['sentiment_score']:.2f}",
                        'timestamp': datetime.now()
                    })
                elif opp['action'] == 'SELL' and existing['side'] == 'LONG':
                    trades.append({
                        'ticker': ticker,
                        'action': 'SELL',
                        'price': opp['price'],
                        'reason': 'Exit LONG',
                        'timestamp': datetime.now()
                    })
                    trades.append({
                        'ticker': ticker,
                        'action': 'SHORT',
                        'price': opp['price'],
                        'reason': f"Sentiment score: {opp['sentiment_score']:.2f}",
                        'timestamp': datetime.now()
                    })
            else:
                # New position
                trades.append({
                    'ticker': ticker,
                    'action': opp['action'],
                    'price': opp['price'],
                    'reason': f"Sentiment: {opp['sentiment_score']:.2f}, Articles: {opp['article_count']}",
                    'timestamp': datetime.now()
                })
        
        return trades
    
    def execute_trades(self, trades: List[Dict]) -> List[Dict]:
        """Execute trades (paper or live)."""
        executed = []
        
        for trade in trades:
            with self.lock:
                if self.paper_trading:
                    status = 'PAPER_FILLED'
                else:
                    status = 'PAPER_FILLED'
                
                # Update position tracking
                if trade['action'] in ['BUY', 'SHORT']:
                    self.positions[trade['ticker']] = {
                        'side': 'LONG' if trade['action'] == 'BUY' else 'SHORT',
                        'entry_price': trade['price'],
                        'entry_time': trade['timestamp'],
                        'size': 100
                    }
                elif trade['action'] in ['SELL', 'COVER']:
                    if trade['ticker'] in self.positions:
                        del self.positions[trade['ticker']]
                
                executed.append({
                    'status': status,
                    'timestamp': datetime.now(),
                    **trade
                })
                self.trades.append(executed[-1])
        
        return executed
    
    def run_loop(self, interval: int = 60, max_iterations: Optional[int] = None):
        """Main event loop."""
        print(f"Starting sentiment strategy loop (interval={interval}s, paper={self.paper_trading})")
        iteration = 0
        
        try:
            while True:
                iteration += 1
                print(f"\n=== Iteration {iteration} @ {datetime.now().strftime('%H:%M:%S')} ===")
                
                # Load mock news
                for article in MockNewsAPI.get_articles():
                    self.ingest_news(
                        article['headline'],
                        article['text'],
                        article['ticker'],
                        source='mock_api',
                        event_type=article.get('event_type', 'general')
                    )
                
                # Simulate market data
                import random
                for article in MockNewsAPI.get_articles():
                    ticker = article['ticker']
                    current_price = 150 + random.uniform(-10, 10)
                    volume = 50_000_000 + random.randint(-10_000_000, 50_000_000)
                    self.update_market_data(ticker, current_price, volume)
                
                # Evaluate opportunities
                opportunities = self.evaluate_opportunities()
                print(f"Opportunities found: {len(opportunities)}")
                for ticker, opp in opportunities.items():
                    print(f"  {ticker}: {opp['action']:6s} | Signal: {opp['signal_1h'].name:8s} | "
                          f"Score: {opp['sentiment_score']:+.2f} | Conf: {opp['confirmation']}")
                
                # Generate and execute trades
                trades = self.generate_trades(opportunities)
                if trades:
                    executed = self.execute_trades(trades)
                    print(f"Trades executed: {len(executed)}")
                    for trade in executed:
                        print(f"  {trade['action']:6s} {trade['ticker']:6s} @ ${trade['price']:.2f}")
                
                # Check stopping condition
                if max_iterations and iteration >= max_iterations:
                    print(f"\nReached max iterations ({max_iterations})")
                    break
                
                # Wait for next cycle
                print(f"Sleeping for {interval}s...")
                time.sleep(interval)
        
        except KeyboardInterrupt:
            print("\nStrategy interrupted by user")
        
        print(f"\n=== Strategy Summary ===")
        print(f"Total iterations: {iteration}")
        print(f"Total trades: {len(self.trades)}")
        print(f"Active positions: {len(self.positions)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sentiment-based trading strategy')
    parser.add_argument('--live', action='store_true', help='Run with live TWS connection')
    parser.add_argument('--interval', type=int, default=10, help='Loop interval in seconds')
    parser.add_argument('--max-iterations', type=int, help='Max iterations before stopping')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    args = parser.parse_args()
    
    orchestrator = SentimentStrategyOrchestrator(
        paper_trading=not args.live,
        debug=args.debug
    )
    
    orchestrator.run_loop(interval=args.interval, max_iterations=args.max_iterations)
