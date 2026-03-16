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
import os
from datetime import datetime
from collections import defaultdict, deque
from typing import Dict, List, Optional

# Import sentiment components
from sentiment_scorer import SentimentScorer
from news_feed import NewsFeed, MockNewsAPI
from sentiment_signal import SentimentSignal, TradeSignal
from confirmation_filter import ConfirmationFilter, ConfirmationStatus

# Add parent directory and workspace ibapi to path for myIBApp import - use absolute path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

# Add workspace ibapi location
workspace_ibapi = os.path.join(parent_dir, 'lib', 'python3.14', 'site-packages')
if os.path.exists(workspace_ibapi):
    sys.path.insert(0, workspace_ibapi)

try:
    from myIBApp import MyIBApp
    TWS_AVAILABLE = True
except ImportError as e:
    TWS_AVAILABLE = False
    # Silently fail - warning will show only if --live is requested
    pass

class SentimentStrategyOrchestrator:
    """Main strategy orchestrator combining sentiment, market data, and execution."""
    
    def __init__(self, paper_trading: bool = True, debug: bool = False, 
                 host: str = '127.0.0.1', port: int = 7496, clientId: int = 1):
        self.paper_trading = paper_trading
        self.debug = debug
        self.host = host
        self.port = port
        self.clientId = clientId
        
        # Components
        self.news_feed = NewsFeed()
        self.sentiment_signal = SentimentSignal()
        self.confirmation_filter = ConfirmationFilter()
        
        # TWS Connection
        self.app = None
        self.connected = False
        if TWS_AVAILABLE and not paper_trading:
            try:
                self.app = MyIBApp()
                self.connect_tws()
            except Exception as e:
                print(f"Failed to initialize TWS connection: {e}")
                print("Falling back to paper trading mode")
                self.paper_trading = True
        elif not paper_trading and not TWS_AVAILABLE:
            print("⚠ Live mode requested but TWS/ibapi unavailable. Using paper trading.")
            print("  To enable live trading: pip install ibapi")
            self.paper_trading = True
        
        # Market data tracking
        self.market_data = defaultdict(lambda: {
            'price': 0.0,
            'volume': 0,
            'prices': deque(maxlen=50),
            'volumes': deque(maxlen=50),
            'bid': 0.0,
            'ask': 0.0
        })
        
        # Active positions
        self.positions = {}
        
        # Trade log
        self.trades = []
        self.stats = {
            'total_trades': 0,
            'profitable_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'daily_pnl': 0.0,
            'win_rate': 0.0
        }
        
        self.lock = threading.Lock()
    
    def connect_tws(self):
        """Connect to TWS API."""
        if not self.app:
            return
        
        try:
            print(f"Connecting to TWS at {self.host}:{self.port}...")
            self.app.connect(self.host, self.port, self.clientId)
            
            # Give connection time to establish
            time.sleep(2)
            self.connected = True
            print("✓ Connected to TWS")
        except Exception as e:
            print(f"✗ Failed to connect to TWS: {e}")
            self.connected = False
    
    def disconnect_tws(self):
        """Disconnect from TWS API."""
        if self.app and self.connected:
            try:
                self.app.disconnect()
                self.connected = False
                print("✓ Disconnected from TWS")
            except Exception as e:
                print(f"Error disconnecting from TWS: {e}")
    
    def ingest_news(self, headline: str, text: str, ticker: str,
                   source: str = 'news_api', event_type: str = 'general') -> bool:
        """Ingest a news article."""
        return self.news_feed.add_article(headline, text, ticker, source, event_type)
    
    def fetch_market_data_from_tws(self, tickers: List[str]):
        """
        Fetch market data from TWS using EClient.reqMktData().
        Updates self.market_data with current prices and volumes from tick callbacks.
        """
        if not self.app or not self.connected:
            return
        
        try:
            from ibapi.contract import Contract
            
            for req_id, ticker in enumerate(tickers):
                # Create contract for market data request
                contract = Contract()
                contract.symbol = ticker
                contract.secType = "STK"
                contract.exchange = "NASDAQ"
                contract.currency = "USD"
                
                # Store mapping for callbacks
                self.app.req_id_to_ticker[req_id] = ticker
                
                # Request real-time market data from TWS via EClient API
                # This triggers tick callbacks: tickPrice(), tickSize(), etc.
                self.app.reqMktData(
                    reqId=req_id,
                    contract=contract,
                    genericTickList="",
                    snapshot=False,
                    regulatorySnapshot=False,
                    mktDataOptions=[]
                )
                
                if self.debug:
                    print(f"Requested market data for {ticker} (req_id: {req_id})")
                
                # Give TWS a moment to process
                time.sleep(0.1)
        except Exception as e:
            if self.debug:
                print(f"Error fetching market data from TWS: {e}")
    
    def ingest_news_from_tws(self):
        """
        Fetch news from TWS using EClient API (reqHistoricalNews if available).
        Falls back to mock data if TWS unavailable or returns no news.
        IBKR Market Data subscription required for news data.
        """
        articles = []
        
        # Try to get real news from TWS API via EClient
        if self.app and self.connected:
            try:
                # TWS can provide news through reqHistoricalNews() or news bulletins
                # This requires Market Data subscription (IBKR, WSJ, etc.)
                # For now, we check if tick callbacks have populated any news data
                # In production, implement: self.app.reqHistoricalNews()
                if hasattr(self.app, 'news_data') and self.app.news_data:
                    articles = self.app.news_data
                    if self.debug:
                        print(f"✓ Fetched {len(articles)} articles from TWS")
            except Exception as e:
                if self.debug:
                    print(f"Note: TWS news data unavailable: {e}")
        
        # Fall back to mock data if no real news available
        if not articles:
            try:
                articles = MockNewsAPI.get_articles()
                if self.debug and articles:
                    print(f"✓ Using {len(articles)} mock articles")
            except Exception as e:
                if self.debug:
                    print(f"Error getting mock articles: {e}")
        
        # Ingest all articles
        try:
            for article in articles:
                # Handle different field names from TWS vs mock
                headline = article.get('headline') or article.get('title') or 'No headline'
                text = article.get('text') or article.get('content') or article.get('summary', '')
                ticker = article.get('ticker') or article.get('symbol') or 'UNKNOWN'
                source = 'tws' if (self.connected and self.app and articles != MockNewsAPI.get_articles()) else 'mock'
                event_type = article.get('event_type', 'general')
                
                self.ingest_news(headline, text, ticker, source, event_type)
        except Exception as e:
            if self.debug:
                print(f"Error ingesting articles: {e}")
    
    def update_market_data(self, ticker: str, price: float, volume: int, 
                          bid: float = 0.0, ask: float = 0.0):
        """Update latest market data for ticker."""
        with self.lock:
            md = self.market_data[ticker]
            md['price'] = price
            md['volume'] = volume
            md['bid'] = bid
            md['ask'] = ask
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
    
    def calculate_portfolio_stats(self) -> Dict:
        """Calculate and return portfolio statistics."""
        total_value = 0.0
        total_pnl = 0.0
        
        for ticker, pos in self.positions.items():
            if ticker in self.market_data:
                current_price = self.market_data[ticker]['price']
                position_value = pos['size'] * current_price
                total_value += position_value
                
                if pos['side'] == 'LONG':
                    pnl = (current_price - pos['entry_price']) * pos['size']
                else:  # SHORT
                    pnl = (pos['entry_price'] - current_price) * pos['size']
                
                total_pnl += pnl
        
        # Calculate win rate
        if self.stats['total_trades'] > 0:
            self.stats['win_rate'] = (self.stats['profitable_trades'] / 
                                     self.stats['total_trades'] * 100)
        
        return {
            'total_positions': len(self.positions),
            'portfolio_value': total_value,
            'unrealized_pnl': total_pnl,
            'total_trades': self.stats['total_trades'],
            'win_rate': self.stats['win_rate'],
            'total_pnl': self.stats['total_pnl'],
            'daily_pnl': self.stats['daily_pnl']
        }
    
    def display_stats(self):
        """Display strategy statistics and current positions."""
        stats = self.calculate_portfolio_stats()
        
        print("\n" + "="*60)
        print("STRATEGY STATISTICS")
        print("="*60)
        print(f"Connection Status: {'✓ Live (TWS)' if self.connected else '✗ Paper Trading'}")
        print(f"Total Trades: {stats['total_trades']}")
        print(f"Win Rate: {stats['win_rate']:.1f}%")
        print(f"Unrealized P&L: ${stats['unrealized_pnl']:,.2f}")
        print(f"Total P&L: ${stats['total_pnl']:,.2f}")
        print(f"Daily P&L: ${stats['daily_pnl']:,.2f}")
        print(f"Portfolio Value: ${stats['portfolio_value']:,.2f}")
        print(f"Active Positions: {stats['total_positions']}")
        
        if self.positions:
            print("\n" + "-"*60)
            print("POSITIONS")
            print("-"*60)
            for ticker, pos in self.positions.items():
                if ticker in self.market_data:
                    current_price = self.market_data[ticker]['price']
                    if pos['side'] == 'LONG':
                        pnl = (current_price - pos['entry_price']) * pos['size']
                    else:
                        pnl = (pos['entry_price'] - current_price) * pos['size']
                    
                    pnl_pct = ((current_price - pos['entry_price']) / pos['entry_price'] * 100 
                              if pos['side'] == 'LONG' 
                              else (pos['entry_price'] - current_price) / pos['entry_price'] * 100)
                    
                    print(f"  {ticker}: {pos['side']:5s} {pos['size']:3d} @ ${pos['entry_price']:.2f} "
                          f"→ ${current_price:.2f} | P&L: ${pnl:,.0f} ({pnl_pct:+.1f}%)")
        
        print("="*60 + "\n")
    
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
        """Execute trades (paper or live via TWS)."""
        executed = []
        
        for trade in trades:
            with self.lock:
                # Determine execution price
                exec_price = trade['price']
                if self.app and self.connected:
                    # Use bid/ask for live trading
                    ticker = trade['ticker']
                    if ticker in self.market_data:
                        md = self.market_data[ticker]
                        if trade['action'] in ['BUY', 'COVER']:
                            exec_price = md['ask'] if md['ask'] > 0 else exec_price
                        else:
                            exec_price = md['bid'] if md['bid'] > 0 else exec_price
                
                # Execute order
                if self.paper_trading or not self.connected:
                    status = 'PAPER_FILLED'
                    if self.debug:
                        print(f"  [PAPER] {trade['action']} {trade['ticker']} @ ${exec_price:.2f}")
                else:
                    # Live execution via TWS using EClient.placeOrder()
                    try:
                        from ibapi.contract import Contract
                        from ibapi.order import Order
                        
                        qty = trade.get('quantity', 100)
                        ticker = trade['ticker']
                        
                        # Create contract
                        contract = Contract()
                        contract.symbol = ticker
                        contract.secType = "STK"
                        contract.exchange = "NASDAQ"
                        contract.currency = "USD"
                        
                        # Create order
                        order = Order()
                        order.action = trade['action'].upper()  # BUY, SELL, SSHORT
                        order.orderType = "LMT"  # Limit order at execution price
                        order.lmtPrice = exec_price
                        order.totalQuantity = qty
                        
                        # Generate next order ID (simplified - in production use nextOrderId callback)
                        order_id = int(time.time()) % 1000000
                        
                        # Place order via EClient API
                        self.app.placeOrder(order_id, contract, order)
                        status = 'LIVE_PENDING'
                        print(f"  [LIVE] {trade['action']} {ticker} x{qty} @ ${exec_price:.2f} (order_id: {order_id})")
                    except Exception as e:
                        status = 'FAILED'
                        print(f"  [ERROR] {trade['action']} {ticker}: {e}")
                
                # Update position tracking
                if trade['action'] in ['BUY', 'SHORT']:
                    self.positions[trade['ticker']] = {
                        'side': 'LONG' if trade['action'] == 'BUY' else 'SHORT',
                        'entry_price': exec_price,
                        'entry_time': trade['timestamp'],
                        'size': trade.get('quantity', 100)
                    }
                elif trade['action'] in ['SELL', 'COVER']:
                    if trade['ticker'] in self.positions:
                        del self.positions[trade['ticker']]
                
                executed.append({
                    'status': status,
                    'execution_price': exec_price,
                    'timestamp': datetime.now(),
                    **trade
                })
                self.trades.append(executed[-1])
                
                # Update stats
                self.stats['total_trades'] += 1
        
        return executed
    
    def run_loop(self, interval: int = 3600, max_iterations: Optional[int] = None):
        """Main event loop. Evaluates opportunities and executes trades."""
        mode = "LIVE (TWS)" if self.connected and not self.paper_trading else "PAPER TRADING"
        hours = interval / 3600
        print(f"\nStarting sentiment strategy loop (interval={interval}s ({hours:.1f}h), mode={mode})")
        iteration = 0
        
        try:
            while True:
                iteration += 1
                print(f"\n{'='*60}")
                print(f"Iteration {iteration} @ {datetime.now().strftime('%H:%M:%S')}")
                print(f"{'='*60}")
                
                # Get list of monitored tickers
                monitored_tickers = ['AAPL', 'TSLA', 'MSFT', 'AMZN']
                
                # Fetch real market data from TWS (if connected) or simulate
                if self.connected and self.app:
                    self.fetch_market_data_from_tws(monitored_tickers)
                else:
                    # Simulate market data for paper trading
                    import random
                    for ticker in monitored_tickers:
                        current_price = 150 + random.uniform(-10, 10)
                        volume = 50_000_000 + random.randint(-10_000_000, 50_000_000)
                        bid = current_price - 0.05
                        ask = current_price + 0.05
                        self.update_market_data(ticker, current_price, volume, bid, ask)
                
                # Ingest news articles
                self.ingest_news_from_tws()
                
                # Evaluate opportunities
                opportunities = self.evaluate_opportunities(tickers=monitored_tickers)
                print(f"\nOpportunities found: {len(opportunities)}")
                for ticker, opp in opportunities.items():
                    print(f"  {ticker}: {opp['action']:6s} | Signal: {opp['signal_1h'].name:8s} | "
                          f"Score: {opp['sentiment_score']:+.2f} | Conf: {opp['confirmation']}")
                
                # Generate and execute trades
                trades = self.generate_trades(opportunities)
                if trades:
                    executed = self.execute_trades(trades)
                    print(f"\nTrades executed: {len(executed)}")
                    for trade in executed:
                        print(f"  {trade['action']:6s} {trade['ticker']:6s} @ ${trade['execution_price']:.2f} [{trade['status']}]")
                
                # Display statistics
                self.display_stats()
                
                # Check stopping condition
                if max_iterations and iteration >= max_iterations:
                    print(f"\nReached max iterations ({max_iterations})")
                    break
                
                # Wait for next cycle
                print(f"Sleeping for {interval}s...")
                time.sleep(interval)
        
        except KeyboardInterrupt:
            print("\n\nStrategy interrupted by user")
        
        finally:
            # Cleanup
            self.disconnect_tws()
            print(f"\n{'='*60}")
            print("FINAL SUMMARY")
            print(f"{'='*60}")
            print(f"Total iterations: {iteration}")
            print(f"Total trades: {len(self.trades)}")
            print(f"Active positions: {len(self.positions)}")
            stats = self.calculate_portfolio_stats()
            print(f"Final Portfolio Value: ${stats['portfolio_value']:,.2f}")
            print(f"Total P&L: ${stats['total_pnl']:,.2f}")
            print(f"Win Rate: {stats['win_rate']:.1f}%")
            print(f"{'='*60}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sentiment-based trading strategy')
    parser.add_argument('--live', action='store_true', help='Run with live TWS connection')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='TWS host (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=7496, help='TWS port (default: 7496)')
    parser.add_argument('--client-id', type=int, default=1, help='TWS client ID (default: 1)')
    parser.add_argument('--interval', type=int, default=3600, help='Loop interval in seconds (default: 3600 = 1 hour)')
    parser.add_argument('--max-iterations', type=int, help='Max iterations before stopping')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    args = parser.parse_args()
    
    orchestrator = SentimentStrategyOrchestrator(
        paper_trading=not args.live,
        debug=args.debug,
        host=args.host,
        port=args.port,
        clientId=args.client_id
    )
    
    orchestrator.run_loop(interval=args.interval, max_iterations=args.max_iterations)
