import argparse
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from collections import defaultdict, deque
import enum

from sentiment_scorer import SentimentScorer

class TradeSignal(enum.Enum):
    """Trade signal directions"""
    LONG = 1
    NEUTRAL = 0
    SHORT = -1

class SentimentSignal:
    """Aggregates news articles into rolling sentiment signals."""
    
    # Thresholds for signal generation
    LONG_THRESHOLD = 0.65
    SHORT_THRESHOLD = -0.65
    NEUTRAL_RANGE = 0.15
    
    # Rolling window sizes (in seconds)
    WINDOWS = {
        '15min': 15 * 60,
        '1hour': 60 * 60,
        '1day': 24 * 60 * 60
    }
    
    def __init__(self):
        self.scorer = SentimentScorer()
        self.rolling_scores = defaultdict(lambda: defaultdict(lambda: deque(maxlen=100)))
        self.last_signal = defaultdict(lambda: {w: TradeSignal.NEUTRAL for w in self.WINDOWS})
    
    def process_articles(self, articles: List[Dict]) -> Dict[str, Dict]:
        """Process articles and generate rolling sentiment signals per ticker."""
        signals = {}
        
        # Group articles by ticker
        by_ticker = defaultdict(list)
        for article in articles:
            by_ticker[article['ticker']].append(article)
        
        # Process each ticker
        for ticker, ticker_articles in by_ticker.items():
            signals[ticker] = {}
            
            # Score each article
            scored_articles = []
            for article in ticker_articles:
                sentiment_data = self.scorer.score(
                    article['text'],
                    article['headline']
                )
                weighted = self.scorer.weighted_score(sentiment_data)
                scored_articles.append({
                    'article': article,
                    'sentiment_score': weighted,
                    'timestamp': article.get('timestamp', datetime.now())
                })
            
            # Add to rolling windows and generate signals
            current_time = datetime.now()
            for window_name, window_seconds in self.WINDOWS.items():
                cutoff_time = current_time - timedelta(seconds=window_seconds)
                
                # Add new scores to rolling window
                for scored in scored_articles:
                    if scored['timestamp'] > cutoff_time:
                        self.rolling_scores[ticker][window_name].append(
                            scored['sentiment_score']
                        )
                
                # Calculate rolling average
                window_scores = list(self.rolling_scores[ticker][window_name])
                if window_scores:
                    avg_sentiment = sum(window_scores) / len(window_scores)
                else:
                    avg_sentiment = 0.0
                
                # Generate signal
                signal = self._generate_signal(avg_sentiment, ticker, window_name)
                
                signals[ticker][window_name] = {
                    'signal': signal,
                    'sentiment_score': avg_sentiment,
                    'article_count': len(window_scores),
                    'direction': signal.value
                }
        
        return signals
    
    def _generate_signal(self, sentiment_score: float, ticker: str, 
                        window: str) -> TradeSignal:
        """Generate signal with hysteresis to avoid flipping on noise."""
        last = self.last_signal[ticker][window]
        
        # Check thresholds
        if sentiment_score > self.LONG_THRESHOLD:
            signal = TradeSignal.LONG
        elif sentiment_score < self.SHORT_THRESHOLD:
            signal = TradeSignal.SHORT
        elif sentiment_score > self.NEUTRAL_RANGE:
            signal = TradeSignal.LONG if last != TradeSignal.SHORT else last
        elif sentiment_score < -self.NEUTRAL_RANGE:
            signal = TradeSignal.SHORT if last != TradeSignal.LONG else last
        else:
            signal = TradeSignal.NEUTRAL
        
        # Update last signal
        self.last_signal[ticker][window] = signal
        return signal
    
    def get_strongest_signals(self, signals: Dict[str, Dict], 
                            window: str = '1hour',
                            exclude_neutral: bool = True) -> List[Tuple[str, float, TradeSignal]]:
        """Get tickers ranked by signal strength within a window."""
        ranked = []
        for ticker, windows_data in signals.items():
            if window in windows_data:
                w_data = windows_data[window]
                signal = w_data['signal']
                score = w_data['sentiment_score']
                
                if exclude_neutral and signal == TradeSignal.NEUTRAL:
                    continue
                
                ranked.append((ticker, score, signal))
        
        ranked.sort(key=lambda x: abs(x[1]), reverse=True)
        return ranked
    
    def get_actionable_signals(self, signals: Dict[str, Dict],
                              min_strength: float = 0.7) -> Dict[str, Dict]:
        """Filter to only actionable signals meeting minimum criteria."""
        actionable = {}
        for ticker, windows_data in signals.items():
            if '1hour' in windows_data:
                h_data = windows_data['1hour']
                if (h_data['signal'] != TradeSignal.NEUTRAL and 
                    abs(h_data['sentiment_score']) >= min_strength and
                    h_data['article_count'] >= 2):
                    actionable[ticker] = windows_data
        
        return actionable


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate sentiment signals from articles')
    parser.add_argument('--test', action='store_true', help='Test with sample articles')
    args = parser.parse_args()
    
    if args.test:
        from news_feed import MockNewsAPI
        
        mock_articles = MockNewsAPI.get_articles()
        signal_gen = SentimentSignal()
        signals = signal_gen.process_articles(mock_articles)
        
        print("\n=== Sentiment Signals ===")
        for ticker, windows_data in signals.items():
            print(f"\n{ticker}:")
            for window, data in windows_data.items():
                print(f"  {window:6s} | Signal: {data['signal'].name:8s} | "
                      f"Score: {data['sentiment_score']:+.2f} | "
                      f"Articles: {data['article_count']}")
        
        print("\n=== Strongest Signals ===")
        strongest = signal_gen.get_strongest_signals(signals, window='1hour')
        for ticker, score, signal in strongest:
            print(f"{ticker:6s} | {signal.name:8s} | Score: {score:+.2f}")
        
        print("\n=== Actionable Signals ===")
        actionable = signal_gen.get_actionable_signals(signals)
        for ticker in actionable:
            print(f"{ticker}")
