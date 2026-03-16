import argparse
import time
from datetime import datetime
from typing import List, Dict, Optional
import threading
from collections import deque

class NewsFeed:
    """
    Ingests financial news articles for sentiment analysis.
    Deduplicates articles, tracks novelty, tags with ticker/event_type.
    """
    
    def __init__(self, max_articles: int = 1000, dedup_window: int = 3600):
        """
        Args:
            max_articles: Maximum articles to keep in memory
            dedup_window: Time window in seconds for deduplication
        """
        self.articles = deque(maxlen=max_articles)
        self.dedup_window = dedup_window
        self.dedup_cache = {}
        self.lock = threading.Lock()
    
    def add_article(self, headline: str, text: str, ticker: str, 
                   source: str = 'unknown', event_type: str = 'general') -> bool:
        """Add article if not duplicate. Returns True if added, False if duplicate."""
        article_hash = self._create_hash(headline, text)
        
        with self.lock:
            current_time = time.time()
            
            # Check if duplicate
            if article_hash in self.dedup_cache:
                cached_time, _ = self.dedup_cache[article_hash]
                if current_time - cached_time < self.dedup_window:
                    return False
            
            # Clean old dedup cache entries
            self.dedup_cache[article_hash] = (current_time, headline)
            expired = [k for k, (t, _) in self.dedup_cache.items() 
                       if current_time - t > self.dedup_window * 2]
            for k in expired:
                del self.dedup_cache[k]
            
            # Add to articles
            article = {
                'headline': headline,
                'text': text,
                'ticker': ticker,
                'source': source,
                'event_type': event_type,
                'timestamp': datetime.now(),
                'hash': article_hash
            }
            self.articles.append(article)
            return True
    
    def get_articles(self, ticker: Optional[str] = None, 
                    since_seconds: int = 3600) -> List[Dict]:
        """Get articles, optionally filtered by ticker and time."""
        with self.lock:
            current_time = datetime.now()
            results = []
            
            for article in self.articles:
                age = (current_time - article['timestamp']).total_seconds()
                if age > since_seconds:
                    continue
                
                if ticker and article['ticker'].upper() != ticker.upper():
                    continue
                
                results.append(article)
            
            return results
    
    def get_latest(self, count: int = 10, ticker: Optional[str] = None) -> List[Dict]:
        """Get most recent articles."""
        with self.lock:
            articles = list(self.articles)
        
        if ticker:
            articles = [a for a in articles if a['ticker'].upper() == ticker.upper()]
        
        return articles[-count:] if len(articles) > 0 else []
    
    def _create_hash(self, headline: str, text: str) -> str:
        """Create dedup hash from headline and first 100 chars of text."""
        import hashlib
        content = (headline + text[:100]).lower().strip()
        return hashlib.md5(content.encode()).hexdigest()


class MockNewsAPI:
    """
    Mock news API for testing sentiment strategy without external API calls.
    """
    
    MOCK_ARTICLES = [
        {
            'headline': 'Apple Beats Q3 Earnings Estimates With Strong iPhone Sales',
            'text': 'Apple Inc. significantly beat Wall Street earnings expectations with remarkable growth in iPhone sales.',
            'ticker': 'AAPL',
            'event_type': 'earnings'
        },
        {
            'headline': 'Tesla Misses Revenue Targets, Stock Slides',
            'text': 'Tesla dramatically missed revenue expectations in latest earnings report. The company warned of challenging market conditions.',
            'ticker': 'TSLA',
            'event_type': 'earnings'
        },
        {
            'headline': 'Microsoft Acquires AI Startups for $10 Billion',
            'text': 'Microsoft announced a major acquisition of leading AI companies to strengthen its cloud capabilities.',
            'ticker': 'MSFT',
            'event_type': 'acquisition'
        },
        {
            'headline': 'Amazon Faces Antitrust Lawsuit from Federal Regulators',
            'text': 'The FTC filed a major lawsuit against Amazon, raising concerns about monopolistic practices.',
            'ticker': 'AMZN',
            'event_type': 'regulatory'
        }
    ]
    
    @staticmethod
    def get_articles(keywords: str = '', limit: int = 10) -> List[Dict]:
        """Mock API call - returns sample articles."""
        return MockNewsAPI.MOCK_ARTICLES[:limit]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='News feed for sentiment analysis')
    parser.add_argument('--test', action='store_true', help='Test with mock data')
    args = parser.parse_args()
    
    feed = NewsFeed()
    
    if args.test:
        for article in MockNewsAPI.get_articles():
            success = feed.add_article(
                article['headline'],
                article['text'],
                article['ticker'],
                source='mock_api',
                event_type=article['event_type']
            )
            print(f"Added: {article['headline'][:60]}... ({success})")
        
        print("\nAPPL Articles:")
        for article in feed.get_articles(ticker='AAPL'):
            print(f"  - {article['headline']}")
        
        print(f"\nTotal articles: {len(list(feed.articles))}")
