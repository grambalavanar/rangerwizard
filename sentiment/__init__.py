"""
Sentiment-Based Trading Strategy Module

Combines news sentiment analysis with market confirmation to generate
trading signals using the Interactive Brokers TWS API.
"""

from .sentiment_scorer import SentimentScorer
from .news_feed import NewsFeed, MockNewsAPI
from .sentiment_signal import SentimentSignal, TradeSignal
from .confirmation_filter import ConfirmationFilter, ConfirmationStatus
from .portfolio_manager import PortfolioManager
from .strategy_orchestrator import SentimentStrategyOrchestrator

__version__ = "1.0.0"
__all__ = [
    'SentimentScorer',
    'NewsFeed',
    'MockNewsAPI',
    'SentimentSignal',
    'TradeSignal',
    'ConfirmationFilter',
    'ConfirmationStatus',
    'PortfolioManager',
    'SentimentStrategyOrchestrator'
]
