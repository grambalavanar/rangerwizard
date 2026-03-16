import argparse
from typing import Dict, Optional, Tuple
from datetime import datetime
from enum import Enum

class ConfirmationStatus(Enum):
    """Confirmation filter results"""
    CONFIRMED = 1
    REJECTED = 0
    INSUFFICIENT_DATA = -1

class ConfirmationFilter:
    """
    Validates sentiment signals with market confirmation.
    Checks: abnormal volume, price direction match, recent breakout.
    """
    
    # Configuration
    VOLUME_MULTIPLIER = 1.5  # 50% above normal = abnormal
    LOOKBACK_BARS = 10       # Check last 10 bars for trend
    BREAKOUT_LOOKBACK = 20   # Check 20-bar high/low for breakout
    
    def __init__(self):
        self.daily_volumes = {}  # ticker -> average volume
        self.price_history = {}  # ticker -> deque of prices
    
    def check_confirmation(self, ticker: str, 
                          sentiment_signal: int,  # 1=LONG, -1=SHORT, 0=NEUTRAL
                          current_price: float,
                          current_volume: float,
                          recent_prices: Optional[list] = None,
                          recent_volumes: Optional[list] = None) -> Dict:
        """
        Check if sentiment signal is confirmed by market data.
        
        Args:
            ticker: Stock ticker
            sentiment_signal: Direction from sentiment (-1, 0, 1)
            current_price: Latest closing price
            current_volume: Latest trading volume
            recent_prices: List of recent prices [oldest -> newest]
            recent_volumes: List of recent volumes [oldest -> newest]
        
        Returns: {
            'confirmed': ConfirmationStatus,
            'volume_check': bool,
            'price_direction_check': bool,
            'breakout_check': bool,
            'details': str
        }
        """
        if sentiment_signal == 0:  # NEUTRAL - no confirmation needed
            return {
                'confirmed': ConfirmationStatus.CONFIRMED,
                'volume_check': None,
                'price_direction_check': None,
                'breakout_check': None,
                'details': 'Neutral signal - no confirmation needed'
            }
        
        # Check for insufficient data
        if recent_prices is None or len(recent_prices) < 5:
            return {
                'confirmed': ConfirmationStatus.INSUFFICIENT_DATA,
                'volume_check': None,
                'price_direction_check': None,
                'breakout_check': None,
                'details': 'Insufficient price history'
            }
        
        # Perform checks
        volume_ok = self._check_volume(ticker, current_volume, recent_volumes)
        price_ok = self._check_price_direction(current_price, recent_prices, sentiment_signal)
        breakout_ok = self._check_breakout(current_price, recent_prices, sentiment_signal)
        
        # Combine checks (all must pass)
        all_checks = [volume_ok is not None and volume_ok,
                     price_ok is not None and price_ok,
                     breakout_ok is not None and breakout_ok]
        
        if all(all_checks):
            status = ConfirmationStatus.CONFIRMED
            details = "All checks passed"
        else:
            status = ConfirmationStatus.REJECTED
            failed = []
            if not (volume_ok or volume_ok is None):
                failed.append("volume")
            if not (price_ok or price_ok is None):
                failed.append("price_direction")
            if not (breakout_ok or breakout_ok is None):
                failed.append("breakout")
            details = f"Failed: {', '.join(failed)}"
        
        return {
            'confirmed': status,
            'volume_check': volume_ok,
            'price_direction_check': price_ok,
            'breakout_check': breakout_ok,
            'details': details
        }
    
    def _check_volume(self, ticker: str, current_volume: float,
                     recent_volumes: Optional[list]) -> Optional[bool]:
        """
        Check for abnormally high volume (at least N% above average).
        """
        if recent_volumes is None or len(recent_volumes) < 5:
            return None
        
        avg_volume = sum(recent_volumes[:-1]) / len(recent_volumes[:-1]) if len(recent_volumes) > 1 else 0
        
        if avg_volume == 0:
            return None
        
        # Volume spike check
        volume_ratio = current_volume / avg_volume
        return volume_ratio >= self.VOLUME_MULTIPLIER
    
    def _check_price_direction(self, current_price: float, 
                               recent_prices: list,
                               sentiment_signal: int) -> Optional[bool]:
        """
        Check if recent price movement agrees with sentiment signal.
        """
        if len(recent_prices) < 2:
            return None
        
        # Calculate price momentum over last 5 bars
        lookback = min(5, len(recent_prices))
        price_change = current_price - recent_prices[-lookback]
        
        if sentiment_signal == 1:  # LONG signal
            return price_change > 0  # Price should be moving up
        elif sentiment_signal == -1:  # SHORT signal
            return price_change < 0  # Price should be moving down
        else:
            return None
    
    def _check_breakout(self, current_price: float,
                       recent_prices: list,
                       sentiment_signal: int) -> Optional[bool]:
        """
        Check if price is breaking out (new highs/lows for period).
        """
        if len(recent_prices) < self.BREAKOUT_LOOKBACK:
            return None
        
        # Get historic high/low
        recent = recent_prices[-self.BREAKOUT_LOOKBACK:]
        recent_high = max(recent[:-1])  # Exclude current price
        recent_low = min(recent[:-1])
        
        if sentiment_signal == 1:  # LONG
            return current_price > recent_high  # Should be new high
        elif sentiment_signal == -1:  # SHORT
            return current_price < recent_low  # Should be new low
        else:
            return None
    
    def check_multi_timeframe(self, ticker: str,
                             sentiment_signals: Dict[str, int],
                             market_data: Dict) -> Dict:
        """
        Check confirmation across multiple timeframes.
        sentiment_signals: {window: signal_direction} e.g., {'15min': 1, '1hour': -1}
        market_data: {ticker: {'price': float, 'volume': float, 'prices': list, 'volumes': list}}
        """
        if ticker not in market_data:
            return {'confirmed': ConfirmationStatus.INSUFFICIENT_DATA, 'details': 'No market data'}
        
        mdata = market_data[ticker]
        
        # Use 1-hour as primary signal
        primary_signal = sentiment_signals.get('1hour', 0)
        
        if primary_signal == 0:
            return {'confirmed': ConfirmationStatus.CONFIRMED, 'details': 'Neutral - no confirmation needed'}
        
        # Check confirmation
        result = self.check_confirmation(
            ticker,
            primary_signal,
            mdata.get('price', 0),
            mdata.get('volume', 0),
            mdata.get('prices'),
            mdata.get('volumes')
        )
        
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Confirmation filter for sentiment signals')
    parser.add_argument('--test', action='store_true', help='Run with test data')
    args = parser.parse_args()
    
    if args.test:
        filter_obj = ConfirmationFilter()
        
        # Test scenario 1: LONG signal with confirming market data
        print("\nTest 1: LONG signal with bullish market")
        result = filter_obj.check_confirmation(
            'AAPL',
            sentiment_signal=1,  # LONG
            current_price=180.0,
            current_volume=80_000_000,
            recent_prices=[160, 162, 165, 170, 175, 180],
            recent_volumes=[40_000_000] * 5 + [80_000_000]
        )
        print(f"  Confirmed: {result['confirmed'].name}")
        print(f"  Volume: {result['volume_check']}, Price: {result['price_direction_check']}, Breakout: {result['breakout_check']}")
        print(f"  Details: {result['details']}")
        
        # Test scenario 2: SHORT signal without confirming volume
        print("\nTest 2: SHORT signal with weak volume")
        result = filter_obj.check_confirmation(
            'TSLA',
            sentiment_signal=-1,  # SHORT
            current_price=250.0,
            current_volume=20_000_000,  # Low volume
            recent_prices=[270, 265, 260, 255, 250],
            recent_volumes=[50_000_000] * 4
        )
        print(f"  Confirmed: {result['confirmed'].name}")
        print(f"  Volume: {result['volume_check']}, Price: {result['price_direction_check']}, Breakout: {result['breakout_check']}")
        print(f"  Details: {result['details']}")
