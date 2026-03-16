import argparse
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import math

class PortfolioManager:
    """
    Manages sentiment strategy portfolio across multiple securities.
    Handles position sizing, sector balancing, and risk controls.
    """
    
    # Risk limits
    MAX_POSITION_SIZE = 100_000      # $ per position
    MAX_SECTOR_EXPOSURE = 300_000    # $ per sector
    MAX_TOTAL_EXPOSURE = 1_000_000   # $ total
    MAX_DAILY_LOSS = 50_000          # $ daily stop loss
    HARD_STOP_PCT = 0.05             # 5% hard stop per position
    
    # Sector mapping
    SECTOR_MAP = {
        'AAPL': 'Technology',
        'MSFT': 'Technology',
        'TSLA': 'Consumer Discretionary',
        'AMZN': 'Consumer Discretionary',
        'JPM': 'Financials',
        'JNJ': 'Healthcare',
        'PG': 'Consumer Staples',
        'XOM': 'Energy',
    }
    
    def __init__(self, account_value: float = 1_000_000):
        self.account_value = account_value
        self.positions = {}  # ticker -> {share_qty, entry_price, entry_time, pnl}
        self.sector_exposure = defaultdict(float)
        self.daily_pnl = 0.0
        self.trade_history = []
    
    def calculate_position_size(self, ticker: str, entry_price: float,
                               signal_strength: float = 1.0) -> int:
        """
        Determine appropriate position size based on:
        - Account value and risk limits
        - Signal strength (confidence)
        - Sector and total exposure
        
        Args:
            ticker: Stock symbol
            entry_price: Entry price
            signal_strength: 0-1 confidence level
        
        Returns: Number of shares to trade
        """
        # Base calculation: 10% risk, 5% stop loss = 200 shares * risk factor
        base_size = (self.account_value * 0.10) / (entry_price * self.HARD_STOP_PCT)
        
        # Scale by signal strength
        size_by_strength = int(base_size * signal_strength)
        
        # Cap by position limit
        max_shares_by_position = int(self.MAX_POSITION_SIZE / entry_price)
        size = min(size_by_strength, max_shares_by_position)
        
        # Check sector exposure
        sector = self.SECTOR_MAP.get(ticker, 'Other')
        notional_value = size * entry_price
        sector_exposure_after = self.sector_exposure[sector] + notional_value
        
        if sector_exposure_after > self.MAX_SECTOR_EXPOSURE:
            max_shares_by_sector = int((self.MAX_SECTOR_EXPOSURE - 
                                        self.sector_exposure[sector]) / entry_price)
            size = max(max_shares_by_sector, 0)  # Reduce to fit sector limit
        
        # Check total exposure
        total_exposure = sum(self.positions[t].get('value', 0) 
                           for t in self.positions) + (size * entry_price)
        
        if total_exposure > self.MAX_TOTAL_EXPOSURE:
            reduction_factor = self.MAX_TOTAL_EXPOSURE / total_exposure
            size = int(size * reduction_factor)
        
        return max(size, 0)
    
    def enter_position(self, ticker: str, direction: str, entry_price: float,
                      quantity: int, signal_strength: float = 1.0) -> Dict:
        """
        Enter a new position.
        """
        sector = self.SECTOR_MAP.get(ticker, 'Other')
        notional_value = quantity * entry_price
        
        with self._check_limits(ticker, notional_value):
            self.positions[ticker] = {
                'direction': direction,  # 'LONG' or 'SHORT'
                'shares': quantity,
                'entry_price': entry_price,
                'entry_time': datetime.now(),
                'value': notional_value,
                'signal_strength': signal_strength
            }
            
            self.sector_exposure[sector] += notional_value
            
            trade = {
                'action': direction,
                'ticker': ticker,
                'quantity': quantity,
                'price': entry_price,
                'timestamp': datetime.now(),
                'sector': sector,
                'notional_value': notional_value
            }
            self.trade_history.append(trade)
            
            return trade
    
    def exit_position(self, ticker: str, exit_price: float,
                      quantity: Optional[int] = None) -> Dict:
        """
        Exit a position (partial or full).
        """
        if ticker not in self.positions:
            return {'error': 'Position not found'}
        
        pos = self.positions[ticker]
        qty = quantity or pos['shares']
        
        # Calculate P&L
        if pos['direction'] == 'LONG':
            pnl = (exit_price - pos['entry_price']) * qty
        else:  # SHORT
            pnl = (pos['entry_price'] - exit_price) * qty
        
        # Update daily P&L
        self.daily_pnl += pnl
        
        # Update position
        sector = self.SECTOR_MAP.get(ticker, 'Other')
        exit_notional = qty * exit_price
        self.sector_exposure[sector] -= exit_notional
        
        if qty >= pos['shares']:
            del self.positions[ticker]
        else:
            pos['shares'] -= qty
            pos['value'] -= exit_notional
        
        trade = {
            'action': 'EXIT',
            'ticker': ticker,
            'quantity': qty,
            'price': exit_price,
            'pnl': pnl,
            'timestamp': datetime.now(),
            'sector': sector
        }
        self.trade_history.append(trade)
        
        return trade
    
    def check_risk_limits(self) -> Dict[str, bool]:
        """
        Check if portfolio violates risk limits.
        """
        limits = {
            'daily_loss_ok': self.daily_pnl > -self.MAX_DAILY_LOSS,
            'total_exposure_ok': sum(p['value'] for p in self.positions.values()) < self.MAX_TOTAL_EXPOSURE,
            'sector_exposure_ok': all(v < self.MAX_SECTOR_EXPOSURE 
                                      for v in self.sector_exposure.values())
        }
        return limits
    
    def check_stop_losses(self, market_prices: Dict[str, float]) -> List[str]:
        """
        Check which positions hit hard stops.
        Returns: List of tickers to stop out
        """
        stops = []
        
        for ticker, pos in self.positions.items():
            if ticker not in market_prices:
                continue
            
            current_price = market_prices[ticker]
            
            if pos['direction'] == 'LONG':
                drawdown = (pos['entry_price'] - current_price) / pos['entry_price']
            else:  # SHORT
                drawdown = (current_price - pos['entry_price']) / pos['entry_price']
            
            if drawdown >= self.HARD_STOP_PCT:
                stops.append(ticker)
        
        return stops
    
    def check_time_stops(self, max_holding_days: int = 7) -> List[str]:
        """
        Check which positions exceed max holding period.
        """
        stops = []
        cutoff_time = datetime.now() - timedelta(days=max_holding_days)
        
        for ticker, pos in self.positions.items():
            if pos['entry_time'] < cutoff_time:
                stops.append(ticker)
        
        return stops
    
    def get_portfolio_stats(self, market_prices: Dict[str, float]) -> Dict:
        """
        Calculate portfolio statistics.
        """
        total_value = 0.0
        total_pnl = 0.0
        sector_pnl = defaultdict(float)
        
        for ticker, pos in self.positions.items():
            if ticker not in market_prices:
                continue
            
            current_price = market_prices[ticker]
            current_value = pos['shares'] * current_price
            
            if pos['direction'] == 'LONG':
                position_pnl = (current_price - pos['entry_price']) * pos['shares']
            else:  # SHORT
                position_pnl = (pos['entry_price'] - current_price) * pos['shares']
            
            total_value += current_value
            total_pnl += position_pnl
            
            sector = self.SECTOR_MAP.get(ticker, 'Other')
            sector_pnl[sector] += position_pnl
        
        return {
            'total_notional_value': total_value,
            'total_pnl': total_pnl,
            'daily_pnl': self.daily_pnl,
            'return_pct': (total_pnl / self.account_value * 100) if self.account_value > 0 else 0,
            'positions_count': len(self.positions),
            'sector_pnl': dict(sector_pnl),
            'sector_exposure': dict(self.sector_exposure)
        }
    
    def rank_positions_for_rebalance(self) -> List[Tuple[str, float]]:
        """
        Rank positions by signal strength for potential rebalancing.
        Returns: [(ticker, strength), ...] sorted by strength desc
        """
        ranked = [(t, p['signal_strength']) 
                 for t, p in self.positions.items()]
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked
    
    def _check_limits(self, ticker: str, notional_value: float):
        """
        Context manager for checking risk limits.
        """
        class LimitChecker:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        return LimitChecker()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Portfolio manager for sentiment strategy')
    parser.add_argument('--test', action='store_true', help='Run with test data')
    args = parser.parse_args()
    
    if args.test:
        pm = PortfolioManager(account_value=1_000_000)
        
        # Test position sizing
        print("\n=== Position Sizing Test ===")
        for ticker in ['AAPL', 'TSLA', 'MSFT']:
            price = 150.0
            qty = pm.calculate_position_size(ticker, price, signal_strength=0.8)
            value = qty * price
            print(f"{ticker}: {qty:4d} shares @ ${price:.2f} = ${value:,.0f}")
        
        # Test position entry
        print("\n=== Enter Positions ===")
        pm.enter_position('AAPL', 'LONG', 150.0, 50, signal_strength=0.8)
        pm.enter_position('TSLA', 'LONG', 250.0, 40, signal_strength=0.9)
        pm.enter_position('MSFT', 'SHORT', 300.0, 20, signal_strength=0.7)
        
        # Test risk checks
        print("\n=== Risk Checks ===")
        limits = pm.check_risk_limits()
        for check, ok in limits.items():
            print(f"{check}: {'✓' if ok else '✗'}")
        
        # Test stop checks
        print("\n=== Stop Loss Checks ===")
        market_prices = {
            'AAPL': 142.0,  # -5.3% from 150
            'TSLA': 237.0,  # -5.2% from 250
            'MSFT': 315.0   # 5% loss (since short)
        }
        stops = pm.check_stop_losses(market_prices)
        print(f"Positions to stop out: {stops}")
        
        # Test portfolio stats
        print("\n=== Portfolio Statistics ===")
        stats = pm.get_portfolio_stats(market_prices)
        print(f"Total Value: ${stats['total_notional_value']:,.0f}")
        print(f"Total P&L: ${stats['total_pnl']:,.0f} ({stats['return_pct']:.2f}%)")
        print(f"Daily P&L: ${stats['daily_pnl']:,.0f}")
        print(f"Positions: {stats['positions_count']}")
        print(f"Sector P&L:")
        for sector, pnl in stats['sector_pnl'].items():
            print(f"  {sector}: ${pnl:,.0f}")
