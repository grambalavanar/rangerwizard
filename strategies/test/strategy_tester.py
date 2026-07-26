"""
strategy_tester.py
==================
Backtesting and strategy comparison library for momentum trading strategies.
Designed to integrate with myIBApp.py (for live IBKR data) and
momentum/momentum_tools.py (for indicators).

Usage flow
----------
1. Define a strategy by subclassing ``Strategy`` and implementing
   ``generate_signals(df)``.
2. Build or fetch an OHLCV DataFrame (from CSV, yfinance, or IBKR via
   ``fetch_ibkr_history``).
3. Call ``run_backtest(strategy, df)`` to get a ``BacktestResult``.
4. Call ``run_comparison({...}, df)`` to compare multiple strategies.
5. Print a summary with ``print_result(result)`` or compare a table
   with ``compare_results(results)``.

Quick example
-------------
    from strategies.test.strategy_tester import (
        Strategy, BacktestConfig, run_backtest, print_result
    )
    from strategies.tools.momentum_tools import macd, adx, crossover

    class MACDStrategy(Strategy):
        name = "MACD + ADX Filter"
        def generate_signals(self, df):
            m, sig, _ = macd(df["Close"])
            adx_v, _, _ = adx(df["High"], df["Low"], df["Close"])
            df["signal"] = 0
            df.loc[crossover(m, sig) & (adx_v > 25), "signal"] = 1
            df.loc[m < sig,                           "signal"] = 0
            df["signal"] = df["signal"].ffill().fillna(0)
            return df

    cfg    = BacktestConfig(initial_capital=100_000)
    result = run_backtest(MACDStrategy(), df, cfg)
    print_result(result)

Dependencies: numpy, pandas, sys, abc, dataclasses
Optional:     matplotlib (only for plot_equity_curve)
"""

import sys
import os
import time
import threading
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Allow imports from the repo root
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from strategies.tools.momentum_tools import (
    sharpe_ratio, sortino_ratio, max_drawdown, calmar_ratio,
    cagr, win_rate, kelly_criterion, volatility, atr,
)


# ============================================================
# STRATEGY BASE CLASS
# ============================================================

class Strategy(ABC):
    """
    Abstract base class for all tradeable strategies.
    ---------------------------------------------------
    What it is
        The contract every strategy must satisfy. Subclass this and
        implement ``generate_signals`` to create your own strategy.
        The backtester and paper trader both accept any ``Strategy``
        subclass, so you write the logic once and test it everywhere.

    Used by
        All strategies in this framework must extend this class.
        It ensures the backtester always receives a DataFrame with a
        standardised ``signal`` column (1 = long, -1 = short, 0 = flat).

    How to implement
        1. Subclass ``Strategy``.
        2. Set the ``name`` class attribute to a human-readable string.
        3. Implement ``generate_signals(df)`` — add a column named
           ``"signal"`` to the DataFrame and return it.
           Signal values: 1 = long, -1 = short, 0 = flat / cash.
        4. Optionally override ``symbols`` to declare which tickers this
           strategy is designed for (used by the paper trader).

    Code example
        >>> from strategies.test.strategy_tester import Strategy
        >>> from strategies.tools.momentum_tools import rsi, crossover
        >>>
        >>> class RSIStrategy(Strategy):
        ...     name = "RSI Mean Reversion"
        ...     symbols = ["AAPL", "MSFT"]
        ...
        ...     def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        ...         df["rsi"] = rsi(df["Close"], window=14)
        ...         df["signal"] = 0
        ...         # Buy when RSI crosses back above 30 (oversold recovery)
        ...         df.loc[df["rsi"] < 30, "signal"] = 1
        ...         df.loc[df["rsi"] > 70, "signal"] = 0
        ...         df["signal"] = df["signal"].ffill().fillna(0)
        ...         return df

    Args for generate_signals
        df (pd.DataFrame): OHLCV DataFrame. Must have columns:
                           Open, High, Low, Close, Volume.
                           Index must be DatetimeIndex.

    Returns from generate_signals
        pd.DataFrame: Same DataFrame with a ``"signal"`` column added.
                      Signals are forward-filled so the position is held
                      until explicitly changed.
    """

    #: Human-readable strategy name. Override in your subclass.
    name: str = "Unnamed Strategy"

    #: Ticker symbols this strategy trades (used by paper_trader.py).
    #: Override in your subclass.
    symbols: List[str] = []

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute entry/exit signals from OHLCV data.

        Args:
            df (pd.DataFrame): OHLCV price data with DatetimeIndex.

        Returns:
            pd.DataFrame: Same df with ``"signal"`` column added.
                          1 = long, -1 = short, 0 = flat.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"Strategy(name='{self.name}')"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class BacktestConfig:
    """
    Configuration for a backtest run.
    -----------------------------------
    What it is
        A dataclass holding all parameters that control how the backtest
        engine simulates trades. Change these to model different account
        sizes, commission structures, and risk rules.

    Used by
        Passed as the ``config`` argument to ``run_backtest()`` and
        ``run_comparison()``. You can share one config across many
        strategy tests for apples-to-apples comparisons.

    When to adjust
        - Change ``initial_capital`` to reflect your actual account size.
        - Set ``commission_per_share`` to match your IBKR pricing plan
          (IBKR Fixed: $0.005/share min $1; Pro Tiered: $0.0035/share+).
        - Use ``position_sizing="atr"`` for volatility-adjusted position
          sizing — better for comparing strategies across different assets.
        - Enable ``allow_short=False`` if your account is a cash account
          or you don't want short exposure.
        - Set ``max_drawdown_pct`` to auto-stop the backtest when equity
          drops below this threshold (simulates a circuit-breaker rule).

    Code example
        >>> from strategies.test.strategy_tester import BacktestConfig
        >>> cfg = BacktestConfig(
        ...     initial_capital=50_000,
        ...     commission_per_share=0.005,
        ...     position_sizing="atr",
        ...     atr_risk_pct=0.01,   # risk 1% of capital per ATR move
        ...     allow_short=False,
        ... )

    Fields
        initial_capital     Starting account balance in dollars.
        commission_per_share Dollars per share charged at entry and exit.
        slippage_pct        Fraction of price lost to spread/slippage
                            on each trade (e.g. 0.001 = 0.1%).
        position_sizing     "fixed"  – size = initial_capital * position_pct
                            "atr"    – size = (capital * atr_risk_pct) / ATR
                            "kelly"  – use Kelly Criterion fraction
        position_pct        Fraction of capital per trade (fixed sizing).
        atr_risk_pct        Capital fraction risked per ATR unit (atr sizing).
        atr_window          ATR window in bars for position sizing.
        allow_short         If False, signal=-1 is treated as flat (exit only).
        max_drawdown_pct    Stop the simulation if equity drawdown exceeds
                            this fraction (0.20 = 20%). None = no limit.
    """
    initial_capital:      float = 100_000.0
    commission_per_share: float = 0.005
    slippage_pct:         float = 0.001
    position_sizing:      str   = "fixed"   # "fixed" | "atr" | "kelly"
    position_pct:         float = 0.10      # used for "fixed" sizing
    atr_risk_pct:         float = 0.01      # used for "atr" sizing
    atr_window:           int   = 14
    allow_short:          bool  = False
    max_drawdown_pct:     Optional[float] = None


# ============================================================
# RESULT CONTAINERS
# ============================================================

@dataclass
class Trade:
    """
    Single completed trade record.
    --------------------------------
    What it is
        A dataclass capturing every detail of one round-trip trade
        (entry + exit). The backtester populates a list of these for
        every strategy, enabling per-trade analysis.

    Used by
        Stored in ``BacktestResult.trades``. Useful for analysing trade
        distributions, identifying losing streaks, and reviewing specific
        entry/exit decisions. The paper trader also logs trades in this
        format to the trade log CSV.

    When to use
        - Review individual trades to identify systematic entry/exit errors.
        - Compute trade-level statistics (e.g. largest winner, longest
          holding period) that position-level metrics miss.
        - Compare the trade log against your brokerage confirms to
          verify backtest accuracy.
    """
    entry_date:   str
    exit_date:    str
    direction:    int    # 1 = long, -1 = short
    entry_price:  float
    exit_price:   float
    shares:       int
    gross_pnl:    float  # before commission
    commission:   float
    net_pnl:      float  # after commission
    return_pct:   float  # net return on the trade


@dataclass
class BacktestResult:
    """
    Complete result from a single backtest run.
    --------------------------------------------
    What it is
        A dataclass containing all performance metrics, the equity curve,
        and every individual trade from a backtest. This is the primary
        output of ``run_backtest()``.

    Used by
        - ``print_result(result)`` — pretty-prints a summary table.
        - ``compare_results(results)`` — builds a comparison DataFrame.
        - ``plot_equity_curve(result)`` — plots the equity curve.
        - The paper trader checks these metrics to determine if a strategy
          is healthy before allowing live execution.

    When to use
        After running a backtest, inspect these fields to decide:
        - Is the Sharpe ratio high enough to justify live trading?
        - Is max_drawdown within your psychological/financial tolerance?
        - Does the win rate match what you expected from the strategy logic?
        - Is CAGR realistic (beware of suspiciously high in-sample numbers)?

    Fields
        strategy_name  Name of the strategy tested.
        symbol         Ticker symbol tested on.
        start_date     First date in the test.
        end_date       Last date in the test.
        initial_capital Starting capital.
        final_equity   Ending equity after all trades.
        total_return   (final_equity / initial_capital) - 1.
        cagr           Compound Annual Growth Rate.
        sharpe         Annualised Sharpe ratio (rf = 0).
        sortino        Annualised Sortino ratio.
        calmar         CAGR / |max_drawdown|.
        max_drawdown   Worst peak-to-trough equity decline.
        volatility     Annualised volatility of daily returns.
        win_rate       Fraction of profitable trades.
        total_trades   Number of completed round-trip trades.
        kelly_fraction Suggested Kelly position size fraction.
        equity_curve   pd.Series of daily equity values.
        daily_returns  pd.Series of daily percentage returns.
        trades         List[Trade] — all completed trades.
    """
    strategy_name:   str
    symbol:          str
    start_date:      str
    end_date:        str
    initial_capital: float
    final_equity:    float
    total_return:    float
    cagr_val:        float
    sharpe:          float
    sortino:         float
    calmar:          float
    max_dd:          float
    vol:             float
    win_rate_val:    float
    total_trades:    int
    kelly_fraction:  float
    equity_curve:    pd.Series
    daily_returns:   pd.Series
    trades:          List[Trade]


# ============================================================
# CORE BACKTEST ENGINE
# ============================================================

def run_backtest(
    strategy: Strategy,
    df: pd.DataFrame,
    config: Optional[BacktestConfig] = None,
    symbol: str = "UNKNOWN",
) -> BacktestResult:
    """
    Run a historical backtest of a single strategy.
    ------------------------------------------------
    What it does
        Simulates trading a strategy on historical OHLCV data, tracking
        capital, positions, and trades bar by bar. Signals generated on
        bar N are executed at the Open of bar N+1 (realistic execution).
        Commission and slippage are applied on every fill.

    Used by
        The primary entry point for validating any strategy before risking
        real capital. Use this to answer: "Would this strategy have worked
        historically?"

    How to use (code)
        >>> from strategies.test.strategy_tester import run_backtest, BacktestConfig
        >>> import pandas as pd
        >>> df = pd.read_csv("AAPL.csv", parse_dates=["Date"], index_col="Date")
        >>> cfg = BacktestConfig(initial_capital=100_000)
        >>> result = run_backtest(MyStrategy(), df, cfg, symbol="AAPL")
        >>> print_result(result)

    How to use (as you the user)
        1. Get historical OHLCV data into a DataFrame (from CSV, yfinance,
           or ``fetch_ibkr_history()``).
        2. Instantiate your strategy and a ``BacktestConfig``.
        3. Call this function and inspect the returned ``BacktestResult``.
        4. If metrics look acceptable, graduate to paper trading.

    When you need this
        - Before running any strategy on paper or live, always run a
          backtest first over at least 2–5 years of data.
        - Use it to tune parameters (e.g. RSI window) before freezing them.
        - Re-run after any strategy modification to check for regressions.

    Args:
        strategy (Strategy):           Instantiated strategy object.
        df       (pd.DataFrame):       OHLCV data. Required columns:
                                       Open, High, Low, Close, Volume.
                                       Must have a DatetimeIndex.
        config   (BacktestConfig):     Simulation parameters. Uses
                                       defaults if None.
        symbol   (str):                Ticker name for labelling only.

    Returns:
        BacktestResult: All metrics, equity curve, and trade list.

    Raises:
        ValueError: If df is missing required columns or has fewer than
                    50 rows (insufficient for meaningful testing).
    """
    if config is None:
        config = BacktestConfig()

    # --- Validate input ---
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"DataFrame is missing required columns: {missing}. "
            f"Ensure your data has Open, High, Low, Close, Volume columns."
        )
    if len(df) < 50:
        raise ValueError(
            f"DataFrame has only {len(df)} rows. Need at least 50 bars for "
            f"a meaningful backtest."
        )

    df = df.copy().sort_index()

    # --- Generate signals ---
    try:
        df = strategy.generate_signals(df)
    except Exception as exc:
        raise RuntimeError(
            f"Strategy '{strategy.name}' raised an error in generate_signals: {exc}"
        ) from exc

    if "signal" not in df.columns:
        raise ValueError(
            f"Strategy '{strategy.name}'.generate_signals() did not add a "
            f"'signal' column. Ensure you assign df['signal'] before returning."
        )

    # Enforce signal values
    df["signal"] = df["signal"].fillna(0)
    if not config.allow_short:
        df["signal"] = df["signal"].clip(lower=0)

    # Pre-compute ATR for position sizing if needed
    if config.position_sizing == "atr":
        df["_atr"] = atr(df["High"], df["Low"], df["Close"], window=config.atr_window)

    # --- Simulate bar by bar ---
    capital     = config.initial_capital
    position    = 0   # current position: 1, -1, or 0
    shares      = 0
    entry_price = 0.0
    entry_date  = ""

    trades_list: List[Trade] = []
    equity_list: List[float] = []
    equity_index: List = []

    def _calc_shares(bar_idx: int, direction: int) -> int:
        """Compute number of shares to buy given current capital and sizing rule."""
        price = float(df["Open"].iloc[bar_idx])
        if price <= 0:
            return 0
        if config.position_sizing == "fixed":
            dollar_size = capital * config.position_pct
        elif config.position_sizing == "atr":
            atr_val = df["_atr"].iloc[bar_idx]
            if pd.isna(atr_val) or atr_val <= 0:
                dollar_size = capital * 0.05
            else:
                dollar_size = (capital * config.atr_risk_pct) / atr_val * price
                dollar_size = min(dollar_size, capital * 0.50)  # cap at 50%
        elif config.position_sizing == "kelly":
            if len(trades_list) >= 10:
                trade_rets = pd.Series([t.return_pct for t in trades_list])
                kf = kelly_criterion(trade_rets)
                kf = max(0.0, min(kf * 0.5, 0.40))   # half-Kelly, capped at 40%
            else:
                kf = config.position_pct
            dollar_size = capital * kf
        else:
            dollar_size = capital * config.position_pct

        n = int(dollar_size / price)
        return max(n, 0)

    def _exit_position(bar_idx: int) -> None:
        nonlocal capital, position, shares, entry_price, entry_date
        if position == 0 or shares == 0:
            return
        raw_exit = float(df["Open"].iloc[bar_idx])
        # Slippage: adverse to the direction of the exit
        slippage = raw_exit * config.slippage_pct
        exit_price = (raw_exit - slippage) if position == 1 else (raw_exit + slippage)
        gross_pnl = shares * (exit_price - entry_price) * position
        commission = shares * config.commission_per_share * 2  # entry + exit
        net_pnl = gross_pnl - commission
        ret_pct = net_pnl / (shares * abs(entry_price)) if entry_price != 0 else 0.0
        capital += net_pnl + (shares * abs(entry_price) * (1 if position == 1 else -1))
        # Actually correct capital tracking: add back cost basis + pnl
        # (capital was reduced by cost basis at entry)
        trades_list.append(Trade(
            entry_date=entry_date,
            exit_date=str(df.index[bar_idx].date()),
            direction=position,
            entry_price=entry_price,
            exit_price=exit_price,
            shares=shares,
            gross_pnl=gross_pnl,
            commission=commission,
            net_pnl=net_pnl,
            return_pct=ret_pct,
        ))
        position = 0
        shares = 0
        entry_price = 0.0
        entry_date = ""

    def _enter_position(bar_idx: int, direction: int) -> None:
        nonlocal capital, position, shares, entry_price, entry_date
        n = _calc_shares(bar_idx, direction)
        if n <= 0:
            return
        raw_entry = float(df["Open"].iloc[bar_idx])
        slippage = raw_entry * config.slippage_pct
        ep = (raw_entry + slippage) if direction == 1 else (raw_entry - slippage)
        cost = n * ep
        if cost > capital:
            n = int(capital / ep)
            cost = n * ep
        if n <= 0:
            return
        capital -= cost
        position = direction
        shares = n
        entry_price = ep
        entry_date = str(df.index[bar_idx].date())

    # Use a simple equity model:
    # - cash capital is tracked after entries/exits
    # - mark-to-market unrealised P&L is included in daily equity

    # Rebuild with a cleaner approach
    capital = config.initial_capital
    position = 0
    shares = 0
    entry_price = 0.0
    entry_date = ""
    trades_list = []
    equity_list = []
    equity_index = []
    cash = config.initial_capital

    circuit_breaker_hit = False

    for i in range(1, len(df)):
        prev_signal = int(df["signal"].iloc[i - 1])
        open_price = float(df["Open"].iloc[i])
        close_price = float(df["Close"].iloc[i])
        date_str = str(df.index[i].date())

        # Execute signal from previous bar at today's open
        if prev_signal != position:
            # Exit current position first
            if position != 0 and shares > 0:
                raw_exit = open_price
                slippage = raw_exit * config.slippage_pct
                exit_p = (raw_exit - slippage) if position == 1 else (raw_exit + slippage)
                gross = shares * (exit_p - entry_price) * position
                comm = shares * config.commission_per_share
                net = gross - comm
                ret_pct = net / (shares * abs(entry_price)) if entry_price else 0.0
                # Return capital
                cash += shares * exit_p * (1 if position == 1 else -1) - comm
                trades_list.append(Trade(
                    entry_date=entry_date,
                    exit_date=date_str,
                    direction=position,
                    entry_price=entry_price,
                    exit_price=exit_p,
                    shares=shares,
                    gross_pnl=gross,
                    commission=comm,
                    net_pnl=net,
                    return_pct=ret_pct,
                ))
                position = 0
                shares = 0
                entry_price = 0.0

            # Enter new position
            if prev_signal != 0:
                direction = prev_signal
                if config.position_sizing == "fixed":
                    dollar_size = cash * config.position_pct
                elif config.position_sizing == "atr":
                    atr_val = float(df["_atr"].iloc[i]) if "_atr" in df.columns else 0
                    if atr_val > 0:
                        dollar_size = min(
                            (cash * config.atr_risk_pct) / atr_val * open_price,
                            cash * 0.50
                        )
                    else:
                        dollar_size = cash * 0.05
                elif config.position_sizing == "kelly":
                    if len(trades_list) >= 10:
                        kf = kelly_criterion(
                            pd.Series([t.return_pct for t in trades_list])
                        )
                        kf = max(0.0, min(kf * 0.5, 0.40))
                    else:
                        kf = config.position_pct
                    dollar_size = cash * kf
                else:
                    dollar_size = cash * config.position_pct

                raw_entry = open_price
                slippage = raw_entry * config.slippage_pct
                ep = (raw_entry + slippage) if direction == 1 else (raw_entry - slippage)
                n = int(min(dollar_size, cash * 0.99) / ep) if ep > 0 else 0
                if n > 0:
                    cost = n * ep + n * config.commission_per_share
                    cash -= cost
                    position = direction
                    shares = n
                    entry_price = ep
                    entry_date = date_str

        # Mark-to-market equity
        unrealised = shares * (close_price - entry_price) * position if shares > 0 else 0.0
        equity = cash + shares * (entry_price if position != 0 else 0) + unrealised
        equity_list.append(equity)
        equity_index.append(df.index[i])

        # Circuit breaker
        if config.max_drawdown_pct is not None:
            drawdown = (equity - config.initial_capital) / config.initial_capital
            if drawdown < -config.max_drawdown_pct:
                circuit_breaker_hit = True
                break

    # Close any open position at last bar's close
    if position != 0 and shares > 0 and len(df) > 0:
        last_close = float(df["Close"].iloc[-1])
        gross = shares * (last_close - entry_price) * position
        comm = shares * config.commission_per_share
        net = gross - comm
        ret_pct = net / (shares * abs(entry_price)) if entry_price else 0.0
        cash += shares * last_close * (1 if position == 1 else -1) - comm
        trades_list.append(Trade(
            entry_date=entry_date,
            exit_date=str(df.index[-1].date()),
            direction=position,
            entry_price=entry_price,
            exit_price=last_close,
            shares=shares,
            gross_pnl=gross,
            commission=comm,
            net_pnl=net,
            return_pct=ret_pct,
        ))

    # --- Build result ---
    equity_series = pd.Series(equity_list, index=equity_index, name="equity")
    daily_rets = equity_series.pct_change().dropna()

    final_equity = float(equity_series.iloc[-1]) if len(equity_series) > 0 else config.initial_capital
    total_return = (final_equity - config.initial_capital) / config.initial_capital

    # Risk metrics (require at least 20 return observations)
    if len(daily_rets) >= 20:
        sharpe_val = sharpe_ratio(daily_rets)
        sortino_val = sortino_ratio(daily_rets)
        calmar_val = calmar_ratio(daily_rets)
        mdd = max_drawdown(daily_rets)
        vol_val = volatility(daily_rets)
        cagr_val = cagr(daily_rets)
    else:
        sharpe_val = sortino_val = calmar_val = mdd = vol_val = cagr_val = float("nan")

    trade_returns = pd.Series([t.return_pct for t in trades_list]) if trades_list else pd.Series([], dtype=float)
    wr = float(win_rate(trade_returns)) if len(trade_returns) > 0 else float("nan")
    kf = float(kelly_criterion(trade_returns)) if len(trade_returns) >= 5 else float("nan")

    if circuit_breaker_hit:
        warnings.warn(
            f"[{strategy.name}] Circuit breaker triggered: drawdown exceeded "
            f"{config.max_drawdown_pct:.0%}. Simulation stopped early.",
            RuntimeWarning,
            stacklevel=2,
        )

    return BacktestResult(
        strategy_name=strategy.name,
        symbol=symbol,
        start_date=str(df.index[0].date()),
        end_date=str(df.index[-1].date()),
        initial_capital=config.initial_capital,
        final_equity=final_equity,
        total_return=total_return,
        cagr_val=cagr_val,
        sharpe=sharpe_val,
        sortino=sortino_val,
        calmar=calmar_val,
        max_dd=mdd,
        vol=vol_val,
        win_rate_val=wr,
        total_trades=len(trades_list),
        kelly_fraction=kf,
        equity_curve=equity_series,
        daily_returns=daily_rets,
        trades=trades_list,
    )


# ============================================================
# MULTI-STRATEGY COMPARISON
# ============================================================

def run_comparison(
    strategies: Dict[str, Strategy],
    df: pd.DataFrame,
    config: Optional[BacktestConfig] = None,
    symbol: str = "UNKNOWN",
) -> Dict[str, BacktestResult]:
    """
    Backtest multiple strategies on the same dataset and return all results.
    -------------------------------------------------------------------------
    What it does
        Runs ``run_backtest`` for each strategy in the dictionary and
        returns a dictionary of ``BacktestResult`` objects keyed by the
        same names you provided. Allows direct apples-to-apples comparison
        across parameter variations or completely different approaches.

    Used by
        Quant analysts use multi-strategy comparisons to select the best
        strategy from a candidate pool before committing to paper trading.
        Also used for parameter sensitivity analysis (e.g., testing RSI
        windows of 7, 10, 14, and 21 on the same data).

    When you need this
        - When you have multiple candidate strategies and want to rank them.
        - When doing parameter sweep tests (create many Strategy subclasses
          with different window sizes and compare them all at once).
        - Before choosing a strategy to deploy in the paper trader.

    Code example
        >>> from strategies.test.strategy_tester import run_comparison, compare_results
        >>> results = run_comparison(
        ...     {"MACD": MACDStrategy(), "RSI": RSIStrategy(), "ADX": ADXStrategy()},
        ...     df,
        ...     config=cfg,
        ...     symbol="AAPL"
        ... )
        >>> table = compare_results(results)
        >>> print(table.to_string())

    Args:
        strategies (Dict[str, Strategy]): Name → Strategy instance mapping.
        df         (pd.DataFrame):        Shared OHLCV dataset.
        config     (BacktestConfig):      Shared simulation config. Uses
                                          defaults if None.
        symbol     (str):                 Ticker label for reporting.

    Returns:
        Dict[str, BacktestResult]: Same keys as input, value = result.
    """
    if config is None:
        config = BacktestConfig()
    results = {}
    for name, strategy in strategies.items():
        print(f"  Running backtest: {name} on {symbol} ...", end=" ", flush=True)
        try:
            result = run_backtest(strategy, df.copy(), config, symbol=symbol)
            results[name] = result
            print("done.")
        except Exception as exc:
            print(f"FAILED: {exc}")
    return results


# ============================================================
# IBKR HISTORICAL DATA FETCHER
# ============================================================

def fetch_ibkr_history(
    app,
    symbol: str,
    duration: str = "2 Y",
    bar_size: str = "1 day",
    exchange: str = "SMART",
    currency: str = "USD",
    what_to_show: str = "TRADES",
    use_rth: int = 1,
    timeout: int = 30,
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data from IBKR TWS via the myIBApp connection.
    -----------------------------------------------------------------------
    What it does
        Requests historical bar data from Interactive Brokers using the
        already-connected ``app`` (returned by ``connect_to_tws()``).
        Attaches temporary callbacks to the app, waits for data, and
        returns a clean OHLCV DataFrame ready for backtesting.

    Used by
        Use this to get historical price data from IBKR instead of relying
        on external data sources. Ensures consistency between your
        backtest data and the live data your paper/live strategy will see.

    When you need this
        - When you want backtest data that matches exactly what IBKR would
          show in live trading (same splits, dividends, closes).
        - When you don't have a separate data subscription (Bloomberg,
          Refinitiv, etc.) and IBKR is your data source.
        - When fetching data for tickers IBKR has that other sources may
          not (e.g., ETFs, OTC stocks, futures).

    Limitations
        - IBKR limits historical data requests: max 6 months of 1-day bars
          per request; use multiple requests and concat for longer periods.
        - Requires an active TWS or IB Gateway connection.
        - ``duration`` and ``bar_size`` must match IBKR's accepted strings
          (see IBKR API docs: https://interactivebrokers.github.io/tws-api/).

    Code example
        >>> from myIBApp import connect_to_tws
        >>> from strategies.test.strategy_tester import fetch_ibkr_history
        >>> app = connect_to_tws()
        >>> df = fetch_ibkr_history(app, "AAPL", duration="1 Y", bar_size="1 day")
        >>> print(df.tail())
        >>> app.disconnect()

    Duration strings (examples)
        "1 D", "5 D", "1 W", "1 M", "3 M", "6 M", "1 Y", "2 Y", "5 Y"

    Bar size strings (examples)
        "1 min", "5 mins", "15 mins", "30 mins", "1 hour", "1 day"

    Args:
        app          (myIBApp):  Connected app from connect_to_tws().
        symbol       (str):      Ticker symbol (e.g. "AAPL").
        duration     (str):      How far back to fetch (default "2 Y").
        bar_size     (str):      Bar granularity (default "1 day").
        exchange     (str):      Exchange (default "SMART" = auto-route).
        currency     (str):      Currency (default "USD").
        what_to_show (str):      Price type: "TRADES", "MIDPOINT", "BID",
                                 "ASK" (default "TRADES").
        use_rth      (int):      1 = regular trading hours only (default 1).
        timeout      (int):      Seconds to wait for data (default 30).

    Returns:
        pd.DataFrame: OHLCV data with DatetimeIndex and columns:
                      Open, High, Low, Close, Volume.

    Raises:
        TimeoutError:   If data does not arrive within ``timeout`` seconds.
        RuntimeError:   If the IBKR request returns an error.
    """
    from ibapi.contract import Contract

    req_id = 9001  # dedicated req_id for historical data requests
    bars_accumulator: List[dict] = []
    done_event = threading.Event()
    error_info: List[str] = []

    # Temporarily monkey-patch callbacks onto the app
    _orig_hist = getattr(app, "historicalData", None)
    _orig_hist_end = getattr(app, "historicalDataEnd", None)
    _orig_error = getattr(app, "error", None)

    def _historicalData(req_id_cb, bar):
        if req_id_cb == req_id:
            bars_accumulator.append({
                "Date":   bar.date,
                "Open":   float(bar.open),
                "High":   float(bar.high),
                "Low":    float(bar.low),
                "Close":  float(bar.close),
                "Volume": float(bar.volume),
            })

    def _historicalDataEnd(req_id_cb, start, end):
        if req_id_cb == req_id:
            done_event.set()

    def _error(req_id_cb, error_code, error_string, *args):
        if req_id_cb == req_id:
            error_info.append(f"[{error_code}] {error_string}")
            done_event.set()
        elif _orig_error:
            _orig_error(req_id_cb, error_code, error_string, *args)

    app.historicalData = _historicalData
    app.historicalDataEnd = _historicalDataEnd
    app.error = _error

    try:
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.exchange = exchange
        contract.currency = currency

        app.reqHistoricalData(
            req_id,
            contract,
            "",          # endDateTime: "" = now
            duration,
            bar_size,
            what_to_show,
            use_rth,
            1,           # formatDate: 1 = yyyyMMdd HH:mm:ss
            False,       # keepUpToDate
            [],          # chartOptions
        )

        if not done_event.wait(timeout=timeout):
            raise TimeoutError(
                f"Historical data request for {symbol} timed out after "
                f"{timeout}s. Check that TWS is connected and the symbol "
                f"is valid."
            )

        if error_info:
            raise RuntimeError(
                f"IBKR returned an error for {symbol}: {'; '.join(error_info)}"
            )

        if not bars_accumulator:
            raise RuntimeError(
                f"No historical data returned for {symbol}. "
                f"Check symbol, exchange, and data permissions."
            )

        df = pd.DataFrame(bars_accumulator)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()
        return df

    finally:
        # Restore original callbacks
        if _orig_hist is not None:
            app.historicalData = _orig_hist
        elif hasattr(app, "historicalData"):
            delattr(app, "historicalData")
        if _orig_hist_end is not None:
            app.historicalDataEnd = _orig_hist_end
        elif hasattr(app, "historicalDataEnd"):
            delattr(app, "historicalDataEnd")
        if _orig_error is not None:
            app.error = _orig_error


# ============================================================
# REPORTING UTILITIES
# ============================================================

def print_result(result: BacktestResult) -> None:
    """
    Print a formatted performance summary for a single backtest result.
    --------------------------------------------------------------------
    What it does
        Prints a human-readable table of all key metrics from a
        ``BacktestResult`` to stdout. Suitable for quick terminal review
        after running a backtest.

    Used by
        Any time you run a backtest and want a quick readout without
        building a full comparison table. Common at the top of a strategy
        development session.

    When you need this
        - After calling ``run_backtest()`` to see if the strategy is
          worth further development.
        - During parameter tuning to compare different configurations.
        - When reviewing an individual strategy in isolation.

    Code example
        >>> from strategies.test.strategy_tester import run_backtest, print_result
        >>> result = run_backtest(MyStrategy(), df, cfg, symbol="AAPL")
        >>> print_result(result)

    Args:
        result (BacktestResult): Output from run_backtest().
    """
    sep = "=" * 52
    print(f"\n{sep}")
    print(f"  Strategy : {result.strategy_name}")
    print(f"  Symbol   : {result.symbol}")
    print(f"  Period   : {result.start_date}  →  {result.end_date}")
    print(sep)
    print(f"  {'Initial Capital':30s} ${result.initial_capital:>12,.2f}")
    print(f"  {'Final Equity':30s} ${result.final_equity:>12,.2f}")
    print(f"  {'Total Return':30s} {result.total_return:>12.2%}")
    print(f"  {'CAGR':30s} {result.cagr_val:>12.2%}")
    print(f"  {'Annualised Volatility':30s} {result.vol:>12.2%}")
    print(f"  {'Max Drawdown':30s} {result.max_dd:>12.2%}")
    print(f"  {'Sharpe Ratio':30s} {result.sharpe:>12.2f}")
    print(f"  {'Sortino Ratio':30s} {result.sortino:>12.2f}")
    print(f"  {'Calmar Ratio':30s} {result.calmar:>12.2f}")
    print(f"  {'Win Rate':30s} {result.win_rate_val:>12.2%}")
    print(f"  {'Total Trades':30s} {result.total_trades:>12d}")
    print(f"  {'Kelly Fraction (full)':30s} {result.kelly_fraction:>12.2%}")
    print(sep)
    if result.trades:
        rets = [t.return_pct for t in result.trades]
        print(f"  {'Avg Trade Return':30s} {np.mean(rets):>12.2%}")
        print(f"  {'Best Trade':30s} {max(rets):>12.2%}")
        print(f"  {'Worst Trade':30s} {min(rets):>12.2%}")
    print(sep + "\n")


def compare_results(results: Dict[str, BacktestResult]) -> pd.DataFrame:
    """
    Build a comparison DataFrame from multiple backtest results.
    -------------------------------------------------------------
    What it does
        Takes the output of ``run_comparison()`` (a dict of results) and
        produces a single pandas DataFrame with one row per strategy and
        all key metrics as columns. Sorted by Sharpe ratio descending.

    Used by
        Quant analysts and systematic traders who want to rank multiple
        strategies or parameter combinations side by side. Easy to export
        to CSV or display in a Jupyter notebook.

    When you need this
        - After running ``run_comparison()`` to rank your candidates.
        - When parameter sweeping (e.g., testing many RSI windows) to
          see which configuration is optimal.
        - When presenting strategy results to a partner or reviewer.

    Code example
        >>> from strategies.test.strategy_tester import run_comparison, compare_results
        >>> results = run_comparison({"MACD": s1, "RSI": s2}, df, cfg)
        >>> table = compare_results(results)
        >>> print(table.to_string())
        >>> table.to_csv("comparison.csv")

    Args:
        results (Dict[str, BacktestResult]): Output from run_comparison().

    Returns:
        pd.DataFrame: One row per strategy, columns = key metrics,
                      sorted by Sharpe ratio descending.
    """
    rows = []
    for name, r in results.items():
        rows.append({
            "Strategy":       name,
            "Symbol":         r.symbol,
            "Total Return":   f"{r.total_return:.2%}",
            "CAGR":           f"{r.cagr_val:.2%}",
            "Sharpe":         f"{r.sharpe:.2f}",
            "Sortino":        f"{r.sortino:.2f}",
            "Calmar":         f"{r.calmar:.2f}",
            "Max Drawdown":   f"{r.max_dd:.2%}",
            "Volatility":     f"{r.vol:.2%}",
            "Win Rate":       f"{r.win_rate_val:.2%}",
            "Trades":         r.total_trades,
            "Kelly Fraction": f"{r.kelly_fraction:.2%}",
        })
    df_out = pd.DataFrame(rows)
    if "Sharpe" in df_out.columns and len(df_out) > 0:
        df_out = df_out.sort_values("Sharpe", ascending=False).reset_index(drop=True)
    return df_out


def plot_equity_curve(
    result: BacktestResult,
    benchmark: Optional[pd.Series] = None,
) -> None:
    """
    Plot the equity curve from a backtest result.
    -----------------------------------------------
    What it does
        Plots the strategy's equity curve over time using matplotlib.
        Optionally overlays a benchmark (e.g., buy-and-hold price series)
        normalised to the same starting capital.

    Used by
        Visual inspection of strategy performance. Equity curve shape
        reveals more than individual metrics — you can see the drawdown
        periods, trend consistency, and whether gains are front-loaded
        (lucky in-sample) or consistent throughout.

    When you need this
        - After backtesting to visually inspect the equity curve before
          trusting the Sharpe/Calmar numbers.
        - Compare strategy curve against a benchmark (e.g., SPY) to see
          if the strategy genuinely adds value.
        - Check for equity curve "cliff-edges" that indicate regime
          dependency or overfitting.

    Code example
        >>> from strategies.test.strategy_tester import run_backtest, plot_equity_curve
        >>> result = run_backtest(MyStrategy(), df, cfg, symbol="AAPL")
        >>> bm_returns = df["Close"] / df["Close"].iloc[0] * cfg.initial_capital
        >>> plot_equity_curve(result, benchmark=bm_returns)

    Args:
        result    (BacktestResult):    Output from run_backtest().
        benchmark (pd.Series, opt):    Price series normalised to the same
                                       starting capital for comparison.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print(
            "matplotlib is not installed. Run: pip install matplotlib\n"
            "Cannot plot equity curve."
        )
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7),
                                   gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(f"{result.strategy_name} — {result.symbol}", fontsize=13)

    # Equity curve
    ax1.plot(result.equity_curve.index, result.equity_curve.values,
             label=result.strategy_name, linewidth=1.5, color="steelblue")
    if benchmark is not None:
        bm_aligned = benchmark.reindex(result.equity_curve.index, method="ffill")
        ax1.plot(bm_aligned.index, bm_aligned.values,
                 label="Benchmark (Buy & Hold)", linewidth=1, color="orange",
                 linestyle="--", alpha=0.8)
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.legend(loc="upper left")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.grid(alpha=0.3)

    # Drawdown
    cum = result.equity_curve
    rolling_max = cum.cummax()
    dd = (cum - rolling_max) / rolling_max * 100
    ax2.fill_between(dd.index, dd.values, 0, color="crimson", alpha=0.4, label="Drawdown %")
    ax2.set_ylabel("Drawdown (%)")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


# ============================================================
# EXAMPLE STRATEGY (for documentation / quick testing)
# ============================================================

class MACDMomentumStrategy(Strategy):
    """
    Example Momentum Strategy: MACD + ADX Trend Filter
    ---------------------------------------------------
    What it does
        Generates long signals when the MACD line crosses above its signal
        line in a trending market (ADX > 25). Exits when MACD crosses back
        below the signal line. No short positions.

    Used by
        Provided as a working example and starting template. Copy this
        class, rename it, and modify ``generate_signals`` to create your
        own strategies.

    When to use this as-is
        - As a quick sanity check that your data pipeline and backtest
          engine are working correctly.
        - As a baseline benchmark to compare against more sophisticated
          strategies.

    Code example
        >>> from strategies.test.strategy_tester import (
        ...     MACDMomentumStrategy, BacktestConfig, run_backtest, print_result
        ... )
        >>> cfg    = BacktestConfig(initial_capital=100_000)
        >>> result = run_backtest(MACDMomentumStrategy(), df, cfg, symbol="AAPL")
        >>> print_result(result)
    """
    name = "MACD + ADX Filter"
    symbols = ["AAPL"]

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        from strategies.tools.momentum_tools import macd, adx, crossover, crossunder

        macd_line, sig_line, _ = macd(df["Close"])
        adx_vals, pdi, mdi = adx(df["High"], df["Low"], df["Close"])

        df["signal"] = 0
        buy  = crossover(macd_line, sig_line)  & (adx_vals > 25)
        sell = crossunder(macd_line, sig_line)

        df.loc[buy,  "signal"] = 1
        df.loc[sell, "signal"] = 0
        df["signal"] = df["signal"].ffill().fillna(0)
        return df
