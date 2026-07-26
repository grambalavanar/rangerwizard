"""
paper_trader.py
===============
Standalone paper-trading execution script. Designed to be run on a
repeating schedule (cron, Task Scheduler, launchd) to check signals and
place orders on an IBKR paper-trading account automatically.

How it works
------------
Each run of this script performs one complete cycle:
  1. Connect to IBKR TWS (paper port 7497 by default).
  2. Load the persistent state file (positions, last signal, P&L log).
  3. Fetch recent historical price data for each configured symbol.
  4. Run the strategy's generate_signals() on that data.
  5. Compare the desired signal against the current held position.
  6. Place BUY / SELL market orders on IBKR if the signal changed.
  7. Save the updated state and append a row to the trade log CSV.
  8. Disconnect.

Scheduling (macOS/Linux — cron)
---------------------------------
  # Edit crontab:  crontab -e
  # Run every weekday at 9:35 AM ET (market open + 5 min):
  35 9 * * 1-5 cd /Users/ram/Documents/GH/rangerwizard && \\
      /path/to/python momentum/test/paper_trader.py \\
      --config paper_config.json >> logs/paper_trader.log 2>&1

  # Run every 30 minutes during market hours:
  */30 9-16 * * 1-5 cd /path/to/rangerwizard && \\
      python momentum/test/paper_trader.py --config paper_config.json

Scheduling (macOS — launchd)
------------------------------
  Create a .plist in ~/Library/LaunchAgents/ pointing to this script.
  Use ``launchctl load`` to register it.

Configuration file (paper_config.json)
----------------------------------------
  {
      "strategy_module":  "my_strategies.MACDMomentumStrategy",
      "symbols":          ["AAPL", "MSFT"],
      "bar_size":         "1 day",
      "duration":         "6 M",
      "position_pct":     0.10,
      "max_shares":       500,
      "tws_port":         7497,
      "state_file":       "paper_state.json",
      "trade_log":        "paper_trades.csv",
      "daily_loss_limit": 0.02
  }

  Tip: Change tws_port to 7496 (live TWS) or 4002 (IB Gateway paper)
       only after thorough paper-trading validation.

Dependencies: numpy, pandas, ibapi (installed with twsapi_macunix)
"""

import argparse
import csv
import importlib
import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Allow imports from the repo root
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from myIBApp import myIBApp, connect_to_tws, disconnect_tws
from strategies.test.strategy_tester import Strategy, fetch_ibkr_history

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("paper_trader")


# ============================================================
# CONFIGURATION DATACLASS
# ============================================================

@dataclass
class PaperConfig:
    """
    Configuration for the paper trader.
    -------------------------------------
    What it is
        A dataclass holding every setting the paper trader needs. Loaded
        from a JSON file at startup (``--config paper_config.json``) so
        you can change settings without modifying code.

    Used by
        ``PaperTrader.__init__`` reads this on every run. The JSON config
        file is the primary way to configure the paper trader for
        scheduling.

    When to adjust
        - Change ``tws_port`` to 7496 (TWS live) or 4002 (Gateway paper)
          when you are ready to move from paper to live.
        - Reduce ``position_pct`` for more conservative position sizing.
        - Set ``daily_loss_limit`` to halt trading if you lose more than
          X% of capital in a single day (e.g., 0.02 = 2%).
        - Adjust ``bar_size`` and ``duration`` to match your strategy's
          required indicator warm-up period (e.g., a 200-day SMA needs at
          least 200 bars of 1-day data → duration "1 Y").

    Fields
        strategy_module   Dotted import path to Strategy subclass,
                          e.g. "my_strategies.MACDMomentumStrategy".
        symbols           List of ticker symbols to trade.
        bar_size          IBKR bar size string for historical data.
        duration          IBKR duration string for historical data.
        position_pct      Fraction of capital per position (e.g. 0.10).
        max_shares        Hard cap on shares per order (safety limit).
        tws_port          IBKR TWS port: 7497=paper, 7496=live, 4002=GW.
        tws_host          IBKR host (default "127.0.0.1").
        tws_client_id     Client ID for the TWS connection.
        state_file        Path to the JSON file storing persistent state.
        trade_log         Path to the CSV file recording every trade.
        daily_loss_limit  Fraction of capital: halt if daily loss exceeds
                          this (e.g., 0.02 = halt if down 2% today).
        dry_run           If True, compute signals but do NOT place orders.
                          Useful for validating signal logic before going live.
    """
    strategy_module:   str        = "momentum.test.strategy_tester.MACDMomentumStrategy"
    symbols:           List[str]  = field(default_factory=lambda: ["AAPL"])
    bar_size:          str        = "1 day"
    duration:          str        = "6 M"
    position_pct:      float      = 0.10
    max_shares:        int        = 500
    tws_port:          int        = 7497    # 7497 = TWS paper account
    tws_host:          str        = "127.0.0.1"
    tws_client_id:     int        = 1
    state_file:        str        = "paper_state.json"
    trade_log:         str        = "paper_trades.csv"
    daily_loss_limit:  float      = 0.02
    dry_run:           bool       = False

    @classmethod
    def from_json(cls, path: str) -> "PaperConfig":
        """
        Load a PaperConfig from a JSON file.
        ---------------------------------------
        What it does
            Reads a JSON configuration file and returns a populated
            ``PaperConfig``. Keys in the JSON map directly to the
            dataclass fields. Missing keys use the dataclass defaults.

        Code example
            >>> cfg = PaperConfig.from_json("paper_config.json")
            >>> print(cfg.tws_port, cfg.symbols)

        Args:
            path (str): Path to the JSON config file.

        Returns:
            PaperConfig: Populated configuration object.
        """
        with open(path, "r") as f:
            data = json.load(f)
        # Only assign fields that exist in the dataclass
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    def to_json(self, path: str) -> None:
        """Save this config to a JSON file."""
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)


# ============================================================
# PERSISTENT STATE
# ============================================================

@dataclass
class SymbolState:
    """
    Persistent position and signal state for a single symbol.
    -----------------------------------------------------------
    What it is
        Stores everything the paper trader needs to remember between
        scheduled runs for one symbol: what position is currently held,
        at what price it was entered, and what the last signal was.

    Used by
        ``TradingState`` holds one ``SymbolState`` per symbol. The paper
        trader compares the current desired signal against
        ``last_signal`` to decide whether to act.

    When it matters
        - The paper trader runs on a schedule and exits between runs, so
          it cannot keep this in memory. It must persist to disk.
        - Prevents re-entering a position on every run if the signal
          hasn't changed.
        - Lets you audit the history of signal changes and fills.

    Fields
        position        Current position: 1=long, -1=short, 0=flat.
        shares          Number of shares currently held.
        entry_price     Average fill price of the current position.
        entry_time      ISO timestamp when the current position was entered.
        last_signal     The signal value on the most recent run.
        last_run        ISO timestamp of the most recent successful run.
        ibkr_order_id   IBKR order ID of the most recent placed order.
    """
    position:       int   = 0
    shares:         int   = 0
    entry_price:    float = 0.0
    entry_time:     str   = ""
    last_signal:    int   = 0
    last_run:       str   = ""
    ibkr_order_id:  int   = -1


@dataclass
class TradingState:
    """
    Complete persistent state for the paper trader across all symbols.
    ------------------------------------------------------------------
    What it is
        A container for all per-symbol states plus account-level info
        (starting capital, realised P&L). Serialised to and from a JSON
        file between scheduled runs.

    Used by
        ``PaperTrader.load_state()`` reads this at startup.
        ``PaperTrader.save_state()`` writes this after every run.

    Fields
        capital         Available cash balance (approximate).
        realised_pnl    Cumulative realised P&L from all closed trades.
        symbols         Dict of {symbol: SymbolState}.
        session_date    Date of the current trading session (YYYY-MM-DD).
        daily_pnl       P&L accumulated so far today (for loss limit check).
    """
    capital:      float                  = 100_000.0
    realised_pnl: float                  = 0.0
    symbols:      Dict[str, SymbolState] = field(default_factory=dict)
    session_date: str                    = ""
    daily_pnl:    float                  = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TradingState":
        sym = {k: SymbolState(**v) for k, v in d.get("symbols", {}).items()}
        return cls(
            capital=d.get("capital", 100_000.0),
            realised_pnl=d.get("realised_pnl", 0.0),
            symbols=sym,
            session_date=d.get("session_date", ""),
            daily_pnl=d.get("daily_pnl", 0.0),
        )


# ============================================================
# PAPER TRADER CLASS
# ============================================================

class PaperTrader:
    """
    Automated paper-trading engine.
    ---------------------------------
    What it is
        Orchestrates the full signal-to-order cycle for one scheduled run:
        connect → load state → fetch data → compute signals → place orders
        → save state → log. Designed to be instantiated fresh each run
        (no persistent in-memory state between cron executions).

    Used by
        Called by ``main()`` at the bottom of this file. You should not
        need to modify this class unless you want custom order routing or
        more complex multi-leg strategies.

    When you need to interact with it
        - Call ``run()`` to execute a complete paper-trading cycle.
        - Use ``dry_run=True`` in the config to test signal logic without
          placing real orders on the paper account.
        - Read the state file (``paper_state.json``) and trade log CSV
          (``paper_trades.csv``) to review performance.
    """

    def __init__(self, strategy: Strategy, config: PaperConfig) -> None:
        self.strategy = strategy
        self.config   = config
        self.app: Optional[myIBApp] = None

    # ── State management ──────────────────────────────────────────────────────

    def load_state(self) -> TradingState:
        """
        Load persistent trading state from the JSON state file.
        ---------------------------------------------------------
        What it does
            Reads the state file written by the previous run. If the file
            does not exist (first run), returns a fresh default state with
            ``initial_capital = 100_000``.

        Used by
            Called at the top of ``run()`` before any signal computation.
            The state tells us what positions are currently open so we
            don't accidentally double-enter.

        When you need this
            - You don't call this directly; ``run()`` calls it automatically.
            - You *can* call it to inspect current positions from a script:
              ``state = trader.load_state()``.

        Code example
            >>> state = trader.load_state()
            >>> print(state.symbols.get("AAPL"))

        Returns:
            TradingState: Loaded or default state.
        """
        if not os.path.exists(self.config.state_file):
            log.info("No state file found — initialising fresh state.")
            return TradingState()
        try:
            with open(self.config.state_file, "r") as f:
                d = json.load(f)
            return TradingState.from_dict(d)
        except (json.JSONDecodeError, KeyError) as exc:
            log.warning(f"State file corrupt ({exc}). Starting fresh.")
            return TradingState()

    def save_state(self, state: TradingState) -> None:
        """
        Write the current trading state to the JSON state file.
        ---------------------------------------------------------
        What it does
            Serialises the ``TradingState`` to disk so the next scheduled
            run can pick up exactly where this run left off.

        Used by
            Called at the end of ``run()`` after all order actions are
            complete. Also called in the exception handler so state is
            preserved even if a run fails midway.

        When you need this
            - Automatically called by ``run()``.
            - If you manually modify positions outside this script (e.g.
              you manually closed a position in TWS), edit the state file
              directly or delete it to reset to a clean state.

        Args:
            state (TradingState): Current state to persist.
        """
        tmp = self.config.state_file + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state.to_dict(), f, indent=2)
        os.replace(tmp, self.config.state_file)
        log.info(f"State saved → {self.config.state_file}")

    # ── Data fetching ─────────────────────────────────────────────────────────

    def fetch_market_data(self, symbol: str) -> pd.DataFrame:
        """
        Fetch recent historical OHLCV data from IBKR for one symbol.
        --------------------------------------------------------------
        What it does
            Calls ``fetch_ibkr_history()`` from strategy_tester.py using
            the parameters in the config. Returns a DataFrame ready for
            passing to ``strategy.generate_signals()``.

        Used by
            Called inside ``run()`` for each symbol. The resulting DataFrame
            is the price history the strategy uses to compute its current
            signal.

        When you need this
            - Automatically called by ``run()``.
            - You can call it manually to inspect the raw data IBKR is
              returning for a symbol before running the strategy.

        Code example
            >>> trader.app = connect_to_tws()
            >>> df = trader.fetch_market_data("AAPL")
            >>> print(df.tail())

        Args:
            symbol (str): Ticker symbol.

        Returns:
            pd.DataFrame: OHLCV data with DatetimeIndex.

        Raises:
            RuntimeError: If TWS is not connected or data fetch fails.
        """
        if self.app is None:
            raise RuntimeError("Not connected to TWS. Call connect() first.")
        log.info(f"Fetching {self.config.duration} of {self.config.bar_size} bars for {symbol}")
        return fetch_ibkr_history(
            self.app,
            symbol=symbol,
            duration=self.config.duration,
            bar_size=self.config.bar_size,
        )

    # ── Signal computation ────────────────────────────────────────────────────

    def compute_signal(self, df: pd.DataFrame) -> int:
        """
        Run the strategy on the price DataFrame and return the current signal.
        -----------------------------------------------------------------------
        What it does
            Calls ``strategy.generate_signals(df)`` and returns the most
            recent signal value (last row of the ``signal`` column).
            This is the signal the paper trader will act on.

        Used by
            Called in ``run()`` for each symbol after fetching price data.
            The returned integer drives the buy/sell/hold decision.

        When you need this
            - Automatically called by ``run()``.
            - You can call it manually to inspect what signal the strategy
              would generate right now without placing any orders.

        Code example
            >>> df = trader.fetch_market_data("AAPL")
            >>> sig = trader.compute_signal(df)
            >>> print(f"Current signal: {sig}")  # 1=long, -1=short, 0=flat

        Args:
            df (pd.DataFrame): OHLCV DataFrame from fetch_market_data().

        Returns:
            int: Current signal (1=long, -1=short, 0=flat/cash).
        """
        df = self.strategy.generate_signals(df.copy())
        if "signal" not in df.columns:
            raise ValueError("Strategy did not produce a 'signal' column.")
        signal_val = int(df["signal"].iloc[-1])
        # Clamp to valid values
        if signal_val not in (-1, 0, 1):
            signal_val = 0
        log.info(f"Current signal: {signal_val}")
        return signal_val

    # ── Order execution ───────────────────────────────────────────────────────

    def get_current_price(self, symbol: str) -> float:
        """
        Request the current bid/ask midpoint price from IBKR.
        -------------------------------------------------------
        What it does
            Makes a snapshot price request via the already-connected app
            and returns the last (or ask) price for position sizing
            and P&L calculations.

        Used by
            Called inside ``execute_signal()`` to determine how many
            shares to buy/sell based on current price and capital.

        When you need this
            - Automatically called by ``execute_signal()``.
            - You can call it manually to check the current market price
              before a run.

        Code example
            >>> price = trader.get_current_price("AAPL")
            >>> print(f"AAPL current price: ${price:.2f}")

        Args:
            symbol (str): Ticker symbol.

        Returns:
            float: Current market price. Returns 0.0 on failure.
        """
        if self.app is None:
            return 0.0

        req_id = 8001
        done_event = threading.Event()
        price_holder: List[float] = []

        _orig = getattr(self.app, "tickPrice", None)

        def _tick(rid, tick_type, price, attrib):
            # tick_type 4=Last, 2=Ask; use either
            if rid == req_id and tick_type in (2, 4) and price > 0:
                price_holder.append(price)
                done_event.set()
            if _orig:
                _orig(rid, tick_type, price, attrib)

        _orig_snap_end = getattr(self.app, "tickSnapshotEnd", None)

        def _snap_end(rid):
            if rid == req_id:
                done_event.set()
            if _orig_snap_end:
                _orig_snap_end(rid)

        self.app.tickPrice = _tick
        self.app.tickSnapshotEnd = _snap_end

        try:
            contract = self.app.make_stock_contract(symbol)
            self.app.req_id_to_ticker[req_id] = symbol
            self.app.reqMktData(req_id, contract, "", True, False, [])
            done_event.wait(timeout=10)
            return float(price_holder[-1]) if price_holder else 0.0
        finally:
            self.app.tickPrice = _orig
            self.app.tickSnapshotEnd = _orig_snap_end
            self.app.cancelMktData(req_id)

    def execute_signal(
        self,
        symbol: str,
        desired_signal: int,
        sym_state: SymbolState,
        state: TradingState,
    ) -> Optional[dict]:
        """
        Compare the desired signal to the current position and place orders.
        ---------------------------------------------------------------------
        What it does
            Implements the core signal-to-order logic:
            - If signal matches current position → do nothing.
            - If signal = 1 (long) and currently flat → BUY.
            - If signal = 0 (flat) and currently long → SELL (exit).
            - If signal = -1 (short) and currently long → SELL then SHORT.
            Also enforces the daily loss limit: refuses to trade if today's
            P&L has already breached the configured threshold.

        Used by
            Called inside ``run()`` for each symbol. Returns a dict with
            trade details if an order was placed, or None if no action taken.

        When you need this
            - Automatically called by ``run()``.
            - Review the returned dict in the logs to confirm what action
              was taken on each run.

        Args:
            symbol         (str):         Ticker symbol.
            desired_signal (int):         Signal from compute_signal().
            sym_state      (SymbolState): Current position/state for this symbol.
            state          (TradingState): Full account state.

        Returns:
            Optional[dict]: Trade action dict if an order was placed, else None.
                            Keys: symbol, action, shares, price, order_id,
                            timestamp, reason.
        """
        # Daily loss limit check
        if state.daily_pnl < -(state.capital * self.config.daily_loss_limit):
            log.warning(
                f"Daily loss limit hit ({state.daily_pnl:.2f}). "
                f"Skipping all trades for {symbol}."
            )
            return None

        # No change in signal
        if desired_signal == sym_state.last_signal and desired_signal == sym_state.position:
            log.info(f"{symbol}: No signal change (signal={desired_signal}, pos={sym_state.position}). Holding.")
            return None

        action_taken = None

        # ── Exit existing position ──────────────────────────────────────────
        if sym_state.position != 0 and sym_state.shares > 0:
            if desired_signal != sym_state.position:
                exit_action = "SELL" if sym_state.position == 1 else "BUY"
                log.info(
                    f"{symbol}: Exiting {sym_state.position} position "
                    f"({sym_state.shares} shares) → {exit_action}"
                )
                price = self.get_current_price(symbol)
                order_id = self._place_order(symbol, exit_action, sym_state.shares)
                if order_id is not None:
                    # Estimate P&L
                    pnl_est = (
                        (price - sym_state.entry_price) * sym_state.shares * sym_state.position
                    )
                    state.realised_pnl += pnl_est
                    state.daily_pnl    += pnl_est
                    state.capital      += sym_state.shares * price * sym_state.position
                    sym_state.position     = 0
                    sym_state.shares       = 0
                    sym_state.entry_price  = 0.0
                    sym_state.ibkr_order_id = order_id
                    action_taken = {
                        "symbol":    symbol,
                        "action":    exit_action,
                        "shares":    sym_state.shares,
                        "price":     price,
                        "order_id":  order_id,
                        "timestamp": datetime.now().isoformat(),
                        "reason":    "signal_change_exit",
                        "pnl_est":   pnl_est,
                    }

        # ── Enter new position ───────────────────────────────────────────────
        if desired_signal != 0:
            price = self.get_current_price(symbol)
            if price <= 0:
                log.warning(f"{symbol}: Could not get price. Skipping entry.")
                return action_taken

            dollar_size = state.capital * self.config.position_pct
            shares = int(min(dollar_size / price, self.config.max_shares))
            if shares <= 0:
                log.warning(f"{symbol}: Computed 0 shares. Skipping entry.")
                return action_taken

            entry_action = "BUY" if desired_signal == 1 else "SELL"
            log.info(
                f"{symbol}: Entering {desired_signal} position. "
                f"{entry_action} {shares} shares @ ~${price:.2f}"
            )
            order_id = self._place_order(symbol, entry_action, shares)
            if order_id is not None:
                cost = shares * price * desired_signal
                state.capital -= abs(cost)
                sym_state.position      = desired_signal
                sym_state.shares        = shares
                sym_state.entry_price   = price
                sym_state.entry_time    = datetime.now().isoformat()
                sym_state.ibkr_order_id = order_id
                action_taken = {
                    "symbol":    symbol,
                    "action":    entry_action,
                    "shares":    shares,
                    "price":     price,
                    "order_id":  order_id,
                    "timestamp": datetime.now().isoformat(),
                    "reason":    "signal_entry",
                    "pnl_est":   0.0,
                }

        sym_state.last_signal = desired_signal
        sym_state.last_run = datetime.now().isoformat()
        return action_taken

    def _place_order(self, symbol: str, action: str, shares: int) -> Optional[int]:
        """
        Place a market order on IBKR and return the assigned order ID.
        ----------------------------------------------------------------
        What it does
            Builds an IBKR contract and market order using the myIBApp
            helper methods, then calls ``placeOrder``. In dry-run mode it
            logs the order but does NOT submit it to IBKR.

        Used by
            Called by ``execute_signal()`` for every buy or sell action.
            This is the only point where real IBKR API calls are made to
            submit orders.

        When you need this
            - Automatically called by ``execute_signal()``.
            - If you want to add custom order types (e.g., limit orders,
              bracket orders with stop-loss), modify this method.

        Code example
            >>> order_id = trader._place_order("AAPL", "BUY", 50)
            >>> print(f"Order placed: {order_id}")

        Args:
            symbol (str):  Ticker symbol.
            action (str):  "BUY" or "SELL".
            shares (int):  Number of shares.

        Returns:
            Optional[int]: IBKR order ID, or None if placement failed.
        """
        if self.config.dry_run:
            log.info(f"[DRY RUN] Would place: {action} {shares} {symbol}")
            return -1  # Fake order ID for dry run

        if self.app is None:
            log.error("Not connected to TWS. Cannot place order.")
            return None

        try:
            contract = self.app.make_stock_contract(symbol)
            order    = self.app.make_market_order(action, shares)
            self.app.next_id += 1
            order_id = self.app.next_id
            self.app.placeOrder(order_id, contract, order)
            log.info(f"Order submitted: {action} {shares} {symbol} — orderId={order_id}")
            # Brief pause to allow TWS to process
            time.sleep(1)
            return order_id
        except Exception as exc:
            log.error(f"Failed to place {action} {shares} {symbol}: {exc}")
            return None

    # ── Trade logging ─────────────────────────────────────────────────────────

    def log_trade(self, trade: dict) -> None:
        """
        Append a completed trade action to the CSV trade log.
        -------------------------------------------------------
        What it does
            Writes one row to the trade log CSV file. Creates the file with
            a header row if it does not already exist. The trade log is a
            permanent record of every order placed by the paper trader.

        Used by
            Called inside ``run()`` after every successful order execution.
            The CSV accumulates over time and can be loaded into a DataFrame
            for P&L analysis.

        When you need this
            - Automatically called by ``run()``.
            - Load the CSV to review your paper trading history:
              ``df = pd.read_csv("paper_trades.csv", parse_dates=["timestamp"])``

        Args:
            trade (dict): Trade detail dict from execute_signal().
        """
        fieldnames = ["timestamp", "symbol", "action", "shares", "price",
                      "order_id", "reason", "pnl_est"]
        file_exists = os.path.exists(self.config.trade_log)
        with open(self.config.trade_log, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
            writer.writerow(trade)
        log.info(f"Trade logged → {self.config.trade_log}")

    # ── Main run cycle ────────────────────────────────────────────────────────

    def run(self) -> None:
        """
        Execute one complete paper-trading cycle.
        ------------------------------------------
        What it does
            The top-level method called by the scheduler on each run.
            Performs the full connect → signal → order → save loop for
            every configured symbol. Handles connection errors gracefully
            and always saves state even if an exception occurs.

        Used by
            Called by ``main()`` at the bottom of this file. If you are
            integrating this into a larger application, call this method
            directly.

        When you need this
            - This is the single entry point for a scheduled run.
            - You can also call it manually in a REPL or test script to
              simulate one run cycle:
                ``trader.run()``

        Scheduling example (cron)
            # 9:35 AM ET, weekdays only
            35 9 * * 1-5 python /path/to/paper_trader.py --config cfg.json
        """
        log.info("=" * 60)
        log.info(f"Paper Trader starting — Strategy: {self.strategy.name}")
        log.info(f"Symbols: {self.config.symbols}  |  dry_run={self.config.dry_run}")
        log.info("=" * 60)

        state = self.load_state()

        # Reset daily P&L if it's a new trading day
        today = str(date.today())
        if state.session_date != today:
            log.info(f"New session: {today}. Resetting daily P&L.")
            state.session_date = today
            state.daily_pnl = 0.0

        # Ensure each symbol has a state entry
        for sym in self.config.symbols:
            if sym not in state.symbols:
                state.symbols[sym] = SymbolState()

        # Connect to TWS
        try:
            log.info(
                f"Connecting to TWS at {self.config.tws_host}:{self.config.tws_port} "
                f"(clientId={self.config.tws_client_id})"
            )
            self.app = myIBApp()
            self.app.connect(
                self.config.tws_host,
                self.config.tws_port,
                self.config.tws_client_id,
            )
            api_thread = threading.Thread(target=self.app.run, daemon=True)
            api_thread.start()
            time.sleep(2)   # wait for connection and nextValidId
            log.info("Connected.")
        except Exception as exc:
            log.error(f"Failed to connect to TWS: {exc}")
            self.save_state(state)
            return

        try:
            for symbol in self.config.symbols:
                log.info(f"--- Processing {symbol} ---")
                sym_state = state.symbols[symbol]

                # 1. Fetch price data
                try:
                    df = self.fetch_market_data(symbol)
                    log.info(f"{symbol}: {len(df)} bars fetched "
                             f"({df.index[0].date()} → {df.index[-1].date()})")
                except Exception as exc:
                    log.error(f"{symbol}: Data fetch failed: {exc}. Skipping.")
                    continue

                # 2. Compute signal
                try:
                    signal = self.compute_signal(df)
                    log.info(f"{symbol}: Signal = {signal} "
                             f"({'LONG' if signal == 1 else 'SHORT' if signal == -1 else 'FLAT'})")
                except Exception as exc:
                    log.error(f"{symbol}: Signal computation failed: {exc}. Skipping.")
                    continue

                # 3. Execute
                try:
                    trade = self.execute_signal(symbol, signal, sym_state, state)
                    if trade:
                        self.log_trade(trade)
                except Exception as exc:
                    log.error(f"{symbol}: Order execution failed: {exc}.")

        finally:
            # Always save state and disconnect
            self.save_state(state)
            try:
                self.app.disconnect()
                log.info("Disconnected from TWS.")
            except Exception:
                pass
            self.app = None

        log.info("Paper Trader run complete.")


# ============================================================
# STRATEGY LOADER
# ============================================================

def load_strategy(module_path: str) -> Strategy:
    """
    Dynamically load a Strategy subclass from a dotted module path.
    ----------------------------------------------------------------
    What it does
        Splits the ``module_path`` string (e.g.
        ``"my_strategies.MACDMomentumStrategy"``) into a module and class
        name, imports the module, and instantiates the class.

    Used by
        Called by ``main()`` to load the strategy specified in the JSON
        config file without requiring you to hard-code the strategy class
        in this file. This allows the paper trader to be reused with any
        strategy.

    When you need this
        - Automatically called by ``main()``.
        - If you want to load a strategy programmatically:
          ``strategy = load_strategy("my_strategies.MACDMomentumStrategy")``

    Code example
        >>> strategy = load_strategy("momentum.test.strategy_tester.MACDMomentumStrategy")
        >>> print(strategy.name)

    Args:
        module_path (str): Dotted path to the Strategy subclass, e.g.
                           ``"momentum.test.strategy_tester.MACDMomentumStrategy"``

    Returns:
        Strategy: Instantiated strategy object.

    Raises:
        ImportError:   If the module cannot be found.
        AttributeError: If the class doesn't exist in the module.
        TypeError:     If the class is not a Strategy subclass.
    """
    parts = module_path.rsplit(".", 1)
    if len(parts) != 2:
        raise ValueError(
            f"strategy_module must be 'module.ClassName', got: '{module_path}'"
        )
    module_name, class_name = parts
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise ImportError(
            f"Cannot import module '{module_name}'. "
            f"Ensure the module is on the Python path."
        ) from exc

    cls = getattr(module, class_name, None)
    if cls is None:
        raise AttributeError(
            f"Class '{class_name}' not found in module '{module_name}'."
        )
    if not (isinstance(cls, type) and issubclass(cls, Strategy)):
        raise TypeError(
            f"'{class_name}' is not a subclass of Strategy."
        )
    return cls()


# ============================================================
# UTILITY: REVIEW PAPER TRADING PERFORMANCE
# ============================================================

def review_paper_performance(
    trade_log_path: str,
    state_file_path: str,
) -> None:
    """
    Print a summary of paper trading performance from the trade log.
    -----------------------------------------------------------------
    What it does
        Reads the CSV trade log and the JSON state file and prints a
        formatted summary: total trades, realised P&L, win rate, and
        current open positions.

    Used by
        Run this at any time to review how your paper trading strategy
        is performing without having to manually inspect the CSV.

    When you need this
        - Periodically review your paper trading results to decide if
          the strategy is ready to move to a live account.
        - After an unusual market day to see if the system behaved as
          expected.
        - Compare paper trading performance against the backtest results
          to check for live-vs-backtest discrepancies.

    Code example
        >>> from momentum.test.paper_trader import review_paper_performance
        >>> review_paper_performance("paper_trades.csv", "paper_state.json")

    Args:
        trade_log_path  (str): Path to the paper_trades.csv file.
        state_file_path (str): Path to the paper_state.json file.
    """
    print("\n" + "=" * 56)
    print("  Paper Trading Performance Review")
    print("=" * 56)

    # State file
    if os.path.exists(state_file_path):
        with open(state_file_path, "r") as f:
            d = json.load(f)
        state = TradingState.from_dict(d)
        print(f"  Cash Available   : ${state.capital:>12,.2f}")
        print(f"  Realised P&L     : ${state.realised_pnl:>+12,.2f}")
        print(f"  Today's P&L      : ${state.daily_pnl:>+12,.2f}")
        print(f"  Session Date     : {state.session_date}")
        print()
        print("  Open Positions:")
        for sym, ss in state.symbols.items():
            if ss.position != 0:
                print(
                    f"    {sym:8s}  {'+LONG' if ss.position == 1 else '-SHORT':6s}  "
                    f"{ss.shares:4d} shares  entry=${ss.entry_price:.2f}"
                )
        if not any(ss.position != 0 for ss in state.symbols.values()):
            print("    (none)")
    else:
        print("  No state file found.")

    print()

    # Trade log
    if os.path.exists(trade_log_path):
        df = pd.read_csv(trade_log_path, parse_dates=["timestamp"])
        print(f"  Total Log Entries: {len(df)}")
        if "pnl_est" in df.columns:
            df["pnl_est"] = pd.to_numeric(df["pnl_est"], errors="coerce").fillna(0)
            completed = df[df["pnl_est"] != 0]
            if len(completed) > 0:
                wins = (completed["pnl_est"] > 0).sum()
                wr = wins / len(completed)
                print(f"  Completed Trades : {len(completed)}")
                print(f"  Win Rate (est.)  : {wr:.1%}")
                print(f"  Total P&L (est.) : ${completed['pnl_est'].sum():>+,.2f}")
    else:
        print("  No trade log found.")

    print("=" * 56 + "\n")


# ============================================================
# ENTRY POINT
# ============================================================

def main() -> None:
    """
    Command-line entry point for the paper trader.
    -----------------------------------------------
    What it does
        Parses command-line arguments, loads the config file and strategy,
        then calls ``PaperTrader.run()`` once. Designed to be invoked by
        a scheduler (cron/launchd) on a repeating basis.

    CLI usage
        python paper_trader.py --config paper_config.json
        python paper_trader.py --config paper_config.json --dry-run
        python paper_trader.py --review  --config paper_config.json

    Arguments
        --config CONFIG_FILE  Path to the JSON config file (required).
        --dry-run             Override config and force dry-run mode
                              (computes signals but does NOT submit orders).
        --review              Print a performance summary and exit without
                              running the trading cycle.

    When to use
        - Schedule with cron for automated daily/intraday execution.
        - Run with ``--dry-run`` to test that the strategy, config, and
          TWS connection all work before enabling live orders.
        - Run with ``--review`` anytime to check current P&L status.

    Cron example (run every weekday at 9:35 AM ET)
        35 9 * * 1-5 cd /path/to/rangerwizard && \\
            python momentum/test/paper_trader.py --config paper_config.json \\
            >> logs/paper_trader.log 2>&1
    """
    parser = argparse.ArgumentParser(
        description="IBKR Paper Trading Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to the JSON configuration file (e.g. paper_config.json).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Compute signals but do NOT submit any orders to IBKR.",
    )
    parser.add_argument(
        "--review", action="store_true",
        help="Print a performance summary and exit.",
    )
    args = parser.parse_args()

    # Load config
    if not os.path.exists(args.config):
        log.error(f"Config file not found: {args.config}")
        sys.exit(1)

    config = PaperConfig.from_json(args.config)

    if args.dry_run:
        config.dry_run = True
        log.info("DRY RUN mode enabled — no orders will be submitted.")

    # Review-only mode
    if args.review:
        review_paper_performance(config.trade_log, config.state_file)
        sys.exit(0)

    # Load strategy
    try:
        strategy = load_strategy(config.strategy_module)
        log.info(f"Loaded strategy: {strategy.name}")
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        log.error(f"Failed to load strategy '{config.strategy_module}': {exc}")
        sys.exit(1)

    # Run
    trader = PaperTrader(strategy, config)
    trader.run()


if __name__ == "__main__":
    main()
