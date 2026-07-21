"""
daily_signal.py
===============
Daily momentum signal runner. Designed to execute at 9:45 AM ET on
weekdays (15 minutes after market open, after the opening noise settles).

What it does each run
----------------------
  1. Connect to IBKR TWS (paper by default; use --live for real money).
  2. Fetch account net liquidation value.
  3. Fetch current open positions.
  4. For each symbol in the configured list:
       a. Download 1 year of daily OHLCV from IBKR historical data.
       b. Run the AlphaCompositeMomentumStrategy with optimised parameters.
       c. Compare the strategy's desired position to what is actually held.
       d. If action required, place a market order capped at 1% of NLV.
  5. Log every signal, decision, and fill to the logs/ folder.
  6. Disconnect cleanly.

Logs folder  (excluded from git — see .gitignore)
-------------------------------------------------
  logs/daily_signal_YYYY-MM-DD.log   Full text log of the run
  logs/decisions_YYYY-MM-DD.csv      Per-symbol signal decisions
  logs/trades_YYYY-MM-DD.csv         Orders actually submitted to IBKR

The CSV logs are structured for future ML training: each row records
the composite score, every indicator sub-score, and the eventual trade
outcome so you can later correlate signal quality with profitability.

Scheduling (cron — runs at 9:45 AM ET on weekdays)
-----------------------------------------------------
  45 9 * * 1-5 cd /Users/ram/Documents/GH/rangerwizard && \\
      python3.11 daily_signal.py >> logs/cron.log 2>&1

Scheduling (launchd — macOS)
------------------------------
  Create ~/Library/LaunchAgents/com.rangerwizard.daily_signal.plist
  pointing to this script with a StartCalendarInterval of {Hour: 9, Minute: 45}.

Usage
------
  python3.11 daily_signal.py                             # paper, default symbols
  python3.11 daily_signal.py --symbols AAPL MSFT SPY     # custom symbols
  python3.11 daily_signal.py --universe megacap           # pre-defined basket
  python3.11 daily_signal.py --live                       # real money (careful!)
  python3.11 daily_signal.py --dry-run                    # compute signals only, no orders
  python3.11 daily_signal.py --risk-pct 0.005            # 0.5% of NLV per trade

Dependencies: numpy, pandas, ibapi (twsapi_macunix)
"""

import argparse
import csv
import json
import logging
import os
import sys
import threading
import time
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Repo root on path ─────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from myIBApp import myIBApp, connect_to_tws
from momentum.strategies.alpha_composite import AlphaCompositeMomentumStrategy

# ── Constants ──────────────────────────────────────────────────────────────────
LOG_DIR      = os.path.join(_ROOT, "logs")
PARAMS_FILE  = os.path.join(_ROOT, "momentum", "test", "runs", "alpha_composite_opt.json")
PORT_PAPER   = 7497
PORT_LIVE    = 7496
CLIENT_ID    = 10       # use a distinct client ID to avoid conflicts with manual TWS
MAX_SHARES   = 1000     # absolute cap per order regardless of NLV size
HISTORY_BARS = "1 Y"    # IBKR historical data duration for signal computation

# Pre-defined symbol universes (same as backtest_optimized.py)
UNIVERSES: Dict[str, List[str]] = {
    "megacap": ["AAPL", "NVDA", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "LLY", "BRK-B"],
    "movers":  ["TSLA", "NVDA", "AMD", "PLTR", "MSTR", "SMCI", "COIN", "MRNA", "SHOP", "CELH"],
    "volume":  ["SPY", "QQQ", "AAPL", "TSLA", "NVDA", "AMZN", "AMD", "SOXL", "BAC", "F"],
}
UNIVERSES["all"] = sorted(set(s for v in UNIVERSES.values() for s in v))
DEFAULT_SYMBOLS = ["AMD"]


# ============================================================
# 1. SETUP
# ============================================================

def setup_logging(today: str) -> logging.Logger:
    """
    Configure file + console logging for the daily run.
    ------------------------------------------------------
    What it does
        Creates the logs/ directory if it doesn't exist, then attaches
        two handlers to the root logger: a rotating file handler writing to
        ``logs/daily_signal_YYYY-MM-DD.log`` and a stream handler for
        terminal output during manual runs.

    Used by
        Called once at the very start of ``main()``. All subsequent
        ``log.info()``, ``log.warning()``, and ``log.error()`` calls in
        every function automatically write to this log file.

    When you need this
        - Manually: not needed — just run the script.
        - If you want to add a new log destination (e.g., email on errors),
          add another handler here.
        - The log file is excluded from git (logs/ in .gitignore) but stays
          on disk for ML training and post-mortem analysis.

    Code example
        >>> log = setup_logging("2026-07-19")
        >>> log.info("Started daily signal run")

    Args:
        today (str): Date string "YYYY-MM-DD" used in the log filename.

    Returns:
        logging.Logger: Configured logger instance.
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"daily_signal_{today}.log")

    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s",
                            datefmt="%H:%M:%S")
    log = logging.getLogger("daily_signal")
    log.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.DEBUG)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    sh.setLevel(logging.INFO)

    log.addHandler(fh)
    log.addHandler(sh)
    return log


def load_params(params_file: str) -> dict:
    """
    Load the optimised strategy parameters from the JSON file produced by
    the genetic optimizer.
    -----------------------------------------------------------------------
    What it does
        Reads ``momentum/test/runs/alpha_composite_opt.json`` and returns
        the ``best_params`` dictionary. If the file is missing it falls
        back to the strategy's DEFAULT_PARAMS, logging a warning.

    Used by
        Called once in ``main()`` before any backtesting or signal
        computation. The returned dict is passed to every
        ``AlphaCompositeMomentumStrategy(params=...)`` instantiation.

    When you need this
        - After re-running the genetic optimizer (``optimize_alpha.py``),
          the new best params are automatically picked up on the next daily
          run without code changes.
        - If you want to test default (un-optimised) params, delete the JSON
          file — the script will fall back automatically.

    Code example
        >>> params = load_params(PARAMS_FILE)
        >>> strategy = AlphaCompositeMomentumStrategy(params=params)

    Args:
        params_file (str): Path to the JSON file.

    Returns:
        dict: Parameter dict to pass to AlphaCompositeMomentumStrategy.
    """
    from momentum.strategies.alpha_composite import DEFAULT_PARAMS

    log = logging.getLogger("daily_signal")
    if not os.path.exists(params_file):
        log.warning(f"Params file not found: {params_file}. Using DEFAULT_PARAMS.")
        return dict(DEFAULT_PARAMS)

    with open(params_file) as f:
        data = json.load(f)

    params   = data.get("best_params", {})
    fitness  = data.get("best_fitness", float("nan"))
    log.info(f"Loaded params from {params_file}  (GA fitness={fitness:.4f})")
    return params


# ============================================================
# 2. IBKR DATA FETCHING
# ============================================================

def connect_ibkr(port: int, client_id: int = CLIENT_ID) -> myIBApp:
    """
    Open a fresh connection to IBKR TWS or IB Gateway.
    ----------------------------------------------------
    What it does
        Instantiates a new ``myIBApp``, connects on the specified port,
        starts the socket thread, and waits for the first ``nextValidId``
        callback (which confirms the connection is ready).

    Used by
        Called once at the start of ``main()``. Returns the live app object
        that all subsequent IBKR API calls use.

    When you need this
        - Port 7497 (default) = TWS paper trading account. Use this until
          you have at least 3 months of consistent paper profit.
        - Port 7496 = TWS live account. Only activated via --live flag.
        - Port 4002 = IB Gateway paper; 4001 = IB Gateway live.
        - If TWS is not running, this will raise a connection error — start
          TWS or IB Gateway manually before running the script.

    Code example
        >>> app = connect_ibkr(PORT_PAPER)
        >>> print("Connected, next order ID:", app.next_id)

    Args:
        port      (int): TWS port (7497 paper, 7496 live).
        client_id (int): IBKR client ID — use a distinct value from the
                         manual TWS session to avoid conflicts.

    Returns:
        myIBApp: Connected app instance.

    Raises:
        ConnectionError: If TWS is not running or refuses the connection.
    """
    log = logging.getLogger("daily_signal")
    log.info(f"Connecting to IBKR TWS on port {port} (clientId={client_id}) ...")

    app = myIBApp()
    app.connect("127.0.0.1", port, client_id)

    api_thread = threading.Thread(target=app.run, daemon=True)
    api_thread.start()
    time.sleep(2)   # wait for connection + nextValidId

    if not hasattr(app, "next_id") or app.next_id is None:
        raise ConnectionError(
            f"Connection to TWS port {port} failed — is TWS running? "
            f"Check that the port matches your TWS API settings."
        )

    log.info(f"Connected. Next valid order ID: {app.next_id}")
    return app


def fetch_account_nlv(app: myIBApp, timeout: int = 15) -> float:
    """
    Fetch the account Net Liquidation Value (NLV) from IBKR.
    ----------------------------------------------------------
    What it measures
        Net Liquidation Value = total account value if all positions were
        closed at current market prices. This is the primary metric used
        for position sizing: every trade is capped at ``risk_pct × NLV``
        dollars, ensuring consistent risk regardless of account size.

    Used by
        Called once per daily run in ``main()``. The returned NLV is used
        in ``calculate_position_size()`` to determine the number of shares
        to buy on each signal.

    When you need this
        - NLV fluctuates daily as positions move. Fetching it fresh every
          morning ensures position sizing always reflects your actual
          account value, not a stale number.
        - If NLV fetch fails (IBKR API timeout), the script aborts rather
          than sizing orders on stale data — a safety feature.

    Code example
        >>> nlv = fetch_account_nlv(app)
        >>> print(f"Account NLV: ${nlv:,.2f}")
        >>> max_per_trade = nlv * 0.01
        >>> print(f"Max per trade (1%): ${max_per_trade:,.2f}")

    Args:
        app     (myIBApp): Connected IBKR app.
        timeout (int):     Seconds to wait for IBKR response.

    Returns:
        float: Account NLV in USD. Returns 0.0 on failure.
    """
    log     = logging.getLogger("daily_signal")
    req_id  = 9200
    holder: List[float] = []
    done    = threading.Event()

    _orig     = getattr(app, "accountSummary",    None)
    _orig_end = getattr(app, "accountSummaryEnd", None)

    def _summary(rid, account, tag, value, currency):
        if rid == req_id and tag == "NetLiquidation" and currency == "USD":
            holder.append(float(value))
        if _orig:
            _orig(rid, account, tag, value, currency)

    def _summary_end(rid):
        if rid == req_id:
            done.set()
        if _orig_end:
            _orig_end(rid)

    app.accountSummary    = _summary
    app.accountSummaryEnd = _summary_end
    try:
        app.reqAccountSummary(req_id, "All", "NetLiquidation")
        done.wait(timeout=timeout)
    finally:
        if _orig is not None:
            app.accountSummary = _orig
        if _orig_end is not None:
            app.accountSummaryEnd = _orig_end
        try:
            app.cancelAccountSummary(req_id)
        except Exception:
            pass

    nlv = holder[-1] if holder else 0.0
    log.info(f"Account NLV: ${nlv:,.2f}")
    return nlv


def fetch_positions(app: myIBApp, timeout: int = 10) -> Dict[str, Tuple[int, float]]:
    """
    Fetch all currently open positions from the IBKR account.
    ----------------------------------------------------------
    What it measures
        Returns a snapshot of every position held in the account at the
        time of the call: symbol → (quantity, average_cost). Quantity is
        positive for long positions and negative for short positions.

    Used by
        Called once per daily run before evaluating any signals. The
        returned positions dict is compared against each strategy signal
        in ``determine_action()`` to decide whether a BUY, SELL, or HOLD
        is required.

    When you need this
        - The script uses this to avoid double-entering a position
          (already long → signal long → HOLD, not another BUY).
        - If you manually change a position in TWS between runs, this fetch
          picks it up automatically on the next run.
        - Review the positions dict in the log if the script's actions
          don't match your expectations.

    Code example
        >>> positions = fetch_positions(app)
        >>> # {"AAPL": (50, 185.20), "SPY": (10, 510.00)}
        >>> aapl_qty = positions.get("AAPL", (0, 0.0))[0]

    Args:
        app     (myIBApp): Connected IBKR app.
        timeout (int):     Seconds to wait.

    Returns:
        Dict[str, Tuple[int, float]]:
            {symbol: (quantity, average_cost)} for all non-zero positions.
    """
    log       = logging.getLogger("daily_signal")
    positions: Dict[str, Tuple[int, float]] = {}
    done      = threading.Event()

    _orig     = getattr(app, "position",    None)
    _orig_end = getattr(app, "positionEnd", None)

    def _position(account, contract, qty, avg_cost):
        if qty != 0:
            positions[contract.symbol] = (int(qty), float(avg_cost))
        if _orig:
            _orig(account, contract, qty, avg_cost)

    def _position_end():
        done.set()
        if _orig_end:
            _orig_end()

    app.position    = _position
    app.positionEnd = _position_end
    try:
        app.reqPositions()
        done.wait(timeout=timeout)
    finally:
        if _orig is not None:
            app.position = _orig
        if _orig_end is not None:
            app.positionEnd = _orig_end
        try:
            app.cancelPositions()
        except Exception:
            pass

    log.info(f"Open positions: {positions or '(none)'}")
    return positions


def fetch_ohlcv(
    app: myIBApp,
    symbol: str,
    duration: str = HISTORY_BARS,
    bar_size: str = "1 day",
    timeout: int  = 30,
) -> pd.DataFrame:
    """
    Download historical OHLCV bars from IBKR for one symbol.
    ---------------------------------------------------------
    What it fetches
        Daily (or intraday) OHLCV bars from IBKR's historical data
        service. This is the exact same data IBKR uses internally and
        what you see in TWS charts — fully adjusted for splits and
        dividends when ``what_to_show="TRADES"`` is used.

    Used by
        Called for each symbol inside ``main()``'s symbol loop. The
        returned DataFrame is passed directly to the strategy's
        ``generate_signals()`` method.

    When you need this
        - The strategy needs at least 252 bars (1 trading year) of
          daily data to compute all indicators reliably. HISTORY_BARS="1 Y"
          satisfies this.
        - For intraday strategies, change bar_size to "5 mins" and duration
          to "5 D" — but note IBKR rate-limits intraday requests aggressively.
        - If IBKR returns an error (symbol not found, no data permissions),
          the symbol is skipped and a warning is logged.

    IBKR duration strings: "1 D", "5 D", "1 W", "1 M", "3 M", "6 M", "1 Y"
    IBKR bar sizes:        "1 min", "5 mins", "15 mins", "1 hour", "1 day"

    Code example
        >>> df = fetch_ohlcv(app, "AAPL")
        >>> print(df.tail())

    Args:
        app      (myIBApp): Connected IBKR app.
        symbol   (str):     Ticker (e.g. "AAPL").
        duration (str):     How far back to fetch (default "1 Y").
        bar_size (str):     Bar granularity (default "1 day").
        timeout  (int):     Seconds to wait for IBKR response.

    Returns:
        pd.DataFrame: OHLCV with DatetimeIndex and columns
                      Open, High, Low, Close, Volume.

    Raises:
        TimeoutError:  If IBKR does not respond within timeout seconds.
        RuntimeError:  If IBKR returns a data error for the symbol.
    """
    from ibapi.contract import Contract

    log    = logging.getLogger("daily_signal")
    req_id = 9300 + abs(hash(symbol)) % 500
    bars: List[dict] = []
    done  = threading.Event()
    error_msgs: List[str] = []

    _orig_hist     = getattr(app, "historicalData",    None)
    _orig_hist_end = getattr(app, "historicalDataEnd", None)
    _orig_err      = getattr(app, "error",             None)

    def _hist(rid, bar):
        if rid == req_id:
            bars.append({"Date":   bar.date,
                         "Open":   float(bar.open),
                         "High":   float(bar.high),
                         "Low":    float(bar.low),
                         "Close":  float(bar.close),
                         "Volume": float(bar.volume)})
        if _orig_hist:
            _orig_hist(rid, bar)

    def _hist_end(rid, start, end):
        if rid == req_id:
            done.set()
        if _orig_hist_end:
            _orig_hist_end(rid, start, end)

    def _err(rid, code, msg, *a):
        if rid == req_id:
            error_msgs.append(f"[{code}] {msg}")
            done.set()
        if _orig_err:
            _orig_err(rid, code, msg, *a)

    app.historicalData    = _hist
    app.historicalDataEnd = _hist_end
    app.error             = _err

    try:
        contract          = Contract()
        contract.symbol   = symbol
        contract.secType  = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"

        app.reqHistoricalData(
            req_id, contract, "",
            duration, bar_size, "TRADES",
            1, 1, False, []
        )

        if not done.wait(timeout=timeout):
            raise TimeoutError(
                f"Historical data for {symbol} timed out after {timeout}s."
            )
        if error_msgs:
            raise RuntimeError(f"IBKR error for {symbol}: {'; '.join(error_msgs)}")
        if not bars:
            raise RuntimeError(f"No historical bars returned for {symbol}.")

        df = pd.DataFrame(bars)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()
        log.debug(f"{symbol}: {len(df)} bars fetched ({df.index[0].date()} → {df.index[-1].date()})")
        return df

    finally:
        if _orig_hist is not None:
            app.historicalData = _orig_hist
        if _orig_hist_end is not None:
            app.historicalDataEnd = _orig_hist_end
        if _orig_err is not None:
            app.error = _orig_err


def fetch_current_price(app: myIBApp, symbol: str, timeout: int = 10) -> float:
    """
    Fetch the current market price for a symbol via a snapshot quote.
    ------------------------------------------------------------------
    What it fetches
        The most recent ask price (tick type 2) or last traded price
        (tick type 4) from IBKR's real-time market data. Used to
        calculate the dollar cost of each trade for position sizing.

    Used by
        Called in ``calculate_position_size()`` to convert the dollar
        risk limit (NLV × risk_pct) into a share count.

    When you need this
        - If the market is closed (pre-market at 9:45 AM isn't open yet),
          IBKR returns the last closing price, which is acceptable for
          next-open order sizing.
        - If no price is returned (e.g., halted stock), the function
          returns 0.0 and the symbol is skipped.

    Code example
        >>> price = fetch_current_price(app, "AAPL")
        >>> print(f"AAPL current: ${price:.2f}")

    Args:
        app     (myIBApp): Connected IBKR app.
        symbol  (str):     Ticker symbol.
        timeout (int):     Seconds to wait.

    Returns:
        float: Current price. Returns 0.0 on failure.
    """
    log    = logging.getLogger("daily_signal")
    req_id = 9400 + abs(hash(symbol)) % 100
    holder: List[float] = []
    done   = threading.Event()

    _orig     = getattr(app, "tickPrice",      None)
    _orig_end = getattr(app, "tickSnapshotEnd", None)
    _orig_err = getattr(app, "error",           None)

    def _tick(rid, tick_type, price, attrib):
        if rid == req_id and tick_type in (2, 4) and price > 0:
            holder.append(price)
            done.set()
        if _orig:
            _orig(rid, tick_type, price, attrib)

    def _snap_end(rid):
        if rid == req_id:
            done.set()
        if _orig_end:
            _orig_end(rid)

    def _err(rid, code, msg, *a):
        # Suppress error 300 "Can't find EId" for snapshot requests —
        # snapshot mode auto-cancels; the cancel-on-end call is redundant.
        if rid == req_id and code == 300:
            return
        if _orig_err:
            _orig_err(rid, code, msg, *a)

    app.tickPrice       = _tick
    app.tickSnapshotEnd = _snap_end
    app.error           = _err
    try:
        contract = app.make_stock_contract(symbol)
        app.req_id_to_ticker[req_id] = symbol
        # snapshot=True: IBKR auto-cancels after tickSnapshotEnd — do NOT
        # call cancelMktData afterward or IBKR will return error 300.
        app.reqMktData(req_id, contract, "", True, False, [])
        done.wait(timeout=timeout)
    finally:
        if _orig is not None:
            app.tickPrice = _orig
        if _orig_end is not None:
            app.tickSnapshotEnd = _orig_end
        if _orig_err is not None:
            app.error = _orig_err

    price = float(holder[-1]) if holder else 0.0
    if price == 0.0:
        log.warning(f"{symbol}: Could not get price. Will skip order.")
    return price


# ============================================================
# 3. SIGNAL COMPUTATION
# ============================================================

def compute_signal(df: pd.DataFrame, params: dict) -> Tuple[int, float]:
    """
    Run the AlphaCompositeMomentumStrategy and return the latest signal.
    ---------------------------------------------------------------------
    What it computes
        Runs all nine momentum sub-scores (Weinstein trend, TSMOM, linear
        trend, 52-week high, MACD quality, RSI zone, KST, volume, Ichimoku)
        through the weighted composite and returns the most recent bar's
        signal (1=long, 0=flat) and the raw composite score (0–1).

    Used by
        Called for each symbol in the main loop. The returned signal
        drives the BUY/SELL/HOLD decision in ``determine_action()``.

    When you need this
        - The composite score is logged alongside the signal so you can
          later analyse borderline cases (score near the threshold).
        - If you want to see all nine sub-scores individually, call
          ``strategy._compute_composite(df, params)`` and also call each
          ``strategy._score_*()`` method separately.

    Code example
        >>> signal, composite = compute_signal(df, params)
        >>> print(f"Signal: {signal}  Composite: {composite:.3f}")

    Args:
        df     (pd.DataFrame): OHLCV DataFrame with DatetimeIndex.
        params (dict):         Optimised strategy parameters.

    Returns:
        Tuple[int, float]: (signal, composite_score).
            signal = 1 (long) or 0 (flat).
            composite_score in [0, 1].
    """
    strategy = AlphaCompositeMomentumStrategy(params=params)
    df_out   = strategy.generate_signals(df.copy())
    signal   = int(df_out["signal"].iloc[-1])
    composite = float(df_out["_composite"].iloc[-1])
    return signal, composite


def determine_action(
    symbol:           str,
    desired_signal:   int,
    current_position: int,
) -> str:
    """
    Determine what trading action to take based on signal vs held position.
    -----------------------------------------------------------------------
    What it does
        Compares the strategy's desired position (1=long, 0=flat) against
        the current position held in the IBKR account, and returns an
        action string: "BUY", "SELL", or "HOLD".

        Logic:
            Desired=1, Held=0  →  BUY  (open new long)
            Desired=0, Held>0  →  SELL (close existing long)
            Desired=1, Held>0  →  HOLD (already positioned correctly)
            Desired=0, Held=0  →  HOLD (correctly flat)

    Used by
        Called for every symbol in the main loop. The action is then
        passed to ``place_order()`` and logged.

    When you need this
        - Review the action log if the script is placing unexpected orders.
          The action is determined purely from signal vs position; there is
          no partial sizing — positions are always fully opened or fully
          closed in one order.
        - To add partial position management (e.g., scale in over multiple
          days), replace this logic with a custom action function.

    Code example
        >>> action = determine_action("AAPL", desired_signal=1, current_position=0)
        >>> print(action)  # "BUY"

    Args:
        symbol           (str): Ticker (for logging only).
        desired_signal   (int): Strategy output: 1=long, 0=flat.
        current_position (int): Shares currently held (0 = flat).

    Returns:
        str: "BUY", "SELL", or "HOLD".
    """
    if desired_signal == 1 and current_position == 0:
        return "BUY"
    elif desired_signal == 0 and current_position > 0:
        return "SELL"
    else:
        return "HOLD"


# ============================================================
# 4. POSITION SIZING
# ============================================================

def calculate_position_size(
    nlv:      float,
    price:    float,
    risk_pct: float = 0.01,
) -> int:
    """
    Calculate the number of shares to buy, capped at risk_pct of NLV.
    -------------------------------------------------------------------
    What it computes
        Converts a dollar risk limit into a share count:
            dollar_limit = NLV × risk_pct
            shares       = floor(dollar_limit / price)
        Then further caps at MAX_SHARES (global constant = 1000).

    Research basis
        Fixed-fractional position sizing (Van Tharp, "Trade Your Way to
        Financial Freedom") with risk_pct = 0.01 (1%). This keeps each
        trade's maximum loss bounded: even a 100% loss on a single position
        only costs 1% of the account. Strongly recommended to keep this at
        or below 2% for momentum strategies (which have moderate win rates).

    Used by
        Called for BUY actions in ``main()`` immediately before calling
        ``place_order()``. For SELL actions, the quantity comes from the
        existing position, not from this function.

    When you need this
        - To take smaller positions (more conservative), reduce risk_pct
          via --risk-pct 0.005 (0.5%).
        - To take larger positions (more aggressive), increase to 0.02,
          but note this multiplies drawdowns accordingly.
        - If NLV fetch fails and nlv=0.0, this returns 0 shares and no
          order is placed — a safe default.

    Code example
        >>> qty = calculate_position_size(nlv=100_000, price=185.0, risk_pct=0.01)
        >>> print(f"Buy {qty} shares at ${185.0:.2f} = ${qty*185.0:.0f}")

    Args:
        nlv      (float): Account net liquidation value in USD.
        price    (float): Current market price of the stock.
        risk_pct (float): Fraction of NLV to risk (default 0.01 = 1%).

    Returns:
        int: Number of shares to buy (0 if price or NLV is zero).
    """
    if price <= 0 or nlv <= 0:
        return 0
    dollar_limit = nlv * risk_pct
    shares = int(dollar_limit / price)
    return min(shares, MAX_SHARES)


# ============================================================
# 5. ORDER PLACEMENT
# ============================================================

def place_order(
    app:     myIBApp,
    symbol:  str,
    action:  str,
    qty:     int,
    dry_run: bool = False,
) -> Optional[int]:
    """
    Submit a market order to IBKR and return the assigned order ID.
    ----------------------------------------------------------------
    What it does
        Builds an IBKR stock contract (SMART routing, USD), creates a
        market order, and calls ``placeOrder``. In dry-run mode it logs
        the intended order without sending it to IBKR. The function waits
        1 second after submission to allow TWS to process the order.

    Used by
        Called in the main loop for BUY and SELL actions after confirming
        the order quantity is > 0.

    When you need this
        - Always run in dry-run mode (--dry-run) first when testing
          a new symbol list or strategy configuration.
        - If you want limit orders instead of market orders, replace
          ``make_market_order()`` with ``make_stop_limit_order()`` from
          myIBApp.py.
        - To add bracket orders (entry + stop-loss + take-profit), modify
          this function to call placeOrder three times with linked IDs.

    IMPORTANT: This function places real orders when dry_run=False.
    On a live account (--live), market orders execute immediately at the
    best available price. Double-check the symbol list and risk parameters
    before disabling dry_run.

    Code example
        >>> order_id = place_order(app, "AAPL", "BUY", 10, dry_run=True)
        >>> # [DRY RUN] Would BUY 10 AAPL  (no order sent)

    Args:
        app     (myIBApp): Connected IBKR app.
        symbol  (str):     Ticker symbol.
        action  (str):     "BUY" or "SELL".
        qty     (int):     Number of shares.
        dry_run (bool):    If True, log but do not submit to IBKR.

    Returns:
        Optional[int]: IBKR order ID, or -1 for dry-run, None on failure.
    """
    log = logging.getLogger("daily_signal")

    if qty <= 0:
        log.warning(f"{symbol}: Order quantity is 0 — skipping.")
        return None

    if dry_run:
        log.info(f"[DRY RUN] Would {action} {qty} {symbol}")
        return -1

    try:
        contract = app.make_stock_contract(symbol)
        order    = app.make_market_order(action, qty)
        app.next_id += 1
        order_id = app.next_id
        app.placeOrder(order_id, contract, order)
        log.info(f"Order submitted: {action} {qty} {symbol}  orderId={order_id}")
        time.sleep(1)
        return order_id
    except Exception as exc:
        log.error(f"{symbol}: Order failed: {exc}")
        return None


# ============================================================
# 6. LOGGING
# ============================================================

def log_decision(
    today:     str,
    symbol:    str,
    signal:    int,
    composite: float,
    action:    str,
    qty:       int,
    price:     float,
    nlv:       float,
    order_id:  Optional[int],
    risk_pct:  float,
    params:    dict,
) -> None:
    """
    Append one row to the daily decisions CSV log file.
    -----------------------------------------------------
    What it logs
        Every signal evaluation — whether or not a trade was placed —
        is recorded with: timestamp, symbol, composite score, signal,
        action, quantity, estimated price, NLV, risk percent, order ID,
        and a SHA-256 hash of the parameters (so you can trace which
        version of the strategy produced which decisions).

        The CSV structure is designed for ML training: you can later join
        decisions with fill prices and compute trade P&L to label each
        row as a successful or failed signal, then train a classifier to
        identify which market conditions make the composite score most
        predictive.

    Used by
        Called for every symbol processed in the main loop, regardless
        of whether a trade was placed. The "HOLD" rows are as valuable
        as the trade rows for training.

    When you need this
        - After several weeks of paper trading, load
          ``logs/decisions_YYYY-MM-DD.csv`` and join with actual price
          data to evaluate the strategy's signal quality.
        - Use the ``params_hash`` column to identify which optimizer run
          produced which set of signals.
        - The file is created with a header on the first row of each day;
          subsequent rows append.

    Code example
        >>> log_decision(today="2026-07-19", symbol="AAPL", signal=1,
        ...              composite=0.72, action="BUY", qty=10, price=185.0,
        ...              nlv=100_000.0, order_id=42, risk_pct=0.01, params={...})

    Args:
        today     (str):         "YYYY-MM-DD" date string.
        symbol    (str):         Ticker.
        signal    (int):         Strategy signal (1=long, 0=flat).
        composite (float):       Composite score [0, 1].
        action    (str):         "BUY", "SELL", or "HOLD".
        qty       (int):         Shares ordered (0 for HOLD).
        price     (float):       Estimated fill price.
        nlv       (float):       Account NLV at time of decision.
        order_id  (Optional[int]): IBKR order ID (-1 dry-run, None failed).
        risk_pct  (float):       Configured risk fraction.
        params    (dict):        Strategy parameter dict.
    """
    import hashlib

    os.makedirs(LOG_DIR, exist_ok=True)
    path        = os.path.join(LOG_DIR, f"decisions_{today}.csv")
    file_exists = os.path.exists(path)

    params_hash = hashlib.sha256(
        json.dumps(params, sort_keys=True).encode()
    ).hexdigest()[:12]

    row = {
        "timestamp":   datetime.now().isoformat(),
        "symbol":      symbol,
        "signal":      signal,
        "composite":   f"{composite:.4f}",
        "action":      action,
        "qty":         qty,
        "price_est":   f"{price:.2f}",
        "dollar_risk": f"{qty * price:.2f}",
        "nlv":         f"{nlv:.2f}",
        "risk_pct":    f"{risk_pct:.4f}",
        "order_id":    order_id if order_id is not None else "",
        "params_hash": params_hash,
    }

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def log_trade(
    today:    str,
    symbol:   str,
    action:   str,
    qty:      int,
    price:    float,
    order_id: Optional[int],
    nlv:      float,
) -> None:
    """
    Append a row to the daily trades CSV (orders actually submitted).
    -----------------------------------------------------------------
    What it logs
        Only rows where an order was actually placed (BUY or SELL with
        order_id not None). Separate from the decisions log so you can
        quickly filter for filled orders without wading through HOLDs.

    Used by
        Called in main() immediately after a successful ``place_order()``
        call (non-None order_id).

    When you need this
        - Load ``logs/trades_YYYY-MM-DD.csv`` to review all orders placed
          on a specific day.
        - Combine with IBKR statement exports to verify fills and compute
          realised P&L.

    Args: same as log_decision (subset of fields).
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    path        = os.path.join(LOG_DIR, f"trades_{today}.csv")
    file_exists = os.path.exists(path)

    row = {
        "timestamp": datetime.now().isoformat(),
        "symbol":    symbol,
        "action":    action,
        "qty":       qty,
        "price_est": f"{price:.2f}",
        "dollar":    f"{qty * price:.2f}",
        "nlv":       f"{nlv:.2f}",
        "order_id":  order_id if order_id is not None else "",
    }

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ============================================================
# 7. MAIN
# ============================================================

def main() -> None:
    """
    Entry point for the daily signal runner.
    -----------------------------------------
    What it does
        Parses arguments, sets up logging, loads params, connects to IBKR,
        then iterates through every symbol: fetches OHLCV, computes signal,
        determines action, and places orders if needed. Logs everything and
        disconnects cleanly.

    Run daily at 9:45 AM ET (cron)
        45 9 * * 1-5 cd /path/to/rangerwizard && \\
            python3.11 daily_signal.py >> logs/cron.log 2>&1

    CLI arguments
        --symbols TICK ...    Custom symbol list
        --universe NAME       megacap | movers | volume | all
        --live                Use live account (port 7496). Default: paper (7497)
        --dry-run             Compute signals but do NOT place orders
        --risk-pct FLOAT      Fraction of NLV per trade (default 0.01 = 1%)
        --years INT           Years of OHLCV history to download (default 1)
        --params-file PATH    Path to optimizer JSON (default: alpha_composite_opt.json)
    """
    parser = argparse.ArgumentParser(description="Daily momentum signal runner")
    parser.add_argument("--symbols",     nargs="+", default=None)
    parser.add_argument("--universe",    default=None,
                        choices=["megacap", "movers", "volume", "all"])
    parser.add_argument("--live",        action="store_true",
                        help="Trade on live account (port 7496). Default: paper (7497).")
    parser.add_argument("--dry-run",     action="store_true",
                        help="Compute signals but do not submit orders.")
    parser.add_argument("--risk-pct",    type=float, default=0.01,
                        help="Fraction of NLV per trade (default 0.01).")
    parser.add_argument("--years",       type=int, default=1)
    parser.add_argument("--params-file", default=PARAMS_FILE)
    args = parser.parse_args()

    today = str(date.today())
    log   = setup_logging(today)

    log.info("=" * 60)
    log.info(f"Daily Signal Runner  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"Account : {'LIVE' if args.live else 'PAPER'}"
             f"  |  dry_run={args.dry_run}  |  risk={args.risk_pct:.2%}")
    log.info("=" * 60)

    # ── Resolve symbols ───────────────────────────────────────────────────────
    if args.universe:
        symbols = UNIVERSES[args.universe]
    elif args.symbols:
        symbols = args.symbols
    else:
        symbols = DEFAULT_SYMBOLS
    log.info(f"Symbols ({len(symbols)}): {symbols}")

    # ── Load params ───────────────────────────────────────────────────────────
    params = load_params(args.params_file)

    # ── Connect ───────────────────────────────────────────────────────────────
    port = PORT_LIVE if args.live else PORT_PAPER
    try:
        app = connect_ibkr(port)
    except ConnectionError as exc:
        log.error(f"Cannot connect to IBKR: {exc}")
        sys.exit(1)

    try:
        # ── Account state ─────────────────────────────────────────────────────
        nlv       = fetch_account_nlv(app)
        positions = fetch_positions(app)

        if nlv == 0.0:
            log.error("NLV fetch returned 0. Aborting to prevent bad position sizing.")
            return

        log.info(f"NLV=${nlv:,.2f}  Max per trade: ${nlv * args.risk_pct:,.2f}")

        # ── Symbol loop ───────────────────────────────────────────────────────
        for symbol in symbols:
            log.info(f"─── {symbol} {'─' * (40 - len(symbol))}")
            price     = 0.0
            signal    = 0
            composite = 0.0
            action    = "HOLD"
            qty       = 0
            order_id  = None

            try:
                # Fetch OHLCV
                duration = f"{args.years} Y"
                df       = fetch_ohlcv(app, symbol, duration=duration)

                # Compute signal
                signal, composite = compute_signal(df, params)
                log.info(f"{symbol}: composite={composite:.3f}  signal={signal}"
                         f"  ({'LONG' if signal == 1 else 'FLAT'})")

                # Use last historical close for sizing — avoids requiring a
                # real-time market data subscription (error 10089).
                # The market order still executes at the live price; we only
                # use this for share-count calculation.
                last_close = float(df["Close"].iloc[-1])
                log.debug(f"{symbol}: last close ${last_close:.2f} (used for sizing)")

                # Determine action
                current_qty = positions.get(symbol, (0, 0.0))[0]
                action      = determine_action(symbol, signal, current_qty)
                log.info(f"{symbol}: held={current_qty}  action={action}")

                if action == "BUY":
                    price = last_close
                    qty   = calculate_position_size(nlv, price, args.risk_pct)
                    log.info(f"{symbol}: BUY {qty} shares @ ~${price:.2f} "
                             f"= ${qty * price:,.0f}")
                    order_id = place_order(app, symbol, "BUY", qty,
                                           dry_run=args.dry_run)
                    if order_id is not None:
                        log_trade(today, symbol, "BUY", qty, price, order_id, nlv)

                elif action == "SELL":
                    price     = last_close
                    max_qty   = calculate_position_size(nlv, price, args.risk_pct)
                    qty       = min(current_qty, max_qty)
                    log.info(f"{symbol}: SELL {qty} of {current_qty} shares @ ~${price:.2f} "
                             f"(capped at 1% NLV = {max_qty} shares)")
                    order_id = place_order(app, symbol, "SELL", qty,
                                           dry_run=args.dry_run)
                    if order_id is not None:
                        log_trade(today, symbol, "SELL", qty, price, order_id, nlv)

                else:
                    log.info(f"{symbol}: HOLD — no action needed.")

            except Exception as exc:
                log.error(f"{symbol}: Error — {exc}")
                action = "ERROR"

            finally:
                log_decision(
                    today=today, symbol=symbol, signal=signal,
                    composite=composite, action=action, qty=qty,
                    price=price, nlv=nlv, order_id=order_id,
                    risk_pct=args.risk_pct, params=params,
                )

    finally:
        app.disconnect()
        log.info("Disconnected from IBKR.")
        log.info(f"Logs → {LOG_DIR}/")
        log.info("=" * 60)


if __name__ == "__main__":
    main()
