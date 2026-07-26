"""
run_backtest_example.py
=======================
Standalone script that runs a complete historical backtest of the
bundled MACDMomentumStrategy and prints a formatted report with ASCII
charts directly in the terminal — no matplotlib or external plotting
libraries required.

What this script does
---------------------
  1. Load historical OHLCV price data (yfinance → IBKR → synthetic fallback).
  2. Run the MACDMomentumStrategy backtest via ``run_backtest()``.
  3. Print a multi-section ASCII report:
       • Performance metrics table
       • Equity curve line chart
       • Drawdown area chart
       • Monthly returns heatmap table
       • Trade return distribution histogram
       • Most recent trades list

Run from the repo root
----------------------
    # Quickest — uses yfinance (pip install yfinance) or synthetic data:
    python momentum/test/run_backtest_example.py

    # Override symbol and lookback:
    python momentum/test/run_backtest_example.py --symbol MSFT --years 3

    # Force yfinance data source:
    python momentum/test/run_backtest_example.py --symbol AAPL --source yfinance

    # Use a live IBKR TWS connection for data:
    python momentum/test/run_backtest_example.py --source ibkr --symbol AAPL

    # Use your own CSV file (must have Date,Open,High,Low,Close,Volume columns):
    python momentum/test/run_backtest_example.py --source csv --csv-path data/AAPL.csv

    # Test a different initial capital:
    python momentum/test/run_backtest_example.py --capital 50000

Dependencies
------------
    Required : numpy, pandas
    Optional : yfinance  (pip install yfinance)  ← recommended data source
    Optional : ibapi     (installed with twsapi_macunix)  ← for IBKR source
"""

import argparse
import os
import sys
import math
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# ── repo-root import path ───────────────────────────────────────────────────
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from strategies.test.strategy_tester import (
    MACDMomentumStrategy,
    BacktestConfig,
    BacktestResult,
    Trade,
    run_backtest,
)

# ── terminal width used by all ASCII charts ─────────────────────────────────
_TERM_WIDTH = 72


# ============================================================
# 1. DATA LOADING
# ============================================================

def load_price_data(
    symbol: str = "AAPL",
    years: int = 3,
    source: str = "auto",
    csv_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, str]:
    """
    Load historical OHLCV price data from the best available source.
    -----------------------------------------------------------------
    What it does
        Tries data sources in priority order until one succeeds:
          1. yfinance  (free, no account needed)
          2. IBKR TWS  (requires active connection on port 7496/7497)
          3. CSV file  (if ``--csv-path`` is supplied)
          4. Synthetic GBM data  (always works; realistic but not real)
        Returns a clean DataFrame and a string label for the report.

    Used by
        Called at the top of ``main()`` to get the price history that
        will be passed to ``run_backtest()``.

    When you need this
        - When running the script standalone without IBKR connected.
        - As a template for loading data in your own backtest scripts.
        - If you want to test against a specific CSV export from your
          brokerage or data provider, pass ``--source csv --csv-path FILE``.

    How to use (code)
        >>> df, label = load_price_data("AAPL", years=3, source="yfinance")
        >>> print(df.tail())

    How to use (CLI)
        python run_backtest_example.py --symbol MSFT --years 2 --source yfinance
        python run_backtest_example.py --source csv --csv-path mydata.csv

    Args:
        symbol   (str): Ticker symbol (e.g. "AAPL").
        years    (int): How many years of history to load (default 3).
        source   (str): "auto" | "yfinance" | "ibkr" | "csv" | "synthetic".
        csv_path (str): Path to a CSV file when source="csv".

    Returns:
        Tuple[pd.DataFrame, str]:
            df    – OHLCV DataFrame with DatetimeIndex.
            label – Short string describing the source (for the report header).
    """
    attempts = []
    if source == "auto":
        attempts = ["yfinance", "ibkr", "synthetic"]
    elif source == "csv":
        attempts = ["csv"]
    else:
        attempts = [source, "synthetic"]

    for attempt in attempts:
        try:
            if attempt == "yfinance":
                df, label = _load_yfinance(symbol, years)
                return df, label
            elif attempt == "ibkr":
                df, label = _load_ibkr(symbol, years)
                return df, label
            elif attempt == "csv":
                df, label = _load_csv(csv_path or "", symbol)
                return df, label
            elif attempt == "synthetic":
                df, label = _load_synthetic(symbol, years)
                return df, label
        except Exception as exc:
            print(f"  [{attempt}] failed: {exc}")

    raise RuntimeError("All data sources failed. Cannot load price data.")


def _load_yfinance(symbol: str, years: int) -> Tuple[pd.DataFrame, str]:
    """
    Download historical OHLCV data from Yahoo Finance via yfinance.
    ----------------------------------------------------------------
    What it fetches
        Adjusted daily OHLCV bars for a stock ticker over the past
        ``years`` years. Prices are split- and dividend-adjusted.

    Used by
        ``load_price_data()`` when source is "yfinance" or "auto".
        yfinance is the default free data source recommended for users
        without a Bloomberg/Refinitiv subscription.

    When you need this
        - Routine backtesting without needing a live TWS connection.
        - Quickly testing a strategy on any publicly listed stock.
        - Getting data for ETFs, indices, or international stocks.

    How to use (code)
        >>> df, label = _load_yfinance("AAPL", years=3)
        >>> df.info()

    How to install yfinance
        pip install yfinance

    Args:
        symbol (str): Yahoo Finance ticker (e.g. "AAPL", "SPY", "^GSPC").
        years  (int): How many years of history to download.

    Returns:
        Tuple[pd.DataFrame, str]: (OHLCV DataFrame, "yfinance") label.

    Raises:
        ImportError: If yfinance is not installed.
        ValueError:  If the symbol is not found or no data is returned.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError(
            "yfinance is not installed. Run: pip install yfinance"
        )

    print(f"  Downloading {symbol} ({years}y) from Yahoo Finance ...", end=" ", flush=True)
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=f"{years}y", auto_adjust=True)

    if df.empty:
        raise ValueError(f"No data returned for '{symbol}' from Yahoo Finance.")

    # Standardise column names
    df = df.rename(columns={"Open": "Open", "High": "High",
                             "Low": "Low", "Close": "Close",
                             "Volume": "Volume"})
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    print(f"done. ({len(df)} bars)")
    return df, "yfinance"


def _load_ibkr(symbol: str, years: int) -> Tuple[pd.DataFrame, str]:
    """
    Fetch historical OHLCV data from a live IBKR TWS connection.
    -------------------------------------------------------------
    What it fetches
        Daily OHLCV bars for ``symbol`` from IBKR's historical data feed.
        Data matches exactly what the TWS platform displays — same splits,
        dividends, and closing prices your live strategy will trade on.

    Used by
        ``load_price_data()`` when source is "ibkr" or "auto" (after
        yfinance fails). Requires TWS or IB Gateway to be running.

    When you need this
        - When you want the most accurate representation of what your
          live momentum strategy will experience (no data discrepancy).
        - When testing instruments that yfinance doesn't cover well
          (some futures, OTC stocks, certain ETFs).
        - For intraday backtesting — change bar_size to "5 mins" etc.

    How to use (code)
        >>> df, label = _load_ibkr("AAPL", years=2)

    How to use (CLI)
        python run_backtest_example.py --source ibkr --symbol AAPL

    Note
        TWS or IB Gateway must be running. Default port is 7497 (paper).
        Change to 7496 in myIBApp.py for live data.

    Args:
        symbol (str): IBKR ticker symbol.
        years  (int): Years of history to fetch.

    Returns:
        Tuple[pd.DataFrame, str]: (OHLCV DataFrame, "IBKR") label.

    Raises:
        RuntimeError: If TWS is not running or the symbol is not found.
    """
    from myIBApp import connect_to_tws
    from strategies.test.strategy_tester import fetch_ibkr_history

    print(f"  Connecting to IBKR TWS for {symbol} ({years}y) ...", end=" ", flush=True)
    app = connect_to_tws()
    duration = f"{years} Y"
    df = fetch_ibkr_history(app, symbol, duration=duration, bar_size="1 day")
    app.disconnect()
    print(f"done. ({len(df)} bars)")
    return df, "IBKR"


def _load_csv(path: str, symbol: str) -> Tuple[pd.DataFrame, str]:
    """
    Load OHLCV data from a local CSV file.
    ---------------------------------------
    What it fetches
        Daily OHLCV price data from a CSV file you provide. The file
        must have at minimum the columns: Date (or index), Open, High,
        Low, Close, Volume.

    Used by
        ``load_price_data()`` when source is "csv". Use this when you
        have exported data from your broker, data vendor, or a previous
        IBKR download.

    When you need this
        - When working offline without Internet or TWS access.
        - When using a proprietary data source (Bloomberg export, Quandl
          CSV, broker statement) for the backtest.
        - For reproducibility: save a CSV once and always test against
          the same data, even if live prices have changed.

    How to use (code)
        >>> df, label = _load_csv("data/AAPL_3y.csv", "AAPL")

    How to use (CLI)
        python run_backtest_example.py --source csv --csv-path data/AAPL.csv

    CSV format required
        Date,Open,High,Low,Close,Volume
        2023-01-03,130.28,130.90,124.17,125.07,112117500
        ...

    Args:
        path   (str): File path to the CSV.
        symbol (str): Ticker label for the report.

    Returns:
        Tuple[pd.DataFrame, str]: (OHLCV DataFrame, "CSV") label.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError:        If required columns are missing.
    """
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found: '{path}'")

    print(f"  Loading CSV: {path} ...", end=" ", flush=True)
    df = pd.read_csv(path, parse_dates=True, index_col=0)
    df.index = pd.to_datetime(df.index)

    required = {"Open", "High", "Low", "Close", "Volume"}
    # Try case-insensitive match
    df.columns = [c.capitalize() if c.lower() in {r.lower() for r in required}
                  else c for c in df.columns]
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    df = df[list(required)].dropna().sort_index()
    print(f"done. ({len(df)} bars)")
    return df, f"CSV:{os.path.basename(path)}"


def _load_synthetic(symbol: str, years: int) -> Tuple[pd.DataFrame, str]:
    """
    Generate realistic synthetic OHLCV data using Geometric Brownian Motion.
    -------------------------------------------------------------------------
    What it generates
        Simulates a stock price path using GBM — the same stochastic
        process underlying the Black-Scholes model. Produces Open, High,
        Low, Close, Volume data that is statistically plausible (but
        entirely fabricated). Useful when no real data source is available.

    Used by
        ``load_price_data()`` as the last-resort fallback when all real
        data sources fail. Always works — requires no network or accounts.
        Also useful for stress-testing strategy code without needing live
        data access.

    When you need this
        - Running the script for the first time without any data source
          configured (instant demonstration mode).
        - Unit testing strategy logic with deterministic (seeded) data.
        - Stress-testing with custom volatility/drift parameters.

    How to use (code)
        >>> df, label = _load_synthetic("FAKE", years=2)

    Note
        Results are clearly labelled "SYNTHETIC" in the report header
        to prevent confusing synthetic results with real backtests.

    Args:
        symbol (str): Label for the synthetic asset.
        years  (int): Number of years of daily bars to generate.

    Returns:
        Tuple[pd.DataFrame, str]: (OHLCV DataFrame, "SYNTHETIC") label.
    """
    print(f"  Generating {years}y synthetic OHLCV for {symbol} ...", end=" ", flush=True)
    np.random.seed(42)
    n_days = years * 252
    dates = pd.bdate_range(
        end=datetime.today().strftime("%Y-%m-%d"), periods=n_days
    )
    # GBM parameters: moderate drift, ~25% annual vol
    mu    = 0.08 / 252
    sigma = 0.25 / math.sqrt(252)
    daily_returns = np.exp(
        (mu - 0.5 * sigma ** 2) + sigma * np.random.randn(n_days)
    )
    close = 150.0 * np.cumprod(daily_returns)

    intraday_vol = sigma * 0.6
    high  = close * np.exp(np.abs(intraday_vol * np.random.randn(n_days)))
    low   = close * np.exp(-np.abs(intraday_vol * np.random.randn(n_days)))
    open_ = close * np.exp(intraday_vol * 0.3 * np.random.randn(n_days))
    # Ensure OHLC relationships are valid
    high  = np.maximum(high,  np.maximum(open_, close))
    low   = np.minimum(low,   np.minimum(open_, close))

    volume = (np.random.lognormal(mean=15, sigma=0.5, size=n_days)).astype(int)

    df = pd.DataFrame({
        "Open":   open_,
        "High":   high,
        "Low":    low,
        "Close":  close,
        "Volume": volume,
    }, index=dates)
    print(f"done. ({len(df)} bars, GBM μ={mu*252:.1%}/yr σ={sigma*math.sqrt(252):.1%}/yr)")
    return df, "SYNTHETIC"


# ============================================================
# 2. ASCII CHART ENGINE
# ============================================================

def ascii_line_chart(
    series: pd.Series,
    title: str,
    width: int = _TERM_WIDTH,
    height: int = 10,
    y_fmt=lambda v: f"${v:>10,.0f}",
) -> str:
    """
    Render a time-series line chart as ASCII art.
    ----------------------------------------------
    What it renders
        A grid of ``width × height`` characters where each column
        represents a time period and the position of the ``▪`` character
        in each column represents the scaled value. Y-axis labels appear
        on the left; date labels appear at the bottom.

    Used by
        ``print_equity_chart()`` and ``print_drawdown_chart()`` to
        produce the main equity curve and drawdown visualisations.
        No external libraries required — pure Python.

    When you need this
        - Any time you want a quick visual of a time series in the
          terminal without opening a browser or notebook.
        - In scheduled paper-trader logs (cron jobs) where only text
          output is available.
        - When sharing strategy results via email or Slack (plain text).

    How to use (code)
        >>> chart = ascii_line_chart(
        ...     equity_series,
        ...     title="Portfolio Equity",
        ...     width=70, height=10,
        ... )
        >>> print(chart)

    Args:
        series (pd.Series): Time series with DatetimeIndex.
        title  (str):       Title displayed above the chart.
        width  (int):       Number of character columns (default 72).
        height (int):       Number of character rows (default 10).
        y_fmt  (callable):  Format function for Y-axis labels.

    Returns:
        str: Multi-line ASCII chart string ready for ``print()``.
    """
    values = series.dropna().values.astype(float)
    dates  = series.dropna().index

    if len(values) < 2:
        return f"  {title}\n  (insufficient data)\n"

    y_min, y_max = values.min(), values.max()
    y_range = y_max - y_min if y_max != y_min else 1.0
    label_w = len(y_fmt(y_max))

    # Build character grid
    grid = [[" "] * width for _ in range(height)]

    prev_row = None
    for col in range(width):
        idx = int(col * (len(values) - 1) / max(width - 1, 1))
        idx = min(idx, len(values) - 1)
        row = height - 1 - int((values[idx] - y_min) / y_range * (height - 1))
        row = max(0, min(height - 1, row))
        grid[row][col] = "▪"
        # Draw vertical connector between consecutive points
        if prev_row is not None and abs(prev_row - row) > 1:
            lo, hi = (min(prev_row, row) + 1, max(prev_row, row))
            for r in range(lo, hi):
                grid[r][col] = "│"
        prev_row = row

    # Compose lines
    lines = [f"  {title}"]
    for r in range(height):
        if r == 0:
            label = y_fmt(y_max)
        elif r == height - 1:
            label = y_fmt(y_min)
        elif r == height // 2:
            label = y_fmt(y_min + y_range / 2)
        else:
            label = " " * label_w
        bar = "│" if r < height - 1 else "└"
        lines.append(f"  {label} {bar}{''.join(grid[r])}")

    # X-axis line + date labels
    lines.append(f"  {' ' * label_w}  {'─' * width}")
    if len(dates) > 0:
        d0 = str(dates[0])[:10]
        dm = str(dates[len(dates) // 2])[:10]
        d1 = str(dates[-1])[:10]
        gap = width - len(d0) - len(dm) - len(d1)
        l_gap = max(1, gap // 2)
        r_gap = max(1, gap - l_gap)
        lines.append(f"  {' ' * (label_w + 2)}{d0}{' ' * l_gap}{dm}{' ' * r_gap}{d1}")

    return "\n".join(lines)


def ascii_area_chart(
    series: pd.Series,
    title: str,
    width: int = _TERM_WIDTH,
    height: int = 8,
    fill_char: str = "█",
    y_fmt=lambda v: f"{v:>7.1%}",
) -> str:
    """
    Render a downward-filled area chart (for drawdowns) as ASCII art.
    ------------------------------------------------------------------
    What it renders
        An inverted filled chart where ``fill_char`` characters fill
        from the zero line downward to the current value. Ideal for
        visualising drawdowns: deeper fills = larger drawdowns.

    Used by
        ``print_drawdown_chart()`` to show the drawdown series from the
        backtest result. The zero line is at the top; deeper drawdowns
        extend further down.

    When you need this
        - Inspecting the severity and duration of drawdown periods
          without a graphical tool.
        - Quickly assessing whether drawdowns cluster (regime-dependent
          strategy) or are spread evenly (market-independent strategy).
        - Including in a cron log file to monitor live paper trading
          drawdown over time.

    How to use (code)
        >>> dd = (equity / equity.cummax() - 1)
        >>> chart = ascii_area_chart(dd, "Drawdown", width=70, height=8)
        >>> print(chart)

    Args:
        series    (pd.Series): Drawdown series (values ≤ 0).
        title     (str):       Title displayed above the chart.
        width     (int):       Character columns (default 72).
        height    (int):       Character rows (default 8).
        fill_char (str):       Fill character (default "█").
        y_fmt     (callable):  Format function for Y-axis labels.

    Returns:
        str: Multi-line ASCII chart string.
    """
    values = series.dropna().values.astype(float)
    dates  = series.dropna().index

    if len(values) < 2:
        return f"  {title}\n  (insufficient data)\n"

    y_min = min(values.min(), -0.001)
    y_max = 0.0
    y_range = y_max - y_min

    label_w = len(y_fmt(y_min))
    grid = [[" "] * width for _ in range(height)]

    for col in range(width):
        idx = int(col * (len(values) - 1) / max(width - 1, 1))
        idx = min(idx, len(values) - 1)
        v = values[idx]
        # Row 0 = top = zero line. Row height-1 = bottom = most negative.
        fill_rows = int((v - y_max) / (y_min - y_max) * height)
        fill_rows = max(0, min(height, fill_rows))
        for r in range(fill_rows):
            grid[r][col] = fill_char

    lines = [f"  {title}"]
    for r in range(height):
        if r == 0:
            label = y_fmt(y_max)
        elif r == height - 1:
            label = y_fmt(y_min)
        elif r == height // 2:
            label = y_fmt(y_min / 2)
        else:
            label = " " * label_w
        bar = "│" if r < height - 1 else "└"
        lines.append(f"  {label} {bar}{''.join(grid[r])}")

    lines.append(f"  {' ' * label_w}  {'─' * width}")
    if len(dates) > 0:
        d0 = str(dates[0])[:10]
        dm = str(dates[len(dates) // 2])[:10]
        d1 = str(dates[-1])[:10]
        gap = width - len(d0) - len(dm) - len(d1)
        l_gap = max(1, gap // 2)
        r_gap = max(1, gap - l_gap)
        lines.append(f"  {' ' * (label_w + 2)}{d0}{' ' * l_gap}{dm}{' ' * r_gap}{d1}")

    return "\n".join(lines)


def ascii_histogram(
    values: List[float],
    title: str,
    bins: int = 12,
    width: int = 38,
) -> str:
    """
    Render a horizontal bar histogram as ASCII art.
    -------------------------------------------------
    What it renders
        A sideways bar chart where each row is a return bin and the
        length of the ``▓`` bar shows how many trades fell in that range.
        A vertical centre line marks the zero-return boundary.

    Used by
        ``print_trade_histogram()`` to show the distribution of
        individual trade returns from the backtest's trade list.

    When you need this
        - Checking whether your strategy has a fat-tailed return
          distribution (a few big wins, many small losses) or a
          normal-ish distribution.
        - Identifying if losses and gains are symmetric or skewed.
        - A quick sanity check that the win/loss profile matches your
          strategy's theoretical behaviour.

    How to use (code)
        >>> trade_returns = [t.return_pct for t in result.trades]
        >>> chart = ascii_histogram(trade_returns, "Trade Returns", bins=10)
        >>> print(chart)

    Args:
        values (List[float]): Trade return percentages (decimal).
        title  (str):         Title displayed above the chart.
        bins   (int):         Number of histogram bins (default 12).
        width  (int):         Max bar length in characters (default 38).

    Returns:
        str: Multi-line ASCII histogram string.
    """
    if len(values) < 2:
        return f"  {title}\n  (not enough trades)\n"

    arr = np.array(values)
    counts, edges = np.histogram(arr, bins=bins)
    max_count = counts.max() if counts.max() > 0 else 1

    lines = [f"  {title}", f"  {'─' * (width + 20)}"]
    for i, count in enumerate(counts):
        lo = edges[i]
        hi = edges[i + 1]
        bar_len = int(count / max_count * width)
        colour = "▓" if lo >= 0 else "░"
        bar = colour * bar_len
        sign = "+" if lo >= 0 else " "
        lines.append(
            f"  {sign}{lo*100:5.1f}% → {hi*100:5.1f}%  │{bar:<{width}}  {count:3d}"
        )
    lines.append(f"  {'─' * (width + 20)}")
    lines.append(f"  ░ = losses  ▓ = gains   n = {len(values)} trades")
    return "\n".join(lines)


def ascii_monthly_table(daily_returns: pd.Series) -> str:
    """
    Render a monthly returns heatmap as an ASCII text table.
    ---------------------------------------------------------
    What it renders
        A grid of years × months showing each month's compounded return.
        Positive months are prefixed with "+" and negative with "-".
        The rightmost column shows the full-year return. Each cell is
        colour-coded in text: uppercase for large positive months,
        lowercase for large negative months.

    Used by
        ``print_monthly_table()`` in the backtest report to show
        seasonality patterns and consistency of returns across months.

    When you need this
        - Looking for seasonality: does your strategy consistently win
          in certain months and lose in others? (If yes, is it robust or
          data-mined?)
        - Identifying years where the strategy underperformed badly —
          is the drawdown concentrated in one year or spread out?
        - Comparing the monthly table against buy-and-hold to see where
          your strategy adds or loses value versus the index.

    How to use (code)
        >>> table = ascii_monthly_table(result.daily_returns)
        >>> print(table)

    Args:
        daily_returns (pd.Series): Daily return series with DatetimeIndex.

    Returns:
        str: Multi-line ASCII monthly returns table.
    """
    if daily_returns.empty or not hasattr(daily_returns.index, "month"):
        return "  Monthly Returns\n  (no data)\n"

    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    monthly = daily_returns.resample("ME").apply(lambda r: (1 + r).prod() - 1)
    monthly.index = monthly.index.to_period("M")
    years = sorted(monthly.index.year.unique())

    col_w = 7
    header = f"  {'Year':>6}  " + "  ".join(f"{m:>{col_w}}" for m in months) + f"  {'EOY':>{col_w}}"
    sep    = f"  {'─' * (8 + (col_w + 2) * 13)}"
    lines  = [f"  Monthly Returns (%)", sep, header, sep]

    for yr in years:
        yr_ret = monthly[monthly.index.year == yr]
        annual = (1 + yr_ret).prod() - 1
        cells = []
        for mo in range(1, 13):
            key = pd.Period(f"{yr}-{mo:02d}", freq="M")
            if key in yr_ret.index:
                v = yr_ret[key]
                sign = "+" if v >= 0 else "-"
                cell = f"{sign}{abs(v)*100:.1f}%"
            else:
                cell = "  n/a "
            cells.append(f"{cell:>{col_w}}")
        eoy_str = f"{'+'if annual>=0 else ''}{annual*100:.1f}%"
        lines.append(f"  {yr:>6}  {'  '.join(cells)}  {eoy_str:>{col_w}}")

    lines.append(sep)
    return "\n".join(lines)


def ascii_trade_table(trades: List[Trade], n: int = 15) -> str:
    """
    Render the most recent N trades as a formatted ASCII table.
    -----------------------------------------------------------
    What it renders
        A table with one row per trade showing entry/exit dates,
        direction, share count, entry price, exit price, and net P&L.
        The most recent trades are shown first.

    Used by
        ``print_recent_trades()`` at the bottom of the backtest report.
        Also useful for reviewing specific trades after noticing an
        unusual period in the equity curve.

    When you need this
        - Verifying that the backtest logic is generating sensible
          entries and exits (correct dates, realistic prices).
        - Identifying which specific trades caused the largest drawdowns.
        - Cross-referencing backtest fills against your paper trade log
          to check for live-vs-backtest discrepancies.

    How to use (code)
        >>> table = ascii_trade_table(result.trades, n=20)
        >>> print(table)

    Args:
        trades (List[Trade]): Trade list from BacktestResult.trades.
        n      (int):         Maximum number of trades to show (default 15).

    Returns:
        str: Formatted ASCII table string.
    """
    if not trades:
        return "  Recent Trades\n  (no trades completed)\n"

    shown = trades[-n:]
    hdr   = f"  {'Entry':>12}  {'Exit':>12}  {'Dir':>5}  {'Shrs':>5}  {'Entry $':>8}  {'Exit $':>8}  {'Net P&L':>10}  {'Ret%':>7}"
    sep   = f"  {'─' * (len(hdr) - 2)}"
    lines = [f"  Last {min(n, len(trades))} of {len(trades)} Trades", sep, hdr, sep]

    for t in reversed(shown):
        direction = "+LONG " if t.direction == 1 else "-SHORT"
        sign = "+" if t.net_pnl >= 0 else ""
        ret_sign = "+" if t.return_pct >= 0 else ""
        lines.append(
            f"  {t.entry_date:>12}  {t.exit_date:>12}  {direction:>5}  "
            f"{t.shares:>5}  ${t.entry_price:>7.2f}  ${t.exit_price:>7.2f}  "
            f"{sign}${t.net_pnl:>8.2f}  {ret_sign}{t.return_pct*100:>5.1f}%"
        )

    lines.append(sep)
    return "\n".join(lines)


# ============================================================
# 3. REPORT SECTIONS
# ============================================================

def print_report_header(result: BacktestResult, data_source: str) -> None:
    """
    Print the report title and period header.
    -------------------------------------------
    What it does
        Prints a formatted banner at the top of the report with the
        strategy name, symbol, test period, data source, and a brief
        note if the data was synthetic (so you know to treat results
        with caution).

    Used by
        ``print_full_report()`` as the first section. Also useful on
        its own when comparing multiple results in a loop.

    When you need this
        - Always printed first; identifies what you are looking at.
        - The "SYNTHETIC DATA" warning is important: if you see it,
          the numbers are not real performance — they're illustrative only.

    Args:
        result      (BacktestResult): Backtest output.
        data_source (str):            Label from ``load_price_data()``.
    """
    W = _TERM_WIDTH + 14
    print("\n" + "═" * W)
    title = f"  BACKTEST REPORT: {result.strategy_name} on {result.symbol}"
    print(title)
    period = f"  Period: {result.start_date}  →  {result.end_date}  |  Data: {data_source}"
    print(period)
    if "SYNTHETIC" in data_source.upper():
        print("  ⚠  SYNTHETIC DATA — results are illustrative only, not real performance")
    print("═" * W)


def print_metrics_box(result: BacktestResult) -> None:
    """
    Print the core performance metrics in a bordered box.
    ------------------------------------------------------
    What it does
        Renders all key backtest metrics (return, CAGR, Sharpe, Sortino,
        Calmar, drawdown, win rate, Kelly fraction) in a clean two-column
        bordered table.

    Used by
        ``print_full_report()`` as the second section, immediately after
        the header. This is the "at a glance" scorecard for the strategy.

    When you need this
        - The first thing to look at after running a backtest.
        - Use this to quickly decide: is the Sharpe ratio high enough?
          Is the max drawdown survivable? Is the CAGR above the benchmark?
        - Print this for multiple strategies to rank them before choosing
          one for paper trading.

    Interpreting the numbers
        Sharpe > 1.0   → acceptable; > 1.5 → good; > 2.0 → excellent
        Calmar > 0.5   → acceptable; > 1.0 → good
        Max DD < -30%  → most traders cannot stomach this psychologically
        Win Rate 45–65% is typical for momentum strategies
        Kelly Fraction: use 25–50% of this number for real sizing

    Args:
        result (BacktestResult): Backtest output.
    """
    W = _TERM_WIDTH + 14

    def row(label_l, val_l, label_r, val_r):
        return (f"│  {label_l:<26}{val_l:>14}   "
                f"{label_r:<26}{val_r:>14}  │")

    def fmt_pct(v):  return f"{v:.2%}" if not math.isnan(v) else "  n/a"
    def fmt_flt(v):  return f"{v:.2f}" if not math.isnan(v) else "  n/a"
    def fmt_dol(v):  return f"${v:>12,.2f}"

    print(f"\n┌{'─' * (W - 2)}┐")
    print(f"│{'  PERFORMANCE METRICS':^{W - 2}}│")
    print(f"├{'─' * (W - 2)}┤")
    print(row("Initial Capital",    fmt_dol(result.initial_capital),
              "Final Equity",       fmt_dol(result.final_equity)))
    print(row("Total Return",       fmt_pct(result.total_return),
              "CAGR",               fmt_pct(result.cagr_val)))
    print(row("Max Drawdown",       fmt_pct(result.max_dd),
              "Annualised Vol",     fmt_pct(result.vol)))
    print(row("Sharpe Ratio",       fmt_flt(result.sharpe),
              "Sortino Ratio",      fmt_flt(result.sortino)))
    print(row("Calmar Ratio",       fmt_flt(result.calmar),
              "Win Rate",           fmt_pct(result.win_rate_val)))
    print(row("Total Trades",       f"{result.total_trades:>14d}",
              "Kelly Fraction",     fmt_pct(result.kelly_fraction)))
    print(f"└{'─' * (W - 2)}┘")


def print_equity_chart(result: BacktestResult) -> None:
    """
    Print the ASCII equity curve chart.
    -------------------------------------
    What it does
        Renders the strategy's cumulative equity over time as an ASCII
        line chart using ``ascii_line_chart()``. The chart shows the
        portfolio value in dollars at each point in time.

    Used by
        ``print_full_report()`` as the third section. This is the visual
        representation of the equity curve.

    When you need this
        - Spotting whether gains are front-loaded (early in the test
          period) or consistent throughout — front-loading often
          indicates regime-dependency or lucky in-sample fitting.
        - Checking if the curve goes sideways for long periods (strategy
          not adapting to the market).
        - Quickly verifying that the dollar amounts in the chart match
          the metrics table above.

    Args:
        result (BacktestResult): Backtest output.
    """
    print(f"\n{'─' * (_TERM_WIDTH + 14)}")
    print(ascii_line_chart(
        result.equity_curve,
        title="Equity Curve",
        width=_TERM_WIDTH,
        height=10,
        y_fmt=lambda v: f"${v:>10,.0f}",
    ))


def print_drawdown_chart(result: BacktestResult) -> None:
    """
    Print the ASCII drawdown chart.
    --------------------------------
    What it does
        Computes the rolling drawdown series from the equity curve and
        renders it as a downward-filled ASCII area chart using
        ``ascii_area_chart()``. Deeper fills = larger drawdowns.

    Used by
        ``print_full_report()`` as the fourth section, directly after
        the equity curve. Together they give a complete picture of
        returns and risk.

    When you need this
        - Finding the maximum drawdown period visually — is it a short
          spike or a long, grinding decline?
        - Checking whether drawdowns recover quickly (momentum market)
          or linger (mean-reverting regime, which hurts momentum).
        - Comparing the drawdown chart against the equity chart to see
          which bull runs are accompanied by elevated volatility.

    Args:
        result (BacktestResult): Backtest output.
    """
    dd = (result.equity_curve / result.equity_curve.cummax()) - 1
    print(f"\n{'─' * (_TERM_WIDTH + 14)}")
    print(ascii_area_chart(
        dd,
        title="Drawdown",
        width=_TERM_WIDTH,
        height=7,
        fill_char="█",
        y_fmt=lambda v: f"{v:>7.1%}",
    ))


def print_monthly_table(result: BacktestResult) -> None:
    """
    Print the ASCII monthly returns heatmap table.
    -----------------------------------------------
    What it does
        Resamples the daily return series to monthly, then renders a
        year × month grid via ``ascii_monthly_table()``. Positive months
        display a "+" prefix; negative months display a "-" prefix.
        An end-of-year column shows the annual compounded return.

    Used by
        ``print_full_report()`` as the fifth section.

    When you need this
        - Identifying seasonal patterns (e.g., "the strategy always
          loses in September" — is this a real pattern or coincidence?).
        - Checking year-over-year consistency: is the strategy profitable
          most years, or just a few great years hiding many bad ones?
        - If a year shows large losses throughout, it likely represents
          a regime the strategy was not designed for — consider adding
          a market-regime filter.

    Args:
        result (BacktestResult): Backtest output.
    """
    print(f"\n{'─' * (_TERM_WIDTH + 14)}")
    print(ascii_monthly_table(result.daily_returns))


def print_trade_histogram(result: BacktestResult) -> None:
    """
    Print the ASCII trade return distribution histogram.
    -----------------------------------------------------
    What it does
        Extracts the per-trade ``return_pct`` from each ``Trade`` in
        ``result.trades`` and renders a horizontal ASCII histogram via
        ``ascii_histogram()``. Shows how trade returns are distributed.

    Used by
        ``print_full_report()`` as the sixth section.

    When you need this
        - Checking the shape of the return distribution:
            • Right-skewed (more big wins than big losses) → ideal for
              trend-following / momentum strategies.
            • Left-skewed (big losses, small wins) → typical of
              mean-reversion or option-selling strategies. Dangerous.
            • Normal / symmetric → typical of random noise. Strategy
              may not have real edge.
        - Spotting outlier trades: if one or two huge wins account for
          all the CAGR, the strategy may not be robust.

    Args:
        result (BacktestResult): Backtest output.
    """
    if not result.trades:
        print("\n  Trade Distribution\n  (no trades)\n")
        return
    trade_rets = [t.return_pct for t in result.trades]
    print(f"\n{'─' * (_TERM_WIDTH + 14)}")
    print(ascii_histogram(trade_rets, "Trade Return Distribution", bins=12, width=40))


def print_recent_trades(result: BacktestResult) -> None:
    """
    Print a table of the most recent completed trades.
    ---------------------------------------------------
    What it does
        Displays the last 15 trades from ``result.trades`` in reverse
        chronological order, showing entry/exit dates, direction, share
        count, fill prices, net P&L, and return percentage.

    Used by
        ``print_full_report()`` as the final section.

    When you need this
        - Verifying that the backtest logic is entering and exiting at
          the correct prices and on the correct dates.
        - Identifying the trades responsible for large P&L swings — look
          at the entry date and check the chart at that time.
        - Cross-referencing against a paper trading log to see if the
          live paper trades matched the historical backtest trades.

    Args:
        result (BacktestResult): Backtest output.
    """
    print(f"\n{'─' * (_TERM_WIDTH + 14)}")
    print(ascii_trade_table(result.trades, n=15))


# ============================================================
# 4. FULL REPORT ORCHESTRATOR
# ============================================================

def print_full_report(result: BacktestResult, data_source: str) -> None:
    """
    Print the complete backtest report with all ASCII charts.
    ----------------------------------------------------------
    What it does
        Calls every report section in order:
          1. Header banner
          2. Performance metrics box
          3. Equity curve chart
          4. Drawdown chart
          5. Monthly returns table
          6. Trade return distribution histogram
          7. Recent trades table
          8. Quick interpretation guide

    Used by
        ``main()`` after ``run_backtest()`` completes. This is the
        single call that produces the entire terminal output.

    When you need this
        - Call this after any ``run_backtest()`` result to get the
          full picture in one shot.
        - You can also call individual sections (``print_metrics_box``,
          ``print_equity_chart``, etc.) if you only need part of the
          report in a larger script.

    How to use (code)
        >>> result = run_backtest(MyStrategy(), df, cfg, symbol="AAPL")
        >>> print_full_report(result, data_source="yfinance")

    Args:
        result      (BacktestResult): Output from run_backtest().
        data_source (str):            Data source label from load_price_data().
    """
    print_report_header(result, data_source)
    print_metrics_box(result)
    print_equity_chart(result)
    print_drawdown_chart(result)
    print_monthly_table(result)
    print_trade_histogram(result)
    print_recent_trades(result)
    _print_guide()


def _print_guide() -> None:
    """
    Print a quick interpretation guide at the bottom of the report.
    ----------------------------------------------------------------
    What it does
        Prints a brief legend/reference showing what the ASCII chart
        symbols mean and how to interpret the key metrics. Helps new
        users read the report without referring to documentation.

    Used by
        ``print_full_report()`` as the final section.
    """
    W = _TERM_WIDTH + 14
    print(f"\n{'─' * W}")
    print("  HOW TO READ THIS REPORT")
    print(f"  {'─' * (W - 4)}")
    print("  Equity Curve  ▪ = portfolio value   │ = vertical connector")
    print("  Drawdown      █ = capital below peak (deeper fill = worse drawdown)")
    print("  Histogram     ░ = losing trades      ▓ = winning trades")
    print(f"  {'─' * (W - 4)}")
    print("  Sharpe > 1.0 acceptable  |  > 1.5 good  |  > 2.0 excellent")
    print("  Calmar > 0.5 acceptable  |  > 1.0 good")
    print("  Win Rate: 45–65% typical for momentum  |  Kelly: use 25% of shown value")
    print(f"{'─' * W}\n")


# ============================================================
# 5. MAIN ENTRY POINT
# ============================================================

def main() -> None:
    """
    Command-line entry point: load data, run backtest, print ASCII report.
    -----------------------------------------------------------------------
    What it does
        1. Parses CLI arguments (symbol, years, source, capital, etc.).
        2. Loads historical OHLCV data via ``load_price_data()``.
        3. Creates a ``BacktestConfig`` with the requested parameters.
        4. Runs ``run_backtest(MACDMomentumStrategy(), df, cfg)``.
        5. Calls ``print_full_report()`` to render the full ASCII output.

    Used by
        Called automatically when you run this file as a script:
            ``python momentum/test/run_backtest_example.py``
        Also importable for use in notebooks or larger scripts:
            ``from strategies.test.run_backtest_example import main; main()``

    CLI Arguments
        --symbol TICKER    Stock ticker to test (default: AAPL)
        --years N          Years of history (default: 3)
        --source SOURCE    Data source: auto|yfinance|ibkr|csv|synthetic
        --csv-path PATH    CSV file path (required if --source csv)
        --capital AMOUNT   Starting capital in dollars (default: 100000)
        --sizing METHOD    Position sizing: fixed|atr|kelly (default: fixed)
        --no-short         Disable short selling (default behaviour is no shorts)
        --allow-short      Allow short selling (strategy must emit -1 signals)

    Examples
        python momentum/test/run_backtest_example.py
        python momentum/test/run_backtest_example.py --symbol SPY --years 5
        python momentum/test/run_backtest_example.py --capital 50000 --sizing atr
        python momentum/test/run_backtest_example.py --source ibkr --symbol TSLA
        python momentum/test/run_backtest_example.py --source csv --csv-path data/AAPL.csv
    """
    parser = argparse.ArgumentParser(
        description="Run the MACDMomentumStrategy backtest and print an ASCII report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--symbol",      default="AAPL",     help="Ticker symbol (default: AAPL)")
    parser.add_argument("--years",       default=3, type=int, help="Years of history (default: 3)")
    parser.add_argument("--source",      default="auto",
                        choices=["auto", "yfinance", "ibkr", "csv", "synthetic"],
                        help="Data source (default: auto → yfinance → synthetic)")
    parser.add_argument("--csv-path",    default=None, help="CSV file path (for --source csv)")
    parser.add_argument("--capital",     default=100_000, type=float,
                        help="Starting capital in $ (default: 100000)")
    parser.add_argument("--sizing",      default="fixed",
                        choices=["fixed", "atr", "kelly"],
                        help="Position sizing method (default: fixed)")
    parser.add_argument("--allow-short", action="store_true",
                        help="Allow short positions (signal=-1)")
    args = parser.parse_args()

    print(f"\n  RangerwizardBacktest  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Symbol: {args.symbol}  |  Source: {args.source}  |  Sizing: {args.sizing}")

    # ── Load data ──────────────────────────────────────────────────────────
    print("\n  Loading data ...")
    df, data_source = load_price_data(
        symbol=args.symbol,
        years=args.years,
        source=args.source,
        csv_path=args.csv_path,
    )

    # ── Configure backtest ─────────────────────────────────────────────────
    cfg = BacktestConfig(
        initial_capital      = args.capital,
        commission_per_share = 0.005,
        slippage_pct         = 0.001,
        position_sizing      = args.sizing,
        position_pct         = 0.10,
        atr_risk_pct         = 0.01,
        allow_short          = args.allow_short,
        max_drawdown_pct     = None,  # no circuit breaker for the example
    )

    # ── Run backtest ───────────────────────────────────────────────────────
    print(f"\n  Running backtest: MACDMomentumStrategy on {args.symbol} ...")
    strategy = MACDMomentumStrategy()
    result   = run_backtest(strategy, df, cfg, symbol=args.symbol)

    # ── Print full report ──────────────────────────────────────────────────
    print_full_report(result, data_source)


if __name__ == "__main__":
    main()
