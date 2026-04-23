# mean_reversion.py
import math
import time
import signal
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from ib_strategy_app import StrategyApp, connect_strategy_app


# ============================================================
# Configuration
# ============================================================
@dataclass
class Config:
    symbol: str = "SPY"
    exchange: str = "ARCA"       # SPY trades on ARCA; override in contract factory if needed
    # Signal
    lookback: int = 20           # bars for rolling mean/std
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 3.5
    # Risk
    starting_equity: float = 100_000.0
    risk_per_trade: float = 0.01
    max_shares: int = 1_000
    # Stationarity filter (applied to historical warm-up)
    adf_p_threshold: float = 0.05
    min_half_life_bars: int = 1
    max_half_life_bars: int = 30
    require_stationary: bool = True   # set False to trade regardless of ADF
    # Warm-up history
    warmup_duration: str = "60 D"
    warmup_bar_size: str = "1 day"    # switch to "5 mins" for intraday trading
    live_bar_size_seconds: int = 5    # reqRealTimeBars is fixed at 5s
    # Execution
    dry_run: bool = True              # True = log only, False = live orders


# ============================================================
# Statistical helpers
# ============================================================
def adf_pvalue(series: pd.Series) -> float:
    s = series.dropna()
    if len(s) < 20:
        return 1.0
    try:
        return adfuller(s, autolag="AIC")[1]
    except Exception:
        return 1.0


def half_life(series: pd.Series) -> float:
    """Estimate OU half-life via regression of Δy_t on y_{t-1}."""
    s = series.dropna().values
    if len(s) < 20:
        return math.inf
    y_lag = s[:-1]
    dy = np.diff(s)
    y_lag = y_lag - y_lag.mean()
    try:
        beta = np.dot(y_lag, dy) / np.dot(y_lag, y_lag)
        if beta >= 0:
            return math.inf
        return -math.log(2) / beta
    except Exception:
        return math.inf


def hurst_exponent(series: pd.Series, max_lag: int = 20) -> float:
    s = series.dropna().values
    if len(s) < max_lag + 2:
        return 0.5
    lags = range(2, max_lag)
    tau = [np.std(np.subtract(s[lag:], s[:-lag])) for lag in lags]
    tau = [t if t > 0 else 1e-10 for t in tau]
    return float(np.polyfit(np.log(lags), np.log(tau), 1)[0])


# ============================================================
# Strategy engine
# ============================================================
@dataclass
class Position:
    qty: int = 0              # signed: +long, -short
    entry_price: float = 0.0
    entry_z: float = 0.0

    @property
    def is_flat(self) -> bool:
        return self.qty == 0


class MeanReversionStrategy:
    def __init__(self, app: StrategyApp, cfg: Config):
        self.app = app
        self.cfg = cfg

        self.contract = app.make_stock_contract(cfg.symbol)
        # Override exchange if needed (myIBApp hardcodes NASDAQ)
        self.contract.exchange = cfg.exchange

        self.prices: deque = deque(maxlen=max(cfg.lookback * 5, 200))
        self.position = Position()
        self.equity = cfg.starting_equity
        self.stationary_ok = False

        self.log = logging.getLogger("MR")

    # ------------------ warm-up ------------------
    def warm_up(self):
        self.log.info("Fetching warm-up history ...")
        df = self.app.fetch_historical(
            self.contract,
            duration=self.cfg.warmup_duration,
            bar_size=self.cfg.warmup_bar_size,
        )
        if df.empty or len(df) < self.cfg.lookback + 10:
            raise RuntimeError("Not enough historical data for warm-up")

        closes = df["close"].astype(float)
        for px in closes.tail(self.prices.maxlen):
            self.prices.append(float(px))

        # Stationarity diagnostics
        pval = adf_pvalue(closes)
        hl   = half_life(closes)
        hurst = hurst_exponent(closes)
        self.log.info(f"ADF p={pval:.4f}  half-life={hl:.2f} bars  Hurst={hurst:.3f}")

        self.stationary_ok = (
            pval < self.cfg.adf_p_threshold
            and self.cfg.min_half_life_bars <= hl <= self.cfg.max_half_life_bars
        )
        if self.cfg.require_stationary and not self.stationary_ok:
            self.log.warning("Series failed stationarity filter — strategy will NOT enter new trades.")
        else:
            self.log.info("Series passed mean-reversion filters (or filter disabled).")

    # ------------------ signal ------------------
    def _zscore(self) -> Optional[float]:
        if len(self.prices) < self.cfg.lookback:
            return None
        window = np.array(list(self.prices)[-self.cfg.lookback:])
        mu = window.mean()
        sd = window.std(ddof=1)
        if sd == 0 or np.isnan(sd):
            return None
        return (self.prices[-1] - mu) / sd

    # ------------------ sizing ------------------
    def _size(self, price: float) -> int:
        window = np.array(list(self.prices)[-self.cfg.lookback:])
        sd = window.std(ddof=1)
        if sd == 0 or np.isnan(sd):
            return 0
        # Dollar distance from entry-z to stop-z on the price
        stop_dist = sd * (self.cfg.stop_z - self.cfg.entry_z)
        dollar_risk = self.equity * self.cfg.risk_per_trade
        shares = int(dollar_risk / max(stop_dist, 1e-6))
        # Cap by notional and hard cap
        notional_cap = int((self.equity * 0.5) / price)
        return max(0, min(shares, notional_cap, self.cfg.max_shares))

    # ------------------ orders ------------------
    def _submit(self, action: str, qty: int, tag: str):
        if qty <= 0:
            return
        self.log.info(f"[{tag}] {action} {qty} {self.cfg.symbol}  dry_run={self.cfg.dry_run}")
        if self.cfg.dry_run:
            return
        self.app.place_market_order(self.contract, action, qty)

    def _enter_long(self, price: float, z: float):
        qty = self._size(price)
        if qty == 0:
            return
        self._submit("BUY", qty, f"ENTRY LONG z={z:.2f}")
        self.position = Position(qty=+qty, entry_price=price, entry_z=z)

    def _enter_short(self, price: float, z: float):
        qty = self._size(price)
        if qty == 0:
            return
        self._submit("SELL", qty, f"ENTRY SHORT z={z:.2f}")
        self.position = Position(qty=-qty, entry_price=price, entry_z=z)

    def _close(self, price: float, reason: str):
        if self.position.is_flat:
            return
        qty = abs(self.position.qty)
        action = "SELL" if self.position.qty > 0 else "BUY"
        pnl = (price - self.position.entry_price) * self.position.qty
        self.log.info(f"[EXIT {reason}] {action} {qty}  pnl≈${pnl:.2f}")
        self._submit(action, qty, f"EXIT {reason}")
        self.equity += pnl
        self.position = Position()

    # ------------------ bar handler ------------------
    def on_bar(self, bar: dict):
        price = float(bar["close"])
        self.prices.append(price)

        z = self._zscore()
        if z is None:
            return

        # --- exit logic first ---
        if not self.position.is_flat:
            if abs(z) < self.cfg.exit_z:
                self._close(price, "reversion")
                return
            if abs(z) > self.cfg.stop_z:
                self._close(price, "stop")
                return
            return  # still in trade, no re-entry

        # --- entries (only if stationary filter passed) ---
        if self.cfg.require_stationary and not self.stationary_ok:
            return

        if z < -self.cfg.entry_z:
            self._enter_long(price, z)
        elif z > self.cfg.entry_z:
            self._enter_short(price, z)


# ============================================================
# Main
# ============================================================
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    cfg = Config()

    app = connect_strategy_app()
    strat = MeanReversionStrategy(app, cfg)

    try:
        strat.warm_up()

        # Subscribe to live bars. reqRealTimeBars gives 5-second bars.
        # For daily-bar strategies, you'd instead loop fetch_historical once per day.
        req_id = app.subscribe_rt_bars(strat.contract, strat.on_bar)
        logging.info(f"Subscribed to real-time bars reqId={req_id}")

        # Graceful shutdown
        stop = False
        def _sigint(*_):
            nonlocal stop
            stop = True
        signal.signal(signal.SIGINT, _sigint)

        while not stop:
            time.sleep(1.0)

    finally:
        # Flatten any open position on shutdown
        if strat.position.qty != 0:
            last = strat.prices[-1] if strat.prices else 0.0
            strat._close(last, "shutdown")
        from myIBApp import disconnect_tws
        try:
            app.cancelRealTimeBars(req_id)
        except Exception:
            pass
        app.disconnect()
        logging.info("Disconnected.")


if __name__ == "__main__":
    main()
