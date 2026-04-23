# ib_strategy_app.py
import threading
import time
import queue
from collections import defaultdict

import pandas as pd

from ibapi.contract import Contract
from ibapi.order import Order

from myIBApp import myIBApp, HOST, PORT, CLIENT_ID


class StrategyApp(myIBApp):
    """
    Extends myIBApp with:
      - historical bar fetching (blocking helper)
      - real-time 5-second bars
      - order placement and order-status tracking
      - position and account callbacks
    """

    def __init__(self):
        super().__init__()

        # Historical data buffers keyed by reqId
        self._hist_buffers = defaultdict(list)
        self._hist_done = {}
        self._hist_events = {}

        # Real-time bar buffers keyed by reqId
        self.rt_bars = defaultdict(list)          # list of dict rows
        self.rt_callbacks = {}                    # reqId -> callable(bar_row)

        # Order bookkeeping
        self.next_order_id = None
        self._oid_event = threading.Event()
        self.order_status = {}                    # orderId -> status dict
        self.positions = {}                       # symbol -> qty
        self._positions_done = threading.Event()
        self.account_values = {}

        # Request-id counter (start above any ids used for market data)
        self._req_id = 9000
        self._req_lock = threading.Lock()

    # -------- utility --------
    def next_req_id(self) -> int:
        with self._req_lock:
            self._req_id += 1
            return self._req_id

    # -------- next valid order id --------
    def nextValidId(self, orderId: int):
        super().nextValidId(orderId) if hasattr(super(), "nextValidId") else None
        self.next_order_id = orderId
        self._oid_event.set()

    def _get_next_order_id(self) -> int:
        self._oid_event.wait(timeout=5)
        oid = self.next_order_id
        self.next_order_id += 1
        return oid

    # -------- historical data --------
    def historicalData(self, reqId, bar):
        self._hist_buffers[reqId].append({
            "date": bar.date,
            "open": bar.open,
            "high": bar.high,
            "low":  bar.low,
            "close": bar.close,
            "volume": bar.volume,
        })

    def historicalDataEnd(self, reqId, start, end):
        self._hist_done[reqId] = True
        ev = self._hist_events.get(reqId)
        if ev:
            ev.set()

    def fetch_historical(
        self,
        contract: Contract,
        duration: str = "60 D",
        bar_size: str = "1 day",
        what_to_show: str = "TRADES",
        use_rth: int = 1,
        timeout: float = 30.0,
    ) -> pd.DataFrame:
        """Blocking historical bar fetch -> DataFrame indexed by datetime."""
        req_id = self.next_req_id()
        ev = threading.Event()
        self._hist_events[req_id] = ev

        self.reqHistoricalData(
            reqId=req_id,
            contract=contract,
            endDateTime="",
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow=what_to_show,
            useRTH=use_rth,
            formatDate=1,
            keepUpToDate=False,
            chartOptions=[],
        )

        if not ev.wait(timeout=timeout):
            raise TimeoutError(f"Historical data request {req_id} timed out")

        rows = self._hist_buffers.pop(req_id, [])
        self._hist_events.pop(req_id, None)
        self._hist_done.pop(req_id, None)

        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).set_index("date").sort_index()
        return df

    # -------- real-time 5-second bars --------
    def realtimeBar(self, reqId, time_, open_, high, low, close, volume, wap, count):
        row = {
            "time": pd.to_datetime(time_, unit="s"),
            "open": open_, "high": high, "low": low, "close": close,
            "volume": volume,
        }
        self.rt_bars[reqId].append(row)
        cb = self.rt_callbacks.get(reqId)
        if cb:
            try:
                cb(row)
            except Exception as e:
                print(f"[realtimeBar callback error] {e}")

    def subscribe_rt_bars(self, contract: Contract, on_bar) -> int:
        req_id = self.next_req_id()
        self.rt_callbacks[req_id] = on_bar
        self.reqRealTimeBars(req_id, contract, 5, "TRADES", True, [])
        return req_id

    # -------- orders --------
    def orderStatus(self, orderId, status, filled, remaining, avgFillPrice,
                    permId, parentId, lastFillPrice, clientId, whyHeld, mktCapPrice):
        self.order_status[orderId] = {
            "status": status, "filled": filled, "remaining": remaining,
            "avg_fill": avgFillPrice,
        }

    def place_market_order(self, contract: Contract, action: str, qty: int) -> int:
        """action = 'BUY' or 'SELL'. Returns orderId."""
        oid = self._get_next_order_id()
        order = Order()
        order.action = action
        order.orderType = "MKT"
        order.totalQuantity = qty
        order.tif = "DAY"
        order.eTradeOnly = False
        order.firmQuoteOnly = False
        self.placeOrder(oid, contract, order)
        return oid

    # -------- positions --------
    def position(self, account, contract, position, avgCost):
        self.positions[contract.symbol] = position

    def positionEnd(self):
        self._positions_done.set()

    def refresh_positions(self, timeout=5.0):
        self._positions_done.clear()
        self.reqPositions()
        self._positions_done.wait(timeout=timeout)
        self.cancelPositions()
        return dict(self.positions)


def connect_strategy_app() -> StrategyApp:
    app = StrategyApp()
    app.connect(HOST, PORT, CLIENT_ID)
    t = threading.Thread(target=app.run, daemon=True)
    t.start()
    # Wait for nextValidId
    app._oid_event.wait(timeout=5)
    print(f"Connected. serverVersion={app.serverVersion()} time={app.twsConnectionTime()}")
    return app
