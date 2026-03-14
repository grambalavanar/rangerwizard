import threading
import time
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.ticktype import TickTypeEnum


# --- Configuration ---
HOST = "127.0.0.1"
PORT = 7496          # 7497 for TWS paper, 7496 for TWS live, 4002 for IB Gateway paper
CLIENT_ID = 0
LOGGING_ENABLED = False


# Singleton connection variables
_singleton_app = None
_api_thread = None

class myIBApp(EWrapper, EClient):
    def __init__(self):
        EWrapper.__init__(self)
        EClient.__init__(self, wrapper=self)
        self.prices = {}
        self.req_id_to_ticker = {}
        self.data_received = {}
        self._lock = threading.Lock()

    def log(self, method, *args):
        if LOGGING_ENABLED:
            print(f"[{method}]: {args}")

    def tickPrice(self, req_id, tick_type, price, attrib):
        """
        Tick types:
          1 = Bid, 2 = Ask, 4 = Last, 6 = High, 7 = Low, 9 = Close
        """
        self.log("tickPrice", "req_id:", req_id, "tick_type:", tick_type, "price:", price, "attrib:", attrib)
        ticker = self.req_id_to_ticker.get(req_id)
        print(f"  {ticker} - {TickTypeEnum.to_str(tick_type)}: ${price:.2f}")

    def tickSize(self, req_id, tick_type, size):
        self.log("tickSize", "request_id:", req_id, "tick_type:", tick_type, "size:", size)

    def tickString(self, req_id, tick_type, value):
        self.log("tickString", "request_id:", req_id, "tick_type:", tick_type, "value:", value)

    def tickGeneric(self, req_id, tick_type, value):
        self.log("tickGeneric", "request_id:", req_id, "tick_type:", tick_type, "value:", value)

    def tickOptionComputation(self, req_id, tick_type, implied_vol, delta, opt_price, pv_dividend, gamma, vega, theta, und_price):
        self.log("tickOptionComputation", "request_id:", req_id, "tick_type:", tick_type, "implied_vol:", implied_vol, "delta:", delta, "opt_price:", opt_price, "pv_dividend:", pv_dividend, "gamma:", gamma, "vega:", vega, "theta:", theta, "und_price:", und_price)

    def tickReqParams(self, req_id, min_tick, bbo_tick_types, snapshot_permissions):
        self.log("tickReqParams", "request_id:", req_id, "min_tick:", min_tick, "bbo_tick_types:", bbo_tick_types, "snapshot_permissions:",  snapshot_permissions)

    def tickByTickAllLast(self, req_id, tick_type, time, price, size, attribs, exchange, special_conditions):
        self.log("tickByTickAllLast", "request_id:", req_id, "tick_type:", tick_type, "time:", time, "price:", price, "size:", size, "attribs:", attribs, "exchange:", exchange, "special_conditions:", special_conditions)

    def tickByTickBidAsk(self, req_id, time, bid_price, ask_price, bid_size, ask_size, attribs):
        self.log("tickByTickBidAsk", "request_id:", req_id, "time:", time, "bid_price:", bid_price, "ask_price:", ask_price, "bid_size:", bid_size, "ask_size:", ask_size, "attribs:", attribs)

    def tickByTickMidPoint(self, req_id, time, mid_price):
        self.log("tickByTickMidPoint", "request_id:", req_id, "time:", time, "mid_price:", mid_price)

    def tickSnapshotEnd(self, req_id):
        """Called when a snapshot request is complete."""
        with self._lock:
            self.data_received[req_id] = True
        self.log("tickSnapshotEnd", "request_id:", req_id)


    def make_stock_contract(self, symbol: str) -> Contract:
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.exchange = "NASDAQ"
        contract.currency = "USD"
        print(f"symbol: {contract.symbol}, secType: {contract.secType}, exchange: {contract.exchange}, currency: {contract.currency}")
        return contract

def connect_to_tws():
    global _singleton_app, _api_thread
    if _singleton_app is not None:
        return _singleton_app
    app = myIBApp()
    app.connect(HOST, PORT, CLIENT_ID)
    print("serverVersion:%s connectionTime:%s" % (app.serverVersion(), app.twsConnectionTime()))
    # Run the socket in a background thread
    api_thread = threading.Thread(target=app.run, daemon=True)
    api_thread.start()

    # Wait for connection
    time.sleep(1)
    _singleton_app = app
    return app

def disconnect_tws():
    global _singleton_app
    if _singleton_app is not None:
        _singleton_app.disconnect()
        _singleton_app = None