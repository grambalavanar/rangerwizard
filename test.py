import threading
import time
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract

# --- Configuration ---
HOST = "127.0.0.1"
PORT = 7497          # 7497 for TWS paper, 7496 for TWS live, 4002 for IB Gateway paper
CLIENT_ID = 1
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
TIMEOUT = 10         # seconds to wait for data


class IBApp(EWrapper, EClient):
    def __init__(self):
        EWrapper.__init__(self)
        EClient.__init__(self, wrapper=self)
        self.prices = {}
        self.req_id_to_ticker = {}
        self.data_received = {}
        self._lock = threading.Lock()

    def error(self, req_id, error_code, error_string, advanced_order_reject_desc=""):
        # Suppress informational messages (2000-2999 are warnings, not errors)
        if error_code not in (2104, 2106, 2158, 2119):
            print(f"[Error] ReqId: {req_id} | Code: {error_code} | Msg: {error_string}")

    def nextValidId(self, order_id: int):
        """Called when connection is established."""
        super().nextValidId(order_id)
        self.next_order_id = order_id
        print(f"Connected. Next valid order ID: {order_id}")

    def tickPrice(self, req_id, tick_type, price, attrib):
        """
        Tick types:
          1 = Bid, 2 = Ask, 4 = Last, 6 = High, 7 = Low, 9 = Close
        """
        if tick_type == 4 and price > 0:  # Last price
            ticker = self.req_id_to_ticker.get(req_id)
            if ticker:
                with self._lock:
                    self.prices[ticker] = price
                    self.data_received[req_id] = True
                print(f"  {ticker}: ${price:.2f}")

    def tickSnapshotEnd(self, req_id):
        """Called when a snapshot request is complete."""
        with self._lock:
            self.data_received[req_id] = True


def make_stock_contract(symbol: str) -> Contract:
    contract = Contract()
    contract.symbol = symbol
    contract.secType = "STK"
    contract.exchange = "SMART"
    contract.currency = "USD"
    return contract


def get_stock_prices(tickers: list) -> dict:
    app = IBApp()
    app.connect(HOST, PORT, CLIENT_ID)

    # Run the socket in a background thread
    api_thread = threading.Thread(target=app.run, daemon=True)
    api_thread.start()

    # Wait for connection
    time.sleep(1)

    print(f"\nRequesting prices for: {tickers}\n")

    # Request a snapshot for each ticker
    for i, ticker in enumerate(tickers):
        req_id = i + 1
        app.req_id_to_ticker[req_id] = ticker
        app.data_received[req_id] = False
        contract = make_stock_contract(ticker)
        # snapshot=True requests a one-time price snapshot instead of streaming
        app.reqMktData(req_id, contract, "", True, False, [])

    # Wait until all data is received or timeout
    deadline = time.time() + TIMEOUT
    while time.time() < deadline:
        with app._lock:
            all_done = all(app.data_received.get(i + 1, False) for i in range(len(tickers)))
        if all_done:
            break
        time.sleep(0.1)

    app.disconnect()
    return app.prices


if __name__ == "__main__":
    prices = get_stock_prices(TICKERS)

    print("\n--- Final Prices ---")
    for ticker in TICKERS:
        price = prices.get(ticker)
        if price:
            print(f"  {ticker:6s}: ${price:.2f}")
        else:
            print(f"  {ticker:6s}: Price not available")
