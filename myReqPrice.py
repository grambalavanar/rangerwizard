import myIBApp
import time
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Request Stock Prices Script")
    parser.add_argument('--tickers', type=str, nargs='+', default=["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"], help='List of ticker symbols')
    parser.add_argument('--timeout', type=int, default=60, help='Seconds to wait for data')
    return parser.parse_args()

def get_stock_prices(app: myIBApp, tickers: list) -> dict:    
    print(f"\nRequesting prices for: {tickers}\n")

    # Request a snapshot for each ticker
    for i, ticker in enumerate(tickers):
        print("Requesting market data for", ticker)
        req_id = i + 1
        app.req_id_to_ticker[req_id] = ticker
        app.data_received[req_id] = False
        contract = app.make_stock_contract(ticker)
        # snapshot=True requests a one-time price snapshot instead of streaming
        app.reqMarketDataType(3)
        app.reqMktData(req_id, contract, "", True, False, [])

    # Wait until all data is received or timeout
    deadline = time.time() + TIMEOUT
    while time.time() < deadline:
        with app._lock:
            all_done = all(app.data_received.get(i + 1, False) for i in range(len(tickers)))
        if all_done:
            print("All data received.")
            break
        time.sleep(0.1)
    print("timeout reached or all data received. Responses:", app.prices)
    return app.prices

if __name__ == "__main__":
    args = parse_args()
    app = myIBApp.connect_to_tws()
    prices = get_stock_prices(app, args.tickers)
    print(f"\n--- Final Prices ---\n{prices}")
    myIBApp.disconnect_tws()

