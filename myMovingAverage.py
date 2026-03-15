import argparse
# --- Configuration ---

def parse_args():
    parser = argparse.ArgumentParser(description="Moving Average Script")
    parser.add_argument('--ticker', type=str, default="AAPL", help='Ticker symbol')
    parser.add_argument('--timeout', type=int, default=60, help='Seconds to wait for data')
    parser.add_argument('--bar_count', type=int, default=20, help='Number of bars for moving average')
    parser.add_argument('--duration_str', type=str, default="1 D", help='Duration string for historical data')
    parser.add_argument('--bar_size_setting', type=str, default="1 min", help='Bar size setting')
    parser.add_argument('--what_to_show', type=str, default="TRADES", help='What to show')
    parser.add_argument('--use_rth', type=int, default=1, help='Use regular trading hours')
    parser.add_argument('--format_date', type=int, default=1, help='Format date')
    parser.add_argument('--keep_up_to_date', type=bool, default=False, help='Keep up to date')
    parser.add_argument('--extra_params', type=str, nargs='*', default=[], help='Extra parameters')
    return parser.parse_args()

import myIBApp
import time


def get_moving_average(
    app,
    ticker=TICKER,
    timeout=TIMEOUT,
    bar_count=BAR_COUNT,
    duration_str=DURATION_STR,
    bar_size_setting=BAR_SIZE_SETTING,
    what_to_show=WHAT_TO_SHOW,
    use_rth=USE_RTH,
    format_date=FORMAT_DATE,
    keep_up_to_date=KEEP_UP_TO_DATE,
    extra_params=EXTRA_PARAMS
) -> float:
    print(f"\nRequesting historical data for: {ticker}\n")

    req_id = 1
    app.req_id_to_ticker[req_id] = ticker
    app.data_received[req_id] = False
    contract = app.make_stock_contract(ticker)

    # Store bar data
    app.bars = []

    # Override historicalData callback to collect bars
    def historicalData(self, reqId, bar):
        self.log("historicalData", reqId, bar)
        self.bars.append(bar.close)
        if len(self.bars) >= bar_count:
            self.data_received[reqId] = True

    import types
    app.historicalData = types.MethodType(historicalData, app)

    # Request historical data (parameters as constants)
    app.reqHistoricalData(
        req_id,
        contract,
        "",  # endDateTime: "" means current time
        duration_str,
        bar_size_setting,
        what_to_show,
        use_rth,
        format_date,
        keep_up_to_date,
        extra_params
    )

    # Wait until enough bars are received or timeout
    deadline = time.time() + timeout
    while time.time() < deadline:
        with app._lock:
            done = app.data_received.get(req_id, False)
        if done:
            print("Enough bars received.")
            break
        time.sleep(0.1)

    if len(app.bars) < bar_count:
        print("Not enough bars received.")
        return None

    moving_avg = sum(app.bars[-bar_count:]) / bar_count
    print(f"\n--- {bar_count}-bar Moving Average for {ticker} ---\n{moving_avg:.2f}")
    return moving_avg

if __name__ == "__main__":
    args = parse_args()
    app = myIBApp.connect_to_tws()
    get_moving_average(
        app,
        ticker=args.ticker,
        timeout=args.timeout,
        bar_count=args.bar_count,
        duration_str=args.duration_str,
        bar_size_setting=args.bar_size_setting,
        what_to_show=args.what_to_show,
        use_rth=args.use_rth,
        format_date=args.format_date,
        keep_up_to_date=args.keep_up_to_date,
        extra_params=args.extra_params
    )
    myIBApp.disconnect_tws()
