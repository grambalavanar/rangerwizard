# --- Configuration ---
TICKER = "AAPL"
TIMEOUT = 60  # seconds to wait for data
BAR_COUNT = 20  # Number of bars for moving average

# --- Historical Data Request Parameters ---
DURATION_STR = "1 D"      # durationStr: 1 day
BAR_SIZE_SETTING = "1 min" # barSizeSetting
WHAT_TO_SHOW = "TRADES"   # whatToShow
USE_RTH = 1                # useRTH: 1 = regular trading hours
FORMAT_DATE = 1            # formatDate: 1 = yyyyMMdd HH:mm:ss
KEEP_UP_TO_DATE = False    # keepUpToDate
EXTRA_PARAMS = []          # extra parameters

import myIBApp
import time


def get_moving_average(
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
    app = myIBApp.connect_to_tws()
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

    app.disconnect()

    if len(app.bars) < bar_count:
        print("Not enough bars received.")
        return None

    moving_avg = sum(app.bars[-bar_count:]) / bar_count
    print(f"\n--- {bar_count}-bar Moving Average for {ticker} ---\n{moving_avg:.2f}")
    return moving_avg

if __name__ == "__main__":
    get_moving_average()
