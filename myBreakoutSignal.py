import myIBApp
import time

TICKER = "V"
TIMEOUT = 60  # seconds to wait for data
BAR_COUNT = 20  # Number of bars for breakout lookback

# --- Historical Data Request Parameters ---
DURATION_STR = "1 Y"      # durationStr: 1 day
BAR_SIZE_SETTING = "1 W" # barSizeSetting
WHAT_TO_SHOW = "TRADES"   # whatToShow
USE_RTH = 1                # useRTH: 1 = regular trading hours
FORMAT_DATE = 1            # formatDate: 1 = yyyyMMdd HH:mm:ss
KEEP_UP_TO_DATE = False    # keepUpToDate
EXTRA_PARAMS = []          # extra parameters

def breakout_signal(prices, lookback=BAR_COUNT):
    """
    Generates breakout signals.
    Buy: price breaks above highest high of lookback period.
    Sell: price breaks below lowest low of lookback period.
    Returns: list of signals ('buy', 'sell', or None)
    """
    signals = []
    for i in range(lookback, len(prices)):
        high = max(prices[i-lookback:i])
        low = min(prices[i-lookback:i])
        if prices[i] > high:
            signals.append('buy')
        elif prices[i] < low:
            signals.append('sell')
        else:
            signals.append(None)
    return signals

def get_breakout_signals(
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
):
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

    # Request historical data
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

    signals = breakout_signal(app.bars, lookback=bar_count)
    print(f"\n--- Breakout Signals for {ticker} ---")
    for idx, signal in enumerate(signals, start=bar_count):
        print(f"Bar {idx}: Price={app.bars[idx]}, Signal={signal}")
    # ASCII plot
    print(f"\n{TICKER}")
    y_map = {'buy': 1, 'sell': -1, None: 0}
    y_labels = {1: ' B ', 0: ' . ', -1: ' S '}
    plot_height = 3
    plot = [['   ' for _ in range(len(signals))] for _ in range(plot_height)]
    for i, signal in enumerate(signals):
        y = y_map[signal]
        row = 1 - y  # 1: buy (top), 1: none (middle), 2: sell (bottom)
        plot[row][i] = y_labels[y]
    for row_idx, row in enumerate(plot):
        if row_idx == 0:
            label = 'BUY '
        elif row_idx == 1:
            label = 'NONE'
        else:
            label = 'SELL'
        print(label + '|' + ''.join(row))
    print('     ' + '-' * (len(signals) * 3))
    print('     ' + ''.join([f'{i+BAR_COUNT:3}' for i in range(len(signals))]))
    return signals

if __name__ == "__main__":
    app = myIBApp.connect_to_tws()
    get_breakout_signals(app)
    myIBApp.disconnect_tws()
