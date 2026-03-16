import myIBApp
import time
import argparse

# --- Constants ---
TICKER = "msft" #["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
BAR_COUNT = 30
TIMEOUT = 60
DURATION_STR = "1 W"
BAR_SIZE_SETTING = "30 mins"
WHAT_TO_SHOW = "TRADES"
USE_RTH = 1
FORMAT_DATE = 1
KEEP_UP_TO_DATE = False
EXTRA_PARAMS = []


def parse_args():
    parser = argparse.ArgumentParser(description="Breakout Signal Script")
    parser.add_argument('--ticker', type=str, default=TICKER, help='Ticker symbol')
    parser.add_argument('--bar_count', type=int, default=BAR_COUNT, help='Lookback period for breakout')
    parser.add_argument('--timeout', type=int, default=TIMEOUT, help='Seconds to wait for data')
    parser.add_argument('--duration_str', type=str, default=DURATION_STR, help='Duration string for historical data')
    parser.add_argument('--bar_size_setting', type=str, default=BAR_SIZE_SETTING, help='Bar size setting')
    parser.add_argument('--what_to_show', type=str, default=WHAT_TO_SHOW, help='What to show')
    parser.add_argument('--use_rth', type=int, default=USE_RTH, help='Use regular trading hours')
    parser.add_argument('--format_date', type=int, default=FORMAT_DATE, help='Format date')
    parser.add_argument('--keep_up_to_date', type=bool, default=KEEP_UP_TO_DATE, help='Keep up to date')
    parser.add_argument('--extra_params', type=str, nargs='*', default=EXTRA_PARAMS, help='Extra parameters')
    return parser.parse_args()

def calculate_atr(prices, lookback=14):
    """
    Calculate Average True Range for volatility.
    """
    if len(prices) < lookback + 1:
        return 0
    true_ranges = []
    for i in range(1, len(prices)):
        tr = prices[i-1] - prices[i]
        true_ranges.append(abs(tr))
    atr = sum(true_ranges[-lookback:]) / lookback
    return atr

def calculate_sma(prices, lookback):
    """
    Calculate Simple Moving Average.
    """
    if len(prices) < lookback:
        return None
    return sum(prices[-lookback:]) / lookback

def breakout_signal(prices, lookback=BAR_COUNT, use_trend_filter=True, consecutive=2, use_atr=True):
    """
    Generates enhanced breakout signals.
    Buy: price breaks above highest high of lookback period.
    Sell: price breaks below lowest low of lookback period.
    
    Enhancements:
    - use_trend_filter: Require price above/below SMA for confirmation
    - consecutive: Number of consecutive bars required for signal
    - use_atr: Scale breakout levels by ATR volatility
    
    Returns: list of signals ('buy', 'sell', or None)
    """
    signals = []
    sma_lookback = min(20, lookback // 2)  # SMA period for trend
    
    for i in range(lookback, len(prices)):
        high = max(prices[i-lookback:i])
        low = min(prices[i-lookback:i])
        current_price = prices[i]
        
        # ATR-based scaling
        if use_atr:
            atr = calculate_atr(prices[:i+1])
            high = high - (atr * 0.5)  # Reduce breakout threshold
            low = low + (atr * 0.5)
        
        # Check breakout
        buy_breakout = current_price > high
        sell_breakout = current_price < low
        
        # Trend filter: check if price is above/below SMA
        sma = calculate_sma(prices[:i+1], sma_lookback)
        trend_ok = True
        if use_trend_filter and sma is not None:
            if buy_breakout and current_price < sma:
                trend_ok = False
            elif sell_breakout and current_price > sma:
                trend_ok = False
        
        # Consecutive bar confirmation
        if consecutive > 1:
            consecutive_count = 0
            for j in range(max(i-consecutive+1, lookback), i+1):
                test_high = max(prices[j-lookback:j])
                test_low = min(prices[j-lookback:j])
                if use_atr:
                    test_atr = calculate_atr(prices[:j+1])
                    test_high -= (test_atr * 0.5)
                    test_low += (test_atr * 0.5)
                if buy_breakout and prices[j] > test_high:
                    consecutive_count += 1
                elif sell_breakout and prices[j] < test_low:
                    consecutive_count += 1
            if consecutive_count < consecutive:
                buy_breakout = False
                sell_breakout = False
        
        # Generate signal
        if buy_breakout and trend_ok:
            signals.append('buy')
        elif sell_breakout and trend_ok:
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

    signals = breakout_signal(app.bars, lookback=bar_count, use_trend_filter=True, consecutive=1, use_atr=True)
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
    args = parse_args()
    app = myIBApp.connect_to_tws()
    get_breakout_signals(
        app,
        ticker=args.ticker if args.ticker is not None else TICKER,
        timeout=args.timeout if args.timeout is not None else TIMEOUT,
        bar_count=args.bar_count if args.bar_count is not None else BAR_COUNT,
        duration_str=args.duration_str if args.duration_str is not None else DURATION_STR,
        bar_size_setting=args.bar_size_setting if args.bar_size_setting is not None else BAR_SIZE_SETTING,
        what_to_show=args.what_to_show if args.what_to_show is not None else WHAT_TO_SHOW,
        use_rth=args.use_rth if args.use_rth is not None else USE_RTH,
        format_date=args.format_date if args.format_date is not None else FORMAT_DATE,
        keep_up_to_date=args.keep_up_to_date if args.keep_up_to_date is not None else KEEP_UP_TO_DATE,
        extra_params=args.extra_params if args.extra_params is not None else EXTRA_PARAMS
    )
    myIBApp.disconnect_tws()
