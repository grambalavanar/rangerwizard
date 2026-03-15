import myIBApp
import myBreakoutSignal
import time
import argparse

# --- Constants ---
TICKER = "V"
BAR_COUNT = 1
TIMEOUT = 60
DURATION_STR = "1 Y"
BAR_SIZE_SETTING = "1 W"
WHAT_TO_SHOW = "TRADES"
USE_RTH = 1
FORMAT_DATE = 1
KEEP_UP_TO_DATE = False
EXTRA_PARAMS = []

def parse_args():
    parser = argparse.ArgumentParser(description="Backtesting Engine for Breakout Logic")
    parser.add_argument('--ticker', type=str, help='Ticker symbol')
    parser.add_argument('--bar_count', type=int, help='Lookback period for breakout')
    parser.add_argument('--timeout', type=int, help='Seconds to wait for data')
    parser.add_argument('--duration_str', type=str, help='Duration string for historical data')
    parser.add_argument('--bar_size_setting', type=str, help='Bar size setting')
    parser.add_argument('--what_to_show', type=str, help='What to show')
    parser.add_argument('--use_rth', type=int, help='Use regular trading hours')
    parser.add_argument('--format_date', type=int, help='Format date')
    parser.add_argument('--keep_up_to_date', type=bool, help='Keep up to date')
    parser.add_argument('--extra_params', type=str, nargs='*', help='Extra parameters')
    return parser.parse_args()


def run_backtest(
    app,
    ticker=TICKER,
    bar_count=BAR_COUNT,
    duration_str=DURATION_STR,
    bar_size_setting=BAR_SIZE_SETTING,
    what_to_show=WHAT_TO_SHOW,
    use_rth=USE_RTH,
    format_date=FORMAT_DATE,
    keep_up_to_date=KEEP_UP_TO_DATE,
    extra_params=EXTRA_PARAMS,
    timeout=TIMEOUT
):
    print(f"\nBacktesting breakout logic for: {ticker}\n")
    req_id = 99
    app.req_id_to_ticker[req_id] = ticker
    app.data_received[req_id] = False
    contract = app.make_stock_contract(ticker)
    app.bars = []

    # Override historicalData callback to collect bars
    def historicalData(self, reqId, bar):
        self.log("historicalData", reqId, bar)
        self.bars.append(bar.close)
        if len(self.bars) >= bar_count:
            self.data_received[reqId] = True

    import types
    app.historicalData = types.MethodType(historicalData, app)

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

    signals = myBreakoutSignal.breakout_signal(
        app.bars,
        lookback=bar_count
    )
    print(f"\n--- Backtest Results for {ticker} ---")
    for idx, signal in enumerate(signals, start=bar_count):
        price_now = app.bars[idx]
        price_next = app.bars[idx+1] if idx+1 < len(app.bars) else None
        # Determine correct decision: buy if price goes up, sell if price goes down, else none
        if price_next is not None:
            if price_next > price_now:
                correct = 'buy'
            elif price_next < price_now:
                correct = 'sell'
            else:
                correct = None
        else:
            correct = None
        print(f"Bar {idx}: Price={price_now}, Signal={signal}, Correct={correct}")
    fitness = calculate_fitness(signals, app.bars, bar_count)
    print(f"\n--- Fitness Score ---\n{fitness:.4f}")
    return signals, fitness
    
def calculate_fitness(signals, prices, lookback):
    """
    Simple fitness function: calculates cumulative returns from signals.
    Assumes buy/sell at next bar's close, ignores slippage/fees.
    Returns total return as fitness score.
    """
    returns = []
    for i, signal in enumerate(signals):
        idx = i + lookback
        if idx + 1 >= len(prices):
            break
        if signal == 'buy':
            ret = (prices[idx + 1] - prices[idx]) / prices[idx]
            returns.append(ret)
        elif signal == 'sell':
            ret = (prices[idx] - prices[idx + 1]) / prices[idx]
            returns.append(ret)
    fitness = sum(returns)
    return fitness


if __name__ == "__main__":
    args = parse_args()
    app = myIBApp.connect_to_tws()
    run_backtest(
        app,
        ticker=args.ticker if args.ticker is not None else TICKER,
        bar_count=args.bar_count if args.bar_count is not None else BAR_COUNT,
        duration_str=args.duration_str if args.duration_str is not None else DURATION_STR,
        bar_size_setting=args.bar_size_setting if args.bar_size_setting is not None else BAR_SIZE_SETTING,
        what_to_show=args.what_to_show if args.what_to_show is not None else WHAT_TO_SHOW,
        use_rth=args.use_rth if args.use_rth is not None else USE_RTH,
        format_date=args.format_date if args.format_date is not None else FORMAT_DATE,
        keep_up_to_date=args.keep_up_to_date if args.keep_up_to_date is not None else KEEP_UP_TO_DATE,
        extra_params=args.extra_params if args.extra_params is not None else EXTRA_PARAMS,
        timeout=args.timeout if args.timeout is not None else TIMEOUT
    )
    myIBApp.disconnect_tws()
