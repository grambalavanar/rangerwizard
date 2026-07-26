# strategies/test — re-exports all testing tools from momentum/test
import os, sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from momentum.test.strategy_tester import (
    Strategy, BacktestConfig, BacktestResult, Trade,
    run_backtest, run_comparison, fetch_ibkr_history,
    print_result, compare_results, plot_equity_curve,
    MACDMomentumStrategy,
)
from momentum.test.genetic_optimizer import (
    IntParam, FloatParam, ChoiceParam, ParameterSpace,
    Individual, GAConfig, OptimizationResult,
    GeneticOptimizer, save_result, load_result,
    print_optimization_report,
)
from momentum.test.run_backtest_example import (
    load_price_data,
    ascii_line_chart, ascii_area_chart, ascii_histogram,
    ascii_monthly_table, ascii_trade_table,
    print_full_report,
)
from momentum.test.paper_trader import (
    PaperConfig, PaperTrader, TradingState, SymbolState,
    load_strategy, review_paper_performance,
)
