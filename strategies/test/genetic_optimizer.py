"""
genetic_optimizer.py
====================
Genetic algorithm for tuning strategy parameters across a basket of stocks.
Finds parameter sets that generalise across different market regimes by
evaluating fitness simultaneously on multiple instruments.

How it works
------------
  Population of chromosomes (parameter dicts) is evolved over ``n_generations``
  using selection → crossover → mutation. Fitness is computed by running a full
  backtest on every symbol in the basket and aggregating the results into a
  single scalar score. The "robust" fitness metric penalises inconsistency
  across stocks, pushing evolution toward parameters that work everywhere rather
  than parameters that only work on one stock.

Quick example
-------------
    from strategies.test.genetic_optimizer import (
        GeneticOptimizer, ParameterSpace, IntParam, FloatParam, GAConfig
    )
    from strategies.test.strategy_tester import Strategy, BacktestConfig
    from strategies.tools.momentum_tools import macd, adx, crossover

    class TunableMACD(Strategy):
        def __init__(self, params):
            self.params = params

        @property
        def name(self):
            return f"MACD(f={self.params['fast']},s={self.params['slow']})"

        def generate_signals(self, df):
            m, sig, _ = macd(df["Close"],
                             window_fast=self.params["fast"],
                             window_slow=self.params["slow"])
            adx_v, _, _ = adx(df["High"], df["Low"], df["Close"])
            df["signal"] = 0
            df.loc[crossover(m, sig) & (adx_v > self.params["adx_min"]), "signal"] = 1
            df.loc[m < sig, "signal"] = 0
            df["signal"] = df["signal"].ffill().fillna(0)
            return df

    space = ParameterSpace(
        params={
            "fast":    IntParam(5,  20),
            "slow":    IntParam(15, 60),
            "adx_min": FloatParam(15.0, 40.0),
        },
        constraints=[lambda p: p["fast"] < p["slow"]],
    )

    opt = GeneticOptimizer(
        strategy_factory = TunableMACD,
        param_space      = space,
        symbols          = ["AAPL", "MSFT", "SPY", "QQQ", "TSLA", "JPM"],
        config           = GAConfig(population_size=30, n_generations=20),
    )
    result = opt.run()
    print_optimization_report(result)

Run as a script
---------------
    python momentum/test/genetic_optimizer.py

Dependencies: numpy, pandas, scipy, yfinance
"""

import json
import math
import os
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from strategies.test.strategy_tester import (
    Strategy, BacktestConfig, BacktestResult, run_backtest,
)
from strategies.test.run_backtest_example import load_price_data

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ============================================================
# 1. PARAMETER SPACE DEFINITION
# ============================================================

@dataclass
class IntParam:
    """
    Integer-valued parameter with inclusive bounds.
    ------------------------------------------------
    What it is
        Defines a single integer hyperparameter with a minimum and maximum
        value. Used as the building block of a ``ParameterSpace`` to tell
        the genetic optimizer what valid values look like for an integer
        parameter (e.g., an RSI window of 5–50 bars).

    Used by
        Passed as values in the ``params`` dict to ``ParameterSpace``.
        The optimizer samples, mutates, and crossovers this type
        automatically — you never call its methods directly.

    When you need this
        Whenever a strategy parameter is a whole number: indicator
        windows (RSI, MACD, ATR), lookback periods, bar counts, etc.

    Code example
        >>> space = ParameterSpace({"rsi_window": IntParam(5, 50)})
        >>> space.random_chromosome()
        {'rsi_window': 23}

    Args:
        min_val (int): Minimum allowable value (inclusive).
        max_val (int): Maximum allowable value (inclusive).
    """
    min_val: int
    max_val: int


@dataclass
class FloatParam:
    """
    Continuous float-valued parameter with inclusive bounds.
    ---------------------------------------------------------
    What it is
        Defines a continuous hyperparameter with a minimum and maximum
        value. Used for thresholds, multipliers, or any parameter that
        doesn't need to be a whole number (e.g., an ADX threshold of
        15.0–40.0, a stop-loss multiplier of 1.5×–3.0× ATR).

    Used by
        Passed as values in the ``params`` dict to ``ParameterSpace``.
        The Gaussian mutation step adds normally-distributed noise scaled
        to the parameter range, allowing smooth exploration of the space.

    When you need this
        For thresholds (RSI overbought level, ADX cutoff), multipliers
        (ATR stop-loss distance, Kelly fraction), or any decimal quantity
        in your strategy.

    Code example
        >>> space = ParameterSpace({"adx_min": FloatParam(15.0, 40.0)})
        >>> space.random_chromosome()
        {'adx_min': 27.43}

    Args:
        min_val (float): Minimum value (inclusive).
        max_val (float): Maximum value (inclusive).
        decimals (int):  Round sampled/mutated values to this many decimal
                         places (default 2).
    """
    min_val: float
    max_val: float
    decimals: int = 2


@dataclass
class ChoiceParam:
    """
    Discrete-choice parameter chosen from a predefined list.
    ---------------------------------------------------------
    What it is
        Defines a parameter that can only take one of a fixed set of
        values. Mutation randomly picks one of the choices; crossover
        randomly inherits one parent's value.

    Used by
        Passed as values in the ``params`` dict to ``ParameterSpace``.
        Use this for categorical strategy parameters such as which
        indicator type to use, the position sizing method, or any
        parameter with a small discrete set of valid values.

    When you need this
        When a parameter represents a mode or category:
        - Choosing between "sma" and "ema" as the trend filter
        - Selecting a bar size ("1 day" vs "1 hour")
        - Switching between "bull_only" and "both_directions" position
          modes.

    Code example
        >>> space = ParameterSpace({
        ...     "ma_type": ChoiceParam(["sma", "ema", "wma"])
        ... })
        >>> space.random_chromosome()
        {'ma_type': 'ema'}

    Args:
        choices (list): All valid values for this parameter.
    """
    choices: list


# ── Parameter space ──────────────────────────────────────────────────────────

class ParameterSpace:
    """
    Defines the search space and all genetic operations for a set of parameters.
    -----------------------------------------------------------------------------
    What it is
        A container that holds the definition of every parameter in your
        strategy's tunable configuration. It knows how to:
          • Draw a random chromosome (random initial individual)
          • Perform crossover between two parent chromosomes
          • Mutate a chromosome by randomly perturbing its values
          • Enforce constraints (e.g., fast EMA window < slow EMA window)

    Used by
        ``GeneticOptimizer.__init__`` stores the space and calls its
        methods during evolution. You create one and pass it in — you
        never call its methods directly during normal use.

    When you need this
        Every time you set up a new optimization run. Define one
        ``ParameterSpace`` per strategy type. You can reuse the same space
        across different stock baskets or ``GAConfig`` settings.

    Code example
        >>> from strategies.test.genetic_optimizer import (
        ...     ParameterSpace, IntParam, FloatParam, ChoiceParam
        ... )
        >>> space = ParameterSpace(
        ...     params={
        ...         "rsi_window": IntParam(5,  50),
        ...         "macd_fast":  IntParam(5,  20),
        ...         "macd_slow":  IntParam(15, 60),
        ...         "adx_min":    FloatParam(15.0, 40.0),
        ...     },
        ...     constraints=[
        ...         lambda p: p["macd_fast"] < p["macd_slow"],  # MACD constraint
        ...     ]
        ... )
        >>> chrom = space.random_chromosome()
        >>> print(chrom)
        {'rsi_window': 14, 'macd_fast': 10, 'macd_slow': 28, 'adx_min': 22.5}

    Args:
        params      (dict):  Mapping of parameter name → Param object.
        constraints (list):  Optional list of callables ``f(chromosome) → bool``.
                             All must return True for the chromosome to be valid.
                             Invalid chromosomes are repaired or resampled.
    """

    def __init__(
        self,
        params: Dict[str, Any],
        constraints: Optional[List[Callable]] = None,
    ) -> None:
        self.params      = params
        self.constraints = constraints or []

    def random_chromosome(self, max_retries: int = 100) -> dict:
        """
        Sample a random valid chromosome from the parameter space.
        -----------------------------------------------------------
        What it does
            Randomly samples each parameter within its bounds, then checks
            all constraints. If any constraint is violated, it resamples
            the chromosome (up to ``max_retries`` times). If constraints
            still cannot be satisfied, it returns the best partial attempt.

        Used by
            ``GeneticOptimizer`` when initialising the starting population.
            Also used when a mutated chromosome violates constraints
            and needs to be regenerated.

        When you need this
            You don't call this directly — the optimizer calls it.
            However, you can use it to preview what chromosomes look like:
            ``print(space.random_chromosome())``

        Args:
            max_retries (int): Max attempts to satisfy constraints.

        Returns:
            dict: A chromosome mapping parameter names to sampled values.
        """
        for _ in range(max_retries):
            chrom = {}
            for name, param in self.params.items():
                if isinstance(param, IntParam):
                    chrom[name] = int(np.random.randint(param.min_val, param.max_val + 1))
                elif isinstance(param, FloatParam):
                    v = np.random.uniform(param.min_val, param.max_val)
                    chrom[name] = round(float(v), param.decimals)
                elif isinstance(param, ChoiceParam):
                    chrom[name] = np.random.choice(param.choices)
                else:
                    raise TypeError(f"Unknown param type for '{name}': {type(param)}")
            if self._valid(chrom):
                return chrom
        return chrom  # return last attempt even if invalid

    def _valid(self, chrom: dict) -> bool:
        """Return True if all constraints are satisfied."""
        return all(c(chrom) for c in self.constraints)

    def crossover(self, parent1: dict, parent2: dict) -> Tuple[dict, dict]:
        """
        Perform uniform crossover between two parent chromosomes.
        ----------------------------------------------------------
        What it does
            For each parameter, randomly assigns the value from parent1
            or parent2 to each child (with equal probability). Float
            parameters use blend crossover (BLX-0.5): a value is sampled
            from [min(p1,p2) - 0.5*range, max(p1,p2) + 0.5*range].
            Constraints are enforced after crossover.

        Used by
            ``GeneticOptimizer._next_generation()`` when building the
            next population from selected parents.

        When you need this
            You don't call this directly — the optimizer uses it
            internally during evolution.

        Args:
            parent1 (dict): First parent chromosome.
            parent2 (dict): Second parent chromosome.

        Returns:
            Tuple[dict, dict]: Two child chromosomes.
        """
        child1, child2 = {}, {}
        for name, param in self.params.items():
            v1, v2 = parent1[name], parent2[name]
            if isinstance(param, IntParam):
                if np.random.random() < 0.5:
                    child1[name], child2[name] = v1, v2
                else:
                    child1[name], child2[name] = v2, v1
            elif isinstance(param, FloatParam):
                # BLX-0.5 blend crossover
                lo, hi = min(v1, v2), max(v1, v2)
                span = hi - lo
                lo_b = max(param.min_val, lo - 0.5 * span)
                hi_b = min(param.max_val, hi + 0.5 * span)
                c1 = round(float(np.random.uniform(lo_b, hi_b)), param.decimals)
                c2 = round(float(np.random.uniform(lo_b, hi_b)), param.decimals)
                child1[name] = float(np.clip(c1, param.min_val, param.max_val))
                child2[name] = float(np.clip(c2, param.min_val, param.max_val))
            elif isinstance(param, ChoiceParam):
                if np.random.random() < 0.5:
                    child1[name], child2[name] = v1, v2
                else:
                    child1[name], child2[name] = v2, v1

        # Repair constraint violations
        for child in (child1, child2):
            if not self._valid(child):
                repaired = self._repair(child)
                child.update(repaired)
        return child1, child2

    def mutate(self, chrom: dict, rate: float = 0.15, sigma: float = 0.15) -> dict:
        """
        Mutate a chromosome by randomly perturbing its parameter values.
        -----------------------------------------------------------------
        What it does
            For each parameter, with probability ``rate``, applies a
            mutation:
              - IntParam:   adds Gaussian noise (scaled to range), clamps.
              - FloatParam: adds Gaussian noise (scaled to range), clamps.
              - ChoiceParam: randomly picks a new choice.
            Constraints are enforced after mutation.

        Used by
            ``GeneticOptimizer._next_generation()`` after crossover to
            introduce diversity and prevent premature convergence.

        When you need this
            You don't call this directly. Adjust ``GAConfig.mutation_rate``
            to control how aggressively the optimizer explores. Higher rate
            (0.2–0.3) explores more; lower rate (0.05–0.10) exploits more.
            Typical sweet spot: 0.10–0.20.

        Args:
            chrom (dict):  Chromosome to mutate (not modified in place).
            rate  (float): Probability of mutating each parameter (0–1).
            sigma (float): Mutation strength as fraction of range (0–1).

        Returns:
            dict: Mutated chromosome.
        """
        mutated = dict(chrom)
        for name, param in self.params.items():
            if np.random.random() > rate:
                continue  # no mutation for this param
            if isinstance(param, IntParam):
                span = max(1, int((param.max_val - param.min_val) * sigma))
                noise = np.random.randint(-span, span + 1)
                mutated[name] = int(np.clip(chrom[name] + noise, param.min_val, param.max_val))
            elif isinstance(param, FloatParam):
                span = (param.max_val - param.min_val) * sigma
                noise = np.random.normal(0, span)
                v = round(float(chrom[name] + noise), param.decimals)
                mutated[name] = round(float(np.clip(v, param.min_val, param.max_val)), param.decimals)
            elif isinstance(param, ChoiceParam):
                mutated[name] = np.random.choice(param.choices)

        if not self._valid(mutated):
            mutated.update(self._repair(mutated))
        return mutated

    def _repair(self, chrom: dict) -> dict:
        """
        Attempt to repair a chromosome that violates constraints.
        ---------------------------------------------------------
        What it does
            Tries up to 50 random resamplings of each violating parameter
            to find a valid chromosome. Used after crossover or mutation
            when the resulting chromosome violates the space constraints
            (e.g., MACD fast >= slow).

        When you need this
            Automatically called by ``crossover`` and ``mutate``. If your
            strategy has many constraints that are hard to satisfy by
            random resampling, consider restructuring your ParameterSpace
            so only the offset/difference is parameterised instead of
            absolute values (e.g., parameterise ``slow_minus_fast``
            instead of ``fast`` and ``slow`` separately).

        Args:
            chrom (dict): Chromosome that violates at least one constraint.

        Returns:
            dict: Repaired chromosome (best effort).
        """
        for _ in range(50):
            candidate = self.random_chromosome(max_retries=1)
            if self._valid(candidate):
                return candidate
        return chrom  # give up, return as-is


# ============================================================
# 2. DATA CONTAINERS
# ============================================================

@dataclass
class Individual:
    """
    A single member of the population — one set of parameters and its fitness.
    ---------------------------------------------------------------------------
    What it is
        A chromosome (parameter dict) combined with its evaluated fitness
        score and per-stock backtest results. This is the atomic unit of
        the genetic algorithm — the population is a list of Individuals.

    Used by
        ``GeneticOptimizer`` creates and manipulates these. The final
        ``OptimizationResult.best_individual`` is the Individual with the
        highest fitness across all generations.

    When you need this
        You don't create these directly. Inspect them in
        ``OptimizationResult.best_individual`` or the generation history
        to understand which parameters the optimizer converged to and
        why.

    Fields
        params   Parameter dict — the chromosome.
        fitness  Scalar fitness score (higher is better). -inf = unevaluated.
        per_stock_results  Dict of {symbol: BacktestResult} from the evaluation.
        generation  Which generation this individual was created in.
    """
    params: dict
    fitness: float = -math.inf
    per_stock_results: Dict[str, "BacktestResult"] = field(default_factory=dict)
    generation: int = 0


@dataclass
class GAConfig:
    """
    Hyperparameters controlling the genetic algorithm's behaviour.
    --------------------------------------------------------------
    What it is
        A dataclass with all settings that control how the GA evolves:
        population size, number of generations, mutation rate, elite
        fraction, and fitness metric. These are the "parameters of the
        parameter optimizer" — tune them to balance exploration vs
        exploitation and runtime vs quality.

    Used by
        Passed to ``GeneticOptimizer.__init__``. Stored and used
        throughout the run by ``GeneticOptimizer.run()``.

    When to adjust
        - Increase ``population_size`` for a larger, more diverse search
          (slower but more thorough). Good values: 30–80.
        - Increase ``n_generations`` to allow more refinement. Good: 20–50.
        - Lower ``mutation_rate`` (0.05) for fine-tuning near a known good
          solution; higher (0.25) for exploring a new, unfamiliar space.
        - Use ``fitness_metric="robust"`` to find parameters that generalise
          across stocks rather than overfit to one instrument.
        - Reduce ``n_workers`` if your machine has fewer cores or if the
          threading overhead is too high.

    Code example
        >>> cfg = GAConfig(
        ...     population_size = 40,
        ...     n_generations   = 30,
        ...     mutation_rate   = 0.15,
        ...     fitness_metric  = "robust",
        ...     min_trades      = 8,
        ... )

    Fields
        population_size   Number of individuals per generation.
        n_generations     How many generations to evolve.
        mutation_rate     Probability of mutating each parameter (0–1).
        mutation_sigma    Mutation strength as fraction of range (0–1).
        crossover_rate    Probability that two parents mate vs clone (0–1).
        elite_fraction    Top fraction of population preserved unchanged.
        tournament_size   Candidates per tournament selection round.
        fitness_metric    "sharpe" | "calmar" | "composite" | "robust".
        consistency_penalty  Weight of the across-stock std penalty for
                             "robust" fitness (0 = ignore, 0.5 = strong penalty).
        min_trades        Discard individuals with fewer trades than this.
        n_workers         Parallel threads for evaluating the population.
        random_seed       Set for reproducible runs (None = random).
        years_of_data     Years of history to download if auto-fetching.
        verbose           Print per-generation progress (default True).
    """
    population_size:     int   = 40
    n_generations:       int   = 30
    mutation_rate:       float = 0.15
    mutation_sigma:      float = 0.15
    crossover_rate:      float = 0.80
    elite_fraction:      float = 0.10
    tournament_size:     int   = 4
    fitness_metric:      str   = "robust"  # "sharpe"|"calmar"|"composite"|"robust"
    consistency_penalty: float = 0.30
    min_trades:          int   = 5
    n_workers:           int   = 4
    random_seed:         Optional[int] = None
    years_of_data:       int   = 3
    verbose:             bool  = True


@dataclass
class OptimizationResult:
    """
    Complete output from a genetic optimization run.
    -------------------------------------------------
    What it is
        All information produced by ``GeneticOptimizer.run()``:
        the best parameters found, the fitness trajectory, how each
        parameter evolved over generations, and the best individual's
        per-stock backtest results.

    Used by
        ``print_optimization_report()`` to render the ASCII summary.
        ``save_result()`` / ``load_result()`` to persist and reload runs.
        You can inspect ``best_params`` directly to use the discovered
        parameters in a new live strategy.

    When you need this
        After any call to ``GeneticOptimizer.run()``. Key things to check:
          1. Did fitness improve consistently? (``fitness_history``)
          2. Did parameters converge? (``param_convergence``)
          3. Are the per-stock metrics consistent? (``best_individual``)
          4. Is the best Sharpe in an acceptable range? (>1.0)

    Fields
        best_params           The champion parameter dict.
        best_fitness          Fitness score of the best individual.
        best_individual       Full Individual object for the best solution.
        fitness_history       Best fitness per generation.
        avg_fitness_history   Mean fitness per generation.
        param_convergence     {param_name: [best value per generation]}.
        total_backtests       Total backtest calls made (pop_size × stocks × gens).
        runtime_seconds       Wall-clock time for the full run.
        config                The GAConfig used for this run.
        symbols               List of symbols used in evaluation.
        n_generations         Actual number of generations run.
    """
    best_params:          dict
    best_fitness:         float
    best_individual:      Individual
    fitness_history:      List[float]
    avg_fitness_history:  List[float]
    param_convergence:    Dict[str, List[float]]
    total_backtests:      int
    runtime_seconds:      float
    config:               GAConfig
    symbols:              List[str]
    n_generations:        int


# ============================================================
# 3. GENETIC OPTIMIZER
# ============================================================

class GeneticOptimizer:
    """
    Genetic algorithm that tunes strategy parameters across a stock basket.
    -----------------------------------------------------------------------
    What it is
        The main engine. Maintains a population of parameter chromosomes,
        evaluates each on every symbol in the basket, selects the best
        parents via tournament selection, breeds them with crossover and
        mutation, and repeats for ``n_generations``. Returns an
        ``OptimizationResult`` with the best parameters found.

    Used by
        Run once per strategy type you want to optimise. Use the returned
        ``best_params`` to configure your paper trading strategy.

    When you need this
        - When the default parameters for an indicator-based strategy
          are clearly sub-optimal on your target instruments.
        - When you have a strategy that has several free parameters and
          you want a principled way to set them.
        - After a significant market regime change — re-run the optimizer
          on recent data to see if new parameters are needed.
        - CAUTION: Always out-of-sample validate. Optimised parameters
          can overfit to the training period — test on a held-out date
          range before deploying.

    Code example
        >>> opt = GeneticOptimizer(
        ...     strategy_factory = TunableMACD,
        ...     param_space      = space,
        ...     symbols          = ["AAPL", "MSFT", "SPY", "QQQ", "TSLA"],
        ...     config           = GAConfig(population_size=30, n_generations=20),
        ... )
        >>> result = opt.run()
        >>> print_optimization_report(result)
        >>> print("Best params:", result.best_params)

    Overfitting warning
        The optimizer will find parameters that maximise fitness on the
        training data. Always validate on a held-out period:
            - Train on years 1-3
            - Validate on year 4 (out-of-sample)
        If out-of-sample performance drops sharply, the optimised
        parameters are likely overfit. Use ``fitness_metric="robust"`` and
        a diverse stock basket to reduce overfitting risk.
    """

    def __init__(
        self,
        strategy_factory: Callable[[dict], Strategy],
        param_space:       ParameterSpace,
        symbols:           List[str],
        config:            Optional[GAConfig]        = None,
        backtest_config:   Optional[BacktestConfig]  = None,
        price_data:        Optional[Dict[str, pd.DataFrame]] = None,
    ) -> None:
        """
        Initialise the optimizer.

        Args:
            strategy_factory (callable): Function/class that takes a params dict
                                         and returns a Strategy instance.
            param_space      (ParameterSpace): The search space definition.
            symbols          (list[str]):  Tickers to evaluate fitness on.
            config           (GAConfig):   GA hyperparameters. Uses defaults if None.
            backtest_config  (BacktestConfig): Backtest settings. Uses defaults if None.
            price_data       (dict, opt): Pre-loaded {symbol: df} OHLCV DataFrames.
                                         If None, data is downloaded automatically
                                         using yfinance (or synthetic fallback).
        """
        self.factory   = strategy_factory
        self.space     = param_space
        self.symbols   = symbols
        self.config    = config or GAConfig()
        self.bt_config = backtest_config or BacktestConfig()
        self._rng      = np.random.default_rng(self.config.random_seed)

        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)

        # Load price data
        self.price_data = price_data or self._load_all(symbols)

    # ── Data loading ──────────────────────────────────────────────────────────

    def _load_all(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Download or generate OHLCV data for all symbols in the basket.
        ---------------------------------------------------------------
        What it does
            Calls ``load_price_data()`` from ``run_backtest_example.py``
            for each symbol (tries yfinance first, then synthetic fallback).
            Caches results in ``self.price_data`` so they are only
            downloaded once per optimizer instance.

        Used by
            ``__init__`` when the caller does not supply ``price_data``.
            Re-downloading every time would be slow for large baskets.

        When you need this
            You don't call this directly. If you want to avoid repeated
            downloads across multiple optimization runs, pre-load the data
            and pass it as ``price_data={...}`` to the constructor.

        Code example
            >>> # Pre-load for multiple runs:
            >>> dfs = {sym: load_price_data(sym, years=3)[0]
            ...        for sym in ["AAPL", "MSFT", "SPY"]}
            >>> opt = GeneticOptimizer(..., price_data=dfs)

        Args:
            symbols (list[str]): Tickers to load.

        Returns:
            dict: {symbol: pd.DataFrame} of OHLCV data.
        """
        data = {}
        print(f"  Loading price data for {len(symbols)} symbols ...")
        for sym in symbols:
            try:
                df, label = load_price_data(sym, years=self.config.years_of_data)
                data[sym] = df
            except Exception as exc:
                print(f"  WARNING: Could not load {sym}: {exc}. Skipping.")
        return data

    # ── Fitness evaluation ────────────────────────────────────────────────────

    def _evaluate_individual(self, ind: Individual) -> Individual:
        """
        Evaluate one individual by backtesting it on all symbols.
        ----------------------------------------------------------
        What it does
            Creates a strategy instance from the individual's params,
            runs ``run_backtest()`` on every symbol, and computes a
            single scalar fitness score using ``_compute_fitness()``.
            Updates ``ind.fitness`` and ``ind.per_stock_results`` in place.

        Used by
            ``_evaluate_population()`` which calls this in parallel using
            a thread pool.

        When you need this
            You don't call this directly. It is the inner loop of the GA.

        Args:
            ind (Individual): Individual to evaluate (mutated in place).

        Returns:
            Individual: Same object with ``fitness`` and ``per_stock_results``
                        populated.
        """
        results = {}
        try:
            strategy = self.factory(ind.params)
        except Exception:
            ind.fitness = -math.inf
            return ind

        for sym, df in self.price_data.items():
            if df is None or len(df) < 60:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    r = run_backtest(strategy, df.copy(), self.bt_config, symbol=sym)
                results[sym] = r
            except Exception:
                pass  # skip failed backtests silently

        ind.per_stock_results = results
        ind.fitness = self._compute_fitness(results)
        return ind

    def _compute_fitness(self, results: Dict[str, BacktestResult]) -> float:
        """
        Aggregate per-stock backtest results into a single fitness scalar.
        -------------------------------------------------------------------
        What it does
            Computes one of four fitness metrics (controlled by
            ``GAConfig.fitness_metric``):
              - "sharpe":    Mean Sharpe ratio across all stocks.
              - "calmar":    Mean Calmar ratio.
              - "composite": 0.4×Sharpe + 0.3×Calmar + 0.3×WinRate_adj.
              - "robust":    Mean Sharpe − consistency_penalty × std(Sharpe).
                             This is the recommended default: it rewards
                             strategies that perform consistently across all
                             stocks, not just those that get lucky on one.

            Individuals with fewer than ``min_trades`` total trades across
            all stocks receive a fitness of -inf.

        Used by
            ``_evaluate_individual()`` after all backtests are complete.

        When you need this
            You don't call this directly. Switch between metrics via
            ``GAConfig.fitness_metric``. The "robust" metric is best for
            finding parameters that generalise. Use "sharpe" for a
            pure risk-adjusted return focus.

        Args:
            results (dict): {symbol: BacktestResult} from all backtests.

        Returns:
            float: Scalar fitness (higher is better, -inf = invalid).
        """
        if not results:
            return -math.inf

        sharpes, calmars, win_rates, trade_counts = [], [], [], []
        for r in results.values():
            if math.isnan(r.sharpe) or math.isnan(r.calmar):
                continue
            trade_counts.append(r.total_trades)
            sharpes.append(r.sharpe)
            calmars.append(r.calmar)
            win_rates.append(r.win_rate_val if not math.isnan(r.win_rate_val) else 0.5)

        if not sharpes:
            return -math.inf

        # Discard individuals that barely trade
        total_trades = sum(trade_counts)
        if total_trades < self.config.min_trades * len(results):
            return -math.inf

        mean_sharpe  = float(np.mean(sharpes))
        mean_calmar  = float(np.mean(np.clip(calmars, -5, 10)))
        mean_winrate = float(np.mean(win_rates))
        std_sharpe   = float(np.std(sharpes)) if len(sharpes) > 1 else 0.0

        metric = self.config.fitness_metric.lower()
        if metric == "sharpe":
            return mean_sharpe
        elif metric == "calmar":
            return mean_calmar
        elif metric == "composite":
            wr_adj = (mean_winrate - 0.5) * 2  # normalise 0-100% → -1 to +1
            return 0.4 * mean_sharpe + 0.3 * mean_calmar + 0.3 * wr_adj
        elif metric == "robust":
            return mean_sharpe - self.config.consistency_penalty * std_sharpe
        else:
            return mean_sharpe

    def _evaluate_population(self, population: List[Individual]) -> List[Individual]:
        """
        Evaluate all individuals in the population in parallel.
        --------------------------------------------------------
        What it does
            Uses a ``ThreadPoolExecutor`` with ``GAConfig.n_workers`` threads
            to call ``_evaluate_individual()`` on every individual
            concurrently. Numpy releases the GIL during computation, so
            threading gives a real speedup here.

        Used by
            ``run()`` at the start of each generation.

        When you need this
            You don't call this directly. Tune ``GAConfig.n_workers`` to
            match the number of CPU cores on your machine. Typical speedup:
            4 workers gives ~3–4× faster per generation.

        Args:
            population (list[Individual]): All individuals to evaluate.

        Returns:
            list[Individual]: Same list with fitness values populated.
        """
        unevaluated = [ind for ind in population if ind.fitness == -math.inf]
        if not unevaluated:
            return population

        with ThreadPoolExecutor(max_workers=self.config.n_workers) as executor:
            futures = {executor.submit(self._evaluate_individual, ind): ind
                       for ind in unevaluated}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception:
                    pass
        return population

    # ── Selection ─────────────────────────────────────────────────────────────

    def _tournament_select(self, population: List[Individual]) -> Individual:
        """
        Select one parent via tournament selection.
        --------------------------------------------
        What it does
            Randomly picks ``tournament_size`` individuals from the
            population and returns the one with the highest fitness.
            This is the most common selection strategy in genetic
            algorithms — it provides selective pressure while maintaining
            diversity (unlike pure elitism which can converge too fast).

        Used by
            ``_next_generation()`` twice per pair of offspring.

        When you need this
            You don't call this directly. The selection pressure is
            controlled by ``GAConfig.tournament_size``: larger values
            (6–8) give higher pressure and faster convergence but risk
            premature convergence; smaller values (2–3) maintain more
            diversity.

        Args:
            population (list[Individual]): Current generation.

        Returns:
            Individual: The winner of the tournament.
        """
        k = min(self.config.tournament_size, len(population))
        contenders = np.random.choice(population, size=k, replace=False)
        return max(contenders, key=lambda ind: ind.fitness)

    # ── Reproduction ──────────────────────────────────────────────────────────

    def _next_generation(
        self, population: List[Individual], generation: int
    ) -> List[Individual]:
        """
        Build the next generation from the current one.
        ------------------------------------------------
        What it does
            Implements the standard generational GA loop:
              1. Elitism: copy the top ``elite_fraction`` directly.
              2. Fill remaining slots by:
                 a. Select two parents via tournament.
                 b. With probability ``crossover_rate``: crossover → 2 children.
                 c. Otherwise: clone both parents.
                 d. Mutate each child with probability ``mutation_rate``.
              3. Return new population (all unevaluated except elites).

        Used by
            ``run()`` at the end of each generation.

        When you need this
            You don't call this directly. The interplay between elite
            fraction, crossover rate, and mutation rate controls how the
            population evolves. Elite fraction 0.10–0.15 is typical.

        Args:
            population (list[Individual]): Current evaluated population.
            generation (int):              Current generation index (for logging).

        Returns:
            list[Individual]: New population (unevaluated individuals).
        """
        n = len(population)
        n_elite = max(1, int(n * self.config.elite_fraction))

        # Sort by fitness descending
        ranked = sorted(population, key=lambda x: x.fitness, reverse=True)

        # Elites carry forward unchanged
        next_gen = [Individual(params=dict(ind.params),
                               fitness=ind.fitness,
                               per_stock_results=ind.per_stock_results,
                               generation=generation)
                    for ind in ranked[:n_elite]]

        # Breed the rest
        while len(next_gen) < n:
            parent1 = self._tournament_select(ranked)
            parent2 = self._tournament_select(ranked)

            if np.random.random() < self.config.crossover_rate:
                c1_params, c2_params = self.space.crossover(parent1.params, parent2.params)
            else:
                c1_params, c2_params = dict(parent1.params), dict(parent2.params)

            c1_params = self.space.mutate(c1_params, self.config.mutation_rate,
                                          self.config.mutation_sigma)
            c2_params = self.space.mutate(c2_params, self.config.mutation_rate,
                                          self.config.mutation_sigma)

            next_gen.append(Individual(params=c1_params, generation=generation))
            if len(next_gen) < n:
                next_gen.append(Individual(params=c2_params, generation=generation))

        return next_gen[:n]

    # ── Main run loop ─────────────────────────────────────────────────────────

    def run(self) -> OptimizationResult:
        """
        Execute the full genetic optimization run.
        -------------------------------------------
        What it does
            1. Creates a random initial population.
            2. Evaluates every individual on all symbols.
            3. For each generation:
               a. Prints a progress line with best/avg fitness.
               b. Breeds the next generation.
               c. Evaluates the new individuals.
               d. Records fitness statistics.
            4. Returns an ``OptimizationResult`` with all findings.

        Used by
            The primary entry point. Call this once and inspect the result.

        When you need this
            Run this on your target stock basket with 2–5 years of data.
            A typical run (30 individuals, 20 generations, 5 stocks) takes
            2–8 minutes depending on CPU speed and data size.

        How to save and reuse results
            >>> result = opt.run()
            >>> save_result(result, "aapl_macd_opt.json")
            >>> # Later:
            >>> result = load_result("aapl_macd_opt.json")

        Returns:
            OptimizationResult: All statistics, best params, and per-stock
                                results for the best individual.
        """
        cfg = self.config
        n_symbols = len(self.price_data)
        total_backtests = 0
        t_start = time.time()

        if cfg.verbose:
            print(f"\n  {'═' * 62}")
            print(f"  GENETIC OPTIMIZER")
            print(f"  Strategy factory : {self.factory.__name__}")
            print(f"  Symbols          : {self.symbols}")
            print(f"  Population       : {cfg.population_size}  "
                  f"Generations: {cfg.n_generations}")
            print(f"  Fitness metric   : {cfg.fitness_metric}  "
                  f"Workers: {cfg.n_workers}")
            print(f"  Parameters       : {list(self.space.params.keys())}")
            print(f"  {'═' * 62}\n")

        # Initialise population
        population = [
            Individual(params=self.space.random_chromosome(), generation=0)
            for _ in range(cfg.population_size)
        ]

        fitness_history: List[float]       = []
        avg_fitness_history: List[float]   = []
        param_convergence: Dict[str, List[float]] = {k: [] for k in self.space.params}

        best_ever: Optional[Individual] = None

        for gen in range(cfg.n_generations):
            # Evaluate unevaluated individuals
            population = self._evaluate_population(population)
            total_backtests += cfg.population_size * n_symbols

            # Sort
            ranked = sorted(population, key=lambda x: x.fitness, reverse=True)
            valid  = [ind for ind in ranked if ind.fitness > -math.inf]

            best_fit  = ranked[0].fitness if ranked else -math.inf
            valid_fits = [ind.fitness for ind in valid]
            avg_fit   = float(np.mean(valid_fits)) if valid_fits else -math.inf

            fitness_history.append(best_fit)
            avg_fitness_history.append(avg_fit)

            # Track parameter evolution (best individual each gen)
            best_ind = ranked[0] if ranked else None
            if best_ever is None or (best_ind and best_ind.fitness > best_ever.fitness):
                best_ever = best_ind
            for pname in self.space.params:
                if best_ind:
                    param_convergence[pname].append(best_ind.params.get(pname, float("nan")))

            if cfg.verbose:
                self._print_generation_line(
                    gen + 1, cfg.n_generations,
                    best_fit, avg_fit,
                    best_ind.params if best_ind else {},
                    valid_fits,
                )

            # Evolve (skip on last generation)
            if gen < cfg.n_generations - 1:
                population = self._next_generation(population, gen + 1)

        runtime = time.time() - t_start

        if cfg.verbose:
            print()

        return OptimizationResult(
            best_params         = dict(best_ever.params) if best_ever else {},
            best_fitness        = best_ever.fitness if best_ever else -math.inf,
            best_individual     = best_ever,
            fitness_history     = fitness_history,
            avg_fitness_history = avg_fitness_history,
            param_convergence   = param_convergence,
            total_backtests     = total_backtests,
            runtime_seconds     = runtime,
            config              = cfg,
            symbols             = list(self.price_data.keys()),
            n_generations       = cfg.n_generations,
        )

    # ── Progress output ───────────────────────────────────────────────────────

    def _print_generation_line(
        self,
        gen:       int,
        total:     int,
        best_fit:  float,
        avg_fit:   float,
        best_params: dict,
        all_fits:  List[float],
    ) -> None:
        """
        Print a single-line progress update for one generation.
        --------------------------------------------------------
        What it does
            Renders a compact line showing the generation number, best and
            average fitness, a mini ASCII progress bar proportional to the
            best fitness value, and a summary of the best individual's
            key parameters.

        Used by
            ``run()`` after each generation is evaluated.

        When you need this
            Always printed unless ``GAConfig.verbose=False``. If you are
            running the optimizer inside a notebook or larger script and
            don't want terminal noise, set ``verbose=False`` and inspect
            ``result.fitness_history`` programmatically instead.
        """
        bar_w  = 20
        # Normalise best_fit to a 0–1 bar (clip between -2 and 3)
        frac   = max(0.0, min(1.0, (best_fit + 2) / 5))
        filled = int(frac * bar_w)
        bar    = "▓" * filled + "░" * (bar_w - filled)

        # Summarise top-3 params
        param_str = "  ".join(
            f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in list(best_params.items())[:3]
        )

        best_s = f"{best_fit:+.3f}" if best_fit > -math.inf else "  n/a "
        avg_s  = f"{avg_fit:+.3f}" if avg_fit  > -math.inf else "  n/a "

        print(
            f"  Gen {gen:>3}/{total}  │{bar}│  "
            f"best={best_s}  avg={avg_s}  │  {param_str}"
        )


# ============================================================
# 4. PERSISTENCE
# ============================================================

def save_result(result: OptimizationResult, path: str) -> None:
    """
    Save an OptimizationResult to a JSON file.
    -------------------------------------------
    What it does
        Serialises the key fields of an ``OptimizationResult`` to a
        human-readable JSON file. Does not save the full per-stock
        backtest results (too large) — only the best_params, fitness
        history, parameter convergence, and config.

    Used by
        After any completed optimization run when you want to preserve
        the results for future reference or to compare across runs.

    When you need this
        - After a long optimization run (save immediately after run!).
        - When comparing two optimizer configurations (save both results
          and inspect the JSON files side by side).
        - When sharing results with a collaborator.
        - Before changing the parameter space or stock basket, save the
          current best result so you can roll back if the new run is worse.

    Code example
        >>> result = opt.run()
        >>> save_result(result, "runs/macd_opt_v1.json")
        >>> # Later:
        >>> old_result = load_result("runs/macd_opt_v1.json")

    Args:
        result (OptimizationResult): Output from GeneticOptimizer.run().
        path   (str): File path for the JSON file.
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    data = {
        "best_params":          result.best_params,
        "best_fitness":         result.best_fitness,
        "fitness_history":      result.fitness_history,
        "avg_fitness_history":  result.avg_fitness_history,
        "param_convergence":    {k: [float(v) if isinstance(v, (int, float)) else str(v)
                                     for v in vals]
                                 for k, vals in result.param_convergence.items()},
        "total_backtests":      result.total_backtests,
        "runtime_seconds":      result.runtime_seconds,
        "symbols":              result.symbols,
        "n_generations":        result.n_generations,
        "config": {
            "population_size": result.config.population_size,
            "n_generations":   result.config.n_generations,
            "fitness_metric":  result.config.fitness_metric,
            "mutation_rate":   result.config.mutation_rate,
        },
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Result saved → {path}")


def load_result(path: str) -> dict:
    """
    Load a previously saved optimization result from a JSON file.
    --------------------------------------------------------------
    What it does
        Reads the JSON file written by ``save_result()`` and returns it
        as a plain dict (not an ``OptimizationResult`` object, since the
        full backtest results were not saved). The dict contains
        ``best_params``, ``fitness_history``, ``param_convergence``, etc.

    Used by
        When you want to review past optimization runs or re-use the
        best parameters from a previous run without re-running the
        full optimization.

    When you need this
        - When starting a new paper trading session using previously
          optimised parameters.
        - When auditing past optimization runs to see if parameters
          have drifted over time.

    Code example
        >>> data = load_result("runs/macd_opt_v1.json")
        >>> best_params = data["best_params"]
        >>> strategy = TunableMACD(best_params)

    Args:
        path (str): Path to a JSON file written by save_result().

    Returns:
        dict: The saved optimization data.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    with open(path, "r") as f:
        return json.load(f)


# ============================================================
# 5. ASCII REPORT
# ============================================================

def _ascii_line_simple(
    values: List[float],
    title:  str,
    width:  int = 60,
    height: int = 8,
    y_fmt: Callable = lambda v: f"{v:+.2f}",
) -> str:
    """
    Render a simple list of floats as an ASCII line chart (no DatetimeIndex).
    --------------------------------------------------------------------------
    What it does
        A lightweight version of ``ascii_line_chart()`` designed for
        plotting fitness histories or parameter evolution curves, where the
        x-axis is just an integer generation index rather than dates.

    Used by
        ``_ascii_fitness_chart()`` and ``_ascii_param_chart()`` inside the
        optimization report.

    When you need this
        - In the optimization report's fitness convergence chart.
        - When plotting any integer-indexed series in a terminal.

    Args:
        values (list[float]): Y values to plot.
        title  (str):         Title line.
        width  (int):         Chart width in characters.
        height (int):         Chart height in characters.
        y_fmt  (callable):    Format function for Y-axis labels.

    Returns:
        str: Multi-line ASCII chart string.
    """
    if not values or len(values) < 2:
        return f"  {title}\n  (no data)\n"

    arr   = [v for v in values if v > -math.inf]
    if not arr:
        return f"  {title}\n  (no valid data)\n"

    y_min, y_max = min(arr), max(arr)
    y_range = y_max - y_min if y_max != y_min else 1.0
    label_w = max(len(y_fmt(y_min)), len(y_fmt(y_max)))

    grid = [[" "] * width for _ in range(height)]
    prev_row = None
    for col in range(width):
        idx = int(col * (len(arr) - 1) / max(width - 1, 1))
        idx = min(idx, len(arr) - 1)
        row = height - 1 - int((arr[idx] - y_min) / y_range * (height - 1))
        row = max(0, min(height - 1, row))
        grid[row][col] = "▪"
        if prev_row is not None and abs(prev_row - row) > 1:
            for r in range(min(prev_row, row) + 1, max(prev_row, row)):
                grid[r][col] = "│"
        prev_row = row

    lines = [f"  {title}"]
    for r in range(height):
        if r == 0:
            lbl = y_fmt(y_max)
        elif r == height - 1:
            lbl = y_fmt(y_min)
        elif r == height // 2:
            lbl = y_fmt(y_min + y_range / 2)
        else:
            lbl = " " * label_w
        bar = "│" if r < height - 1 else "└"
        lines.append(f"  {lbl:>{label_w}} {bar}{''.join(grid[r])}")

    lines.append(f"  {' ' * label_w}  {'─' * width}")
    n = len(arr)
    lines.append(f"  {' ' * (label_w + 2)}Gen 1{' ' * (width // 2 - 8)}"
                 f"Gen {n // 2}{' ' * (width // 2 - 6)}Gen {n}")
    return "\n".join(lines)


def _ascii_param_table(result: OptimizationResult) -> str:
    """
    Render a table showing how each parameter evolved over generations.
    -------------------------------------------------------------------
    What it does
        Builds a grid showing the best individual's parameter values at
        evenly-spaced generation checkpoints (up to 8 columns). Lets you
        see which parameters converged quickly and which kept changing.

    Used by
        ``print_optimization_report()`` as the final section.

    When you need this
        - If a parameter converges to a boundary (min or max), that
          suggests the parameter range in ParameterSpace is too narrow
          and should be expanded.
        - If a parameter never converges (oscillates throughout), it may
          not be very influential — consider removing it to reduce the
          search space.
        - If all parameters converge quickly (by gen 5), the population
          may be too small or mutation_rate too low.

    Args:
        result (OptimizationResult): Output from GeneticOptimizer.run().

    Returns:
        str: Multi-line ASCII table string.
    """
    conv  = result.param_convergence
    n_gen = result.n_generations
    if not conv or not any(conv.values()):
        return "  Parameter Convergence\n  (no data)\n"

    # Pick up to 8 checkpoint generations
    checkpoints = list(range(0, n_gen, max(1, n_gen // 7)))
    if (n_gen - 1) not in checkpoints:
        checkpoints.append(n_gen - 1)
    checkpoints = checkpoints[:8]

    col_w  = 8
    header = f"  {'Parameter':<16}" + "".join(f"  G{cp+1:>4}" for cp in checkpoints)
    sep    = f"  {'─' * (16 + (col_w + 2) * len(checkpoints))}"
    lines  = [f"  Parameter Convergence (best individual's value per generation)", sep, header, sep]

    for pname, vals in conv.items():
        if not vals:
            continue
        row = f"  {pname:<16}"
        for cp in checkpoints:
            idx = min(cp, len(vals) - 1)
            v   = vals[idx]
            if isinstance(v, float):
                row += f"  {v:>{col_w}.2f}"
            elif isinstance(v, int):
                row += f"  {v:>{col_w}d}"
            else:
                row += f"  {str(v):>{col_w}}"
        lines.append(row)

    lines.append(sep)
    return "\n".join(lines)


def _ascii_per_stock_table(result: OptimizationResult) -> str:
    """
    Render the best individual's per-stock performance as a table.
    --------------------------------------------------------------
    What it does
        Shows Sharpe, Calmar, Max Drawdown, CAGR, and number of trades
        for the best individual on each symbol in the basket. This is
        critical for assessing whether the optimised parameters
        genuinely generalise.

    Used by
        ``print_optimization_report()`` as the third section.

    When you need this
        - If some stocks show great Sharpe (>2.0) and others are flat
          (<0.5), the parameters are probably overfit to the best stocks.
          Increase ``consistency_penalty`` or add more diverse symbols.
        - If all stocks show negative Sharpe, the strategy fundamentally
          doesn't work on this data — try a different strategy class or
          time period.

    Args:
        result (OptimizationResult): Output from GeneticOptimizer.run().

    Returns:
        str: Multi-line ASCII table string.
    """
    ind = result.best_individual
    if not ind or not ind.per_stock_results:
        return "  Per-Stock Results\n  (no data)\n"

    hdr = (f"  {'Symbol':>8}  {'Sharpe':>7}  {'Calmar':>7}  "
           f"{'MaxDD':>7}  {'CAGR':>7}  {'Trades':>7}  {'WinRate':>8}")
    sep = f"  {'─' * (len(hdr) - 2)}"
    lines = ["  Best Individual — Per-Stock Results", sep, hdr, sep]

    sharpes = []
    for sym, r in sorted(ind.per_stock_results.items()):
        sh  = f"{r.sharpe:+.2f}"   if not math.isnan(r.sharpe)    else "  n/a"
        ca  = f"{r.calmar:+.2f}"   if not math.isnan(r.calmar)    else "  n/a"
        dd  = f"{r.max_dd:.1%}"    if not math.isnan(r.max_dd)    else "  n/a"
        cg  = f"{r.cagr_val:.1%}"  if not math.isnan(r.cagr_val)  else "  n/a"
        wr  = f"{r.win_rate_val:.1%}" if not math.isnan(r.win_rate_val) else "  n/a"
        lines.append(f"  {sym:>8}  {sh:>7}  {ca:>7}  {dd:>7}  {cg:>7}  "
                     f"{r.total_trades:>7d}  {wr:>8}")
        if not math.isnan(r.sharpe):
            sharpes.append(r.sharpe)

    lines.append(sep)
    if sharpes:
        lines.append(
            f"  {'MEAN':>8}  {np.mean(sharpes):>+7.2f}  {'':>7}  "
            f"{'':>7}  {'':>7}  {'':>7}  {'':>8}"
        )
        lines.append(
            f"  {'STD':>8}  {np.std(sharpes):>7.2f}  "
            f"{'(lower std = more consistent across stocks)':>}"
        )
    return "\n".join(lines)


def print_optimization_report(result: OptimizationResult) -> None:
    """
    Print the complete genetic optimization ASCII report.
    ------------------------------------------------------
    What it does
        Renders a multi-section terminal report covering:
          1. Summary header (runtime, total backtests, best fitness)
          2. Best parameters found
          3. Per-stock performance of the best individual
          4. Fitness convergence chart (best and average per generation)
          5. Parameter evolution table
          6. Interpretation guide

    Used by
        Call this after ``GeneticOptimizer.run()`` to review the full
        results in the terminal. Also called by the ``__main__`` example.

    When you need this
        Always call this after every optimization run before using the
        best_params. The report will tell you:
          - Did the optimization converge? (fitness chart flattening)
          - Are the per-stock results consistent? (important for robustness)
          - Did any parameter hit a boundary? (suggests range is too narrow)

    Code example
        >>> result = opt.run()
        >>> print_optimization_report(result)
        >>> # Then use the best params:
        >>> strategy = TunableMACD(result.best_params)

    Args:
        result (OptimizationResult): Output from GeneticOptimizer.run().
    """
    W = 74
    print(f"\n{'═' * W}")
    print(f"  GENETIC OPTIMIZATION COMPLETE")
    print(f"  Symbols      : {result.symbols}")
    print(f"  Generations  : {result.n_generations}  "
          f"Population: {result.config.population_size}  "
          f"Metric: {result.config.fitness_metric}")
    print(f"  Total tests  : {result.total_backtests:,}  "
          f"Runtime: {result.runtime_seconds:.1f}s")
    print(f"{'═' * W}")

    # Best params box
    print(f"\n┌{'─' * (W - 2)}┐")
    print(f"│{'  BEST PARAMETERS FOUND':^{W - 2}}│")
    print(f"├{'─' * (W - 2)}┤")
    for k, v in result.best_params.items():
        fmt_v = f"{v:.4f}" if isinstance(v, float) else str(v)
        print(f"│  {k:<28} {fmt_v:<{W - 34}}  │")
    print(f"│{'':^{W - 2}}│")
    print(f"│  {'Best Fitness Score':<28} {result.best_fitness:+.4f}"
          f"{'':<{W - 42}}  │")
    print(f"└{'─' * (W - 2)}┘")

    # Per-stock results
    print(f"\n{'─' * W}")
    print(_ascii_per_stock_table(result))

    # Fitness convergence chart
    print(f"\n{'─' * W}")
    print(_ascii_line_simple(
        result.fitness_history,
        title="Fitness Convergence (best per generation)",
        width=60, height=8,
        y_fmt=lambda v: f"{v:+.2f}",
    ))
    print()
    print(_ascii_line_simple(
        result.avg_fitness_history,
        title="Average Fitness per Generation",
        width=60, height=6,
        y_fmt=lambda v: f"{v:+.2f}",
    ))

    # Parameter convergence table
    print(f"\n{'─' * W}")
    print(_ascii_param_table(result))

    # Guide
    print(f"\n{'─' * W}")
    print("  HOW TO USE THESE RESULTS")
    print(f"  {'─' * (W - 4)}")
    print("  1. Check per-stock Sharpe STD: low (<0.5) = robust, high = overfits")
    print("  2. Fitness converging flat? → run more generations or widen param ranges")
    print("  3. Param at boundary? → expand IntParam/FloatParam range and re-run")
    print("  4. ALWAYS validate on a held-out time period before live trading")
    print(f"  {'─' * (W - 4)}")
    print(f"  Best params → strategy_factory(result.best_params) to create strategy")
    print(f"{'─' * W}\n")


# ============================================================
# 6. STANDALONE EXAMPLE
# ============================================================

def _make_example_strategy_factory():
    """
    Build and return a tunable MACD+ADX strategy factory for the demo.
    -------------------------------------------------------------------
    What it does
        Defines a ``TunableMACDStrategy`` class whose ``__init__`` accepts
        a params dict, then returns the class itself as the factory.
        This is the pattern the optimizer expects: a callable that takes
        a params dict and returns a ``Strategy`` instance.

    Used by
        The ``if __name__ == "__main__"`` demo block. Copy this pattern
        when creating your own tunable strategies.

    When you need this
        - As a template when writing your first tunable strategy.
        - To understand the factory pattern before building your own.

    Returns:
        type: The ``TunableMACDStrategy`` class (usable as a factory).
    """
    from strategies.tools.momentum_tools import macd, adx, crossover as _crossover

    class TunableMACDStrategy(Strategy):
        """
        Parameterised MACD + ADX strategy for genetic optimisation.
        Accepts a params dict with keys: fast, slow, adx_period, adx_min.
        """
        def __init__(self, params: dict) -> None:
            self._params = params

        @property
        def name(self) -> str:
            p = self._params
            return (f"MACD(f={p['fast']},s={p['slow']},"
                    f"adx>={p['adx_min']:.0f})")

        @property
        def params(self) -> dict:
            return self._params

        def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
            p = self._params
            fast = max(2, int(p["fast"]))
            slow = max(fast + 2, int(p["slow"]))
            adx_period = max(5, int(p.get("adx_period", 14)))
            adx_min = float(p["adx_min"])

            m, sig, _ = macd(df["Close"], window_fast=fast, window_slow=slow)
            adx_v, _, _ = adx(df["High"], df["Low"], df["Close"], window=adx_period)

            df["signal"] = 0
            df.loc[_crossover(m, sig) & (adx_v > adx_min), "signal"] = 1
            df.loc[m < sig, "signal"] = 0
            df["signal"] = df["signal"].ffill().fillna(0)
            return df

    return TunableMACDStrategy


def main() -> None:
    """
    Run the genetic optimizer demo on a real stock basket.
    -------------------------------------------------------
    What it does
        Defines a ``TunableMACDStrategy`` with four tunable parameters
        (MACD fast/slow windows, ADX period, ADX threshold), sets up a
        ``ParameterSpace`` with realistic bounds, downloads 3 years of
        daily data for a diversified basket of 5 stocks, runs the GA for
        20 generations with 25 individuals, and prints the full report.

    Run it
        python momentum/test/genetic_optimizer.py

    Faster test (smaller run)
        python momentum/test/genetic_optimizer.py --fast

    Expected runtime
        ~3–6 minutes on a modern machine (25 pop × 5 symbols × 20 gens = 2,500 backtests).
        Reduce population_size or n_generations in GAConfig for quicker tests.
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true",
                        help="Use smaller population/generations for a quick test")
    args = parser.parse_args()

    factory = _make_example_strategy_factory()

    space = ParameterSpace(
        params={
            "fast":       IntParam(5,  20),
            "slow":       IntParam(15, 60),
            "adx_period": IntParam(7,  28),
            "adx_min":    FloatParam(10.0, 45.0),
        },
        constraints=[lambda p: p["fast"] < p["slow"] - 4],
    )

    cfg = GAConfig(
        population_size  = 10 if args.fast else 25,
        n_generations    = 5  if args.fast else 20,
        mutation_rate    = 0.15,
        fitness_metric   = "robust",
        consistency_penalty = 0.30,
        min_trades       = 3,
        n_workers        = 4,
        verbose          = True,
        years_of_data    = 2 if args.fast else 3,
    )

    # Diversified basket: large-cap tech, index ETF, financials, defensive
    symbols = ["AAPL", "SPY", "JPM", "XOM"] if args.fast else [
        "AAPL", "SPY", "QQQ", "JPM", "XOM"
    ]

    opt = GeneticOptimizer(
        strategy_factory = factory,
        param_space      = space,
        symbols          = symbols,
        config           = cfg,
    )

    result = opt.run()
    print_optimization_report(result)

    # Save results
    os.makedirs("momentum/test/runs", exist_ok=True)
    save_result(result, "momentum/test/runs/macd_opt_latest.json")

    print(f"\n  To use best params in a strategy:")
    print(f"  >>> factory = _make_example_strategy_factory()")
    print(f"  >>> strategy = factory({result.best_params})")


if __name__ == "__main__":
    main()
