import myIBApp
import myBreakoutSignal
import time
import random
import argparse
from backtestEngine import calculate_fitness, run_backtest, TICKER, BAR_COUNT, WHAT_TO_SHOW, USE_RTH, FORMAT_DATE, KEEP_UP_TO_DATE, EXTRA_PARAMS, TIMEOUT

# --- Genetic Algorithm ---
def genetic_optimize(
    app,
    ticker=TICKER,
    bar_count=BAR_COUNT,
    what_to_show=WHAT_TO_SHOW,
    use_rth=USE_RTH,
    format_date=FORMAT_DATE,
    keep_up_to_date=KEEP_UP_TO_DATE,
    extra_params=EXTRA_PARAMS,
    timeout=TIMEOUT,
    generations=10,
    population_size=8
):
    duration_choices = ["1 D", "5 D", "1 W", "1 M", "3 M", "6 M", "1 Y"]
    bar_size_choices = ["1 min", "5 mins", "15 mins", "1 h", "1 D", "1 W", "1 M"]
    population = [
        (random.choice(duration_choices), random.choice(bar_size_choices))
        for _ in range(population_size)
    ]
    best = None
    best_fitness = float('-inf')
    for gen in range(generations):
        print(f"\n--- Generation {gen+1} ---")
        scores = []
        for indiv in population:
            duration_str, bar_size_setting = indiv
            try:
                _, fitness = run_backtest(
                    app,
                    ticker=ticker,
                    bar_count=bar_count,
                    duration_str=duration_str,
                    bar_size_setting=bar_size_setting,
                    what_to_show=what_to_show,
                    use_rth=use_rth,
                    format_date=format_date,
                    keep_up_to_date=keep_up_to_date,
                    extra_params=extra_params,
                    timeout=timeout
                )
            except Exception as e:
                print(f"Error for {indiv}: {e}")
                fitness = float('-inf')
            scores.append((fitness, indiv))
            if fitness > best_fitness:
                best_fitness = fitness
                best = indiv
        scores.sort(reverse=True)
        # Elitism: keep top 2
        new_population = [indiv for _, indiv in scores[:2]]
        # Crossover and mutation
        while len(new_population) < population_size:
            parent1 = random.choice(scores[:4])[1]
            parent2 = random.choice(scores[:4])[1]
            child = (
                random.choice([parent1[0], parent2[0]]),
                random.choice([parent1[1], parent2[1]])
            )
            # Mutation
            if random.random() < 0.2:
                child = (
                    random.choice(duration_choices),
                    child[1]
                )
            if random.random() < 0.2:
                child = (
                    child[0],
                    random.choice(bar_size_choices)
                )
            new_population.append(child)
        population = new_population
    print(f"\n--- Best Parameters ---")
    print(f"DURATION_STR: {best[0]}, BAR_SIZE_SETTING: {best[1]}, Fitness: {best_fitness:.4f}")
    return best, best_fitness

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genetic Optimization for Backtest Engine")
    parser.add_argument('--generations', type=int, default=5, help='Number of generations')
    parser.add_argument('--population_size', type=int, default=6, help='Population size')
    parser.add_argument('--ticker', type=str, default=TICKER, help='Ticker symbol')
    parser.add_argument('--bar_count', type=int, default=BAR_COUNT, help='Lookback period for breakout')
    parser.add_argument('--timeout', type=int, default=TIMEOUT, help='Seconds to wait for data')
    args = parser.parse_args()
    app = myIBApp.connect_to_tws()
    genetic_optimize(
        app,
        ticker=args.ticker,
        bar_count=args.bar_count,
        timeout=args.timeout,
        generations=args.generations,
        population_size=args.population_size
    )
    myIBApp.disconnect_tws()
