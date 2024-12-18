import ManualStrategy as ms
import StrategyLearner as sl
import experiment1
import experiment2
from marketsimcode import compute_portvals
import util as ut
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd


def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "akommi3"  # replace tb34 with your Georgia Tech username.


def study_group():
    """
    :return: A comma separated string of GT_Name of each member of your study group
    :rtype: str
    # Example:"gburdell3, jdoe77, tbalch7" or "gburdell3" if a single individual working alone
    """
    return "akommi3"


def gtid():
    """
    :return: The GT ID of the student
    :rtype: int
    """
    return 903135337  # replace with your GT ID number

def generate_comparison_chart(symbol, sd, ed, manual_strategy, strategy_learner, verbose=False):
    """
    Generate and save comparison charts for ManualStrategy, StrategyLearner, and Benchmark.
    """
    period = "In-Sample" if sd.year == 2008 else "Out-of-Sample"

    if verbose:
        print(f"Generating comparison chart for {symbol} ({period})...")

    # Create date range based on input start and end dates
    prices = ut.get_data([symbol], pd.date_range(sd, ed))
    dates = prices.index

    # Benchmark portfolio
    benchmark_trades = pd.DataFrame(0, index=dates, columns=[symbol])
    benchmark_trades.iloc[0] = 1000  # Buy 1000 shares on the first day
    benchmark_trades.iloc[-1] = -1000  # Sell 1000 shares on the last day

    benchmark_portvals = compute_portvals(benchmark_trades, start_val=100000, commission=9.95, impact=0.005)
    benchmark_portvals /= benchmark_portvals.iloc[0]  # Normalize benchmark

    # Manual Strategy portfolio
    manual_trades = manual_strategy.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=100000)
    manual_portvals = compute_portvals(manual_trades, start_val=100000, commission=9.95, impact=0.005)
    manual_portvals /= manual_portvals.iloc[0]  # Normalize manual strategy portfolio

    # Strategy Learner portfolio
    strategy_learner.add_evidence(symbol=symbol, sd=sd, ed=ed, sv=100000)
    learner_trades = strategy_learner.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=100000)
    learner_portvals = compute_portvals(learner_trades, start_val=100000, commission=9.95, impact=0.005)
    learner_portvals /= learner_portvals.iloc[0]  # Normalize Q-Learner portfolio

    # Plot
    plt.figure(figsize=(12, 8))
    plt.plot(benchmark_portvals, label="Benchmark (Purple)", color="purple", linewidth=1.5)
    plt.plot(manual_portvals, label="Manual Strategy (Red)", color="red", linewidth=1.5)
    plt.plot(learner_portvals, label="Strategy Learner (Green)", color="green", linewidth=1.5)

    plt.title(f"{symbol} {period} Performance Comparison")
    plt.xlabel("Date")
    plt.ylabel("Normalized Portfolio Value")
    plt.legend(loc="best")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"images/{symbol}_{period}_Comparison.png")
    plt.close()

    if verbose:
        print(f"Comparison chart saved to images/{symbol}_{period}_Comparison.png")


if __name__ == "__main__":
    verbose = False
    symbol = "JPM"

    # Initialize strategies
    manual_strategy = ms.ManualStrategy(verbose=verbose)
    strategy_learner = sl.StrategyLearner(verbose=verbose, impact=0.005, commission=9.95)

    # In-sample period
    in_sample_sd = dt.datetime(2008, 1, 1)
    in_sample_ed = dt.datetime(2009, 12, 31)

    # Out-of-sample period
    out_sample_sd = dt.datetime(2010, 1, 1)
    out_sample_ed = dt.datetime(2011, 12, 31)

    # Generate Manual Strategy charts
    trades_in_sample = manual_strategy.testPolicy(symbol, in_sample_sd, in_sample_ed)
    manual_strategy.generate_charts(
        symbol, in_sample_sd, in_sample_ed, trades_in_sample, commission=9.95, impact=0.005, verbose=verbose
    )

    trades_out_sample = manual_strategy.testPolicy(symbol, out_sample_sd, out_sample_ed)
    manual_strategy.generate_charts(
        symbol, out_sample_sd, out_sample_ed, trades_out_sample, commission=9.95, impact=0.005, verbose=verbose
    )

    manual_strategy.generate_analysis_table(trades_in=trades_in_sample, trades_out=trades_out_sample)

    # Run Experiment 1
    if verbose:
        print("Running Experiment 1...")
    experiment1.run_experiment1()

    # Run Experiment 2
    if verbose:
        print("Running Experiment 2...")
    experiment2.run_experiment2()

    # Generate comparison charts
    generate_comparison_chart(symbol, in_sample_sd, in_sample_ed, manual_strategy, strategy_learner, verbose)
    generate_comparison_chart(symbol, out_sample_sd, out_sample_ed, manual_strategy, strategy_learner, verbose)

    if verbose:
        print("Test project completed successfully.")
