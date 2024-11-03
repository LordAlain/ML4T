import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
from util import get_data
from TheoreticallyOptimalStrategy import testPolicy
from marketsimcode import compute_portvals
from indicators import run_indicators

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

if __name__ == "__main__":
    # if sys.argv[len(sys.argv) - 1] == "-debug":
    #     verbose=True
    # else:
    #     verbose=False

    # Set parameters
    symbol = "JPM"
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    sv = 100000
    verbose = False  # Set to True to display charts

    if verbose:
        print(f"Running full test for {symbol} from {sd} to {ed}")

    # Step 1: Get stock prices and calculate indicators
    prices = get_data([symbol], pd.date_range(sd, ed))[symbol]
    run_indicators(symbol, prices, verbose=verbose)

    # Step 2: Run Theoretical Optimal Strategy (TOS)
    trades = testPolicy(symbol, sd, ed, sv)

    # Step 3: Simulate the portfolio
    strategy_portvals = compute_portvals(trades, start_val=sv, commission=0.0, impact=0.0)
    benchmark_trades = pd.DataFrame(0.0, index=prices.index, columns=[symbol])
    benchmark_trades.iloc[0] = 1000.0  # Buy 1000 shares at the start
    benchmark_portvals = compute_portvals(benchmark_trades, start_val=sv, commission=0.0, impact=0.0)

    # Extract 'PortVal' column to get Series
    strategy_portvals = strategy_portvals['PortVal']
    benchmark_portvals = benchmark_portvals['PortVal']

    # Step 4: Normalize and plot comparison
    norm_strategy = strategy_portvals / strategy_portvals.iloc[0]
    norm_benchmark = benchmark_portvals / benchmark_portvals.iloc[0]

    plt.figure(figsize=(12, 6))
    plt.plot(norm_strategy, label='Theoretical Optimal Strategy', color='red')
    plt.plot(norm_benchmark, label='Benchmark (Buy and Hold)', color='purple')
    plt.title(f'Normalized Portfolio Values: {symbol}')
    plt.xlabel('Date')
    plt.ylabel('Normalized Value')
    plt.legend()
    plt.grid()
    if verbose:
        plt.show()
    else:
        plt.savefig(f'images/{symbol}_strategy_vs_benchmark.png')
        plt.close()

    # Step 5: Calculate statistics
    daily_returns_strategy = strategy_portvals.pct_change().dropna()
    daily_returns_benchmark = benchmark_portvals.pct_change().dropna()

    cum_return_strategy = (strategy_portvals.iloc[-1] / strategy_portvals.iloc[0]) - 1
    cum_return_benchmark = (benchmark_portvals.iloc[-1] / benchmark_portvals.iloc[0]) - 1

    std_strategy = daily_returns_strategy.std()
    std_benchmark = daily_returns_benchmark.std()

    mean_strategy = daily_returns_strategy.mean()
    mean_benchmark = daily_returns_benchmark.mean()

    if verbose:
        print("Statistics:")
        print(f"Cumulative Return of Strategy: {cum_return_strategy:.6f}")
        print(f"Cumulative Return of Benchmark: {cum_return_benchmark:.6f}")
        print(f"Standard Deviation of Strategy: {std_strategy:.6f}")
        print(f"Standard Deviation of Benchmark: {std_benchmark:.6f}")
        print(f"Mean Daily Return of Strategy: {mean_strategy:.6f}")
        print(f"Mean Daily Return of Benchmark: {mean_benchmark:.6f}")

    if verbose:
        print("Full test completed successfully.")