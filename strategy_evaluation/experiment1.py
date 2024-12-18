# experiment1.py

"""
Experiment 1: Compare Manual Strategy and Strategy Learner

Student Name: Aditya Kommi
GT User ID: akommi3
GT ID: 903135337
"""

import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from ManualStrategy import ManualStrategy
from StrategyLearner import StrategyLearner
from marketsimcode import compute_portvals
import util as ut
import numpy as np


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


def run_experiment1():
    """
    Runs experiment 1 to compare the Manual Strategy and Strategy Learner.
    """
    symbol = "JPM"
    sv = 100000

    # In-sample period
    sd_insample = dt.datetime(2008, 1, 1)
    ed_insample = dt.datetime(2009, 12, 31)

    # Out-of-sample period
    sd_outsample = dt.datetime(2010, 1, 1)
    ed_outsample = dt.datetime(2011, 12, 31)

    # Manual Strategy In-sample
    ms = ManualStrategy()
    trades_ms_in = ms.testPolicy(
        symbol=symbol, sd=sd_insample, ed=ed_insample, sv=sv
    )
    portvals_ms_in = compute_portvals(
        trades_ms_in, start_val=sv, commission=9.95, impact=0.005
    )

    # Strategy Learner In-sample
    sl = StrategyLearner(impact=0.005, commission=9.95)
    sl.add_evidence(symbol=symbol, sd=sd_insample, ed=ed_insample, sv=sv)
    trades_sl_in = sl.testPolicy(
        symbol=symbol, sd=sd_insample, ed=ed_insample, sv=sv
    )
    portvals_sl_in = compute_portvals(
        trades_sl_in, start_val=sv, commission=9.95, impact=0.005
    )

    # Benchmark In-sample
    dates_in = pd.date_range(sd_insample, ed_insample)
    prices_in = ut.get_data([symbol], dates_in)
    benchmark_trades_in = pd.DataFrame(index=prices_in.index, columns=[symbol])
    benchmark_trades_in.iloc[:, :] = 0
    benchmark_trades_in.iloc[0, 0] = 1000  # Buy 1000 shares on the first day
    portvals_bench_in = compute_portvals(
        benchmark_trades_in, start_val=sv, commission=9.95, impact=0.005
    )

    # Normalize portfolios
    norm_ms_in = portvals_ms_in / portvals_ms_in.iloc[0]
    norm_sl_in = portvals_sl_in / portvals_sl_in.iloc[0]
    norm_bench_in = portvals_bench_in / portvals_bench_in.iloc[0]

    # Plot in-sample
    plt.figure(figsize=(10, 6))
    plt.plot(norm_ms_in, label="Manual Strategy", color="red")
    plt.plot(norm_sl_in, label="Strategy Learner", color="green")
    plt.plot(norm_bench_in, label="Benchmark", color="purple")
    plt.title("In-Sample Portfolio Value")
    plt.xlabel("Date")
    plt.ylabel("Normalized Portfolio Value")
    plt.legend()
    plt.grid()
    plt.savefig("images/experiment1_in_sample.png")
    plt.close()

    # Manual Strategy Out-of-sample
    trades_ms_out = ms.testPolicy(
        symbol=symbol, sd=sd_outsample, ed=ed_outsample, sv=sv
    )
    portvals_ms_out = compute_portvals(
        trades_ms_out, start_val=sv, commission=9.95, impact=0.005
    )

    # Strategy Learner Out-of-sample
    trades_sl_out = sl.testPolicy(
        symbol=symbol, sd=sd_outsample, ed=ed_outsample, sv=sv
    )
    portvals_sl_out = compute_portvals(
        trades_sl_out, start_val=sv, commission=9.95, impact=0.005
    )

    # Benchmark Out-of-sample
    dates_out = pd.date_range(sd_outsample, ed_outsample)
    prices_out = ut.get_data([symbol], dates_out)
    benchmark_trades_out = pd.DataFrame(
        index=prices_out.index, columns=[symbol]
    )
    benchmark_trades_out.iloc[:, :] = 0
    benchmark_trades_out.iloc[0, 0] = 1000  # Buy 1000 shares on the first day
    portvals_bench_out = compute_portvals(
        benchmark_trades_out, start_val=sv, commission=9.95, impact=0.005
    )

    # Normalize portfolios
    norm_ms_out = portvals_ms_out / portvals_ms_out.iloc[0]
    norm_sl_out = portvals_sl_out / portvals_sl_out.iloc[0]
    norm_bench_out = portvals_bench_out / portvals_bench_out.iloc[0]

    # Plot out-of-sample
    plt.figure(figsize=(10, 6))
    plt.plot(norm_ms_out, label="Manual Strategy", color="red")
    plt.plot(norm_sl_out, label="Strategy Learner", color="green")
    plt.plot(norm_bench_out, label="Benchmark", color="purple")
    plt.title("Out-of-Sample Portfolio Value")
    plt.xlabel("Date")
    plt.ylabel("Normalized Portfolio Value")
    plt.legend()
    plt.grid()
    plt.savefig("images/experiment1_out_sample.png")
    plt.close()


if __name__ == "__main__":
    run_experiment1()
