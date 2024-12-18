# experiment2.py

"""
Experiment 2: Analyze Impact on Strategy Learner

Student Name: Aditya Kommi
GT User ID: akommi3
GT ID: 903135337
"""

import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from StrategyLearner import StrategyLearner
from marketsimcode import compute_portvals
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


def run_experiment2():
    """
    Runs experiment 2 to analyze the impact on Strategy Learner.
    """
    symbol = "JPM"
    sv = 100000
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    impacts = [0.0, 0.005, 0.01]
    cumulative_returns = []
    num_trades = []

    for impact in impacts:
        sl = StrategyLearner(impact=impact, commission=0.0)
        sl.add_evidence(symbol=symbol, sd=sd, ed=ed, sv=sv)
        trades = sl.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)
        portvals = compute_portvals(
            trades, start_val=sv, commission=0.0, impact=impact
        )
        cr = portvals.iloc[-1] / portvals.iloc[0] - 1
        cumulative_returns.append(cr)
        num_trade = trades[symbol].astype(bool).sum()
        num_trades.append(num_trade)

    # Plot Impact vs. Cumulative Return
    plt.figure(figsize=(10, 6))
    plt.plot(impacts, cumulative_returns, marker="o")
    plt.title("Impact vs. Cumulative Return")
    plt.xlabel("Impact")
    plt.ylabel("Cumulative Return")
    plt.grid()
    plt.savefig("images/experiment2_cr.png")
    plt.close()

    # Plot Impact vs. Number of Trades
    plt.figure(figsize=(10, 6))
    plt.plot(impacts, num_trades, marker="o")
    plt.title("Impact vs. Number of Trades")
    plt.xlabel("Impact")
    plt.ylabel("Number of Trades")
    plt.grid()
    plt.savefig("images/experiment2_trades.png")
    plt.close()


if __name__ == "__main__":
    run_experiment2()
