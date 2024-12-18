"""
ManualStrategy.py

Student Name: Aditya Kommi
GT User ID: akommi3
GT ID: 903135337
"""

import datetime as dt
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from marketsimcode import compute_portvals
import util as ut
from indicators import compute_bollinger_bands, compute_rsi, compute_macd


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


class ManualStrategy(object):
    """
    A simple manual strategy that uses indicators to make trading decisions.

    :param verbose: If “verbose” is True, your code can print out information for debugging.
    :type verbose: bool
    """

    def __init__(self, verbose=False):
        self.verbose = verbose

    def testPolicy(
        self,
        symbol="JPM",
        sd=dt.datetime(2008, 1, 1),
        ed=dt.datetime(2009, 12, 31),
        sv=100000,
    ):
        """
        Tests the manual strategy and returns a trades DataFrame.

        :param symbol: The stock symbol to trade
        :type symbol: str
        :param sd: A datetime object that represents the start date
        :type sd: datetime
        :param ed: A datetime object that represents the end date
        :type ed: datetime
        :param sv: The starting value of the portfolio
        :type sv: int
        :return: A DataFrame with values representing trades for each day.
        :rtype: pandas.DataFrame
        """

        # Fetch data
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data([symbol], dates)
        prices = prices_all[symbol].fillna(method="ffill").fillna(method="bfill")

        # Compute indicators
        bbp = compute_bollinger_bands(prices, window=20)
        rsi = compute_rsi(prices, window=14)
        macd = compute_macd(prices)

        # Create a signal DataFrame
        signals = pd.DataFrame(index=prices.index)
        signals["bbp_signal"] = (bbp < 0.35).astype(int) - (bbp > 0.65).astype(int)
        signals["rsi_signal"] = (rsi < 40).astype(int) - (rsi > 60).astype(int)
        signals["macd_signal"] = (macd > 0).astype(int) - (macd < 0).astype(int)

        # Combine signals into a single signal
        signals["combined_signal"] = (
            signals["bbp_signal"] +
            signals["rsi_signal"] +
            signals["macd_signal"]
        )

        # Initialize trades DataFrame
        trades = pd.DataFrame(data=0, index=prices.index, columns=[symbol])

        # Generate trades from combined signals
        holdings = 0  # Current position
        for i in range(1, len(signals)):
            # Buy Signal: Enter or increase a long position
            if signals["combined_signal"].iloc[i] > 1:
                if holdings == 0:
                    trades.iloc[i] = 1000  # Buy
                    holdings = 1000
                elif holdings == -1000:
                    trades.iloc[i] = 2000  # Cover short and go long
                    holdings = 1000
            # Sell Signal: Enter or increase a short position
            elif signals["combined_signal"].iloc[i] < -1:
                if holdings == 0:
                    trades.iloc[i] = -1000  # Sell
                    holdings = -1000
                elif holdings == 1000:
                    trades.iloc[i] = -2000  # Sell long and go short
                    holdings = -1000
            # Do Nothing: Maintain current position
            else:
                trades.iloc[i] = 0

        if self.verbose:
            print(signals)
            print(trades)

        return trades

    def generate_charts(self, symbol, sd, ed, trades, commission, impact, verbose=False):
        """
        Generates and saves charts showing the strategy and benchmark performance.
        """
        # Fetch benchmark data
        dates = pd.date_range(sd, ed)
        prices = ut.get_data([symbol], dates)

        # Benchmark trades
        benchmark_trades = pd.DataFrame(0, index=prices.index, columns=[symbol])
        benchmark_trades.iloc[0] = 1000
        benchmark_portvals = compute_portvals(benchmark_trades, 100000, commission, impact)
        benchmark_portvals /= benchmark_portvals.iloc[0]

        # Manual strategy portfolio
        portvals = compute_portvals(trades, 100000, commission, impact)
        portvals /= portvals.iloc[0]

        # Plot strategy and benchmark
        plt.figure(figsize=(12, 8))
        plt.plot(portvals, label="Manual Strategy (Red)", color="red")
        plt.plot(benchmark_portvals, label="Benchmark (Purple)", color="purple")

        # Add LONG and SHORT entry points
        long_signals = trades[trades[symbol] > 0].index
        short_signals = trades[trades[symbol] < 0].index
        plt.vlines(long_signals, ymin=portvals.min(), ymax=portvals.max(), colors="blue", label="LONG Entry")
        plt.vlines(short_signals, ymin=portvals.min(), ymax=portvals.max(), colors="black", label="SHORT Entry")

        # Add labels and save the plot
        period = "In-Sample" if sd.year == 2008 else "Out-of-Sample"
        plt.title(f"{symbol} Manual Strategy ({period})")
        plt.xlabel("Date")
        plt.ylabel("Normalized Portfolio Value")
        plt.legend(loc="best")
        plt.grid()
        plt.tight_layout()
        plt.savefig(f"images/{symbol}_manual_strategy_{period.lower().replace(' ', '_')}.png")
        plt.close()

        if verbose:
            print(f"Chart saved to images/{symbol}_manual_strategy_{period.lower().replace(' ', '_')}.png")

    def calculate_performance(self, portvals, benchmark_portvals):
        """
        Calculates performance metrics for the strategy and benchmark.
        """
        # Calculate metrics
        cr_strategy = (portvals.iloc[-1] / portvals.iloc[0]) - 1
        cr_benchmark = (benchmark_portvals.iloc[-1] / benchmark_portvals.iloc[0]) - 1
        daily_returns_strategy = portvals.pct_change().iloc[1:]
        daily_returns_benchmark = benchmark_portvals.pct_change().iloc[1:]
        std_strategy = daily_returns_strategy.std()
        std_benchmark = daily_returns_benchmark.std()
        mean_strategy = daily_returns_strategy.mean()
        mean_benchmark = daily_returns_benchmark.mean()

        # Create summary table
        performance = {
            "Cumulative Return": [cr_strategy.item(), cr_benchmark.item()],
            "STDEV of Daily Returns": [std_strategy.item(), std_benchmark.item()],
            "Mean of Daily Returns": [mean_strategy.item(), mean_benchmark.item()],
        }
        summary = pd.DataFrame(performance, index=["Manual Strategy", "Benchmark"])
        return summary

    def generate_analysis_table(
            self,
            trades_in,
            trades_out,
            symbol = "JPM",
            sd_in = dt.datetime(2008, 1, 1),
            ed_in = dt.datetime(2009, 12, 31),
            sd_out = dt.datetime(2010, 1, 1),
            ed_out = dt.datetime(2011, 12, 31),
            commission = 9.95,
            impact = 0.005,
            verbose=False
    ):
        """
        Generates a combined performance summary table for in-sample and out-of-sample periods.
        """

        # In-Sample Benchmark
        dates_in = pd.date_range(sd_in, ed_in)
        prices_in = ut.get_data([symbol], dates_in)
        benchmark_trades_in = pd.DataFrame(0, index=prices_in.index, columns=[symbol])
        benchmark_trades_in.iloc[0] = 1000
        benchmark_portvals_in = compute_portvals(
            benchmark_trades_in, start_val=100000, commission=commission, impact=impact
        )

        # In-Sample Manual Strategy
        portvals_in = compute_portvals(
            trades_in, start_val=100000, commission=commission, impact=impact
        )

        # Out-of-Sample Benchmark
        dates_out = pd.date_range(sd_out, ed_out)
        prices_out = ut.get_data([symbol], dates_out)
        benchmark_trades_out = pd.DataFrame(0, index=prices_out.index, columns=[symbol])
        benchmark_trades_out.iloc[0] = 1000
        benchmark_portvals_out = compute_portvals(
            benchmark_trades_out, start_val=100000, commission=commission, impact=impact
        )

        # Out-of-Sample Manual Strategy
        portvals_out = compute_portvals(
            trades_out, start_val=100000, commission=commission, impact=impact
        )

        # Calculate performance metrics
        performance_in = self.calculate_performance(portvals_in, benchmark_portvals_in)
        performance_out = self.calculate_performance(portvals_out, benchmark_portvals_out)

        # Combine results into a single DataFrame
        performance = {
            "Cumulative Return": [
                performance_in["Cumulative Return"][0],
                performance_in["Cumulative Return"][1],
                performance_out["Cumulative Return"][0],
                performance_out["Cumulative Return"][1],
            ],
            "STDEV of Daily Returns": [
                performance_in["STDEV of Daily Returns"][0],
                performance_in["STDEV of Daily Returns"][1],
                performance_out["STDEV of Daily Returns"][0],
                performance_out["STDEV of Daily Returns"][1],
            ],
            "Mean of Daily Returns": [
                performance_in["Mean of Daily Returns"][0],
                performance_in["Mean of Daily Returns"][1],
                performance_out["Mean of Daily Returns"][0],
                performance_out["Mean of Daily Returns"][1],
            ],
        }

        summary_table = pd.DataFrame(
            performance,
            index=[
                "Manual Strategy (In-Sample)",
                "Benchmark (In-Sample)",
                "Manual Strategy (Out-of-Sample)",
                "Benchmark (Out-of-Sample)",
            ],
        )

        filename = "performance_summary.txt"
        summary_table.to_csv(filename, sep=",")
        if self.verbose:
            print(f"Performance summary saved to {filename}")

    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "akommi3"  # replace tb34 with your Georgia Tech username.

    def study_group(self):
        """
        :return: A comma separated string of GT_Name of each member of your study group
        :rtype: str
        # Example:"gburdell3, jdoe77, tbalch7" or "gburdell3" if a single individual working alone
        """
        return "akommi3"

    def gtid(self):
        """
        :return: The GT ID of the student
        :rtype: int
        """
        return 903135337  # replace with your GT ID number

if __name__ == "__main__":
    print("Testing Manual Strategy")
    ms = ManualStrategy(verbose=False)
    trades = ms.testPolicy()
    print(trades)
