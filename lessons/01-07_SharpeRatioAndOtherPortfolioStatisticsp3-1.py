"""
(c) 2015 by Devpriya Dave and Tucker Balch.
"""
#from optimize_something.optimization import compute_daily_returns

"""=================================================================================="""

import datetime as dt

import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
#from util import get_data, plot_data


def symbol_to_path(symbol, base_dir=None):
    """Return CSV file path given ticker symbol."""
    if base_dir is None:
        base_dir = os.environ.get("MARKET_DATA_DIR", "../data/")
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))


def get_data(symbols, dates, addSPY=True, colname="Adj Close"):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df = pd.DataFrame(index=dates)
    if addSPY and "SPY" not in symbols:  # add SPY for reference, if absent
        symbols = ["SPY"] + list(
            symbols
        )  # handles the case where symbols is np array of 'object'

    for symbol in symbols:
        df_temp = pd.read_csv(
            symbol_to_path(symbol),
            index_col="Date",
            parse_dates=True,
            usecols=["Date", colname],
            na_values=["nan"],
        )
        df_temp = df_temp.rename(columns={colname: symbol})
        df = df.join(df_temp)
        if symbol == "SPY":  # drop dates SPY did not trade
            df = df.dropna(subset=["SPY"])

    return df

start_val = 1000000
start_date = "2009-1-1"
sd = dt.datetime.strptime(start_date, "%Y-%m-%d")
end_date = "2011-12-31"
ed = dt.datetime.strptime(end_date, "%Y-%m-%d")
symbols = ['SPY', 'XOM', 'GOOG', 'GLD']
syms = ['SPY', 'XOM', 'GOOG', 'GLD']
allocs = [0.4, 0.4, 0.1, 0.1]
k = 252

# Read in adjusted closing prices for given symbols, date range
dates = pd.date_range(sd, ed)
prices_all = get_data(symbols=syms, dates=dates)  # automatically adds SPY
prices = prices_all[syms]  # only portfolio symbols

print("prices:\n", prices.iloc[0])
normed = prices/prices.iloc[0]
alloced = normed * allocs
pos_vals = alloced * start_val
port_val = pos_vals.sum(axis=1)

daily_returns = (port_val / port_val.shift(1)) - 1 # much easier with Pandas!
daily_returns.iloc[0] = 0 # Pandas leaves the 0th row full of Nans
daily_rets = daily_returns[1:]
cum_ret = (port_val[-1]/port_val[0] - 1)
avg_daily_ret = daily_rets.mean()
std_daily_ret = daily_rets.std()

daily_rf = 0
SR = np.sqrt(k) * (daily_rets - daily_rf).mean() / daily_rets.std()

# Print statistics
print(f"Start Date: {start_date}")
print(f"End Date: {end_date}")
print(f"Symbols: {symbols}")
print(f"Sharpe Ratio: {SR}")
print(f"Volatility (stdev of daily returns): {std_daily_ret}")
print(f"Average Daily Return: {avg_daily_ret}")
print(f"Cumulative Return: {cum_ret}")

