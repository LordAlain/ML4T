""""""
"""MC2-P1: Market simulator.

Copyright 2018, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 4646/7646

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as github and gitlab.  This copyright statement should not be removed
or edited.

We do grant permission to share solutions privately with non-students such
as potential employers. However, sharing with other current or future
students of CS 7646 is prohibited and subject to being investigated as a
GT honor code violation.

-----do not edit anything above this line---

Student Name: Aditya Kommi (replace with your name)
GT User ID: akommi3 (replace with your User ID)
GT ID: 903135337 (replace with your GT ID)
"""

import datetime as dt
# import os
import numpy as np
import pandas as pd
from util import get_data, plot_data


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

def compute_portvals(orders_file="./orders/orders.csv", start_val=1000000, commission=9.95, impact=0.005):
    """
    Computes the portfolio values.

    :param orders_file: Path of the order file or the file object
    :type orders_file: str or file object
    :param start_val: The starting value of the portfolio
    :type start_val: int
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)
    :type commission: float
    :param impact: The amount the price moves against the trader compared to the historical data at each transaction
    :type impact: float
    :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.
    :rtype: pandas.DataFrame
    """
    # this is the function the autograder will call to test your code
    # NOTE: orders_file may be a string, or it may be a file object. Your
    # code should work correctly with either input
    # TODO: Your code here

    ## Parse Data
    orders = pd.read_csv(orders_file, header=0, index_col='Date', parse_dates=True)
    # orders.index = pd.to_datetime(orders.index)

    start_date = pd.to_datetime(orders.index.min(), format='%Y-%m-%d')
    end_date = pd.to_datetime(orders.index.max(), format='%Y-%m-%d')
    dates = pd.date_range(start_date, end_date)
    syms = orders['Symbol'].unique()
    # syms = np.append(syms, 'Cash')


    # Read in adjusted closing prices for given symbols, date range
    prices_all = get_data(syms, dates)  # automatically adds SPY
    syms = np.append(syms, 'Cash') # add Cash column
    prices_all['Cash'] = 1.0
    # prices_all = prices_all.reindex(dates).fillna(method="ffill")
    prices_all = prices_all.fillna(method="ffill")
    prices_all = prices_all.fillna(method="bfill")

    ## Initialize Portfolio and holdings


    # only portfolio symbols
    prices = prices_all[syms]

    # only SPY, for comparison later
    prices_SPY = prices_all["SPY"]
    normed = prices / prices.iloc[0]
    n = len(syms)

    # portvals = get_data(syms, pd.date_range(start_date, end_date)).drop(columns=['SPY'])
    portfolio = pd.DataFrame(index=prices.index, columns=syms)
    portfolio[syms] = 0
    portfolio['Cash'] = start_val
    cash = start_val
    # holdings = portfolio
    holdings = {symbol: 0 for symbol in syms}  # Initial holdings for each symbol

    ## Iterate through orders

    for dates in portfolio.index:
        if dates not in prices.index:
            continue

        if dates in orders.index:
            date_orders = orders.loc[[dates]]

            for date, row in date_orders.iterrows():
                symbol = row['Symbol']
                order_type = row['Order']
                shares = row['Shares']
                price = prices.loc[date, symbol]  # Get the stock price on the date of the order

                if order_type == 'BUY':
                    # Calculate the cost of buying shares (including market impact and commission)
                    cost = shares * price * (1 + impact)  # Apply market impact for buying
                    cash -= cost + commission  # Deduct cash and commission for the transaction
                    holdings[symbol] += shares  # Update holdings with the bought shares
                    holdings['Cash'] = cash
                    # holdings.iloc[date:, symbol] += shares  # Update holdings with the bought shares
                    # holdings.iloc[date:, 'Cash'] = cash

                elif order_type == 'SELL':
                    # Calculate the revenue from selling shares (including market impact and commission)
                    revenue = shares * price * (1 - impact)  # Apply market impact for selling
                    cash += revenue - commission  # Add revenue and deduct commission
                    holdings[symbol] -= shares  # Update holdings by subtracting the sold shares
                    holdings['Cash'] = cash
                    # holdings[date:, symbol] -= shares  # Update holdings by subtracting the sold shares
                    # holdings[date:, 'Cash'] = cash

        # Update portfolio with current cash and updated holdings
        for sym in syms:
            portfolio.loc[dates, sym] = holdings[sym] * prices.loc[dates, sym]  # Update each symbol holding

        # print("1")

    # print("2")
        # Compute total portfolio value (cash + holdings)
    portvals = portfolio[syms].sum(axis=1)
    # print()

    # Return the total portfolio value as a DataFrame
    return portvals


    # In the template, instead of computing the value of the portfolio, we just
    # read in the value of IBM over 6 months
    # start_date = dt.datetime(2008, 1, 1)
    # end_date = dt.datetime(2008, 6, 1)
    # portvals = get_data(["IBM"], pd.date_range(start_date, end_date))
    # portvals = portvals[["IBM"]]  # remove SPY
    # rv = pd.DataFrame(index=portvals.index, data=portvals.values)

    # return rv
    # return portvals


def test_code():
    """
    Helper function to test code
    """
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    of = "orders/orders-short.csv"
    sv = 1000000

    # Process orders
    portvals = compute_portvals(orders_file=of, start_val=sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]]  # just get the first column
    else:
        "warning, code did not return a DataFrame"

    # Get portfolio stats
    # Here we just fake the data. you should use your code from previous assignments.
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2008, 6, 1)
    # cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [
    #     0.2,
    #     0.01,
    #     0.02,
    #     1.5,
    # ]
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [
        0.2,
        0.01,
        0.02,
        1.5,
    ]

    # Calculate portfolio statistics
    daily_rets = (portvals / portvals.shift(1)) - 1
    daily_rets = daily_rets[1:]  # exclude the first day
    cum_ret = (portvals[-1] / portvals[0]) - 1
    avg_daily_ret = daily_rets.mean()
    std_daily_ret = daily_rets.std()
    sharpe_ratio = -(np.sqrt(252) * avg_daily_ret) / std_daily_ret

    # Compare portfolio against $SPX
    print(f"Date Range: {start_date} to {end_date}")
    print()
    print(f"Sharpe Ratio of Fund: {sharpe_ratio}")
    print(f"Sharpe Ratio of SPY : {sharpe_ratio_SPY}")
    print()
    print(f"Cumulative Return of Fund: {cum_ret}")
    print(f"Cumulative Return of SPY : {cum_ret_SPY}")
    print()
    print(f"Standard Deviation of Fund: {std_daily_ret}")
    print(f"Standard Deviation of SPY : {std_daily_ret_SPY}")
    print()
    print(f"Average Daily Return of Fund: {avg_daily_ret}")
    print(f"Average Daily Return of SPY : {avg_daily_ret_SPY}")
    print()
    print(f"Final Portfolio Value: {portvals[-1]}")


if __name__ == "__main__":
    test_code()
