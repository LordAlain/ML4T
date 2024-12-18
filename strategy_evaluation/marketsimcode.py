import pandas as pd
import datetime as dt
import util


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

def compute_portvals(trades, start_val=100000, commission=0.0, impact=0.0):
    """
    Computes the portfolio values given a set of trades.

    :param trades: DataFrame of trades with dates as index and symbols as columns.
    :type trades: pd.DataFrame
    :param start_val: The starting value of the portfolio.
    :type start_val: int
    :param commission: The fixed amount in dollars charged for each transaction.
    :type commission: float
    :param impact: The amount the price moves against the trader compared to the historical data.
    :type impact: float
    :return: A DataFrame containing the value of the portfolio for each trading day.
    :rtype: pd.DataFrame
    """
    # Generate date range and get price data
    start_date = trades.index.min()
    end_date = trades.index.max()
    dates = pd.date_range(start_date, end_date)
    symbols = trades.columns.tolist()
    prices_all = util.get_data(symbols, dates)
    prices_all['Cash'] = 1.0  # Add Cash column

    # Fill missing data
    prices_all = prices_all.fillna(method="ffill").fillna(method="bfill")

    # Initialize holdings DataFrame
    holdings = pd.DataFrame(index=prices_all.index, columns=symbols + ['Cash'])
    holdings.iloc[0] = 0.0
    holdings.at[holdings.index[0], 'Cash'] = start_val

    # Iterate through each date to update holdings
    for i in range(len(holdings)):
        date = holdings.index[i]
        if i > 0:
            holdings.iloc[i] = holdings.iloc[i - 1]
        if date in trades.index:
            for symbol in symbols:
                shares = trades.at[date, symbol]
                if shares != 0:
                    price = prices_all.at[date, symbol]
                    # Calculate transaction cost
                    transaction_cost = price * shares * (1 + impact)
                    total_cost = transaction_cost + commission
                    # Update holdings
                    holdings.at[date, symbol] += shares
                    holdings.at[date, 'Cash'] -= total_cost

    # Calculate portfolio values
    portvals = (holdings[symbols] * prices_all[symbols]).sum(axis=1) + holdings['Cash']

    # Return the total portfolio value as a DataFrame
    return portvals.to_frame(name='PortVal')


if __name__ == "__main__":
    # Example usage
    import TheoreticallyOptimalStrategy as tos

    # Run the theoretical optimal strategy
    df_trades = tos.testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
    # Compute portfolio values
    portvals = compute_portvals(df_trades, start_val=100000, commission=0.0, impact=0.0)
    # print(portvals)
