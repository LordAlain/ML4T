import pandas as pd
import datetime as dt
from util import get_data

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

def testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000, verbose=False):
    """
    Implements a theoretically optimal strategy for trading a given stock while respecting position limits.

    This strategy buys before the price increases and sells before the price decreases,
    leveraging future price knowledge.

    :param symbol: The stock symbol to act on (e.g., "JPM")
    :type symbol: str
    :param sd: Start date for the time period
    :type sd: datetime
    :param ed: End date for the time period
    :type ed: datetime
    :param sv: Starting value of the portfolio
    :type sv: int
    :param verbose: If True, prints detailed output during calculations
    :type verbose: bool
    :return: A DataFrame of trades (+1000 for buy, -1000 for sell, 0 for hold)
    :rtype: pd.DataFrame
    """
    if verbose:
        print(f"Running theoretical optimal strategy for {symbol} from {sd} to {ed}")

    # Generate date range and get price data
    dates = pd.date_range(sd, ed)
    prices_all = get_data([symbol], dates)
    prices = prices_all[symbol]

    # Initialize trades DataFrame
    trades = pd.DataFrame(index=prices.index, data=0, columns=[symbol])

    # Compute future price changes
    future_prices = prices.shift(-1)
    future_prices.iloc[-1] = prices.iloc[-1]  # Last day price remains the same
    price_diff = future_prices - prices

    # Initialize position
    position = 0  # Current position: +1000 (long), -1000 (short), or 0 (neutral)

    # Iterate over the price differences
    for i in range(len(prices)):
        if price_diff.iloc[i] > 0 and position <= 0:
            # Buy up to 1000 shares
            trade = 1000 - position
            trades.iloc[i] = trade
            position += trade
            if verbose:
                print(f"Buying {trade} shares on {prices.index[i]} at ${prices.iloc[i]:.2f}")
        elif price_diff.iloc[i] < 0 and position >= 0:
            # Sell down to -1000 shares
            trade = -1000 - position
            trades.iloc[i] = trade
            position += trade
            if verbose:
                print(f"Selling {abs(trade)} shares on {prices.index[i]} at ${prices.iloc[i]:.2f}")

    if verbose:
        print("Theoretical optimal strategy executed successfully.")

    return trades


if __name__ == "__main__":
    df_trades = testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000, verbose=True)
    # print(df_trades)
