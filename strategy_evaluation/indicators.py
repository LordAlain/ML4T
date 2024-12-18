import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

def compute_sma(prices, window=20, verbose=False):
    """
    Compute Simple Moving Average (SMA).

    :param prices: Adjusted close prices.
    :type prices: pd.Series
    :param window: Window size for the SMA.
    :type window: int
    :param verbose: If True, prints debugging information.
    :type verbose: bool
    :return: The SMA values.
    :rtype: pd.Series
    """
    if verbose:
        print(f"Calculating Simple Moving Average (SMA) with window {window}...")
    sma = prices.rolling(window=window).mean()
    return sma


def compute_std(prices, window=20, verbose=False):
    """
    Compute Rolling Standard Deviation.

    :param prices: Adjusted close prices.
    :type prices: pd.Series
    :param window: Window size for the standard deviation.
    :type window: int
    :param verbose: bool
    :return: The rolling standard deviation values.
    :rtype: pd.Series
    """
    if verbose:
        print(f"Calculating Rolling Standard Deviation with window {window}...")
    std = prices.rolling(window=window).std()
    return std


def compute_bollinger_bands(prices, window=20, verbose=False):
    """
    Compute Bollinger Band Percentage (%B).

    Bollinger Bands are volatility bands placed above and below a moving average.

    The Bollinger Band Percentage (%B) is calculated as:
    %B = (Price - Lower Band) / (Upper Band - Lower Band)

    :param prices: Adjusted close prices.
    :type prices: pd.Series
    :param window: Window size for the Bollinger Bands.
    :type window: int
    :param verbose: bool
    :return: The Bollinger Band Percentage (%B).
    :rtype: pd.Series
    """
    if verbose:
        print(f"Calculating Bollinger Bands (%B) with window {window}...")
    sma = compute_sma(prices, window)
    rolling_std = compute_std(prices, window)
    upper_band = sma + (rolling_std * 2)
    lower_band = sma - (rolling_std * 2)
    bbp = (prices - lower_band) / (upper_band - lower_band)
    return bbp  # Return %B as the single real results vector


def compute_rsi(prices, window=14, verbose=False):
    """
    Compute Relative Strength Index (RSI).

    :param prices: Adjusted close prices.
    :type prices: pd.Series
    :param window: Window size for the RSI.
    :type window: int
    :param verbose: If True, prints debugging information.
    :type verbose: bool
    :return: The RSI values.
    :rtype: pd.Series
    """
    if verbose:
        print(f"Calculating Relative Strength Index (RSI) with window {window}...")
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_cci(prices, window=20, verbose=False):
    """
    Compute Commodity Channel Index (CCI).

    :param prices: Adjusted close prices.
    :type prices: pd.Series
    :param window: Window size for the CCI.
    :type window: int
    :param verbose: bool
    :return: The CCI values.
    :rtype: pd.Series
    """
    if verbose:
        print(f"Calculating Commodity Channel Index (CCI) with window {window}...")
    tp = prices  # Typical price (since we only have close prices)
    sma = compute_sma(tp, window, verbose)
    mad = tp.rolling(window=window).apply(lambda x: np.fabs(x - x.mean()).mean())
    cci = (tp - sma) / (0.015 * mad)
    return cci


def compute_momentum(prices, window=10, verbose=False):
    """
    Compute Momentum indicator.

    :param prices: Adjusted close prices.
    :type prices: pd.Series
    :param window: Lookback window for momentum calculation.
    :type window: int
    :param verbose: bool
    :return: The Momentum values.
    :rtype: pd.Series
    """
    if verbose:
        print(f"Calculating Momentum with window {window}...")
    momentum = (prices / prices.shift(window)) - 1
    return momentum


def compute_macd(prices, n_fast=12, n_slow=26, verbose=False):
    """
    Compute Moving Average Convergence Divergence (MACD) Line.

    Note: Returns only the MACD Line as a single results vector.

    :param prices: Adjusted close prices.
    :type prices: pd.Series
    :param n_fast: Fast EMA period.
    :type n_fast: int
    :param n_slow: Slow EMA period.
    :type n_slow: int
    :param verbose: bool
    :return: The MACD Line values.
    :rtype: pd.Series
    """
    if verbose:
        print(f"Calculating MACD Line with fast period {n_fast} and slow period {n_slow}...")
    ema_fast = prices.ewm(span=n_fast, adjust=False).mean()
    ema_slow = prices.ewm(span=n_slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    return macd_line


def run_indicators(symbol, prices, verbose=False):
    """
    Run selected technical indicators for a stock, generate and save/show charts.

    :param symbol: Stock symbol.
    :type symbol: str
    :param prices: Adjusted close price data.
    :type prices: pd.Series
    :param verbose: If True, prints debugging information.
    :type verbose: bool
    :return: A dictionary containing all computed indicators.
    :rtype: dict
    """

    if verbose:
        print(f"Running indicators for symbol {symbol}...")

    # Compute indicators
    bbp = compute_bollinger_bands(prices, window=20, verbose=verbose)
    sma = compute_sma(prices, window=20, verbose=verbose)
    rolling_std = compute_std(prices, window=20, verbose=verbose)
    upper_band = sma + (rolling_std * 2)
    lower_band = sma - (rolling_std * 2)
    rsi = compute_rsi(prices, window=14, verbose=verbose)
    cci = compute_cci(prices, window=20, verbose=verbose)
    momentum = compute_momentum(prices, window=10, verbose=verbose)
    macd_line = compute_macd(prices, verbose=verbose)

    indicators = {
        'Bollinger Bands %B': bbp,
        'RSI': rsi,
        'CCI': cci,
        'Momentum': momentum,
        'MACD Line': macd_line
    }

    # Plot indicators with compelling charts
    # 1. Bollinger Bands (%B)
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 1, 1)
    plt.plot(prices, label='Price', color='blue')
    plt.plot(sma, label='SMA (20)', color='orange')
    plt.plot(upper_band, label='Upper Band', color='green', linestyle='--')
    plt.plot(lower_band, label='Lower Band', color='red', linestyle='--')
    plt.fill_between(prices.index, lower_band, upper_band, color='grey', alpha=0.2)
    plt.title(f'Bollinger Bands for {symbol}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(bbp, label='Bollinger Band %B', color='purple')
    plt.axhline(0, color='red', linestyle='--', label='Lower Band (0)')
    plt.axhline(1, color='green', linestyle='--', label='Upper Band (1)')
    plt.xlabel('Date')
    plt.ylabel('Bollinger Band %B')
    plt.legend()
    plt.grid()
    plt.savefig(f'images/{symbol}_BollingerBands.png')

    plt.tight_layout()
    if verbose:
        plt.show()
    else:
        plt.close()

    # 2. RSI
    plt.figure(figsize=(12, 6))
    plt.plot(rsi, label='RSI', color='purple')
    plt.axhline(70, color='red', linestyle='--', label='Overbought (70)')
    plt.axhline(30, color='green', linestyle='--', label='Oversold (30)')
    plt.title(f'Relative Strength Index (RSI) for {symbol}')
    plt.xlabel('Date')
    plt.ylabel('RSI')
    plt.legend()
    plt.grid()
    plt.savefig(f'images/{symbol}_RSI.png')
    if verbose:
        plt.show()
    else:
        plt.close()

    # 3. CCI
    plt.figure(figsize=(12, 6))
    plt.plot(cci, label='CCI', color='brown')
    plt.axhline(100, color='red', linestyle='--', label='Overbought (100)')
    plt.axhline(-100, color='green', linestyle='--', label='Oversold (-100)')
    plt.title(f'Commodity Channel Index (CCI) for {symbol}')
    plt.xlabel('Date')
    plt.ylabel('CCI')
    plt.legend()
    plt.grid()
    plt.savefig(f'images/{symbol}_CCI.png')
    if verbose:
        plt.show()
    else:
        plt.close()

    # 4. Momentum
    plt.figure(figsize=(12, 6))
    plt.plot(momentum, label='Momentum', color='orange')
    plt.axhline(0, color='black', linestyle='--')
    plt.title(f'Momentum Indicator for {symbol}')
    plt.xlabel('Date')
    plt.ylabel('Momentum')
    plt.legend()
    plt.grid()
    plt.savefig(f'images/{symbol}_Momentum.png')
    if verbose:
        plt.show()
    else:
        plt.close()

    # 5. MACD Line
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(prices, label='Price', color='blue')
    plt.title(f'Price and MACD Line for {symbol}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(macd_line, label='MACD Line', color='red')
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('MACD Line')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'images/{symbol}_MACD.png')

    if verbose:
        plt.show()
    else:
        plt.close()

    if verbose:
        print(f"Indicators for {symbol} calculated and visualized successfully.")

    return indicators


if __name__ == "__main__":
    # Example usage
    symbol = "JPM"  # Stock symbol
    sd = pd.to_datetime("2008-01-01")  # Start date
    ed = pd.to_datetime("2009-12-31")  # End date
    verbose = True  # Set to True to display charts

    # Get the adjusted close price data for the symbol
    prices = get_data([symbol], pd.date_range(sd, ed))[symbol]

    # Run indicators and generate charts
    run_indicators(symbol, prices, verbose=verbose)
