""""""  		  	   		 	   		  		  		    	 		 		   		 		  
"""  		  	   		 	   		  		  		    	 		 		   		 		  
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
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

Student Name: Aditya Kommi
GT User ID: akommi3
GT ID: 903135337
"""

import datetime as dt
import random

import pandas as pd
import util as ut
from indicators import compute_bollinger_bands, compute_rsi, compute_macd
from QLearner import QLearner

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


class StrategyLearner(object):
    """
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.

    :param verbose: If “verbose” is True, your code can print out information for debugging.
        If verbose = False your code should not generate ANY output.
    :type verbose: bool
    :param impact: The market impact of each transaction, defaults to 0.0
    :type impact: float
    :param commission: The commission amount charged, defaults to 0.0
    :type commission: float
    """
    # constructor
    def __init__(self, verbose=False, impact=0.0, commission=0.0):
        """
        Constructor method
        """
        self.verbose = verbose
        self.impact = impact
        self.commission = commission
        self.learner = None
        self.num_bins = 10

    # this method should create a QLearner, and train it for trading
    def add_evidence(
        self,
        symbol="IBM",
        sd=dt.datetime(2008, 1, 1),
        ed=dt.datetime(2009, 1, 1),
        sv=10000,
    ):
        """
        Trains your strategy learner over a given time frame.

        :param symbol: The stock symbol to train on
        :type symbol: str
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008
        :type sd: datetime
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009
        :type ed: datetime
        :param sv: The starting value of the portfolio
        :type sv: int
        """

        # add your code to do learning here

        # example usage of the old backward compatible util function
        syms = [symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices_all = prices_all.fillna(method='ffill').fillna(method='bfill')
        prices = prices_all[symbol]  # only portfolio symbols
        prices_SPY = prices_all["SPY"]  # only SPY, for comparison later
        if self.verbose:
            print(prices)

        # example use with new colname
        volume_all = ut.get_data(
            syms, dates, colname="Volume"
        )  # automatically adds SPY
        volume = volume_all[syms]  # only portfolio symbols
        volume_SPY = volume_all["SPY"]  # only SPY, for comparison later
        if self.verbose:
            print(volume)

        # Compute indicators
        bbp = compute_bollinger_bands(prices, window=20)
        rsi = compute_rsi(prices, window=14)
        macd_line = compute_macd(prices)

        # Discretize indicators
        indicators = pd.DataFrame({
            'BBP': bbp,
            'RSI': rsi,
            'MACD': macd_line,
        }).fillna(method='ffill').fillna(method='bfill')

        # Binning
        bbp_bins = pd.qcut(indicators['BBP'], self.num_bins, labels=False, duplicates='drop')
        rsi_bins = pd.qcut(indicators['RSI'], self.num_bins, labels=False, duplicates='drop')
        macd_bins = pd.qcut(indicators['MACD'], self.num_bins, labels=False, duplicates='drop')

        # Create states
        states = bbp_bins * (self.num_bins ** 2) + rsi_bins * self.num_bins + macd_bins
        states = states.astype(int)

        # Initialize QLearner
        num_states = self.num_bins ** 3  # Total possible combinations
        num_actions = 3  # Short, Hold, Long
        self.learner = QLearner(
            num_states=num_states,
            num_actions=num_actions,
            alpha=0.2,
            gamma=0.9,
            rar=0.9,
            radr=0.999,
            dyna=0,
            verbose=self.verbose,
        )

        # Training loop
        epochs = 250  # Number of training epochs
        for epoch in range(epochs):
            if self.verbose:
                print(f"Epoch {epoch + 1}/{epochs}")
            holdings = 0  # Current position
            cash = sv  # Starting cash
            for i in range(len(prices) - 1):
                state = states.iloc[i]
                action = self.learner.querysetstate(state)
                # Map action to trade
                desired_holdings = holdings
                if action == 0:  # Short
                    desired_holdings = -1000
                elif action == 1:  # Hold
                    desired_holdings = holdings
                elif action == 2:  # Long
                    desired_holdings = 1000

                trade = desired_holdings - holdings
                holdings += trade
                trade_cost = abs(trade * prices.iloc[i] * self.impact) + self.commission

                # Calculate reward
                daily_return = holdings * (prices.iloc[i + 1] - prices.iloc[i])
                reward = daily_return - trade_cost
                reward = reward / sv  # Normalize reward

                # Get next state
                next_state = states.iloc[i + 1]
                self.learner.query(next_state, reward)


    # this method should use the existing policy and test it against new data
    def testPolicy(
        self,
        symbol="IBM",
        sd=dt.datetime(2009, 1, 1),
        ed=dt.datetime(2010, 1, 1),
        sv=10000,
    ):
        """
        Tests your learner using data outside of the training data

        :param symbol: The stock symbol that you trained on on
        :type symbol: str
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008
        :type sd: datetime
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009
        :type ed: datetime
        :param sv: The starting value of the portfolio
        :type sv: int
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to
            long so long as net holdings are constrained to -1000, 0, and 1000.
        :rtype: pandas.DataFrame
        """

        # here we build a fake set of trades
        # your code should return the same sort of data
        # dates = pd.date_range(sd, ed)
        # prices_all = ut.get_data([symbol], dates)  # automatically adds SPY
        # trades = prices_all[[symbol,]]  # only portfolio symbols
        # trades_SPY = prices_all["SPY"]  # only SPY, for comparison later
        # trades.values[:, :] = 0  # set them all to nothing
        # trades.values[0, :] = 1000  # add a BUY at the start
        # trades.values[40, :] = -1000  # add a SELL
        # trades.values[41, :] = 1000  # add a BUY
        # trades.values[60, :] = -2000  # go short from long
        # trades.values[61, :] = 2000  # go long from short
        # trades.values[-1, :] = -1000  # exit on the last day
        # if self.verbose:
        #     print(type(trades))  # it better be a DataFrame!
        # if self.verbose:
        #     print(trades)
        # if self.verbose:
        #     print(prices_all)

        dates = pd.date_range(sd, ed)
        syms = [symbol]
        prices_all = ut.get_data(syms, dates)
        prices = prices_all[symbol]
        prices = prices.fillna(method='ffill').fillna(method='bfill')
        # trades = prices_all[[symbol,]]  # only portfolio symbols
        # trades_SPY = prices_all["SPY"]  # only SPY, for comparison later

        # Compute indicators
        bbp = compute_bollinger_bands(prices, window=20)
        rsi = compute_rsi(prices, window=14)
        macd_line = compute_macd(prices)

        # Prepare the indicators DataFrame
        indicators = pd.DataFrame({
            'BBP': bbp,
            'RSI': rsi,
            'MACD': macd_line,
        })
        indicators = indicators.fillna(method='ffill').fillna(method='bfill')

        # Discretize indicators
        bbp_bins = pd.qcut(indicators['BBP'], self.num_bins, labels=False, duplicates='drop')
        rsi_bins = pd.qcut(indicators['RSI'], self.num_bins, labels=False, duplicates='drop')
        macd_bins = pd.qcut(indicators['MACD'], self.num_bins, labels=False, duplicates='drop')

        # Create states
        states = bbp_bins * (self.num_bins ** 2) + rsi_bins * self.num_bins + macd_bins
        states = states.astype(int)

        # Set rar to 0
        self.learner.rar = 0.0

        # Simulation
        holdings = 0
        trades = pd.DataFrame(data=0, index=prices.index, columns=[symbol])
        trades.iloc[:, :] = 0

        for i in range(len(prices) - 1):
            state = states.iloc[i]
            action = self.learner.querysetstate(state)
            # Map action to desired holdings
            desired_holdings = holdings
            if action == 0:  # Short
                desired_holdings = -1000
            elif action == 1:  # Hold
                desired_holdings = holdings
            elif action == 2:  # Long
                desired_holdings = 1000

            trade = desired_holdings - holdings
            trades.iloc[i, 0] = trade
            holdings += trade

        if self.verbose:
            print(type(trades))  # it better be a DataFrame!
        if self.verbose:
            print(trades)
        if self.verbose:
            print(prices_all)

        return trades

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
    print("One does not simply think up a strategy")
