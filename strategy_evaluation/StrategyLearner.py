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
  		  	   		 	 	 		  		  		    	 		 		   		 		  
Student Name: Senjuti Twisha 		  	   		 	 	 		  		  		    	 		 		   		 		  
GT User ID: stwisha3  		  	   		 	 	 		  		  		    	 		 		   		 		  
GT ID: 904080731 		  	   		 	 	 		  		  		    	 		 		   		 		  
"""  		  	   		 	 	 		  		  		    	 		 		   		 		  

import datetime as dt
import numpy as np
import random
import pandas as pd
import util as ut
import RTLearner as rt
import BagLearner as bl
import indicators as id

class StrategyLearner(object):
    """
    Implements a learning-based trading strategy that trains a Random Forest (bag of RTLearners)
    on technical indicators to predict future N-day returns and generate legal daily trades.
    """

    def __init__(self, verbose=False, impact=0.0, commission=0.0):
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
        self.verbose = verbose
        self.impact = impact  		  	   		 	 	 		  		  		    	 		 		   		 		  
        self.commission = commission

        # Create BagLearner of RTLearners
        self.learner = bl.BagLearner(
            learner=rt.RTLearner,
            kwargs={"leaf_size": 10},
            bags= 40,
            boost=False,
            verbose=False
        )

    def add_evidence(self, symbol='IBM', sd=dt.datetime(2008, 1, 1, 0, 0), ed=dt.datetime(2009, 1, 1, 0, 0), sv=100000):
        """  		  	   		 	 	 		  		  		    	 		 		   		 		  
        Trains your strategy learner over a given time frame.
        Parameters
            symbol (str) – The stock symbol to train on
            sd (datetime) – A datetime object that represents the start date, defaults to 1/1/2008
            ed (datetime) – A datetime object that represents the end date, defaults to 1/1/2009
            sv (int) – The starting value of the portfolio
        """

        # 1) Load price data
        syms = [symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)   # automatically adds SPY
        prices = prices_all[syms]               # only portfolio symbols
        prices_SPY = prices_all["SPY"]          # only SPY, for comparison later

        # Fix missing data: ffill → push last valid value forward, bfill → fill at the beginning if needed
        prices = prices.fillna(method="ffill").fillna(method="bfill")
        normed_prices = prices / prices.iloc[0] # normalize for return calculation

        # 2) Compute five indicators on price data
        bbp_df = id.bollinger_band(prices.copy(), windows=20, k=2, plot=False)
        rsi_df = id.rsi(prices.copy(), windows=14, plot=False)
        # macd_df = id.macd(prices.copy(), fast=12, slow=26, signal=9, plot=False)
        k_df = id.stoc_k(prices.copy(), windows=14, plot=False)
        cci_df = id.cci(prices.copy(), windows =14, plot=False)

        # 3) Build feature matrix X from the indicators
        ind_df = pd.DataFrame(index=prices.index)
        ind_df["BBP"] = bbp_df["BBP"]
        ind_df["RSI"] = rsi_df["RSI"]
        # ind_df["MACD_Hist"] = macd_df["MACD_Diff"]
        ind_df["K"] = k_df["Stoc_K"]
        ind_df["CCI"] = cci_df["CCI"]

        ind_df = ind_df.fillna(0)   # Indicators have "warm-up" periods at the start where they are NaN, replace any NaNs with 0

        # 4) Choose how many days ahead we look to define "future return".
        lookback = 5
        x_train = ind_df.iloc[:-lookback].values    # We will create X_train from all rows *except* the last 'lookback' days because they do not have enough future data

        # 5) Compute N-day forward return for each day inside the training window.
        price_arr = normed_prices.values  # # price_arr is a simple 1D array of normalized prices.
        future_ret = (price_arr[lookback:] / price_arr[:-lookback]) - 1.0 # This gives the percentage movement between day t and t+lookback.

        # 6) Turn these future returns into labels y in {-1, 0, +1}.
        thresh = 0.015 # We use a threshold of 1.5% (0.015)
        buy_signal = (future_ret > (thresh + self.impact)).astype(int) # large enough to overcome trading costs/impact.
        sell_signal = (future_ret < -(thresh + self.impact)).astype(int)
        y_train = (buy_signal - sell_signal)    # y_train = +1 → LONG,  -1 → SHORT,  0 → CASH

        # 7) Finally, we pass features X and labels y to the BagLearner.
        self.learner.add_evidence(x_train, y_train) #  Inside BagLearner, multiple RTLearners will be trained on bootstrapped samples of this (X, y).

    def testPolicy(self, symbol='IBM', sd=dt.datetime(2009, 1, 1, 0, 0), ed=dt.datetime(2010, 1, 1, 0, 0), sv=100000):
        """  		  	   		 	 	 		  		  		    	 		 		   		 		  
        Tests your learner using data outside of the training data
        Parameters
            symbol (str) – The stock symbol that you trained on on
            sd (datetime) – A datetime object that represents the start date, defaults to 1/1/2008
            ed (datetime) – A datetime object that represents the end date, defaults to 1/1/2009
            sv (int) – The starting value of the portfolio
        Returns
            A single column data frame, indexed by date representing trades for each day. Legal values are +1000.0 indicating
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to
            long so long as net holdings are constrained to -1000, 0, and 1000.

        Return type
            pandas.DataFrame
        """

        # 1) Load price data for the test period

        syms = [symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)  # loads symbol + SPY
        prices = prices_all[syms]  # keep only our symbol
        prices = prices.fillna(method="ffill").fillna(method="bfill")

        # 2) Compute the same 5 indicators used in training

        bbp_df = id.bollinger_band(prices.copy(), windows=20, k=2, plot=False)
        rsi_df = id.rsi(prices.copy(), windows=14, plot=False)
        # macd_df = id.macd(prices.copy(), fast=12, slow=26, signal=9, plot=False)
        k_df = id.stoc_k(prices.copy(), windows=14, plot=False)
        cci_df = id.cci(prices.copy(), windows=14, plot=False)

        # 3) Build X_test = indicator matrix for all test days

        ind_df = pd.DataFrame(index=prices.index)
        ind_df["BBP"] = bbp_df["BBP"]
        ind_df["RSI"] = rsi_df["RSI"]
        # ind_df["MACD_Hist"] = macd_df["MACD_Diff"]
        ind_df["K"] = k_df["Stoc_K"]
        ind_df["CCI"] = cci_df["CCI"]

        # 4) Use the trained BagLearner to predict actions for each day

        ind_df = ind_df.fillna(0)   # Replace NaN indicator values with 0 (neutral)
        lookback = 5  # same value you used in training

        # X_test = ind_df.values      # Convert the DataFrame to a plain NumPy array
        X_test = ind_df.iloc[:-lookback].values

        # BagLearner already returns classification labels ---
        y_pred = self.learner.query(X_test).astype(int)
        y_pred = np.clip(y_pred, -1, 1)

        n = len(prices) - lookback
        testY = y_pred[:n]  # keep first n predictions

        trades = prices_all[syms].copy()
        trades.loc[:, :] = 0  # start with all zeros

        # 5) Efficient state machine using desired_shares

        desired_shares = testY * 1000  # -1→-1000, 0→0, +1→1000
        trade_series = np.zeros(n, dtype=int)

        share = 0
        for i in range(n):
            target = desired_shares[i]
            trade = np.clip(target - share, -2000, 2000)
            trade_series[i] = trade
            share += trade

        # Write back into the DataFrame (only for the first n dates)
        trades.iloc[:n, 0] = trade_series

        return trades

    def author(self):
        return "stwisha3"

    def study_group(self):
        return "stwisha3"