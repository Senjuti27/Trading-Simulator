"""
Code implementing a TheoreticallyOptimalStrategy (details below).
It should implement testPolicy(), which returns a trades data frame (see below).
"""

import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from util import get_data, plot_data

def author():
    return "stwisha3"

def study_group():
    return "stwisha3"

def testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009,12,31), sv = 100000):
    df_prices = get_data([symbol], pd.date_range(sd,ed), addSPY=True, colname="Adj Close") #Loading price data and keeping SPY for valid days
    df_prices = df_prices.drop(['SPY'], axis=1) #Removing SPY
    df_prices["diff"] = df_prices[symbol].shift(-1) - df_prices[symbol]   #Shift(-1) moves all prices up by one day (to look ahead).
    df_prices["position"] = df_prices["diff"]/abs(df_prices["diff"]) #Now, +1 = price will rise tomorrow, –1 = price will fall.
    df_prices.fillna(method='bfill', inplace= True) #Fills the final NaN value (at the end) using the next valid one (backfill).
    df_prices["prev_pos"] = df_prices["position"].shift(1)

    # print(df_prices)

    # Create the df_trades DataFrame (initialized to 0)
    df_trades = pd.DataFrame(index=df_prices.index, columns=[symbol], data=0.0)

    # Go through each day in the dataset
    for i in range(len(df_prices)):
        # --- CASE 1: First day ---
        if i == 0:
            # Enter initial position: buy/sell 1000 depending on position sign
            df_trades.iloc[i, 0] = df_prices.iloc[i]["position"] * 1000

        # --- CASE 2: Last day ---
        elif i == len(df_prices) - 1:
            # Close all positions (we can’t see the future)
            df_trades.iloc[i, 0] = 0.0

        # --- CASE 3: Signal flipped ---
        elif df_prices.iloc[i]["position"] != df_prices.iloc[i]["prev_pos"]:
            # Flip the trade: double the previous position in opposite direction
            df_trades.iloc[i, 0] = df_prices.iloc[i]["prev_pos"] * -2000

        # --- CASE 4: No change in position ---
        else:
            # Hold position → no trade today
            df_trades.iloc[i, 0] = 0.0

    # print(df_trades)

    return df_trades





