"""
An improved version of your marketsim code accepts a “trades” DataFrame (instead of a file).
More info on the trades data frame is below. It is OK not to submit this file
if you have subsumed its functionality into one of your other required code files.
This file has a different name and a slightly different setup than your previous project.
However, that solution can be used with several edits for the new requirements.
"""

import numpy as np
import TheoreticallyOptimalStrategy as tos
import pandas as pd
from util import get_data

def author():
    return "stwisha3"

def study_group():
    return "stwisha3"

# order = tos.testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009,12,31), sv = 100000)

def compute_portvals(orders, start_val=100000, commission= 0.00, impact=0.00):
    orders = orders.sort_index()        # Sort by date (keep chronological order)
    start_date = orders.index[0]
    end_date = orders.index[-1]
    symbols = orders.columns.tolist()   # e.g., ['JPM']
    symbols = np.unique(symbols)        # keep unique symbols

    # symbols = orders['Symbol'].values
    # print(orders, "\n", start_date, end_date, "\n", symbols)

    # Get price data for those symbols
    dates = pd.date_range(start_date, end_date)
    df_prices = get_data(symbols, dates, addSPY=True, colname="Adj Close")

    # Drop SPY if not in portfolio
    if 'SPY' not in symbols:
        df_prices = df_prices.drop('SPY', axis=1)

    # Add 'Cash' column to track cash balance
    df_prices['Cash'] = 1
    # print(df_prices)

    df_trades = df_prices * 0
    # print(df_trades)

    for date, row in orders.iterrows():
        for sym in symbols:
            shares_amount = row[sym]  # get trade amount directly
            if shares_amount == 0:
                continue  # skip if no trade
            p = df_prices.loc[date, sym]
            # update symbol holdings
            df_trades.loc[date, sym] += shares_amount
            # update cash (buy→subtract, sell→add)
            df_trades.loc[date, "Cash"] += -1 * shares_amount * p * (1 + np.sign(shares_amount) * impact)
            df_trades.loc[date, "Cash"] -= commission

    # print(df_trades)

    # Compute cumulative holdings
    df_holding = df_trades.copy()
    df_holding.iloc[:, :] = 0
    df_holding.iloc[0, :] = df_trades.iloc[0, :]
    df_holding.iloc[0, -1] += start_val
    for i in range(1, df_trades.shape[0]):
        for j in range(df_trades.shape[1]):
            prev = df_holding.iloc[i - 1, j]
            curr = df_trades.iloc[i, j]
            df_holding.iloc[i, j] = prev + curr
    # print(df_holding)

    # Compute portfolio value (holdings × prices)
    df_values = df_holding * df_prices
    df_values["port_val"] = df_values.sum(axis=1)
    # print(df_values)

    # Compute daily returns
    df_values["daily_returns"] = df_values["port_val"].pct_change()
    df_values.loc[df_values.index[0], "daily_returns"] = 0

    # Final portfolio values DataFrame
    df_portvals = df_values[["port_val"]]

    # print(df_portvals)
    return df_portvals


