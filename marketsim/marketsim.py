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
  		  	   		 	 	 		  		  		    	 		 		   		 		  
Student Name: Senjuti Twisha 		  	   		 	 	 		  		  		    	 		 		   		 		  
GT User ID: stwisha3  		  	   		 	 	 		  		  		    	 		 		   		 		  
GT ID: 904080731 	 		  	   		 	 	 		  		  		    	 		 		   		 		  
"""  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
import datetime as dt
import numpy as np
import pandas as pd  		  	   		 	 	 		  		  		    	 		 		   		 		  
from util import get_data, plot_data  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
def compute_portvals(  		  	   		 	 	 		  		  		    	 		 		   		 		  
    orders_file="./orders/orders.csv",  		  	   		 	 	 		  		  		    	 		 		   		 		  
    start_val=1000000,  		  	   		 	 	 		  		  		    	 		 		   		 		  
    commission=9.95,  		  	   		 	 	 		  		  		    	 		 		   		 		  
    impact=0.005,  		  	   		 	 	 		  		  		    	 		 		   		 		  
):  		  	   		 	 	 		  		  		    	 		 		   		 		  
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
    

    orders = pd.read_csv(orders_file,
                         index_col= 'Date',
                         parse_dates= True,
                         na_values=['nan']
                         )
    orders = orders.sort_index()
    start_date = orders.index[0]
    end_date = orders.index[-1]
    symbols = orders['Symbol'].values
    symbols = np.unique(symbols)
    # print(orders, "\n", start_date, end_date, "\n", symbols)

    dates = pd.date_range(start_date,end_date)
    df_prices = get_data(symbols,dates,addSPY= True, colname="Adj Close")
    if 'SPY' not in symbols:
        df_prices = df_prices.drop('SPY', axis=1)
    df_prices['Cash'] = 1
    # print(df_prices)

    df_trades = df_prices * 0
    # print(df_trades)
    for date, row in orders.iterrows():
        sym = row['Symbol']
        order_type = row['Order']
        shares_amount = row['Shares']
        # print(sym,order_type,shares_amount)
        if order_type == "BUY":
            df_trades.loc[date, sym] += shares_amount
            p = df_prices.loc[date, sym]
            df_trades.loc[date, "Cash"] += -1 * (shares_amount * p * (1+impact) + commission)
        elif order_type == "SELL":
            df_trades.loc[date, sym] += -1 * shares_amount
            p = df_prices.loc[date, sym]
            df_trades.loc[date, "Cash"] += shares_amount * p * (1 - impact) - commission
    # print(df_trades)

    df_holding = df_trades.copy()
    df_holding.iloc[:,:] = 0
    df_holding.iloc[0,:] = df_trades.iloc[0,:]
    df_holding.iloc[0,-1] += start_val
    for i in range(1, df_trades.shape[0]):
        for j in range(df_trades.shape[1]):
            prev = df_holding.iloc[i-1, j]
            curr = df_trades.iloc[i, j]
            df_holding.iloc[i,j] = prev+curr
    # print(df_holding)

    df_values = df_holding * df_prices
    df_values["port_val"] = df_values.sum(axis = 1)
    # print(df_values)

    df_values["daily_returns"] = df_values["port_val"].pct_change()
    df_values.loc[df_values.index[0], "daily_returns"] = 0
    portvals = df_values[["port_val"]]
    print(df_values)
    print(portvals)
    return portvals

def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "stwisha3"  # replace tb34 with your Georgia Tech username.

def study_group():
    """
    :return: A comma-separated string of GT usernames of your study group members.
             If working alone, just return your username.
    :rtype: str
    """
    return "stwisha3"  # replace/add other usernames if you have a study group
  		  	   		 	 	 		  		  		    	 		 		   		 		  
def test_code():  		  	   		 	 	 		  		  		    	 		 		   		 		  
    """  		  	   		 	 	 		  		  		    	 		 		   		 		  
    Helper function to test code  		  	   		 	 	 		  		  		    	 		 		   		 		  
    """  		  	   		 	 	 		  		  		    	 		 		   		 		  
    # this is a helper function you can use to test your code  		  	   		 	 	 		  		  		    	 		 		   		 		  
    # note that during autograding his function will not be called.  		  	   		 	 	 		  		  		    	 		 		   		 		  
    # Define input parameters  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
    of = "./orders/orders-01.csv"
    # of = "./additional_orders/orders-short.csv"
    # of = "./testcases_mc2p1/orders-leverage-2.csv"
    sv = 1000000  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
    # Process orders  		  	   		 	 	 		  		  		    	 		 		   		 		  
    portvals = compute_portvals(orders_file=of, start_val=sv)  		  	   		 	 	 		  		  		    	 		 		   		 		  
    if isinstance(portvals, pd.DataFrame):  		  	   		 	 	 		  		  		    	 		 		   		 		  
        portvals = portvals[portvals.columns[0]]  # just get the first column  		  	   		 	 	 		  		  		    	 		 		   		 		  
    else:  		  	   		 	 	 		  		  		    	 		 		   		 		  
        "warning, code did not return a DataFrame"

    start_date = portvals.index[0]
    end_date = portvals.index[-1]
    # Daily Returns: from 2nd row divided by previous row minus 1
    daily_ret = portvals[1:] / portvals[:-1].values - 1
    # Cumulative Return: (ending / starting) - 1
    cum_ret = portvals.iloc[-1] / portvals.iloc[0] - 1
    # Average and Standard Deviation of Daily Returns
    avg_daily_ret = daily_ret.mean()
    std_daily_ret = daily_ret.std()
    # Sharpe Ratio (risk-free rate = 0)
    sharpe_ratio = np.sqrt(252) * (avg_daily_ret / std_daily_ret)
    # Ending value
    end_value = portvals.iloc[-1]

    # Now compare against $SPX
    prices_SPX = get_data(['$SPX'], pd.date_range(start_date, end_date))
    prices_SPX = prices_SPX[['$SPX']]  # Only keep SPX column

    # Simulate SPX portfolio (start_val = 1,000,000, 100% allocation to SPX)
    normed_SPX = prices_SPX / prices_SPX.iloc[0]
    alloced_SPX = normed_SPX * 1.0
    portvals_SPX = alloced_SPX * 1000000

    # Compute SPX stats directly
    daily_ret_SPX = portvals_SPX.pct_change().fillna(0)
    cum_ret_SPX = (portvals_SPX.iloc[-1, 0] / portvals_SPX.iloc[0, 0]) - 1
    avg_daily_ret_SPX = daily_ret_SPX.mean()[0]
    std_daily_ret_SPX = daily_ret_SPX.std()[0]
    sharpe_ratio_SPX = np.sqrt(252) * avg_daily_ret_SPX / std_daily_ret_SPX

    # Compare portfolio against $SPX
    print(f"Date Range: {start_date} to {end_date}")
    print()
    print(f"Sharpe Ratio of Fund: {sharpe_ratio}")
    print(f"Sharpe Ratio of SPX : {sharpe_ratio_SPX}")
    print()
    print(f"Cumulative Return of Fund: {cum_ret}")
    print(f"Cumulative Return of SPX : {cum_ret_SPX}")
    print()
    print(f"Standard Deviation of Fund: {std_daily_ret}")
    print(f"Standard Deviation of SPX : {std_daily_ret_SPX}")
    print()
    print(f"Average Daily Return of Fund: {avg_daily_ret}")
    print(f"Average Daily Return of SPX : {avg_daily_ret_SPX}")
    print()
    print(f"Final Portfolio Value: {portvals[-1]}")

    # Plot portfolio value over time using plot_data
    # plot_data(portvals, title="Portfolio Value Over Time", ylabel="Portfolio Value ($)")

  		  	   		 	 	 		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		 	 	 		  		  		    	 		 		   		 		  
    test_code()  		  	   		 	 	 		  		  		    	 		 		   		 		  
