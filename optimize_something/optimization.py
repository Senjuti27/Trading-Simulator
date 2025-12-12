""""""  		  	   		 	 	 		  		  		    	 		 		   		 		  
"""MC1-P2: Optimize a portfolio.  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
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
import matplotlib.pyplot as plt  		  	   		 	 	 		  		  		    	 		 		   		 		  
import pandas as pd  		  	   		 	 	 		  		  		    	 		 		   		 		  
from util import get_data, plot_data
from scipy.optimize import minimize

def fetch_prices_for_symbols(start_date, end_date, symbols):
    """Load adjusted close prices for the selected symbols and drop missing rows."""
    dates = pd.date_range(start=start_date, end=end_date)
    prices = get_data(symbols, dates)
    return prices[symbols].dropna()

def normalize_prices(df):
    """Scale prices so that all start at 1.0 on the first day."""
    return df / df.iloc[0]

def compute_portfolio_statistics(prices, allocations):
    """
    Calculate cumulative return, average daily return, standard deviation, and Sharpe ratio for given allocations.
    """
    normed = normalize_prices(prices)
    alloced = normed * allocations
    port_val = alloced.sum(axis=1)
    daily_rets = port_val.pct_change().dropna()

    cr = port_val.iloc[-1] / port_val.iloc[0] - 1.0
    adr = daily_rets.mean()
    sddr = daily_rets.std(ddof=1)  # sample std dev
    sr = np.sqrt(252.0) * adr / sddr if sddr != 0 else 0.0

    return cr, adr, sddr, sr, port_val

def objective_negative_sharpe(allocations, prices):
    """Objective function: return negative sharpe so optimizer maximizes Sharpe."""
    return -compute_portfolio_statistics(prices, allocations)[3]

def constraint_sum_to_one(allocations):
    """Equality constraint: sum of weights must equal 1. """
    return np.sum(allocations) - 1.0

  		  	   		 	 	 		  		  		    	 		 		   		 		  
# This is the function that will be tested by the autograder  		  	   		 	 	 		  		  		    	 		 		   		 		  
# The student must update this code to properly implement the functionality
def optimize_portfolio(  		  	   		 	 	 		  		  		    	 		 		   		 		  
    sd=dt.datetime(2008, 1, 1),
    ed=dt.datetime(2009, 1, 1),
    syms=["GOOG", "AAPL", "GLD", "XOM"],
    gen_plot=False,  		  	   		 	 	 		  		  		    	 		 		   		 		  
):
    """  		  	   		 	 	 		  		  		    	 		 		   		 		  
    This function should find the optimal allocations for a given set of stocks. You should optimize for maximum Sharpe  		  	   		 	 	 		  		  		    	 		 		   		 		  
    Ratio. The function should accept as input a list of symbols as well as start and end dates and return a list of  		  	   		 	 	 		  		  		    	 		 		   		 		  
    floats (as a one-dimensional numpy array) that represents the allocations to each of the equities. You can take  		  	   		 	 	 		  		  		    	 		 		   		 		  
    advantage of routines developed in the optional assess portfolio project to compute daily portfolio value and  		  	   		 	 	 		  		  		    	 		 		   		 		  
    statistics.  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :type sd: datetime  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :param ed: A datetime object that represents the end date, defaults to 1/1/2009
    :type ed: datetime  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :param syms: A list of symbols that make up the portfolio (note that your code should support any  		  	   		 	 	 		  		  		    	 		 		   		 		  
        symbol in the data directory)  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :type syms: list  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :param gen_plot: If True, optionally create a plot named plot.png. The autograder will always call your  		  	   		 	 	 		  		  		    	 		 		   		 		  
        code with gen_plot = False.  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :type gen_plot: bool  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :return: A tuple containing the portfolio allocations, cumulative return, average daily returns,  		  	   		 	 	 		  		  		    	 		 		   		 		  
        standard deviation of daily returns, and Sharpe ratio  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :rtype: tuple  		  	   		 	 	 		  		  		    	 		 		   		 		  
    """

    if syms is None or len(syms) < 2:
        raise ValueError("You must provide at least two symbols (n >= 2).")

    prices = fetch_prices_for_symbols(sd, ed, syms)
    n = len(syms)
    x0 = np.array([1.0 / n] * n)  # start with equal weights
    bounds = tuple((0.0, 1.0) for _ in range(n))
    constraints = ({"type": "eq", "fun": constraint_sum_to_one},)

    result = minimize(objective_negative_sharpe, x0, args=(prices,), method="SLSQP", bounds=bounds,
                      constraints=constraints, options={"disp": False, "maxiter": 500})

    # Use result if optimization succeeded, otherwise fallback to uniform
    allocs = result.x if result.success else x0
    allocs = np.clip(allocs, 0.0, 1.0)
    allocs /= allocs.sum()
    # allocs = np.round(allocs, 25)

    cr, adr, sddr, sr, port_val = compute_portfolio_statistics(prices, allocs)

    if gen_plot:
        # Compare normalized portfolio vs SPY and save as Figure1.png
        dates = pd.date_range(start=sd, end=ed)
        prices_all = get_data(["SPY"] + syms, dates)
        norm_port = normalize_prices(port_val.to_frame("Portfolio"))
        spy = normalize_prices(prices_all[["SPY"]])
        df_plot = pd.concat([norm_port, spy], axis=1).dropna()
        ax = df_plot.plot(title="Daily Optimized Portfolio Value vs SPY", fontsize=10)
        ax.set_xlabel("Date")
        ax.set_ylabel("Normalized Value")
        ax.legend(["Optimized Portfolio", "SPY"])
        plt.tight_layout()
        ax.grid(True)
        plt.savefig("Figure1.png")
        plt.close()

    return allocs, cr, adr, sddr, sr


def test_code():  		  	   		 	 	 		  		  		    	 		 		   		 		  
    """  		  	   		 	 	 		  		  		    	 		 		   		 		  
    This function WILL NOT be called by the auto grader.  		  	   		 	 	 		  		  		    	 		 		   		 		  
    """
  	#

    sd = dt.datetime(2008, 6, 1)
    ed = dt.datetime(2009, 6, 1)
    syms = ['IBM', 'X', 'GLD', 'JPM']

    allocations, cr, adr, sddr, sr = optimize_portfolio(sd, ed, syms, gen_plot= False)

    # Print statistics  		  	   		 	 	 		  		  		    	 		 		   		 		  
    print(f"Start Date: {sd}")
    print(f"End Date: {ed}")
    print(f"Symbols: {syms}")
    print(f"Allocations:{allocations}")  		  	   		 	 	 		  		  		    	 		 		   		 		  
    print(f"Sharpe Ratio: {sr}")  		  	   		 	 	 		  		  		    	 		 		   		 		  
    print(f"Volatility (stdev of daily returns): {sddr}")  		  	   		 	 	 		  		  		    	 		 		   		 		  
    print(f"Average Daily Return: {adr}")  		  	   		 	 	 		  		  		    	 		 		   		 		  
    print(f"Cumulative Return: {cr}")  		  	   		 	 	 		  		  		    	 		 		   		 		  

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

if __name__ == "__main__":
    # This code WILL NOT be called by the auto grader  		  	   		 	 	 		  		  		    	 		 		   		 		  
    # Do not assume that it will be called  		  	   		 	 	 		  		  		    	 		 		   		 		  
    test_code()


