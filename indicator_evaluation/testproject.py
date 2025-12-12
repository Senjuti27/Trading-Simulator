"""
This file should be considered the entry point to the project.
The if __name__ == "__main__":section of the code will call the testPolicy function in TheoreticallyOptimalStrategy,
as well as your indicators and marketsimcode as needed,
to generate the plots and statistics for your report (more details below).
"""
import numpy as np
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import TheoreticallyOptimalStrategy as tos
import marketsimcode as ms
import indicators as id
from util import get_data

def author():
    return "stwisha3"

def study_group():
    return "stwisha3"

if __name__ == "__main__":
    symbol = "JPM"
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009,12,31)
    sv = 100000
    """
    ---------------------Indicators----------------------------
    """
    df_prices_id = get_data([symbol],pd.date_range(sd,ed))
    # print(df_prices_id.head())
    # print(df_prices_id.tail())
    df_ret_bb = id.bollinger_band(df_prices_id, windows=20, k=2, plot = True)
    # print("df_ret_bb")
    # print(df_ret_bb)
    df_ret_rsi = id.rsi(df_prices_id, windows= 14, plot = True)
    # print("df_ret_rsi")
    # print(df_ret_rsi)
    df_ret_macd = id.macd(df_prices_id, fast=12, slow=26, signal= 9, plot=True)
    # print("df_ret_macd")
    # print(df_ret_macd)
    df_ret_stoc_k = id.stoc_k(df_prices_id, windows= 14, plot=True)
    # print("df_ret_stoc_k")
    # print(df_ret_stoc_k)
    df_ret_cci = id.cci(df_prices_id, windows=14, plot=True)
    # print("df_ret_cci")
    # print(df_ret_cci)
    """
    ---------------------------Benchmark Portfolio Performance------------
    """
    dates = pd.date_range(sd, ed)
    prices = get_data([symbol], dates)
    prices = prices[[symbol]]
    # Create benchmark order: Buy 1000 on first day
    benchmark_orders = pd.DataFrame(0, index=prices.index, columns=[symbol])
    benchmark_orders.iloc[0, 0] = 1000
    # Compute benchmark portfolio
    benchmark_portvals = ms.compute_portvals(benchmark_orders, sv)
    # Compute stats
    cr_b = ((benchmark_portvals.iloc[-1].values[0]) / benchmark_portvals.iloc[0].values[0]) - 1
    adr_b = benchmark_portvals.pct_change().mean()[0]
    sddr_b = benchmark_portvals.pct_change().std()[0]
    sr_b = (adr_b / sddr_b) * np.sqrt(252)

    """
    ---------------------------TOS Portfolio Performance------------
    """
    # Get trades from TOS
    df_trades = tos.testPolicy(symbol, sd, ed, sv)
    # Compute portfolio values from trades
    df_portvals = ms.compute_portvals(df_trades, sv)
    #  Normalize portfolio for plotting (optional)
    df_portvals_norm = df_portvals / df_portvals.iloc[0]
    # Compute daily returns (drop first NaN)
    daily_ret_tos = df_portvals.pct_change().dropna()
    # Compute cumulative return
    cum_ret_tos = (df_portvals.iloc[-1] / df_portvals.iloc[0]) - 1
    # Compute average and std dev of daily return
    avg_daily_ret_tos = daily_ret_tos.mean()
    std_daily_ret_tos = daily_ret_tos.std()
    # Compute Sharpe ratio (annualized, risk-free rate = 0)
    sharpe_tos = np.sqrt(252) * (avg_daily_ret_tos / std_daily_ret_tos)


    # === Performance Comparison Table ===
    # print("\n==== Portfolio Performance Comparison ====")
    # print(f"{'Metric':<30}{'Benchmark':>20}{'TOS (Best Possible)':>25}")
    # print("-" * 75)
    # print(f"{'Cumulative Return':<30}{cr_b:.6f}{cum_ret_tos.iloc[0]:>25.6f}")
    # print(f"{'Average Daily Return':<30}{adr_b:.6f}{avg_daily_ret_tos.iloc[0]:>25.6f}")
    # print(f"{'Std Dev Daily Return':<30}{sddr_b:.6f}{std_daily_ret_tos.iloc[0]:>25.6f}")
    # print(f"{'Sharpe Ratio':<30}{sr_b:.6f}{sharpe_tos.iloc[0]:>25.6f}")

    """
    ---------------------------Plot: TOS vs Benchmark-------------------------------------
    """
    # Normalize both to start at 1.0
    norm_bench = benchmark_portvals / benchmark_portvals.iloc[0]
    norm_tos = df_portvals / df_portvals.iloc[0]
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(norm_bench, label="Benchmark", color="purple", linewidth=2)
    plt.plot(norm_tos, label="TOS Portfolio", color="red", linewidth=2)
    # Titles and labels
    plt.title(f"TOS vs Benchmark: Normalized Portfolio Comparison ({symbol})", fontsize=14)
    plt.xlabel("Date")
    plt.ylabel("Normalized Value")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    # Save and show
    plt.savefig("./images/TOS_vs_Benchmark.png")
    # plt.show()
    plt.close()

