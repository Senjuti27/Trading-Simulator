"""
This file is the entry point for the project.
It calls ManualStrategy, computes portfolio values, benchmarks performance, produces required plots, and reports statistics.
"""

import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from util import get_data
from ManualStrategy import ManualStrategy
from StrategyLearner import StrategyLearner
from marketsimcode import compute_portvals
import experiment1 as e1
import experiment2 as e2

def author():
    return "stwisha3"

def study_group():
    return "stwisha3"

def run_manual_strategy_period(verbose, symbol, sd, ed, sv, commission, impact, img_filename, label):
    """
    Runs Manual Strategy and Benchmark for a given date range.

    Steps:
    1. Call ManualStrategy.testPolicy() → get trades DataFrame
    2. Compute portfolio values using compute_portvals()
    3. Build benchmark trades (buy 1000 shares on day 1, hold)
    4. Compute portfolio values for benchmark
    5. Normalize both for plotting
    6. Plot both curves + LONG/SHORT vertical lines
    7. Compute performance stats (CR, ADR, SDDR, Sharpe)
    """

    # --------------------- Run Manual Strategy ---------------------

    ma_st = ManualStrategy(verbose=False, impact=impact, commission=commission)
    ms_trades = ma_st.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)
    portvals_ms = compute_portvals(ms_trades, start_val=sv, commission=commission, impact=impact)

    # --------------------- Benchmark (Buy & Hold 1000 shares) ---------------------

    dates = pd.date_range(sd, ed)
    prices = get_data([symbol], dates, addSPY=True, colname="Adj Close")

    # Create empty trades file with same dates
    bm_trades = pd.DataFrame(0, index=prices.index, columns=[symbol])

    # Buy 1000 shares on the first day, hold to the end
    first_day = prices.index[0]
    bm_trades.loc[first_day, symbol] = 1000

    portvals_bm = compute_portvals(bm_trades, start_val=sv,commission=commission, impact=impact)

    # Normalize both to 1.0 at start for plotting
    norm_ms = portvals_ms["port_val"] / portvals_ms["port_val"].iloc[0]
    norm_bm = portvals_bm["port_val"] / portvals_bm["port_val"].iloc[0]

    # --------------------- Plot Manual Strategy vs Benchmark ---------------------
    if verbose == True:

        ax = norm_bm.plot(label="Benchmark", color="purple", figsize=(12, 6))
        norm_ms.plot(ax=ax, label="Manual Strategy", color="red")

        # Long signals → blue dashed lines
        long_dates = ms_trades[ms_trades[symbol] > 0].index
        for d in long_dates:
            ax.axvline(d, color="blue", linestyle="--", alpha=0.4)

        # Short signals → black dashed lines
        short_dates = ms_trades[ms_trades[symbol] < 0].index
        for d in short_dates:
            ax.axvline(d, color="black", linestyle="--", alpha=0.4)

        ax.plot([], [], color="blue", linestyle="--", label="Long Entry")
        ax.plot([], [], color="black", linestyle="--", label="Short Entry")

        ax.set_title(f"Manual Strategy vs Benchmark ({label})")
        ax.set_xlabel("Date")
        ax.set_ylabel("Normalized Portfolio Value")
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(f"images/{img_filename}")
        plt.close()

    # --------------------- Compute Stats for Report ---------------------

    # For Manual Strategy
    cr_ms = portvals_ms.iloc[-1]["port_val"] / portvals_ms.iloc[0]["port_val"] - 1
    adr_ms = portvals_ms["port_val"].pct_change().mean()
    sddr_ms = portvals_ms["port_val"].pct_change().std()
    sr_ms = (adr_ms / sddr_ms) * np.sqrt(252)

    # For Benchmark
    cr_bm = portvals_bm.iloc[-1]["port_val"] / portvals_bm.iloc[0]["port_val"] - 1
    adr_bm = portvals_bm["port_val"].pct_change().mean()
    sddr_bm = portvals_bm["port_val"].pct_change().std()
    sr_bm = (adr_bm / sddr_bm) * np.sqrt(252)

    # Each returns (CR, ADR, SDDR, SR)
    return (cr_ms, adr_ms, sddr_ms, sr_ms), (cr_bm, adr_bm, sddr_bm, sr_bm)

def run_strategy_learner_period(verbose, learner, symbol, sd, ed, sv, commission, impact, img_filename, label):
    """
    Runs StrategyLearner and Benchmark for a given date range.

    Steps:
    1. Call learner.testPolicy() → get trades DataFrame
    2. Compute portfolio values using compute_portvals()
    3. Build benchmark trades (buy 1000 shares on day 1, hold)
    4. Compute portfolio values for benchmark
    5. Normalize both for plotting
    6. Plot both curves + LONG/SHORT vertical lines (based on trades)
    7. Compute performance stats (CR, ADR, SDDR, Sharpe)
    """

    # --------------------- Run StrategyLearner ---------------------

    # testPolicy() uses the model trained in add_evidence() and returns
    # a trades DataFrame indicating how many shares to buy/sell each day.
    sl_trades = learner.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)

    # Compute portfolio values for the StrategyLearner's trades.
    portvals_sl = compute_portvals(sl_trades, start_val=sv, commission=commission, impact=impact)

    # --------------------- Benchmark (Buy & Hold 1000 shares) ---------------------
    dates = pd.date_range(sd, ed)
    prices = get_data([symbol], dates, addSPY=True, colname="Adj Close")
    bm_trades = pd.DataFrame(0, index=prices.index, columns=[symbol])
    first_day = prices.index[0]
    bm_trades.loc[first_day, symbol] = 1000

    portvals_bm = compute_portvals(bm_trades, start_val=sv, commission=commission, impact=impact)

    # Normalize both for plotting.
    norm_sl = portvals_sl["port_val"] / portvals_sl["port_val"].iloc[0]
    norm_bm = portvals_bm["port_val"] / portvals_bm["port_val"].iloc[0]

    # --------------------- Plot StrategyLearner vs Benchmark ---------------------
    if verbose == True:
        ax = norm_bm.plot(label="Benchmark", color="purple", figsize=(12, 6))
        norm_sl.plot(ax=ax, label="Strategy Learner", color="green")

        # Long entries (trades > 0) are shown as blue dashed lines.
        long_dates = sl_trades[sl_trades[symbol] > 0].index
        for d in long_dates:
            ax.axvline(d, color="blue", linestyle="--", alpha=0.4)

        # Short entries (trades < 0) are shown as black dashed lines.
        short_dates = sl_trades[sl_trades[symbol] < 0].index
        for d in short_dates:
            ax.axvline(d, color="black", linestyle="--", alpha=0.4)

        ax.plot([], [], color="blue", linestyle="--", label="Long Entry")
        ax.plot([], [], color="black", linestyle="--", label="Short Entry")

        ax.set_title(f"Strategy Learner vs Benchmark ({label})")
        ax.set_xlabel("Date")
        ax.set_ylabel("Normalized Portfolio Value")
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(f"images/{img_filename}")
        # plt.show()
        plt.close()

    # --------------------- Compute Stats for Report ---------------------
    # Strategy Learner stats
    cr_sl = portvals_sl.iloc[-1]["port_val"] / portvals_sl.iloc[0]["port_val"] - 1
    daily_ret_sl = portvals_sl["port_val"].pct_change().dropna()
    adr_sl = daily_ret_sl.mean()
    sddr_sl = daily_ret_sl.std()
    sr_sl = (adr_sl / sddr_sl) * np.sqrt(252)

    # Benchmark stats
    cr_bm = portvals_bm.iloc[-1]["port_val"] / portvals_bm.iloc[0]["port_val"] - 1
    daily_ret_bm = portvals_bm["port_val"].pct_change().dropna()
    adr_bm = daily_ret_bm.mean()
    sddr_bm = daily_ret_bm.std()
    sr_bm = (adr_bm / sddr_bm) * np.sqrt(252)

    return (cr_sl, adr_sl, sddr_sl, sr_sl), (cr_bm, adr_bm, sddr_bm, sr_bm)

if __name__ == "__main__":
    # setting the random seed
    np.random.seed(904080731)

    symbol = "JPM"
    sv = 100000
    commission = 9.95
    impact = 0.005

    # ------------------ IN-SAMPLE: 2008–2009 ------------------
    sd_in = dt.datetime(2008, 1, 1)
    ed_in = dt.datetime(2009, 12, 31)

    stats_ms_in, stats_bm_in = run_manual_strategy_period(
        verbose= True,
        symbol=symbol,
        sd=sd_in,
        ed=ed_in,
        sv=sv,
        commission=commission,
        impact=impact,
        img_filename="manual_vs_benchmark_insample.png",
        label="In-Sample (2008–2009)",
    )

    # Create and TRAIN StrategyLearner on in-sample data only
    strat_learner = StrategyLearner(verbose=False, impact=impact, commission=commission, )
    strat_learner.add_evidence(symbol=symbol, sd=sd_in, ed=ed_in, sv=sv)

    stats_sl_in, stats_sl_bm_in = run_strategy_learner_period(
        verbose = False,
        learner= strat_learner,
        symbol=symbol,
        sd=sd_in,
        ed=ed_in,
        sv=sv,
        commission=commission,
        impact=impact,
        img_filename="strategy_vs_benchmark_insample.png",
        label="In-Sample (2008–2009)",
    )


    # ------------------ OUT-OF-SAMPLE: 2010–2011 ------------------
    sd_out = dt.datetime(2010, 1, 1)
    ed_out = dt.datetime(2011, 12, 31)

    stats_ms_out, stats_bm_out = run_manual_strategy_period(
        verbose=True,
        symbol=symbol,
        sd=sd_out,
        ed=ed_out,
        sv=sv,
        commission=commission,
        impact=impact,
        img_filename="manual_vs_benchmark_outsample.png",
        label="Out-of-Sample (2010–2011)",
    )

    stats_sl_out, stats_sl_bm_out = run_strategy_learner_period(
        verbose = False,
        learner=strat_learner,
        symbol=symbol,
        sd=sd_out,
        ed=ed_out,
        sv=sv,
        commission=commission,
        impact=impact,
        img_filename="strategy_vs_benchmark_outsample.png",
        label="Out-of-Sample (2010–2011)",
    )

    # ------------------ Print All Stats ------------------
    # print("=== In-Sample Performance (2008–2009) ===")
    # #
    # # print(f"Manual Strategy:    CR={stats_ms_in[0]:.6f}, "f"ADR={stats_ms_in[1]:.6f}, SDDR={stats_ms_in[2]:.6f}, SR={stats_ms_in[3]:.6f}")
    # # print(f"Benchmark (Manual): CR={stats_bm_in[0]:.6f}, "f"ADR={stats_bm_in[1]:.6f}, SDDR={stats_bm_in[2]:.6f}, SR={stats_bm_in[3]:.6f}")
    # print(f"Strategy Learner:   CR={stats_sl_in[0]:.6f}, "f"ADR={stats_sl_in[1]:.6f}, SDDR={stats_sl_in[2]:.6f}, SR={stats_sl_in[3]:.6f}")
    # print(f"Benchmark (SL):     CR={stats_sl_bm_in[0]:.6f}, "f"ADR={stats_sl_bm_in[1]:.6f}, SDDR={stats_sl_bm_in[2]:.6f}, SR={stats_sl_bm_in[3]:.6f}")
    # # print()
    # #
    # print("=== Out-of-Sample Performance (2010–2011) ===")
    # # print(f"Manual Strategy:    CR={stats_ms_out[0]:.6f}, "f"ADR={stats_ms_out[1]:.6f}, SDDR={stats_ms_out[2]:.6f}, SR={stats_ms_out[3]:.6f}")
    # # print(f"Benchmark (Manual): CR={stats_bm_out[0]:.6f}, "f"ADR={stats_bm_out[1]:.6f}, SDDR={stats_bm_out[2]:.6f}, SR={stats_bm_out[3]:.6f}")
    # print(f"Strategy Learner:   CR={stats_sl_out[0]:.6f}, "f"ADR={stats_sl_out[1]:.6f}, SDDR={stats_sl_out[2]:.6f}, SR={stats_sl_out[3]:.6f}")
    # print(f"Benchmark (SL):     CR={stats_sl_bm_out[0]:.6f}, "f"ADR={stats_sl_bm_out[1]:.6f}, SDDR={stats_sl_bm_out[2]:.6f}, SR={stats_sl_bm_out[3]:.6f}")

    e1.run_experiment1()
    e2.run_experiment2()


