"""
Experiment 1:
Compare Manual Strategy, Strategy Learner, and Benchmark
for JPM in-sample (2008–2009) and out-of-sample (2010–2011).
"""

import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from util import get_data
from marketsimcode import compute_portvals
from ManualStrategy import ManualStrategy
from StrategyLearner import StrategyLearner

def author():
    return "stwisha3"

def study_group():
    return "stwisha3"

def compute_stats(portvals):
    """
    Given a portvals DataFrame with a single column 'port_val',
    compute CR, ADR, SDDR, SR.
    """
    vals = portvals["port_val"]
    cr = (vals.iloc[-1] / vals.iloc[0]) - 1.0
    daily_rets = vals.pct_change().iloc[1:]
    adr = daily_rets.mean()
    sddr = daily_rets.std()
    sr = (adr / sddr) * np.sqrt(252) if sddr != 0 else 0.0
    return cr, adr, sddr, sr


def run_experiment1():
    symbol = "JPM"
    sv = 100000
    commission = 9.95
    impact = 0.005

    # ---------------------- In-sample period ----------------------

    sd_in = dt.datetime(2008, 1, 1)
    ed_in = dt.datetime(2009, 12, 31)

    # Manual Strategy
    ms = ManualStrategy(verbose=False, impact=impact, commission=commission)
    ms_trades_in = ms.testPolicy(symbol=symbol, sd=sd_in, ed=ed_in, sv=sv)
    portvals_ms_in = compute_portvals(
        ms_trades_in, start_val=sv, commission=commission, impact=impact
    )

    # Strategy Learner - train on in-sample
    sl = StrategyLearner(verbose=False, impact=impact, commission=commission)
    sl.add_evidence(symbol=symbol, sd=sd_in, ed=ed_in, sv=sv)
    sl_trades_in = sl.testPolicy(symbol=symbol, sd=sd_in, ed=ed_in, sv=sv)
    portvals_sl_in = compute_portvals(
        sl_trades_in, start_val=sv, commission=commission, impact=impact
    )

    # Benchmark
    dates = get_data([symbol], pd.date_range(sd_in, ed_in), addSPY=True).index
    bm_trades = pd.DataFrame(0, index=dates, columns=[symbol])
    bm_trades.iloc[0, 0] = 1000
    portvals_bm_in = compute_portvals(bm_trades, start_val=sv, commission=commission, impact=impact)

    # Normalize
    norm_ms_in = portvals_ms_in["port_val"] / portvals_ms_in["port_val"].iloc[0]
    norm_sl_in = portvals_sl_in["port_val"] / portvals_sl_in["port_val"].iloc[0]
    norm_bm_in = portvals_bm_in["port_val"] / portvals_bm_in["port_val"].iloc[0]

    # Stats
    stats_ms_in = compute_stats(portvals_ms_in)
    stats_sl_in = compute_stats(portvals_sl_in)
    stats_bm_in = compute_stats(portvals_bm_in)

    # Plot in-sample
    plt.figure(figsize=(12, 6))
    norm_bm_in.plot(label="Benchmark", color="purple")
    norm_ms_in.plot(label="Manual Strategy", color="red")
    norm_sl_in.plot(label="Strategy Learner", color="green")
    plt.title("Experiment 1: In-Sample (2008–2009)")
    plt.xlabel("Date")
    plt.ylabel("Normalized Portfolio Value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("images/exp1_in_sample.png")
    # plt.show()
    plt.close()

    # ---------------------- Out-of-sample period ----------------------

    sd_out = dt.datetime(2010, 1, 1)
    ed_out = dt.datetime(2011, 12, 31)

    # NO retrain learner; reuse trained model
    ms_trades_out = ms.testPolicy(symbol=symbol, sd=sd_out, ed=ed_out, sv=sv)
    portvals_ms_out = compute_portvals(
        ms_trades_out, start_val=sv, commission=commission, impact=impact
    )

    sl_trades_out = sl.testPolicy(symbol=symbol, sd=sd_out, ed=ed_out, sv=sv)
    portvals_sl_out = compute_portvals(
        sl_trades_out, start_val=sv, commission=commission, impact=impact
    )

    # Benchmark
    dates_out = get_data([symbol], pd.date_range(sd_out, ed_out), addSPY=True).index
    bm_trades_out = pd.DataFrame(0, index=dates_out, columns=[symbol])
    bm_trades_out.iloc[0, 0] = 1000
    portvals_bm_out = compute_portvals(bm_trades_out, start_val=sv, commission=commission, impact=impact)

    # Normalize
    norm_ms_out = portvals_ms_out["port_val"] / portvals_ms_out["port_val"].iloc[0]
    norm_sl_out = portvals_sl_out["port_val"] / portvals_sl_out["port_val"].iloc[0]
    norm_bm_out = portvals_bm_out["port_val"] / portvals_bm_out["port_val"].iloc[0]

    # Stats
    stats_ms_out = compute_stats(portvals_ms_out)
    stats_sl_out = compute_stats(portvals_sl_out)
    stats_bm_out = compute_stats(portvals_bm_out)

    # Plot out-of-sample
    plt.figure(figsize=(12, 6))
    norm_bm_out.plot(label="Benchmark", color="purple")
    norm_ms_out.plot(label="Manual Strategy", color="red")
    norm_sl_out.plot(label="Strategy Learner", color="green")
    plt.title("Experiment 1: Out-of-Sample (2010–2011)")
    plt.xlabel("Date")
    plt.ylabel("Normalized Portfolio Value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("images/exp1_out_sample.png")
    # plt.show()
    plt.close()

    # Optional: print for debugging / report support
    # print("=== Experiment 1: In-Sample (2008–2009) ===")
    # print("Manual Strategy:   CR={:.6f}, ADR={:.6f}, SDDR={:.6f}, SR={:.6f}".format(*stats_ms_in))
    # print("Strategy Learner:  CR={:.6f}, ADR={:.6f}, SDDR={:.6f}, SR={:.6f}".format(*stats_sl_in))
    # print("Benchmark:         CR={:.6f}, ADR={:.6f}, SDDR={:.6f}, SR={:.6f}".format(*stats_bm_in))
    # print()
    #
    # print("=== Experiment 1: Out-of-Sample (2010–2011) ===")
    # print("Manual Strategy:   CR={:.6f}, ADR={:.6f}, SDDR={:.6f}, SR={:.6f}".format(*stats_ms_out))
    # print("Strategy Learner:  CR={:.6f}, ADR={:.6f}, SDDR={:.6f}, SR={:.6f}".format(*stats_sl_out))
    # print("Benchmark:         CR={:.6f}, ADR={:.6f}, SDDR={:.6f}, SR={:.6f}".format(*stats_bm_out))


if __name__ == "__main__":
    run_experiment1()
