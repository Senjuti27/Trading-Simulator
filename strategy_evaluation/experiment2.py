"""
Experiment 2:
Assess sensitivity of StrategyLearner to different market impacts
for JPM in-sample (2008–2009).
"""

import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from marketsimcode import compute_portvals
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


def run_experiment2():
    symbol = "JPM"
    sv = 100000
    commission = 9.95

    # In-sample period only (per spec)
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)

    # different impact levels to compare
    impacts = [0, 0.0005, 0.005, 0.01]

    portfolio_values = []
    stats_list = []

    # ------------------------ Train & test per impact ------------------------

    for impact in impacts:
        # Create a fresh StrategyLearner for each impact
        sl = StrategyLearner(verbose=False, impact=impact, commission=commission)

        # Train on in-sample data
        sl.add_evidence(symbol=symbol, sd=sd, ed=ed, sv=sv)

        # Get trades produced by the learner (in-sample)
        sl_trades = sl.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)

        # Convert trades to portfolio values using matching impact
        portvals = compute_portvals(
            sl_trades,
            start_val=sv,
            commission=commission,
            impact=impact,
        )

        portfolio_values.append(portvals)
        stats_list.append(compute_stats(portvals))

    # ------------------------ Normalize and plot ------------------------

    # Normalize each strategy learner portfolio to 1.0 on the first day
    pv1 = portfolio_values[0]["port_val"] / portfolio_values[0]["port_val"].iloc[0]
    pv2 = portfolio_values[1]["port_val"] / portfolio_values[1]["port_val"].iloc[0]
    pv3 = portfolio_values[2]["port_val"] / portfolio_values[2]["port_val"].iloc[0]
    pv4 = portfolio_values[3]["port_val"] / portfolio_values[3]["port_val"].iloc[0]

    # Plot using your preferred style
    ax = pv1.plot(
        fontsize=12,
        color="black",
        label="Strategy Learner - impact = 0",
        figsize=(12, 6),
    )
    pv2.plot(ax=ax, color="blue", label="Strategy Learner - impact = 0.0005")
    pv3.plot(ax=ax, color="green", label="Strategy Learner - impact = 0.005")
    pv4.plot(ax=ax, color="purple", label="Strategy Learner - impact = 0.01")

    plt.title("Experiment 2: Impact Sensitivity (2008–2009)")
    plt.xlabel("Date")
    plt.ylabel("Normalized Portfolio Value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig("images/exp2_impact.png")
    plt.close()

    # ------------------- Chart 2: Metrics vs Impact -------------------

    # Unpack stats into separate lists: CR, ADR, SDDR, SR
    cr_list = []
    adr_list = []
    sddr_list = []
    sr_list = []

    for (cr, adr, sddr, sr) in stats_list:
        cr_list.append(cr)
        adr_list.append(adr)
        sddr_list.append(sddr)
        sr_list.append(sr)

    # Extract CR and SR from stats_list
    CRs = [stats[0] for stats in stats_list]
    SRs = [stats[3] for stats in stats_list]

    x = np.arange(len(impacts))  # positions on x-axis
    width = 0.35  # width of each bar

    plt.figure(figsize=(10, 6))

    # CR bars (left side)
    plt.bar(x - width / 2, CRs, width, label="Cumulative Return", color="skyblue")

    # SR bars (right side)
    plt.bar(x + width / 2, SRs, width, label="Sharpe Ratio", color="lightgreen")

    plt.xticks(x, [str(i) for i in impacts])
    plt.xlabel("Impact Value")
    plt.title("Experiment 2: Cumulative Return & Sharpe Ratio vs Market Impact")
    plt.ylabel("Metric Value")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.savefig("images/exp2_combined_bar.png")
    plt.close()



    # # ------------------- PRINT RESULTS -------------------
    # print("=== Experiment 2: Impact Sensitivity (In-Sample 2008–2009) ===")
    # for impact, stats in zip(impacts, stats_list):
    #     cr, adr, sddr, sr = stats
    #     print(
    #         f"Impact={impact:.4f} | "
    #         f"CR={cr:.6f} | ADR={adr:.6f} | SDDR={sddr:.6f} | SR={sr:.6f}"
    #     )


if __name__ == "__main__":
    run_experiment2()
