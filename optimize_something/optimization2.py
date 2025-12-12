
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # ensure no display is needed in Gradescope
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# IMPORTANT: use ONLY util.get_data to read prices
import util as ut


def author():
    """
    Returns:
        str: Georgia Tech username of the author (do NOT include '@gatech.edu').
    """
    return "stwisha3"


def study_group():
    """
    Returns:
        list[str]: GT usernames you discussed the project with. Empty list if none.
    """
    return []


def _fetch_prices(sd, ed, syms):
    """
    Fetches adjusted close prices for symbols between sd and ed (inclusive).

    Args:
        sd (datetime): start date
        ed (datetime): end date
        syms (list[str]): symbols to load

    Returns:
        pd.DataFrame: prices indexed by date with columns as symbols (and possibly SPY).
    """
    dates = pd.date_range(sd, ed)
    # util.get_data handles adding SPY if needed and reading from ../data
    prices = ut.get_data(syms, dates)
    # Some util.get_data versions always add SPY; ensure requested symbols exist
    return prices[syms].dropna(how="all")  # keep rows with any real data


def _normalize(df):
    return df / df.iloc[0]


def _portfolio_stats(prices, allocs):
    """
    Given price DataFrame and allocations, compute portfolio statistics.

    Args:
        prices (pd.DataFrame): price DF for the assets only (no SPY)
        allocs (np.ndarray): allocations that sum to 1, each in [0,1]

    Returns:
        tuple: (cr, adr, sddr, sr, daily_port_val_series)
    """
    # Normalize prices to start at 1
    normed = _normalize(prices)
    # Allocate
    alloced = normed * allocs
    # Daily portfolio value
    port_val = alloced.sum(axis=1)

    # Daily returns (exclude first day)
    daily_rets = port_val.pct_change().dropna()

    # Stats
    cr = port_val.iloc[-1] / port_val.iloc[0] - 1.0
    adr = daily_rets.mean()
    # Sample standard deviation (ddof=1) as required
    sddr = daily_rets.std(ddof=1)

    # Sharpe ratio with 0.0 daily risk-free rate
    k = np.sqrt(252.0)  # trading days per year
    sr = k * (adr / sddr) if sddr != 0 else 0.0

    return cr, adr, sddr, sr, port_val


def _neg_sharpe(allocs, prices):
    """
    Objective for optimizer: negative Sharpe ratio to enable maximization.
    """
    _, _, _, sr, _ = _portfolio_stats(prices, allocs)
    return -sr


def _alloc_sum_to_one(allocs):
    """
    Equality constraint: allocations must sum to 1.
    """
    return np.sum(allocs) - 1.0


def optimize_portfolio(sd=dt.datetime(2008, 6, 1),
                       ed=dt.datetime(2009, 6, 1),
                       syms=None,
                       gen_plot=False):
    """
    Find allocations that maximize Sharpe ratio for the given symbols and date range.
    Long-only, allocations in [0,1], and must sum to 1.

    Args:
        sd (datetime): start date
        ed (datetime): end date
        syms (list[str]): symbols in the portfolio (n >= 2)
        gen_plot (bool): when True, save 'Figure1.png' comparing optimal portfolio vs SPY

    Returns:
        tuple: (allocs, cr, adr, sddr, sr)
            allocs: 1-d np.ndarray of weights (sum ~ 1.0; each in [0,1])
            cr: cumulative return
            adr: average daily return
            sddr: standard deviation of daily returns (sample)
            sr: Sharpe ratio (annualized, rf=0, 252 trading days)
    """
    if syms is None or len(syms) < 2:
        raise ValueError("You must provide at least two symbols (n >= 2).")

    # Load only the requested symbols for optimization
    prices_syms = _fetch_prices(sd, ed, syms)

    # Initial guess: uniform allocation
    n = len(syms)
    x0 = np.array([1.0 / n] * n)

    # Bounds and constraints
    bounds = tuple((0.0, 1.0) for _ in range(n))  # long-only
    constraints = ({"type": "eq", "fun": _alloc_sum_to_one},)

    # Optimize using SLSQP
    result = minimize(_neg_sharpe, x0,
                      args=(prices_syms,),
                      method="SLSQP",
                      bounds=bounds,
                      constraints=constraints,
                      options={"disp": False, "maxiter": 500})

    # If the optimizer fails, fall back to uniform allocation (robustness)
    allocs = result.x if result.success else x0

    # Clip tiny numerical violations and re-normalize to sum to 1 within tolerance
    allocs = np.clip(allocs, 0.0, 1.0)
    s = allocs.sum()
    if s != 0:
        allocs = allocs / s

    # Compute final statistics
    cr, adr, sddr, sr, port_val = _portfolio_stats(prices_syms, allocs)

    if gen_plot:
        # Build comparison plot against SPY normalized
        # Use util.get_data so we comply with I/O rule
        dates = pd.date_range(sd, ed)
        # Ensure SPY is present for comparison
        prices_all = ut.get_data(["SPY"] + syms, dates)
        # Portfolio value built from syms only
        norm_port = _normalize(port_val.to_frame("Portfolio")).rename(columns={"Portfolio": "Portfolio"})
        # SPY normalized
        spy = _normalize(prices_all[["SPY"]])

        # Align indexes
        df_plot = pd.concat([norm_port, spy], axis=1).dropna()
        ax = df_plot.plot(title="Daily Portfolio Value vs SPY", fontsize=10)
        ax.set_xlabel("Date")
        ax.set_ylabel("Normalized Value")
        ax.legend(["Portfolio", "SPY"])
        plt.tight_layout()
        plt.savefig("Figure1.png")
        plt.close()

    return allocs, cr, adr, sddr, sr


if __name__ == "__main__":
    # Example run that produces the Figure1.png used in the report.
    # Parameters specified by the assignment:
    # Start Date: 2008-06-01, End Date: 2009-06-01, Symbols: ['IBM', 'X', 'GLD', 'JPM']
    allocs, cr, adr, sddr, sr = optimize_portfolio(
        sd=dt.datetime(2008, 6, 1),
        ed=dt.datetime(2009, 6, 1),
        syms=["IBM", "X", "GLD", "JPM"],
        gen_plot=True
    )
    # Print a compact summary (safe for local testing; autograder ignores __main__)
    print("Optimized allocations:", np.round(allocs, 4).tolist())
    print(f"CR:  {cr:.6f}")
    print(f"ADR: {adr:.6f}")
    print(f"SDDR:{sddr:.6f}")
    print(f"SR:  {sr:.6f}")
