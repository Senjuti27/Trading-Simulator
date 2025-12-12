"""
Implements a rule-based trading strategy that converts handcrafted indicator thresholds
directly into trades, serving as a human-designed baseline against the StrategyLearner.
"""

import datetime as dt
import pandas as pd
import numpy as np
from util import get_data
import indicators as id

class ManualStrategy:
    """
    A manual learner that uses human-coded rules with the same indicators as StrategyLearner.
    """
    def __init__(self, verbose= False, impact = 0.0, commission= 0.0):
        self.verbose = verbose
        self.impact = impact
        self.commission = commission

    def add_evidence(self, symbol='IBM', sd=dt.datetime(2008, 1, 1, 0, 0), ed=dt.datetime(2009, 1, 1, 0, 0), sv=100000):
        pass

    def testPolicy(self, symbol='IBM', sd=dt.datetime(2009, 1, 1, 0, 0), ed=dt.datetime(2010, 1, 1, 0, 0), sv=100000):

        # --- 1) Load price data into a single DataFrame ---

        dates = pd.date_range(sd, ed)
        self.data = get_data([symbol], dates, addSPY=True, colname="Adj Close").drop(columns=["SPY"])
        self.data = self.data[[symbol]]  # keep only the target symbol

        # --- 2) Compute indicators and add as columns ---

        bbp_df = id.bollinger_band(self.data.copy(), windows=20, k=2, plot=False)
        rsi_df = id.rsi(self.data.copy(), windows=14, plot=False)
        # macd_df = id.macd(self.data.copy(), fast=12, slow=26, signal=9, plot=False)
        k_df = id.stoc_k(self.data.copy(), windows=14, plot=False)
        cci_df = id.cci(self.data.copy(), windows=14, plot=False)

        self.data["BBP"] = bbp_df["BBP"]  # 0..1
        self.data["RSI"] = rsi_df["RSI"]  # 0..100
        # self.data["MACD_Hist"] = macd_df["MACD_Diff"]
        self.data["K"] = k_df["Stoc_K"]  # 0..100
        self.data["CCI"] = cci_df["CCI"]

        # --- 3) Create per-indicator signals ---

        sig_bbp = pd.Series(0, index=self.data.index)
        sig_bbp[self.data["BBP"] < 0.2] = 1
        sig_bbp[self.data["BBP"] > .8] = -1
        self.data["sig_bbp"] = sig_bbp

        sig_rsi = pd.Series(0, index=self.data.index)
        sig_rsi[self.data["RSI"] < 35] = 1
        sig_rsi[self.data["RSI"] > 65] = -1
        self.data["sig_rsi"] = sig_rsi

        # sig_macd = pd.Series(0, index=self.data.index)
        # sig_macd[self.data["MACD_Hist"] > 0] = 1
        # sig_macd[self.data["MACD_Hist"] < 0] = -1
        # self.data["sig_macd"] = sig_macd

        sig_k = pd.Series(0, index=self.data.index)
        sig_k[self.data["K"] < 20] = 1
        sig_k[self.data["K"] > 80] = -1
        self.data["sig_k"] = sig_k

        sig_cci = pd.Series(0, index=self.data.index)
        sig_cci[self.data["CCI"] < -100] = 1
        sig_cci[self.data["CCI"] > 100] = -1
        self.data["sig_cci"] = sig_cci

        # --- 4) Combine signals and apply warm-up window ---

        # Majority Voting
        scores = sig_bbp+sig_rsi+sig_cci+sig_k
        raw_signal = pd.Series(0, index=self.data.index)
        raw_signal[scores >= 3] = +1
        raw_signal[scores <= -3] = -1


        # between -1 and +1 → stay 0 (no trade)
        # lookback = max(20, 14, 26, 14, 14)  # longest indicator window
        lookback = 20
        raw_signal.iloc[:lookback] = 0      # no trades for the first lookback days
        self.data["raw_signal"] = raw_signal # store raw signal in self.data

        # --- 5) Convert raw signals into next-day desired positions ---

        # After warm-up: if raw_signal is 0, keep previous day's position
        # (avoid jumping into cash on every neutral day)
        desired_pos = raw_signal.shift(1).astype(float) # Shift so that today's position uses *yesterday's* signal
        desired_pos.iloc[lookback:] = desired_pos.iloc[lookback:].replace(0, np.nan) # If outputs 0 (CASH), treat it as unknown and wait.
        desired_pos = desired_pos.ffill().fillna(0).astype(int)

        # Store for debugging
        self.data["raw_signal"] = raw_signal
        self.data["desired_pos"] = desired_pos

        # ---------- 6) Convert desired positions into trades ----------

        trades_units = pd.Series(0, index=desired_pos.index, dtype=int)
        for i in range(1, len(desired_pos)):
            yesterday = desired_pos.iloc[i - 1]
            today = desired_pos.iloc[i]
            if yesterday == today:
                trades_units.iloc[i] = 0  # no change
            else:
                trades_units.iloc[i] = today - yesterday  # -2, -1, +1, +2

        # First day: whatever position we want on day 0
        trades_units.iloc[0] = desired_pos.iloc[0]

        # Convert units (-2..2) → shares, obey +/- 2000 cap
        trades_shares = (trades_units * 1000).clip(-2000, 2000).astype(int)

        # Converts the Series → DataFrame with a single column named the symbol (e.g., "JPM")
        trades = trades_shares.to_frame(symbol)
        trades.index.name = "Date"

        # Extra debugging columns
        self.data["trade_units"] = trades_units
        self.data["trade_shares"] = trades_shares
        self.data["final_position"] = desired_pos * 1000

        # print(self.data.head(60))  # optional: inspect signals
        return trades

    def author(self):
        return "stwisha3"

    def study_group(self):
        return "stwisha3"

# if __name__ == "__main__":
#     print("Manual strategy")
#     ms = ManualStrategy()
#     ms.testPolicy(symbol="JPM", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000)