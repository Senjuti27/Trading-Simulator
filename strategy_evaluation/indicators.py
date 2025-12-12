"""
Computes the technical indicators used as features for the learner
(e.g., Bollinger Band %B, RSI, MACD histogram, Stochastic %K, CCI),
returning date-aligned DataFrames for each signal.
"""

import pandas as pd
from util import get_data
import matplotlib.pyplot as plt

def bollinger_band(df_prices_id, windows, k, plot):
    """
    Bollinger Bands are volatility-based bands placed above and below a moving average (typically a 20-day SMA).
    They help identify how stretched the price is from its average.
    Components:

    - SMA (Simple Moving Average)
    - Upper Band = SMA + k * StdDev
    - Lower Band = SMA - k * StdDev
    - BBP (Bollinger Band Percent) = (Price - Lower Band) / (Upper Band - Lower Band)

    BBP gives a normalized value between 0 and 1:
    - BBP > 1: Price is above the upper band (strong up-move or overbought)
    - BBP < 0: Price is below the lower band (strong down-move or oversold)
    - BBP ≈ 0.5: Price is near the middle of the bands
    """

    # Drop SPY if it exists
    if 'SPY' in df_prices_id.columns:
        df_prices_id = df_prices_id.drop('SPY', axis=1)

    symbol = df_prices_id.columns[0]
    sma = df_prices_id[symbol].rolling(windows).mean()
    std_dev = df_prices_id[symbol].rolling(windows).std()
    lower_band = sma - k * std_dev
    upper_band = sma + k * std_dev
    bbp = (df_prices_id[symbol]-lower_band)/ (upper_band-lower_band)
    df_ret = pd.DataFrame({
        "Price": df_prices_id[symbol],
        "SMA": sma,
        "STD_DEV": std_dev,
        "Lower_Band": lower_band,
        "Upper_Band": upper_band,
        "BBP": bbp
    })

    if plot:
        fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        # Upper Plot: Price with Bollinger Bands
        axs[0].plot(df_prices_id[symbol], label="Price", color="black", linewidth=1.2)
        axs[0].plot(sma, label=f"SMA({windows})", color="blue", linewidth=1)
        axs[0].plot(upper_band, label="Upper BB", color="red", linestyle="--")
        axs[0].plot(lower_band, label="Lower BB", color="green", linestyle="--")
        axs[0].set_ylabel("Price")
        axs[0].set_title("Bollinger Bands (BBP) Indicator")
        axs[0].legend(loc="lower left")
        axs[0].grid(True)
        # Lower Plot: BBP Value
        axs[1].plot(bbp, label="BBP", color="purple", linewidth=1)
        axs[1].axhline(1, color="red", linestyle="--", label="Overbought (1)")
        axs[1].axhline(0, color="green", linestyle="--", label="Oversold (0)")
        axs[1].set_ylabel("BBP Value")
        axs[1].set_xlabel("Date")
        axs[1].legend(loc="lower left")
        axs[1].grid(True)
        plt.tight_layout()
        plt.savefig(f"./images/BBP_{symbol}.png")
        # plt.show()
        plt.close()

    return df_ret

def rsi(df_prices_id, windows, plot):
    """
    RSI is a momentum oscillator that measures the speed and magnitude of recent price changes.
    It compares the average gains and losses over a lookback period (typically 14 days).

    Steps:
    1. Gain = max(price_today - price_yesterday, 0)
    2. Loss = max(price_yesterday - price_today, 0)
    3. Avg Gain and Avg Loss are smoothed using Wilder's EMA
    4. RS = Avg Gain / Avg Loss
    5. RSI = 100 - (100 / (1 + RS))

    Interpretation:
    - RSI ∈ [0, 100]
    - RSI > 70: Overbought (price may be too high)
    - RSI < 30: Oversold (price may be too low)
    - RSI ~50: Neutral
    """

    # Drop SPY if it exists
    if 'SPY' in df_prices_id.columns:
        df_prices_id = df_prices_id.drop('SPY', axis=1)

    symbol = df_prices_id.columns[0]
    delta = df_prices_id[symbol].diff()     # daily price change -> (+): price up, (-) -> down
    gain = delta.clip(lower = 0)            # Positive values, negative values-> 0
    loss = - delta.clip(upper = 0)          # Negative value, positive values -> 0

    # avg_gain = gain.rolling(windows).mean()
    # avg_loss = loss.rolling(windows).mean()

    """
    Wilder's RSI
    """
    # Initialize average gain/loss using first N values
    avg_gain = pd.Series(0.0, index=gain.index)
    avg_loss = pd.Series(0.0, index=loss.index)
    avg_gain.iloc[windows] = gain.iloc[1:windows + 1].mean()
    avg_loss.iloc[windows] = loss.iloc[1:windows + 1].mean()

    # Wilder's recursive smoothing (EMA-style)
    for i in range(windows + 1, len(df_prices_id[symbol])):
        avg_gain.iloc[i] = (avg_gain.iloc[i - 1] * (windows - 1) + gain.iloc[i]) / windows
        avg_loss.iloc[i] = (avg_loss.iloc[i - 1] * (windows- 1) + loss.iloc[i]) / windows

    rsi = 100 - ( 100/ (1+ (avg_gain/avg_loss)))
    df_ret = pd.DataFrame({
        "Price": df_prices_id[symbol],
        "Difference": delta,
        "Gain": gain,
        "Loss": loss,
        "Avg_Gain": avg_gain,
        "Avg_Loss": avg_loss,
        "RSI": rsi
    })

    if plot:
        fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={'height_ratios': [1, 2]})
        # Upper Plot: Price
        axs[0].plot(df_prices_id[symbol], label=" Price", color="black", linewidth=1.5)
        axs[0].set_ylabel("Price")
        axs[0].legend(loc="lower left")
        axs[0].grid(True)
        # --- Bottom subplot: RSI ---
        axs[1].plot(rsi, label=f"RSI ({windows}-day)", color="blue", linewidth=1.5)
        axs[1].axhline(70, color="red", linestyle="--", linewidth=1, label="Overbought (70)")
        axs[1].axhline(30, color="green", linestyle="--", linewidth=1, label="Oversold (30)")
        axs[1].set_ylabel("RSI")
        axs[1].set_xlabel("Date")
        axs[1].legend(loc="lower left")
        axs[1].grid(True)
        # --- Common formatting ---
        fig.suptitle(f"Relative Strength Index (RSI) Indicator", fontsize=14)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)  # Make space for the main title
        plt.savefig(f"./images/RSI_{symbol}.png")
        # plt.show()
        plt.close()

    return df_ret

def macd(df_prices_id, fast, slow, signal, plot):
    """
    MACD is a trend-following momentum indicator that shows the relationship between two EMAs.
    fast-> 12, slow->26
    Formulas:
    - MACD Line = EMA(12) - EMA(26)
    - Signal Line = EMA(9) of MACD Line
    - MACD Histogram = MACD Line - Signal Line

    Interpretation:
    - MACD > Signal: Bullish (momentum increasing)
    - MACD < Signal: Bearish (momentum weakening)
    - Histogram crossing zero is often used as a buy/sell trigger
    """
    # Drop SPY if it exists
    if 'SPY' in df_prices_id.columns:
        df_prices_id = df_prices_id.drop('SPY', axis=1)

    symbol = df_prices_id.columns[0]
    ema_f = df_prices_id[symbol].ewm(span=fast, min_periods= fast).mean()
    ema_s = df_prices_id[symbol].ewm(span=slow, min_periods=slow).mean()
    macd_line = ema_f-ema_s
    signal_line = macd_line.ewm(span=signal, min_periods=signal).mean()
    macd_diff = macd_line - signal_line

    df_ret = pd.DataFrame({
        "Price": df_prices_id[symbol],
        "EMA_Fast": ema_f,
        "EMA_Slow": ema_s,
        "MACD_Line": macd_line,
        "Signal_Line": signal_line,
        "MACD_Diff": macd_diff
    })

    if plot:
        fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={'height_ratios': [1, 2]})
        axs[0].plot(df_prices_id[symbol], label=f"{symbol} Price", color="black", linewidth=1.5)
        axs[0].set_ylabel("Price")
        axs[0].legend(loc="lower left")
        axs[0].grid(True)
        # --- Bottom subplot: MACD indicator ---
        axs[1].plot(macd_line, label=f"MACD ({fast},{slow})", color="blue", linewidth=1.5)
        axs[1].plot(signal_line, label=f"Signal ({signal})", color="orange", linewidth=1.5)
        # Histogram bars (green = positive, red = negative)
        axs[1].bar(
            macd_diff.index,
            macd_diff,
            color=["green" if val >= 0 else "red" for val in macd_diff],
            alpha=0.5,
            label="MACD Histogram"
        )
        axs[1].axhline(0, color="black", linestyle="--", linewidth=1)
        axs[1].set_ylabel("MACD Value")
        axs[1].set_xlabel("Date")
        axs[1].legend(loc="lower left")
        axs[1].grid(True)
        # --- Common formatting ---
        fig.suptitle(f"Moving Average Convergence Divergence (MACD) Indicator for {symbol}", fontsize=14)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)  # space for main title
        plt.savefig(f"./images/MACD_{symbol}.png")
        # plt.show()
        plt.close()

    return df_ret

def stoc_k(df_prices_id, windows, plot):
    """
    Stochastic Oscillator is a momentum indicator that compares
    a security's closing price to its price range over a set number of periods (typically 14).

    %K line: Measures a security's closing price in relation to its price range over a specific period,
    such as the last 14 days. A higher %K value indicates the closing price is near the high of the recent range,
    while a lower %K means it is near the low.

    %D line: A simple moving average (SMA) of the %K line, typically over three periods.
    Because it is an average, it is smoother and reacts more slowly to price changes,
    which helps confirm trends and reduce false signals

    Formulas:
    - %K = (Close - Lowest Low) / (Highest High - Lowest Low) * 100
    - %D = SMA(3) of %K (acts as a signal line)

    Interpretation:
    - Values range from 0 to 100.
    - %K > 80: Overbought (price near recent highs)
    - %K < 20: Oversold (price near recent lows)
    - Crossovers of %K and %D can indicate trend reversals

    """
    # Drop SPY if exists
    if 'SPY' in df_prices_id.columns:
        df_prices_id = df_prices_id.drop('SPY', axis=1)

    symbol = df_prices_id.columns[0]
    dates = df_prices_id.index

    df_adj_close = df_prices_id.copy()
    df_close = get_data([symbol], dates, addSPY=False, colname='Close')
    df_high = get_data([symbol], dates, addSPY=False, colname='High')
    df_low = get_data([symbol], dates, addSPY=False, colname='Low')

    adj_factor = df_adj_close[symbol] / df_close[symbol]

    adj_high = df_high[symbol] * adj_factor
    adj_low = df_low[symbol] * adj_factor
    adj_close = df_adj_close[symbol]

    highest_high = adj_high.rolling(window=windows, min_periods=windows).max()
    lowest_low = adj_low.rolling(window=windows, min_periods=windows).min()

    stoc_k = 100 * (adj_close - lowest_low) / (highest_high - lowest_low)
    stoc_d = stoc_k.rolling(window=3).mean()

    df_ret = pd.DataFrame({
        "Price": adj_close,
        "Adj_High": adj_high,
        "Adj_Low": adj_low,
        "Highest_High": highest_high,
        "Lowest_Low": lowest_low,
        "Stoc_K": stoc_k,
        "Stoc_D": stoc_d
    })

    if plot:
        fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={'height_ratios': [1, 2]})
        # Top subplot: Price
        axs[0].plot(df_prices_id[symbol], label=f"{symbol} Price", color="black", linewidth=1.5)
        axs[0].set_ylabel("Price")
        axs[0].legend(loc="lower left")
        axs[0].grid(True)
        # Bottom subplot: %K and %D
        axs[1].plot(stoc_k, label=f"%K ({windows}-day)", color="blue", linewidth=1)
        axs[1].plot(stoc_d, label="%D (3-day SMA)", color="green", linestyle="--", linewidth=1)
        axs[1].axhline(80, color="red", linestyle="--", linewidth=1, label="Overbought (80)")
        axs[1].axhline(20, color="orange", linestyle="--", linewidth=1, label="Oversold (20)")
        axs[1].set_ylabel("Stochastic Value (0–100)")
        axs[1].set_xlabel("Date")
        axs[1].legend(loc="upper left")
        axs[1].grid(True)
        # Figure title and layout
        fig.suptitle(f"Stochastic %K Indicator ({symbol})", fontsize=14)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig(f"./images/StochasticK_{symbol}.png")
        # plt.show()
        plt.close()

    return df_ret

def cci(df_prices_id, windows, plot):
    """
    CCI (Commodity Channel Index) is a momentum-based oscillator that measures
    how far a price has diverged from its historical average (typical price vs moving average).

    Formula:
    - Typical Price (TP) = (High + Low + Close) / 3
    - SMA_TP = SMA of Typical Price over 'n' periods
    - Mean Deviation = avg(|TP - SMA_TP|)
    - CCI = (TP - SMA_TP) / (0.015 * Mean Deviation)

    Interpretation:
    - CCI typically ranges between -100 and +100.
    - CCI > +100: Overbought or strong bullish signal
    - CCI < -100: Oversold or strong bearish signal
    - CCI crossing zero indicates trend shifts
    """

    if 'SPY' in df_prices_id.columns:
        df_prices_id = df_prices_id.drop('SPY', axis=1)

    symbol = df_prices_id.columns[0]
    dates = df_prices_id.index  # use same date range as input

    # Get adjusted and raw OHLC data
    df_adj_close = df_prices_id.copy()  # already adjusted close
    df_close = get_data([symbol], dates, addSPY=False, colname='Close')
    df_high = get_data([symbol], dates, addSPY=False, colname='High')
    df_low = get_data([symbol], dates, addSPY=False, colname='Low')
    df_open = get_data([symbol], dates, addSPY=False, colname='Open')

    # Compute adjustment factor
    adj_factor = df_adj_close[symbol] / df_close[symbol]

    # Compute adjusted OHLC
    adj_high = df_high[symbol] * adj_factor
    adj_low = df_low[symbol] * adj_factor
    adj_close = df_adj_close[symbol]

    tp = (adj_high + adj_low + adj_close) / 3.0
    sma_tp = tp.rolling(window=windows).mean()
    md = tp.rolling(window=windows).apply(lambda x: abs(x - x.mean()).mean(), raw=False)
    cci = (tp - sma_tp) / (0.015 * md)

    df_ret = pd.DataFrame({
        "Price": adj_close,
        "Typical_Price": tp,
        "SMA_TP": sma_tp,
        "MD": md,
        "CCI": cci
    })

    if plot:
        fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={'height_ratios': [1, 2]})

        # --- Top subplot: Adjusted Price ---
        axs[0].plot(df_ret["Price"], label=f"{symbol} Adjusted Close", color="black", linewidth=1.5)
        axs[0].set_ylabel("Price")
        axs[0].legend(loc="lower left")
        axs[0].grid(True)
        # --- Bottom subplot: CCI ---
        axs[1].plot(df_ret["CCI"], label=f"CCI ({windows}-day)", color="blue", linewidth=1.5)
        axs[1].axhline(100, color="red", linestyle="--", linewidth=1, label="Overbought (+100)")
        axs[1].axhline(-100, color="green", linestyle="--", linewidth=1, label="Oversold (-100)")
        axs[1].set_ylabel("CCI")
        axs[1].set_xlabel("Date")
        axs[1].legend(loc="lower left")
        axs[1].grid(True)

        fig.suptitle(f"Commodity Channel Index (CCI) — {symbol}", fontsize=14)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig(f"./images/CCI_{symbol}.png")
        # plt.show()
        plt.close()

    return df_ret

def author():
    return "stwisha3"

def study_group():
    return "stwisha3"







