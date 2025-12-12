import pandas as pd
import os
import matplotlib.pyplot as plt
import time
import numpy as np
import seaborn as sns

def get_max_close(symbol):
    df = pd.read_csv("data/{}.csv".format(symbol))
    return df['Close'].max()

def get_mean_volume(symbol):
    df = pd.read_csv("data/{}.csv".format(symbol))
    return df['Volume'].mean()

def test_run_1_1():
    df = pd.read_csv("data/AAPL.csv")
    print (df)
    print (df.head())
    print(df.tail())
    print(df[10:21])
    print(df['Adj Close'])
    df['Adj Close'].plot()
    plt.show()
    df['High'].plot()
    plt.show()
    df[['Close', 'Adj Close']].plot()
    plt.show()
    for symbol in ['AAPL', 'IBM']:
        print("Max Close")
        print(symbol, get_max_close(symbol))
        print("Mean Volume")
        print(symbol, get_mean_volume(symbol))

def symbol_to_path(symbol, base_dir="data"):
    """Return CSV file path given ticker symbol."""
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))

def get_data(symbols, dates):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df = pd.DataFrame(index=dates)
    if 'SPY' not in symbols:  # add SPY for reference, if absent
        symbols.insert(0, 'SPY')

    for symbol in symbols:
		# Quiz: Read and join data for each symbol
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',
                parse_dates=True, usecols=['Date', 'Adj Close'], na_values=['nan'])
        df_temp = df_temp.rename(columns={'Adj Close': symbol})
        df = df.join(df_temp)
        if symbol == 'SPY':  # drop dates SPY did not trade
            df = df.dropna(subset=["SPY"])

    return df

def plot_data(df, title="Stock prices"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    plt.show()

    """
    Newer ways
    """
    # plt.figure(figsize=(10, 6))
    # plt.plot(df.index, df.values)
    # plt.title(title)
    # plt.xlabel("Date")
    # plt.ylabel("Price")
    # plt.grid(True)
    # plt.show()


def plot_selected(df, columns, start_index, end_index):
    """Plot the desired columns over index values in the given range."""
    """
    Replacing ix with loc
    """
    plot_data(df.loc[start_index:end_index, columns], title="Selected data")

def normalize_data(df):
	"""Normalize stock prices using the first row of the dataframe.
    Replacing ix with iloc
    """
	return df/df.iloc[0,:]

def test_run_1_2():
    start_date = '2010-01-22'
    end_date = '2010-01-26'
    dates = pd.date_range(start_date,end_date)
    print(dates)
    print(dates[0])

    df1 = pd.DataFrame(index=dates)
    dfSPY = pd.read_csv("data/SPY.csv", index_col="Date" , parse_dates=True, usecols=['Date', 'Adj Close'], na_values= ['nan'] )
    # df1 = df1.join(dfSPY)
    # df1 = df1.dropna()
    dfSPY = dfSPY.rename(columns={'Adj Close': 'SPY'})
    df1 = df1.join(dfSPY, how = 'inner')
    print(df1)

    symbols = ['GOOG', 'IBM', 'GLD']
    for s in symbols:
        df_temp = pd.read_csv("data/{}.csv".format(s), index_col= 'Date', parse_dates= True, usecols=['Date', 'Adj Close'], na_values=['nan'])
        df_temp = df_temp.rename(columns={'Adj Close': s})
        df1 = df1.join(df_temp)
    print(df1)

    dates = pd.date_range('2010-01-01', '2010-12-31')
    symbols = ['GOOG', 'IBM', 'GLD']  # SPY will be added in get_data()
    df = get_data(symbols, dates)
    print(df)

    """
    lecture uses.ix that doesn't work anymore, replace it with .loc
    """
    # Slice by row range (dates) using DataFram.ix[] selector
    print(df.loc['2010-01-01':'2010-01-31'])
    #  Slice by column (symbols)
    print(df['GOOG'])
    print(df[['IBM', 'GLD']])  # a list of labels selects multiple columns
    print(df.loc[:, ['SPY', 'IBM']])
    # Slice by row and column
    print(df.loc['2010-03-01':'2010-03-15', ['SPY', 'IBM']])

    plot_data(df)
    plot_selected(df, ['SPY', 'IBM'], '2010-03-01', '2010-04-01')
    df_normalized = normalize_data(df)
    plot_data(df_normalized, title="Normalized Stock Prices")
    corr_matrix = df_normalized.pct_change().corr()

    plt.figure(figsize=(6, 4))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, linewidths=0.5)
    plt.title('Correlation Heatmap: Daily Returns')
    plt.show()





def create_arrays():
    print("=== Creating Arrays ===")
    print(np.array([2, 3, 4]))  # 1D array
    print(np.array([(2,3,4), (5,6,7)]))  # 2D array
    print(np.empty(5))
    print(np.empty((5,4)))
    print(np.empty((5,4,3)))
    print(np.ones((5,4)))
    print(np.ones((5,4), dtype=np.int_))
    print()

def random_numbers():
    print("=== Random Numbers ===")
    print(np.random.random((5,4)))
    print(np.random.rand(5,4))
    print(np.random.normal(size=(2,3)))
    print(np.random.normal(50,10,size=(2,3)))
    print(np.random.randint(0,10))
    print(np.random.randint(0,10, size=5))
    print(np.random.randint(0,10, size=(2,3)))
    print()

def array_attributes():
    print("=== Array Attributes ===")
    a = np.random.random((5,4))
    print("Array:\n", a)
    print("Shape:", a.shape)
    print("Rows:", a.shape[0], "Columns:", a.shape[1])
    print("Number of dimensions:", len(a.shape))
    print("Size:", a.size)
    print("Data type:", a.dtype)
    print()

def array_operations():
    print("=== Array Operations ===")
    np.random.seed(693)
    a = np.random.randint(0, 10, size=(5,4))
    print("Array:\n", a)
    print("Sum of all elements:", a.sum())
    print("Sum of each column:", a.sum(axis=0))
    print("Sum of each row:", a.sum(axis=1))
    print("Min of each column:", a.min(axis=0))
    print("Max of each row:", a.max(axis=1))
    print("Mean of all elements:", a.mean())
    print()

def access_elements():
    print("=== Accessing Array Elements ===")
    a = np.random.rand(5,4)
    print("Array:\n", a)
    print("Element (3,2):", a[3,2])
    print("Slice (0,1:3):", a[0,1:3])
    print("Top-left 2x2:", a[0:2,0:2])
    print("Columns 0 & 2:", a[:,0:3:2])
    a[0,0] = 1
    print("Modified one element:\n", a)
    a[0,:] = 2
    print("Modified row with single value:\n", a)
    a[:,3] = [1,2,3,4,5]
    print("Modified column with list:\n", a)
    print()

def arithmetic_operations():
    print("=== Arithmetic Operations ===")
    a = np.array([(1,2,3,4,5),(10,20,30,40,50)])
    b = np.array([(100,200,300,400,500),(1,2,3,4,5)])
    print("a:\n", a)
    print("a*2:\n", a*2)
    print("a/2:\n", a/2)
    print("a/2.0:\n", a/2.0)
    print("b:\n", b)
    print("a+b:\n", a+b)
    print("a*b:\n", a*b)
    print("a/b:\n", a/b)
    print()

def max_value_example():
    print("=== Maximum Value Example ===")
    a = np.array([9,6,2,3,12,14,7,10], dtype=np.int32)
    print("Array:", a)
    print("Maximum value:", a.max())
    print("Index of max:", np.argmax(a))
    print()

def timing_example():
    print("=== Timing Example ===")
    t1 = time.time()
    print("ML4T")
    t2 = time.time()
    print("Time taken by print statement:", t2-t1, "seconds")
    print()

def masking_example():
    print("=== Masking and Indexing ===")
    a = np.random.rand(5)
    indices = np.array([1,2,3,3])
    print("Original array:", a)
    print("Access using indices:", a[indices])

    a = np.array([(20,25,10,23,26,32,10,5,0),(0,2,50,20,0,1,28,5,0)])
    print("2D Array:\n", a)
    mean = a.mean()
    print("Mean:", mean)
    print("Elements less than mean:", a[a<mean])
    a[a<mean] = mean
    print("After replacing elements < mean with mean:\n", a)
    print()


def test_run_1_3():
    create_arrays()
    random_numbers()
    array_attributes()
    array_operations()
    access_elements()
    arithmetic_operations()
    max_value_example()
    timing_example()
    masking_example()

def get_rolling_mean(values, window):
    """Return rolling mean of given values, using specified window size."""
    return values.rolling(window=window).mean()

def get_rolling_std(values, window):
    """Return rolling standard deviation of given values, using specified window size."""
    return values.rolling(window=window).std()

def get_bollinger_bands(rm, rstd):
    """Return upper and lower Bollinger Bands."""
    upper_band = rm + 2 * rstd
    lower_band = rm - 2 * rstd
    return upper_band, lower_band

def compute_daily_returns(df):
    """Compute and return the daily return values."""
    daily_returns = df.copy()
    daily_returns = (df / df.shift(1)) - 1
    # daily_returns[1:] =(df[1:] / df[:-1].values)-1
    daily_returns.iloc[0, :] = 0
    return daily_returns

def test_run_1_4():
    # Dates and symbols
    dates = pd.date_range('2012-01-01', '2012-12-31')
    symbols = ['SPY','XOM','GOOG','GLD']

    # Get data
    df = get_data(symbols, dates)

    # --- Global statistics ---
    print("=== Global statistics ===")
    print("Mean:\n", df.mean())
    print("Median:\n", df.median())
    print("Std deviation:\n", df.std())

    # --- Rolling mean example ---
    print("=== Rolling Mean ===")
    window = 20
    rm_SPY = get_rolling_mean(df['SPY'], window)
    ax = df['SPY'].plot(title="SPY Rolling Mean", label='SPY')
    rm_SPY.plot(label='Rolling Mean', ax=ax)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc='upper left')
    plt.show()

    # --- Bollinger Bands ---
    print("=== Bollinger Bands ===")
    rstd_SPY = get_rolling_std(df['SPY'], window)
    upper_band, lower_band = get_bollinger_bands(rm_SPY, rstd_SPY)
    ax = df['SPY'].plot(title="Bollinger Bands", label='SPY')
    rm_SPY.plot(label='Rolling Mean', ax=ax)
    upper_band.plot(label='Upper Band', ax=ax)
    lower_band.plot(label='Lower Band', ax=ax)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc='upper left')
    plt.show()

    # --- Daily returns ---
    print("=== Daily Returns ===")
    daily_returns = compute_daily_returns(df)
    plot_data(daily_returns, title="Daily Returns")

def fill_missing_values(df_data):
    """Fill missing values in data frame, in place."""
    pass  # QUIZ: Your code here (DO NOT modify anything else)
    df_data.fillna(method="ffill", inplace=True)
    df_data.fillna(method="bfill", inplace=True)

def test_run_1_5():
    """Function called by Test Run."""
    # Read data
    symbol_list = ["JAVA", "FAKE1", "FAKE2"]  # list of symbols
    start_date = "2005-12-31"
    end_date = "2014-12-07"
    dates = pd.date_range(start_date, end_date)  # date range as index
    df_data = get_data(symbol_list, dates)  # get data for each symbol

    # Fill missing values
    fill_missing_values(df_data)

    # Plot
    plot_data(df_data)

def test_run_1_6():
    # 1. Read data
    dates = pd.date_range('2009-01-01', '2012-12-31')
    symbols = ['SPY', 'XOM', 'GLD']
    df = get_data(symbols, dates)
    plot_data(df, title="Adjusted Close Prices")

    # 2. Compute daily returns
    daily_returns = compute_daily_returns(df)
    plot_data(daily_returns, title="Daily Returns")

    # 3. Histogram for one stock (SPY)
    daily_returns['SPY'].hist(bins=20)
    plt.title("SPY Daily Returns")
    plt.xlabel("Daily return")
    plt.ylabel("Frequency")
    plt.show()

    # 4. Histogram with stats (mean, std, kurtosis)
    mean = daily_returns['SPY'].mean()
    std = daily_returns['SPY'].std()
    kurt = daily_returns['SPY'].kurtosis()
    print("SPY mean =", mean)
    print("SPY std  =", std)
    print("SPY kurt =", kurt)
    ax = daily_returns['SPY'].hist(bins=20)
    ax.axvline(mean, color='w', linestyle='dashed', linewidth=2)
    ax.axvline(std, color='r', linestyle='dashed', linewidth=2)
    ax.axvline(-std, color='r', linestyle='dashed', linewidth=2)
    plt.title("SPY Daily Returns with Stats")
    plt.show()

    # 5. Two histograms together (SPY vs XOM)
    daily_returns['SPY'].hist(bins=20, alpha=0.5, label="SPY")
    daily_returns['XOM'].hist(bins=20, alpha=0.5, label="XOM")
    plt.legend(loc='upper right')
    plt.title("SPY vs XOM Daily Returns")
    plt.show()

    # Overlaying a normal curve
    mu, sigma = daily_returns['SPY'].mean(), daily_returns['SPY'].std()
    x = np.linspace(-0.05, 0.05, 100)
    plt.hist(daily_returns['SPY'], bins=20, density=True, alpha=0.6, color='g')
    plt.plot(x, 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)), 'r--')
    plt.title('SPY Returns vs Normal Distribution')
    plt.show()

    # 6. Scatterplots with regression line
    for symbol in ['XOM', 'GLD']:
        daily_returns.plot(kind='scatter', x='SPY', y=symbol)
        beta, alpha = np.polyfit(daily_returns['SPY'], daily_returns[symbol], 1)
        print(f"{symbol} beta =", beta)
        print(f"{symbol} alpha =", alpha)
        plt.plot(daily_returns['SPY'], beta*daily_returns['SPY'] + alpha, '-', color='r')
        plt.title(f"{symbol} vs SPY")
        plt.show()




if __name__ == "__main__":
    # Pandas
    # test_run_1_1()
    # test_run_1_2()

    # Numpy
    # test_run_1_3()

    # Statistical Analysis of time series
    # test_run_1_4()

    # Incomplete Data
    # test_run_1_5()

    #Histogram and Scatterplot
    test_run_1_6()



