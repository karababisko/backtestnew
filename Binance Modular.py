## 1. Veri Toplama Modülü
import ccxt
import pandas as pd
import mplfinance as mpf
import tkinter as tk

def fetch_data(symbol, timeframe, since):
    try:
        exchange = ccxt.binance({
            'apiKey': '',
            'secret': '',
        })
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since)
        return ohlcv
    except ccxt.BaseError as e:
        print(f"An error occurred while fetching data: {str(e)}")
        return None









## 2. Data Processing Module

def calculate_ema(df, period):
    multiplier = 2 / (period + 1)
    return df['close'].ewm(alpha=multiplier, adjust=False).mean()

def calculate_wma(df, weights):
    weights = pd.Series(weights, index=df['close'].tail(len(weights)).index)
    weighted_sum = (df['close'].tail(len(weights)) * weights).sum()
    return weighted_sum / sum(weights)

def calculate_long_ma(df, period):
    return df['close'].rolling(window=period).mean()

def calculate_macd(df, short_period=12, long_period=26, signal_period=9):
    ema_short = df['close'].ewm(span=short_period, adjust=False).mean()
    ema_long = df['close'].ewm(span=long_period, adjust=False).mean()
    macd_line = ema_short - ema_long
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    return macd_line, signal_line

def calculate_bollinger_bands(df, period=20, std_dev=2):
    rolling_mean = df['close'].rolling(window=period).mean()
    rolling_std = df['close'].rolling(window=period).std()
    upper_band = rolling_mean + (rolling_std * std_dev)
    lower_band = rolling_mean - (rolling_std * std_dev)
    return upper_band, lower_band

def calculate_stochastic_oscillator(df, period):
    high_max = df['high'].rolling(window=period).max()
    low_min = df['low'].rolling(window=period).min()
    percent_k = ((df['close'] - low_min) / (high_max - low_min)) * 100
    percent_d = percent_k.rolling(window=3).mean()
    return percent_k, percent_d

# Average True Range (ATR)
def calculate_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    true_range = high_low.combine(high_close, max).combine(low_close, max)
    return true_range.rolling(window=period).mean()

# Average Directional Index (ADX)
def calculate_adx(df, period=14):
    high_diff = df['high'].diff()
    low_diff = df['low'].diff().abs()
    directional_movement = high_diff.where(high_diff > low_diff, low_diff).fillna(0)
    smoothed_dm = directional_movement.ewm(alpha=1/period, adjust=False).mean()
    true_range = calculate_atr(df, period)
    di = (smoothed_dm / true_range) * 100
    return di.rolling(window=period).mean()

# On-Balance Volume (OBV)
def calculate_obv(df):
    volume_diff = df['volume'].diff().where(df['close'].diff() > 0, -df['volume'].diff()).fillna(0)
    return volume_diff.cumsum()

# Volume Weighted Average Price (VWAP)
def calculate_vwap(df):
    vwap = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    return vwap

# Ichimoku Cloud
def calculate_ichimoku_cloud(df, conversion_period=9, base_period=26, leading_span=52, lagging_span=26):
    conversion_line = (df['high'].rolling(window=conversion_period).max() + df['low'].rolling(window=conversion_period).min()) / 2
    base_line = (df['high'].rolling(window=base_period).max() + df['low'].rolling(window=base_period).min()) / 2
    leading_span_a = (conversion_line + base_line) / 2
    leading_span_b = (df['high'].rolling(window=leading_span).max() + df['low'].rolling(window=leading_span).min()) / 2
    lagging_span = df['close'].shift(-lagging_span)
    return conversion_line, base_line, leading_span_a.shift(leading_span), leading_span_b.shift(leading_span), lagging_span

# Elder's Force Index (EFI)
def calculate_efi(df, period=13):
    return (df['close'].diff() * df['volume']).ewm(span=period).mean()

def process_data(ohlcv, short_ma_period, long_ma_period, parameters):
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    # MACD
    df['macd'], df['signal_line'] = calculate_macd(df, parameters['macd_short'], parameters['macd_long'], parameters['macd_signal'])

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # EMA
    df['ema'] = calculate_ema(df, parameters['ema_period'])

    # WMA
    df['wma'] = calculate_wma(df, parameters['wma_weights'])

    # MA long
    df['ma_long'] = calculate_long_ma(df, long_ma_period)
    df['ma_short'] = calculate_short_ma(df, short_ma_period)

    # Stochastic Oscillator
    df['percent_k'], df['percent_d'] = calculate_stochastic_oscillator(df, parameters['stochastic_period'])

    # ATR
    df['atr'] = calculate_atr(df, parameters['atr_period'])

    # Bollinger Bands
    df['bollinger_upper'], df['bollinger_lower'] = calculate_bollinger_bands(df, parameters['bollinger_period'], parameters['bollinger_std_dev'])

    # ADX
    df['adx'] = calculate_adx(df, parameters['adx_threshold'])

    # OBV
    df['obv'] = calculate_obv(df)

    # VWAP
    df['vwap'] = calculate_vwap(df)

    # Ichimoku Cloud
    df['ichimoku_conversion_line'], df['ichimoku_base_line'], df['ichimoku_leading_span_a'], df['ichimoku_leading_span_b'], df['ichimoku_lagging_span'] = calculate_ichimoku_cloud(
        df,
        parameters['ichimoku_conversion_line_period'],
        parameters['ichimoku_base_line_period'],
        parameters['ichimoku_leading_span_period'],
        parameters['ichimoku_lagging_span2_period']
    )

    # EFI
    df['efi'] = calculate_efi(df, parameters['efi_period'])

    return df
 



## INPUT VALIDATION
def on_submit_trading_pair():
    symbol = symbol_entry.get() # Assuming symbol_entry is the entry widget for the trading pair
    if not validate_trading_pair(symbol):
        print("Invalid trading pair. Please enter a valid trading pair.")
        return
    # Rest of the code to process the trading pair

def validate_trading_pair(symbol):
    if '/' not in symbol:
        return False
    return True






# 3. Strategy Determination
def generate_individual_signals(df, selected_indicators, parameters):
    signals = pd.DataFrame(index=df.index)

    # Moving Averages (EMA/WMA)
    if "EMA" in selected_indicators:
        signals['buy_signal_ema'] = (df['ema'] > df['ma_long']) & (df['ema'].shift(1) <= df['ma_long'].shift(1))
        signals['sell_signal_ema'] = (df['ema'] < df['ma_long']) & (df['ema'].shift(1) >= df['ma_long'].shift(1))
    if "WMA" in selected_indicators:
        signals['buy_signal_wma'] = (df['wma'] > df['ma_long']) & (df['wma'].shift(1) <= df['ma_long'].shift(1))
        signals['sell_signal_wma'] = (df['wma'] < df['ma_long']) & (df['wma'].shift(1) >= df['ma_long'].shift(1))

    # MACD
    if "MACD" in selected_indicators:
        signals['buy_signal_macd'] = (df['macd'] > df['signal_line']) & (df['macd'].shift(1) <= df['signal_line'].shift(1))
        signals['sell_signal_macd'] = (df['macd'] < df['signal_line']) & (df['macd'].shift(1) >= df['signal_line'].shift(1))

    # RSI
    if "RSI" in selected_indicators:
        signals['buy_signal_rsi'] = df['rsi'] < parameters['rsi_lower']
        signals['sell_signal_rsi'] = df['rsi'] > parameters['rsi_upper']

    # OBV
    if "OBV" in selected_indicators:
        signals['buy_signal_obv'] = df['obv'].diff() > 0
        signals['sell_signal_obv'] = df['obv'].diff() < 0

    # Ichimoku Cloud
    if "Ichimoku Cloud" in selected_indicators:
        signals['buy_signal_ichimoku'] = (df['ichimoku_conversion_line'] > df['ichimoku_base_line']) & (df['ichimoku_conversion_line'].shift(1) <= df['ichimoku_base_line'].shift(1))
        signals['sell_signal_ichimoku'] = (df['ichimoku_conversion_line'] < df['ichimoku_base_line']) & (df['ichimoku_conversion_line'].shift(1) >= df['ichimoku_base_line'].shift(1))

    # ATR (Example: You can define specific rules for ATR based on parameters['atr_period'])
    if "ATR" in selected_indicators:
        atr_buy_threshold = parameters['atr_buy_threshold']
        atr_sell_threshold = parameters['atr_sell_threshold']
        signals['buy_signal_atr'] = df['atr'] < atr_buy_threshold
        signals['sell_signal_atr'] = df['atr'] > atr_sell_threshold

    # ADX
    if "ADX" in selected_indicators:
        signals['buy_signal_adx'] = df['adx'] > parameters['adx_threshold']

    # EFI
    if "EFI" in selected_indicators:
        signals['buy_signal_efi'] = df['efi'] > 0
        signals['sell_signal_efi'] = df['efi'] < 0

    # VWAP
    if "VWAP" in selected_indicators:
        signals['buy_signal_vwap'] = df['close'] > df['vwap']
        signals['sell_signal_vwap'] = df['close'] < df['vwap']

    # Stochastic Oscillator
    if "Stochastic Oscillator" in selected_indicators:
        signals['buy_signal_stochastic'] = df['percent_k'] < parameters['stochastic_lower']
        signals['sell_signal_stochastic'] = df['percent_k'] > parameters['stochastic_upper']

    # Bollinger Bands
    if "Bollinger Bands" in selected_indicators:
        bollinger_middle_band = df['close'].rolling(window=parameters['bollinger_period']).mean()
        bollinger_std_dev = df['close'].rolling(window=parameters['bollinger_period']).std()
        bollinger_upper_band = bollinger_middle_band + (bollinger_std_dev * parameters['bollinger_std_dev'])
        bollinger_lower_band = bollinger_middle_band - (bollinger_std_dev * parameters['bollinger_std_dev'])
        signals['buy_signal_bollinger'] = df['close'] < bollinger_lower_band
        signals['sell_signal_bollinger'] = df['close'] > bollinger_upper_band    

    return signals

def on_test_strategy():
    # Retrieve selected indicators and adjusted parameters
    selected_indicators = [indicator for indicator, var in indicators_var.items() if var.get()]
    parameters = {
        'ema_period': int(ema_period_entry.get()),
        'wma_weights': list(map(float, wma_weights_entry.get().split(','))),
        'stochastic_period': int(stochastic_period_entry.get()),
        'atr_period': int(atr_period_entry.get()),
        'rsi_upper': int(rsi_upper_entry.get()),
        'rsi_lower': int(rsi_lower_entry.get()),
        'adx_threshold': int(adx_threshold_entry.get()),
        'efi_period': int(efi_period_entry.get()),
        'macd_short': int(macd_short_entry.get()),
        'macd_long': int(macd_long_entry.get()),
        'macd_signal': int(macd_signal_entry.get()),
        'bollinger_period': int(bollinger_period_entry.get()),
        'bollinger_std_dev': int(bollinger_std_dev_entry.get()),
        'ichimoku_conversion_line_period': int(ichimoku_conversion_line_period_entry.get()),
        'ichimoku_base_line_period': int(ichimoku_base_line_period_entry.get()),
        'ichimoku_leading_span_b_period': int(ichimoku_leading_span_b_period_entry.get()),
        'ichimoku_lagging_span_period': int(ichimoku_lagging_span_period_entry.get())
}

    print("Testing Strategy with Indicators:", selected_indicators)
    print("Parameters:", parameters)

    # Fetch and process data
    ohlcv = fetch_data(symbol, timeframe, since)
    df = process_data(ohlcv, ma_short, ma_long, **parameters)
    df = generate_individual_signals(df, selected_indicators, parameters)
    profit, transactions = calculate_profit(df)

    # Retrieve selected symbol and timeframe
    symbol = symbol_entry.get()
    timeframe = timeframe_entry.get()
    
    # Generate buy/sell signals
    df = generate_individual_signals(df, selected_indicators, parameters)

    # Display profit in GUI
    profit, transactions_df = calculate_profit(df)
    profit_display.config(text=f"{profit:.2f}")

    # Display transactions in GUI
    transactions_text.delete(1.0, tk.END)  # Clear existing text
    for _, row in transactions_df.iterrows():
        transactions_text.insert(tk.END, f"{row['Tarih']} - {row['İşlem']} - {row['Fiyat']:.2f} - {row['Bakiye']:.2f}\n")
    
    # Visualize results
    visualize_data(df)
    show_transactions(df)









## 4. Profit Calculation Module
def calculate_profit(df):
    initial_balance = 100000
    balance = initial_balance
    btc_balance = 0
    transactions = []

    # Loop through the DataFrame
    for index, row in df.iterrows():
        # Check individual buy signals
        buy_signals = [col for col in df.columns if 'buy_signal' in col and row[col]]
        if buy_signals and balance > 0:
            btc_balance += balance / row['close']
            balance = 0
            transactions.append((index, "Buy", row['close'], balance + btc_balance * row['close']))
        
        # Check individual sell signals
        sell_signals = [col for col in df.columns if 'sell_signal' in col and row[col]]
        if sell_signals and btc_balance > 0:
            balance += btc_balance * row['close']
            btc_balance = 0
            transactions.append((index, "Sell", row['close'], balance))
    
    final_balance = balance + btc_balance * df.iloc[-1]['close']
    profit = final_balance - initial_balance
    transactions_df = pd.DataFrame(transactions, columns=['Date', 'Transaction', 'Price', 'Balance'])
    return profit, transactions_df







## 5. Visulation Module
def visualize_data(df, selected_indicators):
    for indicator in selected_indicators:
        buy_signals_df = df['close'].copy()
        buy_signals_df[:] = float('nan')
        buy_signals_df[df[f'buy_signal_{indicator.lower()}']] = df['close'][df[f'buy_signal_{indicator.lower()}']]
        sell_signals_df = df['close'].copy()
        sell_signals_df[:] = float('nan')
        sell_signals_df[df[f'sell_signal_{indicator.lower()}']] = df['close'][df[f'sell_signal_{indicator.lower()}']]
        
        apds = [] # Add your specific addplot configurations for each indicator
        if indicator == "EMA":
            apds.append(mpf.make_addplot(df['ema'], color='blue'))
        # Add similar conditions for other indicators
        
        apds.append(mpf.make_addplot(buy_signals_df, scatter=True, markersize=100, marker='^', color='g'))
        apds.append(mpf.make_addplot(sell_signals_df, scatter=True, markersize=100, marker='v', color='r'))
        
        mpf.plot(df, type='candle', volume=True, addplot=apds, figratio=(12, 10), block=False, title=indicator)

def show_transactions(df, selected_indicators):
    for indicator in selected_indicators:
        transactions_window = tk.Toplevel(root)
        transactions_window.title(f"İşlem Ayrıntıları - {indicator}")
        transactions_label = tk.Label(transactions_window, text="Tarih\t\tSaat\t\tFiyat\t\tİşlem")
        transactions_label.pack()
        for index, row in df.iterrows():
            if row[f'buy_signal_{indicator.lower()}'] or row[f'sell_signal_{indicator.lower()}']:
                date_time = index.strftime("%Y-%m-%d\t%H:%M:%S")
                price = row['close']
                action = "Alım" if row[f'buy_signal_{indicator.lower()}'] else "Satım"
                transaction_detail = f"{date_time}\t{price:.2f}\t\t{action}"
                transaction_label = tk.Label(transactions_window, text=transaction_detail)
                transaction_label.pack()









## 6. GUI Modülü
from tkinter import *
from tkinter import ttk

# Function to handle the selection of indicators
def on_select_indicators():
    selected_indicators = [indicator for indicator, var in indicators_var.items() if var.get()]
    print("Selected Indicators:", selected_indicators)


# Function to handle the adjustment of parameters
def on_adjust_parameters():
    global parameters, selected_indicators

    # Define the variables
    symbol = 'BTC/USD'  # Symbol for the asset you want to analyze
    timeframe = '1h'   # Timeframe for the data, e.g., '1h' for 1-hour intervals
    since = None       # Timestamp for the beginning of the data; None fetches as much as possible
    ma_short = 10      # Short moving average period
    ma_long = 50       # Long moving average period

    parameters = {
        'ema_period': int(ema_period_entry.get()),
        'wma_weights': list(map(float, wma_weights_entry.get().split(','))),
        'stochastic_period': int(stochastic_period_entry.get()),
        'stochastic_upper': int(stochastic_upper_entry.get()),  # Upper bound for Stochastic Oscillator
        'stochastic_lower': int(stochastic_lower_entry.get()),  # Lower bound for Stochastic Oscillator
        'atr_period': int(atr_period_entry.get()),
        'rsi_upper': int(rsi_upper_entry.get()),
        'rsi_lower': int(rsi_lower_entry.get()),
        'adx_threshold': int(adx_threshold_entry.get()),
        'efi_period': int(efi_period_entry.get()),
        'macd_short': int(macd_short_period_entry.get()),
        'macd_long': int(macd_long_period_entry.get()),
        'macd_signal': int(macd_signal_period_entry.get()),
        'bollinger_period': int(bollinger_period_entry.get()),
        'bollinger_std_dev': int(bollinger_std_dev_entry.get()),
        'ichimoku_conversion_line_period': int(ichimoku_conversion_line_period_entry.get()),
        'ichimoku_base_line_period': int(ichimoku_base_line_period_entry.get()),
        'ichimoku_lagging_span_period': int(ichimoku_lagging_span_entry.get()),
        'ichimoku_leading_span_period': int(ichimoku_leading_span_entry.get()),
        'ichimoku_lagging_span2_period': int(ichimoku_lagging_span2_period_entry.get()),
        'ichimoku_displacement': int(ichimoku_displacement_entry.get())
    }
    
    print("Adjusted Parameters:", parameters)
    # Code here to update the strategy with the adjusted parameters
    ohlcv = fetch_data(symbol, timeframe, since)
    df = process_data(ohlcv, ma_short, ma_long, **parameters)
    df = generate_individual_signals(df, selected_indicators, parameters)
    profit, transactions = calculate_profit(df)

    # Update the profit display
    profit_display.config(text=str(profit))

    # Update the transactions text widget
    transactions_text.delete(1.0, END)  # Clear previous content
    for transaction in transactions:
        transactions_text.insert(END, str(transaction) + '\n')

def create_labeled_entry(parent, label_text, default_value="", row_num=0):
    frame = Frame(parent)  # Creating a frame to hold label and entry
    frame.pack(pady=5)     # Packing the frame into the parent
    label = Label(frame, text=label_text)
    label.pack(side=LEFT, padx=5)   # Using pack instead of grid
    entry = Entry(frame)
    entry.pack(side=LEFT, padx=5)   # Using pack instead of grid
    entry.insert(0, default_value)
    return entry


# GUI initialization
root = Tk()
root.title("Trading Strategy Builder")

# Create a notebook (tabbed interface) to organize the indicators
notebook = ttk.Notebook(root)
notebook.pack(expand=1, fill="both")

# Define the indicators_var dictionary
indicators_var = {
    "EMA": IntVar(),
    "WMA": IntVar(),
    "Stochastic Oscillator": IntVar(),
    "ATR": IntVar(),
    "RSI": IntVar(),
    "ADX": IntVar(),
    "OBV": IntVar(),
    "VWAP": IntVar(),
    "Ichimoku Cloud": IntVar(),
    "MACD": IntVar(),
    "EFI": IntVar(),
}


# Indicator Selection Panel
indicator_frame = ttk.Frame(notebook)
indicator_frame.grid(row=0, column=0, sticky=(N, W, E, S))

# Checkboxes for each indicator
for row, (indicator, var) in enumerate(indicators_var.items()):
    c = Checkbutton(indicator_frame, text=indicator, variable=var)
    c.grid(row=row, column=0, padx=5, pady=5)

# Button to select indicators
select_button = Button(indicator_frame, text="Select Indicators", command=on_select_indicators)
select_button.grid(row=len(indicators_var), column=0, padx=5, pady=5)

# Parameter Adjustment Panel
parameter_frame_outer = ttk.Frame(notebook)
parameter_frame_outer.grid(row=0, column=1, sticky=(N, W, E, S))

parameter_canvas = Canvas(parameter_frame_outer)
parameter_canvas.pack(side=LEFT, fill=BOTH, expand=YES)

scrollbar = Scrollbar(parameter_frame_outer, orient="vertical", command=parameter_canvas.yview)
scrollbar.pack(side=RIGHT, fill=Y)
parameter_canvas.config(yscrollcommand=scrollbar.set)

scrollable_frame = ttk.Frame(parameter_canvas)
parameter_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

def configure_scroll_region(event):
    parameter_canvas.config(scrollregion=parameter_canvas.bbox("all"))
scrollable_frame.bind("<Configure>", configure_scroll_region)


# EMA Period
ema_period_entry = create_labeled_entry(scrollable_frame, "EMA Period:", "5")

# WMA Weights
wma_weights_entry = create_labeled_entry(scrollable_frame, "WMA Weights (comma-separated):", "0.2,0.2,0.2,0.2,0.2")

# Stochastic Oscillator Period
stochastic_period_entry = create_labeled_entry(scrollable_frame, "Stochastic Oscillator Period:", "14")

# Stochastic Oscillator Upper Bound
stochastic_upper_entry = create_labeled_entry(scrollable_frame, "Stochastic Oscillator Upper Bound:", "80")

# Stochastic Oscillator Lower Bound
stochastic_lower_entry = create_labeled_entry(scrollable_frame, "Stochastic Oscillator Lower Bound:", "20")

# ATR Period
atr_period_entry = create_labeled_entry(scrollable_frame, "ATR Period:", "14")

# ADX Threshold
adx_threshold_entry = create_labeled_entry(scrollable_frame, "ADX Threshold:", "20")

# RSI Lower Bound
rsi_lower_entry = create_labeled_entry(scrollable_frame, "RSI Lower Bound:", "30")

# RSI Upper Bound
rsi_upper_entry = create_labeled_entry(scrollable_frame, "RSI Upper Bound:", "70")

# EFI Period
efi_period_entry = create_labeled_entry(scrollable_frame, "EFI Period:", "20")

# Bollinger Band Period
bollinger_period_entry = create_labeled_entry(scrollable_frame, "Bollinger Band Period:", "20")

# Bollinger Band Standard Deviation
bollinger_std_dev_entry = create_labeled_entry(scrollable_frame, "Bollinger Band Standard Deviation:", "2")

# Ichimoku Conversion Line Period
ichimoku_conversion_line_period_entry = create_labeled_entry(scrollable_frame, "Ichimoku Conversion Line Period:", "9")

# Ichimoku Base Line Period
ichimoku_base_line_period_entry = create_labeled_entry(scrollable_frame, "Ichimoku Base Line Period:", "26")

# Ichimoku Lagging Span 2 Period
ichimoku_lagging_span2_period_entry = create_labeled_entry(scrollable_frame, "Ichimoku Lagging Span 2 Period:", "52")

# Ichimoku Lagging Span Period
ichimoku_lagging_span_entry = create_labeled_entry(scrollable_frame, "Ichimoku Lagging Span Period:", "30")

# Ichimoku Displacement
ichimoku_displacement_entry = create_labeled_entry(scrollable_frame, "Ichimoku Displacement:", "26")

ichimoku_leading_span_entry = create_labeled_entry(scrollable_frame, "Ichimoku Leading Span Period:", "26")


# MACD Parameters
macd_short_period_entry = create_labeled_entry(scrollable_frame, "MACD Short Period:", "12")
macd_long_period_entry = create_labeled_entry(scrollable_frame, "MACD Long Period:", "26")
macd_signal_period_entry = create_labeled_entry(scrollable_frame, "MACD Signal Period:", "9")


# Button to adjust parameters
adjust_button = Button(scrollable_frame, text="Adjust Parameters", command=on_adjust_parameters)
adjust_button.pack(padx=5, pady=5)  # Using pack instead of grid


# Result Display Panel
result_frame = Frame(root) # or ttk.Frame(root)
notebook.add(result_frame, text="Results") # Add the result_frame to the notebook

# Profit Label
profit_label = Label(result_frame, text="Profit:")
profit_label.pack(side=LEFT, padx=5)
profit_display = Label(result_frame, text="")  # Placeholder for profit value
profit_display.pack(side=LEFT, padx=5)

# Transactions Text Widget with Scrollbar
transactions_label = Label(result_frame, text="Transactions:")
transactions_label.pack()
transactions_text_frame = Frame(result_frame)
transactions_text_frame.pack(fill=BOTH)
transactions_text = Text(transactions_text_frame, height=10, width=80)  # Text widget to display transactions
scrollbar = Scrollbar(transactions_text_frame, command=transactions_text.yview)
transactions_text['yscrollcommand'] = scrollbar.set
transactions_text.pack(side=LEFT, fill=BOTH, expand=YES)
scrollbar.pack(side=RIGHT, fill=Y)

# Trading Pair and Timeframe Entry
symbol_entry = create_labeled_entry(result_frame, "Trading Pair (e.g., 'BTC/USDT'):", "BTC/USDT")
timeframe_entry = create_labeled_entry(result_frame, "Timeframe (e.g., '1h', '4h', '1d'):", "4h")

# Submit Button inside Results Frame
submit_button = Button(result_frame, text="Submit", command=on_submit_trading_pair)
submit_button.pack(pady=5)  # Adjust padding as needed

# Add tabs to the notebook
notebook.add(indicator_frame, text="Indicators")
notebook.add(parameter_frame_outer, text="Parameters")  # Change this line to add the outer frame
notebook.add(result_frame, text="Results")

# Running the GUI
root.mainloop()