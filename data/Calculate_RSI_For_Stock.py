import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta

def calculate_rsi_for_stock(ticker, data_type='Close', csv_file='/Users/maxmagnusson/Documents/TIF360/ANN - Project L/data/OMXS30_underlying_raw.csv', period=14, plot=True):
    raw = pd.read_csv(csv_file, header=None)

    level_0 = raw.iloc[0, 1:]  
    level_1 = raw.iloc[1, 1:]

    multi_index = pd.MultiIndex.from_arrays([level_0, level_1])

    df = raw.iloc[2:].copy()
    df.reset_index(drop=True, inplace=True)
    df.columns = ['Date'] + list(multi_index)
    
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df.set_index('Date', inplace=True)

    try:
        price_series = pd.to_numeric(df[(data_type, ticker)], errors='coerce')
    except KeyError:
        raise ValueError(f"No column found for ({data_type}, {ticker})")

    rsi = ta.momentum.RSIIndicator(close=price_series, window=period).rsi()

    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(rsi, label=f'RSI {period} - {ticker}')
        plt.axhline(70, color='red', linestyle='--', label='Overbought')
        plt.axhline(30, color='green', linestyle='--', label='Oversold')
        plt.title(f'RSI ({period}-day) for {ticker}')
        plt.xlabel('Date')
        plt.ylabel('RSI')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return rsi

if __name__ == '__main__':
    # Change the ticker here to test different stocks
    test_ticker = 'SAAB-B.ST'
    calculate_rsi_for_stock(test_ticker)