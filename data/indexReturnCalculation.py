import os
import sys

import numpy as np
import pandas as pd

def calculate_index_arithmetic_return(df: pd.DataFrame, start_date: str) -> float:
    """
    Calculate the arithmetic (simple) cumulative return of the index from `start_date`
    to the last available date by summing each day's index return.
    """
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    daily_index_return = df.groupby('Date')['WeightedRet'].sum()
    period = daily_index_return[daily_index_return.index >= pd.to_datetime(start_date)]
    return float(period.sum())

def calculate_index_geometric_return(df: pd.DataFrame, start_date: str) -> float:
    """
    Calculate the compounded (geometric) cumulative return of the index
    from `start_date` to the last available date.
    """
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    daily_index_return = df.groupby('Date')['WeightedRet'].sum()
    period = daily_index_return[daily_index_return.index >= pd.to_datetime(start_date)]
    return float((period + 1).prod() - 1)

def get_index_prices(df: pd.DataFrame) -> pd.Series:
    """
    Build the index 'price' series as the sum of Close prices per date.
    """
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    return df.groupby('Date')['Close'].sum()

if __name__ == "__main__":
    csv_path = '/Users/maxmagnusson/Documents/TIF360/ANN - Project L/data/OMXS22_raw_features.csv'

    if not os.path.isfile(csv_path):
        sys.exit(f"ERROR: CSV not found at {csv_path!r} (cwd={os.getcwd()!r})")

    df = pd.read_csv(csv_path, parse_dates=['Date'])
    df.columns = df.columns.str.strip()

    if 'WeightedRet' not in df.columns:
        if {'Return', 'Weight'}.issubset(df.columns):
            df['WeightedRet'] = df['Return'] * df['Weight']
            print("Note: computed `WeightedRet` = Return * Weight", file=sys.stderr)
        else:
            sys.exit(f"ERROR: Neither `WeightedRet` nor both `Return` and `Weight` found. Columns: {df.columns.tolist()}")

    start_date = '2023-04-17'
    end_date = df['Date'].max().date()

    arith = calculate_index_arithmetic_return(df, start_date)
    geo   = calculate_index_geometric_return(df, start_date)

    price_series = get_index_prices(df)
    try:
        price_start = price_series.loc[pd.to_datetime(start_date)]
    except KeyError:
        sys.exit(f"ERROR: No index price available exactly on {start_date}")
    price_end = price_series.loc[pd.to_datetime(end_date)]

    print(f"Index price on {start_date}:          {price_start:.2f}")
    print(f"Index price on {end_date}:          {price_end:.2f}")
    print(f"Arithmetic total return from {start_date} to {end_date}: {arith:.2%}")
    print(f"Geometric total return  from {start_date} to {end_date}: {geo:.2%}")
