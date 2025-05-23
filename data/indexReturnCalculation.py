#!/usr/bin/env python3
import os
import pandas as pd

def calculate_index_final_return_from_df(df: pd.DataFrame, start_date: str) -> float:
    """
    Calculate the cumulative return of an index from `start_date` to the last available date.

    Parameters:
    - df: DataFrame containing at least ['Date', 'WeightedRet'].
    - start_date: 'YYYY-MM-DD' string marking the beginning of the period.

    Returns:
    - Cumulative return (e.g., 0.05 for +5%).
    """
    # 1. Ensure datetime & sort
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    # 2. Sum weighted returns per day
    index_daily = df.groupby('Date')['WeightedRet'].sum()

    # 3. Filter from the start date onward
    period = index_daily[index_daily.index >= pd.to_datetime(start_date)]

    # 4. Compound and return
    return (1 + period).prod() - 1

# ─── Change this to your actual absolute path ───────────────────────────────
csv_path = '/Users/maxmagnusson/Documents/TIF360/ANN - Project L/data/OMXS22_model_features_raw.csv'
# ──────────────────────────────────────────────────────────────────────────

# Sanity-check that the file exists before loading
if not os.path.isfile(csv_path):
    raise FileNotFoundError(
        f"CSV not found at {csv_path!r}. "
        f"Current working dir: {os.getcwd()!r}"
    )

# Load the data
df = pd.read_csv(csv_path, parse_dates=['Date'])

# Compute final return from your chosen start date
start_date = '2023-04-17'
final_return = calculate_index_final_return_from_df(df, start_date)

# Print it out in percent form
print(f"Final return from {start_date} to the last date in the CSV: {final_return:.2%}")
