#%%
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
#%%
###
### NOTE TO SELF: This type of dataset should probably include
###       index returns, repo rate, USD/SEK, and similar as features,
###       as features for the model to capture the market context.
###

class TickerEmbeddedDataset(Dataset):
    def __init__(self, df, tickers, features, window=60, horizon=1):
        """
        df: DataFrame with columns ['Ticker', features..., 'Return']
        tickers: List of ticker-strings
        features: List of feature column names (excluding 'Return')
        window: Number of time steps in the input sequence
        horizon: Int or list of ints for the prediction horizon
                ex horizon=1 means predicting the next time step
                ex horizon=[1, 2] means predicting the next 1 and 2 time steps
        """
        self.df = df
        self.tickers = tickers
        self.features = features
        self.window = window
        self.horizon = list(horizon) if isinstance(horizon, list) else [horizon]
        
        Xs, Ts, ys = [], [], []
        ticker2idx = {ticker: i for i, ticker in enumerate(tickers)}

        for t in tickers:
            sub = df[df['Ticker'] == t]
            data = sub[features].values
            returns = sub['Return'].values
            n = len(sub) - window - max(self.horizon)

            for i in range(n):
                Xs.append(data[i:i+window])
                label = [returns[i + window + h] for h in self.horizon]
                ys.append(label)
                Ts.append(ticker2idx[t])

        self.X = torch.tensor(np.stack(Xs), dtype=torch.float32)    # (N, window, features)
        self.T = torch.tensor(Ts, dtype=torch.long)                 # (N,)
        self.y = torch.tensor(np.stack(ys), dtype=torch.float32)    # (N, len(horizon))

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return {'X': self.X[idx], 'T': self.T[idx], 'y': self.y[idx]}

if __name__ == "__main__":
    print("Testing TickerEmbeddedDataset...")
    # Example usage
    df = pd.read_csv("OMXS22_model_features_raw.csv", index_col= 'Date', parse_dates=True)
    tickers = df["Ticker"].unique().tolist()
    features = [c for c in df.columns if c not in ("Ticker", "Return")]
    window = 60
    horizon = 1

    ds = TickerEmbeddedDataset(df, tickers, features, window, horizon)

    print(f"Dataset length: {len(ds)} samples")
    print(f"First sample: {ds[0]}")
    print(f"First sample X shape: {ds[0]['X'].shape}")
    print(f"First sample T shape: {ds[0]['T'].shape}")
    print(f"First sample y shape: {ds[0]['y'].shape}")

    loader = DataLoader(ds, batch_size=32, shuffle=True)

    print("DataLoader test:")
    print(f"Batch X shape: {next(iter(loader))['X'].shape}")
    print(f"Batch T shape: {next(iter(loader))['T'].shape}")
    print(f"Batch y shape: {next(iter(loader))['y'].shape}")
    print("TickerEmbeddedDataset test complete.")

# %%
