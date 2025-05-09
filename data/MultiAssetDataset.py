#%%
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
###
### NOTE TO SELF: This type of dataset reduces the total amount of data
### and leads to a high increase in number of input features.
### This dataset is not adjusted to include macro-indicators, only
### data for multiple assets (such as stocks, commodities etc.)
###

class MultiAssetDataset(Dataset):
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
        
        dates = df.index.unique().sort_values()

        T = len(dates)
        A = len(tickers)
        F = len(features)
        H = len(self.horizon)
        max_h = max(self.horizon)

        arr = np.zeros((T, A, F), dtype=float)
        rets = np.zeros((T, A), dtype=float)
        for i, t in enumerate(tickers):
            sub = df[df['Ticker'] == t].reindex(dates)
            arr[:, i, :] = sub[features].values
            rets[:, i] = sub['Return'].values
        
        Xs = []
        ys = []
        n_samples = T - window - max_h + 1

        for i in range(n_samples):
            Xs.append(arr[i : i + window])

            cumulative_rets = []
            for h in self.horizon:
                
                slice_rets = rets[i + window : i + window + h]
                cumulative = np.prod(slice_rets + 1, axis=0) - 1
                cumulative_rets.append(cumulative)

            
            y_i = np.stack(cumulative_rets, axis = -1)
            ys.append(y_i)

        self.X = torch.tensor(np.stack(Xs), dtype=torch.float32) # (N, W, A, F)
        self.y = torch.tensor(np.stack(ys), dtype=torch.float32) # (N, A, H)

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
if __name__ == "__main__":
    df = pd.read_csv("OMXS22_model_features_raw.csv", index_col="Date", parse_dates=True)
    tickers = df["Ticker"].unique().tolist()
    features = ["Return","Volume","SMA20","EMA20","RSI14","ReturnVola20"]
    ds = MultiAssetDataset(df, tickers, features, window=60, horizon=[1,5])

    print("Samples:", len(ds))
    X0, y0 = ds[0]
    print("X0 shape:", X0.shape)   # -> (60, 22, 6)
    print("y0 shape:", y0.shape)   # -> (22, 2)
    print("First 3 assets cumulative returns for horizon [1,5]:\n", y0[:3])