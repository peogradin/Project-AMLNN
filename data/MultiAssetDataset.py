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
    def __init__(self, df, tickers, features, target_col, target_is_cumulative = True, window=60, horizon=1):
        """
        df: DataFrame with columns ['Ticker', features...]
        tickers: List of ticker-strings
        features: List of feature column names
        target_col: String with the column name of the target (e.g. Close, Return, Log_return)
        target_is_cumulative: Bool, if True the target is cumulative (for returns, log-returns set to True)
        window: Number of time steps in the input sequence
        horizon: Int or list of ints for the prediction horizon
                ex horizon=1 means predicting the next time step
                ex horizon=[1, 2] means predicting the next 1 and 2 time steps
        """
        self.df = df
        self.tickers = tickers
        self.features = features
        self.target_col = target_col
        self.window = window
        self.horizon = list(horizon) if isinstance(horizon, list) else [horizon]
        
        dates = df.index.unique().sort_values()

        T = len(dates)
        A = len(tickers)
        F = len(features)
        H = len(self.horizon)
        max_h = max(self.horizon)

        arr = np.zeros((T, A, F), dtype=float)
        targets_arr = np.zeros((T, A), dtype=float)
        for i, t in enumerate(tickers):
            sub = df[df['Ticker'] == t].reindex(dates)
            arr[:, i, :] = sub[features].values
            targets_arr[:, i] = sub[target_col].values
        
        Xs = []
        ys = []
        n_samples = T - window - max_h + 1

        for i in range(n_samples):
            Xs.append(arr[i : i + window])

            if target_is_cumulative:      
                targets = []
                for h in self.horizon:
                    slice_rets = targets_arr[i + window : i + window + h]
                    cumulative = np.prod(1 + slice_rets, axis=0) - 1
                    targets.append(cumulative)
            else:
                targets = []
                for h in self.horizon:
                    one_step = targets_arr[i + window + h - 1]  # exakt h steg fram
                    targets.append(one_step)
            
            y_i = np.stack(targets, axis = -1)
            ys.append(y_i)

        self.X = torch.stack([torch.tensor(x.reshape(window, -1), dtype=torch.float32) for x in Xs]) # (N, W, A*F)
        self.y = torch.stack([torch.tensor(y.reshape(-1, H), dtype=torch.float32) for y in ys]) # (N, A*H)

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx].squeeze(-1)
    
if __name__ == "__main__":
    df = pd.read_csv("OMXS22_model_features_raw.csv", index_col="Date", parse_dates=True)
    tickers = df["Ticker"].unique().tolist()
    features = ["Close","Volume","SMA20","EMA20","RSI14","ReturnVola20"]
    ds = MultiAssetDataset(df, tickers, features, "Close", True, window=60, horizon=1)
    loader = DataLoader(ds, batch_size=32, shuffle=True)
    for batch in loader:
        X, y = batch
        print("X shape:", X.shape)   # -> (32, 60, 22*6)
        print("y shape:", y.shape)   # -> (32, 22, 1)
        break

    print("Samples:", len(ds))
    X0, y0 = ds[0]
    print("X0 shape:", X0.shape)   # -> (60, 22*66)
    print("y0 shape:", y0.shape)   # -> (22, 2)
    print("First 3 assets cumulative returns for horizon 1:\n", y0[:3])
# %%
