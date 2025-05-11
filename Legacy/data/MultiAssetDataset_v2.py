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

class MultiAssetSequencedDataset(Dataset):
    def __init__(self, df, tickers, features, target_col, target_is_cumulative = True, window=60, window_step_length = 1, horizon=1):
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
        self.target_is_cumulative = target_is_cumulative
        self.window = window
        self.horizon = list(horizon) if isinstance(horizon, list) else [horizon]
        
        dates = df.index.unique().sort_values()

        T = len(dates)
        A = len(tickers)
        F = len(features)
        H = len(self.horizon)
        max_h = max(self.horizon)

        self.feature_means = {}
        self.feature_stds = {}

        arr = np.zeros((T, A, F), dtype=float)
        targets_arr = np.zeros((T, A), dtype=float)

        for i, t in enumerate(tickers):
            sub = df[df['Ticker'] == t].reindex(dates)
            for j, f in enumerate(features):
                mean = sub[f].mean()
                std = sub[f].std()
                self.feature_means[(t, f)] = mean
                self.feature_stds[(t, f)] = std
                arr[:, i, j] = ((sub[f] - mean) / std).values
            targets_arr[:, i] = sub[target_col].values
        
        Xs = []
        ys = []
        n_samples = T - window - max_h + 1

        for i in range(0, n_samples, window_step_length):
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

        # Normalize the target after potential accumulation
        ys_np = np.stack(ys)  # shape (N, A, H)
        self.y_mean = ys_np.mean(axis=0)  # shape (A, H)
        self.y_std = ys_np.std(axis=0)    # shape (A, H)
        ys_norm = (ys_np - self.y_mean) / self.y_std

        self.X = torch.stack([torch.tensor(x.reshape(window, -1), dtype=torch.float32) for x in Xs]) # (N, W, A*F)
        self.y = self.y = torch.tensor(ys_norm.reshape(len(ys_norm), -1, H), dtype=torch.float32) # (N, A*H)

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        y = self.y[idx]
        if y.shape[-1] == 1:
            y = y.squeeze(-1)  # If H=1 → (A,)
        return self.X[idx], y
    
    def inverse_transform(self, y_pred):
        """
        Inverse transform the normalized target values.
        y: Tensor of shape (B, A, H) or (B, A)
        """
        mean = torch.tensor(self.y_mean, dtype=y_pred.dtype, device=y_pred.device)
        std = torch.tensor(self.y_std, dtype=y_pred.dtype, device=y_pred.device)
        return y_pred * std + mean
    
