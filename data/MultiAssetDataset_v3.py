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
    def __init__(self, df, tickers, features, target_col, target_is_cumulative = True, window=60, window_step_length = 1, horizon=1, num_permutations=1,
                 feature_means=None, feature_stds=None, y_mean=None, y_std=None):
        """
        df: DataFrame with columns ['Ticker', features...] and
        tickers: List of ticker-strings
        features: List of feature column names.
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
        self.window_step_length = window_step_length
        self.num_permutations = num_permutations

        self.X_means = feature_means # if feature_means is not None else {}
        self.X_stds = feature_stds # if feature_stds is not None else {}
        self.y_mean = y_mean
        self.y_std = y_std

        self.sequence_dates = []
        self.target_dates = []
        
    def create(self):
        dates = self.df.index.unique().sort_values()

        T = len(dates)
        A = len(self.tickers)
        F = len(self.features)
        H = len(self.horizon)
        max_h = max(self.horizon)

        arr = np.zeros((T, A, F), dtype=float)
        targets_arr = np.zeros((T, A), dtype=float)
        
        for i, t in enumerate(self.tickers):
            sub = self.df[self.df['Ticker'] == t].reindex(dates)
            # for j, f in enumerate(self.features):
            #     mean = sub[f].mean() if (t, f) not in self.X_means else self.X_means[(t, f)]
            #     std = sub[f].std() if (t, f) not in self.X_stds else self.X_stds[(t, f)]
            #     self.X_means[(t, f)] = mean
            #     self.X_stds[(t, f)] = std
            #     arr[:, i, j] = ((sub[f] - mean) / std).values
            arr[:, i, :] = sub[self.features].values
            targets_arr[:, i] = sub[self.target_col].values
        
        Xs = []
        ys = []
        n_samples = T - self.window - max_h + 1
        
        for i in range(0, n_samples, self.window_step_length):
            Xs.append(arr[i : i + self.window])
            self.sequence_dates.append(dates[i + self.window -1])
            self.target_dates.append(dates[i + self.window + max_h - 1])

            if self.target_is_cumulative:      
                targets = []
                for h in self.horizon:
                    slice_rets = targets_arr[i + self.window : i + self.window + h]
                    cumulative = np.prod(1 + slice_rets, axis=0) - 1
                    targets.append(cumulative)
            else:
                targets = []
                for h in self.horizon:
                    one_step = targets_arr[i + self.window + h - 1]  # exakt h steg fram
                    targets.append(one_step)
            
            y_i = np.stack(targets, axis = -1)
            ys.append(y_i)

        # Normalize the features
        Xs_np = np.stack(Xs)  # shape (N, W, A, F)
        if self.X_means is None: self.X_means = Xs_np.mean(axis=1)
        if self.X_stds is None: self.X_stds = Xs_np.std(axis=1)
        Xs_norm = (Xs_np - self.X_means[:, None, :, :]) / self.X_stds[:, None, :, :]

        # Normalize the target with the feature means and stds from target column
        ys_np = np.stack(ys)  # shape (N, A, H)
        target_col_idx = self.features.index(self.target_col)
        X_target_feature = Xs_np[:, :, :, target_col_idx] # (N, W, A)
        if self.y_mean is None: self.y_mean = X_target_feature.mean(axis=1).reshape(-1, A, 1)
        if self.y_std is None: self.y_std = X_target_feature.std(axis=1).reshape(-1, A, 1)
        ys_norm = (ys_np - self.y_mean) / self.y_std

        # Permute the ticker placements
        if self.num_permutations > 1:
            Xs_norm = np.array(Xs_norm)
            permuted_Xs = []
            permuted_ys = []
            for i in range(self.num_permutations - 1):
                permuted_indices = np.random.permutation(A)
                permuted_Xs.append(Xs_norm[:, :, permuted_indices])
                permuted_ys.append(ys_norm[:, permuted_indices])
            Xs_norm = np.concatenate([Xs_norm] + permuted_Xs, axis=0)
            ys_norm = np.concatenate([ys_norm] + permuted_ys, axis=0)

        # Extend the data with permutation of ticker placements (if num_permutations > 1)
        self.X = torch.stack([torch.tensor(x.reshape(self.window, -1), dtype=torch.float32) for x in Xs_norm]) # (N, W, A*F)
        self.y = torch.tensor(ys_norm.reshape(len(ys_norm), -1), dtype=torch.float32) # (N, A*H)

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        y = self.y[idx]
        # if len(self.horizon) >= 1:
        #     y = y.squeeze(-1)  # If H=1 → (A,)
        return self.X[idx], y
    
    def get_sequence_dates(self):
        """
        Returns the dates of the sequences in the dataset.
        """
        if self.num_permutations > 1:
            raise ValueError("Sequence dates are not supported for permuted datasets.")
        return self.sequence_dates
    
    def get_target_dates(self):
        """
        Returns the dates of the targets in the dataset.
        """
        if self.num_permutations > 1:
            raise ValueError("Target dates are not supported for permuted datasets.")
        return self.target_dates
    
    def inverse_transform(self, y_pred, idx, batch=True):
        """
        Inverse transform the normalized target values.
        y: Tensor of shape (B, A*H) or (B, A)
        """
        # if not batch:
        #     y_pred = y_pred.unsqueeze(0)
        # mean = torch.tensor(self.y_mean, dtype=y_pred.dtype, device=y_pred.device)
        # std = torch.tensor(self.y_std, dtype=y_pred.dtype, device=y_pred.device)
        # return y_pred * std + mean if batch else y_pred.squeeze(-1) * std + mean
        if self.num_permutations > 1:
            raise ValueError("Inverse transform is not supported for permuted datasets.")
        if batch:
            assert y_pred.shape[0] == 1, "Only batch size 1 supported in inverse_transform"
            y_pred = y_pred.squeeze(0)
        
        A = len(self.tickers)
        H = len(self.horizon)
        assert y_pred.shape[0] == A * H, f"Expected shape ({A * H},), got {y_pred.shape}"
        y_reshaped = y_pred.view(A, H)
        
        mean = torch.tensor(self.y_mean[idx], dtype=y_pred.dtype, device=y_pred.device) # (A, 1)
        std = torch.tensor(self.y_std[idx], dtype=y_pred.dtype, device=y_pred.device) # (A, 1)
        
        y_inv = y_reshaped * std + mean
        y_inv = y_inv.view(1, A * H) if batch else y_inv.view(A * H) # (B, A*H) or (A*H)
        return y_inv
        

    
    def inverse_feature_transform(self, X, idx, batch=True):
        """
        Inverse transform the normalized feature values.
        X: Tensor of shape (B, T, A*F) or (T, A*F)
        """
        if self.num_permutations > 1:
            raise ValueError("Inverse feature transform is not supported for permuted datasets.")
        if batch:
            assert X.shape[0] == 1, "Only batch size 1 supported in inverse_feature_transform"
            X = X.squeeze(0)  # (T, A*F)

        T, AF = X.shape
        A = len(self.tickers)
        F = len(self.features)
        assert AF == A * F, f"Feature dimension mismatch. Expected {A * F} features, got {AF}"
        X_reshaped = X.view(T, A, F)
        X_inv = torch.zeros_like(X_reshaped)

        mean = torch.tensor(self.X_means[idx], dtype=X.dtype, device=X.device) # (A, F)
        std = torch.tensor(self.X_stds[idx], dtype=X.dtype, device=X.device) # (A, F)
        X_inv = X_reshaped * std + mean  # (T, A, F)

        return X_inv.view(1, T, A * F) if batch else X_inv.view(T, A * F)
    

##############################################################################################################
### TESTING THE DATASET BELOW
##############################################################################################################
#%%
def verify_inverse_feature_transform(ds: MultiAssetSequencedDataset, df_val: pd.DataFrame, idx: int = 0):
    import matplotlib.pyplot as plt

    # === Hämta inverterad X-sekvens från datasetet ===
    X_norm, y_norm = ds[idx]
    X_inv = ds.inverse_feature_transform(X_norm, idx, batch=False)  # shape (W, A*F)
    y_inv = ds.inverse_transform(y_norm, idx, batch=False)  # shape (A*H)
    W = ds.window
    A = len(ds.tickers)
    F = len(ds.features)
    H = len(ds.horizon)
    X_inv_np = X_inv.view(W, A, F).cpu().numpy()
    y_inv_np = y_inv.view(A, H).cpu().numpy()

    # === Återskapa originaldata utan pivot ===
    dates = df_val.index.unique().sort_values()
    start_date = dates[idx]
    end_date = dates[idx + W - 1]
    date_window = dates[idx : idx + W]

    # Bygg en (W, A, F)-tensor direkt från df_val
    X_true = np.zeros((W, A, F))
    for a, ticker in enumerate(ds.tickers):
        df_ticker = df_val[df_val["Ticker"] == ticker].loc[date_window]
        for f, feature in enumerate(ds.features):
            X_true[:, a, f] = df_ticker[feature].values

    # === Plot ===
    ticker_idx = 0
    feature_idx = 0
    plt.plot(X_true[:, ticker_idx, feature_idx], label="Original")
    plt.plot(X_inv_np[:, ticker_idx, feature_idx] + 10, label="Inverse transformed")
    plt.title(f"{ds.tickers[ticker_idx]} | {ds.features[feature_idx]}")
    plt.legend()
    plt.show()
    
    # === Plot y ===
    date_window_y = dates[idx + W : idx + W + max(ds.horizon)]
    y_true = np.zeros((A, H))
    for a, ticker in enumerate(ds.tickers):
        df_ticker = df_val[df_val["Ticker"] == ticker].loc[date_window_y]
        for h in range(H):
            y_true[a, h] = df_ticker[ds.target_col].values[h]  # sista värdet i sekvensen
    print("y shape: ", y_inv_np.shape)
    print(y_inv)
    plt.plot(y_inv_np[ticker_idx], label="Inverse transformed")
    plt.plot(y_true[ticker_idx], label="Original")
    plt.title(f"{ds.tickers[ticker_idx]} | {ds.target_col}")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    import pandas as pd
    #from data.MultiAssetDataset_v2 import MultiAssetSequencedDataset
    from torch.utils.data import DataLoader

    df = pd.read_csv("../data/OMXS22_model_features_raw.csv", index_col = "Date", parse_dates = True)

    tickers = df["Ticker"].unique().tolist()
    #tickers = ["KINV-B.ST"] #, "SAND.ST"]
    features = ["Close" , "Return" ,"Volume","SMA20","EMA20","RSI14"]
    target_col = "Close"
    window   = 100
    window_step_length = 6
    horizon  = [i for i in range(1, 20)] # [1, 2, 3, 4, 5]


    train_stop = pd.Timestamp("2018-01-01")
    val_stop   = pd.Timestamp("2023-12-31")

    df_train = df[df.index < train_stop].copy()

    df_val   = df[(df.index >= train_stop) & (df.index < val_stop)].copy()
    df_test  = df[df.index >= val_stop].copy()


    ds_train = MultiAssetSequencedDataset(df_train, tickers, features, target_col, False, window, window_step_length, horizon=horizon, num_permutations=10)
    ds_train.create()
    ds_val   = MultiAssetSequencedDataset(df_val, tickers, features, target_col, False, window, window_step_length, horizon=horizon, num_permutations=1)
    ds_val.create()
    Xb, yb = ds_val[0]
    print("X shape: ", Xb.shape, "y shape: ", yb.shape)
    Xb_inv = ds_val.inverse_feature_transform(Xb, 0, batch=False)
    yb_inv = ds_val.inverse_transform(yb, 0, batch=False)
    print("X_inv shape: ", Xb_inv.shape, "y_inv shape: ", yb_inv.shape)
    print(df_val.index)
    verify_inverse_feature_transform(ds_val, df_val, idx=0)
    
# %%
