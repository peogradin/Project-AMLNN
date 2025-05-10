#%%
import torch
import torch.nn as nn

class MultiAssetLSTM(nn.Module):
    """
    An LSTM model that jointly forecasts multiple assets.

    Input:
        x: Tensor of shape (batch_size, window, n_assets, n_features)
    Output:
        y: Tensor of shape(batch_size, n_assets, horizon)
    """

    def __init__(self,
            n_assets: int,
            n_features: int,
            hidden_size: int = 64,
            num_layers: int = 1,
            dropout: float = 0.0,
            horizon: int = 1
        ):
        super().__init__()
        self.n_assets = n_assets
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        if isinstance(horizon, int):
            self.horizon = [horizon]
        else:
            self.horizon = list(horizon)
        self.horizon_length = len(self.horizon)

        self.lstm = nn.LSTM(
            input_size=n_assets*n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.fc = nn.Linear(hidden_size, n_assets * self.horizon_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, W, A*F)
        returns y: (B, A, H)
        """
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        y = self.fc(h)
        return y.view(-1, self.n_assets, self.horizon_length)
    
if __name__ == "__main__":
    # Test MultiAssetLSTM
    print("Testing MultiAssetLSTM.")
    model = MultiAssetLSTM(n_assets=22, n_features=6, hidden_size=64,
                           num_layers=2, dropout=0.1, horizon = [1])
    x = torch.randn(8, 60, 22*6)
    y_hat = model(x)
    print("Output shape: ", y_hat.shape)
    print("Output: ", y_hat[1])
    print("MultiAssetLSTM test complete.")