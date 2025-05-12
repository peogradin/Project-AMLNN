#%%
import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0, N_hidden=1, mode="optimize"):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        self.N_hidden = N_hidden
        self.mode = mode

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        
        if N_hidden > 1:
            hidden_layers = []
            for i in range(N_hidden - 1):
                hidden_layers.extend([
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, hidden_size),
                    nn.SELU()
                ])
            hidden_layers.extend([
                nn.Dropout(dropout),
                nn.Linear(hidden_size, output_size),
            ])
            self.fc = nn.Sequential(*hidden_layers)
        else:
            self.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size, output_size),
            )

    def forward(self, x):
        B = x.size(0)

        h_0 = torch.zeros(self.num_layers, B, self.hidden_size, device=x.device)
        c_0 = torch.zeros(self.num_layers, B, self.hidden_size, device=x.device)

        # Forward propagate LSTM
        out, (h_n, c_n) = self.lstm(x, (h_0.detach(), c_0.detach()))
        last_hidden = out[:, -1, :]

        logits = self.fc(last_hidden)

        if self.mode == "optimize":
            return F.softmax(logits, dim=-1)
        elif self.mode == "predict":
            return logits
        else:
            raise ValueError("Invalid mode. Choose 'optimize' or 'predict'.")