import torch
import torch.nn as nn

class SimpleLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_hidden_layers, output_size, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_hidden_layers, 
            dropout=dropout,
            batch_first=True)
        
        self.fc_out = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, x):
        lstm_out, hidden_state = self.lstm(x)

        last_hidden = lstm_out[:, -1, :]
        output = self.fc_out(last_hidden)

        return output

