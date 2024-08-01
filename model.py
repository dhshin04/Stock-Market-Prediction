import torch
import torch.nn as nn
import torch.nn.functional as F


class StockPredictor(nn.Module):

    def __init__(self, out_channels=64, hidden_size=64, fc1_out=256, fc2_out=128, size=3, dilation=1, num_layers=4, label_size=20, dropout_rate=0.25):
        super(StockPredictor, self).__init__()
        self.size = size
        self.dilation = dilation
        self.label_size = label_size
        
        # Define Neural Network
        self.dropout = nn.Dropout(dropout_rate)
        self.conv = nn.Conv1d(
            in_channels=6, out_channels=out_channels, 
            kernel_size=size, stride=1
        )
        self.conv_bn = nn.BatchNorm1d(out_channels)     # Batch Norm for CNN (applied on entire input)
        
        self.lstm = nn.LSTM(out_channels, hidden_size, num_layers=num_layers, batch_first=True)
        self.lstm_ln = nn.LayerNorm(hidden_size)        # Layer Norm for LSTM (applied on each timestamp)
        
        self.fc1 = nn.Linear(hidden_size, fc1_out)
        self.fc_bn1 = nn.BatchNorm1d(fc1_out)           # Batch Norm for FC
        self.fc2 = nn.Linear(fc1_out, fc2_out)
        self.fc_bn2 = nn.BatchNorm1d(fc2_out)           # Batch Norm for FC
        
        self.output = nn.Linear(fc2_out, label_size)    # Output Layer

    def forward(self, x):
        '''
        Arguments:
            x (torch.tensor): Has shape (batch_size, sequence_length, num_features)
        '''
        
        # CNN
        x = x.transpose(1, 2)
        x = F.pad(x, ((self.dilation * (self.size - 1)), 0))    # Causal padding (only on left side)
        x = self.conv(x)
        x = self.conv_bn(x)
        x = torch.relu(x)
        x = self.dropout(x)

        # LSTM
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)     # Integrated tanh/sigmoid activation
        x = self.lstm_ln(x)
        x = self.dropout(x)
        x = x[:, -1, :]     # Only keep last timestamp's output for each sequence in batch

        # FC
        x = self.fc1(x)
        x = self.fc_bn1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.fc_bn2(x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.output(x)          # (batch_size, label_size * num_features)
        return x
