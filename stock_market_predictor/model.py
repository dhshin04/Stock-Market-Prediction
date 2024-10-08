# TODO: Continue experimenting with different model architectures to reduce MAPE

import torch.nn as nn


class StockPredictor(nn.Module):

    def __init__(self, out_channels=64, hidden_size=64, fc1_out=256, fc2_out=128, size=3, dilation=1, num_layers=4, label_size=20, dropout_rate=0.25):
        super(StockPredictor, self).__init__()
        self.size = size
        self.dilation = dilation
        self.label_size = label_size
        
        ## LSTM Model ##
        self.dropout = nn.Dropout(dropout_rate)
        
        # Replace 6 with out_channels if adding CNN
        self.lstm = nn.LSTM(6, hidden_size, num_layers=num_layers, batch_first=True)
        self.lstm_ln = nn.LayerNorm(hidden_size)        # Layer Norm for LSTM (applied on each timestamp)

        self.output = nn.Linear(hidden_size, label_size)    # Output Layer
             
        ## Additional Layers for future testing (currently not used) ##
        # CNN
        self.conv = nn.Conv1d(
            in_channels=6, out_channels=out_channels, 
            kernel_size=size, stride=1
        )
        self.conv_bn = nn.BatchNorm1d(out_channels)     # Batch Norm for CNN (applied on entire input)

        # FC Layers
        self.fc1 = nn.Linear(hidden_size, fc1_out)
        self.fc_bn1 = nn.BatchNorm1d(fc1_out)           # Batch Norm for FC
        self.fc2 = nn.Linear(fc1_out, fc2_out)
        self.fc_bn2 = nn.BatchNorm1d(fc2_out)           # Batch Norm for FC
        
    def forward(self, x):
        '''
        Arguments:
            x (torch.tensor): Has shape (batch_size, sequence_length, num_features)
        '''
        
        # CNN - current dataset showed similar performance with and without CNN
        # x = x.transpose(1, 2)
        # x = F.pad(x, ((self.dilation * (self.size - 1)), 0))    # Causal padding (only on left side)
        # x = self.conv(x)
        # x = self.conv_bn(x)
        # x = torch.relu(x)
        # x = self.dropout(x)

        # LSTM
        # x = x.transpose(1, 2)
        x, _ = self.lstm(x)     # Integrated tanh/sigmoid activation
        x = self.lstm_ln(x)
        x = self.dropout(x)
        x = x[:, -1, :]     # Only keep last timestamp's output for each sequence in batch

        x = self.output(x)          # (batch_size, label_size * num_features)
        return x
