import torch
import torch.nn as nn
import torch.nn.functional as F

class resBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_conv1=False, strides=1, dropout=0.4):
        super().__init__()
        
        self.process = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=strides, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels)
        )
        
        if use_conv1:
            self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=strides)
        else:
            self.conv1 = None
        
    def forward(self, x):
        left = self.process(x)
        right = x if self.conv1 is None else self.conv1(x)
        
        return F.relu(left + right)

class cnnModule(nn.Module):
    def __init__(self, in_channel, out_channel, hidden_channel=128, dropout=0.4):
        super().__init__()
        
        self.head = nn.Sequential(
            nn.Conv1d(in_channel, hidden_channel, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(hidden_channel),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.MaxPool1d(2)
        )
        
        self.cnn = nn.Sequential(
            resBlock(hidden_channel, out_channel, use_conv1=True, strides=1),
            resBlock(out_channel, out_channel, strides=1),
            resBlock(out_channel, out_channel, strides=1)
        )
    
    def forward(self, x):
        x = self.head(x)
        x = self.cnn(x)
        
        return x

class DeepLPI(nn.Module):
    def __init__(self, molshape, seqshape, dropout=0.4):
        super().__init__()
        
        self.molshape = molshape
        self.seqshape = seqshape

        self.molcnn = cnnModule(1, 64)  # Adjusted out_channel
        self.seqcnn = cnnModule(1, 64)  # Adjusted out_channel
        
        self.pool = nn.AvgPool1d(5, stride=3)
        self.lstm = nn.LSTM(64, 64, num_layers=3, batch_first=True, bidirectional=True)  # Adjusted hidden size and num_layers
        
        self.mlp = nn.Sequential(
            nn.Linear(round(((molshape + seqshape) / 4 - 2) * 2 / 3) * 64, 4096),  # Adjusted hidden units
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            
            nn.Linear(4096, 2048),  # Adjusted hidden units
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            
            nn.Linear(2048, 512),  # Adjusted hidden units
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            
            nn.Linear(512, 1),
        )

    def forward(self, mol, seq):
        mol = self.molcnn(mol.reshape(-1, 1, self.molshape))
        seq = self.seqcnn(seq.reshape(-1, 1, self.seqshape))
        
        # Concatenate along the sequence dimension
        x = torch.cat((mol, seq), 2)
        x = self.pool(x)
        
        # Reshape for LSTM
        batch_size = x.size(0)
        x = x.reshape(batch_size, -1, 64)
        x, _ = self.lstm(x)
        
        # Fully connected layer
        x = self.mlp(x.flatten(1))
        
        x = x.flatten()
        
        return x
