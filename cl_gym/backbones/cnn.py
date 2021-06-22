import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN1D(nn.Module):
    """
    A small 1-D CNN used for time series tasks.
    """
    def __init__(self, dropout_prob=0.0):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.conv1 = nn.Conv1d(18, 64, 5)
        self.conv2 = nn.Conv1d(64, 64, 5)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.pool2 = nn.MaxPool1d(4)
        self.relu = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(64*62, 200)
        self.fc4 = nn.Linear(200, 12)

    def forward(self, x, task_id=None):
        out = x
        batch_size = out.shape[0]
        out = torch.transpose(out, 1, 2)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        if self.dropout_prob:
            out = self.dropout(out)
        out = self.pool2(out)
        out = out.view(batch_size, -1)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        return out