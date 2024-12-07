import numpy as np
import torch
import torch.nn as nn
from torch import autograd, Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# Define RNN Model
class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size, epoch, num_classes,dropout):
        super(RNNClassifier , self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,dropout=dropout)
        self.fc = nn.Linear(hidden_size, hidden_size//2)
        self.fc2 = nn.Linear(hidden_size//2, num_classes-1)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.batch_size = batch_size


    def forward(self, x:Tensor):
        length_of_sequence = (x != 0).any(dim=2).sum(dim=1)

        x = pack_padded_sequence(x, length_of_sequence, batch_first=True,enforce_sorted=False)

        out, (hidden_,_ )= self.rnn(x)  # hidden shape: (num_layers, batch_size, hidden_size)
        out, length_of_sequence = pad_packed_sequence(out, batch_first=True)
        out = out[:, -1, :]
        out = self.fc(out)  # Output layer
        out = self.fc2(out)

        return out.flatten(start_dim=0)

