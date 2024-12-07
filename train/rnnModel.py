import numpy as np
import torch
import torch.nn as nn
from torch import autograd, Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

from config.variables import dim, max_sentences


# Define RNN Model
class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size, epoch, num_classes,dropout):
        super(RNNClassifier , self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes-1)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.batch_size = batch_size


    def forward(self, x:Tensor):
        out, (hidden_,_ )= self.rnn(x)  # hidden shape: (num_layers, batch_size, hidden_size)
        out = hidden_[-1]  # Take the last layer's hidden state
        out = self.fc(out)  # Output layer

        return out.flatten(start_dim=0)

