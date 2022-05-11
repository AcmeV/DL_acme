import json
import os

import torch
from torch import nn
import torch.nn.functional as F

from .BaseRNN import BaseRNN


class ResidualRNN_HAR(BaseRNN):

    def __init__(self):
        super(ResidualRNN_HAR, self).__init__()

        current_path = os.path.abspath(__file__)
        dir = os.path.dirname(current_path)

        stru_file = open(f"{dir}/structure_ResidualRNN.json", 'r')
        stru_conf = json.load(stru_file)

        self.n_layers = stru_conf['n_layers']
        self.n_hiddens = stru_conf['n_hiddens']
        self.n_classes = stru_conf['n_classes']
        self.n_inputs = stru_conf['n_inputs']
        self.n_residuals = stru_conf['n_residuals']
        self.n_residual_layers = stru_conf['n_residual_layers']

        self.rnn1 = nn.RNN(self.n_inputs, self.n_hiddens, self.n_layers, dropout=0.5)
        self.rnn2 = nn.RNN(self.n_hiddens, self.n_hiddens, self.n_layers, dropout=0.5)
        self.fc = nn.Linear(self.n_hiddens, self.n_classes)

        self.init_weights(self.rnn1)
        self.init_weights(self.rnn2)
        self.init_weights(self.fc)
        self.dropout = nn.Dropout(0.5)

    def add_residual_layers(self, x, hidden):
        residual = F.relu(x, inplace=False)
        for i in range(self.n_residual_layers):
            residual, hidden = self.rnn2(residual, hidden)
            residual = F.relu(residual, inplace=False)
        x = x + residual
        return x

    def forward(self, x, hidden):
        x = x.permute(1, 0, 2)
        x, hidden = self.rnn1(x, hidden)
        for i in range(self.n_residuals):
            x = self.add_residual_layers(x, hidden)
        x = self.dropout(x)

        out = x[-1]
        out = out.contiguous().view(-1, self.n_hiddens)
        out = self.fc(out)
        out = F.softmax(out, dim=1)

        return out

    def begin_state(self, batch_size, device):
        ''' Initialize hidden state'''
        hidden = torch.zeros((self.n_layers, batch_size, self.n_hiddens), device=device)
        return hidden
