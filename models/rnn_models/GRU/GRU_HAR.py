import json
import os

import torch
from torch import nn
import torch.nn.functional as F

from .BaseGRU import BaseGRU


class GRU_HAR(BaseGRU):

    def __init__(self):
        super(GRU_HAR, self).__init__()

        current_path = os.path.abspath(__file__)
        dir = os.path.dirname(current_path)

        stru_file = open(f"{dir}/structure_GRU.json", 'r')
        stru_conf = json.load(stru_file)

        self.n_layers = stru_conf['n_layers']
        self.n_hiddens = stru_conf['n_hiddens']
        self.n_classes = stru_conf['n_classes']
        self.n_inputs = stru_conf['n_inputs']

        self.gru = nn.GRU(self.n_inputs, self.n_hiddens, self.n_layers, dropout=0.5)

        self.fc = nn.Linear(self.n_hiddens, self.n_classes)

        self.init_weights(self.gru)
        self.init_weights(self.fc)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, hidden):
        x = x.permute(1, 0, 2)
        x, hidden = self.gru(x, hidden)
        x = self.dropout(x)

        out = x[-1]
        out = out.contiguous().view(-1, self.n_hiddens)
        out = self.fc(out)
        out = F.softmax(out, dim=1)

        return out

    def begin_state(self, batch_size, device):
        hidden = torch.zeros(self.n_layers, batch_size, self.n_hiddens).to(device)
        return hidden