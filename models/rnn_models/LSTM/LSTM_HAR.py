import json
import os

from torch import nn
import torch.nn.functional as F

from .BaseLSTM import BaseLSTM


class LSTM_HAR(BaseLSTM):

    def __init__(self):
        super(LSTM_HAR, self).__init__()

        current_path = os.path.abspath(__file__)
        dir = os.path.dirname(current_path)

        stru_file = open(f"{dir}/structure_LSTM.json", 'r')
        stru_conf = json.load(stru_file)

        self.n_layers = stru_conf['n_layers']
        self.n_hiddens = stru_conf['n_hiddens']
        self.n_classes = stru_conf['n_classes']
        self.n_inputs = stru_conf['n_inputs']

        self.lstm = nn.LSTM(self.n_inputs, self.n_hiddens, self.n_layers, dropout=0.5)
        self.fc = nn.Linear(self.n_hiddens, self.n_classes)

        self.init_weights(self.lstm)
        self.init_weights(self.fc)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, hidden):
        x = x.permute(1, 0, 2)
        x, hidden = self.lstm(x, hidden)
        x = self.dropout(x)

        out = x[-1]
        out = out.contiguous().view(-1, self.n_hiddens)
        out = self.fc(out)
        out = F.softmax(out, dim=1)

        return out

    def begin_state(self, batch_size, device):
        ''' Initialize hidden state'''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.n_hiddens).zero_().to(device),
            weight.new(self.n_layers, batch_size, self.n_hiddens).zero_().to(device))

        return hidden