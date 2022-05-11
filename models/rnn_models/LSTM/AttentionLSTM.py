import json
import os

import torch
from torch import nn
import torch.nn.functional as F

from .BaseLSTM import BaseLSTM


class AttentionLSTM_HAR(BaseLSTM):

    def __init__(self):
        super(AttentionLSTM_HAR, self).__init__()

        current_path = os.path.abspath(__file__)
        dir = os.path.dirname(current_path)
        # load model structure params
        stru_file = open(f"{dir}/structure_AttentionLSTM.json", 'r')
        stru_conf = json.load(stru_file)

        self.n_layers = stru_conf['n_layers']
        self.n_hiddens = stru_conf['n_hiddens']
        self.n_classes = stru_conf['n_classes']
        self.n_inputs = stru_conf['n_inputs']
        self.sequence_size = 128
        self.attention_size = stru_conf['attention_size']

        self.lstm = nn.LSTM(self.n_inputs, self.n_hiddens, self.n_layers, dropout=0.5)
        self.fc = nn.Linear(self.n_hiddens, self.n_classes)

        self.init_weights(self.lstm)
        self.init_weights(self.fc)

        self.w_omega = nn.Parameter(torch.zeros(self.n_hiddens, self.attention_size), requires_grad=True)
        self.u_omega = nn.Parameter(torch.zeros(self.attention_size, 1), requires_grad=True)

        self.dropout = nn.Dropout(0.5)

    def attention_layer(self, lstm_out):
        # reshape out as [(batch_size * sequence_size) x hidden_size]
        output_reshape = torch.Tensor.reshape(lstm_out, [-1, self.n_hiddens])

        attn_tanh = torch.tanh(torch.mm(output_reshape, self.w_omega))
        attn_hidden_layer = torch.mm(attn_tanh, self.u_omega)

        alphas = F.softmax(torch.reshape(attn_hidden_layer, [-1, self.sequence_size]), dim=1)

        alphas_reshape = torch.Tensor.reshape(alphas, [-1, self.sequence_size, 1])
        state = lstm_out.permute(1, 0, 2)
        attn_output = torch.sum(state * alphas_reshape, 1)

        return attn_output

    def forward(self, x, hidden):
        x = x.permute(1, 0, 2)
        x, hidden = self.lstm(x, hidden)
        x = self.dropout(x)
        out = self.attention_layer(x)
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

