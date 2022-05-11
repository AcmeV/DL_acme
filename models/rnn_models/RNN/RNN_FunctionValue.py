import torch
from torch import nn

class RNN_FunctionValue(nn.Module):
    def __init__(self, num_hiddens=512):
        super(RNN_FunctionValue, self).__init__()
        self.num_input, self.num_hiddens = 1, num_hiddens
        self.rnn_net = nn.RNN(input_size=self.num_input, num_layers=1, hidden_size=self.num_hiddens, batch_first=True)
        self.num_directions = 1
        self.linear = nn.Linear(self.num_hiddens, self.num_input)

    def forward(self, inputs, state):
        Y, state = self.rnn_net(inputs)
        output = self.linear(state.reshape(-1, self.num_hiddens))
        return output, state

    def begin_state(self, batch_size, device='cpu'):
        return torch.zeros((1, batch_size, self.num_hiddens), device=device)