import torch
from torch import nn
from torch.nn import functional as F

class RNN_TimeMachine(nn.Module):
    def __init__(self):
        super(RNN_TimeMachine, self).__init__()
        self.num_input, self.num_hiddens = 28, 512
        self.rnn_net = nn.RNN(input_size=self.num_input, num_layers=1, hidden_size=self.num_hiddens)
        self.num_directions = 1
        self.linear = nn.Linear(self.num_hiddens, self.num_input)

    def forward(self, inputs, state):
        inputs = F.one_hot(inputs.T, self.num_input).type(torch.float32)
        Y, state = self.rnn_net(inputs, state)
        output = self.linear(Y.reshape(-1, Y.shape[-1]))
        return output, state

    def begin_state(self, batch_size, device='cpu'):
        return torch.zeros((1, batch_size, self.num_hiddens), device=device)