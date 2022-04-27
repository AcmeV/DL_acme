import torch
from torch.nn import functional as F

class MyRNN_TimeMachine:
    def __init__(self):
        self.num_input, self.num_hiddens  = 28, 512
        self.params = self.init_params(self.num_input, self.num_hiddens)

    def __call__(self, inputs, state):
        inputs = F.one_hot(inputs.T, self.num_input).type(torch.float32)
        W_xh, W_hh, b_h, W_ho, b_o = self.params

        H,  = state

        outputs = []
        for X in inputs:
            H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)

            Y = torch.mm(H, W_ho) + b_o

            outputs.append(Y)
        return torch.cat(outputs, dim=0), (H, )

    def parameters(self):
        return self.params

    def init_params(self, num_input, num_hiddens):
        num_output = num_input

        def normal(shape):
            return torch.randn(size=shape) * 0.01

        W_xh = normal((num_input, num_hiddens))
        W_hh = normal((num_hiddens, num_hiddens))
        b_h = torch.zeros(num_hiddens)
        W_ho = normal((num_hiddens, num_output))
        b_o = torch.zeros(num_output)
        params = [W_xh, W_hh, b_h, W_ho, b_o]
        for param in params:
            param.requires_grad_(True)
        return params

    def begin_state(self, batch_size, device='cpu'):
        return (torch.zeros((batch_size, self.num_hiddens), device=device), )

    def to(self, device):
        for p in self.params:
            p.to(device)
