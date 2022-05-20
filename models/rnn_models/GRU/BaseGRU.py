import torch
from torch import nn


class BaseGRU(nn.Module):

    def __init__(self):
        super(BaseGRU, self).__init__()
        self.n_hiddens = 32

    def init_weights(self, layer):
        if type(layer) == nn.LSTM or type(layer) == nn.GRU:
            for name, param in layer.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.orthogonal_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
        elif type(layer) == nn.Linear:
            torch.nn.init.orthogonal_(layer.weight)
            layer.bias.data.fill_(0)