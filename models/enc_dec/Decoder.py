from torch import nn

from models import BaseModel


class Decoder(BaseModel):
    """The base decoder interface for the encoder-decoder architecture.

    Defined in :numref:`sec_encoder-decoder`"""
    def __init__(self):
        super(Decoder, self).__init__()

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError

    def init_weights(self, layer):
        if type(layer) == nn.Linear:
            nn.init.xavier_uniform_(layer.weight)
        if type(layer) == nn.GRU:
            for param in layer._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(layer._parameters[param])