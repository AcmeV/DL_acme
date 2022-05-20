from torch import nn

from models import BaseModel


class Encoder(BaseModel):
    """The base encoder interface for the encoder-decoder architecture."""

    def __init__(self):
        super(Encoder, self).__init__()

    def forward(self, X, *args):
        raise NotImplementedError

    def init_weights(self, layer):
        if type(layer) == nn.Linear:
            nn.init.xavier_uniform_(layer.weight)
        if type(layer) == nn.GRU:
            for param in layer._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(layer._parameters[param])