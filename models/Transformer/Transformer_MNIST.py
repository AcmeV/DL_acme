from torch import nn

from models import BaseModel
import torch.nn.functional as f
from models.Transformer import TransformerEncoder_MNIST


class Transformer_MNIST(BaseModel):
    """The base class for the encoder-decoder architecture.

    Defined in :numref:`sec_encoder-decoder`"""
    def __init__(self, pool_mode='mean'):
        super(Transformer_MNIST, self).__init__()

        num_layers = 2
        num_hiddens = 768
        key_size = query_size = value_size = num_hiddens
        dropout = 0.5
        norm_shape = [num_hiddens]
        ffn_num_input, ffn_num_hiddens, num_heads = num_hiddens, num_hiddens, 4
        self.pool_mode = pool_mode
        self.num_classes = 10

        self.to_latent = nn.Identity()

        self.encoder = TransformerEncoder_MNIST(key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout)
        self.mlp_head = nn.Linear(num_hiddens, self.num_classes)

    def forward(self, X):
        enc_outputs = self.encoder(X)
        if self.pool_mode == 'mean':
            out = enc_outputs.mean(dim=1)
        else:
            out = enc_outputs[:, 0]
        out = self.mlp_head(self.to_latent(out))
        return f.log_softmax(out, dim=1)