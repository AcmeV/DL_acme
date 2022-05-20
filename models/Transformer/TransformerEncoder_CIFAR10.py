import torch
from torch import nn
from models.enc_dec import Encoder
from models.Transformer.CVEncoderBlock import CVEncoderBlock
from models.Transformer.tranformer_utils import PositionalEncoding, ImageEmbedding


class TransformerEncoder_CIFAR10(Encoder):
    """Transformer encoder.

    Defined in :numref:`sec_transformer`"""
    def __init__(self, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder_CIFAR10, self).__init__()
        self.num_hiddens = num_hiddens
        patch_size = 2
        self.embedding = ImageEmbedding(patch_size, in_channels=3, hidden_size=num_hiddens)
        self.time_step = int((32 / patch_size) ** 2)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.classifer_token = nn.Parameter(torch.zeros(1, 1, num_hiddens))
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(f'block_{i}',
                     CVEncoderBlock(key_size, query_size, value_size, num_hiddens,
                                    norm_shape, ffn_num_input, ffn_num_hiddens,
                                    num_heads, dropout, use_bias))

    def forward(self, X, *args):
        # Since positional encoding values are between -1 and 1, the embedding
        # values are multiplied by the square root of the embedding dimension
        # to rescale before they are summed up
        bsz = X.shape[0]
        cls_tokens = self.classifer_token.expand(bsz, -1, -1)
        X, enc_valid_lens = self.embedding(X)
        X = torch.cat((cls_tokens, X), dim=1)
        for i, blk in enumerate(self.blks):
            X = blk(X, enc_valid_lens)
        return X