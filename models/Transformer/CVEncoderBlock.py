from torch import nn

from models.Transformer.tranformer_utils import AddNorm, PositionWiseFFN, CVAttention


class CVEncoderBlock(nn.Module):
    """Transformer encoder block.

    Defined in :numref:`sec_transformer`"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, use_bias=False, **kwargs):
        super(CVEncoderBlock, self).__init__()
        self.attention = CVAttention(num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(
            ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, *args):
        Y = self.addnorm1(X, self.attention(X))
        return self.addnorm2(Y, self.ffn(Y))