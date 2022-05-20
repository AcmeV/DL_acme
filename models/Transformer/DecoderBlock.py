import torch
from torch import nn

from models.Transformer.tranformer_utils import MultiHeadAttention, AddNorm, PositionWiseFFN


class DecoderBlock(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens, norm_shape,
                 ffn_num_input, ffn_num_hiddens, num_heads, dropout, i, **kwargs):
        super(DecoderBlock, self).__init__()
        self.i = i
        self.attention1 = MultiHeadAttention(key_size, query_size, value_size,
                                             num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)

        self.attention2 = MultiHeadAttention(key_size, query_size, value_size,
                                             num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)

        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)

        self.addnorm3 = AddNorm(norm_shape, dropout)

    def forward(self, X, state):

        enc_outputs, enc_valid_lens = state[0], state[1]

        # 训练阶段，输出序列的所有词元都在同一时间处理，
        # 因此state[2][self.i]初始化为None。
        # 预测阶段，输出序列是通过词元一个接着一个解码的，
        # 因此state[2][self.i]包含着直到当前时间步第i个块解码的输出表示
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i], X), dim=1)

        state[2][self.i] = key_values

        if self.training:
            bsz, num_steps, _ = X.shape
            # 只看当前位置， 屏蔽后面的部分（即当前位置的valid_len=i)
            dec_valid_lens = torch.arange(1, num_steps + 1, device=X.device).repeat(bsz, 1)
        else:
            dec_valid_lens = None

        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)

        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)

        return self.addnorm3(Z, self.ffn(Z)), state