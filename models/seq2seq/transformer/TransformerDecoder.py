import math

from torch import nn

from models.enc_dec import Decoder
from models.Transformer.DecoderBlock import DecoderBlock
from models.Transformer.tranformer_utils import PositionalEncoding


class TransformerDecoder(Decoder):
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hidden,
                 num_heads,  num_layers, dropout=0.1, **kwargs):
        super(TransformerDecoder, self).__init__()
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()

        for i in range(num_layers):
            self.blks.add_module(f'block_{i}', DecoderBlock(
                key_size, query_size, value_size, num_hiddens,
                norm_shape, ffn_num_input, ffn_num_hidden,
                num_heads, dropout, i))

        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, *args):
        return [enc_outputs, args[0], [None] * self.num_layers]

    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range(2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            # 解码器自注意力权重
            self._attention_weights[0][i] = blk.attention1.attention.attention_weights
            # 编码器－解码器自注意力权重
            self._attention_weights[1][i] = blk.attention2.attention.attention_weights

        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights
