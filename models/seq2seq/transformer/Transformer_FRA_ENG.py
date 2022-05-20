from models import BaseModel
from models.seq2seq.transformer.TransformerEncoder import TransformerEncoder
from models.seq2seq.transformer.TransformerDecoder import TransformerDecoder


class Transformer_FRA_ENG(BaseModel):
    """The base class for the encoder-decoder architecture.

    Defined in :numref:`sec_encoder-decoder`"""
    def __init__(self):
        super(Transformer_FRA_ENG, self).__init__()
        # eng_vocab_size = 10012
        # fra_vocab_size = 17851

        eng_vocab_size = 184
        fra_vocab_size = 201

        num_layers = 2
        num_hiddens = 32
        key_size, query_size, value_size = 32, 32, 32
        dropout = 0.1
        norm_shape = [32]
        ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4

        self.encoder = TransformerEncoder(eng_vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout)
        self.decoder = TransformerDecoder(fra_vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout)

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)