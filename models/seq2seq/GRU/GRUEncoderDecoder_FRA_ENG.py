from models import BaseModel
from models.seq2seq.GRU import GRUEncoder, GRUDecoder


class GRUEncoderDecoder_FRA_ENG(BaseModel):
    """The base class for the encoder-decoder architecture.

    Defined in :numref:`sec_encoder-decoder`"""
    def __init__(self):
        super(GRUEncoderDecoder_FRA_ENG, self).__init__()
        # eng_vocab_size = 10012
        # fra_vocab_size = 17851

        eng_vocab_size = 184
        fra_vocab_size = 201

        self.encoder = GRUEncoder(eng_vocab_size, 32, 32, 2, 0.1)
        self.decoder = GRUDecoder(fra_vocab_size, 32, 32, 2, 0.1)

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)