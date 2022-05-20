from torch import nn
from models.enc_dec.Encoder import Encoder


class GRUEncoder(Encoder):

    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0):
        super(GRUEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout)

        self.init_weights(self.gru)

    def forward(self, X, *args):
        X = self.embedding(X)
        X = X.permute(1, 0, 2)
        output, state = self.gru(X)

        return output, state