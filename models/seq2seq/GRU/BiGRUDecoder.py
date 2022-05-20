import torch
from torch import nn
from models.enc_dec.Decoder import Decoder


class BiGRUDecoder(Decoder):

    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0):
        super(BiGRUDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)

        self.gru = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers, bidirectional=True, dropout=dropout)
        self.dense = nn.Linear(num_hiddens * 2, vocab_size)

        self.init_weights(self.gru)
        self.init_weights(self.dense)


    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]

    def forward(self, X, state):
        X = self.embedding(X).permute(1, 0, 2)
        context = state[-1].repeat(X.shape[0], 1, 1)
        X_and_context = torch.cat((X, context), 2)
        output, state = self.gru(X_and_context, state)
        output = self.dense(output).permute(1, 0, 2)

        return output, state