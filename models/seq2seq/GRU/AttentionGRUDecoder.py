import torch
from torch import nn

from Utils.dl_utils import AdditiveAttention
from models.enc_dec.Decoder import Decoder


class AttentionGRUDecoder(Decoder):

    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0):
        super(AttentionGRUDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)

        self.gru = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)
        self.attention = AdditiveAttention(num_hiddens, num_hiddens, num_hiddens, dropout)

        self.init_weights(self.gru)
        self.init_weights(self.dense)


    def init_state(self, enc_outputs, *args):
        outputs, hidden_state = enc_outputs
        return (outputs.permute(1, 0, 2), hidden_state, args[0])

    def forward(self, X, state):

        enc_outputs, hidden_state, enc_valid_lens = state

        X = self.embedding(X).permute(1, 0, 2)
        outputs, self._attention_weights = [], []
        for x in X:
            query = torch.unsqueeze(hidden_state[-1], dim=1)
            context = self.attention(query, enc_outputs, enc_outputs, enc_valid_lens)
            x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)

            out, hidden_state = self.gru(x.permute(1, 0, 2), hidden_state)
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)

        outputs = self.dense(torch.cat(outputs, dim=0))
        return outputs.permute(1, 0, 2), (enc_outputs, hidden_state, enc_valid_lens)

    @property
    def attention_weights(self):
        return self._attention_weights