from torch import nn

class PositionWiseFFN(nn.Module):
    """Positionwise feed-forward network.

    Defined in :numref:`sec_transformer`"""
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs,
                 **kwargs):
        super(PositionWiseFFN, self).__init__()

        self.ffn = nn.Sequential(
            nn.Linear(ffn_num_input, ffn_num_hiddens),
            nn.GELU(),
            nn.Dropout(),
            nn.Linear(ffn_num_hiddens, ffn_num_outputs),
            nn.Dropout()
        )

        # self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        # self.relu = nn.ReLU()
        # self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        # return self.dense2(self.relu(self.dense1(X)))
        return self.ffn(X)