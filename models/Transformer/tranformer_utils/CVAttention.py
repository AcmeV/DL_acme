import math

import torch
from torch import nn
from torch.nn import Linear, Dropout, Softmax

class CVAttention(nn.Module):
    def __init__(self, num_hiddens, num_heads, dropout, **kwargs):
        super(CVAttention,self).__init__()
        self.num_attention_heads = num_heads
        self.attention_head_size = int(num_hiddens / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(num_hiddens, self.all_head_size)#wm,768->768，Wq矩阵为（768,768）
        self.key = Linear(num_hiddens, self.all_head_size)#wm,768->768,Wk矩阵为（768,768）
        self.value = Linear(num_hiddens, self.all_head_size)#wm,768->768,Wv矩阵为（768,768）
        self.out = Linear(num_hiddens, num_hiddens)  # wm,768->768
        self.attn_dropout = Dropout(dropout)
        self.proj_dropout = Dropout(dropout)

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
        self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        # hidden_states为：(bs,197,768)
        mixed_query_layer = self.query(hidden_states)#wm,768->768
        mixed_key_layer = self.key(hidden_states)#wm,768->768
        mixed_value_layer = self.value(hidden_states)#wm,768->768

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output