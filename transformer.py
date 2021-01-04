import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super(SelfAttention, self).__init__()
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x:[b,s,d]
        q, k, v = [w(x) for w in [self.wq, self.wk, self.wv]]
        attn = torch.softmax(torch.matmul(q, k.transpose(-1, -2)), dim=-1)  # [b,s,s]
        output = torch.matmul(attn, v)
        return output


class MultiheadAttention(nn.Module):
    '''
    attention througth the sequence and embedding axis
    '''

    def __init__(self, d_model, n_heads):
        super(MultiheadAttention, self).__init__()
        assert d_model % n_heads == 0
        self.w = nn.Linear(d_model * 3, d_model)
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

    def forward(self, x, mask):
        # x:[b,s,d]
        b, s, d = x.size()
        x_head = self.w(x).reshape(b, s, 3, d)  # [b,s,3,d]
        q, k, v = x_head[:, :, 0, :], x_head[:, :, 1, :], x_head[:, :, 2, :]  # [b,s,d]

        q = q.reshape(b, s, self.n_heads, -1).transpose(1, 2)
        k = k.reshape(b, s, self.n_heads, -1).transpose(1, 2)
        v = v.reshape(b, s, self.n_heads, -1).transpose(1, 2)  # [b,n_head,s,d_head]

        attn = torch.softmax(torch.matmul(q, k.transpose(-1, -2)), dim=-1)  # [b,n_head,s,s]
        attn.masked_fill_()
        output = torch.matmul(attn, v)  # [b,n_head,s,d_head]
        output = output.transpose(1, 2)  # [b,s,n_head]
        output = output.reshape(b, s, -1)
        return output


class MultiCrossheadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiCrossheadAttention, self).__init__()
        assert d_model % n_heads == 0
        self.w = nn.Linear(d_model * 3, d_model)
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

    def forward(self, x):
        b, s, d = x.size()
        x_head = self.w(x).reshape(b, s, 3, d)
        q, k, v = x_head[:, :, 0, :], x_head[:, :, 1, :], x_head[:, :, 2, :]  # [b,s,d]

        q = q.reshape(b, s, self.n_heads, -1)
        k = k.reshape(b, s, self.n_heads, -1)
        v = v.reshape(b, s, self.n_heads, -1)

        attn = torch.softmax(torch.matmul(q, k.transpose(-1, -2)), dim=-1)  # [b,s,n_heads, n_heads]
        output = torch.matmul(attn, v)  # [b,s,n_heads,d_head]
        output = output.reshape(b, s, -1)
        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(self,attention,d_model, d_ffn,dropout=0.1,act_func=nn.ReLU()):
        super(TransformerEncoderLayer, self).__init__()

        self.attention=attention

        self.feedforward=nn.Sequential(nn.Linear(d_model,d_ffn),
                                       act_func,
                                       nn.Dropout(dropout),
                                       nn.Linear(d_ffn,d_model))

        self.norm1=nn.LayerNorm(d_model)
        self.norm2=nn.LayerNorm(d_model)

    def forward(self, src, src_mask):
