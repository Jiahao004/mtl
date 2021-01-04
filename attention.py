import torch
import torch.nn as nn
from torch.nn.parameter import Parameter



class DotproductAttention(nn.Module):
    '''
    Dot Product Attention
    '''

    def __init__(self, d_model, d_attn):
        super(DotproductAttention, self).__init__()
        self.w_seq = nn.Linear(d_model, d_attn)
        self.w_label = nn.Linear(d_model, d_attn)

    def forward(self, seq, label):
        '''
        :param seq: [b,s,d]
        :param label: [b,l,d]
        :return:
        '''
        seq = self.w_seq(seq)  # [b,s,d_attn]
        label = self.w_label(label)  # [b,l,d_attn]
        seq_attn = torch.softmax(torch.matmul(seq, label.transpose(-1, -2)), dim=-1)  # [b,s,l]
        return seq_attn


class ComponentEmbeddingAttention(nn.Module):
    '''
    Component Attention
    '''

    def __init__(self, d_model, d_attn, n_comp):
        super(ComponentEmbeddingAttention, self).__init__()
        self.comp_matrix = Parameter(torch.randn([n_comp, d_attn]))
        self.w_seq = nn.Linear(d_model, d_attn)
        self.w_label = nn.Linear(d_model, d_attn)

    def forward(self, seq, label):
        '''
        :param seq: [b,s,d]
        :param label: [b,l,d]
        :return:
        '''
        seq = self.w_seq(seq)  # [b,s,d_attn]
        label = self.w_label(label)  # [b,l,d_attn]
        attned_seq = torch.matmul(seq, self.comp_matrix.transpose(-1, -2))  # [b,s,n_comp]
        attned_label = torch.matmul(label, self.comp_matrix.transpose(-1, -2))  # [b,l,n_comp]
        seq_attn = torch.softmax(torch.matmul(attned_seq, attned_label.transpose(-1, -2)), dim=-1)  # [b,s,l]
        return seq_attn
