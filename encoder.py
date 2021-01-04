import os
import logging
import pandas as pd
import transformers
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class MTLEncoder(nn.Module):
    '''
    MTL encoder model
    '''

    def __init__(self, keywords_predictor, seq_encoder, n_keywords_selected_sentences, doc_pooling=torch.mean):
        super(MTLEncoder, self).__init__()
        self.keywords_predictor = keywords_predictor

        self.seq_encoder = seq_encoder
        self.n_keywords_selected_sentences = n_keywords_selected_sentences
        self.pooling = doc_pooling

    def forward(self, x):
        # x: [b,n_seq,seq_len]
        _, id_keywords_selected_sentences = self.keywords_predicting(x, self.n_keywords_selected_sentences)

        id_selected_sentences = id_keywords_selected_sentences
        # if want to add more subtask, the 'id_selected_sentences' should be 'id_keywords_selected_sentences' with other task ids

        embedding = self.doc_embedding(x, id_selected_sentences)
        return embedding

    def doc_embedding(self, x, ids):
        # x: [b,n_seq, seq_len]
        topk_seq = self.batch_indexing(x, ids)  # [b,k,seq]
        batch, n_selected_sentences, seq_len = topk_seq.size()
        output = self.seq_encoder(topk_seq.reshape(-1, seq_len)).reshape(batch, n_selected_sentences, -1)  # [b,k,d]
        output = self.pooling(output, dim=1)  # [b,d]
        return output

    def keywords_predicting(self, x, n_selected_sentences=0):
        # x: [b,n_seq,seq_len]
        pred_logits, id_selected_sentences = self.keywords_predictor(x, n_selected_sentences)
        return pred_logits, id_selected_sentences

    def batch_indexing(self, embeddings, id):
        # embeddings:[b, n_seq, d]
        # id:[b, k]
        output = embeddings.gather(dim=1, index=id.unsqueeze(2).expand(-1, -1, embeddings.size(-1)))
        # output: [b, k, d]
        return output


class MTLEncoderTrainer:
    def __init__(self, keywords_track):
        self.keywords_criterion = nn.BCEWithLogitsLoss()
        self.keywords_track = keywords_track

    def get_loss(self, mtl_encoder, input, target, device):
        input = input.to(device)
        keywords_logits,_ = mtl_encoder.keywords_predicting(input,0)

        # convert target index into target vector
        target_vec = torch.zeros(keywords_logits.size()).to(device).scatter(dim=1,index=target,src=torch.ones(target.size()))

        loss = self.keywords_criterion(keywords_logits, target_vec)
        return loss

    def evaluate_keywords_prediction(self, mtl_encoder, test_loader, device, threshold=0.5):
        mtl_encoder.eval()
        with torch.no_grad():
            outputs = []
            targets = []
            for x, x_tgt, pos, pos_tgt, neg, neg_tgt in test_loader:
                # input:[b, n_seq, seq_len]
                for input, target in [[x, x_tgt], [pos, pos_tgt],[neg, neg_tgt]]:
                    output, _ = mtl_encoder.keywords_predicting(input.to(device))  # [b,l]
                    output = (output > threshold).to(torch.int)
                    outputs += output.cpu().tolist()
                    targets += target.tolist()

            p, r, f, _ = precision_recall_fscore_support(targets, outputs, average='micro')
            pp, rr, ff, _ = precision_recall_fscore_support(targets, outputs, average='macro')
        return p, r, f, pp, rr, ff


class KeywordsPredictor(nn.Module):
    def __init__(self, d_model, n_keywords, k_keywords, encoder, attn):
        super(KeywordsPredictor, self).__init__()
        self.encoder = encoder
        self.attn = attn
        self.label_embedding = Parameter(torch.rand(n_keywords, d_model))
        self.pred_matrix = Parameter(torch.rand(n_keywords, d_model))  # [l,d]
        self.k_keywords=k_keywords

    def forward(self, x, n_selected_sentences=0):
        # x:[b,n_seq, seq_len]
        batch, n_seq, seq_len = x.size()
        embeddings = self.encoder(x.reshape(-1, seq_len)).reshape(batch, n_seq, -1)  # [b,n_seq,d]

        attn_matrix = self.attn(embeddings, self.label_embedding)  # [b,n_seq,l]

        doc_embedding_matrix = torch.bmm(attn_matrix.transpose(-1, -2), embeddings)  # [b,n_keywords_class,d]
        pred_logits = (doc_embedding_matrix * self.pred_matrix).sum(-1)  # [b,n_keywords_class]

        if n_selected_sentences == 0:
            id_selected_sentences=None
        else:
            seq_score = attn_matrix.sum(dim=-1)  # [b,n_seq]
            _, id_selected_sentences = torch.topk(seq_score, n_selected_sentences, sorted=False)  # id: [b,k]

        return pred_logits, id_selected_sentences


class LangEncoder(nn.Module):
    '''
    a text encoder, which compress the sequence length dim into 1,
    if want to use huggingface transformers pretrained language model, to keep the compatibility,
    need to create a class adding a pooling method after the huggingface model.
    '''

    def __init__(self, encoder_type='bigru', d_model=128, n_layers=3, pooling=torch.mean, vocab_size=70000):
        super(LangEncoder, self).__init__()

        if 'huggingface' in encoder_type:
            if encoder_type == 'huggingface_albert_base_v1':
                self.encoder = transformers.AlbertModel.from_pretrained('albert-base-v1')
        else:
            self.embedding = nn.Embedding(vocab_size, d_model)

            if encoder_type == 'gru':
                self.encoder = nn.GRU(d_model, d_model, n_layers)
            elif encoder_type == 'bigru':
                self.encoder = nn.GRU(d_model, d_model // 2, n_layers, bidirectional=True)
            elif encoder_type == 'tfm':
                if d_model % 64 != 0:
                    raise ValueError('d_model must be divided by 64 when using tfm as encoder')
                self.encoder = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(d_model, d_model // 64, d_model * 4, dropout=0.1), n_layers)
            else:
                raise ValueError('encoder_type must be gru, bigru or tfm')
            self.encoder_type = encoder_type
        self.pooling = pooling

    def forward(self, x):
        '''
        :param x: [batch, seq_len]
        :return: output: [b, d]
        '''
        if 'huggingface' in self.encoder_type:
            output = self.encoder(x)
        else:
            x = self.embedding(x).transpose(0, 1)  # [s,b,d]
            if self.encoder_type == 'tfm':
                output = self.encoder(x)
            else:
                output, _ = self.encoder(x)
                output = output
            output = output.transpose(0, 1)
        return self.pooling(output, dim=1)  # [b,d]
