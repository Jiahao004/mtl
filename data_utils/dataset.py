import random
import torch
import pandas as pd
from torch.utils.data import Dataset


class MTLDataset(Dataset):
    '''
    standard mtl dataset, all the dataset and dataloaders in codes are assumed to return the same formatted data.
    this dataset only store the paper-citation pairs
    '''

    def __init__(self, paper_citation_pairs, mtl_database):
        super(MTLDataset, self).__init__()
        self.database = mtl_database
        self.pairs = paper_citation_pairs

    def __getitem__(self, item):
        paper, cit = self.pairs[item]
        output = self.database.get_item(paper, cit)
        return output

    def __len__(self):
        return len(self.pairs)


class MTLDatabase:
    '''
    a database which contain the dataframe
    '''

    def __init__(self, df):
        assert isinstance(df,pd.DataFrame)
        self.name_to_index_dict = {name: index for index, name in enumerate(df['title'])}
        self.df = df
        self.length = len(df)


    def get_tensor(self, name):
        id = self.name_to_index_dict[name]
        return self.df.iloc[id]['input'].clone().detach(), self.df.iloc[id]['keywords_tgt'].clone().detach()

    def negative_sampling(self, name):
        id = self.name_to_index_dict[name]
        cits = self.df.iloc[id]['cits']

        neg_id = random.randint(0, self.length)
        while self.df.iloc[neg_id]['title'] in cits:
            neg_id = random.randint(0, self.length)
        return self.df.iloc[neg_id]['input'].clone().detach(),self.df.iloc[neg_id]['keywords_tgt'].clone().detach()

    def get_item(self, paper, cit):
        x, x_tgt=self.get_tensor(paper)
        pos,pos_tgt=self.get_tensor(cit)
        neg,neg_tgt=self.negative_sampling(paper)
        return x, x_tgt, pos, pos_tgt, neg, neg_tgt
