import os
import re
import pandas as pd
import numpy as np
import swifter

from torch.utils.data import DataLoader
from transformers import AlbertTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

from data_utils.dataset import *

DEBUG_MODE = True


def reading_files(cit_file_path, doc_path, is_debug_mode=False):
    papers = {}
    for file in sorted(os.listdir(doc_path)):
        if file.endswith('txt') and file[0].isalpha():
            if is_debug_mode:
                #if not (file.startswith('A')):
                if not (file.startswith('A') or file.startswith('C')):
                    continue
            title = file.split('.')[0]
            with open(os.path.join(doc_path, file), 'r') as open_file:
                papers[title] = open_file.read()
    df = pd.DataFrame(papers.items(), columns=['title', 'content'])

    paper_map = {}
    with open(cit_file_path, 'r') as open_file:
        for i, line in enumerate(open_file.readlines()):
            cit, ori = line.strip().split(' ==> ')
            if ori not in paper_map:
                paper_map[ori] = []
            paper_map[ori].append(cit)
            if cit not in paper_map:
                paper_map[cit] = []
            paper_map[cit].append(ori)
    df['cits'] = df['title'].swifter.apply(paper_map.get)
    return df


def tokenizing(s, n_seq, seq_len, tokenizer=AlbertTokenizer.from_pretrained('albert-base-v2'), len_threshold=5):
    '''
    tokenizing the input document string into structurized tensors
    '''
    sents = ''.join(s.split('\n')).split('. ')
    res = torch.zeros([n_seq, seq_len], dtype=torch.long)
    n_sent_in_res = 0
    for sent in sents:
        if n_sent_in_res == n_seq:
            break
        if len(sent.split()) < len_threshold:
            # filter out the sent with small length
            continue
        tokens = tokenizer.encode(sent, padding='max_length', truncation=True, max_length=seq_len)
        res[n_sent_in_res] = torch.tensor(tokens, dtype=torch.long)
        n_sent_in_res += 1

    return res


def preparing(cit_file_path, paper_doc_path, output_path, batch_size=1, n_seq=256, seq_len=32, n_keywords_per_doc=5,
              training_ratio=0.8, is_debug_mode=False, is_loading_data=False, loading_data_type='structurized'):

    if is_loading_data:


        if loading_data_type=='structurized':
            print('loading previous preparing dataloaders...')
            data_dict = torch.load(os.path.join(output_path, 'data.dict'))
            training_loader = data_dict['training_loader']
            validating_loader = data_dict['validating_loader']
            testing_loader = data_dict['testing_loader']
            keywords_to_label_dict = data_dict['keywords_to_label_dict']
            print('done')
            return training_loader, validating_loader, testing_loader, keywords_to_label_dict

        elif loading_data_type == 'tokenized':
            print('loading previous tokenized data')
            df=pd.read_json(os.path.join(output_path,'df.json'))

        else:
            raise ValueError('the loading_data_type should be either tokenized or structurized')

    else:
        print('reading files...')
        df = reading_files(cit_file_path, paper_doc_path, is_debug_mode)
        print('done')

        print('abstract extracting...')
        df['abstract'] = df['content'].swifter.apply(lambda x: re.findall('Abs.*Int', x, re.DOTALL))
        print('done')

        print('keywords generating...')
        df = tfidf(df, k=n_keywords_per_doc)
        print('done')

        df.to_json(os.path.join(output_path,'df.json'))


    print('tokenizing...')
    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
    df['input'] = df['content'].swifter.apply(tokenizing, args=(n_seq, seq_len, tokenizer))
    print('done')

    print('keywords_labeling...')
    df_training, df_validating, df_testing, keywords_to_label_dict = seperating_dataset(df, training_ratio)
    df_training['keywords_tgt'] = df_training['keywords'].swifter.apply(
        lambda l: torch.tensor([keywords_to_label_dict[w] for w in l]))
    df_validating['keywords_tgt'] = df_validating['keywords'].swifter.apply(
        lambda l: torch.tensor([keywords_to_label_dict[w] for w in l]))
    df_testing['keywords_tgt'] = df_testing['keywords'].swifter.apply(
        lambda l: torch.tensor([keywords_to_label_dict[w] for w in l]))
    print('done')

    print('generating dataloaders...')
    training_loader = generating_dataloader(df_training, batch_size, not is_debug_mode)
    validating_loader = generating_dataloader(df_validating, batch_size, not is_debug_mode)
    testing_loader = generating_dataloader(df_testing, batch_size, not is_debug_mode)
    print('done')

    torch.save({'training_loader': training_loader,
                'validating_loader': validating_loader,
                'testing_loader': testing_loader,
                'keywords_to_label_dict': keywords_to_label_dict},
               os.path.join(output_path, 'data.dict'))

    return training_loader, validating_loader, testing_loader, keywords_to_label_dict


def generating_dataloader(df, batch_size, shuffle=True):
    database = MTLDatabase(df[['title', 'cits', 'input', 'keywords_tgt']])

    paper_cit_pairs = []
    for i in range(len(df)):
        row = df.iloc[i]
        paper = row['title']
        paper_cit_pairs += [(paper, cit) for cit in row['cits']]
    dataset = MTLDataset(paper_cit_pairs, database)

    dataloader = DataLoader(dataset, batch_size, shuffle)

    return dataloader


def seperating_dataset(df, train_ratio=0.8):
    def removing_cross_dataset_citations(df):
        assert isinstance(df, pd.DataFrame)
        papers = set(df['title'].tolist())
        df['cits'] = df['cits'].swifter.apply(lambda l: [item for item in l if item in papers])
        df = df[~df['cits'].str.len().eq(0)]
        return df

    df = df.dropna()  # drop papers with no cits

    n_samples = len(df)
    n_training_samples = int(train_ratio * n_samples)
    n_validating_samples = (n_samples - n_training_samples) // 2
    df_training = df[:n_training_samples]
    df_validating = df[n_training_samples:n_training_samples + n_validating_samples]
    df_testing = df[n_training_samples + n_validating_samples:]
    df_training = removing_cross_dataset_citations(df_training)
    df_validating = removing_cross_dataset_citations(df_validating)
    df_testing = removing_cross_dataset_citations(df_testing)

    # counting keywords
    tmp = []
    for t_df in [df_training, df_validating, df_testing]:
        tmp += [a.tolist() for a in t_df['keywords'].tolist()]
    keywords_list = []
    for l in tmp:
        keywords_list += l
    keywords_list = list(set(keywords_list))
    keywords_to_label_dict = {keyword: i for i, keyword in enumerate(keywords_list)}
    return df_training, df_validating, df_testing, keywords_to_label_dict


def tfidf(df, column='content', max_df=0.9, min_df=0.2, k=10):
    '''
    using tf-idf to find the topk words in each sequence
    :param corpus: dataframe with item 'content'
    :return: the same dataframe df with item 'keywords'
    '''
    tfidf_vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df)
    corpus = df[column].values
    tfidf = tfidf_vectorizer.fit_transform(corpus)
    word_index = np.array(tfidf_vectorizer.get_feature_names())
    df['tfidf_vec'] = tfidf.toarray().tolist()
    df['keywords'] = df['tfidf_vec'].swifter.apply(lambda x: word_index[np.array(x).argsort()[-k:]])
    return df


def main():
    cit_file_path = '../data/aan/release/2014/networks/paper-citation-network-nonself.txt'
    doc_path = '../data/aan/papers_text/'
    dataloader_training, dataloader_validating, dataloader_testing, keywords_20_label_dict = preparing(cit_file_path,
                                                                                                       doc_path,
                                                                                                       is_debug_mode=DEBUG_MODE)
    print(next(iter(dataloader_training)))


if __name__ == '__main__':
    main()
