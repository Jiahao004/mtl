import sys

sys.path.append('..')
import os
import time

import torch.optim as optim

import logging

import argparse
from data_utils.preparing_aan import preparing
from attention import *
from encoder import LangEncoder, KeywordsPredictor, MTLEncoder
from simsiam import SimSiam
from trainer import MTLTrainer


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_epoch',
                        type=int,
                        default=10,
                        help='the number of training epoch')

    parser.add_argument('--batch_size',
                        type=int,
                        default=4,
                        help='the batch size for model')

    parser.add_argument('--lr_simsiam',
                        type=float,
                        default=1e-3,
                        help='the learning rate for simsiam network')

    parser.add_argument('--lr_encoder',
                        type=float,
                        default=1e-3,
                        help='the learning rate for multi task encoder')

    parser.add_argument('--type_simsiam_langcoder',
                        type=str,
                        default='gru',
                        help='the language encoder type for simsiam,()')

    parser.add_argument('--d_model',
                        type=int,
                        default=128,
                        help='the dimension of the model')

    parser.add_argument('--n_seq',
                        type=int,
                        default=256,
                        help='the number of sequences for each paper')

    parser.add_argument('--seq_len',
                        type=int,
                        default=32,
                        help='the number of words in each sequence')

    parser.add_argument('--n_keywords_per_doc',
                        type=int,
                        default=5,
                        help='the number of keywords to predict in keywords prediction task')

    parser.add_argument('--n_keywords_selected_sentences',
                        type=int,
                        default=16,
                        help='the number of keywords-task selected sentences')

    parser.add_argument('--type_keywords_langcoder',
                        type=str,
                        default='gru',
                        help='the language encoder type for keywords predictor,()')

    parser.add_argument('--output_path',
                        type=str,
                        default='../runs/debug/')

    parser.add_argument('--paper_doc_path',
                        type=str,
                        default='../data/aan/papers_text/',
                        help='the document path which contains all the papers')

    parser.add_argument('--cit_file_path',
                        type=str,
                        default='../data/aan/release/2014/networks/paper-citation-network-nonself.txt',
                        help='paper citation record file')

    parser.add_argument('--training_ratio',
                        type=float,
                        default=0.8,
                        help='the ratio which is used to split the dataset for training the model, the validating and testing using each half of the rest samples')

    parser.add_argument('--is_debug_mode',
                        type=bool,
                        default=False,
                        help='if choose debug mode to run the code, only papers start with A or C are kept in dataset')

    parser.add_argument('--is_loading_data',
                        type=bool,
                        default=False,
                        help='if load previous dataloaders')

    parser.add_argument('--loading_data_type',
                        type=str,
                        default='tokenized',
                        help='load either tokenized data or structurized data.')

    parser.add_argument('--is_cuda',
                        type=bool,
                        default=True,
                        help='if use cuda')

    parser.add_argument('--is_langcoder_share_weights',
                        type=bool,
                        default=False,
                        help='if use different language encoder for simsiam and multi task training')

    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    logging.basicConfig(filename=os.path.join(args.output_path, 'run.log'), level=logging.INFO)
    logging.info("------------" + time.asctime() + "---------------")

    for arg, value in sorted(vars(args).items()):
        logging.info("Argument %s: %r", arg, value)

    if not torch.cuda.is_available():
        logging.info('cuda is not surported on your device, use cpu instead.')
        print('cuda is not surported on your device, use cpu instead.')

    device = torch.device('cuda' if args.is_cuda and torch.cuda.is_available() else 'cpu')
    logging.info('device: {}'.format(device))
    print('device: {}'.format(device))

    training_loader, validating_loader, testing_loader, keywords_to_label_dict = preparing(args.cit_file_path,
                                                                                           args.paper_doc_path,
                                                                                           args.output_path,
                                                                                           args.batch_size,
                                                                                           args.n_seq,
                                                                                           args.seq_len,
                                                                                           args.n_keywords_per_doc,
                                                                                           args.training_ratio,
                                                                                           args.is_debug_mode,
                                                                                           args.is_loading_data,
                                                                                           args.loading_data_type)



    n_keywords = len(keywords_to_label_dict)
    langencoder_of_keywords_predictor = LangEncoder(args.type_keywords_langcoder, args.d_model)
    keywords_predictor = KeywordsPredictor(args.d_model, n_keywords, args.n_keywords_per_doc,
                                           langencoder_of_keywords_predictor,
                                           DotproductAttention(args.d_model, args.d_model))

    if args.is_langcoder_share_weights:
        langencoder_of_simsiam = langencoder_of_keywords_predictor
    else:
        langencoder_of_simsiam = LangEncoder(args.type_keywords_langcoder,args.d_model)
    encoder = MTLEncoder(keywords_predictor, langencoder_of_simsiam, args.n_keywords_selected_sentences)
    encoder_optimizer = optim.Adam(encoder.parameters(), args.lr_encoder)

    simsiam_predictor = nn.Sequential(nn.Linear(args.d_model, args.d_model),
                                      nn.ReLU(),
                                      nn.Linear(args.d_model, args.d_model),
                                      nn.ReLU(),
                                      nn.Linear(args.d_model, args.d_model))

    simsiam = SimSiam(simsiam_predictor)
    simsiam_optimizer = optim.Adam(simsiam.parameters(), args.lr_simsiam)

    trainer = MTLTrainer()
    trainer.train(args.n_epoch, encoder, simsiam, encoder_optimizer, simsiam_optimizer, training_loader,
                  validating_loader, testing_loader, device, args.output_path)

    print('done, file save to {}'.format(args.output_path))

if __name__ == '__main__':
    main()
