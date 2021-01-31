import os
import sys
import json

import pandas as pd
from sklearn.model_selection import train_test_split
from evaluate import Evaluate
from tqdm import tqdm_notebook

import re, spacy, copy, random
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertConfig, RobertaTokenizer, RobertaConfig, XLNetTokenizer, XLNetConfig
from transformers import DistilBertTokenizer, DistilBertConfig
from transformers import get_linear_schedule_with_warmup
from transformers import AdamW, BertModel
from layers.GCN import *
from tqdm import tqdm, trange
from sklearn.model_selection import KFold, StratifiedKFold
import pandas as pd
import numpy as np
import time
import gc

from train import train_test_loader, trainer
from models import BertWithGCNAndMWE
from utils import pad_or_truncate
from prepro_wimcor import get_input as get_wimcor_input
from get_results import get_args

def read_trofi_input(file_dir):
    df = pd.read_csv(file_dir, header=0, sep=',')
    # Create sentence and label lists
    tokenized_texts = df.sentence.values
    labels = df['label'].values
    target_token_indices = df['verb_idx'].values
    return tokenized_texts, labels, target_token_indices

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    args = get_args()
    file_dir = args.file_dir
    train_batch_size = args.train_batch_size
    test_batch_size = args.test_batch_size
    n_splits = args.n_splits
    n_epochs = args.epochs
    dropout = args.dropout
    maxlen = args.maxlen
    num_total_steps = args.num_total_steps
    num_warmup_steps = args.num_warmup_steps
    trim_texts = args.trim_texts
    debug_mode = args.debug_mode
    plm = args.plm
    max_grad_norm = 1.0

    tokenized_texts, labels, target_token_indices = get_wimcor_input(file_dir, trim_texts, maxlen, debug_mode)
    maxlen = max([len(sent) for sent in tokenized_texts]) + 2
    print('num of samples: {}'.format(len(tokenized_texts)))
    print('maxlen of tokenized texts: {}'.format(maxlen))
    print('tokenize the first sample: {}'.format(tokenized_texts[0]))
    print('label of the first sample: {}'.format(labels[0]))

    # add special tokens at the beginning and end of each sentence
    for sent in tokenized_texts:
            sent.insert(0, '[CLS]')
            sent.insert(len(sent), '[SEP]')
    # construct the vocabulary
    vocab = list(set([w for sent in tokenized_texts for w in sent]))
    # index the input words
    if plm=='bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert_config = BertConfig(vocab_size_or_config_json_file=len(vocab))
    elif plm=='roberta':
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        bert_config = RobertaConfig(vocab_size_or_config_json_file=len(vocab))
    elif plm=='xlnet':
        tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
        bert_config = XLNetConfig(vocab_size_or_config_json_file=len(vocab))
    elif plm=='distilbert':
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        bert_config = DistilBertConfig(vocab_size_or_config_json_file=len(vocab))
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    input_ids = pad_or_truncate(input_ids,maxlen)

    all_test_indices = []
    all_predictions = []
    all_folds_labels = []
    recorded_results_per_fold = []
    splits = train_test_loader(input_ids, labels, target_token_indices, n_splits, train_batch_size, test_batch_size)

    for i, (train_dataloader, test_dataloader) in enumerate(splits):
        model = BertWithGCNAndMWE(bert_config, dropout, plm)
        model.to(device)

        optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                     num_training_steps=num_total_steps)

        print('fold number {}:'.format(i+1))

        scores, all_preds, all_labels, test_indices = trainer(n_epochs, model, optimizer, scheduler,
                train_dataloader, test_dataloader, train_batch_size, test_batch_size, device)
        recorded_results_per_fold.append((scores.accuracy(),)+scores.precision_recall_fscore())

        all_test_indices.append(test_indices)
        all_predictions.append(all_preds)
        all_folds_labels.append(all_labels)

    print('K-fold cross-validation results:')
    print("Accuracy: {}".format(sum([i for i,j,k,l in recorded_results_per_fold])/n_splits))
    print("Precision: {}".format(sum([j for i,j,k,l in recorded_results_per_fold])/n_splits))
    print("Recall: {}".format(sum([k for i,j,k,l in recorded_results_per_fold])/n_splits))
    print("F-score: {}".format(sum([l for i,j,k,l in recorded_results_per_fold])/n_splits))

    # sanity checks
    # print('recorded_results_per_fold: {}'.format(recorded_results_per_fold))
    # print('len(set(recorded_results_per_fold)): {}'.format(len(set(recorded_results_per_fold))))
