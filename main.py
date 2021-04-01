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
from transformers import get_linear_schedule_with_warmup
from transformers import AdamW, BertModel
from tqdm import tqdm, trange
from sklearn.model_selection import KFold, StratifiedKFold
import pandas as pd
import numpy as np
import time
import gc

from train import train_test_loader, trainer
from models import Seqlab
from models import Seqlabbase
from utils import pad_or_truncate
from utils import get_plm_resources
from prepro_wimcor import get_input as get_wimcor_input
from get_results import get_args

def read_trofi_input(pickle_dir):
    df = pd.read_csv(pickle_dir, header=0, sep=',')
    # Create sentence and label lists
    tokenized_texts = df.sentence.values
    labels = df['label'].values
    target_token_indices = df['verb_idx'].values
    return tokenized_texts, labels, target_token_indices

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    args = get_args()
    pickle_dir = args.pickle_dir
    train_batch_size = args.train_batch_size
    test_batch_size = args.test_batch_size
    n_splits = args.n_splits
    n_epochs = args.epochs
    expt_model_choice = args.expt_model_choice
    window_size = args.window_size
    dropout = args.dropout
    maxlen = args.maxlen
    num_total_steps = args.num_total_steps
    num_warmup_steps = args.num_warmup_steps
    trim_texts = args.trim_texts
    debug_mode = args.debug_mode
    distort_context = args.distort_context
    plm_choice = args.plm_choice
    layer_no = args.layer_no
    max_grad_norm = 1.0

    tokenized_texts, labels, target_token_indices = get_wimcor_input(pickle_dir, trim_texts, maxlen, debug_mode, distort_context)
    maxlen = max([len(sent) for sent in tokenized_texts]) + 2
    print('num of samples: {}'.format(len(tokenized_texts)))
    print('maxlen of tokenized texts: {}'.format(maxlen))
    print('tokenize the first sample: {}'.format(tokenized_texts[0]))
    print('label of the first sample: {}'.format(labels[0]))

    # add special tokens at the beginning and end of each sentence
    for sent, label in zip(tokenized_texts, labels):
            sent.insert(0, '[CLS]')
            sent.insert(len(sent), '[SEP]')
            label.insert(0, 0)
            label.insert(len(label), 0)
    # construct the vocabulary
    vocab = list(set([w for sent in tokenized_texts for w in sent]))
    bert_model, tokenizer, bert_config = get_plm_resources(plm_choice, len(vocab))
    # index the input words
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    input_ids = pad_or_truncate(input_ids,maxlen)
    labels = pad_or_truncate(labels, maxlen)

    all_test_indices = []
    all_predictions = []
    all_folds_labels = []
    recorded_results_per_fold = []
    splits = train_test_loader(input_ids, labels, target_token_indices, n_splits, train_batch_size, test_batch_size, window_size)

    for i, (train_dataloader, test_dataloader) in enumerate(splits):
        print('fold number {}:'.format(i+1))

        if expt_model_choice=='seqlab':
            expt_model = Seqlab(bert_config, dropout, bert_model, layer_no)
        elif expt_model_choice=='seqlabbase':
            expt_model = Seqlabbase(bert_config, dropout, bert_model, window_size, layer_no)
        print('Loaded the {} expt model'.format(expt_model_choice))
        expt_model.to(device)

        optimizer = AdamW(expt_model.parameters(), lr=2e-5, correct_bias=False)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                     num_training_steps=num_total_steps)

        scores, all_preds, all_labels, test_indices = trainer(n_epochs, expt_model, optimizer, scheduler,
                train_dataloader, test_dataloader, train_batch_size, test_batch_size, device, expt_model_choice)

        recorded_results_per_fold.append((scores.accuracy(),)+scores.precision_recall_fscore_coarse())
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
