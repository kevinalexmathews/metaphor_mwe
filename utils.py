import torch
import spacy
import numpy as np

from transformers import BertModel, RobertaModel, XLNetModel, DistilBertModel
from transformers import BertTokenizer, BertConfig, RobertaTokenizer, RobertaConfig, XLNetTokenizer, XLNetConfig
from transformers import DistilBertTokenizer, DistilBertConfig

def get_plm_resources(plm, vocab_len):
    """load PLM resources such as model, tokenizer and config"""
    if plm=='bert':
        bert_model = BertModel.from_pretrained('bert-base-uncased')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert_config = BertConfig(vocab_size_or_config_json_file=vocab_len)
    elif plm=='roberta':
        bert_model = RobertaModel.from_pretrained('roberta-base')
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        bert_config = RobertaConfig(vocab_size_or_config_json_file=vocab_len)
    elif plm=='xlnet':
        bert_model = XLNetModel.from_pretrained('xlnet-base-cased')
        tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
        bert_config = XLNetConfig(vocab_size_or_config_json_file=vocab_len)
    elif plm=='distilbert':
        bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        bert_config = DistilBertConfig(vocab_size_or_config_json_file=vocab_len)
    return bert_model, tokenizer, bert_config

def adjacency(sentences,max_len):
    """compute dependent-to-head adjacency matrices"""
    nlp = spacy.load("en_core_web_sm")
    A = []
    for sent in sentences:
        doc = nlp(sent)
        adj = np.zeros([max_len,max_len])
        for tok in doc:
            if not str(tok).isspace():
                if tok.i+1<max_len and tok.head.i+1<max_len:
                    adj[tok.i+1][tok.head.i+1] = 1
        A.append(adj)
    return A


def pad_or_truncate(input_ids, max_len):
        pad = lambda seq,max_len : seq[0:max_len] if len(seq) > max_len else seq + [0] * (max_len-len(seq))
        return torch.Tensor([pad(seq,max_len) for seq in input_ids]).long()

