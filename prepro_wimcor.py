import os
import sys
import codecs
from spacy.lang.en import English
import spacy
import numpy as np
from collections import Counter

def locate_entity(document, ent, left_w, right_w):
    left_w = '' if len(left_w) == 0 else left_w[-1].text
    right_w = '' if len(right_w) == 0 else right_w[0].text
    for doc in document:
        if doc.text == ent[0]:
            index = doc.i
            if left_w == '' or document[index - 1].text == left_w:
                if right_w == '' or document[index + len(ent)].text == right_w:
                    return index
    raise Exception()  # If this is ever triggered, there are problems parsing the text. Check SpaCy output!

def read_file(path, trim_texts, maxlen, debug_mode):
    # path = "~/metonymy-resolution/harvest-data/disambiguation-pages/corpora/new-corpora/prewin-multi/wiki_LOCATION_train.txt"  # Input file name.

    dirname = os.path.dirname(path)
    name = os.path.basename(path)
    if 'lit' in name or 'literal' in name or 'LOCATION' in name:
        pmw_label = 0
    else:
        pmw_label = 1

    spacy_tokenizer = English(parser=False)
    en_nlp = spacy.load('en')
    inp = codecs.open(path, mode="r", encoding="utf-8")

    tokenized_texts = []
    labels = []
    target_token_indices = []

    # in debug mode, only a few lines are read using this `limit` variable
    limit = 1000 if pmw_label==0 else 250

    for line in inp:
        line = line.split(u"<SEP>")
        sentence = line[1].split(u"<ENT>")
        entity = [t.text for t in spacy_tokenizer(sentence[1])]
        en_doc = en_nlp(u"".join(sentence).strip())

        index = locate_entity(en_doc, entity, spacy_tokenizer(sentence[0].strip()), spacy_tokenizer(sentence[2].strip()))
        tokens = [t.text for t in en_doc]
        if trim_texts:
            # pick a window of context words for both sides of the PMW
            # instead of the entire sample
            old_pmw = tokens[index]
            len_left_context = max(0, index-maxlen//2 - 1)
            len_right_context = index + maxlen//2 - 1
            tokens = tokens[len_left_context: len_right_context]
            index = index - len_left_context
            assert (old_pmw == tokens[index])
        # print(tokens[index])

        tokenized_texts.append(tokens)
        labels.append(pmw_label)
        target_token_indices.append(index)

        if debug_mode:
            if limit==0:
                break
            else:
                limit = limit-1

    print("Processed {} lines/sentences in file \'{}\'".format(len(tokenized_texts), path))
    return tokenized_texts, labels, target_token_indices

def get_input(file_dir, trim_texts, maxlen, debug_mode):
    tokenized_texts, labels, target_token_indices = [], [], []
    for item in os.listdir(file_dir):
        s_out, l_out, t_out = read_file(file_dir+item, trim_texts, maxlen, debug_mode)
        tokenized_texts.extend(s_out)
        labels.extend(l_out)
        target_token_indices.extend(t_out)
    print(Counter(labels))
    labels = np.array(labels)
    target_token_indices = np.array(target_token_indices)
    return tokenized_texts, labels, target_token_indices
