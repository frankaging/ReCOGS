from __future__ import absolute_import, division, print_function

import pandas as pd
import re, random, copy

import pandas as pd
import re, random, copy

import collections
import unicodedata
import torch
import six
from torch.utils.data import Dataset
import random
import numpy as np
import json

np_re = re.compile(r"""
    ^
    \s*(\*)?
    \s*(\w+?)\s*
    \(
    \s*(.+?)\s*
    \)
    \s*$""", re.VERBOSE)

pred_re = re.compile(r"""
    ^
    \s*(\w+?)\s*
    \.
    \s*(\w+?)\s*
    \(
    \s*(.+?)\s*
    ,
    \s*(.+?)\s*
    \)
    \s*$""", re.VERBOSE)

mod_re = re.compile(r"""
    ^
    \s*(\w+?)\s*
    \.
    \s*(\w+?)\s*
    \.
    \s*(\w+?)\s*
    \(
    \s*(.+?)\s*
    ,
    \s*(.+?)\s*
    \)
    \s*$""", re.VERBOSE)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_np(phi):   
    the, pred, var = np_re.search(phi).groups()
    indef = '' if the is None else '*'
    return {'type': 'np', 'definiteness': indef, 'pred': pred, 'entvar': var}

def parse_pred(phi):
    pred, role, eventvar, entvar = pred_re.search(phi).groups()
    return {'type': 'role', 'role': role, 'pred': pred, 'entvar': entvar, 'eventvar': eventvar}

def parse_mod(phi):
    nppred, rel, pred, e1, e2 = mod_re.search(phi).groups()
    # Keeping `rel` even though it is always 'nmod'
    return {'type': 'mod', 'rel': rel, 'pred': pred, 'nppred': nppred, 'e1': e1, 'e2': e2}

def translate_entity_simplied(entvar, data):
    ent = [e for e in data if e['type'] == 'np' and e.get("entvar") == entvar]
    if not ent:
        return entvar, entvar
    else:
        ent = ent[0]
        return f"{ent['definiteness']} {ent['entvar']} ( {ent['pred']} )", f"{ent['definiteness']} {ent['pred']}"

def translate_entity(entvar, data):
    ent = [e for e in data if e['type'] == 'np' and e.get("entvar") == entvar]
    if not ent:
        return entvar
    else:
        ent = ent[0]
        return f"{ent['definiteness']} {ent['entvar']} ( {ent['pred']} )"
    
def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")

def convert_tokens_to_ids(vocab, tokens):
    """Converts a sequence of tokens into ids using the vocab."""
    ids = []
    for token in tokens:
        if token not in vocab.keys():
            ids.append(vocab['[UNK]'])
        else:
            ids.append(vocab[token])
    return ids
        
def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, "r") as reader:
        while True:
            token = convert_to_unicode(reader.readline())
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab