from __future__ import absolute_import, division, print_function

import collections
import unicodedata
import torch
import six
from torch.utils.data import Dataset
import random
import numpy as np

def glove2dict(src_filename):
    """
    GloVe vectors file reader.
    Parameters
    ----------
    src_filename : str
        Full path to the GloVe file to be processed.
    Returns
    -------
    dict
        Mapping words to their GloVe vectors as `np.array`.
    """
    # This distribution has some words with spaces, so we have to
    # assume its dimensionality and parse out the lines specially:
    if '840B.300d' in src_filename:
        line_parser = lambda line: line.rsplit(" ", 300)
    else:
        line_parser = lambda line: line.strip().split()
    data = {}
    with open(src_filename, encoding='utf8') as f:
        while True:
            try:
                line = next(f)
                line = line_parser(line)
                data[line[0]] = np.array(line[1: ], dtype=np.float64)
            except StopIteration:
                break
            except UnicodeDecodeError:
                pass
    return data


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

class WordLevelTokenizer(object):
    """Runs end-to-end tokenziation."""
    def __init__(self, vocab_file, config, delimiter=" ", max_seq_len=128):
        self.vocab = load_vocab(vocab_file)
        self.vocab_reverse = collections.OrderedDict()
        for k, v in self.vocab.items():
            self.vocab_reverse[v] = k
        self.pad_token_id = config.pad_token_id
        self.bos_token_id = config.bos_token_id
        self.eos_token_id = config.eos_token_id
        self.unk_token_id = config.unk_token_id
        self.mask_token_id = config.mask_token_id
        self.special_token_ids = set(
            [config.pad_token_id, config.bos_token_id, config.eos_token_id, 
            config.unk_token_id, config.mask_token_id]
        )
        
        self.max_seq_len = max_seq_len
        self.delimiter = delimiter
        
    def tokenize(self, text):
        split_tokens = []
        for token in text.split(self.delimiter):
            split_tokens.append(token)
        return split_tokens
    
    def convert_tokens_to_ids(self, tokens):
        return convert_tokens_to_ids(self.vocab, tokens)
    
    def __call__(self, text):
        original = self.convert_tokens_to_ids(self.tokenize(text))
        original = original[:(self.max_seq_len-2)]
        return [self.bos_token_id] + original + [self.eos_token_id]
    
    def batch_decode(self, pred_labels, skip_special_tokens=True):
        decode_labels_batch = []
        for labels in pred_labels:
            decode_labels = []
            for l in labels.tolist():
                if l == self.eos_token_id:
                    break
                if l not in self.special_token_ids:
                    decode_labels += [self.vocab_reverse[l]]
            decode_labels_batch += [self.delimiter.join(decode_labels)]
        return decode_labels_batch

class COGSDataset(Dataset):
    def __init__(
        self, cogs_path, 
        src_tokenizer, tgt_tokenizer,
        partition, max_len=512, max_examples=-1,
        least_to_most=False
    ):
        self._items = [] # ()
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
    
        self.eval_cat = []
        is_gen_dev = False
        if partition == "gen-dev":
            partition = "gen"
            is_gen_dev = True
        for l in open(f"{cogs_path}/{partition}.tsv", "r").readlines():
            if max_examples != -1 and len(self._items) > max_examples:
                break
            text, sparse, cat = l.split("\t")
            src_input_ids = src_tokenizer(text)
            tgt_input_ids = tgt_tokenizer(sparse)
            self._items += [(src_input_ids, tgt_input_ids)]
            self.eval_cat += [cat.strip()]
            
        if "train" in partition:
            random.shuffle(self._items)
            if least_to_most:
                self._items = sorted(
                    self._items, key = lambda i: len(i[0]), 
                    reverse=False
                )

        if is_gen_dev:
            # this is a strange partition accordingly to previous works.
            # well, since other ppl are using this, i have to do it as well!
            random.shuffle(self._items)
            self._items = sorted(
                self._items, key = lambda i: len(i[0]), 
                reverse=True if not least_to_most else False
            )
            self._items = self._items[:len(self._items)//10]
            
    def __len__(self):
        return len(self._items)

    def __getitem__(self, item):
        return self._items[item]
    
    def collate_batch(self, batch):
        src_seq_lens = []
        tgt_seq_lens = []
        for i in range(len(batch)):
            src_seq_lens += [len(batch[i][0])]
            tgt_seq_lens += [len(batch[i][1])]
        max_src_seq_lens = max(src_seq_lens)
        max_tgt_seq_lens = max(tgt_seq_lens)
        
        input_ids_batch = []
        mask_batch = []
        labels_batch = []
        for i in range(len(batch)):
            input_ids = batch[i][0] + [0] * (max_src_seq_lens - src_seq_lens[i])
            input_ids_batch += [input_ids]
            
            mask = [1] * src_seq_lens[i] + [0] * (max_src_seq_lens - src_seq_lens[i])
            mask_batch += [mask]
            
            labels = batch[i][1] + [0] * (max_tgt_seq_lens - tgt_seq_lens[i])
            labels_batch += [labels]

        return {"input_ids": torch.tensor(input_ids_batch),
                "labels": torch.tensor(labels_batch),
                "attention_mask": torch.tensor(mask_batch)}
