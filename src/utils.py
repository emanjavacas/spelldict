# encoding: utf-8
from indexer import Indexer, CharIndexer
from collections import Counter

import os
import codecs
import tarfile

import numpy as np


def one_hot(ints, n_rows, n_cols):
    """
    computes the embedding matrix for a possibly padded word.
    `n_cols`: max word length in corpus
    `n_rows`: length of one-hot vector, total number of characters
    """
    mat = np.zeros((n_rows, n_cols), dtype=np.int8)
    mat[np.arange(n_rows), ints] = 1
    return mat


def take(gen, n):
    cnt = 0
    for i in gen:
        if cnt >= n:
            break
        cnt += 1
        yield i


def from_tar(in_fn='../data/postprocessed.tar.gz'):
    """ generator over tokenized sents in tarred """
    with tarfile.open(in_fn, "r|*") as tar:
        for tarinfo in tar:
            for l in tar.extractfile(tarinfo):
                yield l.split()


def get_sents(in_dir):
    fns = os.listdir(in_dir)
    for fn in fns:
        path = os.path.join(in_dir, fn)
        with codecs.open(path, 'r', 'utf-8') as f:
            for l in f:
                yield l.split()


def read_targets(n, in_fn='../data/targets.txt'):
    with codecs.open(in_fn, 'r', 'utf-8') as f:
        cnt = 0
        for l in f:
            if cnt >= n:
                break
            word, count = l.split('\t')
            cnt += 1
            yield word


def get_targets(n, corpus):
    counter = Counter((w for s in corpus for w in s))
    return dict(counter.most_common(n)).keys()


def build_contexts(sents, targets):
    """
    Word-level encoding of target words. Character-level encoding
    of contexts.
    """
    word_indexer = Indexer()
    char_indexer = CharIndexer()
    X, y = [], []
    max_word_len = 0
    for sent in sents:
        for i, target in enumerate(sent[1:]):
            if target not in targets:
                continue
            y.append(word_indexer.encode(target))
            char_idxs = char_indexer.encode_word(sent[i])
            max_word_len = max(max_word_len, len(char_idxs))
            X.append(char_idxs)
    for i, word in enumerate(X):
        padded_idxs = char_indexer.pad(word, max_word_len)
        X[i] = one_hot(padded_idxs, max_word_len, char_indexer.vocab_len())
    return X, y, word_indexer, char_indexer


def get_data(in_dir, n_sents, n_targets):
    sents = take(get_sents(in_dir), n_sents)
    targets = get_targets(n_targets, take(get_sents(in_dir), n_sents))
    X, y, word_indexer, char_indexer = build_contexts(sents, targets)
    return np.asarray(X), np.asarray(y), word_indexer, char_indexer


# text = """from their godly endeavour, assuring themselves,\n
# that though bad Christians carp and repine\n
# at the confidence and respect which Catholick\n
# Princes and Ministers of State shew to their Clergy,\n
# by trusting them with their conscienbces and affaires\n"""
# sents = [sent.split() for sent in text.split('\n') if sent.split()]
# sents = get_sents('../data/scripts/test_files_out/D00000998435240000.htm')
# targets = get_targets(1000, sents)
# sents = get_sents('../data/scripts/test_files_out/D00000998435240000.htm')
# X, y, word, char = build_contexts(sents, targets)
