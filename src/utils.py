# encoding: utf-8
from indexer import Indexer, CharIndexer
from collection import Counter

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


def get_sents(in_fn='../data/merged.txt'):
    with codecs.open(in_fn, 'r', 'utf-8') as f:
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
    Build word-level encoded dataset and vocabulary.
    Build character encoder based on vocabulary.
    Word_idx -> String -> Char_idxs
    """
    word_indexer = Indexer()
    X, y = [], []
    for sent in sents:
        for i, target in enumerate(sent[1:]):
            if target not in targets:
                continue
            X.append(word_indexer.encode(sent[i]))
            y.append(word_indexer.encode(target))
    char_indexer = CharIndexer.from_vocabulary(word_indexer.vocab())
    max_word_len = max([len(w) for w in word_indexer.vocab()])
    for i, word in enumerate(X):
        word_chars = word_indexer.decode(word)
        char_idxs = char_indexer.pad_encode(word_chars, max_word_len)
        X[i] = one_hot(char_idxs, max_word_len + 2, char_indexer.vocab_len())
    return X, y, word_indexer, char_indexer


def get_data(n_sents, n_targets):
    sents = take(get_sents(), n_sents)
    targets = get_targets(n_targets)
    X, y, word_indexer, char_indexer = build_contexts(sents, targets)
    return np.asarray(X), np.asarray(y), word_indexer, char_indexer


# text = """from their godly endeavour, assuring themselves,\n
# that though bad Christians carp and repine\n
# at the confidence and respect which Catholick\n
# Princes and Ministers of State shew to their Clergy,\n
# by trusting them with their conscienbces and affaires\n"""
# sents = [sent.split() for sent in text.split('\n') if sent.split()]
