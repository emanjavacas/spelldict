# encoding: utf-8
from indexer import Indexer, CharIndexer
from text_preprocessing import process_text

import codecs
import tarfile

import numpy as np


def numpy_one_hot(ints, n_rows, n_cols):
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


def from_tar(in_fn='../data/postprocessed.tar.gz', process_fn=process_text):
    """ generator over tokenized sents in tarred """
    with tarfile.open(in_fn, "r|*") as tar:
        for tarinfo in tar:
            for l in tar.extractfile(tarinfo):
                processed = process_fn(l)
                if processed:
                    yield processed


def get_sents(in_fn='../data/merged.txt', process_fn=process_text):
    with codecs.open(in_fn, 'r', 'utf-8') as f:
        for l in f:
            processed = process_fn(l)
            if processed:
                yield processed


def build_contexts(sents, n_targets):
    """
    Encode entire corpus with word_indexer (counts and indexes).
    Extract targets from word_indexer.
    Build word-level encoded dataset and vocabulary.
    Remove OOV words.
    Build character encoder based on vocabulary.
    Word_idx -> String -> Char_idxs
    """
    word_indexer = Indexer()
    X, y, vocabulary = [], [], set()
    indexed_sents = word_indexer.index(sents)
    targets = word_indexer.most_common(n_targets)
    for sent in indexed_sents:
        for i, target in enumerate(sent[1:]):
            if target not in targets:
                continue
            vocabulary.update([word_indexer.decode(target)])
            vocabulary.update([word_indexer.decode(sent[i])])
            X.append(sent[i])
            y.append(target)
    word_indexer.cut(vocabulary)
    char_indexer = CharIndexer.from_vocabulary(vocabulary)
    max_word_len = max([len(w) for w in vocabulary])
    for i, word in enumerate(X):
        word_chars = word_indexer.decode(word)
        char_idxs = char_indexer.pad_encode(word_chars, max_word_len)
        X[i] = numpy_one_hot(char_idxs, max_word_len + 2, char_indexer.max)
    return X, y, word_indexer, char_indexer


def get_data(n_sents, n_targets):
    sents = take(get_sents(), n_sents)
    X, y, word_indexer, char_indexer = build_contexts(sents, n_targets)
    return np.asarray(X), np.asarray(y), word_indexer, char_indexer


# text = """from their godly endeavour, assuring themselves,\n
# that though bad Christians carp and repine\n
# at the confidence and respect which Catholick\n
# Princes and Ministers of State shew to their Clergy,\n
# by trusting them with their conscienbces and affaires\n"""
# sents = [sent.split() for sent in text.split('\n') if sent.split()]
