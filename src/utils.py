# encoding: utf-8
from collections import Counter
from itertools import tee
import os
import re
import codecs
import tarfile

from indexer import Indexer, CharIndexer

from keras.models import model_from_json
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


def save_model(model, word_indexer, char_indexer, fname):
    word_indexer.save(fname + "_word.pkl")
    char_indexer.save(fname + "_char.pkl")
    with open(fname + "_arch.json", "w") as f:
        f.write(model.to_json())
    model.save_weights(fname + "_weights.h5")
    return None


def load_model(fname):
    with open(fname + "_arch.json", "r") as f:
        model = model_from_json(f.read())
    model.load_weights(fname + "_weights.h5")
    word_indexer = Indexer.load(fname + "_word.pkl")
    char_indexer = CharIndexer.load(fname + "_char.pkl")
    return model, word_indexer, char_indexer


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
                yield [re.sub(r'[()]', '', w) for w in l.split()]


def read_targets(n, in_fn='../data/targets.txt'):
    with codecs.open(in_fn, 'r', 'utf-8') as f:
        cnt = 0
        for l in f:
            if cnt >= n:
                break
            word, count = l.split('\t')
            cnt += 1
            yield word


def targets_by_min(min_freq, corpus):
    targets = set()
    freqs = Counter()
    for s in corpus:
        for w in s:
            if w in targets:
                continue
            freqs[w] += 1
            if freqs[w] >= min_freq:
                targets.update([w])
    return list(targets)


def get_targets(n, corpus, min_freq=False):
    """Get most frequent `n` targets. If `min_freq == True`,
    `n` is instead interpreted as a min frequency threshold"""
    if min_freq:
        return targets_by_min(min_freq, corpus)
    else:
        counter = Counter((w for s in corpus for w in s))
        return dict(counter.most_common(n)).keys()


def sliding_window(seq, size=2, fillvalue=None):
    for i, it in enumerate(seq):
        output, j, k = [], i, i
        while len(output) < size:
            j -= 1
            output.extend([seq[j]] if j >= 0 else [fillvalue])
        output.reverse()
        output.extend([seq[i]])
        while len(output) < size * 2 + 1:
            k += 1
            output.extend([seq[k]] if k < len(seq) else [fillvalue])
        yield output


def build_contexts(sents, targets=None, window=15, encoding="one_hot", sep=" "):
    "Word-level encoding of target words. Character-level encoding of contexts"
    word_indexer = Indexer()
    char_indexer = CharIndexer(PAD="|", BOS="", EOS="")
    X, y = [], []
    for sent in sents:          # encoding
        for i, target in enumerate(sent):
            if targets and target not in targets:
                continue
            y.append(word_indexer.encode(target))
            left = sep.join(sent[0:i])[-window:]
            left_idxs = char_indexer.pad_encode(left, window, pad_dir="left")
            X.append(left_idxs)
    for i, context in enumerate(X):
        if encoding and encoding == "one_hot":
            X[i] = one_hot(context, window, char_indexer.vocab_len())
        else:
            X[i] = context
    return X, y, word_indexer, char_indexer


# def get_data(in_dir, n_sents, n_targets, **kwargs):
#     # s1, s2 = tee(take(get_sents(in_dir), n_sents)) # assumes get_sents fits in mem
#     sents = list(take(get_sents(in_dir), n_sents))
#     targets = get_targets(n_targets, sents)
#     X, y, word_indexer, char_indexer = build_contexts(sents, targets, **kwargs)
#     return np.asarray(X), np.asarray(y), word_indexer, char_indexer

def get_data(in_dir, n_sents, n_targets, **kwargs):
    # mem efficient (build 2 gens)
    targets = get_targets(n_targets, take(get_sents(in_dir), n_sents))
    X, y, word_indexer, char_indexer = \
        build_contexts(take(get_sents(in_dir), n_sents), targets, **kwargs)
    return np.asarray(X), np.asarray(y), word_indexer, char_indexer


def get_batches(in_dir, n_sents, n_targets, batch_size, w_idxr, c_idxr, **kwargs):
    # assumes X, y fit in memory
    targets = get_targets(n_targets, take(get_sents(in_dir), n_sents))
    X, y, word_indexer, char_indexer = \
        build_contexts(take(get_sents(in_dir), n_sents), targets, **kwargs)
    w_idxr = word_indexer
    c_idxr = char_indexer
    for start in range(0, y.shape[0], batch_size):
        end = start + batch_size
        yield X[start:end], y[start:end]
