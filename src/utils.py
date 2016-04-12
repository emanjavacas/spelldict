# encoding: utf-8
from collections import Counter
from itertools import tee
import os
import re
import sys
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


def ppl(model, X_test, y_test):
    n_test = test_data.shape[0]
    n_test_chunks = n_test / chunk +1
    total_score = 0
    for chunk_idx in xrange(n_test_chunks):
        test_chunk_x = test_data[chunk_idx*chunk:(chunk_idx+1)*chunk,:-1]
        test_chunk_y = test_data[chunk_idx*chunk:(chunk_idx+1)*chunk,-1]
        log_proba = np.log(model.predict(test_chunk_x))
        predictions = log_proba[np.arange(test_chunk_x.shape[0]),test_chunk_y]
        total_score += np.sum(predictions)

    print "Words: %d, Perplexity %f" % (n_test,np.exp(-1*total_score/n_test))


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


def get_targets_by_min_freq(min_freq, corpus):
    "return words with at least frequency 'min_freq'"
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


def get_targets(n, corpus):
    """Get most frequent `n` targets. If `min_freq == True`,
    `n` is instead interpreted as a min frequency threshold"""
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

class Progress(object):
    def __init__(self, toolbar_width=40):
        sys.stdout.write("[%s]" % (" " * toolbar_width))
        sys.stdout.flush()
        sys.stdout.write("\b" * (toolbar_width+1))

    def update(self, s):
        sys.stdout.write(s)
        sys.stdout.flush()


def build_contexts(sents, targets=None, window=15, encoding="one_hot", sep=" ",
                   verbose=True):
    "Word-level encoding of target words. Character-level encoding of contexts"
    word_indexer = Indexer()
    char_indexer = CharIndexer(PAD="|", BOS="", EOS="")
    X, y = [], []
    if verbose:
        n = 0
        progress = Progress()
    for sent in sents:
        n += 1
        if verbose and (n % 1000) == 0:
            progress.update("Done [%d] sentences" % n)
        for i, target in enumerate(sent):
            if targets and target not in targets:
                continue
            y.append(word_indexer.encode(target))
            left = sep.join(sent[0:i])[-window:]
            left_idxs = char_indexer.pad_encode(left, window, pad_dir="left")
            X.append(left_idxs)
    return X, y, word_indexer, char_indexer


def encode_data(X, y, word_indexer, char_indexer, window=15, encoding="one_hot"):
    for i, context in enumerate(X):
        if encoding and encoding == "one_hot":
            X[i] = one_hot(context, window, char_indexer.vocab_len())
        else:
            X[i] = context
    return X, y, word_indexer, char_indexer


def get_data(in_dir, n_sents, n_targets, **kwargs):
    # mem efficient (build 2 gens)
    targets = get_targets(n_targets, take(get_sents(in_dir), n_sents))
    X, y, word_indexer, char_indexer = \
        build_contexts(take(get_sents(in_dir), n_sents), targets, **kwargs)
    X, y, word_indexer, char_indexer = \
        encode_data(X, y, word_indexer, char_indexer, **kwargs)
    return np.asarray(X), np.asarray(y), word_indexer, char_indexer


class MiniBatchGenerator(object):
    def __init__(self, in_dir, n_sents, n_targets, batch_size, encoding, **kwargs):
        self.batch_size = batch_size
        X, y, word_indexer, char_indexer = \
            get_data(in_dir, n_sents, n_targets, encoding=encoding, **kwargs)
        self.X = X
        self.y = y
        self.word_indexer = word_indexer
        self.char_indexer = char_indexer
        self.iter_start = False

    def train_test_split(self, test_split_size=0.001):
        if self.iter_start:
            print "WARNING: making test-train split after iterating"
        max_test = self.y.shape[0] * test_split_size
        X_test, y_test = self.X[:max_test], self.y[:max_test]
        self.X, self.y = self.X[max_test:], self.y[max_test:]
        return X_test, y_test

    def __iter__(self):
        
        for start in range(0, self.y.shape[0], self.batch_size):
            end = start + self.batch_size
            yield self.X[start:end], self.y[start:end]
