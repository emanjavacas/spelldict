# encoding: utf-8
from collections import Counter
import os
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


def get_targets(n, corpus, min_freq=False):
    """Get most frequent `n` targets. If `min_freq == True`,
    `n` is instead interpreted as a min frequency threshold"""
    if min_freq:
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


def build_contexts(sents, targets=None, window=2, one_hot_enc=True, sep=" "):
    """
    Word-level encoding of target words. Character-level encoding of contexts.
    """
    word_indexer = Indexer()
    char_indexer = CharIndexer(PAD="|", BOS="", EOS="")
    X, y = [], []
    max_word_len = 0
    for sent in sents:
        for i, target in enumerate(sent[window:]):
            if targets and target not in targets:
                continue
            y.append(word_indexer.encode(target))
            left_words = sep.join(sent[max(0, i - window):i])
            right_words = sep.join(sent[i+1:min(len(sent), i+1+window)])
            left_idxs = char_indexer.encode_seq(left_words)
            right_idxs = char_indexer.encode_seq(right_words)
            max_word_len = max(max_word_len, len(left_words), len(right_words))
            X.append(tuple((left_idxs, right_idxs)))
    for i, (left, right) in enumerate(X):
        padded = char_indexer.pad(left, max_word_len, pad_dir="left")
        padded.extend(char_indexer.pad(right, max_word_len, pad_dir="right"))
        if one_hot_enc:
            X[i] = one_hot(padded, max_word_len * 2, char_indexer.vocab_len())
        else:
            X[i] = padded
    return X, y, word_indexer, char_indexer


def get_data(in_dir, n_sents, n_targets, **kwargs):
    sents = take(get_sents(in_dir), n_sents)
    targets = get_targets(n_targets, take(get_sents(in_dir), n_sents))
    X, y, word_indexer, char_indexer = build_contexts(sents, targets, **kwargs)
    return np.asarray(X), np.asarray(y), word_indexer, char_indexer


text = """Therefore it is thought fitt,\n
to giue notice to all persons in general of his Maiesties erpresse.\n
And if any hereafter shall haue occasion of vse any Cloath printed,\n
and that the buyer and buyers may not incurr his Maiesties displeasures,\n
nor bring vpon themselues the paines penalties and imprisonment, that may\n
be inflicted as aforesaid. Therefore they may repaire to Hunny Laine in\n
Cheapside, ouer against Bowe Church, where they shall be reasonably dealt\n
withall for the buying of such things ready Printed, or for the Printing.\n"""

sents = text.split("\n")
X, y, w, c = build_contexts([s.split(" ") for s in sents])
