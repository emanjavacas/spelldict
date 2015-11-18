# encoding: utf-8
from indexer import Indexer, CharIndexer
import tarfile
import string
import re

import numpy as np

punct = re.escape(''.join(c for c in string.punctuation if c != '@'))
subs = re.compile('(%s)' % '|'.join(
    ['|',
     '\[[^\]]+\]',
     '^[' + punct + ']',
     '[' + punct + ']$']))
process_fns = [
    lambda sent: sent.lower(),
    lambda sent: sent.split(),
    lambda sent: [w[:10] for w in sent if '@' not in w],
    lambda sent: [subs.sub('', w) for w in sent]]


def process_text(text):
    for fn in process_fns:
        text = fn(text)
    return text


def one_hot(n, max_len):
    vec = np.zeros(max_len, dtype=np.int8)
    vec[n] = 1
    return vec


def take(gen, n):
    cnt = 0
    for i in gen:
        if cnt >= n:
            break
        yield i


def corpus(n_sents):
    return take(get_sents(), n_sents)


def get_sents(in_fn='../data/postprocessed.tar.gz', process_fn=process_text):
    """ generator over tokenized sents in tarred """
    with tarfile.open(in_fn, "r|*") as tar:
        for tarinfo in tar:
            doc = tar.extractfile(tarinfo)
            for l in doc:
                processed = process_fn(l)
                if processed:
                    yield processed


def build_data(corpus, n_sents, n_targets=None):
    X, y = [], []
    word_indexer = Indexer()    # fit words
    word_indexer.fit(corpus(n_sents))
    char_indexer = CharIndexer.from_word_indexer(word_indexer)
    targets = word_indexer.most_common(n_targets, indexed=False)
    max_word_len = max([len(w) for w in word_indexer.vocab()])
    for sent in corpus(n_sents):
        for i, target in enumerate(sent[1:]):
            if n_targets and target not in targets:
                continue
            target_idx = word_indexer.encode(target)
            left = [one_hot(c, char_indexer.max)
                    for c in char_indexer.pad_encode(sent[i], max_word_len)]
            X.append(left)
            y.append(target_idx)
    return np.asarray(X), np.asarray(y), word_indexer, char_indexer


# text = """from their godly endeavour, assuring themselves,\n
# that though bad Christians carp and repine\n
# at the confidence and respect which Catholick\n
# Princes and Ministers of State shew to their Clergy,\n
# by trusting them with their conscienbces and affaires\n"""
# sents = [sent.split() for sent in text.split('\n') if sent.split()]
X, y, word_indexer, char_indexer = build_data(corpus, 50000, 500)
