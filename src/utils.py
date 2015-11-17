# encoding: utf-8
import tarfile
import re
import numpy as np

in_fn = "data/postprocessed.tar.gz"


def partition(lst, n):
    for ix in xrange(0, len(lst), n):
        yield lst[ix:ix+n]


class TokenIndexer(object):
    def __init__(self):
        self.decoder = {}
        self.encoder = {}
        self.last = 1

    def char_indexer(self):
        """factory method that creates a char-level indexer based on
        a previously learnt word-level indexer"""
        indexer = TokenIndexer()
        chars = set([c for w in self.encoder for c in w])
        for c in chars:
            indexer.encode(c)
        return indexer

    def to_one_hot(self, n):
        vec = np.zeros(len(self.decoder), dtype=np.int8)
        vec[n] = 1
        return vec

    def _encode(self, s):
        idx = self.encoder.get(s, None)
        if idx is not None:
            return idx
        else:
            idx = self.last
            self.encoder[s] = idx
            self.decoder[idx] = s
            self.last += 1
            return idx

    def encode(self, s, output="int"):
        idx = self._encode(s)
        if output == "int":
            return idx
        if output == "one_hot":
            return self.to_one_hot(idx)

    def decode(self, idx, output="int"):
        if idx not in self.decoder:
            raise ValueError("Cannot found index [%d]" % idx)
        return self.decoder[idx]


class SeqIndexer(object):
    def __init__(self, pad='_'):
        self.indexer = TokenIndexer()
        self.pad = pad
        self.indexer.encode(pad)

    def encode(self, seq, size=10):
        assert len(seq) <= size
        ids = []
        for t in seq:
            idx = self.indexer.encode(t)
            ids.append(idx)
        margin = size - len(seq)
        if margin:  # add padding
            ids.extend(margin * [self.indexer.encode(self.pad)])
        assert len(ids) == size
        return ids

    def decode(self, sent):
        tkns = []
        for idx in sent:
            w = self.indexer.decode(idx)
            tkns.append(w)
        return tkns


process_fns = [
    lambda x: re.sub(r'\[[^\]]+\]', "", x),
    lambda x: x.replace("|", ""),
    lambda x: x.lower()
]


def process_text(text):
    for fn in process_fns:
        text = fn(text)
    return text


def itertar(in_fn, process_fn=process_text):
    """ generator over lines in tarred """
    with tarfile.open(in_fn, "r|*") as tar:
        for tarinfo in tar:
            doc = tar.extractfile(tarinfo)
            for l in doc:
                yield process_fn(l)


def load_sents(text_generator, indexer, n_sents=100, ignore_missing=True):
    """
    Returns word-level indexed sents. Modifies the indexer as side effect.
    Ignores words that include '@' used as missing character symbol.
    """
    sents = []
    for line in text_generator:
        sent = []
        for word in line.strip().split():
            if ignore_missing and '@' in word:
                continue
            sent.append(indexer.encode(word))
        if sent:
            sents.append(sent)
    return sents


def padding(char_idx, max_len, default=0):
    "0 is a reserved index in our implementation"
    if len(char_idx) == max_len:
        return char_idx
    if len(char_idx) > max_len:
        raise ValueError("Sequence longer than max")
    return (max_len - len(char_idx)) * [default] + char_idx


def load_data(sents, word_indexer, char_indexer, max_word_len):
    """
    Computes sents to tuples of prev-word, target-word
    transforming the former to one-hot-encoding matrix
    """
    for sent in sents:
        for i, target in enumerate(sent[1:]):
            prev_idx = sent[i]
            prev_chars = [char_indexer.encode(c) for c in word_indexer.decode(prev_idx)]
            yield (np.asarray(prev_emb), target)





def construct_contexts(sents, context_len=5):
    X, y = [], []
    for sent in sents:
        for i, target in enumerate(sent):
            y.append(target)
            offset = i - context_len
            left = padding(offset) + sent[max(0, offset): i]
            X.append(left)
    return zip(X, y)


text = """from their godly endeavour, assuring themselves,\n
that though bad Christians carp and repine\n
at the confidence and respect which Catholick\n
Princes and Ministers of State shew to their Clergy,\n
by trusting them with their conscienbces and affaires\n"""


word_indexer = TokenIndexer()
sents = load_sents(text.split("\n"), word_indexer)
char_indexer = word_indexer.char_indexer()
data = list(load_data(sents, word_indexer, char_indexer))
