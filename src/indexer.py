# encoding: utf-8
from collections import Counter, Iterable


def padding(chars, max_len, padder):
    if len(chars) == max_len:
        return chars
    if len(chars) > max_len:
        raise ValueError("Sequence longer than max")
    return (max_len - len(chars)) * [padder] + chars


def flatten(lst):
    if not isinstance(lst, str) and isinstance(lst, Iterable):
        for i in lst:
            for subi in flatten(lst):
                yield subi
    else:
        yield lst


class Indexer(object):
    def __init__(self):
        self.decoder = {}
        self.encoder = {}
        self.freq = Counter()
        self.max = 0

    def vocab(self):
        return self.encoder.keys()

    def most_common(self, n=None, indexed=True):
        items = [w for (w, f) in self.freq.most_common(n)]
        if indexed:
            return [self.encode(i) for i in items]
        else:
            return items

    def encode(self, s):
        self.freq[s] += 1
        if s in self.encoder:
            return self.encoder[s]
        else:
            idx = self.max
            self.encoder[s] = idx
            self.decoder[idx] = s
            self.max += 1
            return idx

    def decode(self, idx):
        if idx not in self.decoder:
            raise ValueError("Cannot found index [%d]" % idx)
        return self.decoder[idx]

    def fit(self, seqs):
        for i in flatten(seqs):
            self.encode(i)


class CharIndexer(Indexer):
    def __init__(self, PAD=" ", BOS="<", EOS=">"):
        super(CharIndexer, self).__init__()
        for s in [PAD, BOS, EOS]:
            self.encode(s)
        self.PAD, self.BOS, self.EOS = PAD, BOS, EOS

    @classmethod
    def from_word_indexer(cls, word_indexer):
        char_indexer = cls()
        chars = set([c for w in word_indexer.vocab() for c in w])
        char_indexer.encode_word(''.join(chars))
        return char_indexer

    def pad_encode(self, word, max_word_len):
        encoded = self.encode_word(word)
        pad_idx = self.encode(self.PAD)
        return padding(encoded, max_word_len + 2, pad_idx)

    def encode_word(self, word):
        word = self.BOS + word + self.EOS
        return [self.encode(c) for c in word]
