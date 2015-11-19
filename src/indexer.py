# encoding: utf-8
from collections import Counter


def padding(chars, max_len, padder):
    chars_len = len(chars)
    if chars_len == max_len:
        return chars
    if chars_len > max_len:
        raise ValueError("Sequence of lenght [%d] longer than max [%d]"
                         % (chars_len, max_len))
    return (max_len - chars_len) * [padder] + chars


def flatten(lst, nested_types=(tuple, list)):
    if isinstance(lst, nested_types):
        for it in lst:
            for subit in flatten(it):
                yield subit
    else:
        yield lst


class Indexer(object):
    def __init__(self):
        self.freq = Counter()
        self.decoder = {}
        self.encoder = {}
        self.max = 0

    def vocab(self):
        return self.encoder.keys()

    def most_common(self, max_number=None):
        return dict(self.freq.most_common(max_number))

    def cut(self, vocabulary):
        "reduces indexed items to match a new vocabulary"
        for w, idx in self.encoder.items():
            if w not in vocabulary:
                del self.encoder[w]
                del self.decoder[idx]

    def encode(self, s):
        idx = self._encode(s)
        self.freq[idx] += 1
        return idx

    def _encode(self, s):
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

    def index(self, seqs):
        indexed_seqs = []
        for seq in seqs:
            indexed_seq = []
            for s in seq:
                indexed_seq.append(self.encode(s))
            indexed_seqs.append(indexed_seq)
        return indexed_seqs


class CharIndexer(Indexer):
    def __init__(self, PAD=" ", BOS="<", EOS=">"):
        super(CharIndexer, self).__init__()
        for s in [PAD, BOS, EOS]:
            self.encode(s)
        self.PAD, self.BOS, self.EOS = PAD, BOS, EOS

    @classmethod
    def from_vocabulary(cls, vocabulary):
        char_indexer = cls()
        chars = set([c for w in vocabulary for c in w])
        char_indexer.encode_word(''.join(chars))
        return char_indexer

    def pad_encode(self, word, max_word_len):
        encoded = self.encode_word(word)
        pad_idx = self.encode(self.PAD)
        return padding(encoded, max_word_len + 2, pad_idx)

    def encode_word(self, word):
        word = self.BOS + word + self.EOS
        return [self.encode(c) for c in word]
