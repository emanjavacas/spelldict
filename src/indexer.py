# encoding: utf-8
import cPickle as p


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
        self.decoder = {}
        self.encoder = {}
        self._max = 0

    def vocab(self):
        return self.encoder.keys()

    def vocab_len(self):
        return len(self.encoder)

    def encode(self, s):
        if s in self.encoder:
            return self.encoder[s]
        else:
            idx = self._max
            self.encoder[s] = idx
            self.decoder[idx] = s
            self._max += 1
            return idx

    def decode(self, idx):
        if idx not in self.decoder:
            raise ValueError("Cannot found index [%d]" % idx)
        return self.decoder[idx]

    def index(self, seqs):
        """encode a corpus. You could call it fit_transform"""
        indexed_seqs = []
        for seq in seqs:
            indexed_seq = []
            for s in seq:
                indexed_seq.append(self.encode(s))
            indexed_seqs.append(indexed_seq)
        return indexed_seqs

    def save(self, filename):
        with open(filename, 'wb') as f:
            p.dump(self, f)

    @classmethod
    def load(self, filename):
        with open(filename, 'rb') as f:
            return p.load(f)


class CharIndexer(Indexer):
    """Indexer sub-class that knows how to pad itself"""
    def __init__(self, PAD=" ", BOS="<", EOS=">"):
        super(CharIndexer, self).__init__()
        for s in [PAD, BOS, EOS]:
            self.encode(s)
        self.PAD, self.BOS, self.EOS = PAD, BOS, EOS

    @classmethod
    def from_vocabulary(cls, vocabulary):
        """factory method that creates a character encoder
        based on a vocabulary. The vocabulary can originate
        from an indexer of words"""
        char_indexer = cls()
        chars = set([c for w in vocabulary for c in w])
        char_indexer.encode_word(''.join(chars))
        return char_indexer

    def pad(self, char_idxs, max_len):
        pad_idx = self.encode(self.PAD)
        return padding(char_idxs, max_len, pad_idx)

    def encode_word(self, word):
        word = self.BOS + word + self.EOS
        return [self.encode(c) for c in word]

    def decode_word(self, *idxs):
        """note that indexer.decode_word(indexer.encode_word(arg)) != word
        given that BOS and EOS characters might be appended by encode_word"""
        return ''.join([self.decode(c) for c in idxs])
